"""Context layers on each site's label grid, for EDA and figures (not training inputs).

1. Sentinel-2 L2A true-colour preview (TCI, 10 m) from the public AWS archive
   (Element84 earth-search STAC): the scene closest to the lidar flight midpoint
   (+/- 45 days, same year) whose site window is < 1% cloud/shadow per the SCL band.
   The real Sentinel-2 pipeline belongs to the S2 teammates; this is only a preview and
   a check that the label grid lines up with S2 pixels (window offsets must be whole).
2. ESA WorldCover 2021 v200 (10 m, CC BY 4.0) class per label cell (mode resampling):
   water mask for EDA, and a first look at a third-party product vs lidar.

Writes data/context/<site>_s2_tci.tif, data/context/<site>_worldcover.tif and
data/manifest/s2_preview_scenes.csv.

Usage: python -I scripts/context_layers.py
"""
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifest"
LABELS = ROOT / "data/labels"
OUT = ROOT / "data/context"
STAC = "https://earth-search.aws.element84.com/v1"
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
CLOUDY_SCL = {3, 8, 9, 10}  # cloud shadow, cloud medium/high, cirrus


def label_grid(site_id):
    with rasterio.open(LABELS / f"{site_id}_cover10m.tif") as s:
        return s.crs, s.transform, s.width, s.height, s.bounds


def s2_preview(site, crs, bounds, client):
    mid = datetime.fromisoformat(site.flight_start) + (
        datetime.fromisoformat(site.flight_end) - datetime.fromisoformat(site.flight_start)) / 2
    ll = transform_bounds(crs, "EPSG:4326", *bounds)
    lo = max(mid - pd.Timedelta(days=45), mid.replace(month=5, day=1))
    hi = min(mid + pd.Timedelta(days=45), mid.replace(month=10, day=15))
    items = list(client.search(collections=["sentinel-2-l2a"], bbox=ll,
                               datetime=f"{lo:%Y-%m-%d}/{hi:%Y-%m-%d}",
                               query={"eo:cloud_cover": {"lt": 40}}).items())
    epsg = f"EPSG:{crs.to_epsg()}"
    items = [i for i in items if (i.properties.get("proj:code") or
                                  f"EPSG:{i.properties.get('proj:epsg')}") == epsg]
    items.sort(key=lambda i: abs((i.datetime - mid).total_seconds()))
    for it in items:
        with rasterio.open(it.assets["scl"].href) as s:
            w = from_bounds(*bounds, transform=s.transform)
            scl = s.read(1, window=w, boundless=True, fill_value=0)
        if scl.size == 0 or (scl == 0).mean() > 0.01:
            continue  # site not fully inside this tile's data area
        cloud = float(np.isin(scl, list(CLOUDY_SCL)).mean())
        if cloud >= 0.01:
            continue
        with rasterio.open(it.assets["visual"].href) as s:
            w = from_bounds(*bounds, transform=s.transform)
            whole = all(abs(v - round(v)) < 1e-6 for v in (w.col_off, w.row_off, w.width, w.height))
            tci = s.read(window=w.round_offsets().round_lengths(), boundless=True, fill_value=0)
        return it, mid, cloud, whole, tci
    return None


def worldcover(crs, tf, width, height, bounds):
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds)
    tiles = {(math.floor(lat / 3) * 3, math.floor(lon / 3) * 3)
             for lat in (s, n) for lon in (w, e)}
    out = np.zeros((height, width), np.uint8)
    for lat, lon in tiles:
        name = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        with rasterio.open(WC_URL.format(tile=name)) as src, WarpedVRT(
                src, crs=crs, transform=tf, width=width, height=height,
                resampling=Resampling.mode, nodata=0) as vrt:
            a = vrt.read(1)
        np.copyto(out, a, where=(out == 0) & (a > 0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(MANIFEST / "sites.csv")
    client = Client.open(STAC)
    rows = []
    for site in sites.itertuples():
        crs, tf, width, height, bounds = label_grid(site.site_id)
        prof = dict(driver="GTiff", crs=crs, transform=tf, width=width, height=height,
                    compress="deflate")
        wc = worldcover(crs, tf, width, height, bounds)
        with rasterio.open(OUT / f"{site.site_id}_worldcover.tif", "w", count=1,
                           dtype="uint8", nodata=0, **prof) as d:
            d.write(wc, 1)
        res = s2_preview(site, crs, bounds, client)
        if res is None:
            print(f"{site.site_id}: no cloud-free S2 scene within +/-45 days")
            rows.append(dict(site_id=site.site_id, scene=None))
            continue
        it, mid, cloud, whole, tci = res
        with rasterio.open(OUT / f"{site.site_id}_s2_tci.tif", "w", count=3,
                           dtype="uint8", **prof) as d:
            d.write(tci)
        gap = (it.datetime - mid).days
        print(f"{site.site_id}: {it.id} ({gap:+d} d from flight), cloud {cloud:.2%}, "
              f"grid aligned={whole}", flush=True)
        rows.append(dict(site_id=site.site_id, scene=it.id, scene_date=it.datetime.isoformat(),
                         days_from_flight=gap, site_cloud_frac=cloud, grid_aligned=whole))
    pd.DataFrame(rows).to_csv(MANIFEST / "s2_preview_scenes.csv", index=False)


if __name__ == "__main__":
    main()
