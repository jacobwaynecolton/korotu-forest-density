"""Put existing canopy-cover products on each site's 10 m label grid, for benchmarking.

Products (all as cover fraction 0-1, float32, NaN = no data):
  meta_chm_v2  Meta/WRI Global Canopy Height v2 (2026, ~1 m, Vantor/Maxar imagery, CC BY 4.0).
               Cover = share of ~1 m pixels > 2 m per 10 m cell, the same definition as our
               lidar label. Imagery dates per tile are recorded from the product metadata.
  scanfi_v2    SCANFI v2 total crown closure (%), 30 m, NFI-calibrated Landsat model;
               the 5-yearly layer nearest the lidar flight year.
  ntems_2022   NTEMS forest canopy cover 2022 (%), 30 m, lidar-plot-calibrated Landsat
               model (Matasci et al.). Non-forest pixels are nodata -> cover 0 on land.
30 m products use nearest resampling onto the 10 m grid (each 10 m cell takes its 30 m
pixel); evaluation also aggregates everything to 30 m.

Writes data/benchmarks/<site>_<product>.tif and data/manifest/benchmark_meta.csv.

Usage: python -I scripts/benchmarks.py [all|holdout] [products...]
"""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box, shape

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_USERAGENT", "Mozilla/5.0")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, LABELS = ROOT / "data/manifest", ROOT / "data/labels"
OUT = ROOT / "data/benchmarks"
META = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
SCANFI = ("/vsicurl/https://ftp.maps.canada.ca/pub/nrcan_rncan/Forests_Foret/SCANFI/v2/"
          "SCANFI_att_closure_{year}_v2_20260119.tif")
NTEMS = ROOT / "data/raw/ntems"


def grid(site):
    with rasterio.open(LABELS / f"{site}_cover10m.tif") as s:
        return s.profile, s.bounds


def save(site, name, arr, prof):
    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(prof, count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(OUT / f"{site}_{name}.tif", "w", **p) as d:
        d.write(arr.astype(np.float32), 1)


def meta_chm(site, prof, bounds, tiles):
    crs, W, H = prof["crs"], prof["width"], prof["height"]
    F = 10  # 1 m fine grid
    fine = np.full((H * F, W * F), 255, np.uint8)
    fine_tf = from_origin(bounds.left, bounds.top, 1.0, 1.0)
    poly = box(*transform_bounds(crs, "EPSG:4326", *bounds))
    hit = tiles[tiles.intersects(poly)].tile.tolist()
    dates = []
    for q in hit:
        with rasterio.open(f"/vsicurl/{META}/chm/{q}.tif") as s:
            b = transform_bounds(crs, s.crs, *bounds)
            w = from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
            a = s.read(1, window=w, boundless=True, fill_value=255)
            wt = s.window_transform(w)
            reproject(a, fine, src_transform=wt, src_crs=s.crs, src_nodata=255,
                      dst_transform=fine_tf, dst_crs=crs, dst_nodata=255,
                      init_dest_nodata=False, resampling=Resampling.nearest)
        try:
            md = requests.get(f"{META}/metadata/{q}.geojson", timeout=60).json()
            for f in md.get("features", []):
                if shape(f["geometry"]).intersects(poly):
                    dates.append(json.dumps(f.get("properties", {}), default=str))
        except Exception:
            pass
    valid = fine != 255
    shp = (H, F, W, F)
    nv = valid.reshape(shp).sum(axis=(1, 3))
    tree = ((fine > 2) & valid).reshape(shp).sum(axis=(1, 3))
    cover = np.where(nv == F * F, tree / np.maximum(nv, 1), np.nan)
    save(site, "meta_chm_v2", cover, prof)
    return dict(meta_tiles=";".join(hit), meta_props=" | ".join(sorted(set(dates)))[:2000])


def warp_to_grid(path, prof, nodata_as_zero):
    with rasterio.open(path) as src, WarpedVRT(src, crs=prof["crs"], transform=prof["transform"],
                                               width=prof["width"], height=prof["height"],
                                               resampling=Resampling.nearest) as v:
        a = v.read(1, masked=True)
    out = a.astype(np.float32).filled(np.nan)
    if nodata_as_zero:
        out = np.where(np.isnan(out), 0.0, out)
    return out / 100.0


def scanfi(site, prof, year):
    yr = int(min(range(1985, 2030, 5), key=lambda y: (abs(y - year), y)))
    a = warp_to_grid(SCANFI.format(year=yr), prof, nodata_as_zero=False)
    a[a > 1.0] = np.nan  # 255 nodata -> 2.55
    save(site, "scanfi_v2", a, prof)
    return dict(scanfi_year=yr)


def ntems(site, prof):
    tifs = sorted(NTEMS.rglob("*.tif"))
    if not tifs:
        raise FileNotFoundError("NTEMS raster not downloaded yet")
    a = warp_to_grid(tifs[0], prof, nodata_as_zero=True)
    a[a > 1.0] = np.nan
    save(site, "ntems_2022", a, prof)
    return dict(ntems_file=tifs[0].name)


def main(which="holdout", products=("meta_chm_v2", "scanfi_v2", "ntems_2022")):
    sites = pd.read_csv(MANIFEST / "sites.csv")
    if which == "holdout":
        sites = sites[sites.split.isin(["val", "test"])]
    tiles = gpd.read_file(ROOT / "data/raw/meta_chm/tiles.geojson") if "meta_chm_v2" in products else None
    prev = MANIFEST / "benchmark_meta.csv"
    rows = {r["site_id"]: r for r in (pd.read_csv(prev).to_dict("records") if prev.exists() else [])}
    for s in sites.itertuples():
        prof, bounds = grid(s.site_id)
        row = rows.get(s.site_id, dict(site_id=s.site_id))
        if "meta_chm_v2" in products:
            row.update(meta_chm(s.site_id, prof, bounds, tiles))
        if "scanfi_v2" in products:
            row.update(scanfi(s.site_id, prof, int(s.year)))
        if "ntems_2022" in products:
            row.update(ntems(s.site_id, prof))
        rows[s.site_id] = row
        print(f"{s.site_id}: {', '.join(products)} done", flush=True)
        pd.DataFrame(rows.values()).to_csv(prev, index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    which = args[0] if args else "holdout"
    main(which, tuple(args[1:]) or ("meta_chm_v2", "scanfi_v2", "ntems_2022"))
