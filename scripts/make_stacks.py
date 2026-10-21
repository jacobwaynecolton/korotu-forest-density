"""Multi-date Sentinel-2 stacks per site: up to K cloud-free scenes from the flight season.

For each site: search Element84 earth-search for L2A scenes in the same UTM zone within
+/- 45 days of the lidar flight midpoint (clipped to 1 May - 15 Oct), keep those whose
site window is fully inside the tile and < 1% cloud/shadow/snow by SCL, closest to the
flight first, at least 5 days apart. Writes data/pairs/stacks/<site>_k<k>.tif (12 bands,
float32 reflectance, same grid as the label) and data/manifest/s2_scenes.csv.

Training uses every date as augmentation (same label, different atmosphere and phenology);
evaluation can average predictions over dates.

Usage: python -I scripts/make_stacks.py [K]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pairs import BAD_SCL, BANDS, STAC, boa_offset_applied, read_window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, LABELS = ROOT / "data/manifest", ROOT / "data/labels"
OUT = ROOT / "data/pairs/stacks"


def candidates(client, site, crs, bounds):
    mid = datetime.fromisoformat(site.flight_start) + (
        datetime.fromisoformat(site.flight_end) - datetime.fromisoformat(site.flight_start)) / 2
    lo = max(mid - pd.Timedelta(days=45), mid.replace(month=5, day=1))
    hi = min(mid + pd.Timedelta(days=45), mid.replace(month=10, day=15))
    items = list(client.search(collections=["sentinel-2-l2a"],
                               bbox=transform_bounds(crs, "EPSG:4326", *bounds),
                               datetime=f"{lo:%Y-%m-%d}/{hi:%Y-%m-%d}",
                               query={"eo:cloud_cover": {"lt": 50}}).items())
    epsg = f"EPSG:{crs.to_epsg()}"
    items = [i for i in items if (i.properties.get("proj:code")
                                  or f"EPSG:{i.properties.get('proj:epsg')}") == epsg]
    items.sort(key=lambda i: abs((i.datetime - mid).total_seconds()))
    return items, mid


def write_stack(item, path, bounds, prof, shape):
    applied = boa_offset_applied(item, bounds, shape)
    stack = np.full((len(BANDS), *shape), np.nan, np.float32)

    def band(i_key):
        i, (name, key, _) = i_key
        a = item.assets[key]
        rb = a.extra_fields["raster:bands"][0]
        dn = read_window(a.href, bounds, shape, Resampling.nearest)
        off = 0.0 if applied else rb.get("offset", 0.0)
        return i, np.where(dn == rb.get("nodata", 0), np.nan, dn * rb.get("scale", 1.0) + off)

    with ThreadPoolExecutor(6) as ex:
        for i, arr in ex.map(band, enumerate(BANDS)):
            stack[i] = arr
    p = dict(prof, count=len(BANDS), dtype="float32", nodata=np.nan)
    with rasterio.open(path, "w", **p) as d:
        d.write(stack)
        d.update_tags(scene=item.id, datetime=item.datetime.isoformat())


def main(K: int = 4):
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(MANIFEST / "sites.csv")
    client = Client.open(STAC)
    prev = MANIFEST / "s2_scenes.csv"
    done = pd.read_csv(prev) if prev.exists() else pd.DataFrame(columns=["site_id"])
    rows = done.to_dict("records")
    for site in sites.itertuples():
        if site.site_id in set(done.site_id):
            continue
        with rasterio.open(LABELS / f"{site.site_id}_cover10m.tif") as s:
            prof, bounds, shape, crs = s.profile, s.bounds, (s.height, s.width), s.crs
        items, mid = candidates(client, site, crs, bounds)
        kept = []
        for it in items:
            if len(kept) == K:
                break
            if any(abs((it.datetime - k.datetime).days) < 5 for k in kept):
                continue
            scl = read_window(it.assets["scl"].href, bounds, shape, Resampling.nearest)
            if np.isin(scl, list(BAD_SCL)).mean() >= 0.01:
                continue
            k = len(kept)
            write_stack(it, OUT / f"{site.site_id}_k{k}.tif", bounds, prof, shape)
            kept.append(it)
            rows.append(dict(site_id=site.site_id, k=k, scene=it.id,
                             date=it.datetime.date().isoformat(),
                             days_from_flight=(it.datetime - mid).days))
        print(f"{site.site_id}: {len(kept)} scenes "
              f"({', '.join(f'{(i.datetime - mid).days:+d}d' for i in kept)})", flush=True)
        pd.DataFrame(rows).to_csv(prev, index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
