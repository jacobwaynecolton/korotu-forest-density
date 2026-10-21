"""Recalibrate each benchmark product to the lidar cover definition, using TRAINING sites only.

Products define "cover" differently (SCANFI crown closure is photo-interpreted, Meta's is
derived from predicted heights). To compare models rather than definitions, fit a monotone
(isotonic) map product -> lidar cover on training-site cells, then apply it to the val/test
maps as <site>_<product>_cal.tif. Also rasterizes Meta CHM v2 imagery acquisition years for
held-out sites (<site>_meta_year.tif) so results can be split by imagery age.

Usage: python -I scripts/calibrate_benchmarks.py meta_chm_v2 scanfi_v2 [ntems_2022]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
BENCH, LABELS = ROOT / "data/benchmarks", ROOT / "data/labels"
META = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"


def read(p):
    with rasterio.open(p) as s:
        return s.read(1), s.profile


def meta_years(site, prof, tiles):
    yr = np.zeros((prof["height"], prof["width"]), np.int16)
    for q in tiles:
        md = requests.get(f"{META}/metadata/{q}.geojson", timeout=60).json()
        shapes = []
        for f in md.get("features", []):
            d = f.get("properties", {}).get("acq_date")
            if d:
                g = transform_geom("EPSG:4326", prof["crs"], f["geometry"])
                shapes.append((g, int(str(d)[:4])))
        if shapes:
            r = rasterize(shapes, out_shape=yr.shape, transform=prof["transform"], fill=0, dtype="int16")
            yr = np.where(yr == 0, r, yr)
    with rasterio.open(BENCH / f"{site}_meta_year.tif", "w",
                       **dict(prof, count=1, dtype="int16", nodata=0)) as d:
        d.write(yr, 1)


def main(products):
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    rng = np.random.default_rng(0)
    for prod in products:
        xs, ys = [], []
        for s in sites[sites.split == "train"].site_id:
            pth = BENCH / f"{s}_{prod}.tif"
            if not pth.exists():
                continue
            p, _ = read(pth); y, _ = read(LABELS / f"{s}_cover10m.tif")
            ok = np.isfinite(p) & np.isfinite(y)
            idx = rng.choice(ok.sum(), min(40000, ok.sum()), replace=False)
            xs.append(p[ok][idx]); ys.append(y[ok][idx])
        x, y = np.concatenate(xs), np.concatenate(ys)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
        print(f"{prod}: fitted on {len(x):,} train cells from {len(xs)} sites")
        for s in sites[sites.split.isin(["val", "test"])].site_id:
            p, prof = read(BENCH / f"{s}_{prod}.tif")
            out = np.where(np.isfinite(p), iso.predict(np.nan_to_num(p).ravel()).reshape(p.shape), np.nan)
            with rasterio.open(BENCH / f"{s}_{prod}_cal.tif", "w",
                               **dict(prof, count=1, dtype="float32", nodata=np.nan)) as d:
                d.write(out.astype(np.float32), 1)
    if "meta_chm_v2" in products:
        bm = pd.read_csv(ROOT / "data/manifest/benchmark_meta.csv").set_index("site_id")
        for s in sites[sites.split.isin(["val", "test"])].site_id:
            _, prof = read(LABELS / f"{s}_cover10m.tif")
            meta_years(s, prof, str(bm.loc[s, "meta_tiles"]).split(";"))
        print("meta imagery years rasterized for held-out sites")


if __name__ == "__main__":
    main(sys.argv[1:])
