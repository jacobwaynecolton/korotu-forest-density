"""Spot-check the benchmark conversions against the original product files.

For random 10 m cells on every held-out site:
  NTEMS / SCANFI  project the cell centre into the product's CRS with pyproj and read the
                  single 30 m pixel underneath straight from the original raster (local
                  NTEMS file, SCANFI over HTTP). Should equal our warped value.
  Meta CHM v2     read the original ~1.2 m height tile, keep the pixels whose centres fall
                  inside the cell, and take the share above 2 m. Our pipeline resamples
                  to a 1 m grid first, so this should agree closely but not exactly.

Writes verify/results/benchmarks.json.
Usage: python -I verify/check_benchmarks.py [cells_per_site]
"""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmarks import META, SCANFI  # noqa: E402

os.environ.setdefault("GDAL_HTTP_USERAGENT", "Mozilla/5.0")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"


def pixel_values(src, crs, xs, ys):
    tx, ty = Transformer.from_crs(crs, src.crs, always_xy=True).transform(xs, ys)
    out = []
    for x, y in zip(tx, ty):
        r, c = src.index(x, y)
        v = src.read(1, window=Window(c, r, 1, 1), boundless=True, masked=True)
        out.append(np.nan if np.ma.getmaskarray(v).all() else float(v.data[0, 0]))
    return np.array(out)


def main(n=40):
    rng = np.random.default_rng(1)
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    bmeta = pd.read_csv(ROOT / "data/manifest/benchmark_meta.csv").set_index("site_id")
    ntems_tif = sorted((ROOT / "data/raw/ntems").rglob("*.tif"))[0]
    rows = []
    for s in sites[sites.split != "train"].itertuples():
        with rasterio.open(ROOT / "data/labels" / f"{s.site_id}_cover10m.tif") as src:
            tf, crs, (H, W) = src.transform, src.crs, src.shape
        r = rng.integers(0, H, n); c = rng.integers(0, W, n)
        cx, cy = tf * (c + 0.5, r + 0.5)
        mine = {}
        for prod in ("ntems_2022", "scanfi_v2", "meta_chm_v2"):
            with rasterio.open(ROOT / "data/benchmarks" / f"{s.site_id}_{prod}.tif") as src:
                mine[prod] = src.read(1)[r, c]
        with rasterio.open(ntems_tif) as src:
            nt = pixel_values(src, crs, cx, cy) / 100
        nt = np.where(np.isnan(nt), 0.0, nt)  # non-forest is nodata in NTEMS -> cover 0
        year = int(bmeta.loc[s.site_id, "scanfi_year"])
        with rasterio.open(SCANFI.format(year=year)) as src:
            sc = pixel_values(src, crs, cx, cy)
        sc = np.where(sc > 100, np.nan, sc) / 100
        # Meta: original tile pixels inside each cell
        tiles = [t for t in str(bmeta.loc[s.site_id, "meta_tiles"]).split(";") if t]
        mt = np.full(n, np.nan)
        for q in tiles:
            with rasterio.open(f"/vsicurl/{META}/chm/{q}.tif") as src:
                to = Transformer.from_crs(crs, src.crs, always_xy=True)
                back = Transformer.from_crs(src.crs, crs, always_xy=True)
                for i in np.flatnonzero(np.isnan(mt)):
                    x0, y1 = tf * (c[i], r[i]); x1, y0 = x0 + 10, y1 - 10
                    bx = to.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
                    rr0, cc0 = src.index(min(bx[0]), max(bx[1])); rr1, cc1 = src.index(max(bx[0]), min(bx[1]))
                    win = Window(cc0 - 1, rr0 - 1, cc1 - cc0 + 3, rr1 - rr0 + 3)
                    a = src.read(1, window=win, boundless=True, fill_value=255)
                    if (a == 255).all():
                        continue
                    jj, ii = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
                    px, py = src.window_transform(win) * (jj + 0.5, ii + 0.5)
                    ux, uy = back.transform(px.ravel(), py.ravel())
                    ux, uy = np.asarray(ux), np.asarray(uy)
                    inside = (ux >= x0) & (ux < x1) & (uy > y0) & (uy <= y1)
                    v = a.ravel()[inside]
                    if inside.sum() and (v != 255).all():
                        mt[i] = float((v > 2).mean())
        for prod, ref in (("ntems_2022", nt), ("scanfi_v2", sc), ("meta_chm_v2", mt)):
            ok = np.isfinite(ref) & np.isfinite(mine[prod])
            d = np.abs(ref[ok] - mine[prod][ok])
            rows.append(dict(site_id=s.site_id, product=prod, cells=int(ok.sum()),
                             exact=float((d < 1e-4).mean()), mean_abs_diff=float(d.mean()), max_abs_diff=float(d.max())))
        print(s.site_id, [(x["product"], round(x["exact"], 3), round(x["mean_abs_diff"], 4)) for x in rows[-3:]], flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("product").apply(lambda g: pd.Series(dict(
        cells=int(g.cells.sum()), exact=float((g.exact * g.cells).sum() / g.cells.sum()),
        mean_abs_diff=float((g.mean_abs_diff * g.cells).sum() / g.cells.sum()), max_abs_diff=float(g.max_abs_diff.max()))),
        include_groups=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmarks.json").write_text(json.dumps(dict(summary=summ.reset_index().to_dict("records"), per_site=rows), indent=1))
    print(summ.round(4))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
