"""Independent recompute of the 10 m canopy-cover labels from the raw 0.5 m lidar CHM.

make_labels.py warps whole tiles with rasterio/GDAL and block-averages. This check takes a
different route: for random 10 m cells it builds the 400 sub-pixel centres in the
Sentinel-2 CRS, projects each point into the lidar tile's own CRS with pyproj, reads the
CHM pixel under it by index arithmetic, and takes the share above 2 m. If the two routes
agree, the labels are what the report says they are.

Writes verify/results/labels.json.
Negative control: a second argument shifts the sample points east by that many metres;
the match should then break, which shows the check can fail.

Usage: python -I verify/check_labels.py [cells_per_site] [shift_m]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"
CELL, FINE = 10.0, 0.5


def load_tiles(site):
    tiles = []
    for p in sorted((ROOT / "data/raw/fri_chm" / site).glob("*_CHM.tif")):
        with rasterio.open(p) as s:
            tiles.append(dict(a=s.read(1), tf=s.transform, crs=s.crs, nd=s.nodata, b=s.bounds))
    return tiles


def chm_at(tiles, xs, ys, label_crs, cache):
    """Height at each point (label CRS); NaN where no tile or invalid."""
    out = np.full(xs.shape, np.nan)
    todo = np.ones(xs.shape, bool)
    for t in tiles:
        key = (str(label_crs), str(t["crs"]))
        if key not in cache:
            cache[key] = Transformer.from_crs(label_crs, t["crs"], always_xy=True)
        tx, ty = cache[key].transform(xs[todo], ys[todo])
        inv = ~t["tf"]
        col, row = inv * (tx, ty)
        col, row = np.floor(col).astype(int), np.floor(row).astype(int)
        H, W = t["a"].shape
        inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        idx = np.flatnonzero(todo)[inside]
        v = t["a"][row[inside], col[inside]].astype(float)
        bad = ~np.isfinite(v) | (v < -2) | (v > 80)
        if t["nd"] is not None:
            bad |= v == t["nd"]
        v[bad] = np.nan
        out[idx] = v
        todo[idx] = False
        if not todo.any():
            break
    return out


def main(n_per_site=300, shift=0.0):
    rng = np.random.default_rng(0)
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    hold = sites[sites.split != "train"].site_id.tolist()
    train = [] if shift else sites[sites.split == "train"].sample(6, random_state=0).site_id.tolist()
    sub = (np.arange(20) + 0.5) * FINE
    rows, cache = [], {}
    for site in hold + train:
        with rasterio.open(ROOT / "data/labels" / f"{site}_cover10m.tif") as s:
            lab, ltf, lcrs = s.read(1), s.transform, s.crs
        tiles = load_tiles(site)
        r, c = np.nonzero(np.isfinite(lab))
        pick = rng.choice(len(r), min(n_per_site, len(r)), replace=False)
        diffs = []
        for i, j in zip(r[pick], c[pick]):
            x0, y0 = ltf * (j, i)  # top-left corner of the cell
            gx, gy = np.meshgrid(x0 + sub + shift, y0 - sub)
            h = chm_at(tiles, gx.ravel(), gy.ravel(), lcrs, cache)
            mine = np.nan if np.isnan(h).any() else float((h > 2).mean())
            diffs.append(abs(mine - lab[i, j]) if np.isfinite(mine) else np.nan)
        d = np.array(diffs)
        ok = np.isfinite(d)
        rows.append(dict(site_id=site, split=sites.set_index("site_id").split[site], cells=int(len(d)),
                         recomputed=int(ok.sum()), exact=float((d[ok] < 1e-6).mean()),
                         within_1px=float((d[ok] <= 1 / 400 + 1e-6).mean()),
                         within_2pct=float((d[ok] <= 0.02).mean()), mean_abs_diff=float(d[ok].mean()),
                         max_abs_diff=float(d[ok].max())))
        print(f"{site}: {rows[-1]['exact']:.1%} exact, {rows[-1]['within_2pct']:.1%} within 2 pts, "
              f"mean |diff| {rows[-1]['mean_abs_diff']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    w = df.recomputed / df.recomputed.sum()
    summary = dict(cells=int(df.recomputed.sum()), sites=len(df), exact=float((df.exact * w).sum()),
                   within_2pct=float((df.within_2pct * w).sum()),
                   mean_abs_diff=float((df.mean_abs_diff * w).sum()), max_abs_diff=float(df.max_abs_diff.max()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("labels.json" if not shift else f"labels_shift{shift:g}m.json")).write_text(json.dumps(dict(summary=summary, per_site=rows), indent=1))
    print(summary)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
