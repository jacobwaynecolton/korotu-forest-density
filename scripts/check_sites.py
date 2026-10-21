"""Quick sanity check of downloaded CHM tiles: per-site canopy-cover spread.

Computes cover (> 2 m) on 10 m blocks in each tile's native grid. This is a rough EDA
check, not the final labels (those go on the Sentinel-2 grid, see PLAN.md §3).
Bins below are placeholders until class thresholds are agreed.

Usage: python -I scripts/check_sites.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
CHM = ROOT / "data/raw/fri_chm"
BLOCK = 20  # 20 px * 0.5 m = 10 m
BINS = [0, 0.10, 0.40, 0.70, 1.0001]
LABELS = ["<10%", "10-40%", "40-70%", ">=70%"]


def tile_cover(path: Path) -> np.ndarray:
    with rasterio.open(path) as s:
        a = s.read(1, masked=True)
    h, w = (n // BLOCK * BLOCK for n in a.shape)  # a few tiles are 2004 px wide
    a = a[:h, :w]
    valid = ~np.ma.getmaskarray(a)
    tree = (a.filled(0) > 2.0) & valid
    shp = (h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    n_valid = valid.reshape(shp).sum(axis=(1, 3))
    cover = tree.reshape(shp).sum(axis=(1, 3)) / np.maximum(n_valid, 1)
    return cover[n_valid == BLOCK * BLOCK].ravel()


def main():
    rows = []
    for site in sorted(p for p in CHM.iterdir() if p.is_dir()):
        tifs = sorted(site.glob("*_CHM.tif"))
        if not tifs:
            continue
        c = np.concatenate([tile_cover(t) for t in tifs])
        hist = np.histogram(c, BINS)[0] / len(c)
        rows.append(dict(site=site.name, tiles=len(tifs), cells=len(c),
                         mean_cover=c.mean(), **dict(zip(LABELS, hist.round(3)))))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = ROOT / "data/manifest/site_cover_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
