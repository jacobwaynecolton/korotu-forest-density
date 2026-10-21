"""Are the lidar labels and the Sentinel-2 pixels on the same ground?

For every site, correlate the label (canopy cover) with Sentinel-2 NDVI after shifting the
label by dx, dy in -3..3 pixels (10 m steps). If the two grids are registered, the
correlation peaks at zero shift. A parabola through the peak and its neighbours gives a
sub-pixel estimate of any leftover offset.

Writes verify/results/alignment.json.
Usage: python -I verify/check_alignment.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"
S = 3


def ndvi(site):
    vals = []
    for p in sorted((ROOT / "data/pairs/stacks").glob(f"{site}_k*.tif")):
        with rasterio.open(p) as s:
            red, nir = s.read(4).astype(float), s.read(8).astype(float)
        vals.append((nir - red) / (nir + red + 1e-6))
    return np.nanmedian(vals, axis=0)


def corr_at(y, x, dy, dx):
    H, W = y.shape
    ys = y[S + dy:H - S + dy, S + dx:W - S + dx]
    xs = x[S:H - S, S:W - S]
    ok = np.isfinite(ys) & np.isfinite(xs)
    return float(np.corrcoef(ys[ok], xs[ok])[0, 1])


def subpixel(c_minus, c0, c_plus):
    den = c_minus - 2 * c0 + c_plus
    return 0.0 if den == 0 else 0.5 * (c_minus - c_plus) / den


def main():
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    rows, grid_sum = [], np.zeros((2 * S + 1, 2 * S + 1))
    for s in sites.itertuples():
        with rasterio.open(ROOT / "data/labels" / f"{s.site_id}_cover10m.tif") as src:
            y = src.read(1)
        x = ndvi(s.site_id)
        g = np.array([[corr_at(y, x, dy, dx) for dx in range(-S, S + 1)] for dy in range(-S, S + 1)])
        grid_sum += g
        iy, ix = np.unravel_index(np.argmax(g), g.shape)
        sx = subpixel(g[S, S - 1], g[S, S], g[S, S + 1]) * 10
        sy = subpixel(g[S - 1, S], g[S, S], g[S + 1, S]) * 10
        rows.append(dict(site_id=s.site_id, split=s.split, r0=float(g[S, S]), peak_dx=int(ix - S), peak_dy=int(iy - S),
                         subpx_east_m=float(sx), subpx_south_m=float(sy), r_shift1=float(np.mean([g[S, S - 1], g[S, S + 1], g[S - 1, S], g[S + 1, S]]))))
        print(f"{s.site_id}: r(0,0)={g[S, S]:.3f}  peak at ({ix - S},{iy - S})  sub-pixel ({sx:+.1f} m, {sy:+.1f} m)", flush=True)
    df = pd.DataFrame(rows)
    mean_grid = grid_sum / len(df)
    summary = dict(sites=len(df), peak_at_zero=int(((df.peak_dx == 0) & (df.peak_dy == 0)).sum()),
                   median_abs_subpx_m=float(np.median(np.hypot(df.subpx_east_m, df.subpx_south_m))),
                   mean_r0=float(df.r0.mean()), mean_r_shift1=float(df.r_shift1.mean()),
                   mean_grid=mean_grid.round(4).tolist(), shifts=list(range(-S, S + 1)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alignment.json").write_text(json.dumps(dict(summary=summary, per_site=rows), indent=1))
    print({k: v for k, v in summary.items() if k != "mean_grid"})


if __name__ == "__main__":
    main()
