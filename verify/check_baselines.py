"""Simple baselines on the same cells, so the U-Net's score has a reference point.

  constant       predict the mean training cover everywhere
  ndvi_linear    cover = a + b * NDVI, fitted on training cells
  pixel_gbm      gradient boosting on the 12 bands of one pixel (no spatial context),
                 the per-pixel approach most older work uses (e.g. RF on Landsat)

Each baseline averages its prediction over a site's Sentinel-2 dates, like the U-Net.
Writes verify/results/baselines.json.
Usage: python -I verify/check_baselines.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"


def site(s):
    with rasterio.open(ROOT / "data/labels" / f"{s}_cover10m.tif") as src:
        y = src.read(1)
    xs = []
    for p in sorted((ROOT / "data/pairs/stacks").glob(f"{s}_k*.tif")):
        with rasterio.open(p) as src:
            xs.append(src.read().astype(np.float32))
    return y, np.stack(xs)  # (D, 12, H, W)


def feats(x):
    nd = (x[:, 7] - x[:, 3]) / (x[:, 7] + x[:, 3] + 1e-6)
    return np.concatenate([x, nd[:, None]], axis=1)  # 12 bands + NDVI


def main():
    rng = np.random.default_rng(0)
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    X, Y = [], []
    for s in sites[sites.split == "train"].site_id:
        y, x = site(s)
        f = feats(x)
        r, c = np.nonzero(np.isfinite(y))
        k = rng.choice(len(r), min(8000, len(r)), replace=False)
        for d in range(f.shape[0]):
            v = f[d][:, r[k], c[k]].T
            ok = np.isfinite(v).all(1)
            X.append(v[ok]); Y.append(y[r[k], c[k]][ok])
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(f"training rows: {len(Y):,}")
    const = float(Y.mean())
    lin = LinearRegression().fit(X[:, [12]], Y)
    gbm = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.08, random_state=0).fit(X, Y)

    rows = []
    for split in ("val", "test"):
        err = {k: [] for k in ("constant", "ndvi_linear", "pixel_gbm", "unet")}
        for s in sites[sites.split == split].site_id:
            y, x = site(s)
            f = feats(x)
            D, C, H, W = f.shape
            flat = f.transpose(0, 2, 3, 1).reshape(-1, C)
            ok = np.isfinite(flat).all(1)
            P = {}
            for name, m in (("ndvi_linear", lambda v: lin.predict(v[:, [12]])), ("pixel_gbm", gbm.predict)):
                p = np.full(len(flat), np.nan); p[ok] = m(flat[ok])
                P[name] = np.nanmean(np.clip(p, 0, 1).reshape(D, H, W), axis=0)
            P["constant"] = np.full((H, W), const)
            with rasterio.open(ROOT / "data/predictions/unet_r34_ens3" / f"{s}.tif") as src:
                P["unet"] = src.read(1)
            m = np.isfinite(y) & np.all([np.isfinite(v) for v in P.values()], axis=0)
            for k in err:
                err[k].append(np.abs(P[k] - y)[m])
        for k, e in err.items():
            e = np.concatenate(e)
            rows.append(dict(split=split, method=k, n=int(len(e)), mae=float(e.mean())))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baselines.json").write_text(json.dumps(dict(rows=rows, train_rows=int(len(Y))), indent=1))
    print(df.pivot(index="method", columns="split", values="mae").round(4))


if __name__ == "__main__":
    main()
