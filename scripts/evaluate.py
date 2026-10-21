"""Score canopy-cover maps against the lidar labels on held-out sites, at 10 m and 30 m.

Sources: benchmark products in data/benchmarks/<site>_<product>.tif and model predictions
in data/predictions/<model>/<site>.tif (band 1 = cover 0-1). Only cells where the label
and every compared source are present are scored, so all sources share the same cells.
30 m = mean of 3x3 blocks of 10 m cells (the label grid is snapped to 60 m, so blocks
nest cleanly).

Metrics: MAE, RMSE, bias, Pearson r, R^2, and 4-class accuracy / macro-F1 with the
placeholder cut points 10 / 40 / 70 %.

Usage: python -I scripts/evaluate.py <split: val|test> <source> [<source> ...]
  source = product name (meta_chm_v2, scanfi_v2, ntems_2022) or model:<name>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
CUTS = [0.10, 0.40, 0.70]


def load(path):
    with rasterio.open(path) as s:
        return s.read(1).astype(np.float32)


def source_path(src, site):
    if src.startswith("model:"):
        return ROOT / "data/predictions" / src.split(":", 1)[1] / f"{site}.tif"
    return ROOT / "data/benchmarks" / f"{site}_{src}.tif"


def block3(a):
    H, W = (n // 3 * 3 for n in a.shape)
    b = a[:H, :W].reshape(H // 3, 3, W // 3, 3)
    ok = np.isfinite(b).all(axis=(1, 3))
    return np.where(ok, np.nanmean(b, axis=(1, 3)), np.nan)


def macro_f1(y, p):
    f = []
    for k in range(4):
        tp = np.sum((y == k) & (p == k)); fp = np.sum((y != k) & (p == k)); fn = np.sum((y == k) & (p != k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(y, p):
    e = p - y
    yc, pc = np.digitize(y, CUTS), np.digitize(p, CUTS)
    return dict(n=int(len(y)), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                bias=float(e.mean()), r=float(np.corrcoef(y, p)[0, 1]),
                r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                acc4=float((yc == pc).mean()), f1_macro=macro_f1(yc, pc))


def main(split, sources):
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    sites = sites[sites.split == split].site_id.tolist()
    data = {res: {"y": [], **{s: [] for s in sources}} for res in ("10m", "30m")}
    for site in sites:
        y = load(ROOT / "data/labels" / f"{site}_cover10m.tif")
        ps = {s: load(source_path(s, site)) for s in sources}
        for res, f in (("10m", lambda a: a), ("30m", block3)):
            yy = f(y); pp = {s: f(np.clip(v, 0, 1)) for s, v in ps.items()}
            ok = np.isfinite(yy) & np.all([np.isfinite(v) for v in pp.values()], axis=0)
            data[res]["y"].append(yy[ok])
            for s in sources:
                data[res][s].append(pp[s][ok])
    rows = []
    for res, d in data.items():
        y = np.concatenate(d["y"])
        for s in sources:
            rows.append(dict(split=split, res=res, source=s, **metrics(y, np.concatenate(d[s]))))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    out = ROOT / "data/manifest" / f"eval_{split}.json"
    old = json.loads(out.read_text()) if out.exists() else []
    keep = [r for r in old if not any(r["source"] == s for s in sources)]
    out.write_text(json.dumps(keep + rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
