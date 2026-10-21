"""Recompute the headline metrics with scikit-learn, independently of scripts/evaluate.py.

Reads the label and prediction rasters fresh, scores 10 m cells where every compared
source is present (same rule as evaluate.py), and compares against the stored
data/manifest/eval_<split>.json that the report is built from.

Writes verify/results/metrics.json.
Usage: python -I verify/check_metrics.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"
SOURCES = {"model:unet_r34_ens3": "data/predictions/unet_r34_ens3/{s}.tif",
           "meta_chm_v2": "data/benchmarks/{s}_meta_chm_v2.tif",
           "scanfi_v2": "data/benchmarks/{s}_scanfi_v2.tif",
           "ntems_2022": "data/benchmarks/{s}_ntems_2022.tif"}
ALL = SOURCES | {"meta_chm_v2_cal": "data/benchmarks/{s}_meta_chm_v2_cal.tif",
                 "scanfi_v2_cal": "data/benchmarks/{s}_scanfi_v2_cal.tif",
                 "ntems_2022_cal": "data/benchmarks/{s}_ntems_2022_cal.tif"}


def rd(p):
    with rasterio.open(ROOT / p) as s:
        return s.read(1).astype(np.float64)


def classes(v):
    return np.select([v < 0.1, v < 0.4, v < 0.7], [0, 1, 2], 3)


def main():
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    rows = []
    for split in ("val", "test"):
        ys, ps = [], {k: [] for k in ALL}
        for s in sites[sites.split == split].site_id:
            y = rd(f"data/labels/{s}_cover10m.tif")
            P = {k: np.clip(rd(p.format(s=s)), 0, 1) for k, p in ALL.items()}
            ok = np.isfinite(y) & np.all([np.isfinite(v) for v in P.values()], axis=0)
            ys.append(y[ok])
            for k in ALL:
                ps[k].append(P[k][ok])
        y = np.concatenate(ys)
        stored = {r["source"]: r for r in json.loads((ROOT / f"data/manifest/eval_{split}.json").read_text()) if r["res"] == "10m"}
        for k in ALL:
            p = np.concatenate(ps[k])
            mine = dict(mae=mean_absolute_error(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
                        r2=r2_score(y, p), acc4=accuracy_score(classes(y), classes(p)),
                        f1_macro=f1_score(classes(y), classes(p), average="macro"))
            st = stored[k]
            rows.append(dict(split=split, source=k, n=int(len(y)), n_stored=st["n"],
                             **{f"{m}": float(v) for m, v in mine.items()},
                             max_abs_diff=float(max(abs(mine[m] - st[m]) for m in mine))))
    df = pd.DataFrame(rows)
    out = dict(rows=rows, worst_diff=float(df.max_abs_diff.max()), all_n_match=bool((df.n == df.n_stored).all()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    print(df.round(4).to_string(index=False))
    print("largest difference from stored results:", out["worst_diff"], "| cell counts match:", out["all_n_match"])


if __name__ == "__main__":
    main()
