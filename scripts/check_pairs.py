"""Sanity checks on data/pairs/pixels.parquet.

1. Reflectance harmonization: median red / NIR / SWIR of dense-forest cells
   (lidar cover > 0.95, WorldCover tree) should be similar across processing baselines.
2. Learnability: a quick per-pixel gradient-boosting baseline trained on train sites,
   scored on the held-out val and test sites. This is a smoke test, not a model.

Usage: python -I scripts/check_pairs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CUTS = [0.10, 0.40, 0.70]


def main():
    px = pd.read_parquet(ROOT / "data/pairs/pixels.parquet")
    m = pd.read_csv(ROOT / "data/manifest/pairs_manifest.csv").set_index("site_id")
    px = px[px.usable]

    d = px[(px.cover_gt2m > 0.95) & (px.worldcover == 10)]
    g = d.groupby("site_id", observed=True)[["B04", "B08", "B11"]].median()
    g["old_baseline"] = m.loc[g.index, "processing_baseline"].astype(float).values < 4
    print("dense-forest median reflectance by baseline group:")
    print(g.groupby("old_baseline")[["B04", "B08", "B11"]].median().round(4).to_string(), "\n")

    px = px.assign(NDVI=(px.B08 - px.B04) / (px.B08 + px.B04 + 1e-6))
    X = BANDS + ["NDVI"]
    tr = px[px.split == "train"].sample(300_000, random_state=0)
    model = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(tr[X], tr.cover_gt2m)
    rows = []
    for sp in ["val", "test"]:
        te = px[px.split == sp]
        p, y = np.clip(model.predict(te[X]), 0, 1), te.cover_gt2m.values
        land = (te.worldcover != 80).values
        rows.append(dict(split=sp, cells=len(te), mae=np.abs(p - y).mean(),
                         r=np.corrcoef(p, y)[0, 1],
                         acc4=(np.digitize(p, CUTS) == np.digitize(y, CUTS)).mean(),
                         land_mae=np.abs(p - y)[land].mean(),
                         land_acc4=(np.digitize(p, CUTS) == np.digitize(y, CUTS))[land].mean(),
                         mean_baseline_mae=np.abs(tr.cover_gt2m.mean() - y).mean()))
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(ROOT / "data/manifest/baseline_check.csv", index=False)


if __name__ == "__main__":
    main()
