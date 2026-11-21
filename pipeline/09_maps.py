"""Step 9: calibrated conservative, central and optimistic maps (3-band GeoTIFF) for every site.

- Test sites: mixture of the 5 final models, final per-band constants.
- Training-pool sites: out-of-fold mixture of the 5 seeds of their fold, constants from the other folds (as in the
  evaluation), so every map comes from models that never saw the site.
Bands: 1 conservative (q10 calibrated), 2 central (q50), 3 optimistic (q90 calibrated); cover x 10 000, uint16.
Output: data/maps/<site>.tif
Usage: python -I pipeline/09_maps.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, MANIFESTS, MAPS, PREDICTIONS, read_cover, write_cover  # noqa: E402
from model import calibrate, mixture  # noqa: E402

BANDS = ["conservative (q10, calibrated)", "central (q50)", "optimistic (q90, calibrated)"]


def main():
    sp = pd.read_csv(MANIFESTS / "splits.csv")
    cal = json.loads((MANIFESTS / "calibration.json").read_text())
    final = np.array(cal["bands"])
    n = 0
    for r in sp.itertuples():
        if r.split == "test":
            runs, consts = [PREDICTIONS / f"final/g{s}" for s in CFG["train"]["seeds"]], final
        else:
            runs, consts = [PREDICTIONS / f"cv/fold{r.fold}/g{s}" for s in CFG["train"]["seeds"]], \
                np.array(cal["per_fold"][str(r.fold)])
        paths = [p / f"{r.site_id}.tif" for p in runs if (p / f"{r.site_id}.tif").exists()]
        if not paths:
            continue
        q10, q50, q90 = mixture([read_cover(p) for p in paths])
        lo, hi = calibrate(q10, q50, q90, consts)
        with rasterio.open(paths[0]) as s:
            prof = s.profile
        write_cover(MAPS / f"{r.site_id}.tif", np.stack([lo, q50, hi]), prof, BANDS)
        n += 1
    print(f"{n} maps -> {MAPS}")


if __name__ == "__main__":
    main()
