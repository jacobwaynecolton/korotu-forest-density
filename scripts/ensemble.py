"""Average the per-site predictions of several trained models into a new prediction set.

Usage: python -I scripts/ensemble.py <out_name> <model> <model> [...]
Writes data/predictions/<out_name>/<site>.tif for every site all models have predicted.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data/predictions"


def main(out, models):
    (PRED / out).mkdir(parents=True, exist_ok=True)
    sites = set.intersection(*[{p.stem for p in (PRED / m).glob("*.tif")} for m in models])
    for s in sorted(sites):
        arrs = []
        for m in models:
            with rasterio.open(PRED / m / f"{s}.tif") as src:
                arrs.append(src.read(1)); prof = src.profile
        with rasterio.open(PRED / out / f"{s}.tif", "w", **prof) as d:
            d.write(np.mean(arrs, axis=0).astype(np.float32), 1)
    print(f"{out}: {len(sites)} sites from {len(models)} models")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
