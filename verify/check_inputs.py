"""Sanity of every input and label file.

For each site: label and every Sentinel-2 stack share CRS, transform and shape (same
pixel grid); label values lie in [0, 1]; reflectance is physically plausible (1st
percentile of blue >= -0.01, median NIR over cells the lidar calls forest (cover > 0.5)
in a vegetated range, so lake-heavy sites don't trip it); each scene is inside
the leaf-on window and within the stated distance of the lidar flight. This is the
check that caught two double-offset scenes (see verify/README.md).

Writes verify/results/inputs.json.
Usage: python -I verify/check_inputs.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"


def main():
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    scenes = pd.read_csv(ROOT / "data/manifest/s2_scenes.csv")
    problems, n_stacks = [], 0
    for s in sites.site_id:
        with rasterio.open(ROOT / "data/labels" / f"{s}_cover10m.tif") as src:
            grid, y = (src.crs, src.transform, src.shape), src.read(1)
        v = y[np.isfinite(y)]
        if v.min() < 0 or v.max() > 1:
            problems.append(f"{s}: label outside [0,1]")
        for p in sorted((ROOT / "data/pairs/stacks").glob(f"{s}_k*.tif")):
            n_stacks += 1
            with rasterio.open(p) as src:
                if (src.crs, src.transform, src.shape) != grid:
                    problems.append(f"{p.name}: grid differs from label")
                blue, nir = src.read(2), src.read(8)
            if np.nanpercentile(blue, 1) < -0.01:
                problems.append(f"{p.name}: negative blue reflectance (offset applied twice?)")
            f = np.nanmedian(nir[y > 0.5]) if (y > 0.5).any() else np.nan
            if not 0.1 < f < 0.6:
                problems.append(f"{p.name}: implausible forest NIR median {f:.3f}")
    d = pd.to_datetime(scenes.date)
    outside = scenes[(d.dt.month < 5) | (d.dt.month > 10)]
    md = d.dt.month * 100 + d.dt.day
    out = dict(sites=len(sites), stacks=n_stacks, scenes_listed=len(scenes),
               max_days_from_flight=int(scenes.days_from_flight.abs().max()),
               scenes_outside_may_oct=len(outside),
               scenes_outside_jun1_sep15=int(((md < 601) | (md > 915)).sum()), problems=problems)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
