"""No leakage between training and held-out sites.

Checks: (1) no site footprints overlap, (2) no lidar tile is used by two sites,
(3) distance from every validation/test site to its nearest training site,
(4) the U-Net's normalisation statistics and checkpoint selection use only train/val
sites (read straight from the training code).

Writes verify/results/splits.json.
Usage: python -I verify/check_splits.py
"""
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify/results"


def main():
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    fp = gpd.read_file(ROOT / "data/manifest/sites.geojson")[["site_id", "geometry"]].merge(sites[["site_id", "split", "lat", "lon"]])
    overlaps = [(a.site_id, b.site_id) for i, a in fp.iterrows() for j, b in fp.iterrows()
                if i < j and a.geometry.intersection(b.geometry).area > 0]
    tiles = sites.tiles.str.split(";").explode()
    shared = tiles[tiles.duplicated(keep=False)].unique().tolist()

    m = fp.to_crs("EPSG:3978")  # Canada Atlas Lambert, metres
    train = m[m.split == "train"]
    near = []
    for r in m[m.split != "train"].itertuples():
        d = train.geometry.distance(r.geometry)
        k = d.idxmin()
        deg = float(np.hypot(sites.lat[sites.site_id == r.site_id].item() - train.loc[k, "lat"],
                             sites.lon[sites.site_id == r.site_id].item() - train.loc[k, "lon"]))
        near.append(dict(site_id=r.site_id, split=r.split, nearest_train=train.loc[k, "site_id"],
                         gap_km=float(d.min() / 1000), centre_deg=deg))

    code = (ROOT / "scripts/train_unet.py").read_text()
    code_checks = dict(
        norm_stats_from_train_only=bool(re.search(r"mean, std = norm_stats\(train_ids\)", code)),
        train_crops_from_train_only="train_ids = [s for s in sites.site_id if split[s] == \"train\"]" in code,
        checkpoint_selected_on_val='v = evaluate("val")' in code and "if v < best" in code,
        test_predicted_once_after_selection=code.count('evaluate("test"') == 1,
    )
    out = dict(n_sites=len(sites), split_counts=sites.split.value_counts().to_dict(),
               footprint_overlaps=overlaps, shared_tiles=shared, nearest_train=near,
               min_gap_km=min(n["gap_km"] for n in near), code_checks=code_checks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "nearest_train"}, indent=1))
    print(pd.DataFrame(near).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
