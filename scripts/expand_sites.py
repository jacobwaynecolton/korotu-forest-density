"""Add more TRAINING sites, spread out, kept away from validation and test sites.

New blocks come from the same eligible pool (select_sites.candidate_blocks) with the
same flight-date rules. They must be >= HOLDOUT_SEP_DEG from every val/test site
(spatial independence of the evaluation) and >= MIN_SEP_DEG from every existing site.
Appended to data/manifest/sites.csv with split=train, selection=expansion.

Usage: python -I scripts/expand_sites.py [n_add]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_sites import (OUT, SEED, block_tiles, candidate_blocks, check_dates,  # noqa: E402
                          farthest_point_order, parse_tiles)

HOLDOUT_SEP_DEG = 0.5
MIN_SEP_DEG = 0.3


def main(n_add: int = 33):
    df = parse_tiles()
    blocks = candidate_blocks(df)
    sites = pd.read_csv(OUT / "sites.csv")
    hold = sites[sites.split.isin(["val", "test"])][["lon", "lat"]].values
    taken = [tuple(x) for x in sites[["lon", "lat"]].values]

    def dist(b, pts):
        return np.min(np.hypot(b.lon - pts[:, 0], b.lat - pts[:, 1])) if len(pts) else np.inf

    rows = []
    for i in farthest_point_order(blocks, np.random.default_rng(SEED + 2)):
        if len(rows) == n_add:
            break
        b = blocks.loc[i]
        if dist(b, hold) < HOLDOUT_SEP_DEG or dist(b, np.array(taken)) < MIN_SEP_DEG:
            continue
        tiles = block_tiles(df, b)
        try:
            dates = check_dates(tiles)
        except Exception as e:
            print("skip", e)
            continue
        if dates is None:
            continue
        site_id = f"Z{int(b.zone)}_E{int(b.be):04d}_N{int(b.bn):04d}_{int(b.year)}"
        taken.append((b.lon, b.lat))
        rows.append(dict(site_id=site_id, zone=int(b.zone), year=int(b.year),
                         e0_m=int(b.be) * 1000, n0_m=int(b.bn) * 1000, lon=b.lon, lat=b.lat,
                         flight_start=dates[0].isoformat(), flight_end=dates[1].isoformat(),
                         tiles=";".join(tiles.Tilename.sort_values()),
                         selection="expansion", split="train"))
        tiles.assign(site_id=site_id).to_csv(OUT / f"tiles_{site_id}.csv", index=False)
        print(f"accept {site_id} lon={b.lon:.2f} lat={b.lat:.2f} "
              f"flown {dates[0]:%Y-%m-%d}..{dates[1]:%Y-%m-%d}", flush=True)
    sites = pd.concat([sites, pd.DataFrame(rows)], ignore_index=True)
    sites.to_csv(OUT / "sites.csv", index=False)
    print(f"added {len(rows)}; total {len(sites)} "
          f"({(sites.split == 'train').sum()} train)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 33)
