"""Add sites chosen for sparse/moderate canopy cover (10-70%), which the first 12 lack.

For a spread-out pool of eligible candidate blocks, download only the centre tile's
CHM, measure the share of 10 m cells with 10-70% cover, and accept the best blocks
whose flights fall in the leaf-on window. Appends to data/manifest/sites.csv.

Usage: python -I scripts/add_sparse_sites.py [n_add] [n_probe]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sites import tile_cover  # noqa: E402
from select_sites import (BLOCK_KM, OUT, SEED, block_tiles, candidate_blocks,  # noqa: E402
                          check_dates, farthest_point_order, parse_tiles)

PROBE_DIR = Path(__file__).resolve().parents[1] / "data/raw/probe_centres"
MIN_SEP_DEG = 0.3


def centre_tile(tiles: pd.DataFrame, b) -> pd.Series:
    ce, cn = b.be + BLOCK_KM // 2, b.bn + BLOCK_KM // 2
    return tiles[(tiles.e_km == ce) & (tiles.n_km == cn)].iloc[0]


def mid_share(url: str, path: Path) -> float:
    if not path.exists():
        with requests.get(url, timeout=120) as r:
            r.raise_for_status()
            path.write_bytes(r.content)
    c = tile_cover(path)
    return float(((c >= 0.10) & (c < 0.70)).mean())


def main(n_add: int = 5, n_probe: int = 45):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    df = parse_tiles()
    blocks = candidate_blocks(df)
    sites = pd.read_csv(OUT / "sites.csv")
    taken = list(zip(sites.lon, sites.lat))

    far = lambda b, pts: all(np.hypot(b.lon - x, b.lat - y) >= MIN_SEP_DEG for x, y in pts)
    order = farthest_point_order(blocks, np.random.default_rng(SEED + 1))
    pool = []
    for i in order:
        b = blocks.loc[i]
        if far(b, taken + [(p.lon, p.lat) for p in pool]):
            pool.append(b)
        if len(pool) == n_probe:
            break

    scored = []
    for b in pool:
        tiles = block_tiles(df, b)
        t = centre_tile(tiles, b)
        try:
            s = mid_share(t.Download_CHM, PROBE_DIR / f"{t.Tilename}.tif")
        except Exception as e:
            print("probe failed", t.Tilename, e)
            continue
        scored.append((s, b))
        print(f"probe Z{int(b.zone)} E{int(b.be)} N{int(b.bn)} {int(b.year)} "
              f"lat={b.lat:.2f}: 10-70% share {s:.2f}", flush=True)

    rows = []
    for s, b in sorted(scored, key=lambda x: -x[0]):
        if len(rows) == n_add:
            break
        if not far(b, taken):
            continue
        tiles = block_tiles(df, b)
        dates = check_dates(tiles)
        if dates is None:
            print(f"reject (flight dates) share={s:.2f}")
            continue
        site_id = f"Z{int(b.zone)}_E{int(b.be):04d}_N{int(b.bn):04d}_{int(b.year)}"
        print(f"accept {site_id} share={s:.2f} flown {dates[0]:%Y-%m-%d}..{dates[1]:%Y-%m-%d}")
        taken.append((b.lon, b.lat))
        rows.append(dict(site_id=site_id, zone=int(b.zone), year=int(b.year),
                         e0_m=int(b.be) * 1000, n0_m=int(b.bn) * 1000, lon=b.lon, lat=b.lat,
                         flight_start=dates[0].isoformat(), flight_end=dates[1].isoformat(),
                         tiles=";".join(tiles.Tilename.sort_values()),
                         selection="sparse_targeted"))
        tiles.assign(site_id=site_id).to_csv(OUT / f"tiles_{site_id}.csv", index=False)

    sites["selection"] = sites.get("selection", "spread")
    sites = pd.concat([sites, pd.DataFrame(rows)], ignore_index=True)
    sites.to_csv(OUT / "sites.csv", index=False)
    print(f"added {len(rows)} sites; total {len(sites)}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
