"""Step 2: draw the sites from the bank, with one spacing rule and more blocks with partial cover.

1. The 64 sites we already have (the project's 49 and the 15 of the Ontario verification study) come first.
2. The rest is drawn at random (fixed seed) within each partial-cover stratum, always from the stratum furthest
   below its target share (50% high, 30% medium, 20% low by default). A candidate is accepted if:
   - its centre is at least min_spacing_km from every accepted site (the only spacing rule);
   - all its lidar flights fall in the leaf-on window, within max_flight_span_days (read from the COPC headers);
   - at least one Sentinel-2 scene passes the image rules (window, distance to the flight, < 1% bad by SCL).
   The scenes found here are the ones the pipeline will download: the scene list is frozen at this step.
3. The draw stops at `target` sites.

Every examined candidate is logged with its outcome, so an interrupted run resumes where it stopped and gives the
same result (same order, same decisions).
Outputs (manifests/): sites.csv, tiles.csv, s2_scenes.csv, selection_log.csv.
Usage: python -I pipeline/02_select_sites.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, MANIFESTS, block_grid, km_between  # noqa: E402
from lidar import block_tiles, flight_dates, read_tile_index  # noqa: E402
from sentinel2 import choose_scenes, client  # noqa: E402

S = CFG["sites"]
LOG, SITES, TILES, SCENES = (MANIFESTS / f for f in ("selection_log.csv", "sites.csv", "tiles.csv", "s2_scenes.csv"))


def quotas(n):
    q = np.floor(np.array(S["strata_shares"]) * n).astype(int)
    q[0] += n - q.sum()                       # rounding remainder to the first stratum
    return dict(zip(S["strata"], q.tolist()))


def candidate_order(bank):
    """Existing sites first, then each stratum in a random order (fixed seed)."""
    rng = np.random.default_rng(S["seed"])
    existing = bank[bank.existing].sort_values("site_id").site_id.tolist()
    per = {s: rng.permutation(bank[(bank.stratum == s) & ~bank.existing].site_id.to_numpy()).tolist()
           for s in S["strata"]}
    return existing, per


def append(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def examine(b, tiles, cl, accepted_xy):
    """Decision for one candidate block: (outcome, detail, flight dates, scenes)."""
    if accepted_xy:
        d = km_between(b.lon, b.lat, np.array([p[0] for p in accepted_xy]), np.array([p[1] for p in accepted_xy]))
        if d.min() < S["min_spacing_km"]:
            return "spacing", f"{d.min():.1f} km from an accepted site", None, None
    t = block_tiles(tiles, b.zone, b.year, b.be, b.bn)
    for attempt in range(2):
        try:
            fd = flight_dates(t)
            if isinstance(fd, str):
                return "flight_dates", fd, None, None
            scenes = choose_scenes(cl, b.zone, block_grid(b.zone, b.be, b.bn), *fd)
            if not scenes:
                return "no_scene", "no Sentinel-2 scene passes the rules", fd, None
            return "accepted", f"{len(scenes)} scenes", fd, scenes
        except Exception as e:  # network: one retry, then the candidate is logged as an error and skipped
            err = f"{type(e).__name__}: {e}"[:300]
            time.sleep(5)
    return "error", err, None, None


def main():
    bank = pd.read_csv(MANIFESTS / "bank.csv")
    bank.index = bank.site_id.to_numpy()
    tiles = read_tile_index(CFG["paths"]["tile_index"])
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=["site_id", "outcome"])
    done = dict(zip(log.site_id, log.outcome))
    acc = bank.loc[[s for s, o in done.items() if o == "accepted"]]
    accepted_xy = list(zip(acc.lon, acc.lat))
    count = acc.stratum.value_counts().to_dict()
    target, q = S["target"], quotas(S["target"])
    existing, per = candidate_order(bank)
    pointer = {s: 0 for s in per}
    cl = client()
    print(f"targets {q}; resuming with {len(acc)} accepted, {len(done)} examined", flush=True)

    def next_candidate():
        for sid in existing:
            if sid not in done:
                return sid
        open_strata = [s for s in S["strata"] if pointer[s] < len(per[s])]
        while open_strata:
            s = max(open_strata, key=lambda s: (q[s] - count.get(s, 0)) / max(q[s], 1))
            while pointer[s] < len(per[s]) and per[s][pointer[s]] in done:
                pointer[s] += 1
            if pointer[s] < len(per[s]):
                return per[s][pointer[s]]
            open_strata.remove(s)
        return None

    while sum(count.values()) < target:
        sid = next_candidate()
        if sid is None:
            print("the bank is exhausted before the target")
            break
        b = bank.loc[sid]
        outcome, detail, fd, scenes = examine(b, tiles, cl, accepted_xy)
        done[sid] = outcome
        if outcome == "accepted":
            accepted_xy.append((b.lon, b.lat))
            count[b.stratum] = count.get(b.stratum, 0) + 1
            k = CFG["sites"]["block_km"] * 1000
            append(SITES, [dict(site_id=sid, zone=b.zone, year=b.year, be=b.be, bn=b.bn, e0_m=b.be * 1000,
                                n0_m=b.bn * 1000, block_m=k, lon=round(b.lon, 5), lat=round(b.lat, 5),
                                flight_start=fd[0].isoformat(), flight_end=fd[1].isoformat(),
                                ntems_partial=b.ntems_partial, stratum=b.stratum,
                                source="existing" if b.existing else "bank")])
            t = block_tiles(tiles, b.zone, b.year, b.be, b.bn)
            append(TILES, [dict(site_id=sid, tile=r.Tilename, url_chm=r.Download_CHM) for r in t.itertuples()])
            append(SCENES, [dict(site_id=sid, k=i, **sc) for i, sc in enumerate(scenes)])
        append(LOG, [dict(site_id=sid, stratum=b.stratum, existing=bool(b.existing), outcome=outcome, detail=detail)])
        if outcome != "spacing":
            print(f"{sid} [{b.stratum}] {outcome}: {detail} | accepted {sum(count.values())}/{target} {count}",
                  flush=True)
    log = pd.read_csv(LOG)
    print("\noutcomes:\n" + log.groupby(["outcome", "stratum"]).size().unstack(fill_value=0).to_string())
    print(f"accepted sites: {sum(count.values())} {count}")


if __name__ == "__main__":
    main()
