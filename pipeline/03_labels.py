"""Step 3: 10 m canopy-cover labels from the lidar CHM, one site at a time.

For each site: get its 25 CHM tiles (from the read-only cache folders if there, else downloaded), build the label
on the site's grid, then delete the downloaded tiles (keep_chm = false), so the disk only ever holds the tiles of
the sites being processed. A site whose label and status file exist is skipped.

Label: data/labels/<site>.tif, float32, NaN = no label. Band 1: share of the cell with canopy > 2 m (the label);
band 2: > 5 m; band 3: share of the cell with valid lidar. Cover is NaN unless the whole cell has valid lidar.
Missing lidar: non-finite, the declared no-data value, -9999, -32767 or FLT_MIN (even undeclared), or a height
outside [-2, 80] m. Their count per site goes in the status file.

Slurm array: each task takes every n-th site (SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT).
Output: data/labels/<site>.tif, data/status/labels/<site>.json
Usage: python -I pipeline/03_labels.py [site_id ...]
"""
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CFG, CHM, LABELS, MANIFESTS, STATUS, array_task, block_grid, grid_profile, sites,  # noqa: E402
                    task_sites, write_json)
from lidar import build_label, cache_index, fetch_tile  # noqa: E402


def one_site(row, tiles, cache):
    sid = row.site_id
    out, status = LABELS / f"{sid}.tif", STATUS / "labels" / f"{sid}.json"
    if out.exists() and status.exists():
        return "skip"
    t0 = time.time()
    t = tiles[tiles.site_id == sid]
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda r: fetch_tile(r.tile, r.url_chm, sid, cache), t.itertuples()))
    bounds = block_grid(row.zone, row.be, row.bn)
    label, stats = build_label(row.zone, bounds, [p for p, _ in got])
    prof = grid_profile(row.zone, bounds, count=3, dtype="float32", nodata=np.nan)
    LABELS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with rasterio.open(tmp, "w", **dict(prof, predictor=3)) as d:
        d.write(label)
        for i, n in enumerate(["cover_gt2m", "cover_gt5m", "valid_frac"], 1):
            d.set_band_description(i, n)
        d.update_tags(source="Ontario FRI leaf-on lidar CHM 0.5 m (OGL-Ontario)",
                      method="share of 0.5 m CHM pixels above the threshold in each 10 m cell")
    tmp.rename(out)
    downloaded = [p for p, new in got if new]
    if not CFG["labels"]["keep_chm"]:
        for p in downloaded:
            p.unlink(missing_ok=True)
        shutil.rmtree(CHM / sid, ignore_errors=True)
    write_json(status, dict(site_id=sid, **stats, tiles=len(got), tiles_downloaded=len(downloaded),
                            minutes=round((time.time() - t0) / 60, 2)))
    return f"{stats['cells']:,} cells, mean cover {stats['mean_cover']:.2f}, partial {stats['partial_share']:.2f}, " \
           f"{len(downloaded)} tiles downloaded, {stats['missing_chm_pixels']:,} missing CHM pixels"


def main(only=None):
    s = sites()
    if only:
        s = s[s.site_id.isin(only)]
    task, n = array_task()
    mine = task_sites(list(s.itertuples()), task, n)
    tiles = pd.read_csv(MANIFESTS / "tiles.csv")
    cache = cache_index()
    print(f"task {task}/{n}: {len(mine)} sites; {len(cache):,} CHM tiles in the cache folders", flush=True)
    failed = []
    for row in mine:
        try:
            print(f"{row.site_id}: {one_site(row, tiles, cache)}", flush=True)
        except Exception as e:  # keep going; the site is reported and has no status file
            failed.append(row.site_id)
            print(f"FAILED {row.site_id}: {type(e).__name__}: {e}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
