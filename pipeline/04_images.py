"""Step 4: download the Sentinel-2 stacks of the scenes frozen in manifests/s2_scenes.csv.

No catalogue search here: each scene is fetched by its id, so a rerun gives the same inputs. For each scene, the 12
L2A bands and the scene classification (SCL) are read on the site's label grid and written as one 13-band uint16
file (DN as delivered, 0 = no data; reflectance = DN x 0.0001). A scene whose blue band still contains the BOA
offset stops with an error instead of being converted wrongly. Files are written through a .part file and an
existing file is skipped, so the step resumes safely.

Slurm array: each task takes every n-th site.
Output: data/stacks/<site>_k<k>.tif, data/status/images/<site>.json
Usage: python -I pipeline/04_images.py [site_id ...]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, STACKS, STATUS, array_task, block_grid, grid_profile, sites, task_sites, write_json  # noqa: E402
from sentinel2 import client, items_by_id, write_stack  # noqa: E402


def main(only=None):
    s = sites()
    if only:
        s = s[s.site_id.isin(only)]
    task, n = array_task()
    mine = task_sites(list(s.itertuples()), task, n)
    scenes = pd.read_csv(MANIFESTS / "s2_scenes.csv")
    scenes = scenes[scenes.site_id.isin([r.site_id for r in mine])]
    cl = client()
    items = items_by_id(cl, scenes.scene.unique())
    print(f"task {task}/{n}: {len(mine)} sites, {len(scenes)} scenes", flush=True)
    failed = 0
    for row in mine:
        bounds = block_grid(row.zone, row.be, row.bn)
        prof = grid_profile(row.zone, bounds)
        ok, errors = [], []
        for r in scenes[scenes.site_id == row.site_id].itertuples():
            path = STACKS / f"{row.site_id}_k{r.k}.tif"
            try:
                if not path.exists():
                    if r.scene not in items:
                        raise LookupError("scene not found in the catalogue")
                    write_stack(items[r.scene], path, bounds, prof)
                ok.append(int(r.k))
            except Exception as e:  # the scene is reported; the others of the site go on
                errors.append(dict(k=int(r.k), scene=r.scene, error=f"{type(e).__name__}: {e}"[:300]))
        failed += len(errors)
        write_json(STATUS / "images" / f"{row.site_id}.json", dict(site_id=row.site_id, scenes_ok=ok, errors=errors))
        print(f"{row.site_id}: {len(ok)} scenes" + (f", {len(errors)} failed: {errors}" if errors else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
