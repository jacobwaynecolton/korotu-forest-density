"""Rebuild the multi-date Sentinel-2 stacks from the exact scene list in data/manifest/s2_scenes.csv.

make_stacks.py searches the catalogue again and skips every site already listed in
s2_scenes.csv, so it cannot rebuild the committed dataset. This script reads the committed
list and writes the same files (data/pairs/stacks/<site>_k<k>.tif) with make_stacks.write_stack,
so the band order, reflectance conversion and BOA-offset check are the project's own.

Resumable: a stack that already exists is skipped; each file is written to a .part file
first, so an interrupted job never leaves a file that looks complete.

Usage: python -I repro/stacks_from_list.py [site_id ...]
"""
import sys
from pathlib import Path

import pandas as pd
import rasterio
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_stacks import OUT, STAC, write_stack  # noqa: E402


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = pd.read_csv(ROOT / "data/manifest/s2_scenes.csv")
    if only:
        scenes = scenes[scenes.site_id.isin(only)]
    client = Client.open(STAC)
    written, skipped, failed = 0, 0, []
    for r in scenes.itertuples():
        path = OUT / f"{r.site_id}_k{r.k}.tif"
        if path.exists():
            skipped += 1
            continue
        tmp = path.with_suffix(".tif.part")
        try:
            with rasterio.open(ROOT / "data/labels" / f"{r.site_id}_cover10m.tif") as s:
                prof, bounds, shape = s.profile, s.bounds, (s.height, s.width)
            items = list(client.search(ids=[r.scene], collections=["sentinel-2-l2a"]).items())
            if not items:
                raise LookupError("scene not found in the catalogue")
            write_stack(items[0], tmp, bounds, prof, shape)
            tmp.rename(path)
            written += 1
            print(f"{r.site_id} k{r.k}: {r.scene}", flush=True)
        except Exception as e:  # keep going, report at the end
            tmp.unlink(missing_ok=True)
            failed.append((r.site_id, r.k, r.scene, f"{type(e).__name__}: {e}"))
            print(f"FAIL {r.site_id} k{r.k} {r.scene}: {e}", flush=True)
    n = len(list(OUT.glob("*_k*.tif")))
    print(f"written {written}, already present {skipped}, failed {len(failed)}; {n} stack files in {OUT}")
    for f in failed:
        print("  failed:", *f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
