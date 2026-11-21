"""Step 1: the bank of candidate blocks, with an estimate of how much partial canopy cover each one has.

Every 5 x 5 km block of the FRI leaf-on tile index with 25 CHM tiles, one UTM zone and one year >= min_year is a
candidate. Its partial-cover score is the share of NTEMS 2022 30 m pixels (Matasci et al. canopy cover) with
cover in [partial_low, partial_high) inside the block; NTEMS non-forest (no data) counts as 0 cover. On the
64 sites we already have, this score ranks sites like the lidar label does (rank correlation 0.75), which is
enough to stratify the draw. Strata are terciles of the score over all candidates: high, medium, low.
NTEMS is never used as a label.

Inputs: the tile index and the NTEMS raster (paths in config.toml; downloaded if missing).
Output: manifests/bank.csv (one row per candidate block).
Usage: python -I pipeline/01_bank.py
"""
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, CSRS_UTM_EPSG, MANIFESTS, REPO, site_id  # noqa: E402
from lidar import eligible_blocks, read_tile_index  # noqa: E402

P, S = CFG["paths"], CFG["sites"]


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, 1 << 20)
    tmp.rename(path)


def ensure_inputs():
    index, ntems = Path(P["tile_index"]), Path(P["ntems"])
    if not index.exists():
        print("downloading the FRI tile index...", flush=True)
        download(P["tile_index_url"], index)
    if not ntems.exists():
        z = ntems.with_suffix(".zip")
        if not z.exists():
            print("downloading NTEMS 2022 canopy cover...", flush=True)
            download(P["ntems_url"], z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ntems.parent)
    return index, ntems


def ntems_scores(blocks, ntems_path):
    """Share of NTEMS pixels with partial cover, and mean NTEMS cover, inside each block."""
    k = S["block_km"] * 1000
    part, mean = np.full(len(blocks), np.nan), np.full(len(blocks), np.nan)
    with rasterio.open(ntems_path) as src:
        nd = src.nodata
        for i, b in enumerate(blocks.itertuples()):
            bb = transform_bounds(f"EPSG:{CSRS_UTM_EPSG[b.zone]}", src.crs, b.be * 1000, b.bn * 1000,
                                  b.be * 1000 + k, b.bn * 1000 + k)
            w = from_bounds(*bb, transform=src.transform).round_offsets().round_lengths()
            a = src.read(1, window=w, boundless=True, fill_value=nd).astype(np.float32)
            a = np.where((a == nd) | ~np.isfinite(a) | (a < 0) | (a > 100), 0.0, a) / 100.0
            part[i] = ((a >= S["partial_low"]) & (a < S["partial_high"])).mean()
            mean[i] = a.mean()
            if i % 2000 == 0:
                print(f"  {i:,}/{len(blocks):,} blocks scored", flush=True)
    return part, mean


def main():
    index, ntems = ensure_inputs()
    blocks = eligible_blocks(read_tile_index(index))
    blocks["site_id"] = [site_id(b.zone, b.be, b.bn, b.year) for b in blocks.itertuples()]
    print(f"{len(blocks):,} eligible blocks (25 CHM tiles, one zone, year >= {S['min_year']})", flush=True)
    blocks["ntems_partial"], blocks["ntems_mean"] = ntems_scores(blocks, ntems)
    # terciles of the score: the third of blocks with the most partial cover is "high"
    blocks["stratum"] = pd.qcut(blocks.ntems_partial.rank(method="first"), 3, labels=S["strata"][::-1]).astype(str)
    existing = set()
    for f in P["existing_sites"]:
        existing |= set(pd.read_csv(REPO / f).site_id)
    blocks["existing"] = blocks.site_id.isin(existing)
    missing = existing - set(blocks.site_id)
    if missing:
        print(f"warning: {len(missing)} existing sites are not eligible blocks and will be skipped: {sorted(missing)}")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    cols = ["site_id", "zone", "year", "be", "bn", "lon", "lat", "ntems_partial", "ntems_mean", "stratum", "existing"]
    blocks[cols].round(5).to_csv(MANIFESTS / "bank.csv", index=False)
    print(blocks.groupby("stratum").ntems_partial.agg(["size", "min", "max"]).round(3))
    print(f"{int(blocks.existing.sum())} existing sites found in the bank -> {MANIFESTS / 'bank.csv'}")


if __name__ == "__main__":
    main()
