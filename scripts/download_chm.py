"""Download CHM GeoTIFFs for the selected sites (data/manifest/tiles_<site>.csv).

Resumable: a tile is skipped if its file exists and opens and its last row decodes.
Writes to a .part file first so interrupted downloads never look complete.

Usage: python -I scripts/download_chm.py [workers]
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import rasterio
import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifest"
DEST = ROOT / "data/raw/fri_chm"


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as s:
            s.read(1, window=((s.height - 1, s.height), (0, s.width)))  # last row decodes
            return s.width >= 2000 and s.height >= 2000
    except rasterio.errors.RasterioIOError:
        return False


def fetch(url: str, path: Path, retries: int = 3) -> str:
    if is_complete(path):
        return "skip"
    tmp = path.with_suffix(".tif.part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(path)
            if is_complete(path):
                return "ok"
            raise IOError("downloaded file is not a readable >= 2000 px raster")
        except Exception as e:
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                return f"FAIL {e}"
    return "FAIL"


def main(workers: int = 6):
    sites = pd.read_csv(MANIFEST / "sites.csv")
    jobs = []
    for site_id in sites.site_id:
        tiles = pd.read_csv(MANIFEST / f"tiles_{site_id}.csv")
        out = DEST / site_id
        out.mkdir(parents=True, exist_ok=True)
        jobs += [(row.Download_CHM, out / f"{row.Tilename}_CHM.tif") for row in tiles.itertuples()]

    results = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, u, p): p for u, p in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            status = fut.result()
            results[futs[fut]] = status
            if status.startswith("FAIL") or i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {futs[fut].name}: {status}", flush=True)

    failed = [p for p, s in results.items() if s.startswith("FAIL")]
    print(f"done: {len(jobs) - len(failed)}/{len(jobs)} complete, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
