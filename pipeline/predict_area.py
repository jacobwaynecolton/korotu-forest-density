"""Conservative, central and optimistic canopy-cover maps for any area, with the final models.

The area (lon/lat box) is put on a 10 m grid in its WGS84 / UTM zone, snapped to 60 m. Scenes are chosen with the
same rules as the training data, centred on the middle of the leaf-on window of the given year: same UTM zone, at
most max_scenes, at least min_days_apart apart, less than 1% of the area flagged by the scene classification
(no data counts as flagged, so the area must fall inside one Sentinel-2 tile). The 5 final models predict each
scene, dates are averaged over unflagged pixels, the seeds are combined as a mixture and the bounds calibrated with
the final per-band constants (manifests/calibration.json).
Output: one GeoTIFF, uint16 (cover x 10 000, 65535 = no data), bands conservative / central / optimistic.
Usage: python -I pipeline/predict_area.py --bbox -84.10 48.20 -84.00 48.27 --year 2024 --out maps/area.tif
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, MANIFESTS, MODELS, grid_profile, write_cover  # noqa: E402
from model import build, calibrate, mixture, normalise, predict  # noqa: E402
from sentinel2 import (BANDS, check_scale_and_offset, choose_scenes, client, items_by_id, read_window,  # noqa: E402
                       reflectance_and_mask, shape_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-km", type=float, default=40.0, help="refuse larger areas (memory)")
    a = ap.parse_args()
    lon0, lat0, lon1, lat1 = a.bbox
    zone = int(math.floor(((lon0 + lon1) / 2 + 180) / 6)) + 1
    snap = CFG["labels"]["snap_m"]
    l, b, r, t = transform_bounds("EPSG:4326", f"EPSG:{32600 + zone}", lon0, lat0, lon1, lat1)
    bounds = (math.floor(l / snap) * snap, math.floor(b / snap) * snap, math.ceil(r / snap) * snap, math.ceil(t / snap) * snap)
    if max(bounds[2] - bounds[0], bounds[3] - bounds[1]) > a.max_km * 1000:
        sys.exit(f"area larger than {a.max_km} km: split it, or raise --max-km")
    (m0, d0), (m1, d1) = (tuple(int(x) for x in w.split("-")) for w in CFG["sentinel2"]["window"])
    start = datetime(a.year, m0, d0, tzinfo=timezone.utc)
    end = datetime(a.year, m1, d1, 23, 59, tzinfo=timezone.utc)
    cl = client()
    scenes = choose_scenes(cl, zone, bounds, start, end)
    if not scenes:
        sys.exit("no Sentinel-2 scene of that year passes the rules for this area")
    items = items_by_id(cl, [s["scene"] for s in scenes])
    shape = shape_of(bounds)
    xs = []
    for s in scenes:
        it = items[s["scene"]]
        arr = np.stack([read_window(it.assets[k].href, bounds, shape) for _, k, _ in BANDS] +
                       [read_window(it.assets["scl"].href, bounds, shape)]).astype(np.uint16)
        check_scale_and_offset(it, arr[11])
        xs.append(arr)
    refl, bad = reflectance_and_mask(np.stack(xs))
    print(f"zone {zone}, grid {shape[1]} x {shape[0]} cells, scenes: {[s['date'] for s in scenes]}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for seed in CFG["train"]["seeds"]:
        meta = json.loads((MODELS / f"final_g{seed}.json").read_text())
        m = build(pretrained=False)
        m.load_state_dict({k: v.float() for k, v in torch.load(MODELS / f"final_g{seed}.pt", map_location="cpu").items()})
        m = m.to(dev).eval()
        x = torch.from_numpy(normalise(refl, np.array(meta["mean"]), np.array(meta["std"]))).to(dev)
        preds.append(predict(m, x, torch.from_numpy(bad).to(dev)))
    q10, q50, q90 = mixture(preds)
    consts = np.array(json.loads((MANIFESTS / "calibration.json").read_text())["bands"])
    lo, hi = calibrate(q10, q50, q90, consts)
    write_cover(a.out, np.stack([lo, q50, hi]), grid_profile(zone, bounds, count=3),
                ["conservative (q10, calibrated)", "central (q50)", "optimistic (q90, calibrated)"])
    with rasterio.open(a.out, "r+") as d:
        d.update_tags(scenes=",".join(s["scene"] for s in scenes), year=a.year)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
