"""From an area of interest (lon/lat box) to calibrated conservative / central / optimistic canopy-cover maps.

Same rules and model as the final pipeline (pipeline/config.toml), applied to the most recent leaf-on season:
  - scenes from the pipeline's Sentinel-2 collection, between 1 June and 15 September of the most recent summer
    (the current one if it has started, up to today), most recent first, at least 5 days apart, with less than 1%
    of the area flagged by the scene classification (cloud, shadow, cirrus, snow, saturated, no data: the area must
    lie inside one Sentinel-2 tile). If no scene of that summer passes, the previous summers are tried;
  - reflectance = DN x 0.0001, with the pipeline's alarm if a scene still carries the BOA offset;
  - the 5 final U-Nets (pipeline/data/models/final_g*.pt), each averaging dates pixel by pixel over unflagged
    dates and 4 flips; the 5 predictions combined as a mixture; bounds calibrated with the final per-band constants.
Inference runs on CPU in one background process that keeps the 5 networks loaded. They run one after the other
with 8 threads each: apps0 allows 8 CPUs per user, so running them side by side would only make each one slower.
"""
import json
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from rasterio.warp import transform as warp_points
from shapely.geometry import Point, shape

APP = Path(__file__).resolve().parent
REPO = APP.parent
sys.path.insert(0, str(REPO / "pipeline"))
from common import CFG, MANIFESTS, MODELS, NODATA_U16, SCALE_U16, write_cover  # noqa: E402
from model import calibrate, mixture  # noqa: E402
from sentinel2 import BANDS, check_scale_and_offset, client, read_window, reflectance_and_mask  # noqa: E402

S2 = CFG["sentinel2"]
MAX_KM = 20.0                 # longest side of an area
MAX_CANDIDATES = 30           # scenes whose cloud mask is read, per season and UTM zone
SEASONS_BACK = 3              # most recent summer, then up to two earlier ones
THREADS_PER_MODEL = 8
SEEDS = CFG["train"]["seeds"]
ONTARIO = shape({"type": "MultiPolygon", "coordinates": [[ring] for ring in
                 json.loads((REPO / "report/ontario_outline.json").read_text())["ontario"]]})


class AnalysisError(Exception):
    """A problem the user can act on (area too large, no clean image...)."""


# ---------- area and grid ----------
def km_size(bbox):
    w, s, e, n = bbox
    lat = math.radians((s + n) / 2)
    return 6371.0 * math.radians(e - w) * math.cos(lat), 6371.0 * math.radians(n - s)


def check_bbox(bbox):
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
        raise AnalysisError("The corners do not form a valid area (west < east, south < north).")
    kx, ky = km_size(bbox)
    if max(kx, ky) > MAX_KM:
        raise AnalysisError(f"The area is {kx:.1f} x {ky:.1f} km; the limit is {MAX_KM:.0f} km on each side.")
    if min(kx, ky) < 0.5:
        raise AnalysisError("The area is smaller than 500 m on one side; draw a larger area.")
    return kx, ky


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def grid_for(bbox, epsg):
    """Box -> grid bounds in the UTM CRS, snapped outward to 60 m so that every Sentinel-2 pixel lines up."""
    s = CFG["labels"]["snap_m"]
    l, b, r, t = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox, densify_pts=21)
    return (math.floor(l / s) * s, math.floor(b / s) * s, math.ceil(r / s) * s, math.ceil(t / s) * s)


def shape_of(bounds):
    c = CFG["labels"]["cell_m"]
    return int(round((bounds[3] - bounds[1]) / c)), int(round((bounds[2] - bounds[0]) / c))


# ---------- scenes ----------
def seasons(today, year=None):
    (m0, d0), (m1, d1) = (tuple(int(x) for x in w.split("-")) for w in S2["window"])
    now = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)
    first = today.year if today >= date(today.year, m0, d0) else today.year - 1
    years = [year] if year else [first - k for k in range(SEASONS_BACK)]
    return [(y, datetime(y, m0, d0, tzinfo=timezone.utc), min(datetime(y, m1, d1, 23, 59, 59, tzinfo=timezone.utc), now))
            for y in years]


def item_epsg(it):
    code = it.properties.get("proj:code") or f"EPSG:{it.properties.get('proj:epsg')}"
    return int(str(code).split(":")[-1])


def find_scenes(cl, bbox, max_scenes, year, progress):
    """Most recent clean scenes of the most recent summer that has some: (year, epsg, bounds, [(item, bad share)])."""
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    tried = []
    for y, start, end in seasons(date.today(), year):
        progress(f"Searching Sentinel-2 scenes of summer {y}", 0.05)
        items = list(cl.search(collections=[S2["collection"]], bbox=bbox,
                               datetime=f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
                               query={"eo:cloud_cover": {"lt": 50}}, max_items=500).items())
        by_epsg = {}
        for it in items:
            by_epsg.setdefault(item_epsg(it), []).append(it)
        order = sorted(by_epsg, key=lambda e: (e != utm_epsg(lon, lat), -len(by_epsg[e])))
        checked = 0
        for epsg in order:
            bounds = grid_for(bbox, epsg)
            shp = shape_of(bounds)
            kept = []
            for it in sorted(by_epsg[epsg], key=lambda i: i.datetime, reverse=True)[:MAX_CANDIDATES]:
                if len(kept) == max_scenes:
                    break
                if any(abs((it.datetime - k.datetime).days) < S2["min_days_apart"] for k, _ in kept):
                    continue
                checked += 1
                progress(f"Summer {y}: checking clouds on {it.datetime:%d %b %Y} ({len(kept)} clean so far)",
                         0.05 + 0.25 * min(checked / 12, 1))
                scl = read_window(it.assets["scl"].href, bounds, shp)
                bad = float(np.isin(scl, S2["bad_scl"]).mean())
                if bad < S2["max_bad_fraction"]:
                    kept.append((it, bad))
            if kept:
                return y, epsg, bounds, kept, tried
        tried.append(y)
    raise AnalysisError(
        f"No Sentinel-2 image of summer {' / '.join(map(str, tried))} has under 1% cloud over this area, inside a single "
        "Sentinel-2 tile. Try a smaller area, or move it slightly.")


def read_scene(it, bounds):
    shp = shape_of(bounds)
    keys = [k for _, k, _ in BANDS] + ["scl"]
    with ThreadPoolExecutor(8) as ex:
        arrs = list(ex.map(lambda k: read_window(it.assets[k].href, bounds, shp), keys))
    x = np.stack(arrs).astype(np.uint16)
    check_scale_and_offset(it, x[11])
    return x


# ---------- inference (worker processes) ----------
_MODELS = {}


def _init_worker():
    import torch
    torch.set_num_threads(THREADS_PER_MODEL)


def _predict_one(seed, x_path, bad_path):
    import torch
    from model import build, normalise, predict
    if seed not in _MODELS:
        meta = json.loads((MODELS / f"final_g{seed}.json").read_text())
        m = build(pretrained=False)
        m.load_state_dict({k: v.float() for k, v in torch.load(MODELS / f"final_g{seed}.pt", map_location="cpu").items()})
        _MODELS[seed] = (m.eval(), np.array(meta["mean"]), np.array(meta["std"]))
    m, mean, std = _MODELS[seed]
    refl = np.load(x_path)
    bad = np.load(bad_path)
    H, W = refl.shape[-2:]
    t = CFG["train"]["tile"]
    ph, pw = max(0, t - H), max(0, t - W)
    if ph or pw:   # areas smaller than one 256 px window: mirror them up to the window size, then crop back
        refl = np.pad(refl, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
        bad = np.pad(bad, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return predict(m, torch.from_numpy(normalise(refl, mean, std)), torch.from_numpy(bad))[:, :H, :W]


def warm(seed):
    """Load one model in a worker process with a tiny prediction, so the first analysis does not wait for it."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/x.npy", np.full((1, 12, 40, 40), 0.05, np.float32))
        np.save(f"{d}/b.npy", np.zeros((1, 40, 40), bool))
        _predict_one(seed, f"{d}/x.npy", f"{d}/b.npy")
    return seed


POOL = None


def pool():
    global POOL
    if POOL is None:
        import multiprocessing as mp
        POOL = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=_init_worker)
    return POOL


# ---------- outputs ----------
def true_colour(x):
    """(13, H, W) uint16 -> RGB uint8, bands 4/3/2, 2-98% stretch over the area."""
    rgb = x[[3, 2, 1]].astype(np.float32)
    out = np.zeros(rgb.shape, np.uint8)
    for i in range(3):
        v = rgb[i][rgb[i] > 0]
        if v.size:
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            out[i] = (np.clip((rgb[i] - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    return out.transpose(1, 2, 0)


def to_u16(a):
    return np.where(np.isfinite(a), np.round(np.clip(a, 0, 1) * SCALE_U16), NODATA_U16).astype("<u2")


def web_mercator(maps, epsg, bounds):
    """Maps reprojected to EPSG:3857 for the overlay on the web map, with their lat/lon bounds."""
    H, W = maps.shape[1:]
    src_tf = rasterio.transform.from_origin(bounds[0], bounds[3], 10, 10)
    tf, w, h = calculate_default_transform(f"EPSG:{epsg}", "EPSG:3857", W, H, *bounds)
    out = np.full((3, h, w), NODATA_U16, np.uint16)
    for i in range(3):
        reproject(to_u16(maps[i]), out[i], src_transform=src_tf, src_crs=f"EPSG:{epsg}", src_nodata=NODATA_U16,
                  dst_transform=tf, dst_crs="EPSG:3857", dst_nodata=NODATA_U16, resampling=Resampling.nearest)
    l, t = tf * (0, 0)
    r, b = tf * (w, h)
    lons, lats = warp_points("EPSG:3857", "EPSG:4326", [l, r], [b, t])
    return out, w, h, [[lats[0], lons[0]], [lats[1], lons[1]]]


def run(job_dir, bbox, max_scenes=4, year=None, progress=lambda m, p: None):
    """Full analysis; writes outputs into job_dir and returns the result metadata."""
    t0 = time.time()
    kx, ky = check_bbox(bbox)
    warnings = []
    centre = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    if not ONTARIO.contains(centre):
        warnings.append("This area is outside Ontario. The model was trained on Ontario's managed forest; elsewhere "
                        "its maps can be much less accurate, especially in open landscapes with low green vegetation.")
    cl = client()
    y, epsg, bounds, kept, tried = find_scenes(cl, bbox, max_scenes, year, progress)
    if tried:
        warnings.append(f"No clean image in summer {' / '.join(map(str, tried))}; summer {y} was used instead.")
    if len(kept) < max_scenes:
        warnings.append(f"Only {len(kept)} clean image{'s' if len(kept) > 1 else ''} in summer {y}: the maps rest on "
                        "fewer dates than in training (up to 4).")
    t_search = time.time()

    progress(f"Reading {len(kept)} image{'s' if len(kept) > 1 else ''} (12 bands + cloud mask)", 0.35)
    with ThreadPoolExecutor(4) as ex:
        xs = list(ex.map(lambda k: read_scene(k[0], bounds), kept))
    stack = np.stack(xs)
    refl, bad = reflectance_and_mask(stack)
    valid = np.isfinite(refl).all(axis=1).any(axis=0)
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="canopy_"))       # local disk: the model processes read the inputs from here
    try:
        np.save(tmp / "refl.npy", refl.astype(np.float32))
        np.save(tmp / "bad.npy", bad)
        t_read = time.time()
        futures = [pool().submit(_predict_one, s, str(tmp / "refl.npy"), str(tmp / "bad.npy")) for s in SEEDS]
        preds = []
        for i, f in enumerate(futures):
            progress(f"Running the final models ({i + 1} of {len(SEEDS)})", 0.4 + 0.5 * i / len(SEEDS))
            preds.append(f.result())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    t_model = time.time()

    progress("Combining the models and calibrating the maps", 0.9)
    q10, q50, q90 = mixture(preds)
    consts = np.array(json.loads((MANIFESTS / "calibration.json").read_text())["bands"])
    lo, hi = calibrate(q10, q50, q90, consts)
    maps = np.stack([lo, q50, hi])
    maps[:, ~valid] = np.nan
    H, W = maps.shape[1:]
    l, b, r, t = bounds
    prof = dict(driver="GTiff", width=W, height=H, count=3, dtype="float32", crs=f"EPSG:{epsg}",
                transform=rasterio.transform.from_origin(l, t, 10, 10), nodata=np.nan, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    write_cover(job_dir / "canopy_cover.tif", maps, prof,
                ["conservative (q10, calibrated)", "central (q50)", "optimistic (q90, calibrated)"])
    to_u16(maps).tofile(job_dir / "maps.u16")
    ov, ow, oh, ob = web_mercator(maps, epsg, bounds)
    ov.astype("<u2").tofile(job_dir / "overlay.u16")
    scenes = []
    for i, ((it, badf), x) in enumerate(zip(kept, xs)):
        Image.fromarray(true_colour(x)).save(job_dir / f"rgb_{i}.jpg", quality=88)
        scenes.append(dict(index=i, id=it.id, date=it.datetime.date().isoformat(), flagged_share=round(badf, 5),
                           tile_cloud_cover=it.properties.get("eo:cloud_cover")))
    lons, lats = warp_points(f"EPSG:{epsg}", "EPSG:4326", [l, r, r, l], [t, t, b, b])
    stats = {name: dict(mean=float(np.nanmean(maps[i])), valid_cells=int(np.isfinite(maps[i]).sum()))
             for i, name in enumerate(("conservative", "central", "optimistic"))}
    result = dict(bbox=bbox, size_km=[round(kx, 2), round(ky, 2)], year=y, epsg=epsg, bounds=list(bounds), width=W,
                  height=H, cell_m=CFG["labels"]["cell_m"], corners=[[la, ln] for la, ln in zip(lats, lons)],
                  scenes=scenes, warnings=warnings, stats=stats,
                  overlay=dict(width=ow, height=oh, bounds=ob),
                  seconds=dict(search=round(t_search - t0, 1), read=round(t_read - t_search, 1),
                               model=round(t_model - t_read, 1), total=round(time.time() - t0, 1)))
    (job_dir / "result.json").write_text(json.dumps(result, indent=1))
    progress("Done", 1.0)
    return result
