"""From an area of interest (lon/lat box) to calibrated conservative / central / optimistic canopy-cover maps.

Same model as the final pipeline (pipeline/config.toml), applied to the most recent leaf-on season:
  - scenes from the pipeline's Sentinel-2 collection, between 1 June and 15 September of the most recent summer
    (the current one if it has started, up to today), inside one Sentinel-2 tile. Seasonal choice (A below): up to
    6 scenes, 3 before 1 August and 3 from 1 August, each with at most 10% of the area flagged by the scene
    classification (cloud, shadow, cirrus, snow, saturated, no data), cleanest first, at least 5 days apart. With
    fewer than 4 such scenes, the pipeline's rule: up to 4 scenes, most recent first, under 1% flagged. If no scene
    of that summer passes, the previous summers are tried;
  - reflectance = DN x 0.0001, with the pipeline's alarm if a scene still carries the BOA offset;
  - the 5 final U-Nets (pipeline/data/models/final_g*.pt) predict each date (4 flips). Base maps: each network
    averages the dates pixel by pixel over unflagged dates, and the 5 networks are combined as a mixture, as in the
    pipeline. With the seasonal choice, date-disagreement rule: where the central value of two clear dates differs by
    more than 0.35 and at least 2 clear dates are below 0.3 (typically fields, read as canopy while the crop is dense
    and green), the maps of the date with the lowest central value (mixture of the 5 networks) are used. Bounds
    calibrated with the final per-band constants.
  The seasonal choice and the rule were tested on 20 held-out test sites with fields and grassland (summer of the
  lidar flight): mean absolute error 0.0975 against 0.1020 with the pipeline's image rule, 0.026 against 0.055 on
  cropland (ESA WorldCover).
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
MAX_CANDIDATES = 60           # scenes whose cloud mask is read, per season and UTM zone (least cloudy tiles first)
# Image choice and date rule (same names and values as the deliverable's config.toml [app])
A = dict(season_scenes=6, season_per_half=3, season_split="08-01", season_max_bad=0.10, season_min_scenes=4, max_scenes=4,
         rule_spread=0.35, rule_low=0.3, rule_min_low=2)
SPLIT = tuple(int(v) for v in A["season_split"].split("-"))
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


def _far_enough(it, kept):
    return all(abs((it.datetime - k.datetime).days) >= S2["min_days_apart"] for k, _ in kept)


def choose_scenes(cands, year):
    """[(item, flagged share)] of one summer -> (chosen [(item, share)] sorted by date, mode).

    'seasonal': up to season_scenes scenes with at most season_max_bad flagged, season_per_half before season_split and
    the rest from it, cleanest first, >= 5 days apart, filled from the other half if one half lacks; used when at
    least season_min_scenes qualify. Otherwise 'standard', the pipeline's rule: max_scenes most recent under 1%."""
    ok = sorted([c for c in cands if c[1] <= float(A["season_max_bad"])], key=lambda c: (c[1], -c[0].datetime.timestamp()))
    cut = datetime(year, *SPLIT, tzinfo=timezone.utc)
    kept = []
    for half in ([c for c in ok if c[0].datetime < cut], [c for c in ok if c[0].datetime >= cut]):
        n = 0
        for c in half:
            if n < int(A["season_per_half"]) and _far_enough(c[0], kept):
                kept.append(c); n += 1
    for c in ok:
        if len(kept) < int(A["season_scenes"]) and c not in kept and _far_enough(c[0], kept):
            kept.append(c)
    if len(kept) >= int(A["season_min_scenes"]):
        return sorted(kept, key=lambda c: c[0].datetime), "seasonal"
    kept = []
    for c in sorted(cands, key=lambda c: c[0].datetime, reverse=True):
        if len(kept) < int(A["max_scenes"]) and c[1] < S2["max_bad_fraction"] and _far_enough(c[0], kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c[0].datetime), "standard"


def find_scenes(cl, bbox, year, progress):
    """Scenes of the most recent summer that has some: (year, epsg, bounds, [(item, flagged share)], mode, tried)."""
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
        for epsg in order:
            bounds = grid_for(bbox, epsg)
            shp = shape_of(bounds)
            todo = sorted(by_epsg[epsg], key=lambda i: i.properties.get("eo:cloud_cover", 100))[:MAX_CANDIDATES]
            progress(f"Summer {y}: checking clouds on {len(todo)} candidate images", 0.08)

            def flagged(it):
                scl = read_window(it.assets["scl"].href, bounds, shp)
                return it, float(np.isin(scl, S2["bad_scl"]).mean())
            with ThreadPoolExecutor(8) as ex:
                cands = list(ex.map(flagged, todo))
            kept, mode = choose_scenes(cands, y)
            if kept:
                return y, epsg, bounds, kept, mode, tried
        tried.append(y)
    raise AnalysisError(
        f"No Sentinel-2 image of summer {' / '.join(map(str, tried))} is clear enough over this area, inside a single "
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
    x = torch.from_numpy(normalise(refl, mean, std))
    b = torch.from_numpy(bad)
    # each date on its own: (dates, 3, H, W). Averaging them over the clear dates gives the pipeline's prediction.
    return np.stack([predict(m, x[d:d + 1], b[d:d + 1])[:, :H, :W] for d in range(x.shape[0])]).astype(np.float32)


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


# ---------- combining dates and networks ----------
def _mix(preds, rows=256):
    """mixture() by blocks of rows, to bound memory on large areas."""
    H = preds[0].shape[-2]
    return np.concatenate([mixture([p[..., r:r + rows, :] for p in preds]) for r in range(0, H, rows)], axis=-2)


def combine(per_seed, good, rule):
    """per_seed: one (dates, 3, H, W) array per network; good: (dates, H, W) clear pixels.

    Base: each network averages the clear dates (all dates where none is clear), then the networks are mixed: the
    pipeline's prediction. With rule, where the clear dates disagree (central value spread > rule_spread) and at least
    rule_min_low clear dates are below rule_low, the maps of the date with the lowest central value (networks mixed)
    replace the base. Returns (3, H, W) uncalibrated q10, q50, q90 and the (H, W) mask where the rule applied."""
    g = good.astype(np.float32)
    g = np.where(g.sum(0, keepdims=True) == 0, 1.0, g)[:, None]                   # (dates, 1, H, W)
    q = _mix([(p * g).sum(0) / g.sum(0) for p in per_seed])
    applied = np.zeros(good.shape[1:], bool)
    if not rule or good.shape[0] < 2:
        return q, applied
    by_date = np.stack([_mix([p[d] for p in per_seed]) for d in range(good.shape[0])])   # (dates, 3, H, W)
    q50 = np.where(good, by_date[:, 1], np.nan)
    with np.errstate(all="ignore"):
        spread = np.nan_to_num(np.nanmax(q50, 0) - np.nanmin(q50, 0))
    low = (q50 < float(A["rule_low"])).sum(0)                                      # NaN compares False
    applied = (spread > float(A["rule_spread"])) & (low >= int(A["rule_min_low"]))
    lowest = np.argmin(np.where(np.isfinite(q50), q50, np.inf), axis=0)
    pick = np.take_along_axis(by_date, lowest[None, None], axis=0)[0]
    return np.where(applied[None], pick, q), applied


def run(job_dir, bbox, year=None, progress=lambda m, p: None):
    """Full analysis; writes outputs into job_dir and returns the result metadata."""
    t0 = time.time()
    kx, ky = check_bbox(bbox)
    warnings = []
    centre = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    if not ONTARIO.contains(centre):
        warnings.append("This area is outside Ontario. The model was trained on Ontario's managed forest; elsewhere "
                        "its maps can be much less accurate, especially in open landscapes with low green vegetation.")
    cl = client()
    y, epsg, bounds, kept, mode, tried = find_scenes(cl, bbox, year, progress)
    if tried:
        warnings.append(f"No clean image in summer {' / '.join(map(str, tried))}; summer {y} was used instead.")
    if mode == "standard":
        warnings.append(f"Only {len(kept)} image{'s' if len(kept) > 1 else ''} under 1% cloud in summer {y}, and fewer "
                        f"than {A['season_min_scenes']} under {float(A['season_max_bad']):.0%}: the dates are averaged "
                        "without the correction for fields, which may be read as forest early in the summer.")
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

    progress("Combining the dates and the models, calibrating the maps", 0.9)
    (q10, q50, q90), applied = combine(preds, ~bad, rule=mode == "seasonal")
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
                  scenes=scenes, warnings=warnings, stats=stats, image_rule=mode,
                  corrected_share=round(float(applied[valid].mean()) if valid.any() else 0.0, 4),
                  overlay=dict(width=ow, height=oh, bounds=ob),
                  seconds=dict(search=round(t_search - t0, 1), read=round(t_read - t_search, 1),
                               model=round(t_model - t_read, 1), total=round(time.time() - t0, 1)))
    (job_dir / "result.json").write_text(json.dumps(result, indent=1))
    progress("Done", 1.0)
    return result
