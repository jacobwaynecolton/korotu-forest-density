"""Sentinel-2 L2A: scene choice, aligned reads and multi-date stacks on the label grid."""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from common import CFG, makedirs, s2_epsg

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

C = CFG["sentinel2"]
# (band, earth-search asset key, native resolution m), in the order of the stack
BANDS = [("B01", "coastal", 60), ("B02", "blue", 10), ("B03", "green", 10), ("B04", "red", 10),
         ("B05", "rededge1", 20), ("B06", "rededge2", 20), ("B07", "rededge3", 20),
         ("B08", "nir", 10), ("B8A", "nir08", 20), ("B09", "nir09", 60),
         ("B11", "swir16", 20), ("B12", "swir22", 20)]
SCL_BAND = len(BANDS)          # index of the SCL band in a stack (13th band)


def client():
    return Client.open(C["stac"])


def read_window(href, bounds, shape, resampling=Resampling.nearest):
    """One band on the label grid. 20 m and 60 m bands are repeated onto 10 m (nearest): no invented values.
    The grid is snapped to 60 m, so the window starts on a whole pixel of every band."""
    with rasterio.open(href) as s:
        w = from_bounds(*bounds, transform=s.transform)
        assert abs(w.col_off - round(w.col_off)) < 1e-6 and abs(w.row_off - round(w.row_off)) < 1e-6, \
            f"window not pixel-aligned: {w}"
        w = w.round_offsets().round_lengths()
        return s.read(1, window=w, out_shape=shape, resampling=resampling, boundless=True, fill_value=0)


def shape_of(bounds):
    cell = CFG["labels"]["cell_m"]
    l, b, r, t = bounds
    return int(round((t - b) / cell)), int(round((r - l) / cell))


def _window(mid):
    (m0, d0), (m1, d1) = (tuple(int(x) for x in w.split("-")) for w in C["window"])
    lo = max(mid - timedelta(days=C["days_around_flight"]), mid.replace(month=m0, day=d0, hour=0, minute=0, second=0))
    hi = min(mid + timedelta(days=C["days_around_flight"]),
             mid.replace(month=m1, day=d1, hour=23, minute=59, second=59))
    return lo, hi


def choose_scenes(cl, zone, bounds, flight_start, flight_end):
    """Up to max_scenes scenes for a site, closest to the flight first, at least min_days_apart apart.

    Scenes: same UTM zone as the grid, between the start and end of the S2 window and within days_around_flight of
    the flight midpoint, and with less than max_bad_fraction of the site flagged bad by the scene classification
    (SCL). Returns a list of dicts (scene, date, days_from_flight, bad_fraction)."""
    mid = flight_start + (flight_end - flight_start) / 2
    lo, hi = _window(mid)
    if lo > hi:
        return []
    bbox = transform_bounds(f"EPSG:{s2_epsg(zone)}", "EPSG:4326", *bounds)
    items = list(cl.search(collections=[C["collection"]], bbox=bbox,
                           datetime=f"{lo:%Y-%m-%dT%H:%M:%SZ}/{hi:%Y-%m-%dT%H:%M:%SZ}",
                           query={"eo:cloud_cover": {"lt": 50}}).items())
    epsg = f"EPSG:{s2_epsg(zone)}"
    items = [i for i in items if (i.properties.get("proj:code") or f"EPSG:{i.properties.get('proj:epsg')}") == epsg]
    items.sort(key=lambda i: (abs((i.datetime - mid).total_seconds()), i.id))
    shape, kept = shape_of(bounds), []
    for it in items:
        if len(kept) == C["max_scenes"]:
            break
        if any(abs((it.datetime - k[0].datetime).days) < C["min_days_apart"] for k in kept):
            continue
        scl = read_window(it.assets["scl"].href, bounds, shape)
        bad = float(np.isin(scl, C["bad_scl"]).mean())
        if bad < C["max_bad_fraction"]:
            kept.append((it, bad))
    return [dict(scene=it.id, date=it.datetime.date().isoformat(), days_from_flight=(it.datetime - mid).days,
                 bad_fraction=round(bad, 5)) for it, bad in kept]


def check_scale_and_offset(item, swir_dn):
    """Reflectance = DN x scale. Stops if a band has another scale, or if the scene still contains the -0.1 BOA
    offset: then every band has DN >= 1000, including B12 (SWIR 2.2 um), whose 1st percentile is otherwise far below
    (dark water, shade). B12 is used because haze and wildfire smoke raise blue and red, but barely B12: a smoky
    scene must not be mistaken for an offset problem. This collection delivers the offset already removed."""
    for _, key, _ in BANDS:
        s = item.assets[key].extra_fields["raster:bands"][0].get("scale", 1.0)
        if abs(s - C["scale"]) > 1e-12:
            raise ValueError(f"{item.id} {key}: scale {s}, expected {C['scale']}")
    v = swir_dn[swir_dn > 0]
    p1 = float(np.percentile(v, 1)) if v.size else 0.0
    if p1 >= C["offset_alarm_dn"]:
        raise ValueError(f"{item.id}: 1st percentile of B12 DN is {p1:.0f}; the BOA offset seems not removed")
    return p1


def write_stack(item, path, bounds, profile):
    """12 bands + SCL of one scene on the label grid -> 13-band uint16 GeoTIFF (DN as delivered, 0 = no data)."""
    shape = shape_of(bounds)
    keys = [k for _, k, _ in BANDS] + ["scl"]
    with ThreadPoolExecutor(6) as ex:
        arrs = list(ex.map(lambda k: read_window(item.assets[k].href, bounds, shape), keys))
    stack = np.stack(arrs).astype(np.uint16)
    p1 = check_scale_and_offset(item, stack[11])
    prof = dict(profile, count=stack.shape[0], dtype="uint16", nodata=0, predictor=2)
    makedirs(path.parent)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(stack)
        for i, (name, _, res) in enumerate(BANDS, 1):
            d.set_band_description(i, f"{name} ({res} m native), DN; reflectance = DN x {C['scale']}")
        d.set_band_description(SCL_BAND + 1, "SCL scene classification")
        d.update_tags(scene=item.id, datetime=item.datetime.isoformat(), b12_p1_dn=p1,
                      processing_baseline=str(item.properties.get("s2:processing_baseline")))
    tmp.rename(path)


def items_by_id(cl, ids):
    found = {}
    ids = list(ids)
    for i in range(0, len(ids), 100):
        for it in cl.search(ids=ids[i:i + 100], collections=[C["collection"]]).items():
            found[it.id] = it
    return found


def load_stack(paths):
    """Stacks of one site -> (dates, 13, H, W) uint16, in the order given."""
    xs = []
    for p in paths:
        with rasterio.open(p) as s:
            xs.append(s.read())
    return np.stack(xs)


def reflectance_and_mask(x):
    """(..., 13, H, W) uint16 -> reflectance (..., 12, H, W) float32 with NaN for no data, and bad (..., H, W) bool."""
    dn = x[..., :SCL_BAND, :, :].astype(np.float32)
    refl = np.where(dn == 0, np.nan, dn * C["scale"])
    bad = np.isin(x[..., SCL_BAND, :, :], C["bad_scl"])
    return refl, bad


def parse_time(s):
    return datetime.fromisoformat(str(s))
