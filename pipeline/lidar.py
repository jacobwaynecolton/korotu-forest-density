"""Ontario FRI leaf-on lidar: tile index, eligible blocks, flight dates, CHM tiles and 10 m cover labels."""
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from common import CFG, CHM, CSRS_UTM_EPSG, s2_epsg

S, L = CFG["sites"], CFG["labels"]


# ---------- tile index and eligible blocks ----------
def read_tile_index(path):
    """FRI tile index -> one row per 1 km tile with zone, lower-left km, year and download links."""
    df = pyogrio.read_dataframe(path, layer="FRI_Tile_Index", read_geometry=False)
    t = df.Tilename  # e.g. "1kmZ173420548502021L": zone 17, easting 342.0 km (100 m units), northing 5485 km, 2021
    df["zone"] = t.str[4:6].astype(int)
    df["e_km"] = t.str[6:10].astype(int) // 10
    df["n_km"] = t.str[10:14].astype(int)
    df["year"] = t.str[15:19].astype(int)
    return df


def eligible_blocks(tiles):
    """5 x 5 km blocks of one zone and one year, all 25 tiles with a CHM, year >= min_year. Centre in lon/lat."""
    k = S["block_km"]
    df = tiles[tiles.Download_CHM.notna() & (tiles.year >= S["min_year"])].copy()
    df["be"], df["bn"] = df.e_km // k * k, df.n_km // k * k
    blocks = df.groupby(["zone", "year", "be", "bn"]).size().rename("n").reset_index()
    blocks = blocks[blocks.n == k * k].drop(columns="n").reset_index(drop=True)
    blocks["lon"], blocks["lat"] = np.nan, np.nan
    for z, idx in blocks.groupby("zone").groups.items():
        tr = Transformer.from_crs(f"EPSG:{CSRS_UTM_EPSG[z]}", "EPSG:4326", always_xy=True)
        b = blocks.loc[idx]
        lon, lat = tr.transform(((b.be + k / 2) * 1000).values, ((b.bn + k / 2) * 1000).values)
        blocks.loc[idx, "lon"], blocks.loc[idx, "lat"] = lon, lat
    return blocks


def block_tiles(tiles, zone, year, be, bn):
    k = S["block_km"]
    return tiles[(tiles.zone == zone) & (tiles.year == year)
                 & (tiles.e_km // k * k == be) & (tiles.n_km // k * k == bn)]


# ---------- flight dates from COPC headers ----------
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP_SECONDS = 18  # GPS - UTC since 2017


def copc_time_range(url, timeout=60):
    """First and last point time of a COPC .laz, read from its first kilobyte (HTTP range request)."""
    with requests.get(url, headers={"Range": "bytes=0-1023"}, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        b = r.raw.read(1024)  # a server that ignores Range must not send the whole file
    if b[:4] != b"LASF":
        raise ValueError(f"not a LAS file: {url}")
    encoding = struct.unpack_from("<H", b, 6)[0]
    header = struct.unpack_from("<H", b, 94)[0]
    if b[header + 2:header + 18].rstrip(b"\0") != b"copc":
        raise ValueError(f"first VLR is not copc info: {url}")
    gmin, gmax = struct.unpack_from("<dd", b, header + 54 + 56)
    # FRI files leave global-encoding bit 0 unset but store adjusted standard GPS time (week time never > 604800)
    if not (encoding & 1) and gmax <= 604800:
        raise ValueError("GPS week time: the date needs the GPS week number")
    to_utc = lambda g: GPS_EPOCH + timedelta(seconds=g + 1e9 - LEAP_SECONDS)  # noqa: E731
    return to_utc(gmin), to_utc(gmax)


def _in_window(d, window):
    (m0, d0), (m1, d1) = (tuple(int(x) for x in w.split("-")) for w in window)
    return date(d.year, m0, d0) <= d.date() <= date(d.year, m1, d1)


def flight_dates(block_tile_rows):
    """(first, last) flight time of a block, or a rejection reason (str)."""
    with ThreadPoolExecutor(8) as ex:
        ranges = list(ex.map(copc_time_range, block_tile_rows.Download_LAZ))
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    if not all(_in_window(a, S["leaf_on"]) and _in_window(b, S["leaf_on"]) for a, b in ranges):
        return "flights outside the leaf-on window"
    if (hi - lo).days > S["max_flight_span_days"]:
        return f"flights spread over {(hi - lo).days} days"
    return lo, hi


# ---------- CHM tiles ----------
def _complete(path):
    try:
        with rasterio.open(path) as s:
            s.read(1, window=Window(0, s.height - 1, s.width, 1))  # the last row decodes
            return s.width >= 2000 and s.height >= 2000
    except rasterio.errors.RasterioIOError:
        return False


def cache_index():
    """{tile name: path} of CHM tiles already on disk in the read-only cache folders."""
    out = {}
    for d in CFG["paths"]["chm_cache"]:
        for p in sorted(Path(d).glob("*/*_CHM.tif")):
            out.setdefault(p.name[:-len("_CHM.tif")], p)
    return out


def fetch_tile(name, url, site, cache):
    """Path to the CHM tile: from the cache if there, else downloaded into DATA/chm/<site>/ (resumable).

    Returns (path, downloaded) so the caller deletes only what it downloaded."""
    if name in cache and _complete(cache[name]):
        return cache[name], False
    path = CHM / site / f"{name}_CHM.tif"
    if _complete(path):
        return path, True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r.raw, f, 1 << 20)
            tmp.rename(path)
            if _complete(path):
                return path, True
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
    raise IOError(f"CHM tile {name} does not decode after 3 attempts")


# ---------- labels ----------
def tile_codes(path):
    """CHM tile -> uint8 codes (0 missing, 1 valid <= 2 m, 2 > 2 m, 3 > 5 m) and the number of missing pixels.

    Missing: not finite, the declared no-data value, any known no-data convention (-9999, -32767, FLT_MIN) even
    when undeclared, or a height outside height_range."""
    with rasterio.open(path) as s:
        a = s.read(1)
        nd = s.nodata
        meta = dict(crs=s.crs, transform=s.transform, bounds=s.bounds)
    lo, hi = L["height_range"]
    bad = ~np.isfinite(a) | (a < lo) | (a > hi)
    for v in ([nd] if nd is not None else []) + L["extra_nodata"]:
        bad |= a == np.float32(v)
    code = np.zeros(a.shape, np.uint8)
    code[~bad] = 1
    code[~bad & (a > L["canopy_m"])] = 2
    code[~bad & (a > L["canopy_alt_m"])] = 3
    return code, int(bad.sum()), meta


def build_label(zone, bounds, tile_paths):
    """CHM tiles -> (3, H, W) float32 label on the grid: cover > 2 m, cover > 5 m, valid fraction; and stats.

    Codes are reprojected (nearest) onto a 0.5 m grid in the Sentinel-2 CRS (PROJ's NAD83(CSRS) -> WGS84 Helmert,
    about 1 m), then 20 x 20 blocks are averaged. Cover is NaN unless every 0.5 m pixel of the cell is valid."""
    fine, cell = L["fine_m"], L["cell_m"]
    k = int(round(cell / fine))
    l, b, r, t = bounds
    W, H = int(round((r - l) / fine)), int(round((t - b) / fine))
    dst_crs = rasterio.crs.CRS.from_epsg(s2_epsg(zone))
    fine_tf = from_origin(l, t, fine, fine)
    grid = np.zeros((H, W), np.uint8)
    missing = {}
    for p in tile_paths:
        code, n_missing, m = tile_codes(p)
        missing[p.name] = n_missing
        tb = transform_bounds(m["crs"], dst_crs, *m["bounds"])
        win = from_bounds(*tb, transform=fine_tf).round_offsets().round_lengths()
        r0, c0 = max(win.row_off - 2, 0), max(win.col_off - 2, 0)
        r1, c1 = min(win.row_off + win.height + 2, H), min(win.col_off + win.width + 2, W)
        if r1 <= r0 or c1 <= c0:
            continue
        sub = np.zeros((r1 - r0, c1 - c0), np.uint8)
        reproject(code, sub, src_transform=m["transform"], src_crs=m["crs"], src_nodata=0,
                  dst_transform=rasterio.windows.transform(Window(c0, r0, c1 - c0, r1 - r0), fine_tf),
                  dst_crs=dst_crs, dst_nodata=0, resampling=Resampling.nearest)
        view = grid[r0:r1, c0:c1]
        np.copyto(view, sub, where=(view == 0) & (sub > 0))  # first tile wins on overlaps
    shp = (H // k, k, W // k, k)
    n_valid = (grid > 0).reshape(shp).sum(axis=(1, 3)).astype(np.float32)
    full = n_valid == k * k
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.where(full, (grid >= 2).reshape(shp).sum(axis=(1, 3)) / n_valid, np.nan).astype(np.float32)
        c5 = np.where(full, (grid == 3).reshape(shp).sum(axis=(1, 3)) / n_valid, np.nan).astype(np.float32)
    v = c2[full]
    stats = dict(cells=int(full.sum()), full_share=float(full.mean()),
                 mean_cover=float(v.mean()) if v.size else np.nan,
                 partial_share=float(((v >= S["partial_low"]) & (v < S["partial_high"])).mean()) if v.size else np.nan,
                 missing_chm_pixels=int(sum(missing.values())),
                 tiles_with_missing=int(sum(n > 0 for n in missing.values())))
    return np.stack([c2, c5, n_valid / (k * k)]).astype(np.float32), stats

