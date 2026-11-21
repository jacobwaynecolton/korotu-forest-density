"""Shared configuration, paths and small helpers of the final pipeline."""
import json
import math
import os
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get("PIPELINE_CONFIG", HERE / "config.toml"))
CFG = tomllib.loads(CONFIG_FILE.read_text())


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


DATA = Path(os.environ.get("PIPELINE_DATA", CFG["paths"]["data"]))
MANIFESTS = _path(os.environ.get("PIPELINE_MANIFESTS", CFG["paths"]["manifests"]))
RESULTS = _path(os.environ.get("PIPELINE_RESULTS", CFG["paths"]["results"]))
LABELS, STACKS, CHM, PREDICTIONS, MODELS, MAPS, STATUS = (
    DATA / d for d in ("labels", "stacks", "chm", "predictions", "models", "maps", "status"))

# NAD83(CSRS) / UTM zone N, the CRS of the FRI lidar products (codes are not sequential)
CSRS_UTM_EPSG = {15: 3159, 16: 3160, 17: 2958, 18: 2959}
NODATA_U16 = 65535
SCALE_U16 = 10000          # predictions stored as cover x 10 000 in uint16


def s2_epsg(zone):
    """Sentinel-2 L2A grids use WGS84 / UTM (EPSG:326zz)."""
    return 32600 + int(zone)


def site_id(zone, be, bn, year):
    return f"Z{int(zone)}_E{int(be):04d}_N{int(bn):04d}_{int(year)}"


def parse_site_id(sid):
    z, e, n, y = sid.split("_")
    return int(z[1:]), int(e[1:]), int(n[1:]), int(y)


def block_grid(zone, be, bn):
    """Label grid of a 5 x 5 km block, from the block's nominal extent (no data needed).

    The block covers [be, be + 5] x [bn, bn + 5] km in NAD83(CSRS) / UTM. Its bounds are moved to the Sentinel-2
    CRS and snapped inward to multiples of snap_m, so every cell is inside the lidar block and every S2 10/20/60 m
    pixel lines up with the grid. Returns (left, bottom, right, top) in metres, EPSG:326zz.
    """
    k, s = CFG["sites"]["block_km"] * 1000, CFG["labels"]["snap_m"]
    l, b, r, t = transform_bounds(f"EPSG:{CSRS_UTM_EPSG[zone]}", f"EPSG:{s2_epsg(zone)}",
                                  be * 1000, bn * 1000, be * 1000 + k, bn * 1000 + k)
    return (math.ceil(l / s) * s, math.ceil(b / s) * s, math.floor(r / s) * s, math.floor(t / s) * s)


def grid_profile(zone, bounds, count=1, dtype="float32", nodata=np.nan):
    """GeoTIFF profile of a label-grid raster."""
    cell = CFG["labels"]["cell_m"]
    l, b, r, t = bounds
    return dict(driver="GTiff", width=int(round((r - l) / cell)), height=int(round((t - b) / cell)), count=count,
                dtype=dtype, crs=f"EPSG:{s2_epsg(zone)}", transform=rasterio.transform.from_origin(l, t, cell, cell),
                nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256)


def km_between(lon1, lat1, lon2, lat2):
    """Great-circle distance in km (vectorised over the second point)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def sites():
    """The selected sites (step 2), with zone, year, block origin, flight dates and stratum."""
    return pd.read_csv(MANIFESTS / "sites.csv")


def usable_sites():
    """Sites with a label and at least one image (steps 3 and 4)."""
    s = sites()
    st = pd.read_csv(MANIFESTS / "site_status.csv")
    return s.merge(st, on="site_id")[lambda d: d.usable].reset_index(drop=True)


def read_band(path, band=1):
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32)


def write_cover(path, arr, profile, descriptions=None):
    """(bands, H, W) cover fractions -> uint16 GeoTIFF (x 10 000, 65535 = no data). Atomic via a .part file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.where(np.isfinite(arr), np.round(np.clip(arr, 0, 1) * SCALE_U16), NODATA_U16).astype(np.uint16)
    prof = dict(profile, count=a.shape[0], dtype="uint16", nodata=NODATA_U16, compress="deflate", predictor=2)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(a)
        for i, text in enumerate(descriptions or [], start=1):
            d.set_band_description(i, text)
    tmp.rename(path)


def read_cover(path):
    """uint16 cover GeoTIFF -> (bands, H, W) float32, NaN where no data."""
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)
    return np.where(a == NODATA_U16, np.nan, a / SCALE_U16)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def task_sites(all_sites, task, n_tasks):
    """Sites handled by one task of a Slurm array: every n_tasks-th site, starting at the task index."""
    return [s for i, s in enumerate(all_sites) if n_tasks <= 1 or i % n_tasks == task]


def array_task():
    """(task index, number of tasks) from Slurm, or (0, 1) outside an array."""
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
