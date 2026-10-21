"""Build 10 m canopy-cover labels on the Sentinel-2 grid for every site.

Per site (PLAN.md §3):
  1. Each 0.5 m CHM tile -> code raster: 0 nodata, 1 valid & <= 2 m, 2 > 2 m, 3 > 5 m.
     Heights outside [-2, 80] m are treated as nodata (lidar spikes / pits).
  2. Codes are reprojected (nearest) onto a 0.5 m grid in WGS84 / UTM (EPSG:326zz),
     the CRS Sentinel-2 L2A uses, with PROJ's NAD83(CSRS)->WGS84 Helmert (~1 m).
  3. 20x20 blocks are averaged into 10 m cells. The 10 m grid is snapped to multiples of
     60 m, so it lines up with S2's 10/20/60 m pixel grids.

Output: data/labels/<site>_cover10m.tif, float32, nodata NaN:
  band 1 cover_gt2m   fraction of valid area with canopy > 2 m (main label)
  band 2 cover_gt5m   fraction with canopy > 5 m (FAO/Canada height threshold)
  band 3 valid_frac   fraction of the cell with valid lidar
Cover is NaN unless valid_frac == 1. Also writes data/manifest/sites.geojson (label
footprints, for the Sentinel-2 side) and data/manifest/label_stats.csv.

Usage: python -I scripts/make_labels.py [site_id ...]
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
CHM = ROOT / "data/raw/fri_chm"
LABELS = ROOT / "data/labels"
MANIFEST = ROOT / "data/manifest"
FINE, CELL, SNAP = 0.5, 10, 60
K = int(CELL / FINE)


def s2_epsg(zone: int) -> int:
    return 32600 + zone


def tile_codes(path: Path):
    with rasterio.open(path) as s:
        a = s.read(1)
        nd = s.nodata
        meta = dict(crs=s.crs, transform=s.transform, bounds=s.bounds)
    valid = np.isfinite(a) & (a >= -2) & (a <= 80)
    if nd is not None:
        valid &= a != nd
    code = np.zeros(a.shape, np.uint8)
    code[valid] = 1
    code[valid & (a > 2)] = 2
    code[valid & (a > 5)] = 3
    return code, meta


def site_grid(tifs, dst_crs):
    lefts, bottoms, rights, tops = [], [], [], []
    for t in tifs:
        with rasterio.open(t) as s:
            l, b, r, tp = transform_bounds(s.crs, dst_crs, *s.bounds)
        lefts.append(l); bottoms.append(b); rights.append(r); tops.append(tp)
    # snap inward so every cell is inside the lidar block
    x0 = np.ceil(min(lefts) / SNAP) * SNAP
    y0 = np.ceil(min(bottoms) / SNAP) * SNAP
    x1 = np.floor(max(rights) / SNAP) * SNAP
    y1 = np.floor(max(tops) / SNAP) * SNAP
    return x0, y0, x1, y1


def build_site(site_id: str, zone: int):
    tifs = sorted((CHM / site_id).glob("*_CHM.tif"))
    dst_crs = rasterio.crs.CRS.from_epsg(s2_epsg(zone))
    x0, y0, x1, y1 = site_grid(tifs, dst_crs)
    W, H = int((x1 - x0) / FINE), int((y1 - y0) / FINE)
    fine_tf = from_origin(x0, y1, FINE, FINE)
    fine = np.zeros((H, W), np.uint8)

    for t in tifs:
        code, m = tile_codes(t)
        tb = transform_bounds(m["crs"], dst_crs, *m["bounds"])
        win = from_bounds(*tb, transform=fine_tf).round_offsets().round_lengths()
        r0, c0 = max(win.row_off - 2, 0), max(win.col_off - 2, 0)
        r1, c1 = min(win.row_off + win.height + 2, H), min(win.col_off + win.width + 2, W)
        if r1 <= r0 or c1 <= c0:
            continue
        sub = np.zeros((r1 - r0, c1 - c0), np.uint8)
        reproject(code, sub, src_transform=m["transform"], src_crs=m["crs"], src_nodata=0,
                  dst_transform=rasterio.windows.transform(
                      rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), fine_tf),
                  dst_crs=dst_crs, dst_nodata=0, resampling=Resampling.nearest)
        view = fine[r0:r1, c0:c1]
        np.copyto(view, sub, where=(view == 0) & (sub > 0))  # first tile wins on overlaps

    shp = (H // K, K, W // K, K)
    n_valid = (fine > 0).reshape(shp).sum(axis=(1, 3)).astype(np.float32)
    n_gt2 = (fine >= 2).reshape(shp).sum(axis=(1, 3))
    n_gt5 = (fine == 3).reshape(shp).sum(axis=(1, 3))
    valid_frac = n_valid / K**2
    full = valid_frac == 1
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.where(full, n_gt2 / n_valid, np.nan).astype(np.float32)
        c5 = np.where(full, n_gt5 / n_valid, np.nan).astype(np.float32)

    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / f"{site_id}_cover10m.tif"
    prof = dict(driver="GTiff", width=W // K, height=H // K, count=3, dtype="float32",
                crs=dst_crs, transform=from_origin(x0, y1, CELL, CELL), nodata=np.nan,
                compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out, "w", **prof) as d:
        d.write(np.stack([c2, c5, valid_frac.astype(np.float32)]))
        for i, n in enumerate(["cover_gt2m", "cover_gt5m", "valid_frac"], 1):
            d.set_band_description(i, n)
        d.update_tags(source="Ontario FRI leaf-on SPL lidar CHM 0.5 m (OGL-Ontario)",
                      method="fraction of 0.5 m CHM pixels above threshold per 10 m cell; "
                             "grid snapped to 60 m in EPSG:326zz to match Sentinel-2")
    v = c2[full]
    return dict(site_id=site_id, epsg_s2=s2_epsg(zone), xmin=x0, ymin=y0, xmax=x1, ymax=y1,
                width=W // K, height=H // K, n_cells=int(full.sum()),
                frac_full=float(full.mean()), mean_cover_gt2m=float(v.mean()),
                mean_cover_gt5m=float(c5[full].mean()), label_file=str(out.relative_to(ROOT)))


def main(only=None):
    sites = pd.read_csv(MANIFEST / "sites.csv")
    if only:
        sites = sites[sites.site_id.isin(only)]
    stats = []
    for r in sites.itertuples():
        st = build_site(r.site_id, r.zone)
        print(f"{r.site_id}: {st['width']}x{st['height']} cells, "
              f"{st['frac_full']:.1%} full, mean cover {st['mean_cover_gt2m']:.2f}", flush=True)
        stats.append(st)
    stats = pd.DataFrame(stats)
    prev = MANIFEST / "label_stats.csv"
    if only and prev.exists():
        old = pd.read_csv(prev)
        stats = pd.concat([old[~old.site_id.isin(stats.site_id)], stats], ignore_index=True)
    stats.to_csv(prev, index=False)

    # footprint polygons (lon/lat) for the Sentinel-2 teammates
    all_sites = pd.read_csv(MANIFEST / "sites.csv").merge(stats, on="site_id")
    geoms = [box(*transform_bounds(f"EPSG:{int(e)}", "EPSG:4326", a, b, c, d))
             for e, a, b, c, d in all_sites[["epsg_s2", "xmin", "ymin", "xmax", "ymax"]].values]
    gpd.GeoDataFrame(all_sites.drop(columns=["tiles"]), geometry=geoms, crs="EPSG:4326"
                     ).to_file(MANIFEST / "sites.geojson", driver="GeoJSON")
    print(f"wrote {prev} and {MANIFEST / 'sites.geojson'}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
