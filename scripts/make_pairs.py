"""Build Sentinel-2 L2A input stacks aligned to the lidar labels, plus a pixel table.

For each site, uses the scene in data/manifest/s2_preview_scenes.csv (closest cloud-free
scene to the lidar flight) and reads the 12 L2A bands + SCL for the label window.
- Reflectance = DN * scale + offset, with scale from each scene's STAC raster:bands.
  Baselines >= 04.00 carry a -0.1 BOA offset, but the Element84 COGs already have it
  removed when `earthsearch:boa_offset_applied` is true (checked: applying it again put
  dense-forest red reflectance at -0.08). So the offset is used only when not yet applied,
  which puts every baseline on one reflectance scale.
- 20 m and 60 m bands are upsampled to 10 m with nearest neighbour (each value is
  repeated, no new values are invented). The label grid is snapped to 60 m, so every
  band window starts on a whole pixel.

Outputs
  data/pairs/<site>_s2.tif     float32, 12 bands (B01..B12 order below), nodata NaN
  data/pairs/<site>_scl.tif    uint8 scene classification (cloud mask source)
  data/pairs/pixels.parquet    one row per 10 m cell: site, split, row, col, x, y,
                               12 bands, scl, cover_gt2m, cover_gt5m, worldcover, usable
  data/manifest/pairs_manifest.csv

`usable` = label present, all bands present, SCL not cloud/shadow/cirrus/snow/saturated.

Usage: python -I scripts/make_pairs.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, LABELS, CONTEXT = ROOT / "data/manifest", ROOT / "data/labels", ROOT / "data/context"
OUT = ROOT / "data/pairs"
STAC = "https://earth-search.aws.element84.com/v1"
# (band name, earth-search asset key, native resolution m)
BANDS = [("B01", "coastal", 60), ("B02", "blue", 10), ("B03", "green", 10), ("B04", "red", 10),
         ("B05", "rededge1", 20), ("B06", "rededge2", 20), ("B07", "rededge3", 20),
         ("B08", "nir", 10), ("B8A", "nir08", 20), ("B09", "nir09", 60),
         ("B11", "swir16", 20), ("B12", "swir22", 20)]
BAD_SCL = {0, 1, 3, 8, 9, 10, 11}  # nodata, saturated, shadow, cloud med/high, cirrus, snow


def read_window(href, bounds, shape, resampling):
    with rasterio.open(href) as s:
        w = from_bounds(*bounds, transform=s.transform)
        assert abs(w.col_off - round(w.col_off)) < 1e-6 and abs(w.row_off - round(w.row_off)) < 1e-6, \
            f"window not pixel-aligned: {w}"
        w = w.round_offsets().round_lengths()
        return s.read(1, window=w, out_shape=shape, resampling=resampling, boundless=True, fill_value=0)


def boa_offset_applied(item, bounds, shape):
    """Whether the -0.1 BOA offset is already baked into this scene's COGs.

    The `earthsearch:boa_offset_applied` flag is not reliable: some 04.00-baseline scenes
    say False although the pixels already have the offset removed (found by
    verify/check_baselines.py: applying it again gave reflectance near -0.09 everywhere).
    So the pixels decide too: with the offset still in, real surfaces have blue DN >= 1000.
    """
    if item.properties.get("earthsearch:boa_offset_applied", False):
        return True
    rb = item.assets["blue"].extra_fields["raster:bands"][0]
    if not rb.get("offset", 0.0):
        return True
    dn = read_window(item.assets["blue"].href, bounds, shape, Resampling.nearest)
    dn = dn[dn != rb.get("nodata", 0)]
    return bool(dn.size) and float(np.percentile(dn, 1)) < 1000


def build_site(site, item, split):
    with rasterio.open(LABELS / f"{site}_cover10m.tif") as s:
        prof, bounds = s.profile, s.bounds
        c2, c5 = s.read(1), s.read(2)
    H, W = c2.shape
    stack = np.full((len(BANDS), H, W), np.nan, np.float32)
    offsets = set()
    applied = boa_offset_applied(item, bounds, (H, W))
    for i, (name, key, _) in enumerate(BANDS):
        a = item.assets[key]
        rb = a.extra_fields["raster:bands"][0]
        dn = read_window(a.href, bounds, (H, W), Resampling.nearest)
        scale = rb.get("scale", 1.0)
        offset = 0.0 if applied else rb.get("offset", 0.0)
        offsets.add(offset)
        stack[i] = np.where(dn == rb.get("nodata", 0), np.nan, dn * scale + offset)
    scl = read_window(item.assets["scl"].href, bounds, (H, W), Resampling.nearest).astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(prof, count=len(BANDS), dtype="float32", nodata=np.nan)
    with rasterio.open(OUT / f"{site}_s2.tif", "w", **p) as d:
        d.write(stack)
        for i, (name, _, res) in enumerate(BANDS, 1):
            d.set_band_description(i, f"{name} ({res} m native)")
        d.update_tags(scene=item.id, datetime=item.datetime.isoformat(),
                      processing_baseline=str(item.properties.get("s2:processing_baseline")),
                      units="surface reflectance (DN*scale+offset)", upsampling="nearest")
    with rasterio.open(OUT / f"{site}_scl.tif", "w", **dict(prof, count=1, dtype="uint8", nodata=0)) as d:
        d.write(scl, 1)

    with rasterio.open(CONTEXT / f"{site}_worldcover.tif") as s:
        wc = s.read(1)
    rr, cc = np.mgrid[0:H, 0:W]
    tf = prof["transform"]
    usable = np.isfinite(c2) & np.isfinite(stack).all(axis=0) & ~np.isin(scl, list(BAD_SCL))
    df = pd.DataFrame({
        "site_id": site, "split": split, "row": rr.ravel().astype(np.int16),
        "col": cc.ravel().astype(np.int16),
        "x": (tf.c + (cc.ravel() + 0.5) * 10).astype(np.float64),
        "y": (tf.f - (rr.ravel() + 0.5) * 10).astype(np.float64),
        **{name: stack[i].ravel() for i, (name, _, _) in enumerate(BANDS)},
        "scl": scl.ravel(), "cover_gt2m": c2.ravel(), "cover_gt5m": c5.ravel(),
        "worldcover": wc.ravel(), "usable": usable.ravel()})
    meta = dict(site_id=site, split=split, scene=item.id, scene_date=item.datetime.date().isoformat(),
                processing_baseline=item.properties.get("s2:processing_baseline"),
                boa_offset_already_applied=applied,
                offsets=";".join(map(str, sorted(offsets))), epsg=prof["crs"].to_epsg(),
                height=H, width=W, usable_cells=int(usable.sum()),
                usable_frac=float(usable.sum() / np.isfinite(c2).sum()))
    return df, meta


def main():
    sites = pd.read_csv(MANIFEST / "sites.csv").set_index("site_id")
    scenes = pd.read_csv(MANIFEST / "s2_preview_scenes.csv")
    client = Client.open(STAC)
    frames, metas = [], []
    for r in scenes.itertuples():
        item = next(client.search(ids=[r.scene], collections=["sentinel-2-l2a"]).items())
        df, meta = build_site(r.site_id, item, sites.loc[r.site_id, "split"])
        frames.append(df); metas.append(meta)
        print(f"{r.site_id}: {item.id} baseline {meta['processing_baseline']} offset {meta['offsets']} "
              f"usable {meta['usable_cells']:,} ({meta['usable_frac']:.1%})", flush=True)
    pix = pd.concat(frames, ignore_index=True)
    for c in ("site_id", "split"):
        pix[c] = pix[c].astype("category")
    pix.to_parquet(OUT / "pixels.parquet", index=False, compression="zstd")
    pd.DataFrame(metas).to_csv(MANIFEST / "pairs_manifest.csv", index=False)
    print(f"pixels.parquet: {len(pix):,} rows, {int(pix.usable.sum()):,} usable")


if __name__ == "__main__":
    main()
