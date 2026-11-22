"""Step 11 (optional): compare the final central map with existing canopy-cover maps on the held-out test set.

Products, put on each site's 10 m label grid (same method as the first version's scripts/benchmarks.py):
  meta_chm_v2  Meta/WRI Global Canopy Height v2: share of ~1 m pixels above 2 m in each 10 m cell (our definition);
  scanfi_v2    SCANFI v2 crown closure, 30 m, the 5-yearly layer nearest the lidar year;
  ntems_2022   NTEMS 2022 canopy cover, 30 m; non-forest (no data) counts as 0 on land.
Each product is also given "recalibrated": an isotonic map product -> lidar label learned on the training sites only
(40,000 cells per site), so that models are compared rather than cover definitions.
Scored on the 39 test sites, on cells where the label, the central map and the three products are all valid.
With --first-version DIR, the first version's three U-Nets (DIR/<name>/best.pt and config.json) are also scored, on
the test sites they never trained on (first-version sites.csv in data/manifest/), next to the final model on the
same sites.
Outputs: data/references/<site>_<product>.tif, results/references.csv
Usage: python -I pipeline/10_references.py [--first-version DIR]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, DATA, LABELS, MANIFESTS, MAPS, REPO, RESULTS, STACKS, makedirs, read_band, read_cover  # noqa: E402
from model import predict  # noqa: E402
from sentinel2 import load_stack, reflectance_and_mask  # noqa: E402

REF = DATA / "references"
META = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
SCANFI = "/vsicurl/https://ftp.maps.canada.ca/pub/nrcan_rncan/Forests_Foret/SCANFI/v2/SCANFI_att_closure_{year}_v2_20260119.tif"
META_TILES = REPO / "data/raw/meta_chm/tiles.geojson"
PRODUCTS = {"meta_chm_v2": "Meta/WRI CHM v2", "scanfi_v2": "SCANFI v2", "ntems_2022": "NTEMS 2022"}


def meta_chm(prof, tiles):
    crs, W, H, b = prof["crs"], prof["width"], prof["height"], rasterio.transform.array_bounds(prof["height"], prof["width"], prof["transform"])
    left, bottom, right, top = b[0], b[1], b[2], b[3]
    k = 10
    fine = np.full((H * k, W * k), 255, np.uint8)
    poly = box(*transform_bounds(crs, "EPSG:4326", left, bottom, right, top))
    for q in tiles[tiles.intersects(poly)].tile:
        with rasterio.open(f"/vsicurl/{META}/chm/{q}.tif") as s:
            w = from_bounds(*transform_bounds(crs, s.crs, left, bottom, right, top), transform=s.transform)
            w = w.round_offsets().round_lengths()
            a = s.read(1, window=w, boundless=True, fill_value=255)
            reproject(a, fine, src_transform=s.window_transform(w), src_crs=s.crs, src_nodata=255,
                      dst_transform=from_origin(left, top, 1.0, 1.0), dst_crs=crs, dst_nodata=255,
                      init_dest_nodata=False, resampling=Resampling.nearest)
    valid = fine != 255
    shp = (H, k, W, k)
    nv = valid.reshape(shp).sum(axis=(1, 3))
    tree = ((fine > 2) & valid).reshape(shp).sum(axis=(1, 3))
    return np.where(nv == k * k, tree / np.maximum(nv, 1), np.nan).astype(np.float32)


def warp(path, prof, nodata_as_zero):
    with rasterio.open(path) as src, WarpedVRT(src, crs=prof["crs"], transform=prof["transform"], width=prof["width"],
                                               height=prof["height"], resampling=Resampling.nearest) as v:
        a = v.read(1, masked=True).astype(np.float32).filled(np.nan)
    if nodata_as_zero:
        a = np.where(np.isnan(a), 0.0, a)
    a = a / 100.0
    a[a > 1.0] = np.nan
    return a


def products_for(sid, year, tiles):
    out = {p: REF / f"{sid}_{p}.tif" for p in PRODUCTS}
    if all(f.exists() for f in out.values()):
        return
    with rasterio.open(LABELS / f"{sid}.tif") as s:
        prof = dict(s.profile, count=1, dtype="float32", nodata=np.nan)
    arrs = {"meta_chm_v2": lambda: meta_chm(prof, tiles),
            "scanfi_v2": lambda: warp(SCANFI.format(year=min(range(1985, 2030, 5), key=lambda y: (abs(y - year), y))), prof, False),
            "ntems_2022": lambda: warp(CFG["paths"]["ntems"], prof, True)}
    for p, f in arrs.items():
        if not out[p].exists():
            a = f()
            tmp = out[p].with_name(out[p].name + ".part")
            with rasterio.open(tmp, "w", **prof) as d:
                d.write(a, 1)
            tmp.rename(out[p])


class FirstBandOnly(torch.nn.Module):
    """A one-output (L1) U-Net, wrapped so that model.predict returns its sigmoid as the central map."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        z = self.m(x)
        return torch.cat([z, z * 0, z * 0], 1)


def first_version(dirpath, sites):
    import segmentation_models_pytorch as smp
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for d in sorted(Path(dirpath).iterdir()):
        cfg = json.loads((d / "config.json").read_text())
        m = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=12, classes=1)
        m.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
        nets.append((FirstBandOnly(m).to(dev).eval(), np.array(cfg["mean"]), np.array(cfg["std"])))
    out = {}
    for sid in sites:
        refl, bad = reflectance_and_mask(load_stack(sorted(STACKS.glob(f"{sid}_k*.tif"))))
        preds = []
        for m, mean, std in nets:
            z = np.nan_to_num((refl - mean[:, None, None]) / std[:, None, None], nan=0.0).astype(np.float16)
            preds.append(predict(m, torch.from_numpy(z).to(dev), torch.from_numpy(bad).to(dev))[1])
        out[sid] = np.mean(preds, axis=0)
    print(f"first version: {len(nets)} models, {len(out)} sites", flush=True)
    return out


def metrics(y, p):
    e = p - y
    return dict(cells=int(len(y)), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-version", type=Path)
    args = ap.parse_args()
    sp = pd.read_csv(MANIFESTS / "splits.csv").merge(pd.read_csv(MANIFESTS / "sites.csv")[["site_id", "year"]], on="site_id")
    makedirs(REF)
    tiles = gpd.read_file(META_TILES)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda r: products_for(r.site_id, int(r.year), tiles), sp.itertuples()))
    print("products ready", flush=True)

    rng = np.random.default_rng(0)
    xs, ys = {p: [] for p in PRODUCTS}, []
    for sid in sp[sp.split == "train"].site_id:
        y = read_band(LABELS / f"{sid}.tif")
        a = {p: read_band(REF / f"{sid}_{p}.tif") for p in PRODUCTS}
        ok = np.isfinite(y) & np.logical_and.reduce([np.isfinite(v) for v in a.values()])
        i = rng.choice(ok.sum(), min(40000, ok.sum()), replace=False)
        ys.append(y[ok][i])
        for p in PRODUCTS:
            xs[p].append(a[p][ok][i])
    y_fit = np.concatenate(ys)
    iso = {p: IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.concatenate(xs[p]), y_fit) for p in PRODUCTS}

    test = sp[sp.split == "test"].site_id.tolist()
    seen = set()
    if args.first_version:
        old = pd.read_csv(REPO / "data/manifest/sites.csv")
        seen = set(old[old.split == "train"].site_id)
        fv = first_version(args.first_version, [s for s in test if s not in seen])
    cells = {k: [] for k in ["label", "final", "first", "unseen"] + [f"{p}|{v}" for p in PRODUCTS for v in ("raw", "cal")]}
    for sid in test:
        y = read_band(LABELS / f"{sid}.tif")
        q50 = read_cover(MAPS / f"{sid}.tif")[1]
        a = {p: read_band(REF / f"{sid}_{p}.tif") for p in PRODUCTS}
        ok = np.isfinite(y) & np.isfinite(q50) & np.logical_and.reduce([np.isfinite(v) for v in a.values()])
        cells["label"].append(y[ok]); cells["final"].append(q50[ok])
        unseen = args.first_version is not None and sid not in seen
        cells["unseen"].append(np.full(ok.sum(), unseen))
        cells["first"].append(fv[sid][ok] if unseen else np.full(ok.sum(), np.nan))
        for p in PRODUCTS:
            cells[f"{p}|raw"].append(a[p][ok]); cells[f"{p}|cal"].append(iso[p].predict(a[p][ok]))
    c = {k: np.concatenate(v) for k, v in cells.items()}
    y = c["label"]
    rows = [dict(sites=len(test), source="Final model, central map (q50)", version="", **metrics(y, c["final"]))]
    for p, name in PRODUCTS.items():
        rows.append(dict(sites=len(test), source=name, version="as published", **metrics(y, c[f"{p}|raw"])))
        rows.append(dict(sites=len(test), source=name, version="recalibrated on the training sites", **metrics(y, c[f"{p}|cal"])))
    if args.first_version:
        u = c["unseen"].astype(bool)
        n = len([s for s in test if s not in seen])
        rows.append(dict(sites=n, source="First-version U-Net (3 seeds)", version="test sites it never trained on", **metrics(y[u], c["first"][u])))
        rows.append(dict(sites=n, source="Final model, central map (q50)", version="same sites", **metrics(y[u], c["final"][u])))
    t = pd.DataFrame(rows)
    makedirs(RESULTS)
    t.to_csv(RESULTS / "references.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
