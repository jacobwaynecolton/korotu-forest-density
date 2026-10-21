"""EDA over the labelled sites; writes numbers for the report and the image figures.

- Assigns the proposed spatial split per site (whole sites only) into sites.csv.
- Cover distribution overall, land only (WorldCover water removed), and per split,
  using placeholder class cut points 10 / 40 / 70 % (to be agreed with Korotu).
- Lidar cover vs ESA WorldCover 2021 classes (first look at a third-party product).
- Figures: pipeline (S2 | CHM 0.5 m | tree mask | 10 m cover) and per-site thumbnails.

Usage: python -I scripts/eda.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from rasterio.warp import reproject, Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, LABELS, CONTEXT = ROOT / "data/manifest", ROOT / "data/labels", ROOT / "data/context"
FIG = ROOT / "report/figures"
CUTS = [0.10, 0.40, 0.70]
CLASSES = ["Non-forest (<10%)", "Sparse (10–40%)", "Moderate (40–70%)", "Dense (≥70%)"]
WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
      60: "Bare", 80: "Water", 90: "Wetland", 100: "Moss & lichen"}
SPLITS = {  # whole sites, spread out; at least one sparse-targeted site in val and test
    "test": ["Z15_E0415_N5390_2020", "Z18_E0305_N5105_2022", "Z16_E0350_N5600_2023"],
    "val": ["Z15_E0690_N5475_2020", "Z17_E0400_N5110_2021", "Z17_E0355_N5560_2019"],
}
# sequential blue ramp (dataviz reference palette, steps 100..650)
COVER_CMAP = LinearSegmentedColormap.from_list(
    "cover", ["#f3f2ee", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])


def cls(c):
    return np.digitize(c, CUTS)


def load(site):
    with rasterio.open(LABELS / f"{site}_cover10m.tif") as s:
        c2, c5 = s.read(1), s.read(2)
    with rasterio.open(CONTEXT / f"{site}_worldcover.tif") as s:
        wc = s.read(1)
    return c2, c5, wc


def shares(c):
    c = c[np.isfinite(c)]
    return (np.bincount(cls(c), minlength=4) / len(c)).tolist(), int(len(c))


def assign_splits():
    sites = pd.read_csv(MANIFEST / "sites.csv")
    sites["selection"] = sites.get("selection", pd.Series(dtype=str)).fillna("spread")
    sites["split"] = "train"
    for k, v in SPLITS.items():
        sites.loc[sites.site_id.isin(v), "split"] = k
    sites.to_csv(MANIFEST / "sites.csv", index=False)
    return sites


def most_mixed_window(candidates, n=30, step=15):
    """Window whose 10 m cells spread most evenly over the four classes (max entropy)."""
    best = (-1, None, 0, 0)
    for site in candidates:
        c2, _, wc = load(site)
        for r in range(0, c2.shape[0] - n, step):
            for c in range(0, c2.shape[1] - n, step):
                w = c2[r:r + n, c:c + n]
                if not np.isfinite(w).all() or (wc[r:r + n, c:c + n] == 80).mean() > 0.05:
                    continue
                p = np.bincount(cls(w.ravel()), minlength=4) / w.size
                h = -(p[p > 0] * np.log(p[p > 0])).sum()
                if h > best[0]:
                    best = (h, site, r, c)
    return best[1:]


def pipeline_figure(site, r0, c0, n=30):
    """n x n 10 m cells window: S2 RGB | CHM | tree mask | cover."""
    with rasterio.open(LABELS / f"{site}_cover10m.tif") as s:
        crs, tf = s.crs, s.transform
        cov = s.read(1)[r0:r0 + n, c0:c0 + n]
    with rasterio.open(CONTEXT / f"{site}_s2_tci.tif") as s:
        rgb = s.read()[:, r0:r0 + n, c0:c0 + n].transpose(1, 2, 0)
    x0, y1 = tf.c + c0 * 10, tf.f - r0 * 10
    fine_tf = from_origin(x0, y1, 0.5, 0.5)
    chm = np.full((n * 20, n * 20), np.nan, np.float32)
    for t in sorted((ROOT / "data/raw/fri_chm" / site).glob("*.tif")):
        with rasterio.open(t) as s:
            a = s.read(1)
            a[(a == s.nodata) | (a < -2) | (a > 80)] = np.nan
            tmp = np.full_like(chm, np.nan)
            reproject(a, tmp, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                      dst_transform=fine_tf, dst_crs=crs, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        chm = np.where(np.isnan(chm), tmp, chm)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4), dpi=130)
    ext = [0, n * 10, 0, n * 10]
    ax[0].imshow(np.clip(rgb.astype(float) * 1.6, 0, 255).astype(np.uint8), extent=ext,
                 interpolation="nearest")
    ax[1].imshow(chm, cmap="Greens", vmin=0, vmax=15, extent=ext, interpolation="nearest")
    ax[2].imshow(chm > 2, cmap=LinearSegmentedColormap.from_list("m", ["#f3f2ee", "#1b6b3a"]),
                 extent=ext, interpolation="nearest")
    im = ax[3].imshow(cov, cmap=COVER_CMAP, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    titles = ["Sentinel-2 true colour (10 m)", "Lidar canopy height (0.5 m)",
              "Tree mask: height > 2 m (0.5 m)", "Label: canopy cover per 10 m cell"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9.5)
        a.set_xticks([0, 150, 300]); a.set_yticks([0, 150, 300])
        a.tick_params(labelsize=7, colors="#6b6a65")
        for sp in a.spines.values():
            sp.set_color("#c3c2b7")
    ax[0].set_ylabel("metres", fontsize=8, color="#6b6a65")
    cb = fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", facecolor="white")
    plt.close(fig)


def thumbs(site):
    c2, _, _ = load(site)
    with rasterio.open(CONTEXT / f"{site}_s2_tci.tif") as s:
        rgb = s.read().transpose(1, 2, 0)
    for name, img, kw in [("s2", np.clip(rgb.astype(float) * 1.6, 0, 255).astype(np.uint8), {}),
                          ("cover", c2, dict(cmap=COVER_CMAP, vmin=0, vmax=1))]:
        fig = plt.figure(figsize=(3, 3), dpi=110)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(img, interpolation="nearest", **kw)
        fig.savefig(FIG / f"{site}_{name}.jpg", pil_kwargs=dict(quality=82))
        plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sites = assign_splits()
    stats = pd.read_csv(MANIFEST / "label_stats.csv").set_index("site_id")
    scenes = pd.read_csv(MANIFEST / "s2_preview_scenes.csv").set_index("site_id")

    all_c, land_c, split_c, per_site = [], [], {"train": [], "val": [], "test": []}, []
    wc_rows = {}
    for s in sites.itertuples():
        c2, c5, wc = load(s.site_id)
        ok = np.isfinite(c2)
        land = ok & (wc != 80)
        all_c.append(c2[ok]); land_c.append(c2[land]); split_c[s.split].append(c2[land])
        sh, n = shares(c2[land])
        per_site.append(dict(site_id=s.site_id, split=s.split, selection=s.selection,
                             lon=round(s.lon, 3), lat=round(s.lat, 3), year=int(s.year),
                             flight=f"{s.flight_start[:10]} to {s.flight_end[:10]}",
                             s2_scene=scenes.loc[s.site_id, "scene"],
                             s2_gap_days=int(scenes.loc[s.site_id, "days_from_flight"]),
                             water=float((wc[ok] == 80).mean()), land_cells=n,
                             mean_cover=float(np.mean(c2[land])),
                             mean_cover_gt5=float(np.nanmean(c5[land])), shares=sh))
        for k in np.unique(wc[ok]):
            sel = ok & (wc == k)
            wc_rows.setdefault(int(k), []).append(c2[sel])
        thumbs(s.site_id)

    allc, landc = np.concatenate(all_c), np.concatenate(land_c)
    hist = np.histogram(landc, bins=20, range=(0, 1.0000001))[0]
    wc_tab = []
    for k, arrs in sorted(wc_rows.items()):
        v = np.concatenate(arrs)
        if len(v) < 2000 or k not in WC:
            continue
        wc_tab.append(dict(code=k, name=WC[k], cells=int(len(v)), mean_cover=float(v.mean()),
                           shares=(np.bincount(cls(v), minlength=4) / len(v)).tolist()))
    wct = np.concatenate([np.concatenate(wc_rows[10])]) if 10 in wc_rows else np.array([])
    non_tree = np.concatenate([np.concatenate(a) for k, a in wc_rows.items() if k not in (10, 80)])
    summary = dict(
        n_sites=len(sites), area_km2=round(float(stats.width.mul(stats.height).sum() / 1e4), 1),
        n_cells=int(len(allc)), n_land_cells=int(len(landc)),
        water_frac=float(1 - len(landc) / len(allc)),
        shares_all=shares(allc)[0], shares_land=shares(landc)[0],
        shares_split={k: dict(zip(["shares", "cells"], shares(np.concatenate(v))))
                      for k, v in split_c.items()},
        hist_land=hist.tolist(),
        gap_days_median=float(scenes.days_from_flight.abs().median()),
        gap_days_max=int(scenes.days_from_flight.abs().max()),
        classes=CLASSES, cuts=CUTS, per_site=per_site, worldcover=wc_tab,
        wc_tree_lidar_lt10=float((wct < 0.10).mean()),
        wc_nontree_lidar_ge40=float((non_tree >= 0.40).mean()),
    )
    (MANIFEST / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    site, r0, c0 = most_mixed_window(["Z17_E0600_N5480_2025", "Z17_E0355_N5560_2019"])
    pipeline_figure(site, r0, c0)
    summary["pipeline_window"] = dict(site_id=site, row=r0, col=c0, size_cells=30)
    (MANIFEST / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_site", "worldcover")},
                     indent=1))
    for w in wc_tab:
        print(w["name"], w["cells"], round(w["mean_cover"], 2), [round(x, 2) for x in w["shares"]])


if __name__ == "__main__":
    main()
