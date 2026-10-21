"""Fairness checks and figures for the model vs benchmark comparison (val + test sites).

1. Per-site MAE for every source (does the win hold site by site?).
2. Meta CHM v2 restricted to cells whose imagery is within 2 years of the lidar flight
   (removes Meta's disadvantage from old imagery).
3. MAE by true-cover bin and 4-class confusion matrix for the model.
4. Figure: per held-out site, S2 | lidar label | model | Meta | SCANFI [| NTEMS].

Writes data/manifest/final_results.json and report/figures/compare_<site>.png.

Usage: python -I scripts/final_analysis.py <model_name> [extra sources...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CUTS = [0.10, 0.40, 0.70]
COVER = LinearSegmentedColormap.from_list(
    "cover", ["#f3f2ee", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])


def rd(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float32)


def main(model, extra):
    sources = {"model": ROOT / "data/predictions" / model,
               "meta_chm_v2": None, "scanfi_v2": None, **{e: None for e in extra}}
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    hold = sites[sites.split.isin(["val", "test"])]
    per_site, recent, bins, conf = [], {"model": [], "meta": [], "y": []}, [], np.zeros((4, 4), int)
    for s in hold.itertuples():
        y = rd(ROOT / "data/labels" / f"{s.site_id}_cover10m.tif")
        P = {"model": rd(ROOT / "data/predictions" / model / f"{s.site_id}.tif")}
        for k in list(sources)[1:]:
            P[k] = rd(ROOT / "data/benchmarks" / f"{s.site_id}_{k}.tif")
        ok = np.isfinite(y) & np.all([np.isfinite(v) for v in P.values()], axis=0)
        row = dict(site_id=s.site_id, split=s.split, cells=int(ok.sum()))
        for k, v in P.items():
            row[f"mae_{k}"] = float(np.abs(np.clip(v, 0, 1) - y)[ok].mean())
        per_site.append(row)
        my = rd(ROOT / "data/benchmarks" / f"{s.site_id}_meta_year.tif")
        rok = ok & (my > 0) & (np.abs(my - s.year) <= 2)
        recent["model"].append(P["model"][rok]); recent["meta"].append(P["meta_chm_v2"][rok]); recent["y"].append(y[rok])
        row["meta_imagery_years"] = sorted({int(v) for v in np.unique(my[ok]) if v > 0})
        row["meta_recent_frac"] = float(rok.sum() / ok.sum())
        if s.split == "test":
            yc, pc = np.digitize(y[ok], CUTS), np.digitize(P["model"][ok], CUTS)
            np.add.at(conf, (yc, pc), 1)
            for lo in np.arange(0, 1, 0.1):
                m = ok & (y >= lo) & (y < lo + 0.1 + (1e-6 if lo >= 0.9 else 0))
                bins.append(dict(site=s.site_id, lo=round(float(lo), 1), n=int(m.sum()),
                                 **{f"abs_{k}": float(np.abs(np.clip(v, 0, 1) - y)[m].sum()) for k, v in P.items()}))
        figure(s, y, P)
    y = np.concatenate(recent["y"])
    rec = dict(cells=int(len(y)), share=float(len(y) / sum(r["cells"] for r in per_site)),
               mae_model=float(np.abs(np.concatenate(recent["model"]) - y).mean()) if len(y) else None,
               mae_meta=float(np.abs(np.concatenate(recent["meta"]) - y).mean()) if len(y) else None)
    b = pd.DataFrame(bins).groupby("lo").sum(numeric_only=True)
    by_bin = [dict(lo=float(lo), n=int(r.n), **{k[4:]: float(r[k] / max(r.n, 1)) for k in b.columns if k.startswith("abs_")})
              for lo, r in b.iterrows()]
    out = dict(model=model, per_site=per_site, meta_recent_imagery=rec,
               test_mae_by_cover_bin=by_bin, test_confusion=conf.tolist(), class_cuts=CUTS)
    (ROOT / "data/manifest/final_results.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(per_site).drop(columns=["meta_imagery_years"]).round(3).to_string(index=False))
    print("\nMeta imagery within 2 years of lidar:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
    print("test confusion (rows = lidar class, cols = model):\n", conf)


def figure(s, y, P):
    with rasterio.open(ROOT / "data/context" / f"{s.site_id}_s2_tci.tif") as src:
        rgb = np.clip(src.read().transpose(1, 2, 0).astype(float) * 1.6, 0, 255).astype(np.uint8)
    panels = [("Sentinel-2 input", rgb), ("Lidar truth", y), ("Our model", P["model"]),
              ("Meta/WRI CHM v2", P["meta_chm_v2"]), ("SCANFI v2", P["scanfi_v2"])] + ([("NTEMS 2022", P["ntems_2022"])] if "ntems_2022" in P else [])
    fig, ax = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.3), dpi=110)
    for a, (t, img) in zip(ax, panels):
        if img.ndim == 3:
            a.imshow(img, interpolation="nearest")
        else:
            a.imshow(img, cmap=COVER, vmin=0, vmax=1, interpolation="nearest")
        if t not in ("Sentinel-2 input", "Lidar truth"):
            ok = np.isfinite(y) & np.isfinite(img)
            t += f"  MAE {np.abs(np.clip(img, 0, 1) - y)[ok].mean():.3f}"
        a.set_title(t, fontsize=10); a.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "report/figures" / f"compare_{s.site_id}.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
