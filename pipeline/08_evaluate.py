"""Step 8: seed ensembles, conformal calibration, and evaluation on cross-validation and on the held-out test set.

1. Out-of-fold ensembles: each training-pool site is predicted by the 5 seeds of its fold; the seeds are combined
   as a mixture (model.mixture).
2. Calibration per band of the central map. In cross-validation, each fold's constants come from the other folds,
   so the reported coverage is honest. The final constants come from all folds and are saved for the maps and for
   predictions on new areas. A single pair of constants (the first calibration) is reported for comparison.
3. Test set: the 5 final models, combined and calibrated with the final constants. Never used for any choice.
Reported: central-map MAE, RMSE, bias, R2; quantile score; share of labels below q10, above q90 and inside;
interval width; overall, by fold, by site, by lidar partial-cover stratum, and coverage by band of the central map.
Outputs: manifests/calibration.json, results/*.csv, results/summary.json, results/coverage_by_band.png
Usage: python -I pipeline/08_evaluate.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, LABELS, MANIFESTS, PREDICTIONS, RESULTS, makedirs, read_band, read_cover, write_json  # noqa: E402
from model import band_of, calibrate, constants, metrics, mixture  # noqa: E402

T, K = CFG["train"], CFG["calibration"]


def cells(site_ids, run_of):
    """Pooled cells: label, mixture q10/q50/q90, one-seed predictions, and site index."""
    out = dict(y=[], q=[], single=[], site=[])
    kept = []
    for sid in site_ids:
        paths = [p for p in (run_of(sid, s) / f"{sid}.tif" for s in T["seeds"]) if p.exists()]
        if len(paths) < len(T["seeds"]):
            print(f"  {sid}: {len(paths)}/{len(T['seeds'])} seeds")
        if not paths:
            continue
        preds = [read_cover(p) for p in paths]
        y = read_band(LABELS / f"{sid}.tif")
        q = mixture(preds)
        ok = np.isfinite(y) & np.isfinite(q).all(0)
        out["y"].append(y[ok]); out["q"].append(q[:, ok]); out["single"].append(preds[0][:, ok])
        out["site"].append(np.full(ok.sum(), len(kept))); kept.append(sid)
    return dict(y=np.concatenate(out["y"]), q=np.concatenate(out["q"], 1), single=np.concatenate(out["single"], 1),
                site=np.concatenate(out["site"])), kept


def coverage_by_band(name, y, q50, lo, hi):
    b = band_of(q50, K["bands"])
    n = np.bincount(b, minlength=K["bands"])
    below = np.bincount(b, weights=y < lo, minlength=K["bands"]) / np.maximum(n, 1)
    above = np.bincount(b, weights=y > hi, minlength=K["bands"]) / np.maximum(n, 1)
    return [dict(set=name, band=i / K["bands"], cells=int(n[i]), below_q10=below[i], above_q90=above[i])
            for i in range(K["bands"]) if n[i]]


def grouped(name, d, lo, hi, sites, sp):
    """Metrics per site and per lidar stratum."""
    strat = sp.set_index("site_id").lidar_stratum
    by_site, by_stratum = [], []
    for i, sid in enumerate(sites):
        m = d["site"] == i
        by_site.append(dict(set=name, site_id=sid, stratum=strat[sid], **metrics(d["y"][m], lo[m], d["q"][1][m], hi[m])))
    st = np.array([strat[s] for s in sites])[d["site"]]
    for s in CFG["sites"]["strata"]:
        m = st == s
        if m.any():
            by_stratum.append(dict(set=name, stratum=s, sites=int(sum(strat[x] == s for x in sites)),
                                   **metrics(d["y"][m], lo[m], d["q"][1][m], hi[m])))
    return by_site, by_stratum


def main():
    sp = pd.read_csv(MANIFESTS / "splits.csv")
    pool = sp[sp.split == "train"]
    fold_of = dict(zip(pool.site_id, pool.fold))
    makedirs(RESULTS)

    # 1-2. cross-validation
    cv, cv_sites = cells(pool.site_id.tolist(), lambda sid, s: PREDICTIONS / f"cv/fold{fold_of[sid]}/g{s}")
    y, (q10, q50, q90) = cv["y"], cv["q"]
    fold = np.array([fold_of[s] for s in cv_sites])[cv["site"]]
    lo, hi, lo_g, hi_g = (np.empty_like(y) for _ in range(4))
    per_fold = {}
    for k in np.unique(fold):
        m = fold == k
        c = constants(y[~m], q10[~m], q50[~m], q90[~m]) if (~m).any() else np.zeros((K["bands"], 2))
        g = constants(y[~m], q10[~m], q50[~m], q90[~m], 1) if (~m).any() else np.zeros((1, 2))
        lo[m], hi[m] = calibrate(q10[m], q50[m], q90[m], c)
        lo_g[m], hi_g[m] = calibrate(q10[m], q50[m], q90[m], g)
        per_fold[int(k)] = c.round(6).tolist()
    final = constants(y, q10, q50, q90)
    write_json(MANIFESTS / "calibration.json", dict(
        bands=final.round(6).tolist(), global_pair=constants(y, q10, q50, q90, 1)[0].round(6).tolist(),
        per_fold=per_fold, rule="per band of 0.1 of q50: (1 - alpha) quantile of q10 - label and of label - q90"))
    s1 = cv["single"]
    rows = [dict(set="cv", ensemble="one model", calibration="none", **metrics(y, s1[0], s1[1], s1[2])),
            dict(set="cv", ensemble="5 seeds", calibration="none", **metrics(y, q10, q50, q90)),
            dict(set="cv", ensemble="5 seeds", calibration="single pair", **metrics(y, lo_g, q50, hi_g)),
            dict(set="cv", ensemble="5 seeds", calibration="per band", **metrics(y, lo, q50, hi))]
    by_fold = [dict(fold=int(k), sites=int(sum(fold_of[s] == k for s in cv_sites)),
                    **metrics(y[fold == k], lo[fold == k], q50[fold == k], hi[fold == k])) for k in np.unique(fold)]
    cov = coverage_by_band("cv, per band", y, q50, lo, hi) + coverage_by_band("cv, single pair", y, q50, lo_g, hi_g)
    sites_rows, strata_rows = grouped("cv", cv, lo, hi, cv_sites, sp)

    # 3. held-out test set
    test_ids = sp[sp.split == "test"].site_id.tolist()
    if all((PREDICTIONS / f"final/g{s}" / "DONE").exists() for s in T["seeds"]):
        te, te_sites = cells(test_ids, lambda sid, s: PREDICTIONS / f"final/g{s}")
        ty, (t10, t50, t90) = te["y"], te["q"]
        tlo, thi = calibrate(t10, t50, t90, final)
        rows += [dict(set="test", ensemble="5 seeds", calibration="none", **metrics(ty, t10, t50, t90)),
                 dict(set="test", ensemble="5 seeds", calibration="per band", **metrics(ty, tlo, t50, thi))]
        cov += coverage_by_band("test, per band", ty, t50, tlo, thi)
        a, b = grouped("test", te, tlo, thi, te_sites, sp)
        sites_rows += a; strata_rows += b
    else:
        print("final models not finished: test set not evaluated")

    tables = dict(metrics=pd.DataFrame(rows), by_fold=pd.DataFrame(by_fold), by_site=pd.DataFrame(sites_rows),
                  by_stratum=pd.DataFrame(strata_rows), coverage_by_band=pd.DataFrame(cov))
    for name, df in tables.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    write_json(RESULTS / "summary.json", dict(
        sites_cv=len(cv_sites), sites_test=len(test_ids), cells_cv=int(len(y)),
        metrics=tables["metrics"].round(5).to_dict("records"), by_stratum=tables["by_stratum"].round(5).to_dict("records")))

    c = tables["coverage_by_band"]
    sets = c["set"].unique()
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 3.8), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sets):
        d = c[c["set"] == s]
        ax.plot(d.band + 0.05, d.below_q10, "o-", label="label below q10")
        ax.plot(d.band + 0.05, d.above_q90, "o-", label="label above q90")
        ax.axhline(K["alpha"], color="k", ls=":", lw=1)
        ax.set(title=s, xlabel="central map (bands of 0.1)", xlim=(0, 1))
    axes[0][0].set_ylabel("share of cells"); axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "coverage_by_band.png", dpi=120); plt.close(fig)
    pd.set_option("display.width", 200)
    print(tables["metrics"].round(4).to_string(index=False))
    print(tables["by_stratum"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
