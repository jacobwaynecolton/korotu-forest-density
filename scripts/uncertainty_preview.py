"""Does the spread between ensemble members flag where the model is wrong?

For every held-out site: spread = std of the three seed predictions per 10 m cell,
error = |ensemble - lidar|. Reports mean error by spread decile (pooled over val + test)
and the Spearman rank correlation, and draws one site: truth | ensemble | error | spread.
A first look at per-pixel confidence for the weighted confidence map, not a calibrated
uncertainty.

Writes data/manifest/uncertainty_preview.json and report/figures/uncertainty_<site>.png.

Usage: python -I scripts/uncertainty_preview.py <ensemble> <member> <member> ... --site <site_id>
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
PRED = ROOT / "data/predictions"
COVER = LinearSegmentedColormap.from_list(
    "cover", ["#f3f2ee", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
WARM = LinearSegmentedColormap.from_list("warm", ["#f3f2ee", "#f6c9b4", "#eb6834", "#8f2d0b"])


def rd(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float32)


def main(ens, members, fig_site):
    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    hold = sites[sites.split.isin(["val", "test"])].site_id
    S, E = [], []
    for site in hold:
        y = rd(ROOT / "data/labels" / f"{site}_cover10m.tif")
        p = rd(PRED / ens / f"{site}.tif")
        sd = np.std([rd(PRED / m / f"{site}.tif") for m in members], axis=0)
        ok = np.isfinite(y) & np.isfinite(p)
        S.append(sd[ok]); E.append(np.abs(p - y)[ok])
        if site == fig_site:
            err = np.where(ok, np.abs(p - y), np.nan)
            fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), dpi=110)
            for a, (t, img, cm, vmax) in zip(ax, [("Lidar truth", y, COVER, 1), ("Model (ensemble)", p, COVER, 1),
                                                   ("Actual error |model − lidar|", err, WARM, 0.5),
                                                   ("Ensemble spread (no lidar needed)", sd, WARM, 0.1)]):
                im = a.imshow(img, cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
                a.set_title(t, fontsize=10); a.axis("off")
                cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=7)
            fig.tight_layout()
            fig.savefig(ROOT / "report/figures" / f"uncertainty_{site}.png", facecolor="white")
            plt.close(fig)
    s, e = np.concatenate(S), np.concatenate(E)
    q = np.quantile(s, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(q, s, side="right") - 1, 0, 9)
    deciles = [dict(decile=int(k + 1), spread_lo=float(q[k]), spread_hi=float(q[k + 1]),
                    mean_error=float(e[idx == k].mean()), n=int((idx == k).sum())) for k in range(10)]
    rs = pd.Series(s).rank().corr(pd.Series(e).rank())
    out = dict(ensemble=ens, members=members, spearman=float(rs), deciles=deciles, figure_site=fig_site,
               error_top10_over_bottom10=float(deciles[-1]["mean_error"] / deciles[0]["mean_error"]))
    (ROOT / "data/manifest/uncertainty_preview.json").write_text(json.dumps(out, indent=1))
    print(f"Spearman(spread, error) = {rs:.3f}")
    for d in deciles:
        print(f"decile {d['decile']:2d}: spread {d['spread_lo']:.3f}-{d['spread_hi']:.3f}  mean error {d['mean_error']:.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    site = a[a.index("--site") + 1]
    rest = a[:a.index("--site")]
    main(rest[0], rest[1:], site)
