"""Data and images for the final-pipeline report (scratchpad)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

REPO = Path("/h/460/tsalin/projects/MLOps/korotu-forest-density")
sys.path.insert(0, str(REPO / "pipeline"))
from common import LABELS, MAPS, STACKS  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"; IMG.mkdir(exist_ok=True)
M, R = REPO / "pipeline/manifests", REPO / "pipeline/results"


def recs(df):
    return json.loads(df.to_json(orient="records", double_precision=5))


def rgb(path):
    with rasterio.open(path) as s:
        x = s.read([4, 3, 2]).astype(np.float32)
    x[x == 0] = np.nan
    out = np.zeros(x.shape, np.uint8)
    for i in range(3):
        v = x[i][np.isfinite(x[i])]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out[i] = (np.nan_to_num((x[i] - lo) / (hi - lo)).clip(0, 1) * 255).astype(np.uint8)
    return out.transpose(1, 2, 0)


def main():
    sites = pd.read_csv(M / "sites.csv")
    sp = pd.read_csv(M / "splits.csv")
    st = pd.read_csv(M / "site_status.csv")
    scenes = pd.read_csv(M / "s2_scenes.csv")
    s = sp.merge(sites[["site_id", "year", "source", "flight_start"]], on="site_id").merge(st[["site_id", "images"]], on="site_id", suffixes=("", "_n"))
    d = dict(
        sites=recs(s[["site_id", "lon", "lat", "split", "fold", "near_test", "lidar_partial", "lidar_stratum", "mean_cover", "images", "year", "source"]]),
        log=pd.read_csv(M / "selection_log.csv").outcome.value_counts().to_dict(),
        bank=int(len(pd.read_csv(M / "bank.csv"))),
        scenes=int(len(scenes)),
        metrics=recs(pd.read_csv(R / "metrics.csv")),
        by_stratum=recs(pd.read_csv(R / "by_stratum.csv")),
        by_fold=recs(pd.read_csv(R / "by_fold.csv")),
        by_site=recs(pd.read_csv(R / "by_site.csv")),
        coverage=recs(pd.read_csv(R / "coverage_by_band.csv")),
        steps=recs(pd.read_csv(R / "steps_choice.csv")),
        contour=json.loads((REPO / "report/ontario_outline.json").read_text()),
    )
    test = s[s.split == "test"]
    out = []
    for r in test.itertuples():
        with rasterio.open(LABELS / f"{r.site_id}.tif") as f:
            y = f.read(1)
        with rasterio.open(MAPS / f"{r.site_id}.tif") as f:
            q = f.read().astype(np.float32)
        q = np.where(q == 65535, np.nan, q / 1e4)
        h, w = y.shape
        pile = np.concatenate([y, q[0], q[1], q[2]], axis=1)
        g = np.where(np.isfinite(pile), np.round(np.clip(pile, 0, 1) * 250), 255).astype(np.uint8)
        Image.fromarray(g, mode="L").save(IMG / f"{r.site_id}_maps.png", optimize=True)
        k0 = scenes[scenes.site_id == r.site_id].sort_values("k").iloc[0]
        Image.fromarray(rgb(STACKS / f"{r.site_id}_k{k0.k}.tif")).save(IMG / f"{r.site_id}_s2.jpg", quality=85, optimize=True)
        out.append(dict(id=r.site_id, w=w, h=h, s2_date=k0.date))
    d["viewer"] = out
    (HERE / "data.json").write_text(json.dumps(d, separators=(",", ":")))
    print(len(out), "test sites;", round(sum(p.stat().st_size for p in IMG.iterdir()) / 1e6, 1), "MB images;",
          round((HERE / "data.json").stat().st_size / 1e3), "kB data")


if __name__ == "__main__":
    main()
