"""Step 5: usable sites, held-out test set and cross-validation folds, all by spatial groups.

1. A site is usable if its label covers at least min_valid_share of its grid and at least one image was written.
   Each site gets its lidar partial-cover stratum: terciles of the share of labelled cells with cover in
   [partial_low, partial_high), measured on the label itself.
2. Spatial groups: sites in the same cell_km x cell_km cell (Canada Atlas Lambert) stay together, in the test set
   or in one fold, never split.
3. Test set: whole groups, about test_share of the sites. Among `draws` random draws (fixed seed), the one whose
   size and stratum mix are closest to the target and to the whole dataset is kept, with at least two sites of
   each stratum. Training sites within buffer_km of a test site are flagged near_test: the final models never
   train on them (they still take part in cross-validation).
4. Folds: the other groups are spread over `folds` folds, again the most balanced of `draws` draws (sizes and
   stratum mix). During cross-validation, a model also skips the training sites within buffer_km of its fold.
Outputs (manifests/): site_status.csv, splits.csv
Usage: python -I pipeline/05_splits.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, MANIFESTS, STATUS, km_between, sites  # noqa: E402

SP, S = CFG["split"], CFG["sites"]


def status_table(s):
    rows = []
    for r in s.itertuples():
        lab, img = STATUS / "labels" / f"{r.site_id}.json", STATUS / "images" / f"{r.site_id}.json"
        d = dict(site_id=r.site_id, label_ok=lab.exists(), images=0)
        if lab.exists():
            j = json.loads(lab.read_text())
            d.update(cells=j["cells"], full_share=j["full_share"], mean_cover=j["mean_cover"],
                     lidar_partial=j["partial_share"], missing_chm_pixels=j["missing_chm_pixels"])
        if img.exists():
            d["images"] = len(json.loads(img.read_text())["scenes_ok"])
        rows.append(d)
    st = pd.DataFrame(rows)
    st["usable"] = st.label_ok & (st.get("full_share", 0) >= CFG["labels"]["min_valid_share"]) & (st.images > 0)
    return st


def mix(df, col="lidar_stratum"):
    return df[col].value_counts(normalize=True).reindex(S["strata"], fill_value=0).to_numpy()


def choose_test(u, rng):
    groups = u.group.unique()
    n_target = SP["test_share"] * len(u)
    p_all = mix(u)
    best, best_score = None, np.inf
    for _ in range(SP["draws"]):
        chosen, n = [], 0
        for g in rng.permutation(groups):
            if n >= n_target:
                break
            chosen.append(g)
            n += (u.group == g).sum()
        t = u[u.group.isin(chosen)]
        counts = t.lidar_stratum.value_counts().reindex(S["strata"], fill_value=0)
        score = abs(len(t) - n_target) / n_target + np.abs(mix(t) - p_all).sum() + (counts < 2).sum()
        if score < best_score:
            best, best_score = set(chosen), score
    return best


def choose_folds(tr, rng):
    groups = tr.group.unique()
    p_all = mix(tr)
    best, best_score = None, np.inf
    for _ in range(SP["draws"]):
        a = dict(zip(groups, rng.integers(0, SP["folds"], len(groups))))
        f = tr.group.map(a)
        sizes = np.bincount(f, minlength=SP["folds"])
        if (sizes == 0).any():
            continue
        score = sizes.std() / sizes.mean() + np.mean([np.abs(mix(tr[f == k]) - p_all).sum() for k in range(SP["folds"])])
        if score < best_score:
            best, best_score = a, score
    return best


def main():
    s = sites()
    st = status_table(s)
    st.to_csv(MANIFESTS / "site_status.csv", index=False)
    u = s.merge(st[st.usable], on="site_id").reset_index(drop=True)
    print(f"{len(u)} usable sites out of {len(s)} "
          f"(no label: {(~st.label_ok).sum()}, no image: {(st.images == 0).sum()})")
    u["lidar_stratum"] = pd.qcut(u.lidar_partial.rank(method="first"), 3, labels=S["strata"][::-1]).astype(str)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True).transform(u.lon.values, u.lat.values)
    c = SP["cell_km"] * 1000
    u["group"] = [f"g{int(a // c)}_{int(b // c)}" for a, b in zip(x, y)]
    rng = np.random.default_rng(SP["seed"])
    test_groups = choose_test(u, rng)
    u["split"] = np.where(u.group.isin(test_groups), "test", "train")
    te = u[u.split == "test"]
    near = np.array([km_between(r.lon, r.lat, te.lon.values, te.lat.values).min() < SP["buffer_km"]
                     if r.split == "train" else False for r in u.itertuples()])
    u["near_test"] = near
    tr = u[u.split == "train"]
    folds = choose_folds(tr, rng)
    u["fold"] = [int(folds[g]) if sp == "train" else -1 for g, sp in zip(u.group, u.split)]
    cols = ["site_id", "lon", "lat", "group", "split", "fold", "near_test", "lidar_partial", "lidar_stratum",
            "stratum", "mean_cover", "images"]
    u[cols].to_csv(MANIFESTS / "splits.csv", index=False)
    print(f"{u.group.nunique()} spatial groups of {SP['cell_km']} km")
    print(f"test: {len(te)} sites in {te.group.nunique()} groups; {int(near.sum())} training sites within "
          f"{SP['buffer_km']} km of a test site (left out of the final models)")
    t = u.groupby(["split", "fold", "lidar_stratum"]).size().unstack(fill_value=0)
    print(t.to_string())


if __name__ == "__main__":
    main()
