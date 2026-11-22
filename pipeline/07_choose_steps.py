"""Step 7: choose the number of training steps from the step-candidate runs of step 6 (fold 0, first seed).

Rule fixed in advance (config.toml): the smallest candidate whose pooled out-of-fold quantile score on fold 0 is
within steps_tolerance (1%) of the best candidate. The 8,000 steps of the first version were tuned for 43 sites;
with several times more data, more steps may help.
Outputs: manifests/steps.json (read by step 6 for the cv and final runs), results/steps_choice.csv
Usage: python -I pipeline/07_choose_steps.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, LABELS, MANIFESTS, PREDICTIONS, RESULTS, makedirs, read_band, read_cover, write_json  # noqa: E402
from model import metrics  # noqa: E402

T = CFG["train"]


def main():
    sp = pd.read_csv(MANIFESTS / "splits.csv")
    fold0 = sp[(sp.split == "train") & (sp.fold == 0)].site_id.tolist()
    rows = []
    for s in T["steps_candidates"]:
        run = PREDICTIONS / f"steps/s{s}"
        if not (run / "DONE").exists():
            print(f"steps {s}: run not finished, skipped")
            continue
        ys, qs = [], []
        for sid in fold0:
            y = read_band(LABELS / f"{sid}.tif")
            q = read_cover(run / f"{sid}.tif")
            ok = np.isfinite(y) & np.isfinite(q).all(0)
            ys.append(y[ok]); qs.append(q[:, ok])
        y, q = np.concatenate(ys), np.concatenate(qs, axis=1)
        rows.append(dict(steps=s, sites=len(fold0), **metrics(y, *q)))
    t = pd.DataFrame(rows)
    best = t.quantile_score.min()
    ok = t[t.quantile_score <= best * (1 + T["steps_tolerance"])]
    chosen = int(ok.steps.min())
    t["chosen"] = t.steps == chosen
    makedirs(RESULTS)
    t.to_csv(RESULTS / "steps_choice.csv", index=False)
    write_json(MANIFESTS / "steps.json", dict(steps=chosen, rule=f"smallest candidate within "
                                              f"{T['steps_tolerance']:.0%} of the best quantile score on fold 0",
                                              scores=dict(zip(t.steps.astype(int).astype(str), t.quantile_score))))
    print(t[["steps", "mae", "quantile_score", "inside", "chosen"]].round(4).to_string(index=False))
    print(f"chosen: {chosen} steps -> {MANIFESTS / 'steps.json'}")


if __name__ == "__main__":
    main()
