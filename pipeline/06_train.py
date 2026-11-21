"""Step 6 (GPU): train one U-Net with three quantile outputs and predict the sites it has not seen.

Three kinds of runs, chosen with --mode and the task index (Slurm array):
  steps : fold 0, first seed, one run per candidate step count (steps_candidates). Used by step 7 to choose the
          number of steps. Predicts the sites of fold 0.
  cv    : task = fold x n_seeds + seed index. Trains on the other folds, minus the sites within buffer_km of the
          fold, predicts the sites of the fold (out-of-fold predictions for calibration and evaluation).
  final : one run per seed. Trains on every training site except those within buffer_km of a test site, predicts
          the test sites, and saves the model.

Training: random 128 px crops from random sites (probability proportional to labelled cells) and random dates,
random 90-degree rotations and flips; label cells that the scene classification flags (cloud, shadow, ...) on
that date are left out of the loss. AdamW, one-cycle schedule, bfloat16 on the GPU. Inputs: 12 bands, reflectance
z-scored with the training sites' statistics; no data -> 0 (the band mean). The data stay in CPU memory (uint16)
and only the batches go to the GPU, so hundreds of sites fit.
Prediction: overlapping 256 px windows, 4 flips, dates averaged pixel by pixel over the unflagged dates.

A run whose DONE file exists is skipped.
Outputs: data/predictions/<run>/<site>.tif (uint16 q10, q50, q90 x 10 000), data/predictions/<run>/DONE,
data/models/logs/<run>.csv; final runs also data/models/final_g<seed>.pt and .json.
Usage: python -I pipeline/06_train.py --mode cv --task 7   [--steps N for a quick test]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, LABELS, MANIFESTS, MODELS, PREDICTIONS, STACKS, km_between, write_cover, write_json  # noqa: E402
from model import build, normalise, norm_stats, pinball_loss, predict, quantiles  # noqa: E402
from sentinel2 import load_stack, reflectance_and_mask  # noqa: E402

T, SP = CFG["train"], CFG["split"]


def chosen_steps():
    f = MANIFESTS / "steps.json"
    if f.exists():
        return int(json.loads(f.read_text())["steps"])
    print(f"warning: {f} not found, using steps_default = {T['steps_default']}")
    return T["steps_default"]


def run_spec(mode, task):
    seeds = T["seeds"]
    if mode == "steps":
        s = T["steps_candidates"][task]
        return dict(name=f"steps/s{s}", fold=0, seed=seeds[0], steps=s)
    if mode == "cv":
        fold, seed = task // len(seeds), seeds[task % len(seeds)]
        return dict(name=f"cv/fold{fold}/g{seed}", fold=fold, seed=seed, steps=chosen_steps())
    if mode == "final":
        return dict(name=f"final/g{seeds[task]}", fold=None, seed=seeds[task], steps=chosen_steps())
    raise ValueError(mode)


def site_lists(fold):
    sp = pd.read_csv(MANIFESTS / "splits.csv")
    pool = sp[sp.split == "train"]
    if fold is None:
        return pool[~pool.near_test].site_id.tolist(), sp[sp.split == "test"].site_id.tolist()
    held = pool[pool.fold == fold]
    rest = pool[pool.fold != fold]
    far = [km_between(r.lon, r.lat, held.lon.values, held.lat.values).min() >= SP["buffer_km"] for r in rest.itertuples()]
    return rest[far].site_id.tolist(), held.site_id.tolist()


def stacks_of(sid):
    return sorted(STACKS.glob(f"{sid}_k*.tif"))


def load_site(sid):
    with rasterio.open(LABELS / f"{sid}.tif") as s:
        y, prof = s.read(1).astype(np.float32), s.profile
    return load_stack(stacks_of(sid)), y, prof


def batch(data, probs, mean, std, rng, n, cs):
    xb, yb = [], []
    for i in rng.choice(len(data), n, p=probs):
        x, y = data[i]
        d = rng.integers(x.shape[0])
        r, c = rng.integers(0, y.shape[0] - cs + 1), rng.integers(0, y.shape[1] - cs + 1)
        refl, bad = reflectance_and_mask(x[d, :, r:r + cs, c:c + cs])
        yy = np.where(bad, np.nan, y[r:r + cs, c:c + cs])
        xx = normalise(refl, mean, std)
        k = rng.integers(4)
        xx, yy = np.rot90(xx, k, (-2, -1)), np.rot90(yy, k, (-2, -1))
        if rng.random() < 0.5:
            xx, yy = xx[..., ::-1], yy[..., ::-1]
        xb.append(np.ascontiguousarray(xx)); yb.append(np.ascontiguousarray(yy))
    return np.stack(xb), np.stack(yb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["steps", "cv", "final"])
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--steps", type=int, help="override the step count (quick tests)")
    ap.add_argument("--max-sites", type=int, help="use only the first N training sites (quick tests)")
    args = ap.parse_args()
    spec = run_spec(args.mode, args.task)
    if args.steps:
        spec["steps"] = args.steps
    out_dir = PREDICTIONS / spec["name"]
    if (out_dir / "DONE").exists():
        print(f"{spec['name']}: already done")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(spec["seed"])
    rng = np.random.default_rng(spec["seed"])
    train_ids, pred_ids = site_lists(spec["fold"])
    train_ids = train_ids[:args.max_sites] if args.max_sites else train_ids
    t0 = time.time()
    data = []
    for sid in train_ids:
        x, y, _ = load_site(sid)
        data.append((x, y))
    mean, std = norm_stats([reflectance_and_mask(x[:, :, ::7, ::7])[0] for x, _ in data])
    sizes = np.array([np.isfinite(y).sum() for _, y in data], float)
    probs = sizes / sizes.sum()
    gb = sum(x.nbytes for x, _ in data) / 1e9
    print(f"{spec['name']}: {spec['steps']} steps, {len(data)} training sites ({gb:.1f} GB in memory), "
          f"{len(pred_ids)} sites to predict, device {dev}", flush=True)

    model = build().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=T["lr"], total_steps=spec["steps"], pct_start=0.05)
    log = []
    model.train()
    for step in range(1, spec["steps"] + 1):
        xb, yb = batch(data, probs, mean, std, rng, T["batch"], T["crop"])
        xb = torch.from_numpy(xb).to(dev).float()
        yb = torch.from_numpy(yb).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(xb).float()
        loss = pinball_loss(quantiles(logits), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 500 == 0 or step == spec["steps"]:
            log.append(dict(step=step, loss=loss.item(), minutes=(time.time() - t0) / 60))
            print(f"  step {step:6d}  loss {loss.item():.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    (MODELS / "logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(MODELS / "logs" / (spec["name"].replace("/", "_") + ".csv"), index=False)

    model.eval()
    for sid in pred_ids:
        x, y, prof = load_site(sid)
        refl, bad = reflectance_and_mask(x)
        xt = torch.from_numpy(normalise(refl, mean, std)).to(dev)
        q = predict(model, xt, torch.from_numpy(bad).to(dev))
        write_cover(out_dir / f"{sid}.tif", q, dict(prof, count=3), ["q10", "q50", "q90"])
    info = dict(spec, train_sites=train_ids, predicted_sites=pred_ids, mean=mean.tolist(), std=std.tolist(),
                minutes=round((time.time() - t0) / 60, 1))
    if args.mode == "final":
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.half() for k, v in model.state_dict().items()}, MODELS / f"final_g{spec['seed']}.pt")
        write_json(MODELS / f"final_g{spec['seed']}.json", info)
    write_json(out_dir / "DONE", info)
    print(f"{spec['name']}: done in {info['minutes']} min")


if __name__ == "__main__":
    main()
