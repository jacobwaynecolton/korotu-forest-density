"""Train a U-Net to predict 10 m canopy cover from 12-band Sentinel-2 L2A.

Data: data/pairs/stacks/<site>_k<k>.tif (all dates) + data/labels/<site>_cover10m.tif.
Everything is held on the GPU; each step samples random 128x128 crops from random
training sites and dates, with flips / 90-degree rotations.
Loss: L1 on cover (the headline metric is MAE), masked where the label is missing.
Model selection: MAE on the validation sites, predicting full sites with overlapping
windows and averaging over all dates (+ flips). Test sites are only predicted at the end
with the selected checkpoint.

Outputs: models/<name>/{best.pt,config.json,log.csv}, data/predictions/<name>/<site>.tif

Usage: python -I scripts/train_unet.py --name unet_r34 [--steps 6000] [--encoder resnet34]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
STACKS, LABELS = ROOT / "data/pairs/stacks", ROOT / "data/labels"


def load_site(site):
    with rasterio.open(LABELS / f"{site}_cover10m.tif") as s:
        y, prof = s.read(1), s.profile
    xs = []
    for p in sorted(STACKS.glob(f"{site}_k*.tif")):
        with rasterio.open(p) as s:
            xs.append(s.read())
    return (np.stack(xs) if xs else None), y, prof


def norm_stats(train_sites):
    acc = []
    for s in train_sites:
        x, _, _ = load_site(s)
        if x is None:
            continue
        v = x[:, :, ::7, ::7].transpose(1, 0, 2, 3).reshape(12, -1)
        acc.append(v)
    v = np.concatenate(acc, axis=1)
    return np.nanmean(v, axis=1), np.nanstd(v, axis=1)


def to_tensor(x, mean, std, dev):
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    x = np.nan_to_num(x, nan=0.0).astype(np.float16)
    return torch.from_numpy(x).to(dev)


@torch.no_grad()
def predict_site(model, xs, tile=256, overlap=64, flips=True):
    """xs: (D, C, H, W) tensor on device. Returns (H, W) numpy mean over dates/flips."""
    D, C, H, W = xs.shape
    out = torch.zeros(H, W, device=xs.device); wsum = torch.zeros(H, W, device=xs.device)
    w1 = torch.hann_window(tile, periodic=False, device=xs.device).clamp_min(1e-3)
    wt = w1[:, None] * w1[None, :]
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) + [max(H - tile, 0)]
    xs_ = list(range(0, max(W - tile, 0) + 1, step)) + [max(W - tile, 0)]
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    for r in sorted(set(ys)):
        for c in sorted(set(xs_)):
            patch = xs[:, :, r:r + tile, c:c + tile].float()
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
            preds = []
            variants = [(False, False), (True, False), (False, True), (True, True)] if flips else [(False, False)]
            for fh, fw in variants:
                p = patch
                if fh: p = p.flip(-2)
                if fw: p = p.flip(-1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = torch.sigmoid(model(p).float())
                if fh: o = o.flip(-2)
                if fw: o = o.flip(-1)
                preds.append(o)
            o = torch.stack(preds).mean(dim=(0, 1))[0][: tile - pad_h, : tile - pad_w]
            ww = wt[: o.shape[0], : o.shape[1]]
            out[r:r + o.shape[0], c:c + o.shape[1]] += o * ww
            wsum[r:r + o.shape[0], c:c + o.shape[1]] += ww
    return (out / wsum).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="unet_r34")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    out_dir = ROOT / "models" / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = ROOT / "data/predictions" / args.name; pred_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(ROOT / "data/manifest/sites.csv")
    split = dict(zip(sites.site_id, sites.split))
    train_ids = [s for s in sites.site_id if split[s] == "train"]
    mean, std = norm_stats(train_ids)

    train = []
    for s in train_ids:
        x, y, _ = load_site(s)
        if x is None:
            print("no imagery, skipped:", s); continue
        train.append((to_tensor(x, mean, std, dev), torch.from_numpy(y).to(dev)))
    holdout = {}
    for s in sites.site_id:
        if split[s] in ("val", "test"):
            x, y, prof = load_site(s)
            holdout[s] = (to_tensor(x, mean, std, dev), y, prof, split[s])
    print(f"train sites {len(train)} ({sum(t[0].shape[0] for t in train)} site-dates); "
          f"GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=12, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    sizes = np.array([t[1].numel() for t in train], dtype=float); probs = sizes / sizes.sum()
    cs = args.crop

    def batch():
        xb, yb = [], []
        for i in np.random.choice(len(train), args.batch, p=probs):
            x, y = train[i]
            d = np.random.randint(x.shape[0])
            r = np.random.randint(0, y.shape[0] - cs + 1); c = np.random.randint(0, y.shape[1] - cs + 1)
            xx, yy = x[d, :, r:r + cs, c:c + cs].float(), y[r:r + cs, c:c + cs]
            k = np.random.randint(4)
            xx, yy = torch.rot90(xx, k, (-2, -1)), torch.rot90(yy, k, (-2, -1))
            if np.random.rand() < 0.5: xx, yy = xx.flip(-1), yy.flip(-1)
            xb.append(xx); yb.append(yy)
        return torch.stack(xb), torch.stack(yb)

    def evaluate(which="val", save=False):
        model.eval(); errs = []
        for s, (x, y, prof, sp) in holdout.items():
            if sp != which: continue
            p = predict_site(model, x)
            ok = np.isfinite(y)
            errs.append(np.abs(p - y)[ok])
            if save:
                with rasterio.open(pred_dir / f"{s}.tif", "w", **dict(prof, count=1, dtype="float32", nodata=np.nan)) as d:
                    d.write(np.where(ok, p, np.nan).astype(np.float32), 1)
        model.train()
        return float(np.concatenate(errs).mean())

    log, best, t0 = [], math.inf, time.time()
    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb).float()[:, 0]
        p = torch.sigmoid(out)
        m = torch.isfinite(yb)
        loss = (p[m] - yb[m]).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate("val")
            log.append(dict(step=step, train_l1=loss.item(), val_mae=v, minutes=(time.time() - t0) / 60))
            flag = ""
            if v < best:
                best = v; flag = " *"
                torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"step {step:5d}  train L1 {loss.item():.4f}  val MAE {v:.4f}{flag}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best.pt"))
    val_mae = evaluate("val", save=True); test_mae = evaluate("test", save=True)
    cfg = dict(vars(args), mean=mean.tolist(), std=std.tolist(), best_val_mae=val_mae,
               test_mae=test_mae, train_sites=len(train), minutes=(time.time() - t0) / 60)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"done: best val MAE {val_mae:.4f}, test MAE {test_mae:.4f}, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
