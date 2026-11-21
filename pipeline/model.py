"""The final model: U-Net with three quantile outputs, seed ensemble, per-band conformal calibration.

Choices come from modeles-probabilistes/ (site-level cross-validation, November 2026): ResNet-34 ImageNet encoder,
q10/q50/q90 with a pinball loss, 5 seeds combined as a mixture, conformal constants per band of 0.1 of q50.
"""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from common import CFG

T, K = CFG["train"], CFG["calibration"]
QUANTILES = tuple(T["quantiles"])


def build(pretrained=True):
    m = smp.Unet(T["encoder"], encoder_weights=T["weights"] if pretrained else None, in_channels=12,
                 classes=len(QUANTILES))
    return m


def quantiles(logits):
    """Network outputs -> q10 <= q50 <= q90 in [0, 1]:
    q50 = s(z0), q10 = q50 (1 - s(z1)), q90 = q50 + (1 - q50) s(z2)."""
    q50 = torch.sigmoid(logits[:, 0])
    q10 = q50 * (1 - torch.sigmoid(logits[:, 1]))
    q90 = q50 + (1 - q50) * torch.sigmoid(logits[:, 2])
    return torch.stack([q10, q50, q90], dim=1)


def pinball_loss(q, y):
    """Sum of pinball losses at 10, 50 and 90%, over cells with a label (NaN = no label or bad pixel)."""
    m = torch.isfinite(y)
    total = 0.0
    for i, tau in enumerate(QUANTILES):
        e = y[m] - q[:, i][m]
        total = total + torch.maximum(tau * e, (tau - 1) * e).mean()
    return total


def norm_stats(refl_list):
    """Per-band mean and std of reflectance over the training sites (every 7th pixel), ignoring NaN."""
    v = np.concatenate([r[:, :, ::7, ::7].transpose(1, 0, 2, 3).reshape(12, -1) for r in refl_list], axis=1)
    return np.nanmean(v, axis=1), np.nanstd(v, axis=1)


def normalise(refl, mean, std):
    """(..., 12, H, W) reflectance -> z-scores, NaN -> 0 (the band mean)."""
    z = (refl - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(z, nan=0.0).astype(np.float16)


@torch.no_grad()
def predict(model, x, bad, tile=None, overlap=None):
    """Predict one site.

    x: (dates, 12, H, W) normalised tensor on the device; bad: (dates, H, W) bool tensor (SCL cloud, shadow...).
    Overlapping windows with Hann weights, 4 flips per window; dates are averaged pixel by pixel over the dates
    where the pixel is not bad (all dates if it is bad everywhere). Returns (3, H, W) numpy: q10, q50, q90."""
    tile, overlap = tile or T["tile"], overlap or T["overlap"]
    D, C, H, W = x.shape
    good = (~bad).float()
    good = torch.where(good.sum(0, keepdim=True) == 0, torch.ones_like(good), good)   # all bad: use all dates
    out = torch.zeros(3, H, W, device=x.device)
    wsum = torch.zeros(H, W, device=x.device)
    w1 = torch.hann_window(tile, periodic=False, device=x.device).clamp_min(1e-3)
    wt = w1[:, None] * w1[None, :]
    step = tile - overlap
    rows = sorted(set(list(range(0, max(H - tile, 0) + 1, step)) + [max(H - tile, 0)]))
    cols = sorted(set(list(range(0, max(W - tile, 0) + 1, step)) + [max(W - tile, 0)]))
    ph, pw = max(0, tile - H), max(0, tile - W)
    for r in rows:
        for c in cols:
            p = x[:, :, r:r + tile, c:c + tile].float()
            if ph or pw:
                p = F.pad(p, (0, pw, 0, ph), mode="reflect")
            variants = []
            for fh, fw in ((False, False), (True, False), (False, True), (True, True)):
                v = p.flip(-2) if fh else p
                v = v.flip(-1) if fw else v
                with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
                    o = quantiles(model(v).float())
                o = o.flip(-2) if fh else o
                o = o.flip(-1) if fw else o
                variants.append(o)
            o = torch.stack(variants).mean(0)[:, :, : tile - ph, : tile - pw]         # (D, 3, h, w)
            g = good[:, r:r + o.shape[2], c:c + o.shape[3]][:, None]                 # (D, 1, h, w)
            o = (o * g).sum(0) / g.sum(0)
            ww = wt[: o.shape[1], : o.shape[2]]
            out[:, r:r + o.shape[1], c:c + o.shape[2]] += o * ww
            wsum[r:r + o.shape[1], c:c + o.shape[2]] += ww
    return (out / wsum).cpu().numpy()


# ---------- seed ensemble ----------
LEVELS = np.linspace(0.05, 0.95, 19)


def levels(q):
    """(3, ...) q10, q50, q90 -> (19, ...) quantiles 5% to 95%, linearly interpolated and extended."""
    q10, q50, q90 = q
    out = [q10 + (u - 0.1) / 0.4 * (q50 - q10) if u <= 0.5 else q50 + (u - 0.5) / 0.4 * (q90 - q50) for u in LEVELS]
    return np.clip(np.stack(out), 0, 1)


def mixture(preds):
    """List of (3, ...) predictions, one per seed -> (3, ...) q10, q50, q90 of the mixture of the seeds.

    Each seed is a distribution given by 19 quantiles at k/20; the 'weibull' plotting position k/(n+1) matches
    these levels, so one seed (or identical seeds) gives back its own q10, q50, q90."""
    return np.quantile(np.concatenate([levels(p) for p in preds]), QUANTILES, axis=0,
                       method="weibull").astype(np.float32)


# ---------- conformal calibration ----------
def band_of(q50, n):
    return np.minimum(np.nan_to_num(q50 * n).astype(int), n - 1)


def constants(y, q10, q50, q90, n=None):
    """Split-conformal corrections, one (c_low, c_high) pair per band of q50: the finite-sample (1 - alpha) quantile
    of (q10 - label) and of (label - q90) within the band. n = 1 gives one pair for the whole map. A band with
    fewer than min_cells cells uses the global pair. A negative constant narrows the interval."""
    n = n or K["bands"]

    def one(m):
        k = int(m.sum())
        level = min(1.0, np.ceil((k + 1) * (1 - K["alpha"])) / k)
        return float(np.quantile(q10[m] - y[m], level)), float(np.quantile(y[m] - q90[m], level))

    glob = one(np.ones(len(y), bool))
    b = band_of(q50, n)
    return np.array([one(b == i) if (b == i).sum() >= K["min_cells"] else glob for i in range(n)])


def calibrate(q10, q50, q90, consts):
    """Calibrated bounds q10 - c_low and q90 + c_high (constants of the q50 band), kept in [0, 1] and around q50."""
    c = np.asarray(consts)[band_of(q50, len(consts))]
    return (np.minimum(np.clip(q10 - c[..., 0], 0, 1), q50),
            np.maximum(np.clip(q90 + c[..., 1], 0, 1), q50))


def pinball(y, q, tau):
    e = y - q
    return float(np.maximum(tau * e, (tau - 1) * e).mean())


def metrics(y, q10, q50, q90):
    """Central-map error and interval coverage on flat arrays of cells with a label."""
    e = q50 - y
    return dict(cells=int(len(y)), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                bias=float(e.mean()),
                r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 else np.nan,
                quantile_score=float(np.mean([pinball(y, q, t) for q, t in zip((q10, q50, q90), QUANTILES)])),
                below_q10=float((y < q10).mean()), above_q90=float((y > q90).mean()),
                inside=float(((y >= q10) & (y <= q90)).mean()), width=float((q90 - q10).mean()))
