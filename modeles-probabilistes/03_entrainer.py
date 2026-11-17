"""Étape 3 (GPU) : entraîner une configuration pour un pli et une graine, puis prédire les sites qu'elle n'a pas vus.

  - pli 0 à 4 : entraînement sur les 4 autres plis (39 ou 40 sites), prédiction des sites du pli ;
  - pli « tous » : modèle final, entraîné sur les 49 sites, qui prédit les sites externes (15 nouveaux sites
    ontariens, 7 sites NEON). Seuls ces modèles finaux sont enregistrés, en demi-précision (quota du disque).

Entraînement identique à celui du projet (scripts/train_unet.py) : recadrages de 128 px tirés au hasard,
rotations et retournements, une date Sentinel-2 au hasard à chaque fois, AdamW, OneCycle, 8 000 pas, calcul en
bfloat16. Différence : pas de choix du meilleur point sur des sites de validation ; on garde le modèle final.

Sorties quantiles : le réseau prédit q50 et deux écarts, si bien que q10 ≤ q50 ≤ q90 restent dans [0, 1] :
  q50 = σ(z0),  q10 = q50 × (1 − σ(z1)),  q90 = q50 + (1 − q50) × σ(z2).
Perte : somme des pertes quantiles (« pinball ») pour 10, 50 et 90 %. La configuration L1 garde une sortie.

Prédiction comme dans le projet : fenêtres de 256 px qui se chevauchent, moyenne sur toutes les dates et sur
4 retournements. Pour les quantiles, c'est la moyenne des quantiles prédits pour chaque variante, qui garde
l'ordre q10 ≤ q50 ≤ q90.

Avec --tache, le pli et la graine sont lus dans SLURM_ARRAY_TASK_ID : tâche = indice du pli × 5 + indice de la
graine, avec les plis 0 à 4, puis « tous » pour les configurations quantiles.
Sorties : SORTIES/predictions/<config>/<pli0..4|final>/g<graine>/<site>.tif (entiers 16 bits, valeur × 10 000 ;
1 bande, ou q10/q50/q90), SORTIES/journaux/<config>_<pli>_g<graine>.csv, SORTIES/modeles/<config>/g<graine>.pt
Usage : python -I modeles-probabilistes/03_entrainer.py --config q_r34_imagenet --pli 0 --graine 0
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
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (CONFIGS, ECHELLE, GRAINES, JOURNAUX, LOT, LR, MODELES, N_PLIS, NODATA, PAS, POIDS,  # noqa: E402
                    PREDICTIONS, QUANTILES, RECADRAGE, fichiers_projet, nom_pli, sites_externes, sites_projet)


def lire_site(label, images):
    with rasterio.open(label) as s:
        y, prof = s.read(1), s.profile
    xs = []
    for p in images:
        with rasterio.open(p) as s:
            xs.append(s.read())
    return np.stack(xs), y, prof


def normalisation(cfg, xs):
    """Moyenne et écart-type par bande sur les sites d'entraînement (comme le projet), ou réflectance brute."""
    if cfg["normalisation"] == "reflectance":
        return np.zeros(12), np.ones(12)
    v = np.concatenate([x[:, :, ::7, ::7].transpose(1, 0, 2, 3).reshape(12, -1) for x in xs], axis=1)
    return np.nanmean(v, axis=1), np.nanstd(v, axis=1)


def vers_tenseur(x, moyenne, ecart, dev):
    x = (x - moyenne[None, :, None, None]) / ecart[None, :, None, None]
    return torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float16)).to(dev)


def construire(cfg, dev):
    modele = smp.Unet(cfg["encodeur"], encoder_weights="imagenet" if cfg["poids"] == "imagenet" else None,
                      in_channels=12, classes=1 if cfg["perte"] == "l1" else 3)
    if cfg["poids"] == "ssl4eo":
        w = torch.load(POIDS / "resnet50_ssl4eo_12bandes.pth", map_location="cpu", weights_only=True)
        manquantes = set(modele.encoder.state_dict()) - set(w)
        assert not manquantes, f"poids SSL4EO incomplets : {sorted(manquantes)[:5]}"
        modele.encoder.load_state_dict(w)
    return modele.to(dev)


def quantites(logits, perte):
    """Sorties du réseau -> couvert (1 canal) ou quantiles q10, q50, q90 (3 canaux), tous dans [0, 1]."""
    if perte == "l1":
        return torch.sigmoid(logits)
    q50 = torch.sigmoid(logits[:, 0])
    q10 = q50 * (1 - torch.sigmoid(logits[:, 1]))
    q90 = q50 + (1 - q50) * torch.sigmoid(logits[:, 2])
    return torch.stack([q10, q50, q90], dim=1)


def fonction_perte(q, y, perte):
    m = torch.isfinite(y)
    if perte == "l1":
        return (q[:, 0][m] - y[m]).abs().mean()
    total = 0.0
    for i, tau in enumerate(QUANTILES):
        e = y[m] - q[:, i][m]
        total = total + torch.maximum(tau * e, (tau - 1) * e).mean()
    return total


@torch.no_grad()
def predire(modele, xs, perte, tuile=256, recouvrement=64):
    """xs : (dates, 12, H, W) sur le GPU. Renvoie (canaux, H, W) en numpy, moyenne sur dates et retournements."""
    D, C, H, W = xs.shape
    K = 1 if perte == "l1" else 3
    out = torch.zeros(K, H, W, device=xs.device)
    poids = torch.zeros(H, W, device=xs.device)
    w1 = torch.hann_window(tuile, periodic=False, device=xs.device).clamp_min(1e-3)
    wt = w1[:, None] * w1[None, :]
    pas = tuile - recouvrement
    lignes = sorted(set(list(range(0, max(H - tuile, 0) + 1, pas)) + [max(H - tuile, 0)]))
    colonnes = sorted(set(list(range(0, max(W - tuile, 0) + 1, pas)) + [max(W - tuile, 0)]))
    pad_h, pad_w = max(0, tuile - H), max(0, tuile - W)
    for r in lignes:
        for c in colonnes:
            p = xs[:, :, r:r + tuile, c:c + tuile].float()
            if pad_h or pad_w:
                p = F.pad(p, (0, pad_w, 0, pad_h), mode="reflect")
            variantes = []
            for fh, fw in ((False, False), (True, False), (False, True), (True, True)):
                v = p.flip(-2) if fh else p
                v = v.flip(-1) if fw else v
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = quantites(modele(v).float(), perte)
                o = o.flip(-2) if fh else o
                o = o.flip(-1) if fw else o
                variantes.append(o)
            o = torch.stack(variantes).mean(dim=(0, 1))[:, : tuile - pad_h, : tuile - pad_w]
            ww = wt[: o.shape[1], : o.shape[2]]
            out[:, r:r + o.shape[1], c:c + o.shape[2]] += o * ww
            poids[r:r + o.shape[1], c:c + o.shape[2]] += ww
    return (out / poids).cpu().numpy()


def ecrire(chemin, p, prof):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    a = np.where(np.isfinite(p), np.round(np.clip(p, 0, 1) * ECHELLE), NODATA).astype(np.uint16)
    profil = dict(prof, count=a.shape[0], dtype="uint16", nodata=NODATA, compress="deflate", predictor=2)
    partiel = chemin.with_name(chemin.name + ".part")
    with rasterio.open(partiel, "w", **profil) as d:
        d.write(a)
    partiel.rename(chemin)


def tache_tableau(cfg):
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    plis = list(range(N_PLIS)) + (["tous"] if cfg["perte"] == "quantile" else [])
    return plis[i // len(GRAINES)], GRAINES[i % len(GRAINES)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--pli", default="0", help="0 à 4, ou « tous » pour le modèle final")
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--tache", action="store_true", help="pli et graine tirés de SLURM_ARRAY_TASK_ID")
    ap.add_argument("--pas", type=int, default=PAS)
    ap.add_argument("--max-sites", type=int, default=0, help="test rapide : limite le nombre de sites")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    pli, graine = tache_tableau(cfg) if args.tache else (args.pli if args.pli == "tous" else int(args.pli), args.graine)
    dossier = PREDICTIONS / args.config / nom_pli(pli) / f"g{graine}"
    if (dossier / "FAIT").exists():
        print(f"{args.config} {nom_pli(pli)} g{graine} : déjà fait")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(graine)
    np.random.seed(graine)
    t0 = time.time()

    s = sites_projet()
    entrainement = list(s.site_id if pli == "tous" else s[s.pli != pli].site_id)
    if pli == "tous":
        a_predire = [(e["site_id"], e["label"], e["images"]) for e in sites_externes()]
    else:
        a_predire = [(sid, *fichiers_projet(sid)) for sid in s[s.pli == pli].site_id]
    if args.max_sites:
        entrainement, a_predire = entrainement[:args.max_sites], a_predire[:args.max_sites]
    donnees = [lire_site(*fichiers_projet(sid))[:2] for sid in entrainement]
    moyenne, ecart = normalisation(cfg, [x for x, _ in donnees])
    train = [(vers_tenseur(x, moyenne, ecart, dev), torch.from_numpy(y).to(dev)) for x, y in donnees]
    del donnees
    print(f"{args.config} {nom_pli(pli)} g{graine} : {len(train)} sites d'entraînement, {len(a_predire)} à prédire, "
          f"{dev}{' ' + torch.cuda.get_device_name() if dev == 'cuda' else ''}", flush=True)

    modele = construire(cfg, dev)
    opt = torch.optim.AdamW(modele.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=args.pas, pct_start=0.05)
    tailles = np.array([y.numel() for _, y in train], dtype=float)
    probas = tailles / tailles.sum()
    cs = RECADRAGE

    def lot():
        xb, yb = [], []
        for i in np.random.choice(len(train), LOT, p=probas):
            x, y = train[i]
            d = np.random.randint(x.shape[0])
            r, c = np.random.randint(0, y.shape[0] - cs + 1), np.random.randint(0, y.shape[1] - cs + 1)
            xx, yy = x[d, :, r:r + cs, c:c + cs].float(), y[r:r + cs, c:c + cs]
            k = np.random.randint(4)
            xx, yy = torch.rot90(xx, k, (-2, -1)), torch.rot90(yy, k, (-2, -1))
            if np.random.rand() < 0.5:
                xx, yy = xx.flip(-1), yy.flip(-1)
            xb.append(xx)
            yb.append(yy)
        return torch.stack(xb), torch.stack(yb)

    journal = []
    modele.train()
    for etape in range(1, args.pas + 1):
        xb, yb = lot()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sortie = modele(xb).float()
        perte = fonction_perte(quantites(sortie, cfg["perte"]), yb, cfg["perte"])
        opt.zero_grad(set_to_none=True)
        perte.backward()
        opt.step()
        sched.step()
        if etape % 500 == 0 or etape == args.pas:
            journal.append(dict(etape=etape, perte=perte.item(), minutes=(time.time() - t0) / 60))
            print(f"étape {etape:5d}  perte {perte.item():.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    JOURNAUX.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(journal).to_csv(JOURNAUX / f"{args.config}_{nom_pli(pli)}_g{graine}.csv", index=False)

    modele.eval()
    for site_id, label, images in a_predire:
        x, _, prof = lire_site(label, images)
        p = predire(modele, vers_tenseur(x, moyenne, ecart, dev), cfg["perte"])
        ecrire(dossier / f"{site_id}.tif", p, prof)
        print(f"  {site_id} : {len(images)} dates, médiane prédite moyenne {np.nanmean(p[p.shape[0] // 2]):.3f}", flush=True)
    if pli == "tous":
        (MODELES / args.config).mkdir(parents=True, exist_ok=True)
        torch.save({k: v.half() for k, v in modele.state_dict().items()}, MODELES / args.config / f"g{graine}.pt")
        (MODELES / args.config / f"g{graine}.json").write_text(json.dumps(
            dict(config=args.config, **cfg, graine=graine, pas=args.pas, moyenne=list(map(float, moyenne)),
                 ecart=list(map(float, ecart)), sites=entrainement), indent=1))
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "FAIT").write_text(f"{(time.time() - t0) / 60:.1f} min\n")
    print(f"terminé en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
