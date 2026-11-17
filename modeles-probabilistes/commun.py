"""Constantes et petites fonctions partagées par les scripts des modèles probabilistes.

Trois configurations de U-Net, toutes entraînées comme le modèle du projet (scripts/train_unet.py : recadrages
de 128 px, retournements et rotations, chaque date Sentinel-2 comme augmentation, AdamW, 8 000 pas) :
  - l1_r34_imagenet : le modèle actuel (encodeur ResNet-34 ImageNet, perte L1, une sortie) ;
  - q_r34_imagenet  : le même encodeur, trois sorties (quantiles 10, 50 et 90 %) et perte quantile ;
  - q_r50_ssl4eo    : trois sorties, encodeur ResNet-50 pré-entraîné sur Sentinel-2 (SSL4EO-S12, MoCo).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJET / "scripts"))
ICI = Path(os.environ.get("PROBA_TRAVAIL", Path(__file__).resolve().parent))
RESULTATS = ICI / "resultats"
SORTIES = Path(os.environ.get("PROBA_SORTIES", "/w/20261/tsalin/MLOps/modeles_probabilistes"))
PREDICTIONS, MODELES, JOURNAUX, POIDS = (SORTIES / d for d in ("predictions", "modeles", "journaux", "poids"))
POIDS_SSL4EO = ("torchgeo/resnet50_sentinel2_all_moco", "resnet50_sentinel2_all_moco-df8b932e.pth")

# Sites de test externes (jamais utilisés pour entraîner ni pour choisir) et sites hors domaine
ONTARIO = dict(sites=PROJET / "independant-verification/model-verification/sites.csv",
               donnees=Path("/w/20261/tsalin/MLOps/independant_verification/modele"))
NEON = dict(sites=PROJET / "independant-verification/model-verification-USA/sites.csv",
            donnees=Path("/w/20261/tsalin/MLOps/independant_verification/modele_usa"))

QUANTILES = (0.1, 0.5, 0.9)
N_PLIS = 5
GRAINES = (0, 1, 2, 3, 4)
PAS, LOT, RECADRAGE, LR = 8000, 32, 128, 1e-3          # comme le modèle du projet
CONFIGS = {
    "l1_r34_imagenet": dict(encodeur="resnet34", poids="imagenet", perte="l1", normalisation="zscore"),
    "q_r34_imagenet": dict(encodeur="resnet34", poids="imagenet", perte="quantile", normalisation="zscore"),
    # Le pré-entraînement SSL4EO a vu des réflectances brutes (comptes numériques / 10 000) : on les lui donne telles quelles.
    "q_r50_ssl4eo": dict(encodeur="resnet50", poids="ssl4eo", perte="quantile", normalisation="reflectance"),
}
ECHELLE = 10000          # prédictions enregistrées en entiers 16 bits : valeur × 10 000
NODATA = 65535


def sites_projet():
    """Les 49 sites du projet avec leur pli de validation croisée (plis.csv, étape 1)."""
    return pd.read_csv(ICI / "plis.csv")


def sites_externes():
    """Les 15 nouveaux sites ontariens et les 7 sites NEON : identifiant, groupe, chemins du label et des images."""
    lignes = []
    for groupe, src in (("ontario", ONTARIO), ("neon", NEON)):
        s = pd.read_csv(src["sites"])
        for r in s.itertuples():
            images = sorted((src["donnees"] / "stacks").glob(f"{r.site_id}_k*.tif"))
            if images:
                lignes.append(dict(site_id=r.site_id, groupe=groupe if groupe == "ontario" else f"neon_{r.groupe}",
                                   label=src["donnees"] / "labels" / f"{r.site_id}_cover10m.tif", images=images))
    return lignes


def fichiers_projet(site_id):
    return (PROJET / "data/labels" / f"{site_id}_cover10m.tif",
            sorted((PROJET / "data/pairs/stacks").glob(f"{site_id}_k*.tif")))


def lire(path, bande=1):
    with rasterio.open(path) as s:
        return s.read(bande).astype(np.float32)


def lire_prediction(path):
    """Prédiction enregistrée -> tableau (bandes, lignes, colonnes) en fraction de couvert, NaN hors données."""
    with rasterio.open(path) as s:
        a = s.read().astype(np.float32)
    return np.where(a == NODATA, np.nan, a / ECHELLE)


def nom_pli(pli):
    return "final" if pli == "tous" else f"pli{pli}"


# ---------- ensemble de graines et calibration (04_evaluer.py, 05_cartes.py) ----------
NIVEAUX = np.linspace(0.05, 0.95, 19)
ALPHA = 0.1              # intervalle 10-90 % : 10 % de manques visés de chaque côté
TRANCHES = 10            # calibration par tranche de 0,1 de la carte centrale
MIN_CASES = 1000         # tranche plus petite : constantes globales


def niveaux(q):
    """(3, ...) q10, q50, q90 -> (19, ...) quantiles 5 % à 95 %, interpolés et prolongés linéairement."""
    q10, q50, q90 = q
    out = [q10 + (u - 0.1) / 0.4 * (q50 - q10) if u <= 0.5 else q50 + (u - 0.5) / 0.4 * (q90 - q50) for u in NIVEAUX]
    return np.clip(np.stack(out), 0, 1)


def melange(preds):
    """Liste de (3, ...) par graine -> (3, ...) quantiles 10, 50, 90 % du mélange des graines.

    Les 19 niveaux sont k/20 (k = 1..19) : la position k/(n+1) de la méthode « weibull » leur correspond, si bien
    qu'une graine seule, ou cinq graines identiques, redonnent exactement leurs q10, q50, q90. La méthode linéaire
    par défaut resserrait l'intervalle d'environ 5 % de chaque demi-largeur.
    """
    return np.quantile(np.concatenate([niveaux(p) for p in preds]), QUANTILES, axis=0,
                       method="weibull").astype(np.float32)


def tranche(q50, n):
    """Indice de tranche (0..n-1) de la carte centrale."""
    return np.minimum(np.nan_to_num(q50 * n).astype(int), n - 1)


def constantes(y, q10, q50, q90, n=TRANCHES):
    """Corrections conformes des deux bornes : tableau (n, 2) de (c_bas, c_haut), une ligne par tranche de q50.

    c_bas est le quantile à 90 % de (q10 − label), c_haut celui de (label − q90), au niveau corrigé pour
    l'effectif de la tranche. Une constante négative resserre la borne. n = 1 donne une constante unique pour
    toute la carte. Une tranche de moins de MIN_CASES cases reprend les constantes globales.
    """
    def une(m):
        k = int(m.sum())
        niveau = min(1.0, np.ceil((k + 1) * (1 - ALPHA)) / k)
        return float(np.quantile(q10[m] - y[m], niveau)), float(np.quantile(y[m] - q90[m], niveau))
    globale = une(np.ones(len(y), bool))
    t = tranche(q50, n)
    return np.array([une(t == i) if (t == i).sum() >= MIN_CASES else globale for i in range(n)])


def calibrer(q10, q50, q90, consts):
    """Bornes calibrées q10 − c_bas et q90 + c_haut (constantes de la tranche de q50), dans [0, 1] et autour de q50."""
    c = np.asarray(consts)[tranche(q50, len(consts))]
    return (np.minimum(np.clip(q10 - c[..., 0], 0, 1), q50),
            np.maximum(np.clip(q90 + c[..., 1], 0, 1), q50))


def ecrire(chemin, p, prof, descriptions=None):
    """Tableau (bandes, lignes, colonnes) en fraction de couvert -> GeoTIFF uint16 (× ECHELLE, NODATA hors données)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    a = np.where(np.isfinite(p), np.round(np.clip(p, 0, 1) * ECHELLE), NODATA).astype(np.uint16)
    profil = dict(prof, count=a.shape[0], dtype="uint16", nodata=NODATA, compress="deflate", predictor=2)
    partiel = chemin.with_name(chemin.name + ".part")
    with rasterio.open(partiel, "w", **profil) as d:
        d.write(a)
        for i, texte in enumerate(descriptions or [], start=1):
            d.set_band_description(i, texte)
    partiel.rename(chemin)
