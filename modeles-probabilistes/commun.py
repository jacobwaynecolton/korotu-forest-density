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
