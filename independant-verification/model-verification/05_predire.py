"""Étape 5 (GPU) : prédire les nouveaux sites avec les 3 modèles, puis leur moyenne (l'ensemble).

Même procédure que l'évaluation du projet, avec sa fonction train_unet.predict_site : fenêtres de 256 px
qui se chevauchent de 64 px, moyenne sur toutes les dates et sur 4 retournements, normalisation par la
moyenne et l'écart-type enregistrés dans le config.json de chaque modèle. L'ensemble est la moyenne
des 3 prédictions, comme scripts/ensemble.py.

Les 3 sites de test du projet sont prédits aussi : l'étape 6 vérifie qu'on retrouve les prédictions
enregistrées à la fin de l'entraînement, ce qui contrôle toute la chaîne de prédiction.
Sans GPU, le script tourne sur le processeur (plus lentement, en float32) : utile seulement pour tester.
Reprise possible : une prédiction déjà écrite est sautée.
Sortie : SORTIES/predictions/<modèle>/<site>.tif (couvert 0-1 sur la grille du label, sans masque)
Usage : python -I independant-verification/model-verification/05_predire.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import ENSEMBLE, ICI, MODELES, PREDICTIONS, PROJET, fichiers_site, sites_controle  # noqa: E402
from train_unet import predict_site, to_tensor  # noqa: E402


def charger_modele(nom, dev):
    cfg = json.loads((PROJET / "models" / nom / "config.json").read_text())
    modele = smp.Unet(cfg["encoder"], encoder_weights=None, in_channels=12, classes=1)
    modele.load_state_dict(torch.load(PROJET / "models" / nom / "best.pt", map_location="cpu"))
    return modele.to(dev).eval(), np.array(cfg["mean"]), np.array(cfg["std"])


def lire_images(chemins):
    """(dates, 12 bandes, lignes, colonnes), comme train_unet.load_site."""
    images = []
    for p in chemins:
        with rasterio.open(p) as s:
            images.append(s.read())
    return np.stack(images)


def ecrire(chemin, p, prof):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(chemin, "w", **dict(prof, count=1, dtype="float32", nodata=np.nan)) as d:
        d.write(p.astype(np.float32), 1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("Pas de GPU : prédiction sur le processeur, en float32 (pour un test seulement).", flush=True)
    scenes = pd.read_csv(ICI / "scenes_s2.csv")
    a_predire = [(s, False) for s in scenes.site_id.unique()] + [(s, True) for s in sites_controle().site_id]
    print(f"{len(a_predire) - len(sites_controle())} nouveaux sites avec images + "
          f"{len(sites_controle())} sites de test du projet (contrôle)", flush=True)

    for nom in MODELES:
        restants = [(s, c) for s, c in a_predire if not (PREDICTIONS / nom / f"{s}.tif").exists()]
        if not restants:
            continue
        modele, moyenne, ecart_type = charger_modele(nom, dev)
        for site_id, controle in restants:
            label, images = fichiers_site(site_id, controle)
            with rasterio.open(label) as s:
                prof = s.profile
            p = predict_site(modele, to_tensor(lire_images(images), moyenne, ecart_type, dev))
            ecrire(PREDICTIONS / nom / f"{site_id}.tif", p, prof)
            print(f"{nom} · {site_id} : {len(images)} dates, couvert prédit moyen {np.nanmean(p):.3f}", flush=True)
        del modele
        if dev == "cuda":
            torch.cuda.empty_cache()

    for site_id, _ in a_predire:
        sortie = PREDICTIONS / ENSEMBLE / f"{site_id}.tif"
        if sortie.exists():
            continue
        tableaux = []
        for nom in MODELES:
            with rasterio.open(PREDICTIONS / nom / f"{site_id}.tif") as s:
                tableaux.append(s.read(1)); prof = s.profile
        ecrire(sortie, np.mean(tableaux, axis=0), prof)
    print(f"prédictions -> {PREDICTIONS}")


if __name__ == "__main__":
    main()
