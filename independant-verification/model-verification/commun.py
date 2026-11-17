"""Constantes et petites fonctions partagées par les scripts de vérification des modèles.

On réutilise le code du projet là où il définit les données ou le modèle testés : règles
d'éligibilité des sites (select_sites.py), téléchargement des CHM (download_chm.py), calcul des
labels (make_labels.py, validé par l'étude GEDI), images Sentinel-2 (make_stacks.py) et prédiction
(train_unet.predict_site). Les fichiers produits vont dans nos propres dossiers, jamais dans data/
du projet, et les métriques sont recalculées ici, sans evaluate.py.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

PROJET = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJET / "scripts"))   # modules du projet : select_sites, make_labels, train_unet…
# Deux variables d'environnement facultatives redirigent les sorties, pour les tests à petite échelle :
# MODELE_TRAVAIL (sites.csv, resultats/…, par défaut ce dossier) et MODELE_SORTIES (données lourdes).
ICI = Path(os.environ.get("MODELE_TRAVAIL", Path(__file__).resolve().parent))
RESULTATS = ICI / "resultats"
SORTIES = Path(os.environ.get("MODELE_SORTIES", "/w/20261/tsalin/MLOps/independant_verification/modele"))
CHM, LABELS, STACKS, PREDICTIONS = (SORTIES / d for d in ("chm", "labels", "stacks", "predictions"))
INDEX = SORTIES / "FRI_Leaf_On_Tile_Index.gpkg"
URL_INDEX = ("https://download.fri.mnrf.gov.on.ca/api/api/Download/tile-index/"
             "FRI_Leaf_On_Tile_Index_GeoPackage/FRI_Leaf_On_Tile_Index_GeoPackage.gpkg")

N_SITES = 15
ECART_MIN_DEG = 0.5    # séparation utilisée par le projet entre sites tenus à part et sites d'entraînement
GRAINE = 3110          # tirage du 31 octobre
K_DATES = 4            # images Sentinel-2 par site au plus, comme make_stacks.py
MODELES = ("unet_r34", "unet_r34_s1", "unet_r34_s2")
ENSEMBLE = "unet_r34_ens3"   # moyenne des 3 modèles, comme scripts/ensemble.py
CSRS_UTM_EPSG = {15: 3159, 16: 3160, 17: 2958, 18: 2959}   # NAD83(CSRS) / UTM, système des noms de tuiles


def centre_site(site_id):
    """Z17_E0600_N5175_2020 -> (lon, lat) du centre du bloc de 5 × 5 km."""
    z, e, n = (int(v) for v in re.match(r"Z(\d+)_E(\d+)_N(\d+)_", site_id).groups())
    vers_lonlat = Transformer.from_crs(f"EPSG:{CSRS_UTM_EPSG[z]}", "EPSG:4326", always_xy=True)
    return vers_lonlat.transform((e + 2.5) * 1000, (n + 2.5) * 1000)


def sites_projet():
    """Tous les sites du projet, abandonnés compris (un fichier tiles_<site>.csv chacun), avec leur split."""
    manifeste = PROJET / "data/manifest"
    split = pd.read_csv(manifeste / "sites.csv").set_index("site_id").split
    ids = sorted(p.stem.removeprefix("tiles_") for p in manifeste.glob("tiles_*.csv"))
    lonlat = [centre_site(s) for s in ids]
    return pd.DataFrame(dict(site_id=ids, split=[split.get(s, "abandonné") for s in ids],
                             lon=[p[0] for p in lonlat], lat=[p[1] for p in lonlat]))


def nouveaux_sites():
    return pd.read_csv(ICI / "sites.csv")


def sites_controle():
    """Sites de test du projet : prédits aussi, pour contrôler notre chaîne et servir de référence."""
    s = pd.read_csv(PROJET / "data/manifest/sites.csv")
    return s[s.split == "test"].reset_index(drop=True)


def fichiers_site(site_id, controle=False):
    """Label et images Sentinel-2 d'un site : ceux du projet pour un site de contrôle, les nôtres sinon."""
    if controle:
        return (PROJET / "data/labels" / f"{site_id}_cover10m.tif",
                sorted((PROJET / "data/pairs/stacks").glob(f"{site_id}_k*.tif")))
    return LABELS / f"{site_id}_cover10m.tif", sorted(STACKS.glob(f"{site_id}_k*.tif"))


def lire(path, bande=1):
    with rasterio.open(path) as s:
        return s.read(bande).astype(np.float32)
