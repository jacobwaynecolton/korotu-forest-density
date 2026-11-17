"""Constantes et petites fonctions partagées par les scripts de vérification des modèles aux États-Unis.

Les étapes 4 à 6 (labels, images Sentinel-2, prédiction) sont celles de l'étude ontarienne
(../model-verification/), lancées telles quelles avec MODELE_TRAVAIL et MODELE_SORTIES pointant ici :
d'où les mêmes noms de dossiers (chm/, labels/, stacks/, predictions/) et les mêmes colonnes dans sites.csv.
Le reste est propre à cette étude : calibration sur l'Ontario, choix des blocs, téléchargement des nuages de
points NEON, construction des CHM à 0,5 m (nuages.py) et évaluation.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

PROJET = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJET / "scripts"))
ICI = Path(os.environ.get("MODELE_TRAVAIL", Path(__file__).resolve().parent))
RESULTATS = ICI / "resultats"
SORTIES = Path(os.environ.get("MODELE_SORTIES", "/w/20261/tsalin/MLOps/independant_verification/modele_usa"))
CHM, LABELS, STACKS, PREDICTIONS = (SORTIES / d for d in ("chm", "labels", "stacks", "predictions"))
ONTARIO = PROJET / "independant-verification/model-verification"   # étude de référence
CHM_ONTARIO = Path("/w/20261/tsalin/MLOps/independant_verification/modele/chm")

API = "https://data.neonscience.org/api/v0"
PRODUIT_NUAGES = "DP1.30003.001"   # nuage de points lidar classé, tuiles de 1 km
PRODUIT_CHM = "DP3.30015.001"      # CHM à 1 m : sert seulement à lire le système de coordonnées
JETON = Path.home() / ".neon_token"
NUAGES = SORTIES / "nuages"
CALIBRATION = SORTIES / "calibration"
# Sites et survols retenus (README) : pour chaque site, le survol publié le plus récent de 2021 à 2025 dont
# la densité atteint 25 premiers retours par m² (médiane mesurée le 31 octobre 2026 sur les en-têtes des
# tuiles classées : BART 43, BONA 47, HARV 29, BLAN 43, SERC 27, SCBI 48, OSBS 37).
# Groupe A : forêts proches de celles de l'Ontario ; groupe B : forêts très éloignées de l'entraînement.
VOLS = {"BART": ("A", "2024-08"), "BONA": ("A", "2024-07"), "HARV": ("A", "2024-08"),
        "BLAN": ("B", "2021-08"), "SERC": ("B", "2021-08"), "SCBI": ("B", "2021-08"), "OSBS": ("B", "2021-09")}
NOM_TUILE = re.compile(r"_DP1_\d{6}_\d{7}_classified_point_cloud.*\.laz$")   # pas les lignes de vol ni les fichiers QA
BLOC = 5          # tuiles de 1 km par côté, comme les sites ontariens
TUILE = 1000      # m
MODELES = ("unet_r34", "unet_r34_s1", "unet_r34_s2")
ENSEMBLE = "unet_r34_ens3"


def jeton():
    """Clé d'API NEON, lue dans ~/.neon_token (jamais affichée)."""
    if not JETON.exists():
        sys.exit(f"{JETON} absent : il faut une clé d'API NEON (voir le README).")
    return JETON.read_text().strip()


def api(chemin, avec_jeton=True):
    """Appel à l'API NEON ; renvoie le champ « data » de la réponse."""
    entetes = {"X-API-Token": jeton()} if avec_jeton else {}
    r = requests.get(f"{API}/{chemin}", headers=entetes, timeout=120)
    r.raise_for_status()
    return r.json()["data"]


def coin_tuile(nom):
    """NEON_D01_BART_DP1_314000_4878000_classified_point_cloud.laz -> (314000, 4878000), coin sud-ouest en m."""
    e, n = re.search(r"_(\d{6})_(\d{7})_", nom).groups()
    return int(e), int(n)


def densite_entete(url):
    """Premiers retours par m² d'après l'en-tête d'un fichier LAS/LAZ (lecture des 375 premiers octets)."""
    import struct
    r = requests.get(url, headers={"Range": "bytes=0-374"}, timeout=60, stream=True)
    r.raise_for_status()
    b = r.raw.read(375)
    xmax, xmin, ymax, ymin = struct.unpack_from("<4d", b, 179)
    premiers = struct.unpack_from("<5I", b, 111)[0]            # champ commun à toutes les versions
    if b[25] >= 4 and struct.unpack_from("<15Q", b, 255)[0]:
        premiers = struct.unpack_from("<15Q", b, 255)[0]      # champ étendu de LAS 1.4
    return premiers / ((xmax - xmin) * (ymax - ymin))


def telecharger(url, chemin, verifier, essais=3, entetes=None):
    """Téléchargement par un fichier .part ; `verifier(chemin)` dit si le fichier est complet."""
    if chemin.exists() and verifier(chemin):
        return "déjà là"
    partiel = chemin.with_name(chemin.name + ".part")
    for essai in range(1, essais + 1):
        try:
            with requests.get(url, stream=True, timeout=300, headers=entetes or {}) as r:
                r.raise_for_status()
                with open(partiel, "wb") as f:
                    for morceau in r.iter_content(1 << 22):
                        f.write(morceau)
            if not verifier(partiel):
                raise IOError("fichier incomplet ou somme de contrôle différente")
            partiel.rename(chemin)
            return "ok"
        except Exception as e:
            if essai == essais:
                partiel.unlink(missing_ok=True)
                return f"ÉCHEC {type(e).__name__}: {e}"


def nouveaux_sites():
    return pd.read_csv(ICI / "sites.csv")


def fichiers_site(site_id, controle=False):
    """Label et images d'un site : ceux du projet pour un site de contrôle (test ontarien), les nôtres sinon."""
    if controle:
        return (PROJET / "data/labels" / f"{site_id}_cover10m.tif",
                sorted((PROJET / "data/pairs/stacks").glob(f"{site_id}_k*.tif")))
    return LABELS / f"{site_id}_cover10m.tif", sorted(STACKS.glob(f"{site_id}_k*.tif"))


def sites_controle():
    s = pd.read_csv(PROJET / "data/manifest/sites.csv")
    return s[s.split == "test"].reset_index(drop=True)


def lire(path, bande=1):
    with rasterio.open(path) as s:
        return s.read(bande).astype(np.float32)
