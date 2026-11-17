"""Constantes et petites fonctions partagées par les scripts GEDI.

Aucune fonction du projet n'est importée : on lit seulement ses fichiers de sortie
(data/manifest/sites.csv, data/labels/, data/raw/fri_chm/) et on refait les calculs ici.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

PROJET = Path(__file__).resolve().parents[2]
# Deux variables d'environnement facultatives redirigent les sorties, pour les tests à petite échelle :
# GEDI_TRAVAIL (fichiers.csv et resultats/, par défaut ce dossier) et GEDI_SORTIES (données lourdes).
ICI = Path(os.environ.get("GEDI_TRAVAIL", Path(__file__).resolve().parent))
RESULTATS = ICI / "resultats"
SORTIES = Path(os.environ.get("GEDI_SORTIES", "/w/20261/tsalin/MLOps/independant_verification/gedi"))

VERSION = "003"
DEBUT_SAISON, FIN_SAISON = (6, 1), (9, 15)
ECART_ANNEES_MAX = 1
PLEINE_PUISSANCE = ("BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011")
EPOQUE_GEDI = pd.Timestamp("2018-01-01", tz="UTC")   # delta_time compte les secondes depuis cette date
RAYON = 12.5                       # m : empreinte GEDI d'environ 25 m de diamètre
SIGMA = RAYON / 2                  # m : écart-type de la pondération gaussienne (variante)
DECALAGES = (-20, -10, 0, 10, 20)  # m : test de position
HAUTEUR_MIN, HAUTEUR_MAX = -2.0, 80.0   # hauteurs CHM valides, comme dans la définition du label
CLASSE_EAU = 80                    # WorldCover


def cle_fichier(nom):
    """GEDI02_B_2020229151418_O09507_02_T04531_02_004_01_V003.h5 -> 2020229151418_O09507_02_T04531.

    La clé est commune aux fichiers L2A et L2B d'un même passage.
    """
    m = re.search(r"GEDI02_[AB]_(\d{13}_O\d{5}_\d{2}_T\d{5})", nom)
    return m.group(1) if m else None


def date_fichier(cle):
    """Début d'acquisition du fichier, lu dans sa clé (année, jour de l'année, heure)."""
    return pd.to_datetime(cle[:13], format="%Y%j%H%M%S", utc=True)


def dans_saison(dates):
    md = dates.dt.month * 100 + dates.dt.day
    return (md >= DEBUT_SAISON[0] * 100 + DEBUT_SAISON[1]) & (md <= FIN_SAISON[0] * 100 + FIN_SAISON[1])


def sites():
    """Les 49 sites : identifiant, split, année et dates du vol lidar, système et emprise du label."""
    s = pd.read_csv(PROJET / "data/manifest/sites.csv")
    lignes = []
    for r in s.itertuples():
        with rasterio.open(PROJET / "data/labels" / f"{r.site_id}_cover10m.tif") as src:
            b, crs = src.bounds, src.crs
        vers_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = vers_lonlat.transform([b.left, b.right, b.right, b.left], [b.bottom, b.bottom, b.top, b.top])
        lignes.append(dict(site_id=r.site_id, split=r.split, annee_vol=int(r.year), zone=int(r.zone),
                           vol_debut=r.flight_start, vol_fin=r.flight_end, epsg=crs.to_epsg(),
                           lon_min=min(xs), lon_max=max(xs), lat_min=min(ys), lat_max=max(ys)))
    return pd.DataFrame(lignes)


def disque(pas):
    """Points d'un disque de rayon RAYON, espacés de `pas` mètres : décalages (dx, dy) et poids gaussiens.

    Chaque point représente la même surface (pas²), donc une moyenne simple sur les points est une
    moyenne pondérée par la surface.
    """
    c = np.arange(-RAYON + pas / 2, RAYON, pas)
    dx, dy = np.meshgrid(c, c)
    dedans = np.hypot(dx, dy) <= RAYON
    dx, dy = dx[dedans], dy[dedans]
    w = np.exp(-(dx ** 2 + dy ** 2) / (2 * SIGMA ** 2))
    return dx, dy, w / w.sum()
