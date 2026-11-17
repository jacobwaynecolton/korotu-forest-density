"""Étape 4 : images Sentinel-2 des nouveaux sites, avec les règles et les fonctions du projet.

Comme make_stacks.py : scènes L2A de la même zone UTM à ±45 jours du milieu du vol (entre le 1er mai
et le 15 octobre), fenêtre du site entièrement dans la tuile, moins de 1 % de nuages, ombres ou neige
d'après la SCL, les plus proches du vol d'abord, espacées d'au moins 5 jours, 4 au plus. Écriture avec
make_stacks.write_stack (ordre des bandes, réflectance, contrôle du décalage BOA du projet).

Un site sans aucune image exploitable est signalé et sera exclu de l'évaluation (règle fixée d'avance).
Reprise possible : un site déjà présent dans scenes_s2.csv est sauté.
Sorties : SORTIES/stacks/<site>_k<k>.tif et model-verification/scenes_s2.csv
Usage : python -I independant-verification/model-verification/04_images_s2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import ICI, K_DATES, STACKS, fichiers_site, nouveaux_sites  # noqa: E402
from make_pairs import BAD_SCL, STAC, read_window  # noqa: E402
from make_stacks import candidates, write_stack  # noqa: E402


def images_site(client, site):
    """Écrit jusqu'à K_DATES images pour un site ; renvoie les lignes de scenes_s2.csv."""
    label, _ = fichiers_site(site.site_id)
    with rasterio.open(label) as s:
        prof, bounds, shape, crs = s.profile, s.bounds, (s.height, s.width), s.crs
    items, milieu = candidates(client, site, crs, bounds)
    gardees, lignes = [], []
    for it in items:
        if len(gardees) == K_DATES:
            break
        if any(abs((it.datetime - g.datetime).days) < 5 for g in gardees):
            continue
        scl = read_window(it.assets["scl"].href, bounds, shape, Resampling.nearest)
        if np.isin(scl, list(BAD_SCL)).mean() >= 0.01:
            continue
        k = len(gardees)
        chemin = STACKS / f"{site.site_id}_k{k}.tif"
        partiel = chemin.with_suffix(".tif.part")
        write_stack(it, partiel, bounds, prof, shape)
        partiel.rename(chemin)
        gardees.append(it)
        lignes.append(dict(site_id=site.site_id, k=k, scene=it.id, date=it.datetime.date().isoformat(),
                           days_from_flight=(it.datetime - milieu).days))
    return lignes


def main():
    STACKS.mkdir(parents=True, exist_ok=True)
    fichier = ICI / "scenes_s2.csv"
    faites = pd.read_csv(fichier) if fichier.exists() else pd.DataFrame(columns=["site_id", "k"])
    lignes, sans_image, echecs = faites.to_dict("records"), [], []
    client = Client.open(STAC)
    for site in nouveaux_sites().itertuples():
        if site.site_id in set(faites.site_id) or not fichiers_site(site.site_id)[0].exists():
            continue
        try:
            nouvelles = images_site(client, site)
        except Exception as e:   # réseau, catalogue : on continue et on signale à la fin
            echecs.append(site.site_id)
            print(f"ÉCHEC {site.site_id} : {type(e).__name__}: {e}", flush=True)
            continue
        if not nouvelles:
            sans_image.append(site.site_id)
        lignes += nouvelles
        ecarts = ", ".join(f"{n['days_from_flight']:+d} j" for n in nouvelles)
        print(f"{site.site_id} : {len(nouvelles)} images ({ecarts})", flush=True)
        pd.DataFrame(lignes).to_csv(fichier, index=False)
    print(f"{len(lignes)} images au total -> {STACKS}")
    if sans_image:
        print(f"sans image exploitable (exclus de l'évaluation) : {sans_image}")
    if echecs:
        print(f"{len(echecs)} sites en échec : relancer l'étape, seuls ceux-là seront refaits.")
        sys.exit(1)


if __name__ == "__main__":
    main()
