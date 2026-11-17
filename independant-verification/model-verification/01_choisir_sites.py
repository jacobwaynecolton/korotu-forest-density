"""Étape 1 : choisir les nouveaux sites lidar, jamais vus par les modèles.

Mêmes règles d'éligibilité que le projet (fonctions de scripts/select_sites.py) : bloc de 5 × 5 tuiles
de 1 km toutes munies d'un CHM, une zone UTM, une année ≥ 2019, vols dans la fenêtre feuillée
(1er juin – 15 septembre) et étalés sur 60 jours au plus. En plus :
  - au moins 0,5° de tout site du projet (entraînement, validation, test, et sites abandonnés) ;
  - au moins 0,5° entre nouveaux sites.
Les blocs sont tirés dans un ordre aléatoire (graine fixe) : l'échantillon représente les blocs lidar
disponibles, au lieu de privilégier les bords de la province comme un tirage « le plus loin possible ».

L'index des tuiles (334 Mo) est téléchargé une fois dans le dossier des sorties.
La sélection n'est faite qu'une fois : si sites.csv existe, le script s'arrête (--refaire pour l'écraser).
Sorties : model-verification/sites.csv et model-verification/tuiles.csv (tuiles de chaque site et leurs URL).
Usage : python -I independant-verification/model-verification/01_choisir_sites.py [--refaire]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import ECART_MIN_DEG, GRAINE, ICI, INDEX, N_SITES, URL_INDEX, sites_projet  # noqa: E402
import select_sites  # noqa: E402  (module du projet, rendu importable par commun)


def telecharger_index():
    if INDEX.exists():
        return
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    partiel = INDEX.with_suffix(".gpkg.part")
    print("téléchargement de l'index des tuiles FRI…", flush=True)
    with requests.get(URL_INDEX, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(partiel, "wb") as f:
            for morceau in r.iter_content(1 << 20):
                f.write(morceau)
    partiel.rename(INDEX)


def distance(b, points):
    return np.min(np.hypot(b.lon - points[:, 0], b.lat - points[:, 1])) if len(points) else np.inf


def main(refaire=False):
    if (ICI / "sites.csv").exists() and not refaire:
        sys.exit(f"{ICI / 'sites.csv'} existe déjà : la sélection est faite (--refaire pour la recommencer).")
    telecharger_index()
    select_sites.INDEX = INDEX   # le projet lit l'index dans data/raw/fri_tile_index/, absent ici
    tuiles = select_sites.parse_tiles()
    blocs = select_sites.candidate_blocks(tuiles)
    projet = sites_projet()
    existants = projet[["lon", "lat"]].values
    print(f"{len(blocs):,} blocs éligibles ; {len(projet)} sites du projet à éviter")

    lignes, morceaux, retenus, refus = [], [], [], 0
    for i in np.random.default_rng(GRAINE).permutation(len(blocs)):
        if len(lignes) == N_SITES:
            break
        b = blocs.loc[i]
        if distance(b, existants) < ECART_MIN_DEG or distance(b, np.array(retenus)) < ECART_MIN_DEG:
            continue
        t = select_sites.block_tiles(tuiles, b)
        site_id = f"Z{int(b.zone)}_E{int(b.be):04d}_N{int(b.bn):04d}_{int(b.year)}"
        try:
            dates = select_sites.check_dates(t)
        except Exception as e:   # en-tête LAZ illisible ou réseau : bloc ignoré, comme dans le projet
            print(f"ignoré {site_id} : {type(e).__name__}: {e}", flush=True)
            continue
        if dates is None:
            refus += 1
            continue
        retenus.append((b.lon, b.lat))
        lignes.append(dict(site_id=site_id, zone=int(b.zone), year=int(b.year), e0_m=int(b.be) * 1000,
                           n0_m=int(b.bn) * 1000, lon=b.lon, lat=b.lat, flight_start=dates[0].isoformat(),
                           flight_end=dates[1].isoformat(), split="nouveau"))
        morceaux.append(t.assign(site_id=site_id)[["site_id", "Tilename", "Download_CHM", "Download_LAZ"]])
        print(f"retenu {site_id}  lon {b.lon:.2f}  lat {b.lat:.2f}  vol {dates[0]:%Y-%m-%d}..{dates[1]:%Y-%m-%d}",
              flush=True)

    if len(lignes) < N_SITES:
        print(f"Attention : seulement {len(lignes)} sites trouvés sur {N_SITES}.")
    sites = pd.DataFrame(lignes)
    sites.to_csv(ICI / "sites.csv", index=False)
    pd.concat(morceaux, ignore_index=True).to_csv(ICI / "tuiles.csv", index=False)
    print(f"\n{len(sites)} sites retenus ({refus} blocs refusés pour les dates de vol) -> {ICI / 'sites.csv'}")
    print(sites.groupby(["zone", "year"]).size().unstack(fill_value=0))
    print(f"latitude des nouveaux sites : {sites.lat.min():.2f} à {sites.lat.max():.2f} ; "
          f"sites d'entraînement : {projet[projet.split == 'train'].lat.min():.2f} à "
          f"{projet[projet.split == 'train'].lat.max():.2f}")


if __name__ == "__main__":
    main("--refaire" in sys.argv[1:])
