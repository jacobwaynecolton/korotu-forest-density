"""Étape 2 : télécharger les CHM à 0,5 m des nouveaux sites (25 tuiles de 1 km par site).

Même fonction que le projet (download_chm.fetch) : un fichier .part d'abord, et une tuile déjà
complète est sautée, donc le script peut être relancé après une coupure.
Sortie : SORTIES/chm/<site>/<tuile>_CHM.tif
Usage : python -I independant-verification/model-verification/02_telecharger_chm.py [nb_téléchargements]
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import CHM, ICI  # noqa: E402
from download_chm import fetch  # noqa: E402


def main(nb=6):
    tuiles = pd.read_csv(ICI / "tuiles.csv")
    taches = []
    for r in tuiles.itertuples():
        (CHM / r.site_id).mkdir(parents=True, exist_ok=True)
        taches.append((r.Download_CHM, CHM / r.site_id / f"{r.Tilename}_CHM.tif"))
    etats = {}
    with ThreadPoolExecutor(nb) as ex:
        futurs = {ex.submit(fetch, url, chemin): chemin for url, chemin in taches}
        for i, fut in enumerate(as_completed(futurs), 1):
            etats[futurs[fut]] = etat = fut.result()
            if etat.startswith("FAIL") or i % 25 == 0 or i == len(taches):
                print(f"[{i}/{len(taches)}] {futurs[fut].name} : {etat}", flush=True)
    echecs = [p for p, e in etats.items() if e.startswith("FAIL")]
    print(f"{len(taches) - len(echecs)}/{len(taches)} tuiles complètes, {len(echecs)} en échec")
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
