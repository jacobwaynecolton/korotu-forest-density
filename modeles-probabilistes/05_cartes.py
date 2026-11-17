"""Étape 5 : écrire les cartes prudente, centrale et optimiste de la configuration retenue (GeoTIFF à 3 bandes).

- Sites externes (15 ontariens, 7 NEON) : mélange des 5 modèles finaux ; constantes conformes finales, par
  tranche de la carte centrale (resultats/resume.json, étape 4).
- 49 sites du projet : mélange des 5 modèles du pli qui ne les a pas vus ; constantes tirées des 4 autres plis,
  comme dans l'évaluation. Ces cartes servent aux comparaisons de l'étape 6.

Bandes : 1 prudente (q10 calibré), 2 centrale (q50), 3 optimiste (q90 calibré) ; couvert × 10 000, entiers 16 bits.
Sorties : SORTIES/cartes/<configuration>/{externes,validation_croisee}/<site>.tif
Usage : python -I modeles-probabilistes/05_cartes.py   (après 04_evaluer.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (GRAINES, PREDICTIONS, RESULTATS, SORTIES, calibrer, ecrire, lire_prediction,  # noqa: E402
                    melange, sites_externes, sites_projet)

BANDES = ("prudente (q10 calibré)", "centrale (q50)", "optimiste (q90 calibré)")


def carte(chemins, consts, sortie):
    """Mélange des graines présentes, calibration, écriture. Rend False si aucune graine n'est disponible."""
    chemins = [c for c in chemins if c.exists()]
    if not chemins:
        return False
    q10, q50, q90 = melange([lire_prediction(c) for c in chemins])
    q10c, q90c = calibrer(q10, q50, q90, consts)
    with rasterio.open(chemins[0]) as s:
        prof = s.profile
    ecrire(sortie, np.stack([q10c, q50, q90c]), prof, BANDES)
    return True


def main():
    resume = json.loads((RESULTATS / "resume.json").read_text())
    config = resume["configuration_retenue"]
    finales = np.array([[t["c_bas"], t["c_haut"]] for t in resume[config]["constantes_par_tranche"]])
    par_pli = {int(k): np.array(v) for k, v in resume[config]["constantes_par_pli"].items()}
    dossier = SORTIES / "cartes" / config
    faites, manques = 0, []
    for e in sites_externes():
        chemins = [PREDICTIONS / config / "final" / f"g{g}" / f"{e['site_id']}.tif" for g in GRAINES]
        if carte(chemins, finales, dossier / "externes" / f"{e['site_id']}.tif"):
            faites += 1
        else:
            manques.append(e["site_id"])
    for r in sites_projet().itertuples():
        chemins = [PREDICTIONS / config / f"pli{r.pli}" / f"g{g}" / f"{r.site_id}.tif" for g in GRAINES]
        if carte(chemins, par_pli[int(r.pli)], dossier / "validation_croisee" / f"{r.site_id}.tif"):
            faites += 1
        else:
            manques.append(r.site_id)
    print(f"{config} : {faites} cartes écrites dans {dossier}" + (f" ; sans prédiction : {manques}" if manques else ""))


if __name__ == "__main__":
    main()
