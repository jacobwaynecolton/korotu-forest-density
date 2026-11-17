"""Étape 1 : répartir les 49 sites du projet en 5 plis de validation croisée, par blocs de longitude.

Les sites sont triés d'ouest en est et coupés en 5 groupes contigus de 9 ou 10 sites. Chaque pli teste donc
une région que le modèle n'a pas vue, ce qui est plus exigeant qu'un tirage au hasard des sites. Les sites de
validation et de test d'origine du projet deviennent des sites comme les autres.
Sortie : modeles-probabilistes/plis.csv
Usage : python -I modeles-probabilistes/01_plis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import ICI, N_PLIS, PROJET  # noqa: E402


def main():
    s = pd.read_csv(PROJET / "data/manifest/sites.csv").sort_values("lon").reset_index(drop=True)
    s["pli"] = 0
    for pli, idx in enumerate(np.array_split(np.arange(len(s)), N_PLIS)):
        s.loc[idx, "pli"] = pli
    s = s[["site_id", "pli", "split", "lon", "lat", "year", "zone"]].rename(columns={"split": "split_origine"})
    s.to_csv(ICI / "plis.csv", index=False)
    print(s.groupby("pli").agg(sites=("site_id", "size"), lon_min=("lon", "min"), lon_max=("lon", "max"),
                               lat_min=("lat", "min"), lat_max=("lat", "max")).round(2).to_string())
    print(f"-> {ICI / 'plis.csv'}")


if __name__ == "__main__":
    main()
