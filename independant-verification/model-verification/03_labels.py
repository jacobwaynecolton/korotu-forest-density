"""Étape 3 : calculer les labels des nouveaux sites, exactement comme ceux du jeu de données.

On appelle make_labels.build_site, la fonction du projet (part des pixels de 0,5 m au-dessus de 2 m et
de 5 m par case de 10 m, grille calée sur celle de Sentinel-2), validée par l'étude GEDI. Seuls ses
dossiers sont redirigés vers les nôtres.
Sorties : SORTIES/labels/<site>_cover10m.tif et model-verification/labels_stats.csv
Usage : python -I independant-verification/model-verification/03_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import CHM, ICI, LABELS, SORTIES, nouveaux_sites  # noqa: E402
import make_labels  # noqa: E402

# build_site lit les CHM dans make_labels.CHM et écrit dans make_labels.LABELS ;
# ROOT ne lui sert qu'à écrire le chemin relatif du label dans les statistiques.
make_labels.CHM, make_labels.LABELS, make_labels.ROOT = CHM, LABELS, SORTIES


def main():
    stats, incomplets = [], []
    for r in nouveaux_sites().itertuples():
        n = len(list((CHM / r.site_id).glob("*_CHM.tif")))
        if n != 25:
            incomplets.append(r.site_id)
            print(f"{r.site_id} : {n} tuiles CHM sur 25, site sauté (relancer l'étape 2)", flush=True)
            continue
        st = make_labels.build_site(r.site_id, r.zone)
        print(f"{r.site_id} : {st['width']}x{st['height']} cases, {st['frac_full']:.1%} complètes, "
              f"couvert moyen {st['mean_cover_gt2m']:.2f}", flush=True)
        stats.append(st)
    pd.DataFrame(stats).to_csv(ICI / "labels_stats.csv", index=False)
    print(f"{len(stats)} labels -> {LABELS}")
    sys.exit(1 if incomplets else 0)


if __name__ == "__main__":
    main()
