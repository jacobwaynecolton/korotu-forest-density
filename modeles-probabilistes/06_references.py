"""Étape 6 : comparer le modèle final aux cartes de couvert existantes (Meta/WRI CHM v2, SCANFI v2, NTEMS 2022).

Deux comparaisons, chacune sur des sites que nos modèles n'ont pas vus :
1. Les 49 sites du projet, en validation croisée : carte centrale hors pli de la configuration retenue (5 graines,
   étape 5) et modèle L1 actuel (5 graines, hors pli). Cartes de référence préparées par le projet
   (scripts/benchmarks.py, data/benchmarks/).
2. Les 15 nouveaux sites ontariens : modèles finaux (étape 5) et modèle d'origine du projet (3 graines, prédictions
   de independant-verification/model-verification/). Les cartes de référence sont mises sur la grille des labels
   avec les fonctions de scripts/benchmarks.py (lecture à distance, rien de lourd n'est téléchargé).

Chaque produit est aussi donné « recalibré », comme dans le projet (scripts/calibrate_benchmarks.py) : régression
isotone produit -> label, apprise sans voir les sites évalués (sur les 4 autres plis en validation croisée, sur les
49 sites du projet pour les nouveaux sites). On compare ainsi les modèles plutôt que les définitions du couvert.
Mesures à 10 m sur les cases où le label et toutes les cartes sont valides : MAE, RMSE, biais, R².
Sorties : resultats/references_metriques.csv, resultats/references_par_site.csv ; cartes de référence des 15 sites
dans SORTIES/references/.
Usage : python -I modeles-probabilistes/06_references.py   (après 05_cartes.py)
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (GRAINES, ONTARIO, PREDICTIONS, PROJET, RESULTATS, SORTIES, fichiers_projet, lire,  # noqa: E402
                    lire_prediction, sites_projet)
import benchmarks  # noqa: E402  (scripts/ du projet, ajouté au chemin par commun)

PRODUITS = {"meta_chm_v2": "Meta/WRI CHM v2", "scanfi_v2": "SCANFI v2", "ntems_2022": "NTEMS 2022"}
REFERENCES = SORTIES / "references"
ORIGINE = ONTARIO["donnees"] / "predictions" / "unet_r34_ens3"
ECHANTILLON = 40000      # cases tirées par site pour apprendre la recalibration, comme le projet


def preparer_nouveaux(sites):
    """Cartes de référence des nouveaux sites sur la grille de leurs labels (déjà faites : sautées)."""
    benchmarks.OUT = REFERENCES
    tuiles = gpd.read_file(PROJET / "data/raw/meta_chm/tiles.geojson")
    for r in sites.itertuples():
        if all((REFERENCES / f"{r.site_id}_{p}.tif").exists() for p in PRODUITS):
            continue
        with rasterio.open(ONTARIO["donnees"] / "labels" / f"{r.site_id}_cover10m.tif") as s:
            prof, bornes = s.profile, s.bounds
        benchmarks.meta_chm(r.site_id, prof, bornes, tuiles)
        benchmarks.scanfi(r.site_id, prof, int(r.year))
        benchmarks.ntems(r.site_id, prof)
        print(f"  {r.site_id} : cartes de référence prêtes", flush=True)


def site(y, modeles, refs):
    """Cases valides partout : dict de vecteurs (label, modèles, produits bruts)."""
    tout = {"label": y, **modeles, **refs}
    ok = np.logical_and.reduce([np.isfinite(v) for v in tout.values()])
    return {k: v[ok] for k, v in tout.items()}


def echantillon(d, rng):
    i = rng.choice(len(d["label"]), min(ECHANTILLON, len(d["label"])), replace=False)
    return {k: d[k][i] for k in ("label", *PRODUITS)}


def isotone(echantillons, produit):
    x = np.concatenate([e[produit] for e in echantillons])
    y = np.concatenate([e["label"] for e in echantillons])
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)


def mesures(y, p):
    e = p - y
    return dict(n=int(len(y)), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                biais=float(e.mean()), r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def evaluer(jeu, donnees, recalibres, modeles):
    """donnees : site -> cases ; recalibres : site -> {produit: vecteur recalibré}. Lignes globales et par site."""
    sources = [(nom, "—", lambda d, s, k=cle: d[k]) for cle, nom in modeles.items()]
    sources += [(nom, "brut", lambda d, s, k=p: d[k]) for p, nom in PRODUITS.items()]
    sources += [(nom, "recalibré", lambda d, s, k=p: recalibres[s][k]) for p, nom in PRODUITS.items()]
    glob, par_site = [], []
    for nom, version, f in sources:
        y = np.concatenate([d["label"] for d in donnees.values()])
        p = np.concatenate([f(d, s) for s, d in donnees.items()])
        glob.append(dict(jeu=jeu, sites=len(donnees), source=nom, version=version, **mesures(y, p)))
        for s, d in donnees.items():
            m = mesures(d["label"], f(d, s))
            par_site.append(dict(jeu=jeu, site_id=s, source=nom, version=version, mae=m["mae"], biais=m["biais"]))
    return glob, par_site


def main():
    resume = json.loads((RESULTATS / "resume.json").read_text())
    config = resume["configuration_retenue"]
    cartes = SORTIES / "cartes" / config
    rng = np.random.default_rng(0)

    # 1. validation croisée sur les 49 sites du projet
    plis = sites_projet()
    projet, ech, pli_de = {}, {}, {}
    for r in plis.itertuples():
        y = lire(fichiers_projet(r.site_id)[0])
        final = lire(cartes / "validation_croisee" / f"{r.site_id}.tif", 2)
        final = np.where(final == 65535, np.nan, final / 10000)
        l1 = [PREDICTIONS / "l1_r34_imagenet" / f"pli{r.pli}" / f"g{g}" / f"{r.site_id}.tif" for g in GRAINES]
        l1 = np.mean([lire_prediction(c)[0] for c in l1 if c.exists()], axis=0)
        refs = {p: lire(PROJET / "data/benchmarks" / f"{r.site_id}_{p}.tif") for p in PRODUITS}
        projet[r.site_id] = site(y, {"final": final, "origine": l1}, refs)
        ech[r.site_id], pli_de[r.site_id] = echantillon(projet[r.site_id], rng), int(r.pli)
    recal = {s: {} for s in projet}
    for k in sorted(set(pli_de.values())):
        autres = [ech[s] for s in projet if pli_de[s] != k]
        for p in PRODUITS:
            iso = isotone(autres, p)
            for s in projet:
                if pli_de[s] == k:
                    recal[s][p] = iso.predict(projet[s][p])
    modeles = {"final": "Modèle final (carte centrale)", "origine": "Modèle d'origine (perte L1)"}
    g1, s1 = evaluer("validation croisée, 49 sites du projet", projet, recal, modeles)

    # 2. les 15 nouveaux sites ontariens
    nouveaux_sites = pd.read_csv(ONTARIO["sites"])
    preparer_nouveaux(nouveaux_sites)
    isos = {p: isotone(list(ech.values()), p) for p in PRODUITS}
    nouveaux, recal = {}, {}
    for r in nouveaux_sites.itertuples():
        y = lire(ONTARIO["donnees"] / "labels" / f"{r.site_id}_cover10m.tif")
        final = lire(cartes / "externes" / f"{r.site_id}.tif", 2)
        final = np.where(final == 65535, np.nan, final / 10000)
        refs = {p: lire(REFERENCES / f"{r.site_id}_{p}.tif") for p in PRODUITS}
        nouveaux[r.site_id] = site(y, {"final": final, "origine": lire(ORIGINE / f"{r.site_id}.tif")}, refs)
        recal[r.site_id] = {p: isos[p].predict(nouveaux[r.site_id][p]) for p in PRODUITS}
    modeles["origine"] = "Modèle d'origine (perte L1, 3 graines)"
    g2, s2 = evaluer("15 nouveaux sites ontariens", nouveaux, recal, modeles)

    glob, par_site = pd.DataFrame(g1 + g2), pd.DataFrame(s1 + s2)
    glob.to_csv(RESULTATS / "references_metriques.csv", index=False)
    par_site.to_csv(RESULTATS / "references_par_site.csv", index=False)
    pd.set_option("display.width", 200)
    print(glob.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
