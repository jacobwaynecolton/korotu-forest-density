"""Étape 6 : évaluer les modèles sur les nouveaux sites et appliquer les critères fixés dans le README.

Référence : les mêmes modèles sur les 3 sites de test du projet, prédits par notre chaîne (étape 5) et
notés par ce script, pour comparer à l'identique. Mesures sur les cases où le label et toutes les
prédictions existent, à 10 m et à 30 m (moyenne de 3 × 3 cases, comme le projet) : MAE, RMSE, biais
(prédiction - label), r, R², exactitude et F1 macro sur 4 classes (seuils 10 / 40 / 70 %, ceux du projet).
Les sites sont regroupés en comptant chaque case une fois, comme scripts/evaluate.py.

Contrôles, sans lesquels la comparaison n'a pas de sens :
  - nos prédictions des sites de test égalent celles enregistrées à la fin de l'entraînement ;
  - notre calcul redonne la MAE du projet (data/manifest/eval_test.json) sur ses propres prédictions.

Sorties : model-verification/resultats/ (criteres.json, controle.json, tableaux .csv, figures .png).
Usage : python -I independant-verification/model-verification/06_evaluer.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (ENSEMBLE, MODELES, PREDICTIONS, PROJET, RESULTATS, fichiers_site, lire,  # noqa: E402
                    nouveaux_sites, sites_controle, sites_projet)

SOURCES = MODELES + (ENSEMBLE,)
COUPES = (0.10, 0.40, 0.70)
# Critères de réussite (README), appliqués à l'ensemble, à 10 m, sur les nouveaux sites regroupés
MAE_RELATIVE_MAX = 1.25                    # MAE ≤ 1,25 × celle des sites de test
BIAIS_MAX = 0.05
ECART_TRANCHE_MAX, PART_TRANCHE_MIN = 0.05, 0.01
MAE_SITE_MAX = 0.25
ECART_MODELES_MAX = 0.01
TOLERANCE_ENSEMBLE = 0.002
# Contrôles
ECART_PRED_MAX, ECART_PRED_MOYEN, ECART_MAE_PROJET = 0.02, 0.002, 0.001


def blocs3(a):
    """Moyenne des blocs de 3 × 3 cases (30 m) ; NaN si une case du bloc manque."""
    H, W = (n // 3 * 3 for n in a.shape)
    b = a[:H, :W].reshape(H // 3, 3, W // 3, 3)
    return np.where(np.isfinite(b).all(axis=(1, 3)), b.mean(axis=(1, 3)), np.nan)


def cases(y, p, resolution):
    """Label et prédictions sur les cases communes : (y, {source: p})."""
    f = blocs3 if resolution == "30m" else (lambda a: a)
    yy, pp = f(y), {s: f(v) for s, v in p.items()}
    ok = np.isfinite(yy) & np.all([np.isfinite(v) for v in pp.values()], axis=0)
    return yy[ok], {s: v[ok] for s, v in pp.items()}


def f1_macro(yc, pc):
    f = []
    for k in range(len(COUPES) + 1):
        vp, fp, fn = np.sum((yc == k) & (pc == k)), np.sum((yc != k) & (pc == k)), np.sum((yc == k) & (pc != k))
        f.append(0.0 if vp == 0 else 2 * vp / (2 * vp + fp + fn))
    return float(np.mean(f))


def metriques(y, p):
    e = p - y
    yc, pc = np.digitize(y, COUPES), np.digitize(p, COUPES)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan
        r2 = float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(n=len(y), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), biais=float(e.mean()),
                r=r, r2=r2, acc4=float((yc == pc).mean()), f1_macro=f1_macro(yc, pc))


def rassembler(sites, controle):
    """{résolution: {site: (y, {source: p})}} pour une liste de sites."""
    donnees = {"10m": {}, "30m": {}}
    for s in sites:
        label, _ = fichiers_site(s, controle)
        y = lire(label)
        p = {src: np.clip(lire(PREDICTIONS / src / f"{s}.tif"), 0, 1) for src in SOURCES}
        for res in donnees:
            donnees[res][s] = cases(y, p, res)
    return donnees


def regrouper(par_site, sites=None):
    """Concatène les cases de plusieurs sites : (y, {source: p})."""
    garder = [v for s, v in par_site.items() if sites is None or s in sites]
    return (np.concatenate([y for y, _ in garder]),
            {src: np.concatenate([p[src] for _, p in garder]) for src in SOURCES})


def tableau_global(groupes):
    lignes = []
    for nom, donnees in groupes.items():
        for res, par_site in donnees.items():
            y, p = regrouper(par_site)
            lignes += [dict(groupe=nom, resolution=res, source=src, **metriques(y, p[src])) for src in SOURCES]
    return pd.DataFrame(lignes)


def tableau_sites(groupes, infos):
    lignes = []
    for nom, donnees in groupes.items():
        for s, (y, p) in donnees["10m"].items():
            for src in SOURCES:
                lignes.append(dict(groupe=nom, site_id=s, source=src, label_moyen=float(y.mean()),
                                   **metriques(y, p[src])))
    return pd.DataFrame(lignes).merge(infos, on="site_id", how="left")


def tableau_tranches(groupes):
    lignes = []
    for nom, donnees in groupes.items():
        y, p = regrouper(donnees["10m"])
        tranche = np.minimum(np.floor(y * 10), 9) / 10
        for lo in np.unique(tranche):
            m = tranche == lo
            for src in SOURCES:
                e = p[src][m] - y[m]
                lignes.append(dict(groupe=nom, tranche=round(float(lo), 1), source=src, n=int(m.sum()),
                                   part=float(m.mean()), mae=float(np.abs(e).mean()), biais=float(e.mean())))
    return pd.DataFrame(lignes)


def tableau_groupes(par_site, infos):
    """Ensemble, nouveaux sites, 10 m : par année du vol, zone UTM et position par rapport à l'entraînement."""
    lignes = []
    for colonne in ("year", "zone", "position"):
        for valeur, g in infos[infos.site_id.isin(list(par_site))].groupby(colonne):
            y, p = regrouper(par_site, set(g.site_id))
            lignes.append(dict(groupement=colonne, groupe=str(valeur), n_sites=len(g), **metriques(y, p[ENSEMBLE])))
    return pd.DataFrame(lignes)


def controle(sites_test):
    """Nos prédictions des sites de test contre celles du projet, et notre MAE contre la sienne."""
    lignes = []
    for s in sites_test:
        for src in SOURCES:
            stockee = PROJET / "data/predictions" / src / f"{s}.tif"
            if not stockee.exists():
                lignes.append(dict(site_id=s, source=src, n=0, ecart_max=np.nan, ecart_moyen=np.nan))
                continue
            a, b = lire(stockee), lire(PREDICTIONS / src / f"{s}.tif")
            ok = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a[ok] - b[ok])
            lignes.append(dict(site_id=s, source=src, n=int(ok.sum()), ecart_max=float(d.max()),
                               ecart_moyen=float(d.mean())))
    preds = pd.DataFrame(lignes)

    ys, ps = [], []
    for s in sites_test:
        y = lire(fichiers_site(s, True)[0])
        p = np.clip(lire(PROJET / "data/predictions" / ENSEMBLE / f"{s}.tif"), 0, 1)
        yy, pp = cases(y, {ENSEMBLE: p}, "10m")
        ys.append(yy); ps.append(pp[ENSEMBLE])
    notre = metriques(np.concatenate(ys), np.concatenate(ps))
    projet = [r for r in json.loads((PROJET / "data/manifest/eval_test.json").read_text())
              if r["source"] == f"model:{ENSEMBLE}" and r["res"] == "10m"]
    mae_projet = projet[0]["mae"] if projet else np.nan

    ok_preds = bool(preds.n.gt(0).all() and preds.ecart_max.max() <= ECART_PRED_MAX
                    and preds.ecart_moyen.max() <= ECART_PRED_MOYEN)
    ok_mae = bool(abs(notre["mae"] - mae_projet) <= ECART_MAE_PROJET)
    return preds, dict(predictions_identiques=ok_preds, ecart_max=float(preds.ecart_max.max()),
                       ecart_moyen_max=float(preds.ecart_moyen.max()), mae_notre_code=notre["mae"],
                       n_notre_code=notre["n"], mae_projet=mae_projet, mae_identique=ok_mae, ok=ok_preds and ok_mae)


def figures(groupes, st, tr, infos, projet):
    ens = st[st.source == ENSEMBLE]
    nv, te = ens[ens.groupe == "nouveaux"].sort_values("mae"), ens[ens.groupe == "test"].sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * (len(nv) + len(te)))))
    ax.barh(nv.site_id, nv.mae, color="tab:blue", label="nouveaux sites")
    ax.barh(te.site_id, te.mae, color="tab:gray", label="sites de test du projet")
    ax.axvline(MAE_SITE_MAX, color="k", ls="--", lw=1, label=f"critère {MAE_SITE_MAX}")
    ax.set(xlabel="MAE de l'ensemble (couvert 0-1)", title="Erreur par site, 10 m")
    ax.legend(loc="lower right"); ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(RESULTATS / "sites.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (nom, donnees) in zip(axes, groupes.items()):
        y, p = regrouper(donnees["10m"])
        hb = ax.hexbin(y, p[ENSEMBLE], gridsize=40, mincnt=1, bins="log", cmap="viridis", extent=(0, 1, 0, 1))
        ax.plot((0, 1), (0, 1), "k--", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="label (lidar)", ylabel="prédiction de l'ensemble",
               title=f"Sites {nom}, 10 m")
        plt.colorbar(hb, ax=ax, label="nombre de cases")
    fig.tight_layout(); fig.savefig(RESULTATS / "nuages.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nom, couleur in (("nouveaux", "tab:blue"), ("test", "tab:gray")):
        t = tr[(tr.groupe == nom) & (tr.source == ENSEMBLE)]
        ax.plot(t.tranche + 0.05, t.mae, "o-", color=couleur, label=f"sites {nom}")
    ax.set(xlabel="label (tranches de 0,1)", ylabel="MAE de l'ensemble", title="Erreur par tranche de couvert, 10 m")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTATS / "tranches.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for split, style in (("train", dict(c="lightgray", s=25)), ("val", dict(c="k", marker="s", s=25)),
                         ("test", dict(c="k", marker="^", s=35))):
        q = projet[projet.split == split]
        ax.scatter(q.lon, q.lat, label=f"projet : {split}", **style)
    m = infos.merge(nv[["site_id", "mae"]], on="site_id")
    sc = ax.scatter(m.lon, m.lat, c=m.mae, cmap="viridis", s=70, edgecolor="k", label="nouveaux sites")
    plt.colorbar(sc, ax=ax, label="MAE de l'ensemble")
    ax.set(xlabel="longitude", ylabel="latitude", title="Sites du projet et nouveaux sites")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(RESULTATS / "carte.png", dpi=130)
    plt.close(fig)


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    projet = sites_projet()
    entrainement = projet[projet.split == "train"]
    infos = nouveaux_sites()[["site_id", "year", "zone", "lon", "lat"]].copy()
    infos["distance_entrainement_deg"] = [np.min(np.hypot(entrainement.lon - r.lon, entrainement.lat - r.lat))
                                          for r in infos.itertuples()]
    infos["position"] = np.where(infos.lat > entrainement.lat.max(), "au nord de l'entraînement",
                                 "dans l'étendue de l'entraînement")
    test = sites_controle()
    infos = pd.concat([infos, test[["site_id", "year", "zone", "lon", "lat"]]], ignore_index=True)

    nouveaux = [s for s in nouveaux_sites().site_id if (PREDICTIONS / ENSEMBLE / f"{s}.tif").exists()]
    exclus = sorted(set(nouveaux_sites().site_id) - set(nouveaux))
    if not nouveaux:
        sys.exit("Aucun nouveau site prédit : vérifier les étapes 3 à 5.")
    groupes = {"nouveaux": rassembler(nouveaux, False), "test": rassembler(test.site_id, True)}

    preds, ctrl = controle(test.site_id)
    sg, st, tr = tableau_global(groupes), tableau_sites(groupes, infos), tableau_tranches(groupes)
    gr = tableau_groupes(groupes["nouveaux"]["10m"], infos)
    for nom, df in (("metriques_globales", sg), ("metriques_sites", st), ("metriques_tranches", tr),
                    ("metriques_groupes", gr), ("controle_predictions", preds)):
        df.to_csv(RESULTATS / f"{nom}.csv", index=False)
    (RESULTATS / "controle.json").write_text(json.dumps(ctrl, indent=1, ensure_ascii=False))
    figures(groupes, st, tr, infos, projet)

    g10 = sg[sg.resolution == "10m"].set_index(["groupe", "source"])
    nv, te = g10.loc[("nouveaux", ENSEMBLE)], g10.loc[("test", ENSEMBLE)]
    t_ens = tr[tr.source == ENSEMBLE].pivot(index="tranche", columns="groupe", values=["mae", "part"])
    a_tester = t_ens[t_ens[("part", "nouveaux")] >= PART_TRANCHE_MIN]
    tranches_ko = [float(lo) for lo, r in a_tester.iterrows()
                   if not r[("mae", "nouveaux")] <= r[("mae", "test")] + ECART_TRANCHE_MAX]
    sites_ens = st[(st.groupe == "nouveaux") & (st.source == ENSEMBLE)]
    sites_ko = sites_ens.loc[sites_ens.mae > MAE_SITE_MAX, "site_id"].tolist()
    mae_modeles = {m: float(g10.loc[("nouveaux", m), "mae"]) for m in MODELES}
    ecart_modeles = max(mae_modeles.values()) - min(mae_modeles.values())

    criteres = [
        (f"MAE ≤ {MAE_RELATIVE_MAX} × celle des sites de test", {"nouveaux": round(nv.mae, 4), "test": round(te.mae, 4)},
         nv.mae <= MAE_RELATIVE_MAX * te.mae),
        (f"biais entre -{BIAIS_MAX} et +{BIAIS_MAX}", round(nv.biais, 4), abs(nv.biais) <= BIAIS_MAX),
        (f"par tranche de couvert : MAE ≤ celle des sites de test + {ECART_TRANCHE_MAX}", tranches_ko, not tranches_ko),
        (f"aucun site avec une MAE > {MAE_SITE_MAX}", sites_ko, not sites_ko),
        (f"écart de MAE entre les 3 modèles ≤ {ECART_MODELES_MAX}", {k: round(v, 4) for k, v in mae_modeles.items()},
         ecart_modeles <= ECART_MODELES_MAX),
        (f"ensemble au moins aussi bon que le meilleur modèle (à {TOLERANCE_ENSEMBLE} près)", round(nv.mae, 4),
         nv.mae <= min(mae_modeles.values()) + TOLERANCE_ENSEMBLE),
    ]
    sortie = dict(n_sites=len(nouveaux), sites_exclus=exclus, n_cases=int(nv.n), controle_ok=ctrl["ok"],
                  criteres=[dict(nom=n, valeur=v, ok=bool(ok)) for n, v, ok in criteres],
                  reussis=sum(bool(ok) for *_, ok in criteres), total=len(criteres))
    (RESULTATS / "criteres.json").write_text(json.dumps(sortie, indent=1, ensure_ascii=False, default=float))

    pd.set_option("display.width", 200)
    print(f"{len(nouveaux)} nouveaux sites évalués" + (f", exclus faute d'images : {exclus}" if exclus else ""))
    print(f"contrôle : prédictions des sites de test {'identiques' if ctrl['predictions_identiques'] else 'DIFFÉRENTES'} "
          f"(écart max {ctrl['ecart_max']:.4f}) ; MAE du projet {ctrl['mae_projet']:.4f}, "
          f"notre code {ctrl['mae_notre_code']:.4f} -> {'OK' if ctrl['ok'] else 'ÉCHEC'}\n")
    print(sg.round(3).to_string(index=False), "\n")
    print("ensemble, par site (10 m) :")
    print(st[st.source == ENSEMBLE][["groupe", "site_id", "year", "zone", "n", "label_moyen", "mae", "biais", "r2"]]
          .sort_values(["groupe", "mae"]).round(3).to_string(index=False), "\n")
    print("ensemble, par tranche de label (10 m) :")
    print(t_ens.round(3).to_string(), "\n")
    print(gr.round(3).to_string(index=False), "\n")
    if not ctrl["ok"]:
        print("ATTENTION : contrôle en échec, les critères ci-dessous ne sont pas interprétables.\n")
    for nom, valeur, ok in criteres:
        print(f"{'PASS' if ok else 'FAIL'}  {nom}  ->  {valeur}")
    print(f"\n{sortie['reussis']}/{sortie['total']} critères réussis | tableaux et figures dans {RESULTATS}")


if __name__ == "__main__":
    main()
