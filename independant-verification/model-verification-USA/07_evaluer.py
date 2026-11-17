"""Étape 7 : décrire ce que font les modèles sur les sites NEON (étude descriptive, sans critère de réussite).

Mesures identiques à celles de l'étude ontarienne (../model-verification/06_evaluer.py) : sur les cases où
le label et toutes les prédictions existent, à 10 m et à 30 m (moyenne de 3 × 3 cases), MAE, RMSE, biais
(prédiction - label), r, R², exactitude et F1 macro sur 4 classes (seuils 10 / 40 / 70 %).
  - Groupe A : forêts proches de celles de l'Ontario (BART, BONA, HARV).
  - Groupe B : forêts très éloignées de l'entraînement (BLAN, SERC, SCBI, OSBS).
Référence : l'ensemble sur les 15 nouveaux sites ontariens (../model-verification/resultats/).
Contrôle : les sites de test ontariens, prédits de nouveau à l'étape 6, doivent redonner les prédictions
enregistrées par l'entraînement.

Sorties : model-verification-USA/resultats/ (resume.json, controle.json, tableaux .csv, figures .png).
Usage : python -I independant-verification/model-verification-USA/07_evaluer.py
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
from commun import (ENSEMBLE, ICI, MODELES, ONTARIO, PREDICTIONS, PROJET, RESULTATS, fichiers_site, lire,  # noqa: E402
                    nouveaux_sites, sites_controle)

SOURCES = MODELES + (ENSEMBLE,)
COUPES = (0.10, 0.40, 0.70)
ECART_PRED_MAX, ECART_PRED_MOYEN = 0.02, 0.002   # contrôle, comme dans l'étude ontarienne
COULEURS = {"A": "tab:blue", "B": "tab:orange"}


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


def rassembler(sites):
    """{résolution: {site: (y, {source: p})}}."""
    donnees = {"10m": {}, "30m": {}}
    for s in sites:
        y = lire(fichiers_site(s)[0])
        p = {src: np.clip(lire(PREDICTIONS / src / f"{s}.tif"), 0, 1) for src in SOURCES}
        for res in donnees:
            donnees[res][s] = cases(y, p, res)
    return donnees


def regrouper(par_site):
    return (np.concatenate([y for y, _ in par_site.values()]),
            {src: np.concatenate([p[src] for _, p in par_site.values()]) for src in SOURCES})


def controle(sites_test):
    """Prédictions des sites de test ontariens refaites ici, contre celles enregistrées par l'entraînement."""
    lignes = []
    for s in sites_test:
        for src in SOURCES:
            a, b = lire(PROJET / "data/predictions" / src / f"{s}.tif"), lire(PREDICTIONS / src / f"{s}.tif")
            ok = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a[ok] - b[ok])
            lignes.append(dict(site_id=s, source=src, n=int(ok.sum()), ecart_max=float(d.max()),
                               ecart_moyen=float(d.mean())))
    preds = pd.DataFrame(lignes)
    ok = bool(preds.n.gt(0).all() and preds.ecart_max.max() <= ECART_PRED_MAX
              and preds.ecart_moyen.max() <= ECART_PRED_MOYEN)
    return preds, dict(ecart_max=float(preds.ecart_max.max()), ecart_moyen_max=float(preds.ecart_moyen.max()), ok=ok)


def reference_ontario():
    """Ensemble sur les 15 nouveaux sites ontariens : métriques globales à 10 m et MAE par tranche."""
    g = pd.read_csv(ONTARIO / "resultats/metriques_globales.csv")
    t = pd.read_csv(ONTARIO / "resultats/metriques_tranches.csv")
    glob = g.query("groupe == 'nouveaux' and resolution == '10m' and source == @ENSEMBLE").iloc[0]
    return glob, t.query("groupe == 'nouveaux' and source == @ENSEMBLE").set_index("tranche").mae


def figures(groupes, st, tr, tranches_onta, mae_onta):
    ens = st[st.source == ENSEMBLE].sort_values(["groupe", "mae"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(ens))))
    ax.barh(ens.site_id + " (" + ens.etat + ")", ens.mae, color=[COULEURS[g] for g in ens.groupe])
    ax.axvline(mae_onta, color="k", lw=1, label=f"Ontario, nouveaux sites ({mae_onta:.3f})")
    for g, c in COULEURS.items():
        ax.barh([], [], color=c, label=f"groupe {g}")
    ax.set(xlabel="MAE de l'ensemble (couvert 0-1)", title="Erreur par site NEON, 10 m")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); fig.savefig(RESULTATS / "sites.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(groupes), figsize=(5.5 * len(groupes), 4.8), squeeze=False)
    for ax, (g, donnees) in zip(axes[0], groupes.items()):
        y, p = regrouper(donnees["10m"])
        hb = ax.hexbin(y, p[ENSEMBLE], gridsize=40, mincnt=1, bins="log", cmap="viridis", extent=(0, 1, 0, 1))
        ax.plot((0, 1), (0, 1), "k--", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="label (lidar NEON)", ylabel="prédiction de l'ensemble",
               title=f"Groupe {g}, 10 m")
        plt.colorbar(hb, ax=ax, label="nombre de cases")
    fig.tight_layout(); fig.savefig(RESULTATS / "nuages.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for g, c in COULEURS.items():
        t = tr[(tr.groupe == g) & (tr.source == ENSEMBLE)]
        if len(t):
            ax.plot(t.tranche + 0.05, t.mae, "o-", color=c, label=f"groupe {g}")
    ax.plot(tranches_onta.index + 0.05, tranches_onta.values, "o-", color="tab:gray", label="Ontario, nouveaux sites")
    ax.set(xlabel="label (tranches de 0,1)", ylabel="MAE de l'ensemble", title="Erreur par tranche de couvert, 10 m")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTATS / "tranches.png", dpi=130); plt.close(fig)


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    sites = nouveaux_sites()
    sites = sites[[(PREDICTIONS / ENSEMBLE / f"{s}.tif").exists() for s in sites.site_id]]
    exclus = sorted(set(nouveaux_sites().site_id) - set(sites.site_id))
    if sites.empty:
        sys.exit("Aucun site NEON prédit : vérifier les étapes 3 à 6.")
    groupes = {g: rassembler(sites[sites.groupe == g].site_id) for g in ("A", "B") if (sites.groupe == g).any()}
    preds, ctrl = controle(sites_controle().site_id)
    onta, tranches_onta = reference_ontario()

    lignes_g, lignes_s, lignes_t = [], [], []
    for g, donnees in groupes.items():
        for res, par_site in donnees.items():
            y, p = regrouper(par_site)
            lignes_g += [dict(groupe=g, resolution=res, source=src, **metriques(y, p[src])) for src in SOURCES]
        for s, (y, p) in donnees["10m"].items():
            lignes_s += [dict(groupe=g, site_id=s, source=src, label_moyen=float(y.mean()), **metriques(y, p[src]))
                         for src in SOURCES]
        y, p = regrouper(donnees["10m"])
        tranche = np.minimum(np.floor(y * 10), 9) / 10
        for lo in np.unique(tranche):
            m = tranche == lo
            for src in SOURCES:
                e = p[src][m] - y[m]
                lignes_t.append(dict(groupe=g, tranche=round(float(lo), 1), source=src, n=int(m.sum()),
                                     part=float(m.mean()), mae=float(np.abs(e).mean()), biais=float(e.mean())))
    sg, tr = pd.DataFrame(lignes_g), pd.DataFrame(lignes_t)
    chm = pd.read_csv(ICI / "chm_stats.csv").groupby("site_id").agg(
        densite_avant=("densite_avant", "median"), densite_apres=("densite_apres", "median"),
        tuiles_plusieurs_jours=("jours", lambda j: int((j > 1).sum())))
    st = (pd.DataFrame(lignes_s).merge(sites[["site_id", "site_neon", "etat", "year", "lat", "lon"]], on="site_id")
          .merge(chm, on="site_id", how="left"))
    for nom, df in (("metriques_globales", sg), ("metriques_sites", st), ("metriques_tranches", tr),
                    ("controle_predictions", preds)):
        df.to_csv(RESULTATS / f"{nom}.csv", index=False)
    (RESULTATS / "controle.json").write_text(json.dumps(ctrl, indent=1, ensure_ascii=False))
    figures(groupes, st, tr, tranches_onta, float(onta.mae))

    g10 = sg[(sg.resolution == "10m") & (sg.source == ENSEMBLE)].set_index("groupe")
    resume = dict(sites_exclus=exclus, controle_ok=ctrl["ok"],
                  ontario=dict(mae=round(float(onta.mae), 4), biais=round(float(onta.biais), 4), r2=round(float(onta.r2), 4)),
                  **{f"groupe_{g}": dict(n_sites=len(groupes[g]["10m"]), mae=round(float(r.mae), 4),
                                         biais=round(float(r.biais), 4), r2=round(float(r.r2), 4))
                     for g, r in g10.iterrows()})
    (RESULTATS / "resume.json").write_text(json.dumps(resume, indent=1, ensure_ascii=False))

    pd.set_option("display.width", 200)
    print(f"{len(sites)} sites NEON évalués" + (f" ; exclus faute d'images : {exclus}" if exclus else ""))
    print(f"contrôle : écart maximal {ctrl['ecart_max']:.4f}, écart moyen maximal {ctrl['ecart_moyen_max']:.5f} "
          f"-> {'OK' if ctrl['ok'] else 'ÉCHEC'}\n")
    print(sg.round(3).to_string(index=False), "\n")
    print("ensemble, par site (10 m) :")
    print(st[st.source == ENSEMBLE][["groupe", "site_id", "etat", "densite_avant", "densite_apres", "n", "label_moyen",
                                     "mae", "biais", "r2"]].sort_values(["groupe", "mae"]).round(3).to_string(index=False), "\n")
    print("ensemble, MAE par tranche de label (10 m), avec la référence ontarienne :")
    t = tr[tr.source == ENSEMBLE].pivot(index="tranche", columns="groupe", values="mae")
    print(t.assign(Ontario=tranches_onta).round(3).to_string(), "\n")
    print(f"référence Ontario (15 nouveaux sites) : MAE {onta.mae:.3f}, biais {onta.biais:+.3f}, R² {onta.r2:.3f}")
    for g, r in g10.iterrows():
        print(f"groupe {g} : MAE {r.mae:.3f}, biais {r.biais:+.3f}, R² {r.r2:.3f}")
    if not ctrl["ok"]:
        print("\nATTENTION : contrôle en échec, les chiffres ci-dessus ne sont pas interprétables.")
    print(f"\ntableaux et figures dans {RESULTATS}")


if __name__ == "__main__":
    main()
