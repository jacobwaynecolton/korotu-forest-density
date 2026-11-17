"""Étape 4 : comparer GEDI à nos données et appliquer les critères fixés dans le README.

Sélection finale, parmi les tirs de l'étape 3 : sensibilité ≥ 0,95 ; leaf_off_flag = 0 ; pas de
perturbation Hansen ; pas d'eau (classe WorldCover 80 fournie par GEDI : sur l'eau, l'accord est
trivial et gonflerait les corrélations) ; références labels et CHM complètes.
Variantes : faisceaux pleine puissance seulement, tirs de nuit seulement, même année que le vol,
moyenne gaussienne au lieu de la moyenne simple.

Sorties : independant-verification/gedi/resultats/ (criteres.json, tableaux .csv, figures .png).
Usage : python -I independant-verification/gedi/04_comparer.py
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
from commun import CLASSE_EAU, DECALAGES, RESULTATS, SORTIES  # noqa: E402

SENSIBILITE_MIN = 0.95
N_MIN_SITE, N_MIN_TRANCHE, N_MIN_GROUPE = 30, 30, 100
TOLERANCE_MONOTONIE = 0.01


def selection(t):
    return ((t.sensitivity >= SENSIBILITE_MIN) & (t.leaf_off == 0) & ~t.perturbe & t.hansen_lu
            & (t.worldcover != CLASSE_EAU) & t.label_gt2.notna() & t.chm_p98.notna() & (t.cover_z5 >= 0))


def stats(d, x, y):
    """Accord entre une référence `x` (nos données) et une mesure `y` ; biais = y - x."""
    if len(d) < 3:
        return dict(n=len(d))
    e = d[y] - d[x]
    return dict(n=len(d), pearson=d[x].corr(d[y]), spearman=d[x].corr(d[y], method="spearman"),
                biais_median=float(e.median()), mae=float(e.abs().mean()), rmse=float(np.sqrt((e ** 2).mean())))


def tableau_variantes(d, eau):
    variantes = {"principale": d, "pleine_puissance": d[d.pleine_puissance], "nuit": d[d.soleil < 0],
                 "meme_annee": d[d.ecart_annees == 0]}
    lignes = []
    for nom, v in variantes.items():
        lignes += [dict(variante=nom, comparaison="hauteur : rh98 GEDI / centile 98 CHM (m)", **stats(v, "chm_p98", "rh98")),
                   dict(variante=nom, comparaison="couvert : cover GEDI / label > 2 m", **stats(v, "label_gt2", "cover")),
                   dict(variante=nom, comparaison="couvert > 5 m : cover_z5 GEDI / label > 5 m", **stats(v, "label_gt5", "cover_z5"))]
    lignes += [dict(variante="gaussienne", comparaison="couvert : cover GEDI / label > 2 m (gaussien)", **stats(d, "label_gt2_gauss", "cover")),
               dict(variante="controle", comparaison="label > 2 m / CHM > 2 m, même disque", **stats(d, "chm_gt2", "label_gt2")),
               dict(variante="eau", comparaison="sur l'eau : cover GEDI / label > 2 m", **stats(eau, "label_gt2", "cover"))]
    return pd.DataFrame(lignes)


def tableau_sites(d):
    lignes = []
    for site_id, g in d.groupby("site_id"):
        lignes.append(dict(site_id=site_id, split=g.split.iloc[0], annee_vol=g.annee_vol.iloc[0], zone=g.zone.iloc[0], n=len(g),
                           spearman_couvert=g.label_gt2.corr(g.cover, method="spearman") if len(g) >= 3 else np.nan,
                           biais_couvert=(g.cover - g.label_gt2).median(),
                           pearson_hauteur=g.chm_p98.corr(g.rh98) if len(g) >= 3 else np.nan,
                           biais_hauteur=(g.rh98 - g.chm_p98).median()))
    st = pd.DataFrame(lignes)
    assez = st.n >= N_MIN_SITE
    moy, et = st.loc[assez, "biais_couvert"].mean(), st.loc[assez, "biais_couvert"].std()
    st["aberrant"] = assez & ((st.spearman_couvert < 0.40) | ((st.biais_couvert - moy).abs() > 2 * et))
    return st


def tableau_groupes(d):
    d = d.assign(tuiles=np.where(d.annee_vol == 2025, "2025 (WGS84)", "autres (NAD83)"))
    lignes = []
    for colonne in ("annee_vol", "zone", "tuiles"):
        for valeur, g in d.groupby(colonne):
            if len(g) >= N_MIN_GROUPE:
                lignes.append(dict(groupement=colonne, groupe=str(valeur), n=len(g),
                                   biais_couvert=(g.cover - g.label_gt2).median(),
                                   biais_hauteur=(g.rh98 - g.chm_p98).median()))
    return pd.DataFrame(lignes, columns=["groupement", "groupe", "n", "biais_couvert", "biais_hauteur"])


def tableau_decalage(d):
    cols = [f"{p}_dx{ox}_dy{oy}" for p in ("lab2", "p98") for ox in DECALAGES for oy in DECALAGES]
    dd = d.dropna(subset=cols)
    lignes = [dict(dx=ox, dy=oy, n=len(dd),
                   spearman_couvert=dd[f"lab2_dx{ox}_dy{oy}"].corr(dd.cover, method="spearman"),
                   pearson_hauteur=dd[f"p98_dx{ox}_dy{oy}"].corr(dd.rh98))
              for ox in DECALAGES for oy in DECALAGES]
    return pd.DataFrame(lignes)


def figures(d, tranches, st, dec):
    def nuage(ax, x, y, titre, lim, unite):
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1, bins="log", cmap="viridis", extent=(lim[0], lim[1], lim[0], lim[1]))
        ax.plot(lim, lim, "k--", lw=1, label="1:1")
        ax.set(xlim=lim, ylim=lim, title=titre, xlabel=f"nos données {unite}", ylabel=f"GEDI {unite}")
        plt.colorbar(hb, ax=ax, label="nombre de tirs")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    nuage(ax, d.chm_p98, d.rh98, "Hauteur de la canopée", (0, 40), "(m)")
    fig.tight_layout(); fig.savefig(RESULTATS / "hauteurs.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    nuage(axes[0], d.label_gt2, d.cover, "Couvert : label > 2 m / GEDI cover", (0, 1), "")
    axes[0].plot(tranches.label_moyen, tranches.cover_mediane, "o-", color="tab:red", label="médiane par tranche")
    axes[0].legend(loc="upper left")
    nuage(axes[1], d.label_gt5, d.cover_z5, "Couvert > 5 m : label / GEDI cover_z", (0, 1), "")
    nuage(axes[2], d.chm_gt2, d.label_gt2, "Contrôle : CHM > 2 m / label > 2 m", (0, 1), "")
    axes[2].set(xlabel="CHM 0,5 m sur le disque", ylabel="labels 10 m sur le disque")
    fig.tight_layout(); fig.savefig(RESULTATS / "couvert.png", dpi=130); plt.close(fig)

    st = st.sort_values("spearman_couvert")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.22 * len(st))), sharey=True)
    couleurs = np.where(st.aberrant, "tab:red", np.where(st.split == "train", "tab:gray", "tab:blue"))
    axes[0].barh(st.site_id, st.spearman_couvert, color=couleurs)
    axes[0].axvline(0.40, color="k", ls="--", lw=1)
    axes[0].set(title="Corrélation de rang du couvert, par site", xlabel="Spearman")
    axes[1].barh(st.site_id, st.biais_couvert, color=couleurs)
    axes[1].set(title="Biais médian du couvert (GEDI - label)", xlabel="biais")
    axes[0].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(RESULTATS / "sites.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, titre in zip(axes, ("spearman_couvert", "pearson_hauteur"), ("Couvert (Spearman)", "Hauteur (Pearson)")):
        m = dec.pivot(index="dy", columns="dx", values=col).sort_index(ascending=False)
        im = ax.imshow(m.values, cmap="viridis")
        ax.set(xticks=range(len(m.columns)), xticklabels=m.columns, yticks=range(len(m.index)), yticklabels=m.index,
               xlabel="décalage vers l'est (m)", ylabel="décalage vers le nord (m)", title=f"Test de position : {titre}")
        for (i, j), v in np.ndenumerate(m.values):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="w")
        plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(RESULTATS / "decalage.png", dpi=130); plt.close(fig)


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(SORTIES / "gedi_tirs_refs.parquet")
    d = t[selection(t)].copy()
    eau = t[(t.worldcover == CLASSE_EAU) & t.label_gt2.notna()]
    print(f"{len(t):,} tirs pré-sélectionnés -> {len(d):,} retenus dans {d.site_id.nunique()} sites "
          f"({len(eau):,} tirs sur l'eau mis à part)\n")
    if len(d) < 30:
        sys.exit("Moins de 30 tirs retenus : pas assez pour comparer. Vérifier les étapes 2 et 3.")

    sv = tableau_variantes(d, eau)
    bornes = np.round(np.arange(0, 1.01, 0.1), 1)
    tranches = (d.assign(tranche=pd.cut(d.label_gt2, bornes, include_lowest=True))
                .groupby("tranche", observed=True)
                .agg(n=("cover", "size"), label_moyen=("label_gt2", "mean"), cover_mediane=("cover", "median"),
                     cover_z5_mediane=("cover_z5", "median")).reset_index())
    tranches = tranches[tranches.n >= N_MIN_TRANCHE]
    st, sg, dec = tableau_sites(d), tableau_groupes(d), tableau_decalage(d)
    for nom, df in (("stats_variantes", sv), ("tranches", tranches), ("stats_sites", st), ("stats_groupes", sg), ("decalage", dec)):
        df.to_csv(RESULTATS / f"{nom}.csv", index=False)
    figures(d, tranches, st, dec)

    h, c = stats(d, "chm_p98", "rh98"), stats(d, "label_gt2", "cover")
    medianes = tranches.cover_mediane.values
    croissante = bool(np.all(np.diff(medianes) >= -TOLERANCE_MONOTONIE))
    ecart_groupes = (float(sg.groupby("groupement").biais_couvert.agg(lambda x: x.max() - x.min()).max())
                     if len(sg) else float("nan"))

    def meilleur(col):
        return dec.loc[dec[col].idxmax(), ["dx", "dy"]].astype(int).tolist() if dec[col].notna().any() else [99, 99]

    meilleur_c, meilleur_h = meilleur("spearman_couvert"), meilleur("pearson_hauteur")
    position_ok = all(abs(v) <= 10 for v in meilleur_c + meilleur_h)
    criteres = [
        ("hauteurs : corrélation ≥ 0,85", round(h["pearson"], 3), h["pearson"] >= 0.85),
        ("hauteurs : biais médian entre -2 et +2 m", round(h["biais_median"], 2), -2 <= h["biais_median"] <= 2),
        ("couvert : Spearman ≥ 0,70", round(c["spearman"], 3), c["spearman"] >= 0.70),
        ("couvert : relation croissante d'une tranche à l'autre", [round(float(v), 3) for v in medianes], croissante),
        ("sites : aucun site aberrant", st.loc[st.aberrant, "site_id"].tolist(), not st.aberrant.any()),
        ("groupes : écart de biais du couvert ≤ 0,05", round(ecart_groupes, 3), ecart_groupes <= 0.05),
        ("position : corrélation maximale à ±10 m près", {"couvert": meilleur_c, "hauteur": meilleur_h}, position_ok),
    ]
    sortie = dict(n_tirs=len(d), n_sites=int(d.site_id.nunique()),
                  criteres=[dict(nom=n, valeur=v, ok=bool(ok)) for n, v, ok in criteres],
                  reussis=sum(bool(ok) for *_, ok in criteres), total=len(criteres))
    (RESULTATS / "criteres.json").write_text(json.dumps(sortie, indent=1, ensure_ascii=False, default=float))

    pd.set_option("display.width", 200)
    print(sv.round(3).to_string(index=False), "\n")
    print("médiane de GEDI cover par tranche de label :")
    print(tranches.round(3).to_string(index=False), "\n")
    print(f"sites (≥ {N_MIN_SITE} tirs) : Spearman médian {st[st.n >= N_MIN_SITE].spearman_couvert.median():.3f}, "
          f"min {st[st.n >= N_MIN_SITE].spearman_couvert.min():.3f}\n")
    for nom, valeur, ok in criteres:
        print(f"{'PASS' if ok else 'FAIL'}  {nom}  ->  {valeur}")
    print(f"\n{sortie['reussis']}/{sortie['total']} critères réussis | figures et tableaux dans {RESULTATS}")


if __name__ == "__main__":
    main()
