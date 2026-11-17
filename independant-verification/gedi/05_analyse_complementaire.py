"""Étape 5 (exploratoire) : analyses faites après avoir vu les résultats de l'étape 4.

Ces analyses ne font pas partie du protocole validé le 30 octobre 2026. Elles servent à comprendre les
critères en échec, pas à les remplacer : le verdict officiel reste celui de resultats/criteres.json.

  - Effet du niveau de couvert : l'écart entre GEDI et le label dépend du couvert (relation convexe,
    due aux deux définitions du couvert). On le retire avec la médiane de GEDI par tranche de 0,05 de
    label, interpolée ; ce qui reste est le « biais corrigé ».
  - Sites : corrélation de rang selon l'étendue des labels dans le site ; biais brut et corrigé.
  - Groupes : biais brut et corrigé par année du vol, année du tir GEDI, zone UTM et tuiles 2025, plus
    le tableau croisé année du vol × année du tir, qui sépare un effet GEDI d'un effet propre aux sites.
  - Tirs manqués : label > 0,9 mais GEDI cover < 0,1.
  - Test de position refait sur les tuiles 2025 (WGS84) seules.

Même sélection que l'étape 4 (fonction importée de 04_comparer.py).
Sorties dans independant-verification/gedi/resultats/ : complementaire_sites.csv, complementaire_groupes.csv,
complementaire_croise.csv, complementaire_decalage.csv, et rapport_donnees.json (données des graphiques
du rapport en anglais, avec les mêmes clés que la page).
Usage : python -I independant-verification/gedi/05_analyse_complementaire.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import CLASSE_EAU, DECALAGES, RESULTATS, SORTIES  # noqa: E402

etape4 = importlib.import_module("04_comparer")   # nom de module commençant par un chiffre : pas d'import direct
selection, N_MIN_SITE, N_MIN_GROUPE = etape4.selection, etape4.N_MIN_SITE, etape4.N_MIN_GROUPE

PAS_COURBE = 0.05                      # largeur des tranches de label pour la courbe médiane de GEDI
N_CASES = 20                           # cases par axe des cartes de densité du rapport
MANQUE_LABEL, MANQUE_GEDI = 0.9, 0.1   # tir « manqué » : label > 0,9 et GEDI cover < 0,1


def par_tranche(x, y, bornes, n_min=30):
    """Par tranche de x : nombre de tirs, moyenne de x, quartiles et médiane de y (tranches d'au moins n_min tirs)."""
    g = pd.DataFrame(dict(x=x, y=y, t=pd.cut(x, bornes, include_lowest=True))).groupby("t", observed=True)
    r = g.agg(n=("y", "size"), x=("x", "mean"), q25=("y", lambda v: v.quantile(0.25)),
              med=("y", "median"), q75=("y", lambda v: v.quantile(0.75)))
    return r[r.n >= n_min].reset_index(drop=True)


def ajouter_biais(d):
    """Biais brut (GEDI - label) et biais corrigé de l'effet du niveau de couvert."""
    courbe = par_tranche(d.label_gt2, d.cover, np.linspace(0, 1, round(1 / PAS_COURBE) + 1), n_min=1)
    d["biais"] = d.cover - d.label_gt2
    d["biais_corrige"] = d.cover - np.interp(d.label_gt2, courbe.x, courbe.med)
    return d


def tableau_sites(d):
    lignes = []
    for site_id, g in d.groupby("site_id"):
        if len(g) < N_MIN_SITE:
            continue
        lignes.append(dict(site_id=site_id, annee_vol=int(g.annee_vol.iloc[0]), zone=int(g.zone.iloc[0]), n=len(g),
                           spearman_couvert=g.label_gt2.corr(g.cover, method="spearman"),
                           label_moyen=g.label_gt2.mean(), label_ecart_type=g.label_gt2.std(),
                           part_label_sous_0_1=(g.label_gt2 < 0.1).mean(),
                           biais=g.biais.median(), biais_corrige=g.biais_corrige.median()))
    st = pd.DataFrame(lignes)
    st["rang_faible"] = st.spearman_couvert < 0.40
    for col in ("biais", "biais_corrige"):
        st[f"{col}_hors_2et"] = (st[col] - st[col].mean()).abs() > 2 * st[col].std()
    st["nord_km"] = st.site_id.str.extract(r"_N(\d{4})_", expand=False).astype(int)   # nord UTM, proche de la latitude
    return st


def tableau_groupes(d):
    d = d.assign(tuiles=np.where(d.annee_vol == 2025, "2025 (WGS84)", "autres (NAD83)"))
    lignes = []
    for colonne in ("annee_vol", "annee_tir", "zone", "tuiles"):
        for valeur, g in d.groupby(colonne):
            if len(g) >= N_MIN_GROUPE:
                lignes.append(dict(groupement=colonne, groupe=str(valeur), n=len(g), n_sites=g.site_id.nunique(),
                                   label_moyen=g.label_gt2.mean(), biais=g.biais.median(),
                                   biais_corrige=g.biais_corrige.median()))
    return pd.DataFrame(lignes)


def decalage_par_tuiles(d):
    """Test de position par groupe de tuiles ; chaque décalage garde tous ses tirs valides (pas d'intersection)."""
    lignes = []
    for nom, g in (("2025 (WGS84)", d[d.annee_vol == 2025]), ("autres (NAD83)", d[d.annee_vol != 2025])):
        for ox in DECALAGES:
            for oy in DECALAGES:
                c, p = f"lab2_dx{ox}_dy{oy}", f"p98_dx{ox}_dy{oy}"
                gg = g[[c, p, "cover", "rh98"]].dropna()
                lignes.append(dict(tuiles=nom, dx=ox, dy=oy, n=len(gg),
                                   spearman_couvert=gg[c].corr(gg.cover, method="spearman"),
                                   pearson_hauteur=gg[p].corr(gg.rh98)))
    return pd.DataFrame(lignes)


def entonnoir(t):
    """Nombre de tirs après chaque filtre, dans l'ordre de la sélection de l'étape 4."""
    m = t.sensitivity >= etape4.SENSIBILITE_MIN
    etapes = [("extraits dans l'emprise des sites", len(pd.read_parquet(SORTIES / "gedi_tirs.parquet", columns=["cle"]))),
              ("pré-sélectionnés (étape 3)", len(t)), ("sensibilité", int(m.sum()))]
    for nom, filtre in (("feuillage", t.leaf_off == 0), ("sans perturbation", ~t.perturbe & t.hansen_lu),
                        ("hors eau", t.worldcover != CLASSE_EAU),
                        ("références complètes", t.label_gt2.notna() & t.chm_p98.notna() & (t.cover_z5 >= 0))):
        m &= filtre
        etapes.append((nom, int(m.sum())))
    assert etapes[-1][1] == int(selection(t).sum()), "l'entonnoir ne reproduit plus la sélection de l'étape 4"
    return etapes


def donnees_rapport(d, st, sg, croise, etapes):
    """Données des graphiques du rapport, avec les clés (en anglais) que lit la page."""
    def tranches(x, y, bornes):
        return [dict(n=int(r.n), x=round(r.x, 4), q25=round(r.q25, 4), med=round(r.med, 4), q75=round(r.q75, 4))
                for r in par_tranche(x, y, bornes).itertuples()]

    def densite(x, y, borne):
        return np.histogram2d(x, y, bins=N_CASES, range=[[0, borne], [0, borne]])[0].astype(int).tolist()

    b01, bh = np.linspace(0, 1, round(1 / PAS_COURBE) + 1), np.arange(0, 42, 2)
    dec = etape4.tableau_decalage(d)
    manques = (d.label_gt2 > MANQUE_LABEL) & (d.cover < MANQUE_GEDI)
    annees = sg[sg.groupement == "annee_vol"]
    return dict(
        n=len(d), n_sites=int(d.site_id.nunique()),
        cover2=dict(heat=densite(d.label_gt2, d.cover, 1), bins=tranches(d.label_gt2, d.cover, b01)),
        cover5=dict(heat=densite(d.label_gt5, d.cover_z5, 1), bins=tranches(d.label_gt5, d.cover_z5, b01)),
        height=dict(heat=densite(d.chm_p98, d.rh98, 40), bins=tranches(d.chm_p98, d.rh98, bh)),
        shift=[dict(dx=int(r.dx), dy=int(r.dy), c=round(r.spearman_couvert, 4), h=round(r.pearson_hauteur, 4))
               for r in dec.itertuples()],
        sites=[dict(site=r.site_id, n=int(r.n), year=int(r.annee_vol), rho=round(r.spearman_couvert, 3),
                    bias=round(r.biais, 3), resid=round(r.biais_corrige, 3), lab_mean=round(r.label_moyen, 3),
                    lab_sd=round(r.label_ecart_type, 3), flag_rho=bool(r.rang_faible), flag_bias=bool(r.biais_hors_2et),
                    flag_resid=bool(r.biais_corrige_hors_2et)) for r in st.itertuples()],
        sites_bias_mu_sd=[round(st.biais.mean(), 4), round(st.biais.std(), 4)],
        sites_resid_mu_sd=[round(st.biais_corrige.mean(), 4), round(st.biais_corrige.std(), 4)],
        by_lidar_year=[dict(k=r.groupe, n=int(r.n), bias=round(r.biais, 4), resid=round(r.biais_corrige, 4),
                            lab=round(r.label_moyen, 3)) for r in annees.itertuples()],
        sites_per_year={r.groupe: int(r.n_sites) for r in annees.itertuples()},
        cross=[dict(lidar=int(r.annee_vol), gedi=int(r.annee_tir), n=int(r.n), resid=round(r.biais_corrige, 3))
               for r in croise.itertuples()],
        odd=dict(n=int(manques.sum()), share=round(manques.mean(), 4),
                 share_dense=round(manques.sum() / (d.label_gt2 > MANQUE_LABEL).sum(), 4),
                 rh98_med=round(d.loc[manques, "rh98"].median(), 2), p98_med=round(d.loc[manques, "chm_p98"].median(), 2),
                 n_sites=int(d.loc[manques, "site_id"].nunique())),
        funnel=[[nom, n] for nom, n in etapes],
    )


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(SORTIES / "gedi_tirs_refs.parquet")
    d = ajouter_biais(t[selection(t)].copy())
    st, sg = tableau_sites(d), tableau_groupes(d)
    croise = (d.groupby(["annee_vol", "annee_tir"])
              .agg(n=("biais", "size"), biais=("biais", "median"), biais_corrige=("biais_corrige", "median"))
              .reset_index())
    dt, etapes = decalage_par_tuiles(d), entonnoir(t)
    for nom, df in (("sites", st), ("groupes", sg), ("croise", croise), ("decalage", dt)):
        df.to_csv(RESULTATS / f"complementaire_{nom}.csv", index=False)
    rapport = donnees_rapport(d, st, sg, croise, etapes)
    (RESULTATS / "rapport_donnees.json").write_text(json.dumps(rapport, separators=(",", ":"), ensure_ascii=False))

    pd.set_option("display.width", 200)
    print(f"{len(d):,} tirs retenus dans {d.site_id.nunique()} sites (même sélection que l'étape 4)\n")
    print("entonnoir :", " -> ".join(f"{nom} {n:,}" for nom, n in etapes), "\n")
    print(f"sites (≥ {N_MIN_SITE} tirs) : {len(st)} ; Spearman du site / écart-type du label dans le site : "
          f"{st.spearman_couvert.corr(st.label_ecart_type, method='spearman'):.2f}")
    print(f"  rang faible (< 0,40) : {st.loc[st.rang_faible, 'site_id'].tolist()}")
    print(f"  biais brut hors 2 écarts-types : {st.loc[st.biais_hors_2et, 'site_id'].tolist()}")
    print(f"  biais corrigé hors 2 écarts-types : {st.loc[st.biais_corrige_hors_2et, 'site_id'].tolist()}")
    print(f"  biais corrigé / nord du site : Spearman {st.biais_corrige.corr(st.nord_km, method='spearman'):.2f}\n")
    for colonne, g in sg.groupby("groupement", sort=False):
        print(f"{colonne} : écart brut {g.biais.max() - g.biais.min():.3f}, "
              f"corrigé {g.biais_corrige.max() - g.biais_corrige.min():.3f}")
        print(g.drop(columns="groupement").round(3).to_string(index=False), "\n")
    print("biais corrigé, année du vol × année du tir :")
    print(croise.pivot(index="annee_vol", columns="annee_tir", values="biais_corrige").round(3).to_string(), "\n")
    o = rapport["odd"]
    print(f"tirs manqués (label > {MANQUE_LABEL}, GEDI < {MANQUE_GEDI}) : {o['n']} ({o['share']:.1%}) dans {o['n_sites']} sites ; "
          f"RH98 médian {o['rh98_med']} m contre {o['p98_med']} m dans le CHM")
    for nom, g in dt.groupby("tuiles"):
        mc, mh = g.loc[g.spearman_couvert.idxmax()], g.loc[g.pearson_hauteur.idxmax()]
        print(f"position, tuiles {nom} : couvert max en ({mc.dx}, {mc.dy}) ρ {mc.spearman_couvert:.3f} ; "
              f"hauteur max en ({mh.dx}, {mh.dy}) r {mh.pearson_hauteur:.3f}")
    print(f"\ntableaux et données du rapport dans {RESULTATS}")


if __name__ == "__main__":
    main()
