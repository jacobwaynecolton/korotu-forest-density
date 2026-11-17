"""Étape 4 : évaluer les trois configurations, calibrer les intervalles et choisir l'encodeur (README).

1. Validation croisée par site : chaque site du projet est prédit par les 5 modèles (graines) de son pli, qui
   ne l'ont jamais vu. Ensemble des graines : moyenne pour la configuration L1 ; pour les quantiles, mélange
   des 5 distributions (chacune approchée par 19 quantiles interpolés entre q10, q50 et q90), dont on reprend
   les quantiles 10, 50 et 90 %.
2. Calibration conforme, borne par borne : c_bas = quantile à 90 % de (q10 − label), c_haut = quantile à 90 %
   de (label − q90) ; bornes calibrées q10 − c_bas et q90 + c_haut, ramenées dans [0, 1] et autour de q50.
   En validation croisée, les constantes de chaque pli viennent des 4 autres plis (couverture honnête) ; les
   constantes finales viennent des 5 plis et s'appliquent aux modèles finaux.
3. Choix de l'encodeur (règle fixée d'avance) : le ResNet-50 pré-entraîné sur Sentinel-2 est retenu si son
   score quantile hors pli est meilleur que celui du ResNet-34 ImageNet dans au moins 4 plis sur 5.
4. Test externe : modèles finaux (entraînés sur les 49 sites) sur les 15 nouveaux sites ontariens ; sites NEON
   pour décrire le comportement hors domaine.

Mesures : MAE, RMSE, biais et R² de l'estimation centrale (q50 ou moyenne) ; score quantile (moyenne des
pertes « pinball » à 10, 50 et 90 %, plus bas = meilleur) ; part des labels sous la carte prudente (q10) et
au-dessus de la carte optimiste (q90), cible 10 % chacune ; part dans l'intervalle (cible 80 %) ; largeur
moyenne de l'intervalle. Les sites ou graines dont la prédiction manque sont sautés et signalés.
Sorties : modeles-probabilistes/resultats/ (tableaux .csv, resume.json, figures .png).
Usage : python -I modeles-probabilistes/04_evaluer.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (CONFIGS, GRAINES, N_PLIS, PREDICTIONS, QUANTILES, RESULTATS, fichiers_projet, lire,  # noqa: E402
                    lire_prediction, sites_externes, sites_projet)

NIVEAUX = np.linspace(0.05, 0.95, 19)
ALPHA = 0.1
CONFIG_L1, CONFIG_A, CONFIG_B = "l1_r34_imagenet", "q_r34_imagenet", "q_r50_ssl4eo"
PLIS_MIN_B = 4


# ---------- ensembles ----------
def niveaux(q):
    """(3, ...) q10, q50, q90 -> (19, ...) quantiles 5 % à 95 %, interpolés et prolongés linéairement."""
    q10, q50, q90 = q
    out = [q10 + (u - 0.1) / 0.4 * (q50 - q10) if u <= 0.5 else q50 + (u - 0.5) / 0.4 * (q90 - q50) for u in NIVEAUX]
    return np.clip(np.stack(out), 0, 1)


def melange(preds):
    """Liste de (3, ...) par graine -> (3, ...) quantiles 10, 50, 90 % du mélange des graines."""
    return np.quantile(np.concatenate([niveaux(p) for p in preds]), QUANTILES, axis=0).astype(np.float32)


def ensemble(config, chemins):
    preds = [lire_prediction(c) for c in chemins]
    if CONFIGS[config]["perte"] == "l1":
        return np.mean(preds, axis=0)[0], None, None, preds
    q10, q50, q90 = melange(preds)
    return q50, q10, q90, preds


# ---------- mesures ----------
def pinball(y, q, tau):
    e = y - q
    return float(np.maximum(tau * e, (tau - 1) * e).mean())


def mesures(y, c, q10=None, q90=None):
    e = c - y
    d = dict(n=int(len(y)), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), biais=float(e.mean()),
             r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 else np.nan)
    if q10 is not None:
        d |= dict(score_quantile=float(np.mean([pinball(y, q, t) for q, t in zip((q10, c, q90), QUANTILES)])),
                  sous_q10=float((y < q10).mean()), sur_q90=float((y > q90).mean()),
                  dans_intervalle=float(((y >= q10) & (y <= q90)).mean()), largeur=float((q90 - q10).mean()))
    return d


def constantes(y, q10, q90):
    """Corrections conformes des deux bornes (niveau corrigé pour l'effectif)."""
    n = len(y)
    niveau = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(q10 - y, niveau)), float(np.quantile(y - q90, niveau))


def calibrer(q10, q50, q90, c_bas, c_haut):
    return np.minimum(np.clip(q10 - c_bas, 0, 1), q50), np.maximum(np.clip(q90 + c_haut, 0, 1), q50)


# ---------- rassemblement des cases ----------
def cases_hors_pli(config, plis):
    """Cases valides de chaque site, prédites par les graines de son pli : dict site -> dict de tableaux."""
    out, manques = {}, []
    for r in plis.itertuples():
        chemins = [PREDICTIONS / config / f"pli{r.pli}" / f"g{g}" / f"{r.site_id}.tif" for g in GRAINES]
        presents = [c for c in chemins if c.exists()]
        if len(presents) < len(chemins):
            manques.append(f"{r.site_id} ({len(presents)}/{len(chemins)} graines)")
        if not presents:
            continue
        y = lire(fichiers_projet(r.site_id)[0])
        c, q10, q90, preds = ensemble(config, presents)
        ok = np.isfinite(y) & np.isfinite(c)
        d = dict(pli=r.pli, y=y[ok], c=c[ok], seules=[p[p.shape[0] // 2][ok] for p in preds])
        if q10 is not None:
            d |= dict(q10=q10[ok], q90=q90[ok], seules_q=[(p[0][ok], p[1][ok], p[2][ok]) for p in preds])
        out[r.site_id] = d
    if manques:
        print(f"{config} : prédictions incomplètes pour {len(manques)} sites : {manques[:5]}…")
    return out


def cases_externes(config):
    out = {}
    for e in sites_externes():
        chemins = [PREDICTIONS / config / "final" / f"g{g}" / f"{e['site_id']}.tif" for g in GRAINES]
        presents = [c for c in chemins if c.exists()]
        if not presents:
            continue
        y = lire(e["label"])
        c, q10, q90, _ = ensemble(config, presents)
        ok = np.isfinite(y) & np.isfinite(c)
        out[e["site_id"]] = dict(groupe=e["groupe"], y=y[ok], c=c[ok], q10=q10[ok], q90=q90[ok], graines=len(presents))
    return out


def concat(cases, cle, sites=None):
    return np.concatenate([d[cle] for s, d in cases.items() if sites is None or s in sites])


# ---------- figures ----------
def figures(cv, ext, constantes_finales, retenue, choix):
    # couverture par tranche de couvert, configuration retenue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (titre, y, q10, q90) in zip(axes, (
            ("Validation croisée (calibration croisée)", cv["y"], cv["q10_cal"], cv["q90_cal"]),
            ("15 nouveaux sites ontariens (test externe)", ext["y"], ext["q10_cal"], ext["q90_cal"]))):
        t = np.minimum(np.floor(y * 10), 9) / 10
        x = np.unique(t)
        ax.plot(x + 0.05, [(y[t == v] < q10[t == v]).mean() for v in x], "o-", color="tab:blue", label="label sous q10")
        ax.plot(x + 0.05, [(y[t == v] > q90[t == v]).mean() for v in x], "o-", color="tab:orange", label="label au-dessus de q90")
        ax.axhline(0.10, color="k", lw=1, ls="--", label="cible 10 %")
        ax.set(title=titre, xlabel="label (tranches de 0,1)", xlim=(0, 1))
    axes[0].set_ylabel("part des cases")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Couverture des cartes prudente et optimiste, {retenue}")
    fig.tight_layout(); fig.savefig(RESULTATS / "couverture_tranches.png", dpi=130); plt.close(fig)

    # score quantile et MAE par pli pour les deux encodeurs
    t = choix["par_pli"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, titre in zip(axes, ("score_quantile", "mae"), ("Score quantile hors pli (plus bas = mieux)", "MAE de q50 hors pli")):
        for config, couleur in ((CONFIG_A, "tab:blue"), (CONFIG_B, "tab:green")):
            v = t[t.config == config]
            ax.plot(v.pli, v[col], "o-", color=couleur, label=config)
        if col == "mae":
            v = t[t.config == CONFIG_L1]
            ax.plot(v.pli, v[col], "s--", color="tab:gray", label=CONFIG_L1)
        ax.set(title=titre, xlabel="pli", xticks=range(N_PLIS))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTATS / "plis_encodeurs.png", dpi=130); plt.close(fig)

    # carte d'exemple : un site ontarien externe
    externes = {e["site_id"]: e for e in sites_externes()}
    site = next((s for s in ext["sites"] if externes[s]["groupe"] == "ontario"), None)
    if site is None:
        return
    e = externes[site]
    y = lire(e["label"])
    q = melange([lire_prediction(PREDICTIONS / retenue / "final" / f"g{g}" / f"{site}.tif")
                 for g in GRAINES if (PREDICTIONS / retenue / "final" / f"g{g}" / f"{site}.tif").exists()])
    q10c, q90c = calibrer(q[0], q[1], q[2], *constantes_finales[retenue])
    with rasterio.open(e["images"][0]) as s:
        x = s.read([4, 3, 2]).astype(float)
    v = x[np.isfinite(x)]
    rgb = (np.nan_to_num((x - np.percentile(v, 2)) / (np.percentile(v, 98) - np.percentile(v, 2))).clip(0, 1)).transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    cmap = plt.get_cmap("Greens").copy(); cmap.set_bad("#d9dcd4")
    axes[0].imshow(rgb); axes[0].set_title("Sentinel-2")
    for ax, (titre, a) in zip(axes[1:], (("Label lidar", y), ("Carte prudente (q10 calibré)", q10c),
                                          ("Carte centrale (q50)", q[1]), ("Carte optimiste (q90 calibré)", q90c))):
        im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=0, vmax=1); ax.set_title(titre)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="couvert")
    fig.suptitle(f"{site} (test externe), {retenue}")
    fig.savefig(RESULTATS / "exemple_cartes.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    plis = sites_projet()
    lignes_cv, lignes_pli, lignes_site, resume, constantes_finales, cv_cal = [], [], [], {}, {}, {}

    for config in CONFIGS:
        cases = cases_hors_pli(config, plis)
        if not cases:
            print(f"{config} : aucune prédiction hors pli, configuration sautée")
            continue
        quantile = CONFIGS[config]["perte"] == "quantile"
        y, c = concat(cases, "y"), concat(cases, "c")
        if quantile:
            q10, q90 = concat(cases, "q10"), concat(cases, "q90")
            pli_case = np.concatenate([np.full(len(d["y"]), d["pli"]) for d in cases.values()])
            # calibration croisée : constantes de chaque pli tirées des 4 autres
            q10c, q90c, consts = np.empty_like(q10), np.empty_like(q90), {}
            for k in np.unique(pli_case):
                m = pli_case == k
                consts[int(k)] = constantes(y[~m], q10[~m], q90[~m]) if (~m).any() else (0.0, 0.0)
                q10c[m], q90c[m] = calibrer(q10[m], c[m], q90[m], *consts[int(k)])
            constantes_finales[config] = constantes(y, q10, q90)
            cv_cal[config] = dict(y=y, q10_cal=q10c, q90_cal=q90c)
            lignes_cv += [dict(config=config, ensemble="5 graines", calibration="brute", **mesures(y, c, q10, q90)),
                          dict(config=config, ensemble="5 graines", calibration="conforme croisée", **mesures(y, c, q10c, q90c))]
            seules = [mesures(concat(cases, "y"), np.concatenate([d["seules_q"][g][1] for d in cases.values() if len(d["seules_q"]) > g]),
                              np.concatenate([d["seules_q"][g][0] for d in cases.values() if len(d["seules_q"]) > g]),
                              np.concatenate([d["seules_q"][g][2] for d in cases.values() if len(d["seules_q"]) > g]))
                      for g in range(min(len(d["seules_q"]) for d in cases.values()))]
        else:
            lignes_cv.append(dict(config=config, ensemble="5 graines", calibration="—", **mesures(y, c)))
            seules = [mesures(y, np.concatenate([d["seules"][g] for d in cases.values()]))
                      for g in range(min(len(d["seules"]) for d in cases.values()))]
        if seules:
            lignes_cv.append(dict(config=config, ensemble="un modèle (moyenne des graines)", calibration="brute",
                                  **{k: float(np.mean([s[k] for s in seules])) for k in seules[0]}))
        for k in sorted({d["pli"] for d in cases.values()}):
            sites = {s for s, d in cases.items() if d["pli"] == k}
            args = (concat(cases, "y", sites), concat(cases, "c", sites)) + (
                (concat(cases, "q10", sites), concat(cases, "q90", sites)) if quantile else ())
            lignes_pli.append(dict(config=config, pli=k, sites=len(sites), **mesures(*args)))
        for s, d in cases.items():
            lignes_site.append(dict(config=config, site_id=s, pli=d["pli"],
                                    **mesures(d["y"], d["c"], d.get("q10"), d.get("q90"))))
        if quantile:
            resume[config] = dict(constantes_conformes=dict(zip(("c_bas", "c_haut"), constantes_finales[config])),
                                  constantes_par_pli={k: dict(zip(("c_bas", "c_haut"), v)) for k, v in consts.items()})

    cv, par_pli, par_site = pd.DataFrame(lignes_cv), pd.DataFrame(lignes_pli), pd.DataFrame(lignes_site)
    for nom, df in (("cv_metriques", cv), ("cv_par_pli", par_pli), ("cv_par_site", par_site)):
        df.to_csv(RESULTATS / f"{nom}.csv", index=False)

    # choix de l'encodeur
    choix = dict(par_pli=par_pli)
    retenue = CONFIG_A
    if {CONFIG_A, CONFIG_B} <= set(par_pli.config):
        a = par_pli[par_pli.config == CONFIG_A].set_index("pli").score_quantile
        b = par_pli[par_pli.config == CONFIG_B].set_index("pli").score_quantile
        communs = a.index.intersection(b.index)
        gagnes = int((b[communs] < a[communs]).sum())
        retenue = CONFIG_B if gagnes >= PLIS_MIN_B and len(communs) == N_PLIS else CONFIG_A
        resume["choix_encodeur"] = dict(plis_compares=len(communs), plis_gagnes_par_ssl4eo=gagnes,
                                        regle=f"SSL4EO retenu si meilleur dans au moins {PLIS_MIN_B} plis sur {N_PLIS}",
                                        retenue=retenue)
    resume["configuration_retenue"] = retenue

    # test externe
    lignes_ext, lignes_ext_site, ext_cal = [], [], None
    for config in (CONFIG_A, CONFIG_B):
        if config not in constantes_finales:
            continue
        cases = cases_externes(config)
        if not cases:
            continue
        for groupe in sorted({d["groupe"] for d in cases.values()}):
            sites = {s for s, d in cases.items() if d["groupe"] == groupe}
            y, c, q10, q90 = (concat(cases, k, sites) for k in ("y", "c", "q10", "q90"))
            q10c, q90c = calibrer(q10, c, q90, *constantes_finales[config])
            lignes_ext += [dict(config=config, groupe=groupe, sites=len(sites), calibration="brute", **mesures(y, c, q10, q90)),
                           dict(config=config, groupe=groupe, sites=len(sites), calibration="conforme", **mesures(y, c, q10c, q90c))]
            if config == retenue and groupe == "ontario":
                ext_cal = dict(y=y, q10_cal=q10c, q90_cal=q90c, sites=sorted(sites))
        for s, d in cases.items():
            q10c, q90c = calibrer(d["q10"], d["c"], d["q90"], *constantes_finales[config])
            lignes_ext_site.append(dict(config=config, site_id=s, groupe=d["groupe"], graines=d["graines"],
                                        **mesures(d["y"], d["c"], q10c, q90c)))
    ext, ext_site = pd.DataFrame(lignes_ext), pd.DataFrame(lignes_ext_site)
    ext.to_csv(RESULTATS / "externe_metriques.csv", index=False)
    ext_site.to_csv(RESULTATS / "externe_par_site.csv", index=False)
    if len(ext_site):
        resume["lien_largeur_erreur"] = {g: float(v.largeur.corr(v.mae, method="spearman"))
                                        for g, v in ext_site[ext_site.config == retenue].groupby("groupe") if len(v) > 2}
    (RESULTATS / "resume.json").write_text(json.dumps(resume, indent=1, ensure_ascii=False, default=float))
    if retenue in cv_cal and ext_cal is not None:
        figures(cv_cal[retenue], ext_cal, constantes_finales, retenue, choix)

    pd.set_option("display.width", 220)
    print("\nvalidation croisée par site (cases regroupées) :")
    print(cv.round(4).to_string(index=False))
    print("\npar pli :")
    print(par_pli.round(4).to_string(index=False))
    if "choix_encodeur" in resume:
        print(f"\nchoix de l'encodeur : {resume['choix_encodeur']}")
    for config, (cb, ch) in constantes_finales.items():
        print(f"constantes conformes {config} : c_bas {cb:+.4f}, c_haut {ch:+.4f}")
    print("\ntest externe (modèles finaux) :")
    print(ext.round(4).to_string(index=False))
    print(f"\nconfiguration retenue : {retenue} | résultats dans {RESULTATS}")


if __name__ == "__main__":
    main()
