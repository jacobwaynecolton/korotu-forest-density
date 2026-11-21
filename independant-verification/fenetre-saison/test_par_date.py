"""Test de la fenêtre saisonnière : les scènes Sentinel-2 de début ou de fin de saison donnent-elles plus d'erreur ?

La sélection actuelle (scripts/make_stacks.py) garde des scènes du 1er mai au 15 octobre. Les vols lidar sont tous
entre le 1er juin et le 15 septembre. Hypothèse : avant le 1er juin et après le 15 septembre, le feuillage des
feuillus est incomplet, et le modèle sous-estime le couvert.

Chaque date est prédite séparément, sans réentraînement, et seulement par des modèles qui n'ont jamais vu le site :
  - « ontario_nouveaux » : 15 nouveaux sites ontariens, 5 modèles finaux q_r34_imagenet (carte centrale q50) ;
  - « projet_retenus »   : 6 sites de validation et de test du projet, 3 modèles d'origine (L1). Les sites de
                           validation ont servi à choisir le point d'arrêt, ce qui joue de la même façon sur toutes
                           leurs dates ;
  - « neon »             : 7 sites NEON, modèles finaux, à titre descriptif (hors domaine).
Prédiction comme dans le projet (fenêtres de 256 px, 4 retournements), moyenne sur les modèles, une date à la fois.

Saisons : « début » avant le 1er juin, « cœur » du 1er juin au 15 septembre, « fin » après le 15 septembre.
Mesures par scène : MAE et biais (prédiction − label) sur toutes les cases, MAE sur le couvert intermédiaire
(label entre 0,1 et 0,9), biais sur la forêt (label ≥ 0,5).
Comparaisons, fixées avant de lancer :
  1. chaque scène hors cœur contre la moyenne des scènes du cœur du même site (même groupe de modèles) ;
  2. pour ces sites, MAE de la moyenne sur toutes les dates contre la moyenne sur les seules dates du cœur
     (c'est ce que changerait une fenêtre restreinte à la prédiction).
Sorties : resultats/scenes.csv, resultats/comparaison_scenes.csv, resultats/comparaison_sites.csv, resultats/resume.json,
resultats/ecart_par_date.png
Usage (GPU) : python -I independant-verification/fenetre-saison/test_par_date.py [nombre de sites, pour un essai]
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

ICI = Path(__file__).resolve().parent
PROJET = ICI.parents[1]
PROBA = PROJET / "modeles-probabilistes"
sys.path.insert(0, str(PROBA))
from commun import CONFIGS, GRAINES, MODELES, fichiers_projet, lire, sites_externes  # noqa: E402

_spec = importlib.util.spec_from_file_location("entrainer", PROBA / "03_entrainer.py")
entrainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entrainer)

RESULTATS = ICI / "resultats"
FINAL = "q_r34_imagenet"
ORIGINE = Path("/w/20261/tsalin/MLOps/repro_resultats/20261030-1632/modeles")
MODELES_ORIGINE = ("unet_r34", "unet_r34_s1", "unet_r34_s2")
SCENES = {"ontario_nouveaux": PROJET / "independant-verification/model-verification/scenes_s2.csv",
          "neon": PROJET / "independant-verification/model-verification-USA/scenes_s2.csv",
          "projet_retenus": PROJET / "data/manifest/s2_scenes.csv"}


def saison(date):
    md = date[5:10]
    return "début" if md < "06-01" else "fin" if md > "09-15" else "cœur"


def charger_finaux(dev):
    cfg = CONFIGS[FINAL]
    out = []
    for g in GRAINES:
        m = entrainer.construire(dict(cfg, poids="aucun"), dev)
        m.load_state_dict({k: v.float() for k, v in torch.load(MODELES / FINAL / f"g{g}.pt", map_location=dev).items()})
        meta = json.loads((MODELES / FINAL / f"g{g}.json").read_text())
        out.append((m.eval(), np.array(meta["moyenne"]), np.array(meta["ecart"]), "quantile"))
    return out


def charger_origine(dev):
    out = []
    for nom in MODELES_ORIGINE:
        cfg = json.loads((ORIGINE / nom / "config.json").read_text())
        m = entrainer.construire(dict(encodeur=cfg["encoder"], poids="aucun", perte="l1"), dev)
        m.load_state_dict(torch.load(ORIGINE / nom / "best.pt", map_location=dev))
        out.append((m.eval(), np.array(cfg["mean"]), np.array(cfg["std"]), "l1"))
    return out


def predire_date(modeles, x, dev):
    """x : (12, H, W) d'une seule date -> carte centrale moyenne sur les modèles."""
    cartes = []
    for m, moyenne, ecart, perte in modeles:
        p = entrainer.predire(m, entrainer.vers_tenseur(x[None], moyenne, ecart, dev), perte)
        cartes.append(p[p.shape[0] // 2])
    return np.mean(cartes, axis=0)


def mesures(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    e = p - y
    mi = (y >= 0.1) & (y < 0.9)
    fo = y >= 0.5
    return dict(cases=int(ok.sum()), mae=float(np.abs(e).mean()), biais=float(e.mean()),
                mae_intermediaire=float(np.abs(e[mi]).mean()) if mi.any() else np.nan,
                biais_foret=float(e[fo].mean()) if fo.any() else np.nan)


def sites_a_tester():
    """(groupe, site, chemin du label, {k: chemin de l'image}) pour les trois groupes."""
    out = []
    for e in sites_externes():
        groupe = "ontario_nouveaux" if e["groupe"] == "ontario" else "neon"
        images = {int(re.search(r"_k(\d+)\.tif$", str(p)).group(1)): p for p in e["images"]}
        out.append((groupe, e["site_id"], e["label"], images))
    sites = pd.read_csv(PROJET / "data/manifest/sites.csv")
    for s in sites[sites.split.isin(["val", "test"])].site_id:
        label, images = fichiers_projet(s)
        out.append(("projet_retenus", s, label, {int(re.search(r"_k(\d+)\.tif$", str(p)).group(1)): p for p in images}))
    return out


def main(max_sites=None):
    RESULTATS.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modeles = {"final": charger_finaux(dev), "origine": charger_origine(dev)}
    scenes = pd.concat([pd.read_csv(p).assign(groupe=g) for g, p in SCENES.items()], ignore_index=True)
    lignes, sites_res = [], []
    for groupe, site, label, images in sites_a_tester()[:max_sites]:
        y = lire(label)
        jeu = modeles["origine" if groupe == "projet_retenus" else "final"]
        cartes = {}
        for k, chemin in sorted(images.items()):
            sc = scenes[(scenes.groupe == groupe) & (scenes.site_id == site) & (scenes.k == k)]
            assert len(sc) == 1, (groupe, site, k)
            sc = sc.iloc[0]
            with rasterio.open(chemin) as s:
                x = s.read()
            cartes[k] = predire_date(jeu, x, dev)
            lignes.append(dict(groupe=groupe, site_id=site, k=k, scene=sc.scene, date=sc.date,
                               jour=pd.Timestamp(sc.date).dayofyear, saison=saison(sc.date),
                               jours_du_vol=int(sc.days_from_flight), **mesures(y, cartes[k])))
        d = pd.DataFrame([l for l in lignes if l["site_id"] == site and l["groupe"] == groupe])
        coeur = d[d.saison == "cœur"].k.tolist()
        if len(coeur) and len(coeur) < len(d):
            toutes = mesures(y, np.mean([cartes[k] for k in d.k], axis=0))
            seul_coeur = mesures(y, np.mean([cartes[k] for k in coeur], axis=0))
            sites_res.append(dict(groupe=groupe, site_id=site, dates=len(d), dates_coeur=len(coeur),
                                  mae_toutes_dates=toutes["mae"], mae_dates_coeur=seul_coeur["mae"],
                                  biais_toutes_dates=toutes["biais"], biais_dates_coeur=seul_coeur["biais"]))
        print(f"{groupe:17s} {site:22s} " + "  ".join(f"{r.date} {r.saison:5s} MAE {r.mae:.3f}"
                                                       for r in d.itertuples()), flush=True)

    sc = pd.DataFrame(lignes)
    sc.to_csv(RESULTATS / "scenes.csv", index=False)

    # 1. chaque scène hors cœur contre les scènes du cœur du même site
    comp = []
    for (g, s), d in sc.groupby(["groupe", "site_id"]):
        c = d[d.saison == "cœur"]
        if c.empty:
            continue
        for r in d[d.saison != "cœur"].itertuples():
            comp.append(dict(groupe=g, site_id=s, date=r.date, saison=r.saison, jours_du_vol=r.jours_du_vol,
                             jours_du_vol_coeur=float(c.jours_du_vol.abs().mean()),
                             **{f"ecart_{m}": getattr(r, m) - c[m].mean()
                                for m in ("mae", "biais", "mae_intermediaire", "biais_foret")}))
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTATS / "comparaison_scenes.csv", index=False)
    sites_res = pd.DataFrame(sites_res)
    sites_res["gain"] = sites_res.mae_toutes_dates - sites_res.mae_dates_coeur
    sites_res.to_csv(RESULTATS / "comparaison_sites.csv", index=False)

    def resume(c, s):
        return dict(scenes_hors_coeur=int(len(c)), dont_debut=int((c.saison == "début").sum()),
                    dont_fin=int((c.saison == "fin").sum()),
                    ecart_mae_moyen=float(c.ecart_mae.mean()), ecart_mae_median=float(c.ecart_mae.median()),
                    scenes_plus_mauvaises=int((c.ecart_mae > 0).sum()),
                    ecart_biais_moyen=float(c.ecart_biais.mean()),
                    ecart_biais_foret_moyen=float(c.ecart_biais_foret.mean()),
                    ecart_mae_intermediaire_moyen=float(c.ecart_mae_intermediaire.mean()),
                    sites=int(len(s)), gain_moyen_si_coeur_seul=float(s.gain.mean()),
                    sites_ameliores=int((s.gain > 0).sum()))
    sans_neon = comp.groupby("groupe").filter(lambda d: d.name != "neon")
    res = {"ontario (nouveaux sites + sites retenus du projet)": resume(sans_neon, sites_res[sites_res.groupe != "neon"])}
    for g in sorted(comp.groupe.unique()):
        res[g] = resume(comp[comp.groupe == g], sites_res[sites_res.groupe == g])
    (RESULTATS / "resume.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # figure : écart de chaque scène à la moyenne des scènes du cœur de son site, selon la date
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    couleurs = {"ontario_nouveaux": "tab:blue", "projet_retenus": "tab:gray", "neon": "tab:orange"}
    for ax, m, titre in zip(axes, ("mae", "biais"), ("MAE de la scène − MAE moyenne des scènes du cœur",
                                                     "biais de la scène − biais moyen des scènes du cœur")):
        for (g, s), d in sc.groupby(["groupe", "site_id"]):
            c = d[d.saison == "cœur"]
            if c.empty or len(d) < 2:
                continue
            ax.scatter(d.jour, d[m] - c[m].mean(), s=22, color=couleurs[g], alpha=0.8,
                       label=g if (m == "mae" and s == sc[sc.groupe == g].site_id.iloc[0]) else None)
        for j in (pd.Timestamp("2021-06-01").dayofyear, pd.Timestamp("2021-09-15").dayofyear):
            ax.axvline(j, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(title=titre, xlabel="jour de l'année (lignes : 1er juin et 15 septembre)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTATS / "ecart_par_date.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 200)
    print("\nscènes hors cœur contre scènes du cœur du même site :")
    print(comp.round(4).to_string(index=False))
    print("\nmoyenne sur toutes les dates contre dates du cœur seules :")
    print(sites_res.round(4).to_string(index=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
