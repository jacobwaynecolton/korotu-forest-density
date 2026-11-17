"""Étape 0 : effet de la résolution du CHM (0,5 m en Ontario, 1 m chez NEON) sur le label.

Sur les CHM ontariens des 15 nouveaux sites de l'étude précédente, on calcule le label de chaque case de
10 m deux fois, dans la grille d'origine des tuiles :
  - à 0,5 m, comme le projet (part des 400 pixels au-dessus de 2 m) ;
  - à 1 m, en prenant le maximum de chaque groupe de 2 × 2 pixels (part des 100 pixels au-dessus de 2 m).
Le maximum imite un CHM à 1 m construit sur les points les plus hauts de chaque mètre carré. NEON comble
aussi les petits trous du CHM (méthode « pit-free ») ; cet effet-là ne peut pas être reproduit sans les
nuages de points, et reste une limite.
Variante : la moyenne des 2 × 2 pixels au lieu du maximum.

Seules les cases entièrement valides sont gardées. Écart = label à 1 m - label à 0,5 m.
Sortie : model-verification-USA/resultats/effet_resolution.csv (par site) et effet_resolution_tranches.csv.
Usage : python -I independant-verification/model-verification-USA/00_effet_resolution.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import CHM_ONTARIO, RESULTATS  # noqa: E402

HAUTEUR_MIN, HAUTEUR_MAX, SEUIL = -2.0, 80.0, 2.0   # comme make_labels.py
CASE = 20   # pixels de 0,5 m par côté de case de 10 m


def labels_tuile(chemin):
    """Labels à 0,5 m et à 1 m (max et moyenne 2 × 2) des cases entièrement valides d'une tuile."""
    with rasterio.open(chemin) as s:
        h, nd = s.read(1).astype(np.float32), s.nodata
    valide = np.isfinite(h) & (h >= HAUTEUR_MIN) & (h <= HAUTEUR_MAX)
    if nd is not None:
        valide &= h != nd
    H, W = h.shape[0] // CASE * CASE, h.shape[1] // CASE * CASE
    h, valide = h[:H, :W], valide[:H, :W]
    cases_ok = valide.reshape(H // CASE, CASE, W // CASE, CASE).all(axis=(1, 3))
    fin = (h > SEUIL).reshape(H // CASE, CASE, W // CASE, CASE).mean(axis=(1, 3))
    h2 = h.reshape(H // 2, 2, W // 2, 2)
    grossier = {}
    for nom, agr in (("max", h2.max(axis=(1, 3))), ("moyenne", h2.mean(axis=(1, 3)))):
        grossier[nom] = (agr > SEUIL).reshape(H // CASE, CASE // 2, W // CASE, CASE // 2).mean(axis=(1, 3))
    return fin[cases_ok], {k: v[cases_ok] for k, v in grossier.items()}


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    sites = sorted(p for p in CHM_ONTARIO.iterdir() if p.is_dir())
    if not sites:
        sys.exit(f"Aucun CHM dans {CHM_ONTARIO} : l'étude ontarienne doit avoir tourné.")
    lignes, tout = [], {"fin": [], "max": [], "moyenne": []}
    for site in sites:
        f, g = [], {"max": [], "moyenne": []}
        for tuile in sorted(site.glob("*_CHM.tif")):
            a, b = labels_tuile(tuile)
            f.append(a)
            for k in g:
                g[k].append(b[k])
        f = np.concatenate(f)
        tout["fin"].append(f)
        ligne = dict(site_id=site.name, n=len(f), label_05m=float(f.mean()))
        for k in g:
            v = np.concatenate(g[k])
            tout[k].append(v)
            ligne |= {f"ecart_{k}": float((v - f).mean()), f"mae_{k}": float(np.abs(v - f).mean())}
        lignes.append(ligne)
        print(f"{site.name} : {len(f):,} cases, label {f.mean():.3f}, écart (max) {ligne['ecart_max']:+.4f}", flush=True)
    st = pd.DataFrame(lignes)
    st.to_csv(RESULTATS / "effet_resolution.csv", index=False)

    f = np.concatenate(tout["fin"])
    tranche = np.minimum(np.floor(f * 10), 9) / 10
    tr = []
    for lo in np.unique(tranche):
        m = tranche == lo
        tr.append(dict(tranche=round(float(lo), 1), part=float(m.mean()),
                       **{f"ecart_{k}": float((np.concatenate(tout[k])[m] - f[m]).mean()) for k in ("max", "moyenne")}))
    tr = pd.DataFrame(tr)
    tr.to_csv(RESULTATS / "effet_resolution_tranches.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(f):,} cases de 10 m dans {len(st)} sites")
    for k in ("max", "moyenne"):
        v = np.concatenate(tout[k])
        print(f"1 m ({k} 2 × 2) - 0,5 m : écart moyen {(v - f).mean():+.4f}, écart absolu moyen {np.abs(v - f).mean():.4f}")
    print("\npar tranche de label à 0,5 m :")
    print(tr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
