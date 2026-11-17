"""Étape 3 : pour chaque tir GEDI, calculer ce que disent nos données sur la même empreinte.

Pré-sélection (plus large que l'étape 4, pour pouvoir y tester des variantes) : qualité L2A et L2B
à 1, pas de dégradation, sensibilité ≥ 0,90, saison, écart ≤ 1 an avec le vol, rh98 et cover renseignés.

Pour chaque tir retenu, sur un disque de 25 m de diamètre centré sur le tir :
  - depuis les labels à 10 m : moyenne de cover_gt2m et cover_gt5m sur des points espacés de 0,5 m,
    donc pondérée par la surface de chaque case (moyenne simple, et gaussienne en variante) ;
  - depuis le CHM brut à 0,5 m : part des hauteurs > 2 m et > 5 m, centile 98 et maximum ;
  - test de position : labels > 2 m et centile 98 du CHM avec le disque décalé de -20 à +20 m
    vers l'est et vers le nord (points espacés de 1 m) ;
  - perturbation : une perte forestière Hansen (3 × 3 pixels autour du tir) entre l'année du vol
    et celle du tir.
Les conversions de coordonnées passent par pyproj ; aucune fonction du projet n'est utilisée.

Sortie : SORTIES/gedi_tirs_refs.parquet.
Usage : python -I independant-verification/gedi/03_references.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import rowcol

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (DECALAGES, ECART_ANNEES_MAX, HAUTEUR_MAX, HAUTEUR_MIN, PROJET, SORTIES,  # noqa: E402
                    dans_saison, disque, sites)

HANSEN = ("https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/"
          "Hansen_GFC-2025-v1.13_lossyear_{t}.tif")
SENSIBILITE_MIN = 0.90
VALIDE_MIN = 0.99   # part minimale de points CHM valides dans le disque
PAQUET = 500        # tirs traités à la fois, pour limiter la mémoire
warnings.filterwarnings("ignore", category=RuntimeWarning)   # centiles et maxima de lignes entièrement vides


def preselection(t):
    return ((t.q_l2a == 1) & (t.q_l2b == 1) & (t.degrade == 0) & (t.sensitivity >= SENSIBILITE_MIN)
            & dans_saison(t.date) & (t.ecart_annees.abs() <= ECART_ANNEES_MAX)
            & t.rh98.notna() & (t.cover >= 0))


def valeurs(a, tf, X, Y):
    """Valeur de la grille `a` sous chaque point (X, Y dans le système de la grille) ; NaN en dehors."""
    col = np.floor((X - tf.c) / tf.a).astype(np.int64)
    row = np.floor((Y - tf.f) / tf.e).astype(np.int64)
    ok = (row >= 0) & (row < a.shape[0]) & (col >= 0) & (col < a.shape[1])
    out = np.full(X.shape, np.nan, np.float32)
    out[ok] = a[row[ok], col[ok]]
    return out


def charger_chm(site_id, crs_label):
    """Tuiles CHM du site, hauteurs invalides mises à NaN, regroupées par système de coordonnées."""
    groupes = {}
    for p in sorted((PROJET / "data/raw/fri_chm" / site_id).glob("*_CHM.tif")):
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
            invalide = ~np.isfinite(a) | (a < HAUTEUR_MIN) | (a > HAUTEUR_MAX)
            if src.nodata is not None:
                invalide |= a == np.float32(src.nodata)
            a[invalide] = np.nan
            cle = src.crs.to_wkt()
            if cle not in groupes:
                groupes[cle] = (Transformer.from_crs(crs_label, src.crs, always_xy=True), [])
            groupes[cle][1].append((a, src.transform, src.bounds))
    return list(groupes.values())


def hauteurs(groupes, X, Y):
    """Hauteur du CHM sous chaque point (X, Y dans le système du label) ; NaN hors tuile ou invalide."""
    out = np.full(X.size, np.nan, np.float32)
    reste = np.ones(X.size, bool)
    for transformeur, tuiles in groupes:
        tx, ty = transformeur.transform(X.ravel(), Y.ravel())
        for a, tf, b in tuiles:
            m = reste & (tx >= b.left) & (tx < b.right) & (ty > b.bottom) & (ty <= b.top)
            if m.any():
                idx = np.flatnonzero(m)
                out[idx] = valeurs(a, tf, tx[idx], ty[idx])
                reste[idx] = False
    return out.reshape(X.shape)


def tuiles_hansen(s):
    """Télécharge (une fois) les tuiles Hansen « lossyear » de 10° × 10° qui couvrent les sites."""
    noms = set()
    for r in s.itertuples():
        for lat in (r.lat_min, r.lat_max):
            for lon in (r.lon_min, r.lon_max):
                haut, gauche = int(np.ceil(lat / 10) * 10), int(np.floor(lon / 10) * 10)
                noms.add(f"{haut:02d}N_{abs(gauche):03d}{'W' if gauche < 0 else 'E'}")
    dossier = SORTIES / "hansen"
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = []
    for nom in sorted(noms):
        p = dossier / f"lossyear_{nom}.tif"
        if not p.exists():
            tmp = p.with_suffix(".part")
            with requests.get(HANSEN.format(t=nom), stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for morceau in r.iter_content(1 << 20):
                        f.write(morceau)
            tmp.rename(p)
            print(f"tuile Hansen téléchargée : {p.name}", flush=True)
        chemins.append(p)
    return chemins


def perturbations(t, chemins):
    """Vrai si un pixel Hansen 3 × 3 autour du tir a perdu sa forêt entre l'année du vol et celle du tir."""
    lon, lat = t.lon.values, t.lat.values
    a_min = np.minimum(t.annee_vol.values, t.annee_tir.values) - 2000   # codes Hansen : 1 = 2001 … 25 = 2025
    a_max = np.maximum(t.annee_vol.values, t.annee_tir.values) - 2000
    perturbe, vu = np.zeros(len(t), bool), np.zeros(len(t), bool)
    for p in chemins:
        with rasterio.open(p) as src:
            b = src.bounds
            m = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
            if not m.any():
                continue
            rows, cols = (np.asarray(v) for v in rowcol(src.transform, lon[m], lat[m]))
            r0, c0 = max(rows.min() - 1, 0), max(cols.min() - 1, 0)
            r1, c1 = min(rows.max() + 2, src.height), min(cols.max() + 2, src.width)
            a = src.read(1, window=((r0, r1), (c0, c1)))
        for k, i in enumerate(np.flatnonzero(m)):
            rr, cc = rows[k] - r0, cols[k] - c0
            v = a[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2]
            perturbe[i] = bool(np.any((v > 0) & (v >= a_min[i]) & (v <= a_max[i])))
            vu[i] = True
    return perturbe, vu


def references_site(site_id, t, chemins_hansen):
    with rasterio.open(PROJET / "data/labels" / f"{site_id}_cover10m.tif") as src:
        lab2, lab5, tf, crs = src.read(1), src.read(2), src.transform, src.crs
    chm = charger_chm(site_id, crs)
    X0, Y0 = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(t.lon.values, t.lat.values)
    dx, dy, w = disque(0.5)
    dx1, dy1, _ = disque(1.0)
    n = len(t)
    r = {k: np.full(n, np.nan) for k in ("label_gt2", "label_gt5", "label_gt2_gauss", "label_gt5_gauss",
                                          "chm_gt2", "chm_gt5", "chm_gt2_gauss", "chm_p98", "chm_max", "chm_valide")}
    for ox in DECALAGES:
        for oy in DECALAGES:
            r[f"lab2_dx{ox}_dy{oy}"] = np.full(n, np.nan)
            r[f"p98_dx{ox}_dy{oy}"] = np.full(n, np.nan)

    for i in range(0, n, PAQUET):
        sl = slice(i, min(i + PAQUET, n))
        X, Y = X0[sl, None] + dx[None, :], Y0[sl, None] + dy[None, :]
        L2, L5 = valeurs(lab2, tf, X, Y), valeurs(lab5, tf, X, Y)
        ok = np.isfinite(L2).all(axis=1)
        r["label_gt2"][sl] = np.where(ok, L2.mean(axis=1), np.nan)
        r["label_gt5"][sl] = np.where(ok, L5.mean(axis=1), np.nan)
        r["label_gt2_gauss"][sl] = np.where(ok, (L2 * w).sum(axis=1), np.nan)
        r["label_gt5_gauss"][sl] = np.where(ok, (L5 * w).sum(axis=1), np.nan)

        H = hauteurs(chm, X, Y)
        valide = np.isfinite(H)
        part = valide.mean(axis=1)
        okh = part >= VALIDE_MIN
        g2, g5 = valide & (H > 2), valide & (H > 5)
        nv, wv = valide.sum(axis=1), (valide * w).sum(axis=1)
        r["chm_valide"][sl] = part
        r["chm_gt2"][sl] = np.where(okh, g2.sum(axis=1) / np.maximum(nv, 1), np.nan)
        r["chm_gt5"][sl] = np.where(okh, g5.sum(axis=1) / np.maximum(nv, 1), np.nan)
        r["chm_gt2_gauss"][sl] = np.where(okh, (g2 * w).sum(axis=1) / np.where(wv > 0, wv, 1), np.nan)
        r["chm_p98"][sl] = np.where(okh, np.nanpercentile(H, 98, axis=1), np.nan)
        r["chm_max"][sl] = np.where(okh, np.nanmax(H, axis=1), np.nan)

        for ox in DECALAGES:
            for oy in DECALAGES:
                Xs, Ys = X0[sl, None] + ox + dx1[None, :], Y0[sl, None] + oy + dy1[None, :]
                Ls = valeurs(lab2, tf, Xs, Ys)
                r[f"lab2_dx{ox}_dy{oy}"][sl] = np.where(np.isfinite(Ls).all(axis=1), Ls.mean(axis=1), np.nan)
                Hs = hauteurs(chm, Xs, Ys)
                oks = np.isfinite(Hs).mean(axis=1) >= VALIDE_MIN
                r[f"p98_dx{ox}_dy{oy}"][sl] = np.where(oks, np.nanpercentile(Hs, 98, axis=1), np.nan)

    perturbe, vu = perturbations(t, chemins_hansen)
    return pd.DataFrame(r).assign(perturbe=perturbe, hansen_lu=vu)


def main():
    s = sites()
    tirs = pd.read_parquet(SORTIES / "gedi_tirs.parquet")
    tirs = tirs.join(s.set_index("site_id")[["split", "annee_vol", "zone"]], on="site_id")
    tirs["annee_tir"] = tirs.date.dt.year
    tirs["ecart_annees"] = tirs.annee_tir - tirs.annee_vol
    sel = tirs[preselection(tirs)].reset_index(drop=True)
    print(f"{len(tirs):,} tirs extraits -> {len(sel):,} pré-sélectionnés dans {sel.site_id.nunique()} sites", flush=True)

    chemins = tuiles_hansen(s)
    morceaux = []
    for site_id, t in sel.groupby("site_id"):
        t = t.reset_index(drop=True)
        r = references_site(site_id, t, chemins)
        morceaux.append(pd.concat([t, r], axis=1))
        print(f"{site_id} : {len(t)} tirs | références labels complètes {r.label_gt2.notna().mean():.0%}, "
              f"CHM {r.chm_p98.notna().mean():.0%} | perturbés {r.perturbe.mean():.1%}", flush=True)

    out = pd.concat(morceaux, ignore_index=True)
    out.to_parquet(SORTIES / "gedi_tirs_refs.parquet", index=False)
    print(f"\n{len(out):,} tirs -> {SORTIES / 'gedi_tirs_refs.parquet'}")
    print(f"tirs non couverts par Hansen : {(~out.hansen_lu).sum()} (attendu : 0)")


if __name__ == "__main__":
    main()
