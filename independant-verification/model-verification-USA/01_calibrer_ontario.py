"""Étape 1 (point d'arrêt) : nos CHM construits sur les points redonnent-ils les labels du projet en Ontario ?

Pour la tuile centrale de chacun des 15 nouveaux sites ontariens (étude ../model-verification), on compare
les labels à 10 m (part des pixels de 0,5 m au-dessus de 2 m, dans la grille de la tuile) du CHM officiel
(DSM - DEM du fournisseur) à ceux de nos CHM, construits sur le nuage de points de la même tuile :
  - « tous_points » : tous les points, sans choix du jour ni éclaircissement ;
  - « methode »     : la méthode appliquée aux sites NEON (nuages.preparer puis nuages.chm) ;
  - « densite_30 »  : la même, ramenée à 30 premiers retours par m², la densité des vols NEON les moins denses.
Écart = notre label - label officiel, sur les cases valides des deux côtés.

Point d'arrêt fixé d'avance : sur les tuiles survolées en un seul jour, l'écart absolu moyen de la
« methode » doit rester ≤ 0,02. Sinon le script s'arrête en erreur, et la chaîne Slurm avec lui.
Sorties : resultats/calibration_ontario.csv et resultats/calibration.json
Usage : python -I independant-verification/model-verification-USA/01_calibrer_ontario.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuages  # noqa: E402
from commun import CALIBRATION, CHM_ONTARIO, ONTARIO, RESULTATS, telecharger  # noqa: E402

ECART_MAX = 0.02
URL_DEM = "https://download.fri.mnrf.gov.on.ca/api/api/Download/geohub_compressed_dem/utm{zone}/{tuile}_DEM.tif"


def lisible(chemin):
    try:
        with rasterio.open(chemin):
            return True
    except rasterio.errors.RasterioIOError:
        return False


def laz_complet(chemin):
    try:
        with nuages.laspy.open(chemin) as f:
            return f.header.point_count > 0
    except Exception:
        return False


def tuiles_centrales():
    t = pd.read_csv(ONTARIO / "tuiles.csv").sort_values(["site_id", "Tilename"])
    return t.groupby("site_id").nth(12).reset_index()   # 13e des 25 tuiles : le centre du bloc


def ecarts(h, officiel, ok_off):
    lab, ok = nuages.labels_10m(h)
    m = ok & ok_off
    return float((lab - officiel)[m].mean()), float(np.abs(lab - officiel)[m].mean()), float(m.mean())


def main():
    RESULTATS.mkdir(parents=True, exist_ok=True)
    (CALIBRATION / "laz").mkdir(parents=True, exist_ok=True)
    lignes = []
    for r in tuiles_centrales().itertuples():
        laz = CALIBRATION / "laz" / f"{r.Tilename}.copc.laz"
        dem = CALIBRATION / f"{r.Tilename}_DEM.tif"
        etats = [telecharger(r.Download_LAZ, laz, laz_complet),
                 telecharger(URL_DEM.format(zone=r.Tilename[4:6], tuile=r.Tilename), dem, lisible)]
        if any(e.startswith("ÉCHEC") for e in etats):
            print(f"{r.site_id} {r.Tilename} : {etats}", flush=True)
            continue
        with rasterio.open(CHM_ONTARIO / r.site_id / f"{r.Tilename}_CHM.tif") as s:
            off, tr, forme = s.read(1).astype(np.float32), s.transform, s.shape
            off[off == s.nodata] = np.nan
        officiel, ok_off = nuages.labels_10m(off)
        p = nuages.lire(laz)
        q, info = nuages.preparer(p)
        q30, _ = nuages.preparer(p, cible=30)
        ligne = dict(site_id=r.site_id, tuile=r.Tilename, label_officiel=float(officiel[ok_off].mean()), **info)
        for nom, h in (("tous_points", nuages.chm(p, tr, forme)), ("methode", nuages.chm(q, tr, forme)),
                       ("densite_30", nuages.chm(q30, tr, forme))):
            ligne[f"ecart_{nom}"], ligne[f"ecart_abs_{nom}"], ligne[f"part_cases_{nom}"] = ecarts(h, officiel, ok_off)
        lignes.append(ligne)
        print(f"{r.site_id} {r.Tilename} : {info['jours']} jour(s), {info['densite_avant']:.0f} -> "
              f"{info['densite_apres']:.0f} pts/m², label {ligne['label_officiel']:.3f}, écart méthode "
              f"{ligne['ecart_methode']:+.4f}, tous points {ligne['ecart_tous_points']:+.4f}, à 30 pts/m² "
              f"{ligne['ecart_densite_30']:+.4f}", flush=True)
    t = pd.DataFrame(lignes)
    t.to_csv(RESULTATS / "calibration_ontario.csv", index=False)

    un_jour = t[t.jours == 1]
    critere = float(un_jour.ecart_abs_methode.mean())
    sortie = dict(n_tuiles=len(t), n_un_jour=len(un_jour), ecart_abs_methode_un_jour=critere,
                  ecart_methode_un_jour=float(un_jour.ecart_methode.mean()),
                  ecart_abs_methode_toutes=float(t.ecart_abs_methode.mean()),
                  ecart_abs_tous_points_toutes=float(t.ecart_abs_tous_points.mean()),
                  effet_densite_30=float((t.ecart_densite_30 - t.ecart_methode).mean()),
                  effet_densite_30_min_max=[float((t.ecart_densite_30 - t.ecart_methode).min()),
                                            float((t.ecart_densite_30 - t.ecart_methode).max())],
                  seuil=ECART_MAX, ok=bool(critere <= ECART_MAX))
    (RESULTATS / "calibration.json").write_text(json.dumps(sortie, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print("\n" + t.drop(columns=[c for c in t.columns if c.startswith("part_")]).round(4).to_string(index=False))
    print(f"\ntuiles survolées en un seul jour : {len(un_jour)}/{len(t)} ; écart absolu moyen de la méthode "
          f"{critere:.4f} (seuil {ECART_MAX}) ; sur toutes les tuiles {sortie['ecart_abs_methode_toutes']:.4f}")
    print(f"effet d'une densité de 30 pts/m² : {sortie['effet_densite_30']:+.4f} en moyenne "
          f"({sortie['effet_densite_30_min_max'][0]:+.4f} à {sortie['effet_densite_30_min_max'][1]:+.4f})")
    if not sortie["ok"]:
        sys.exit("POINT D'ARRÊT : la méthode ne redonne pas les labels ontariens, on n'applique pas la suite.")
    print("Point d'arrêt franchi : la méthode peut être appliquée aux sites NEON.")


if __name__ == "__main__":
    main()
