"""Étape 3 : pour chaque tuile NEON, télécharger le nuage de points, en tirer le CHM à 0,5 m, puis l'effacer.

Le quota du volume de travail (100 Go) ne permet pas de garder les 91 Go de nuages de points : la première
exécution s'est arrêtée sans message en atteignant le quota (ni les fichiers ni le journal ne pouvaient plus
être écrits). Chaque nuage est donc effacé dès que son CHM (environ 10 Mo) est écrit ; au plus un nuage par
processus est présent à la fois.

  - Téléchargement : liste des fichiers redemandée à l'API site par site (les liens signés expirent), curl
    avec délais maximaux (moins de 1 Mo/s pendant 2 minutes, ou plus de 30 minutes : transfert recommencé,
    3 essais), somme md5 comparée à celle de NEON. Un nuage déjà présent et intact est utilisé tel quel.
  - CHM : méthode calibrée sur l'Ontario (nuages.py) : un seul jour de vol, densité ramenée à 44 premiers
    retours par m² si elle est plus forte, sol tiré des points de la classe 2, hauteur maximale par case de
    0,5 m. Grille de la tuile NEON (1 km, coin sud-ouest dans le nom), système UTM du site (EPSG:326zz).
Les CHM prennent la place de ceux de l'Ontario : l'étape 4 en fait les labels avec le script du projet.

Reprise possible : une tuile dont le CHM et ses informations existent est sautée.
Sorties : SORTIES/chm/<site>/<tuile>_CHM.tif (+ .json) et model-verification-USA/chm_stats.csv
Usage : python -I independant-verification/model-verification-USA/03_nuages_chm.py [nb_processus]
"""
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuages  # noqa: E402
from commun import CHM, ICI, NUAGES, PRODUIT_NUAGES, TUILE, api, coin_tuile, nouveaux_sites  # noqa: E402

PAS = 0.5
DUREE_MAX = 1800   # s par téléchargement
CURL = ["curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "30",
        "--speed-limit", "1000000", "--speed-time", "120"]


def md5(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for morceau in iter(lambda: f.read(1 << 22), b""):
            h.update(morceau)
    return h.hexdigest()


def intact(chemin, somme):
    if isinstance(somme, str) and somme:
        return md5(chemin) == somme
    try:
        with laspy.open(chemin) as f:
            return f.header.point_count > 0
    except Exception:
        return False


def telecharger(url, chemin, somme, essais=3):
    """Renvoie l'état : « déjà là », « ok » ou « ÉCHEC … »."""
    if chemin.exists() and intact(chemin, somme):
        return "déjà là"
    partiel = chemin.with_name(chemin.name + ".part")
    for essai in range(1, essais + 1):
        try:
            subprocess.run(CURL + ["--output", str(partiel), url], check=True, timeout=DUREE_MAX,
                           capture_output=True, text=True)
            if not intact(partiel, somme):
                raise IOError("somme md5 différente de celle annoncée par NEON")
            partiel.rename(chemin)
            return "ok"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IOError) as e:
            partiel.unlink(missing_ok=True)
            if essai == essais:
                detail = getattr(e, "stderr", None) or str(e)
                return f"ÉCHEC après {essais} essais : {type(e).__name__} {str(detail).strip()[:200]}"
            time.sleep(15 * essai)


def traiter(url, site_id, nom, somme, zone):
    sortie = CHM / site_id / nom.replace(".laz", "_CHM.tif")
    infos = sortie.with_suffix(".json")
    if sortie.exists() and infos.exists():
        return json.loads(infos.read_text()) | dict(etat="déjà fait")
    t0 = time.time()
    laz = NUAGES / site_id / nom
    etat = telecharger(url, laz, somme)
    if etat.startswith("ÉCHEC"):
        return dict(site_id=site_id, tuile=nom, etat=etat)
    e0, n0 = coin_tuile(nom)
    transform, forme = from_origin(e0, n0 + TUILE, PAS, PAS), (int(TUILE / PAS), int(TUILE / PAS))
    p, info = nuages.preparer(nuages.lire(laz))
    h = nuages.chm(p, transform, forme)
    prof = dict(driver="GTiff", width=forme[1], height=forme[0], count=1, dtype="float32",
                crs=f"EPSG:{32600 + zone}", transform=transform, nodata=np.nan, compress="deflate",
                predictor=3, tiled=True, blockxsize=256, blockysize=256)
    partiel = sortie.with_name(sortie.name + ".part")
    with rasterio.open(partiel, "w", **prof) as d:
        d.write(h, 1)
        d.update_tags(source=f"NEON {nom}", methode="hauteur maximale des points par case de 0,5 m (nuages.py)")
    partiel.rename(sortie)
    info |= dict(site_id=site_id, tuile=nom, part_vide=float(np.isnan(h).mean()))
    infos.write_text(json.dumps(info))
    laz.unlink()   # le quota ne permet pas de garder les nuages de points
    return info | dict(etat=f"{etat}, CHM écrit, nuage effacé", secondes=time.time() - t0)


def main(nb=2):
    sites = nouveaux_sites().set_index("site_id")
    tuiles = pd.read_csv(ICI / "tuiles.csv")
    faits, echecs = 0, []
    for (site_id, site_neon, mois), g in tuiles.groupby(["site_id", "site_neon", "mois"], sort=False):
        (NUAGES / site_id).mkdir(parents=True, exist_ok=True)
        (CHM / site_id).mkdir(parents=True, exist_ok=True)
        restantes = [r for r in g.itertuples()
                     if not (CHM / site_id / r.name.replace(".laz", "_CHM.json")).exists()]
        urls = {f["name"]: f["url"] for f in api(f"data/{PRODUIT_NUAGES}/{site_neon}/{mois}")["files"]} if restantes else {}
        zone = int(sites.loc[site_id, "zone"])
        with ProcessPoolExecutor(nb) as ex:
            futurs = [ex.submit(traiter, urls.get(r.name), site_id, r.name, r.md5, zone) for r in g.itertuples()]
            for fut in futurs:
                faits += 1
                try:
                    info = fut.result()
                except Exception as e:   # nuage illisible, mémoire : on signale et on continue
                    info = dict(etat=f"ÉCHEC {type(e).__name__}: {e}")
                if info["etat"].startswith("ÉCHEC"):
                    echecs.append(site_id)
                    print(f"[{faits}/{len(tuiles)}] {site_id} : {info['etat']}", flush=True)
                    continue
                print(f"[{faits}/{len(tuiles)}] {site_id}/{info['tuile']} : {info['etat']} ; {info['jours']} jour(s) "
                      f"({info['part_jour_principal']:.0%} des points gardés), {info['densite_avant']:.0f} -> "
                      f"{info['densite_apres']:.0f} pts/m², {info['part_vide']:.1%} de cases vides"
                      + (f", {info['secondes']:.0f} s" if "secondes" in info else ""), flush=True)

    st = pd.DataFrame([json.loads(p.read_text()) for p in sorted(CHM.glob("*/*_CHM.json"))])
    st.to_csv(ICI / "chm_stats.csv", index=False)
    if len(st):
        print("\n" + st.groupby("site_id").agg(tuiles=("tuile", "size"), plusieurs_jours=("jours", lambda j: int((j > 1).sum())),
                                              densite_avant=("densite_avant", "median"),
                                              densite_apres=("densite_apres", "median"),
                                              cases_vides=("part_vide", "mean")).round(3).to_string())
    print(f"\n{len(tuiles) - len(echecs)}/{len(tuiles)} tuiles traitées, {len(echecs)} en échec")
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
