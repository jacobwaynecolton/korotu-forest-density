"""Étape 2 : extraire les tirs GEDI qui tombent dans les sites, en lisant à distance seulement les variables utiles.

Pour chaque passage listé à l'étape 1 :
  - fichier L2B : position, date, couvert total (`cover`), profil de couvert (`cover_z`, tranches de 5 m),
    drapeaux de qualité (version 3), sensibilité, élévation du soleil, saison, classe WorldCover ;
  - fichier L2A : hauteurs relatives rh50, rh98 et rh100, appariées tir par tir grâce à `shot_number`.
Les fichiers restent dans le cloud de la NASA : on n'en lit que des morceaux, avec un cache par blocs
de 4 Mo (environ 10 fois plus rapide que le réglage par défaut).

Reprise possible : chaque passage traité donne SORTIES/tirs/<clé>.parquet, et les passages déjà faits
sont sautés. Sortie finale : SORTIES/gedi_tirs.parquet.
Usage : python -I independant-verification/gedi/02_extraire_tirs.py [nb_processus]
"""
import logging
import multiprocessing
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import aiohttp
import earthaccess
import fsspec
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import EPOQUE_GEDI, ICI, PLEINE_PUISSANCE, SORTIES, date_fichier, sites  # noqa: E402

DOSSIER = SORTIES / "tirs"
MARGE = 0.001    # degré (~100 m) autour de l'emprise de chaque site
CACHE = dict(block_size=4 * 2 ** 20, cache_type="blockcache")
# Sans délai maximal, une requête sans réponse bloque un processus indéfiniment (vu en test) ;
# avec, elle échoue et le passage est réessayé.
DELAIS = aiohttp.ClientTimeout(total=120, sock_connect=30, sock_read=60)
# Ces délais n'ont pas suffi : lors de la première exécution complète, 2 passages sont restés bloqués
# 9 heures sans erreur. Une alarme du système borne donc aussi la durée totale d'un passage
# (1 à 2 minutes en temps normal).
DUREE_MAX_PASSAGE = 600   # s


class DureeDepassee(Exception):
    pass


def _alarme(signum, frame):
    raise DureeDepassee(f"passage interrompu au bout de {DUREE_MAX_PASSAGE} s")
VARIABLES_L2B = {   # nom de colonne : chemin dans le faisceau
    "shot_number": "shot_number", "delta_time": "delta_time", "cover": "cover", "pai": "pai",
    "sensitivity": "sensitivity", "q_l2a": "l2a_quality_flag_rel3", "q_l2b": "l2b_quality_flag_rel3",
    "degrade": "geolocation/degrade_flag", "soleil": "geolocation/solar_elevation",
    "leaf_off": "land_cover_data/leaf_off_flag", "worldcover": "land_cover_data/worldcover_class",
}
COLONNES = (["cle", "faisceau", "pleine_puissance", "site_id", "lat", "lon"] + list(VARIABLES_L2B)
            + ["cover_z0", "cover_z5", "cover_z10", "dz", "rh50", "rh98", "rh100", "date"])

_fs = None


def initialiser(jeton):
    """Session HTTPS de chaque processus : jeton NASA obtenu une seule fois par le processus principal."""
    global _fs
    _fs = fsspec.filesystem("https", skip_instance_cache=True, client_kwargs=dict(
        headers={"Authorization": f"Bearer {jeton}"}, trust_env=False, timeout=DELAIS))
    signal.signal(signal.SIGALRM, _alarme)


def lire_l2b(url, emprises, cle):
    morceaux = []
    with _fs.open(url, **CACHE) as f, h5py.File(f, "r") as h:
        for nom in sorted(k for k in h if k.startswith("BEAM")):
            b = h[nom]
            lat, lon = b["lat_lowestmode"][:], b["lon_lowestmode"][:]
            site = np.full(lat.shape, "", dtype=object)
            for sid, x0, y0, x1, y1 in emprises:
                site[(lon >= x0 - MARGE) & (lon <= x1 + MARGE) & (lat >= y0 - MARGE) & (lat <= y1 + MARGE)] = sid
            idx = np.flatnonzero(site != "")
            if idx.size == 0:
                continue
            i0, i1 = idx.min(), idx.max() + 1   # on lit un seul bloc contigu, puis on garde les tirs utiles
            garder = idx - i0
            d = dict(cle=cle, faisceau=nom, pleine_puissance=nom in PLEINE_PUISSANCE,
                     site_id=site[idx], lat=lat[idx], lon=lon[idx])
            for col, chemin in VARIABLES_L2B.items():
                d[col] = b[chemin][i0:i1][garder]
            cz = b["cover_z"][i0:i1, :3][garder]
            d["cover_z0"], d["cover_z5"], d["cover_z10"] = cz[:, 0], cz[:, 1], cz[:, 2]
            d["dz"] = float(b["ancillary/dz"][0])
            morceaux.append(pd.DataFrame(d))
    return pd.concat(morceaux, ignore_index=True) if morceaux else None


def ajouter_hauteurs(df, url):
    """Ajoute rh50, rh98 et rh100 du fichier L2A, tir par tir (jointure sur shot_number)."""
    df["rh50"] = df["rh98"] = df["rh100"] = np.nan
    with _fs.open(url, **CACHE) as f, h5py.File(f, "r") as h:
        for nom, g in df.groupby("faisceau"):
            sn = h[nom]["shot_number"][:]
            ordre = np.argsort(sn)
            pos = ordre[np.clip(np.searchsorted(sn[ordre], g.shot_number.values), 0, sn.size - 1)]
            ok = sn[pos] == g.shot_number.values
            if not ok.any():
                continue
            j0, j1 = pos[ok].min(), pos[ok].max() + 1
            rh = h[nom]["rh"][j0:j1, :]
            lignes = g.index[ok]
            df.loc[lignes, "rh50"] = rh[pos[ok] - j0, 50]
            df.loc[lignes, "rh98"] = rh[pos[ok] - j0, 98]
            df.loc[lignes, "rh100"] = rh[pos[ok] - j0, 100]
    return df


def traiter(cle, url_l2b, url_l2a, emprises, essais=3):
    """Extrait un passage. Renvoie (clé, nombre de tirs, durée en s, message d'erreur ou None)."""
    t0 = time.time()
    for essai in range(1, essais + 1):
        signal.alarm(DUREE_MAX_PASSAGE)
        try:
            df = lire_l2b(url_l2b, emprises, cle)
            if df is None:
                df = pd.DataFrame(columns=COLONNES)
            else:
                if isinstance(url_l2a, str):
                    df = ajouter_hauteurs(df, url_l2a)
                else:
                    df["rh50"] = df["rh98"] = df["rh100"] = np.nan
                df["date"] = EPOQUE_GEDI + pd.to_timedelta(df["delta_time"], unit="s")
            df[COLONNES].to_parquet(DOSSIER / f"{cle}.parquet", index=False)
            return cle, len(df), time.time() - t0, None
        except Exception as e:  # coupure réseau, fichier momentanément inaccessible, durée dépassée…
            signal.alarm(0)
            if essai == essais:
                return cle, 0, time.time() - t0, f"{type(e).__name__}: {e}"
            time.sleep(15 * essai)
        finally:
            signal.alarm(0)


def main(nb_processus=6):
    DOSSIER.mkdir(parents=True, exist_ok=True)
    fichiers = pd.read_csv(ICI / "fichiers.csv")
    fichiers = fichiers[fichiers.url_l2b.notna()]
    s = sites().set_index("site_id")
    taches = []
    for cle, g in fichiers.groupby("cle"):
        if (DOSSIER / f"{cle}.parquet").exists():
            continue
        emprises = [(sid, *s.loc[sid, ["lon_min", "lat_min", "lon_max", "lat_max"]]) for sid in g.site_id.unique()]
        taches.append((cle, g.url_l2b.iloc[0], g.url_l2a.iloc[0], emprises))
    total = fichiers.cle.nunique()
    print(f"{total} passages, {total - len(taches)} déjà faits, {len(taches)} à traiter avec {nb_processus} processus",
          flush=True)

    logging.getLogger("earthaccess").setLevel(logging.WARNING)
    jeton = earthaccess.login(strategy="netrc").token["access_token"]   # identifiants lus dans ~/.netrc
    echecs = []
    # « spawn » : processus créés à neuf. Avec « fork » (le défaut sous Linux), les processus héritaient
    # d'un état réseau défaillant et certaines lectures devenaient ~100 fois plus lentes (vu en test).
    contexte = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(nb_processus, mp_context=contexte, initializer=initialiser, initargs=(jeton,)) as ex:
        futurs = [ex.submit(traiter, *t) for t in taches]
        for i, fut in enumerate(as_completed(futurs), 1):
            cle, n, duree, err = fut.result()
            if err:
                echecs.append((cle, err))
                print(f"[{i}/{len(taches)}] ÉCHEC {cle} : {err}", flush=True)
            elif i % 10 == 0 or i == len(taches):
                print(f"[{i}/{len(taches)}] {cle} : {n} tirs ({duree:.0f} s)", flush=True)

    morceaux = [m for m in (pd.read_parquet(p) for p in sorted(DOSSIER.glob("*.parquet"))) if len(m)]
    if not morceaux:
        sys.exit("Aucun tir extrait : vérifier les échecs ci-dessus.")
    tirs = pd.concat(morceaux, ignore_index=True)
    tirs = tirs.merge(fichiers[["cle", "site_id", "ecart_annees"]].drop_duplicates(), on=["cle", "site_id"], how="left")
    debut_fichier = tirs.cle.map({c: date_fichier(c) for c in tirs.cle.unique()})
    decalage_h = (tirs["date"] - debut_fichier).dt.total_seconds().abs() / 3600
    tirs.to_parquet(SORTIES / "gedi_tirs.parquet", index=False)
    print(f"\n{len(tirs):,} tirs dans {tirs.site_id.nunique()} sites -> {SORTIES / 'gedi_tirs.parquet'}")
    print(f"contrôle des dates : {(decalage_h > 3).sum()} tirs à plus de 3 h du début de leur fichier (attendu : 0)")
    print(f"hauteurs L2A manquantes : {tirs.rh98.isna().sum():,} tirs")
    print("tirs par écart d'années avec le vol lidar :", tirs.ecart_annees.value_counts().sort_index().to_dict())
    if echecs:
        print(f"\n{len(echecs)} passages en échec : relancer l'étape, seuls ceux-là seront refaits.")
        sys.exit(1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
