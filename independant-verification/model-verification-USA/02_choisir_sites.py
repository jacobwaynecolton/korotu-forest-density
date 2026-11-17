"""Étape 2 : choisir le bloc de 5 × 5 km de chaque survol NEON retenu (commun.VOLS), selon une règle fixée d'avance.

  - Bloc : parmi les blocs de 5 × 5 tuiles de 1 km dont le nuage de points classé existe, celui dont le centre est
    le plus proche du centre de l'ensemble des tuiles du survol. On ne regarde pas la forêt pour choisir.
  - Zone UTM : lue dans le système de coordonnées d'une tuile CHM NEON du même survol (EPSG:326zz,
    celui de Sentinel-2).
  - Dates de vol : lues dans le nom des fichiers de trajectoire (SBET) livrés avec les données ; à défaut,
    le mois entier du survol (signalé).
  - Densité : médiane des premiers retours par m² des 25 tuiles, d'après leurs en-têtes.

La sélection n'est faite qu'une fois : si sites.csv existe, le script s'arrête (--refaire pour l'écraser).
Sorties : model-verification-USA/sites.csv et tuiles.csv (les 25 nuages de points de chaque site).
Usage : python -I independant-verification/model-verification-USA/02_choisir_sites.py [--refaire]
"""
import calendar
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (BLOC, ICI, NOM_TUILE, PRODUIT_CHM, PRODUIT_NUAGES, TUILE, VOLS, api,  # noqa: E402
                    coin_tuile, densite_entete)


def choisir_bloc(coins):
    """Coin sud-ouest du bloc de BLOC × BLOC tuiles complet le plus proche du centre des tuiles."""
    presentes = set(coins)
    centre = np.mean(coins, axis=0) + TUILE / 2
    meilleur, distance = None, np.inf
    for e0, n0 in coins:
        if all((e0 + i * TUILE, n0 + j * TUILE) in presentes for i in range(BLOC) for j in range(BLOC)):
            d = np.hypot(e0 + BLOC * TUILE / 2 - centre[0], n0 + BLOC * TUILE / 2 - centre[1])
            if d < distance:
                meilleur, distance = (e0, n0), d
    return meilleur


def dans_bloc(nom, coin):
    e, n = coin_tuile(nom)
    return 0 <= e - coin[0] < BLOC * TUILE and 0 <= n - coin[1] < BLOC * TUILE


def dates_vol(fichiers, mois):
    """Début et fin des vols d'après les noms de trajectoire (2024062513_..._SBET_QAQC.pdf)."""
    heures = sorted(datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
                    for f in fichiers if (m := re.match(r"(\d{10})_.*SBET", f["name"])))
    if heures:
        return heures[0], heures[-1], True
    an, mo = int(mois[:4]), int(mois[5:])
    return (datetime(an, mo, 1, tzinfo=timezone.utc),
            datetime(an, mo, calendar.monthrange(an, mo)[1], 23, tzinfo=timezone.utc), False)


def main(refaire=False):
    if (ICI / "sites.csv").exists() and not refaire:
        sys.exit(f"{ICI / 'sites.csv'} existe déjà : la sélection est faite (--refaire pour la recommencer).")
    lignes, tuiles = [], []
    for site, (groupe, mois) in VOLS.items():
        d = api(f"data/{PRODUIT_NUAGES}/{site}/{mois}")
        laz = [f for f in d["files"] if NOM_TUILE.search(f["name"])]
        coin = choisir_bloc([coin_tuile(f["name"]) for f in laz])
        if coin is None:
            print(f"{site} {mois} : aucun bloc complet de {BLOC} × {BLOC} tuiles, site écarté", flush=True)
            continue
        bloc = [f for f in laz if dans_bloc(f["name"], coin)]
        chm = api(f"data/{PRODUIT_CHM}/{site}/{mois}")
        tuile_chm = next(f for f in chm["files"] if f["name"].endswith("_CHM.tif") and dans_bloc(f["name"], coin))
        with rasterio.open(tuile_chm["url"]) as s:   # lecture de l'en-tête seulement
            crs = s.crs
        epsg = crs.to_epsg()
        if not (epsg and 32601 <= epsg <= 32660):
            sys.exit(f"{site} : système de coordonnées inattendu ({crs}), à examiner.")
        debut, fin, dates_sures = dates_vol(d["files"] + chm["files"], mois)
        densite = float(np.median([densite_entete(f["url"]) for f in bloc]))
        lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(
            coin[0] + BLOC * TUILE / 2, coin[1] + BLOC * TUILE / 2)
        info = api(f"sites/{site}", avec_jeton=False)
        site_id = f"{site}_{mois[:4]}"
        lignes.append(dict(site_id=site_id, site_neon=site, groupe=groupe, etat=info["stateCode"],
                           domaine=info["domainCode"], zone=epsg - 32600, year=int(mois[:4]), mois=mois,
                           version=d["release"], e0_m=coin[0], n0_m=coin[1], lon=lon, lat=lat,
                           flight_start=debut.isoformat(), flight_end=fin.isoformat(),
                           dates_vol_sures=dates_sures, densite_entetes=round(densite, 1)))
        tuiles += [dict(site_id=site_id, site_neon=site, mois=mois, name=f["name"], md5=f.get("md5"), size=f["size"])
                   for f in bloc]
        print(f"groupe {groupe} · {site_id} ({info['stateCode']}) : {d['release']}, vols {debut:%Y-%m-%d}.."
              f"{fin:%Y-%m-%d}{'' if dates_sures else ' (mois entier)'}, bloc E{coin[0]} N{coin[1]} zone {epsg - 32600}, "
              f"{densite:.0f} premiers retours/m², {sum(f['size'] for f in bloc) / 1e9:.1f} Go", flush=True)
    pd.DataFrame(lignes).to_csv(ICI / "sites.csv", index=False)
    pd.DataFrame(tuiles).to_csv(ICI / "tuiles.csv", index=False)
    print(f"\n{len(lignes)} sites, {len(tuiles)} nuages de points ({sum(t['size'] for t in tuiles) / 1e9:.0f} Go) "
          f"-> {ICI / 'sites.csv'}")


if __name__ == "__main__":
    main("--refaire" in sys.argv[1:])
