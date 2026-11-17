"""Étape 1 : lister les fichiers GEDI (L2A et L2B, version 003) qui survolent chaque site.

Pour chaque site, on cherche les passages du 1er juin au 15 septembre de l'année du vol lidar,
de l'année d'avant et de l'année d'après. La recherche passe par le catalogue de la NASA (CMR)
et ne demande pas de connexion.

Sortie : independant-verification/gedi/fichiers.csv, une ligne par couple (site, passage GEDI).
Usage : python -I independant-verification/gedi/01_lister_fichiers.py
"""
import logging
import sys
from pathlib import Path

import earthaccess
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import (DEBUT_SAISON, ECART_ANNEES_MAX, FIN_SAISON, ICI, VERSION,  # noqa: E402
                    cle_fichier, date_fichier, sites)

logging.getLogger("earthaccess").setLevel(logging.WARNING)


def chercher(produit, site, annee):
    """{clé du passage: lien https du fichier .h5} pour un produit, un site et une saison."""
    debut = f"{annee}-{DEBUT_SAISON[0]:02d}-{DEBUT_SAISON[1]:02d}"
    fin = f"{annee}-{FIN_SAISON[0]:02d}-{FIN_SAISON[1]:02d}T23:59:59"
    resultats = earthaccess.search_data(short_name=produit, version=VERSION, temporal=(debut, fin),
                                        bounding_box=(site.lon_min, site.lat_min, site.lon_max, site.lat_max))
    liens = {}
    for g in resultats:
        for url in g.data_links(access="external"):
            if url.endswith(".h5"):
                liens[cle_fichier(url.rsplit("/", 1)[-1])] = url
    return liens


def main():
    lignes, s = [], sites()
    for site in s.itertuples():
        n = 0
        for ecart in range(-ECART_ANNEES_MAX, ECART_ANNEES_MAX + 1):
            annee = site.annee_vol + ecart
            l2a, l2b = chercher("GEDI02_A", site, annee), chercher("GEDI02_B", site, annee)
            for cle in sorted(set(l2a) | set(l2b)):
                lignes.append(dict(site_id=site.site_id, split=site.split, annee_vol=site.annee_vol,
                                   ecart_annees=ecart, cle=cle, date=date_fichier(cle).isoformat(),
                                   url_l2a=l2a.get(cle), url_l2b=l2b.get(cle)))
                n += 1
        print(f"{site.site_id} ({site.split}, vol {site.annee_vol}) : {n} passages", flush=True)

    df = pd.DataFrame(lignes)
    df.to_csv(ICI / "fichiers.csv", index=False)
    incomplets = df[df.url_l2a.isna() | df.url_l2b.isna()].cle.unique()
    par_site = df.groupby("site_id").size()
    sans = sorted(set(s.site_id) - set(par_site.index))
    print(f"\n{df.cle.nunique()} passages GEDI distincts | médiane {par_site.median():.0f} par site "
          f"(min {par_site.min()}, max {par_site.max()})")
    print(f"sites sans aucun passage : {sans or 'aucun'}")
    print(f"passages où il manque L2A ou L2B : {len(incomplets)} {list(incomplets)[:5]}")
    print(f"écrit : {ICI / 'fichiers.csv'}")


if __name__ == "__main__":
    main()
