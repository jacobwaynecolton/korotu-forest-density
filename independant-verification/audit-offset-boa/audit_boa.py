"""Audit : pour chaque scène Sentinel-2 utilisée, quelle branche de make_pairs.boa_offset_applied décide.

Scènes : 181 du projet, 51 des nouveaux sites ontariens, 22 NEON. Pour chacune, le catalogue Earth Search
(sentinel-2-l2a) donne la version de traitement, le drapeau earthsearch:boa_offset_applied et le décalage des
métadonnées ; le test des pixels (1er percentile du bleu) n'est fait que si les deux premiers ne tranchent pas.
Sortie : audit_boa.csv (à côté du script).
Usage : python -I independant-verification/audit-offset-boa/audit_boa.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from pystac_client import Client
from rasterio.enums import Resampling
P = Path("/h/460/tsalin/projects/MLOps/korotu-forest-density")
sys.path.insert(0, str(P / "scripts"))
from make_pairs import STAC, read_window  # noqa: E402
W = Path("/w/20261/tsalin/MLOps/independant_verification")
jeux = [("projet", P / "data/manifest/s2_scenes.csv", lambda s: P / "data/labels" / f"{s}_cover10m.tif"),
        ("ontario_nouveaux", P / "independant-verification/model-verification/scenes_s2.csv", lambda s: W / "modele/labels" / f"{s}_cover10m.tif"),
        ("neon", P / "independant-verification/model-verification-USA/scenes_s2.csv", lambda s: W / "modele_usa/labels" / f"{s}_cover10m.tif")]
c = Client.open(STAC)
rows = []
for jeu, csv, lab in jeux:
    sc = pd.read_csv(csv)
    items = {i.id: i for i in c.search(ids=sc.scene.unique().tolist(), collections=["sentinel-2-l2a"]).items()}
    for r in sc.itertuples():
        it = items.get(r.scene)
        if it is None:
            rows.append(dict(jeu=jeu, site_id=r.site_id, scene=r.scene, branche="introuvable")); continue
        flag = it.properties.get("earthsearch:boa_offset_applied")
        rb = it.assets["blue"].extra_fields["raster:bands"][0]
        off = rb.get("offset", 0.0)
        d = dict(jeu=jeu, site_id=r.site_id, scene=r.scene, date=r.date, baseline=it.properties.get("s2:processing_baseline"),
                 drapeau=flag, offset_meta=off)
        if flag:
            d.update(branche="drapeau vrai", retire=True)
        elif not off:
            d.update(branche="pas d'offset", retire=True)
        else:
            with rasterio.open(lab(r.site_id)) as s:
                b, shp = s.bounds, (s.height, s.width)
            dn = read_window(it.assets["blue"].href, b, shp, Resampling.nearest)
            dn = dn[dn != rb.get("nodata", 0)]
            p1 = float(np.percentile(dn, 1))
            d.update(branche="pixels", bleu_p1_dn=p1, retire=p1 < 1000)
        rows.append(d)
        print(jeu, r.scene, d.get("branche"), d.get("baseline"), d.get("drapeau"), d.get("offset_meta"), d.get("bleu_p1_dn", ""), flush=True)
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_name("audit_boa.csv"), index=False)
print(df.groupby(["jeu", "branche", "retire"], dropna=False).size())
print(df.groupby(["baseline", "drapeau", "offset_meta"], dropna=False).size())
