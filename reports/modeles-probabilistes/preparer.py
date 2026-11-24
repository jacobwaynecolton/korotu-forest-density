"""Données et images du rapport « modèle probabilistes » (scratchpad, pas dans le dépôt)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

PROJET = Path("/h/460/tsalin/projects/MLOps/korotu-forest-density")
sys.path.insert(0, str(PROJET / "modeles-probabilistes"))
from commun import NEON, ONTARIO, RESULTATS, SORTIES, lire, sites_externes  # noqa: E402

ICI = Path(__file__).resolve().parent
IMG = ICI / "img"
IMG.mkdir(exist_ok=True)
CONFIG = "q_r34_imagenet"
NOMS_NEON = {"BART_2024": "BART (NH)", "BONA_2024": "BONA (AK)", "HARV_2024": "HARV (MA)", "BLAN_2021": "BLAN (VA)",
             "SERC_2021": "SERC (MD)", "SCBI_2021": "SCBI (VA)", "OSBS_2021": "OSBS (FL)"}


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def tableau(df):
    return [{k: (r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
             for k, v in row.items()} for row in df.to_dict("records")]


def rgb(chemin):
    with rasterio.open(chemin) as s:
        x = s.read([4, 3, 2]).astype(np.float32)
    x[x <= 0] = np.nan
    out = np.zeros(x.shape, np.uint8)
    for i in range(3):
        v = x[i][np.isfinite(x[i])]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        out[i] = (np.nan_to_num((x[i] - lo) / (hi - lo)).clip(0, 1) * 255).astype(np.uint8)
    return out.transpose(1, 2, 0)


def main():
    d = {}
    d["resume"] = json.loads((RESULTATS / "resume.json").read_text())
    for nom in ("cv_metriques", "cv_par_pli", "externe_metriques", "externe_par_site", "couverture_tranches",
                "references_metriques", "references_par_site", "cv_par_site"):
        d[nom] = tableau(pd.read_csv(RESULTATS / f"{nom}.csv"))
    plis = pd.read_csv(PROJET / "modeles-probabilistes/plis.csv")
    d["plis"] = tableau(plis[["site_id", "pli", "lon", "lat", "split_origine"]])
    ont = pd.read_csv(ONTARIO["sites"])
    d["nouveaux"] = tableau(ont[["site_id", "lon", "lat", "year"]])
    neon = pd.read_csv(NEON["sites"])
    d["neon"] = tableau(neon[[c for c in ("site_id", "lon", "lat", "groupe") if c in neon.columns]])
    d["contour"] = json.loads((PROJET / "report/ontario_outline.json").read_text())

    # images des sites externes
    cartes = SORTIES / "cartes" / CONFIG / "externes"
    sites = []
    for e in sites_externes():
        s = e["site_id"]
        y = lire(e["label"])
        with rasterio.open(cartes / f"{s}.tif") as src:
            q = src.read().astype(np.float32)
        q = np.where(q == 65535, np.nan, q / 10000)
        h, w = y.shape
        assert q.shape[1:] == (h, w), (s, q.shape, y.shape)
        pile = np.concatenate([y, q[0], q[1], q[2]], axis=1)
        g = np.where(np.isfinite(pile), np.round(np.clip(pile, 0, 1) * 250), 255).astype(np.uint8)
        Image.fromarray(g, mode="L").save(IMG / f"{s}_cartes.png", optimize=True)
        Image.fromarray(rgb(e["images"][0])).save(IMG / f"{s}_s2.jpg", quality=86, optimize=True)
        ok = np.isfinite(y) & np.isfinite(q[1])
        sites.append(dict(id=s, groupe=e["groupe"], nom=NOMS_NEON.get(s, s), largeur=w, hauteur=h,
                          images=len(e["images"]), label_moyen=r(np.nanmean(y[ok]))))
    d["sites"] = sites
    (ICI / "donnees.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
    taille = sum(p.stat().st_size for p in IMG.iterdir())
    print(f"{len(sites)} sites, images {taille / 1e6:.1f} Mo, données {(ICI / 'donnees.json').stat().st_size / 1e3:.0f} ko")


if __name__ == "__main__":
    main()
