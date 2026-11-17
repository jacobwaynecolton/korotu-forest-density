"""Construire un CHM à 0,5 m à partir d'un nuage de points lidar, de la même façon en Ontario et chez NEON.

Méthode, retenue après comparaison avec le CHM officiel de l'Ontario (DSM - DEM du fournisseur, étape 1) :
  - points de bruit (classes 7 et 18) et points mis à l'écart (« withheld ») retirés ;
  - un seul jour de vol par tuile, celui qui a le plus de points : superposer deux survols de jours
    différents gonfle le label (+0,06 sur une tuile ontarienne survolée deux fois) ;
  - densité ramenée à celle des sites d'entraînement ontariens (44 premiers retours par m², médiane des
    tuiles) en gardant des impulsions entières au hasard ; un nuage moins dense est laissé tel quel ;
  - sol : moyenne des points de la classe 2 par case de 1 m, trous comblés par interpolation linéaire,
    puis par la valeur la plus proche ; hauteur d'un point = altitude - sol (interpolation bilinéaire) ;
  - CHM : hauteur maximale des points de chaque case de 0,5 m ; une case vide prend la valeur de la case
    pleine la plus proche si elle est à 1 m au plus, sinon elle reste sans donnée.
Sur des tuiles ontariennes survolées un seul jour, le maximum par case redonne les labels officiels à
0,005 près ; les quantiles plus bas s'en écartent fortement.
"""
import laspy
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

BRUIT = (7, 18)
SOL = 2
PAS_SOL = 1.0         # m
COMBLE_MAX = 1.0      # m
DENSITE_CIBLE = 44.0  # premiers retours par m², médiane des tuiles d'entraînement ontariennes
SEMAINE = 604800      # s


def lire(chemin):
    """Points utiles d'un fichier LAZ : dict de tableaux x, y, z, classe, retour, temps, jour."""
    las = laspy.read(chemin)
    ok = ~np.isin(np.asarray(las.classification), BRUIT) & ~np.asarray(las.withheld, bool)
    temps = np.asarray(las.gps_time)[ok]
    # Temps GPS « standard ajusté » (secondes GPS - 1e9) ou secondes dans la semaine GPS. Les fichiers
    # ontariens n'indiquent pas toujours le format dans l'en-tête : on le déduit des valeurs.
    absolu = temps + 1e9 if temps.size and temps.max() > SEMAINE else temps
    return dict(x=np.asarray(las.x)[ok], y=np.asarray(las.y)[ok], z=np.asarray(las.z)[ok],
                classe=np.asarray(las.classification)[ok], retour=np.asarray(las.return_number)[ok],
                temps=temps, jour=(absolu // 86400).astype(np.int64))


def garder(p, masque):
    return {k: v[masque] for k, v in p.items()}


def densite_premiers(p):
    """Premiers retours par m² de surface couverte (cases de 1 m qui contiennent au moins un point)."""
    if not p["x"].size:
        return 0.0
    c = (p["x"] - p["x"].min()).astype(np.int64)
    r = (p["y"] - p["y"].min()).astype(np.int64)
    couverte = int((np.bincount(r * (c.max() + 1) + c) > 0).sum())
    return float((p["retour"] == 1).sum() / couverte)


def eclaircir(p, facteur, graine=0):
    """Garde chaque impulsion avec la probabilité `facteur`, avec tous ses retours."""
    if facteur >= 1:
        return p
    _, impulsion = np.unique(p["temps"], return_inverse=True)   # les retours d'une impulsion partagent son temps
    choisies = np.random.default_rng(graine).random(impulsion.max() + 1) < facteur
    return garder(p, choisies[impulsion])


def preparer(p, cible=DENSITE_CIBLE):
    """Un seul jour de vol, puis densité ramenée à `cible` si besoin. Renvoie (points, informations)."""
    jours, n = np.unique(p["jour"], return_counts=True)
    principal = jours[n.argmax()]
    q = garder(p, p["jour"] == principal)
    avant = densite_premiers(q)
    q = eclaircir(q, cible / avant) if avant > cible else q
    return q, dict(jours=len(jours), part_jour_principal=float(n.max() / n.sum()),
                   densite_avant=avant, densite_apres=densite_premiers(q))


def mnt(p, x0, y1, largeur, hauteur):
    """Modèle de sol au pas PAS_SOL sur l'emprise [x0, x0 + largeur] × [y1 - hauteur, y1]."""
    W, H = int(round(largeur / PAS_SOL)), int(round(hauteur / PAS_SOL))
    sol = p["classe"] == SOL
    c = ((p["x"][sol] - x0) // PAS_SOL).astype(np.int64)
    r = ((y1 - p["y"][sol]) // PAS_SOL).astype(np.int64)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    somme, n = np.zeros(H * W), np.zeros(H * W)
    np.add.at(somme, r[ok] * W + c[ok], p["z"][sol][ok])
    np.add.at(n, r[ok] * W + c[ok], 1)
    g = np.where(n > 0, somme / np.maximum(n, 1), np.nan).reshape(H, W)
    vides = np.isnan(g)
    if vides.any() and (~vides).any():
        rr, cc = np.nonzero(~vides)
        g[vides] = griddata((rr, cc), g[rr, cc], np.nonzero(vides), method="linear")
        if np.isnan(g).any():
            idx = ndimage.distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
            g = g[tuple(idx)]
    return g


def chm(p, transform, forme, sol=None):
    """CHM au pas de `transform` (0,5 m) sur la grille `forme` ; `sol` : MNT au pas PAS_SOL sur la même emprise."""
    res, x0, y1 = transform.a, transform.c, transform.f
    H, W = forme
    if sol is None:
        sol = mnt(p, x0, y1, W * res, H * res)
    gr = (y1 - p["y"]) / PAS_SOL - 0.5
    gc = (p["x"] - x0) / PAS_SOL - 0.5
    h = p["z"] - ndimage.map_coordinates(sol, [gr, gc], order=1, mode="nearest")
    c = ((p["x"] - x0) // res).astype(np.int64)
    r = ((y1 - p["y"]) // res).astype(np.int64)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    haut = np.full(H * W, -np.inf)
    np.maximum.at(haut, r[ok] * W + c[ok], h[ok])
    haut = haut.reshape(H, W)
    vide = ~np.isfinite(haut)
    if vide.any() and (~vide).any():
        d, idx = ndimage.distance_transform_edt(vide, return_indices=True)
        haut = np.where(vide & (d * res <= COMBLE_MAX), haut[tuple(idx)], haut)
    haut[~np.isfinite(haut)] = np.nan
    return haut.astype(np.float32)


def labels_10m(h, case=20):
    """Part des pixels au-dessus de 2 m par bloc de case × case pixels, et blocs entièrement valides."""
    v = np.isfinite(h) & (h >= -2) & (h <= 80)
    H, W = h.shape[0] // case * case, h.shape[1] // case * case
    h, v = h[:H, :W], v[:H, :W]
    forme = (H // case, case, W // case, case)
    return (np.where(v, h, 0) > 2).reshape(forme).mean(axis=(1, 3)), v.reshape(forme).all(axis=(1, 3))
