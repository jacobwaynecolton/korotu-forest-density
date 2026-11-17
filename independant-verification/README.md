# Vérifications indépendantes

Deux études, menées **sans réutiliser le code du projet** : on lit seulement ses sorties
(labels, CHM, images, prédictions, `data/manifest/sites.csv`) avec nos propres scripts.

1. **Les labels sont-ils corrects ?** Comparaison avec un autre lidar, GEDI (`gedi/`).
2. **Les modèles généralisent-ils ?** Évaluation sur de nouveaux sites lidar de l'Ontario (`model-verification/`,
   protocole dans son README).
3. **Que font les modèles hors de l'Ontario ?** Sites lidar NEON aux États-Unis, proches ou très éloignés des
   forêts d'entraînement (`model-verification-USA/`, étude descriptive).

Les sorties lourdes vont sur le volume de travail : `/w/20261/tsalin/MLOps/independant_verification/`.
Le code et les résultats légers (tableaux, figures) restent ici.

Environnement : le `.venv` du projet, avec deux ajouts (`earthaccess`, `h5py`) installés le
30 octobre 2026. Effet de bord : `python-dateutil` est passé de 3.0.0 à 2.9.0.post0, une contrainte
de `botocore`. Listes complètes des versions : `env_avant_installation.txt` et `env_apres_installation.txt`.

---

## 1. Labels contre GEDI : protocole (validé le 30 octobre 2026)

Décisions validées avant tout calcul : GEDI version 003, tirs à ±1 an du vol lidar, masque des
perturbations Hansen Global Forest Change, critères de réussite ci-dessous.

### Pourquoi GEDI

GEDI est un lidar spatial, installé sur la Station spatiale internationale depuis 2019. Le capteur,
l'équipe et la méthode sont différents de ceux du lidar ontarien. Chaque tir mesure une empreinte
d'environ 25 m de diamètre.

### Ce que la comparaison peut montrer, et ce qu'elle ne peut pas montrer

- **Elle peut montrer** des erreurs grossières (hauteurs, seuil, unités, position décalée de plus
  de ~10 m), des biais propres à un site, une année ou une zone, et la forme de la relation.
- **Elle ne peut pas prouver** la précision à 10 m. La position d'un tir GEDI est connue à ~10 m près,
  et les définitions diffèrent. GEDI mesure la probabilité que la lumière traverse la canopée, trous
  dans les couronnes compris. Le label mesure la surface sous un CHM de plus de 2 m. On attend donc
  une corrélation forte avec un biais, pas l'égalité.

### Données GEDI

| Produit | Variables | Usage |
|---|---|---|
| L2A, version 003 | `rh` (centile 98), `quality_flag`, `degrade_flag`, `sensitivity`, `solar_elevation`, `lat_lowestmode`, `lon_lowestmode`, `delta_time` | Hauteur de la canopée |
| L2B, version 003 | `cover`, `cover_z` (couvert cumulé par tranches de 5 m), `l2b_quality_flag`, `leaf_off_flag` | Couvert total et au-dessus de 5 m |

Accès avec `earthaccess` (compte NASA Earthdata). On lit à distance seulement les variables utiles,
dans le cloud de la NASA (AWS us-west-2), sans télécharger les fichiers complets (0,4 à 2 Go chacun).

### Sélection des tirs

1. Le tir tombe dans l'emprise d'un des 49 sites.
2. Qualité : `quality_flag = 1`, `degrade_flag = 0`, `l2b_quality_flag = 1`, `sensitivity ≥ 0,95`.
3. Saison : du 1er juin au 15 septembre, avec `leaf_off_flag = 0`.
4. Date : même année que le vol lidar, ou à ±1 an. C'est nécessaire pour les 9 sites survolés en 2023,
   car GEDI était en pause de mars 2023 à avril 2024.
5. Aucune perturbation (coupe, feu) entre les deux dates, d'après le masque Hansen Global Forest Change.
6. Variantes analysées à part : faisceaux pleine puissance seulement, tirs de nuit seulement.

Volume estimé d'après le catalogue de la NASA (L2B v003) : **478 fichiers GEDI**, médiane de 12 par
site. Un seul site n'a aucune donnée : `Z15_E0480_N5780_2025`, à 52,2°N, au-delà de l'orbite.

### Références calculées sur nos données, pour chaque tir

On prend un disque de 25 m de diamètre autour du centre du tir, avec une pondération gaussienne
testée en variante.

- Depuis le **CHM brut à 0,5 m** : part au-dessus de 2 m, part au-dessus de 5 m, centile 98 des hauteurs.
- Depuis les **labels à 10 m** : moyenne de `cover_gt2m` et `cover_gt5m`, pondérée par la surface de
  chaque case dans le disque. **C'est ce calcul qui teste les labels du dataset.**

Les conversions de coordonnées se font avec pyproj, sans les fonctions du projet.

### Comparaisons

| Ce qu'on compare | Ce que ça vérifie |
|---|---|
| RH98 de GEDI et centile 98 du CHM | Les hauteurs du CHM et la position |
| `cover` de GEDI et part au-dessus de 2 m (CHM, labels) | Le label principal |
| `cover_z` à 5 m et part au-dessus de 5 m | La bande `cover_gt5m` |
| Labels et CHM sur le même disque | L'effet du regroupement en cases de 10 m |
| Tirs décalés de ±10 m et ±20 m | La position, de façon grossière |

Les résultats sont détaillés par site, par année, par zone UTM, par tranche de densité, et pour les
tuiles 2025 (déjà en WGS84) face aux autres.

### Critères de réussite, fixés avant de voir les résultats

- **Hauteurs** : corrélation ≥ 0,85 et biais médian entre −2 m et +2 m.
- **Couvert** : corrélation de rang (Spearman) ≥ 0,70, et relation croissante d'une tranche de densité
  à la suivante.
- **Sites** : aucun site avec au moins 30 tirs retenus n'a une corrélation de rang < 0,40, ni un biais
  à plus de 2 écarts-types de la moyenne des sites.
- **Groupes** : pas d'écart de biais supérieur à 0,05 entre années, zones UTM ou tuiles 2025.
- **Position** : la corrélation est maximale sans décalage, à la précision de GEDI près (±10 m).

### Étapes

| Script | Rôle | Sortie |
|---|---|---|
| `gedi/01_lister_fichiers.py` | Liste des fichiers GEDI par site, depuis le catalogue de la NASA | `gedi/fichiers.csv` |
| `gedi/02_extraire_tirs.py` | Lecture à distance des variables utiles | `…/gedi_tirs.parquet` |
| `gedi/03_references.py` | Masque des perturbations, références CHM et labels | `…/gedi_tirs_refs.parquet` |
| `gedi/04_comparer.py` | Statistiques, figures, critères | `gedi/resultats/` |
| `gedi/05_analyse_complementaire.py` | Analyses exploratoires, faites après avoir vu les résultats : effet du niveau de couvert, sites, années, tirs manqués | `gedi/resultats/complementaire_*.csv`, `rapport_donnees.json` |

Lancement sur Slurm, avec les dépendances entre étapes : `bash independant-verification/gedi/lancer_gedi.sh`
(options `--etape N`, `--depuis N`, `--dry-run`). `gedi/commun.py` regroupe les chemins et les constantes.
L'étape 5 ne fait pas partie du protocole et ne change pas le verdict de `criteres.json`. Elle tourne en
quelques secondes, directement : `python -I independant-verification/gedi/05_analyse_complementaire.py`.

### Détails de mise en œuvre

Ces points ont été fixés en écrivant les scripts, après lecture de vrais fichiers GEDI version 003 :

- **Drapeaux de qualité** : la version 3 en fournit deux jeux ; on utilise ceux de la version 3
  (`l2a_quality_flag_rel3`, `l2b_quality_flag_rel3`).
- **Couvert au-dessus de 5 m** : deuxième valeur de `cover_z`, puisque le pas vertical `ancillary/dz`
  vaut 5 m dans les fichiers.
- **Date d'un tir** : `delta_time` secondes après le 1er janvier 2018 (UTC), contrôlée contre la date
  du nom de fichier.
- **Deux sélections successives** : l'étape 3 garde les tirs de sensibilité ≥ 0,90 (pour pouvoir
  tester des variantes) ; l'analyse principale de l'étape 4 applique 0,95.
- **Eau** : les tirs que GEDI classe en eau (WorldCover 80, fourni dans le fichier L2B) sont exclus de
  l'analyse principale, car l'accord y est trivial (couvert nul des deux côtés). Ils sont rapportés à part.
- **Références** : points tous les 0,5 m dans le disque (1 m pour le test de position). Le CHM doit être
  valide sur au moins 99 % du disque ; les labels, sur 100 %. Pondération gaussienne en variante,
  avec un écart-type de 6,25 m.
- **Perturbations** : Hansen Global Forest Change v1.13 (pertes jusqu'en 2025), 3 × 3 pixels autour du
  tir, toute perte entre l'année du vol et celle du tir, bornes comprises.
- **Monotonie** : la médiane de GEDI ne doit pas baisser de plus de 0,01 d'une tranche de 0,1 à la
  suivante (tranches d'au moins 30 tirs).
- **Robustesse de la lecture à distance** : processus parallèles créés à neuf (`spawn`), délais
  maximaux par requête, et durée maximale de 10 minutes par passage (alarme `SIGALRM`). Cette dernière
  sécurité a été ajoutée après la première exécution complète (31 octobre 2026) : 476 passages sur 478
  ont été traités en 2 h 45, mais 2 sont restés bloqués sans erreur jusqu'à la limite Slurm de 12 heures.
  Relus seuls, ces 2 passages se lisaient normalement : le blocage était passager.
