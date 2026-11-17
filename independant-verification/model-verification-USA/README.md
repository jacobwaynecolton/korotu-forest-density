# Vérification des modèles aux États-Unis, avec le lidar NEON (étude 3)

**Question** : que prédisent les 3 U-Net et leur ensemble sur des forêts américaines, proches de celles de
l'Ontario (groupe A) ou très éloignées de leur distribution d'entraînement (groupe B) ?
L'étude est **descriptive** : pas de critère de réussite. Le groupe A ne compte que 3 sites, et le groupe B
est volontairement hors domaine.

Les sorties lourdes vont sur le volume de travail : `/w/20261/tsalin/MLOps/independant_verification/modele_usa/`.
Les nuages de points (91 Go) ne tiennent pas dans le quota de 100 Go : chacun est effacé dès que son CHM est
écrit, et l'étude occupe au final environ 2 Go.

---

## Pourquoi construire nos propres CHM

NEON fournit un CHM à 1 m, mais il ne donne pas le même label que le CHM ontarien à 0,5 m :

- **Résolution (étape 0)** : sur les 15 nouveaux sites ontariens, le même CHM passé à 1 m (maximum de chaque
  groupe de 2 × 2 pixels) donne un label plus élevé de **+0,08** en moyenne, et jusqu'à +0,17 pour les couverts
  de 0,3 à 0,6 : les trouées de moins d'un mètre disparaissent.
- **Densité de points** : le label dépend aussi du nombre de mesures par pixel. Les tuiles ontariennes ont une
  médiane de **44 premiers retours par m²**, alors que la plupart des vols NEON en ont entre 5 et 25.

On construit donc un CHM à 0,5 m à partir des nuages de points NEON, avec une méthode vérifiée d'abord sur
l'Ontario (étape 1), et seulement sur les vols NEON assez denses.

## Méthode de construction du CHM (`nuages.py`)

1. Retirer les points de bruit (classes 7 et 18).
2. Garder un seul jour de vol par tuile, celui qui a le plus de points. Sur une tuile ontarienne survolée deux
   jours différents, superposer les deux survols gonflait le label de +0,06.
3. Ramener la densité à 44 premiers retours par m² en gardant des impulsions entières au hasard. Un nuage
   moins dense est laissé tel quel.
4. Sol : moyenne des points de la classe 2 par case de 1 m, trous comblés par interpolation.
5. CHM : hauteur maximale des points de chaque case de 0,5 m. Une case vide prend la valeur de sa voisine
   pleine la plus proche à 1 m au plus.

Le CHM officiel ontarien est exactement DSM − DEM du fournisseur. Sur des tuiles survolées en un seul jour, le
maximum par case en redonne les labels à 0,005 près, alors que les quantiles plus bas s'en écartent fortement
(jusqu'à −0,29 pour la médiane).

## Protocole (31 octobre 2026)

### Étape 1 : calibration sur l'Ontario, avec point d'arrêt

Sur la tuile centrale de chacun des 15 nouveaux sites ontariens, on compare les labels à 10 m du CHM officiel à
ceux de notre méthode, construite sur le nuage de points de la même tuile.

**Point d'arrêt** : sur les tuiles survolées en un seul jour, l'écart absolu moyen doit rester ≤ 0,02. Sinon,
la chaîne s'arrête.

La même étape mesure l'effet d'une densité de 30 premiers retours par m², celle des vols NEON les moins denses
retenus.

### Sites et survols

Pour chaque site : le survol publié le plus récent de 2021 à 2025 qui atteint 25 premiers retours par m²
(médiane des en-têtes des tuiles).

| Groupe | Site | État | Forêt | Survol | Densité (pts/m²) |
|---|---|---|---|---|---|
| A | BART, Bartlett | NH | Feuillus du nord (hêtre, érable, bouleau), pruche et épinette en altitude | août–sept. 2024 | 44 |
| A | BONA, Caribou-Poker Creeks | AK | Forêt boréale (épinette noire, bouleau, peuplier) | juillet 2024 | 48 |
| A | HARV, Harvard Forest | MA | Forêt mixte tempérée (chêne rouge, érable rouge, pruche, pin blanc) | août–sept. 2024 | 30 |
| B | BLAN, Blandy | VA | Mosaïque agricole et boisés de feuillus | août 2021 | 45 |
| B | SERC, Smithsonian Environmental Research Center | MD | Grands feuillus de la plaine côtière, champs, marais | août 2021 | 41 |
| B | SCBI, Smithsonian Conservation Biology Institute | VA | Feuillus des Appalaches (chênes, tulipier) | août 2021 | 51 |
| B | OSBS, Ordway-Swisher | FL | Pins des marais sur sable, savane, zones humides, lacs | août–sept. 2021 | 36 |

Densités du tableau : médianes des 25 tuiles retenues. HARV est en dessous de la densité ontarienne
habituelle : son label risque d'être un peu plus bas qu'avec le lidar ontarien (l'étape 1 en donne l'ordre de
grandeur).

### Bloc de chaque site

Parmi les blocs de 5 × 5 tuiles de 1 km complets, celui dont le centre est le plus proche du centre du survol.
On ne regarde pas la forêt pour choisir.

### Ensuite, comme en Ontario

- **Labels** : `make_labels.build_site`, appliqué à nos CHM.
- **Images Sentinel-2** : jusqu'à 4 images à ±45 jours du milieu du vol.
- **Prédiction** : les 3 modèles et leur ensemble, avec la fonction du projet.
- **Mesures** : MAE, RMSE, biais, r, R², exactitude et F1 sur 4 classes, à 10 m et à 30 m, par groupe, par site
  et par tranche de couvert.
- **Référence** : l'ensemble sur les 15 nouveaux sites ontariens (MAE de 0,089 à 10 m).
- **Contrôle** : les 3 sites de test ontariens sont prédits de nouveau et doivent redonner les prédictions de
  l'entraînement.

### Ce que l'étude peut montrer, et ce qu'elle ne peut pas montrer

- **Elle peut montrer** l'ordre de grandeur de l'erreur dans des forêts proches de l'Ontario mais mesurées par
  un autre lidar, et le comportement du modèle très loin de sa distribution d'entraînement.
- **Elle ne peut pas conclure** sur la généralisation avec 3 sites dans le groupe A.
- **La comparaison reste imparfaite.** Le capteur de NEON n'est pas celui de l'Ontario (lidar classique contre
  lidar à photon unique, et une densité qu'on peut réduire mais pas augmenter), et le sol est calculé par nous.
  Une partie de l'écart peut donc venir des labels eux-mêmes.

## Étapes

| Script | Rôle | Sortie |
|---|---|---|
| `00_effet_resolution.py` | Effet du passage de 0,5 m à 1 m sur les labels ontariens | `resultats/effet_resolution*.csv` |
| `01_calibrer_ontario.py` | Méthode contre CHM officiel sur 15 tuiles ontariennes, point d'arrêt | `resultats/calibration*` |
| `02_choisir_sites.py` | Bloc de 5 × 5 km de chaque survol, zone UTM, dates, densité | `sites.csv`, `tuiles.csv` |
| `03_nuages_chm.py` | Pour chacune des 175 tuiles : nuage de points NEON téléchargé, CHM à 0,5 m, nuage effacé | `…/chm/`, `chm_stats.csv` |
| `04_labels.py` | Labels à 10 m (script ontarien `03_labels.py`) | `…/labels/`, `labels_stats.csv` |
| `05_images_s2.py` | Images Sentinel-2 (script ontarien `04_images_s2.py`) | `…/stacks/`, `scenes_s2.csv` |
| `06_predire.py` | Prédictions, GPU (script ontarien `05_predire.py`) | `…/predictions/` |
| `07_evaluer.py` | Contrôle, métriques, figures | `resultats/` |

Lancement sur Slurm : `bash independant-verification/model-verification-USA/lancer_usa.sh` (options
`--etape N`, `--depuis N`, `--jusqua N`, `--dry-run`). Les étapes 4 à 6 exécutent les scripts de l'étude
ontarienne sans les modifier, après avoir fixé `MODELE_TRAVAIL` et `MODELE_SORTIES` sur nos dossiers.

## Détails de mise en œuvre

- **Clé NEON** : depuis juin 2026, NEON exige une clé d'API pour télécharger. Elle est lue dans
  `~/.neon_token` (fichier lisible par toi seul) et n'est jamais affichée.
- **Liens signés** : les liens de téléchargement expirent, d'où une nouvelle demande à l'API juste avant de
  télécharger.
- **Fichiers retenus** : seules les tuiles classées (`…_DP1_<est>_<nord>_classified_point_cloud…laz`), pas
  les lignes de vol ni les fichiers de qualité livrés à côté.
- **Coordonnées** : les tuiles NEON sont déjà en WGS84 / UTM, le système de Sentinel-2. Il n'y a rien à
  reprojeter.
- **Mémoire** : environ 8 Go par tuile en cours de traitement. L'étape 3 en traite 2 à la fois.
- **Quota** : la première exécution (31 octobre 2026) s'est arrêtée sans message après 36 nuages de points
  (21 Go) : le quota de 100 Go du volume de travail était atteint, et même le journal ne pouvait plus être
  écrit. D'où l'étape 3 actuelle, qui efface chaque nuage après usage, avec des téléchargements par curl
  bornés dans le temps. Les 15 nuages de points ontariens de la calibration ont aussi été effacés
  (résultats gardés dans `resultats/calibration*`) ; relancer l'étape 1 les téléchargerait de nouveau.
- **Environnement** : `laspy` 2.7.0 et `lazrs` 0.8.2 ajoutés au `.venv` le 31 octobre 2026, sans autre
  changement.
