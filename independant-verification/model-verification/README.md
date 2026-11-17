# Vérification des modèles sur de nouveaux sites (étude 2)

**Question** : les 3 U-Net entraînés (`unet_r34`, `unet_r34_s1`, `unet_r34_s2`, graines 0, 1 et 2) et leur
ensemble (`unet_r34_ens3`, la moyenne des trois) gardent-ils leurs performances sur des sites lidar de
l'Ontario qu'ils n'ont jamais vus ?

Les sorties lourdes vont sur le volume de travail : `/w/20261/tsalin/MLOps/independant_verification/modele/`.
Le code, les listes de sites et d'images, et les résultats légers (tableaux, figures) restent ici.

---

## Pourquoi de nouveaux sites lidar

- Le test du projet repose sur **3 sites seulement** (MAE par site de 0,078 à 0,134), et les 3 sites de
  validation ont servi à choisir le meilleur point de l'entraînement. L'estimation de la performance est
  donc fragile.
- Le lidar provincial couvre bien plus que les 49 sites du projet : **14 015 blocs** de 5 × 5 km
  remplissent les mêmes conditions.
- Sur de nouveaux sites, les labels se calculent **exactement comme ceux du jeu de données**. On compare donc
  le modèle à la même grandeur que celle qu'il a apprise, sans conversion. Ce n'est pas le cas avec GEDI
  (autre définition du couvert) ni avec SCANFI (un autre modèle, deux fois moins précis que le nôtre sur
  les sites du projet).

## Protocole (validé le 31 octobre 2026, avant les prédictions)

### Choix des sites

1. **Mêmes règles que le projet** (fonctions de `scripts/select_sites.py`) : 25 tuiles de 1 km contiguës
   toutes munies d'un CHM, une zone UTM, une année ≥ 2019, vols du 1er juin au 15 septembre et étalés
   sur 60 jours au plus.
2. **Indépendance** : au moins 0,5° (environ 40 à 55 km) de chacun des 50 sites du projet (entraînement,
   validation, test et site abandonné), et 0,5° entre nouveaux sites. C'est la séparation que le projet
   impose entre ses sites de validation ou de test et ses sites d'entraînement. Il reste 7 035 blocs.
3. **Tirage aléatoire** avec une graine fixe (3110), jusqu'à **15 sites**. Un tirage aléatoire représente
   les blocs lidar disponibles. Un tirage « le plus loin possible », comme celui du projet, privilégierait
   les bords de la province.
4. La sélection est faite une seule fois, avant toute prédiction, et enregistrée dans `sites.csv`.

### Données de chaque site

- **Labels** : `make_labels.build_site`, la fonction du projet, validée par l'étude GEDI.
- **Images Sentinel-2** : mêmes règles et mêmes fonctions que `scripts/make_stacks.py`. Ce sont des scènes
  L2A à ±45 jours du vol, avec moins de 1 % de nuages, d'ombres ou de neige, espacées d'au moins 5 jours,
  4 au plus.

### Prédiction

Avec la fonction du projet, `train_unet.predict_site` : fenêtres de 256 px qui se chevauchent, moyenne
sur toutes les dates et sur 4 retournements, normalisation enregistrée dans le `config.json` de chaque
modèle. L'ensemble est la moyenne des 3 prédictions, comme `scripts/ensemble.py`.

### Mesures

Calculées par nos soins, sans `scripts/evaluate.py`, mais avec les mêmes définitions :

- **Métriques** : MAE, RMSE, biais (prédiction − label), r, R², exactitude et F1 macro sur 4 classes
  (seuils 10 / 40 / 70 %).
- **Résolution** : à 10 m, et à 30 m (moyenne de 3 × 3 cases).
- **Cases retenues** : celles où le label et toutes les prédictions existent. Les sites sont regroupés en
  comptant chaque case une fois.
- **Détails** : par site, par tranche de couvert de 0,1, par année du vol, par zone UTM, et selon que le
  site est au nord des sites d'entraînement ou dans leur étendue.

### Référence

Les **mêmes modèles sur les 3 sites de test du projet**, prédits par notre chaîne et notés par notre code.
La comparaison se fait donc à l'identique.

### Contrôles

Sans eux, la comparaison n'a pas de sens :

- Nos prédictions des sites de test doivent égaler celles enregistrées à la fin de l'entraînement :
  écart maximal ≤ 0,02, écart moyen ≤ 0,002 par site et par modèle. Une petite différence vient du calcul
  en bfloat16 sur GPU.
- Notre code doit redonner la MAE du projet (`data/manifest/eval_test.json`) sur ses propres prédictions,
  à 0,001 près.

### Critères de réussite, fixés avant de voir les résultats

Ils s'appliquent à l'ensemble, à 10 m, sur les nouveaux sites regroupés.

- **Erreur globale** : MAE ≤ 1,25 × celle des sites de test. Avec 15 sites, et des MAE par site qui
  varient de 0,05 à 0,17 sur les sites tenus à part du projet, une dégradation de 25 % est détectable.
- **Biais** : entre −0,05 et +0,05.
- **Par tranche de couvert** : dans chaque tranche de 0,1 qui contient au moins 1 % des cases, la MAE ne
  dépasse pas celle des sites de test de plus de 0,05. Ce critère ne dépend pas de la composition des
  sites tirés : des sites plus denses ou plus clairsemés ne changent pas la note d'une tranche.
- **Aucun échec grossier** : aucun site avec une MAE > 0,25.
- **Stabilité** : l'écart de MAE entre les 3 modèles est ≤ 0,01.
- **Ensemble** : sa MAE ne dépasse pas celle du meilleur des 3 modèles de plus de 0,002.

### Règles d'exclusion, fixées d'avance

Un nouveau site sans aucune image Sentinel-2 exploitable est exclu et signalé. Aucun site n'est
remplacé ni retiré après la prédiction.

### Ce que l'étude peut montrer, et ce qu'elle ne peut pas montrer

- **Elle peut montrer** si l'erreur reste du même ordre ailleurs dans l'Ontario couvert par le lidar
  provincial (2019–2025), et si elle se dégrade dans certaines régions, années ou tranches de couvert.
- **Elle ne peut pas montrer** la performance hors de l'Ontario, ni dans les zones sans lidar provincial
  (en particulier le Grand Nord). Elle ne dit rien non plus des images prises loin de la date du vol : les
  images sont choisies à ±45 jours du vol, comme pour l'entraînement.
- **Comme toute évaluation contre ces labels**, elle mesure l'accord avec le couvert des couronnes calculé
  sur le CHM, pas avec une autre définition du couvert (voir l'étude GEDI).

## Étapes

| Script | Rôle | Sortie |
|---|---|---|
| `01_choisir_sites.py` | Index des tuiles FRI, tirage des nouveaux sites, contrôle des dates de vol | `sites.csv`, `tuiles.csv` |
| `02_telecharger_chm.py` | CHM à 0,5 m, 25 tuiles par site | `…/chm/<site>/` |
| `03_labels.py` | Labels à 10 m sur la grille Sentinel-2 | `…/labels/`, `labels_stats.csv` |
| `04_images_s2.py` | Jusqu'à 4 images Sentinel-2 par site | `…/stacks/`, `scenes_s2.csv` |
| `05_predire.py` | Prédictions des 3 modèles et de l'ensemble (GPU), sites de test compris | `…/predictions/<modèle>/` |
| `06_evaluer.py` | Contrôles, métriques, critères, figures | `resultats/` |

Lancement sur Slurm, avec les dépendances entre étapes : `bash independant-verification/model-verification/lancer_modele.sh`
(options `--etape N`, `--depuis N`, `--jusqua N`, `--dry-run`). L'étape 5 tourne sur la partition GPU, les
autres sur les nœuds de calcul. `commun.py` regroupe les chemins et les constantes.

## Détails de mise en œuvre

- **Code du projet réutilisé** : `select_sites` (éligibilité), `download_chm.fetch`, `make_labels.build_site`,
  `make_stacks.candidates` et `write_stack`, `train_unet.predict_site` et `to_tensor`. Ces fonctions
  définissent les données et le modèle que l'on teste. Leurs dossiers d'entrée et de sortie sont redirigés
  vers les nôtres, et rien n'est écrit dans `data/` du projet.
- **Index des tuiles** : téléchargé le 31 octobre 2026 (334 Mo). Il donne les mêmes 14 015 blocs éligibles
  que `data/manifest/candidate_blocks.csv`.
- **Modèles évalués** : ceux de `models/`, réentraînés lors de la reproduction du 30 octobre 2026 et
  équivalents à ceux du camarade (MAE de test de 0,109 à 0,111 par modèle, 0,108 pour l'ensemble).
- **Volume** : 375 tuiles CHM (3,9 Go) et 51 images Sentinel-2 (196 Mo).
- **Test à petite échelle** : les variables `MODELE_TRAVAIL` et `MODELE_SORTIES` redirigent les sorties.
  Sans GPU, l'étape 5 tourne sur le processeur, en float32, ce qui n'est utile que pour un test.

---

## Résultats (exécution du 31 octobre 2026)

**Verdict : 6 critères sur 6 réussis.** Sur 15 sites jamais vus, l'ensemble garde l'erreur mesurée sur
le test du projet ; il fait même un peu mieux. L'erreur augmente nettement vers le nord, et le sud de la
province n'a pas pu être testé (voir « Limites de l'échantillon »).

Détails dans `resultats/` : `criteres.json`, `controle.json`, `metriques_*.csv` et 4 figures
(`sites.png`, `nuages.png`, `tranches.png`, `carte.png`).

### Exécution

- **Sites** : 15 retenus, entre 47,3 et 50,2° N, de 2020 à 2025 (aucun en 2019 ni en 2024), dans les
  zones UTM 15 (2), 16 (5) et 17 (8). Sur les 51 blocs examinés, 36 ont été refusés par la règle des
  dates de vol.
- **Labels** : 100 % des cases sont complètes sur chaque site ; couvert moyen de 0,54 à 0,84.
- **Images** : 51 images Sentinel-2, de 1 à 4 par site ; aucun site exclu.

### Contrôles : réussis

- Nos prédictions des 3 sites de test égalent celles de l'entraînement : écart nul pour `unet_r34_s1`
  et `unet_r34_s2`, et au plus 0,003 pour `unet_r34` (écart moyen < 0,0002).
- Notre code redonne la MAE du projet sur ses propres prédictions : 0,1077 contre 0,1077.

### Critères

| Critère | Valeur | Résultat |
|---|---|---|
| MAE ≤ 1,25 × celle des sites de test | 0,089 contre 0,108 (rapport 0,82) | réussi |
| Biais entre −0,05 et +0,05 | +0,012 | réussi |
| Par tranche de couvert, MAE ≤ celle du test + 0,05 | aucune tranche au-delà | réussi |
| Aucun site avec une MAE > 0,25 | maximum 0,128 | réussi |
| Écart de MAE entre les 3 modèles ≤ 0,01 | 0,088 à 0,094 (écart 0,005) | réussi |
| Ensemble au plus 0,002 au-dessus du meilleur modèle | 0,0886 contre 0,0884 (`unet_r34_s1`) | réussi |

### Métriques de l'ensemble

| | Nouveaux sites, 10 m | Test, 10 m | Nouveaux sites, 30 m | Test, 30 m |
|---|---|---|---|---|
| Cases | 3 663 504 | 726 191 | 407 056 | 80 687 |
| MAE | **0,089** | 0,108 | **0,066** | 0,082 |
| RMSE | 0,135 | 0,169 | 0,096 | 0,127 |
| Biais | +0,012 | −0,014 | +0,012 | −0,014 |
| R² | 0,840 | 0,815 | 0,907 | 0,880 |
| Exactitude sur 4 classes | 0,786 | 0,744 | 0,835 | 0,786 |
| F1 macro | 0,698 | 0,651 | 0,775 | 0,731 |

### Par tranche de couvert (ensemble, 10 m)

| Label | 0–0,1 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Part des cases, nouveaux | 13 % | 2 % | 3 % | 4 % | 5 % | 7 % | 8 % | 10 % | 13 % | 36 % |
| MAE, nouveaux | 0,043 | 0,173 | 0,183 | 0,179 | 0,164 | 0,143 | 0,124 | 0,112 | 0,093 | 0,048 |
| MAE, test | 0,056 | 0,172 | 0,184 | 0,187 | 0,177 | 0,150 | 0,124 | 0,115 | 0,133 | 0,100 |

### Par site (ensemble, 10 m, du nord au sud)

| Site | Latitude | Images | Label moyen | MAE | Biais | R² |
|---|---|---|---|---|---|---|
| `Z17_E0485_N5560_2023` | 50,2 | 4 | 0,55 | 0,127 | −0,080 | 0,55 |
| `Z16_E0610_N5555_2020` | 50,2 | 3 | 0,58 | 0,128 | +0,067 | 0,47 |
| `Z17_E0450_N5540_2023` | 50,0 | 4 | 0,64 | 0,108 | +0,023 | 0,50 |
| `Z17_E0405_N5530_2021` | 49,9 | 3 | 0,57 | 0,128 | +0,017 | 0,72 |
| `Z17_E0565_N5500_2025` | 49,7 | 3 | 0,61 | 0,103 | −0,040 | 0,77 |
| `Z17_E0360_N5495_2021` | 49,6 | 4 | 0,70 | 0,111 | +0,025 | 0,69 |
| `Z16_E0470_N5480_2021` | 49,5 | 1 | 0,78 | 0,054 | +0,011 | 0,91 |
| `Z15_E0545_N5470_2022` | 49,4 | 4 | 0,54 | 0,029 | −0,004 | 0,98 |
| `Z16_E0570_N5450_2021` | 49,2 | 4 | 0,73 | 0,101 | −0,003 | 0,78 |
| `Z16_E0490_N5420_2020` | 49,0 | 4 | 0,84 | 0,073 | +0,036 | 0,78 |
| `Z17_E0395_N5415_2021` | 48,9 | 4 | 0,72 | 0,054 | +0,011 | 0,93 |
| `Z16_E0685_N5385_2022` | 48,6 | 3 | 0,62 | 0,092 | +0,048 | 0,84 |
| `Z15_E0675_N5335_2021` | 48,2 | 3 | 0,78 | 0,073 | −0,016 | 0,87 |
| `Z17_E0370_N5305_2021` | 47,9 | 3 | 0,64 | 0,063 | +0,015 | 0,91 |
| `Z17_E0565_N5240_2020` | 47,3 | 4 | 0,70 | 0,082 | +0,069 | 0,86 |

Sites de test du projet, pour comparaison : 0,078, 0,111 et 0,134.

### Lecture

- **L'erreur réelle est d'environ 0,09 à 10 m.** Avec 15 sites, l'intervalle de confiance à 95 % de la MAE
  va de 0,073 à 0,103 (bootstrap sur les sites). Le jeu de test de 3 sites (0,108) était du côté
  pessimiste. La différence ne vient pas de la composition : avec la répartition des couverts des sites
  de test, la MAE des nouveaux sites serait de 0,083.
- **Le nord est plus difficile.** Les 6 sites au-dessus de 49,6° N ont une MAE moyenne de 0,118
  (R² moyen 0,62), contre 0,069 (R² 0,87) pour les 9 autres. La corrélation de rang entre la latitude et
  la MAE est de 0,66. Le pire site de validation du projet (`Z17_E0355_N5560_2019`, 50,2° N, MAE 0,168)
  va dans le même sens. Hypothèse non vérifiée : des épinettes clairsemées et des tourbières, avec des
  arbres proches du seuil de 2 m. La distance aux sites d'entraînement, elle, n'explique pas l'erreur
  (corrélation de rang −0,23).
- **Les couverts intermédiaires restent le point faible du modèle.** Entre 0,1 et 0,5, la MAE est de 0,16
  à 0,18 sur les nouveaux sites comme sur le test. C'est une limite du modèle, pas un défaut de
  généralisation.
- **Biais local.** Le biais global est presque nul (+0,012), mais il varie de −0,08 à +0,07 selon les
  sites, et 3 sites dépassent ±0,05. Une carte d'une zone de quelques kilomètres peut donc être décalée de
  quelques points de couvert.
- **Les 3 modèles se valent** (0,088 à 0,094). L'ensemble égale le meilleur à 10 m et est le meilleur à
  30 m. Il reste le bon choix.
- **Le nombre d'images Sentinel-2 ne joue pas** (corrélation de rang 0,00 avec la MAE). Le site qui n'a
  qu'une image a une MAE de 0,054.
- **Signal faible** : les 3 sites de 2020 ont un biais moyen de +0,058. Avec si peu de sites, on ne peut
  rien en conclure ; c'est à garder en tête avec la question des premières années de lidar soulevée par
  l'étude GEDI.

### Limites de l'échantillon

- **Le sud n'est pas représenté.** En rejouant le tirage, les 13 blocs rencontrés au sud de 47,3° N ont
  tous été refusés par la règle des dates de vol (lidar pris hors de la saison feuillée, ou étalé sur plus
  de 60 jours). Le résultat vaut pour le domaine où le lidar a été pris en saison feuillée, pas pour la
  forêt mixte des Grands Lacs et du Saint-Laurent.
- **Le grand nord non plus.** Aucun site au-dessus de 50,2° N n'a été tiré, alors que 19 % des blocs
  éligibles s'y trouvent. Comme l'erreur augmente vers le nord, c'est la zone à vérifier en priorité.
- **Analyses exploratoires.** Les critères et les tableaux ci-dessus viennent de `06_evaluer.py` et
  étaient fixés d'avance. En revanche, l'intervalle de confiance, la correction de composition, l'analyse
  par latitude et le tirage rejoué ont été calculés après avoir vu les résultats, à partir de
  `resultats/metriques_sites.csv` et `resultats/metriques_tranches.csv`.
