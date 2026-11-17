# Modèles probabilistes de couvert forestier

**But** : produire, pour chaque pixel de 10 m, trois cartes de couvert :
- **prudente** : quantile 10 %. Le vrai couvert n'est en dessous que dans 10 % des cas.
- **centrale** : médiane, quantile 50 %.
- **optimiste** : quantile 90 %. Le vrai couvert n'est au-dessus que dans 10 % des cas.

Au passage, on compare l'encodeur actuel du U-Net à un encodeur pré-entraîné sur Sentinel-2.

Les sorties lourdes vont sur le volume de travail : `/w/20261/tsalin/MLOps/modeles_probabilistes/`.
Le code, les plis et les résultats légers restent ici.

---

## Protocole (fixé le 17 novembre 2026, avant les entraînements)

Choix faits avec l'équipe : cartes pixel par pixel, bornes à 10 % et 90 %, 5 graines par configuration, pas
de modèle temporel, pas de modèle de fondation, pas de loi de probabilité complète.

### Étape 1 : une évaluation capable de distinguer deux modèles

Le jeu de test du projet ne compte que 3 sites : la MAE y varie de 0,078 à 0,134 selon le site, plus que
l'écart attendu entre deux architectures. On évalue donc :

- **Validation croisée par site** (`01_plis.py`) : les 49 sites du projet sont triés d'ouest en est et coupés
  en 5 blocs de longitude de 9 ou 10 sites. Chaque bloc sert une fois de test, et le modèle est entraîné sur
  les 4 autres. Chaque pli teste ainsi une région que le modèle n'a pas vue.
- **Test externe** : les 15 nouveaux sites ontariens de `independant-verification/model-verification/`. Ils
  ne servent jamais à entraîner ni à choisir un modèle.
- **Hors domaine** : les 7 sites NEON de `independant-verification/model-verification-USA/`, à titre
  descriptif.

Le modèle actuel (U-Net ResNet-34 ImageNet, perte L1) est réévalué dans ce cadre (configuration
`l1_r34_imagenet`), pour servir de référence.

Différence avec le projet : on n'a plus de sites de validation pour choisir le meilleur point de
l'entraînement. Avec 3 sites, ce choix était lui-même bruité : la meilleure MAE de validation était à l'étape
4 500, mais retombait à 0,108 à la fin. Chaque modèle est donc entraîné 8 000 pas, et on garde le modèle final.

### Étape 2 : U-Net à trois quantiles, 5 graines, calibration conforme

- **Sorties** : le réseau prédit q50 et deux écarts, si bien que les quantiles restent ordonnés et dans [0, 1] :
  `q50 = σ(z0)`, `q10 = q50 × (1 − σ(z1))`, `q90 = q50 + (1 − q50) × σ(z2)`.
- **Perte** : somme des pertes quantiles (« pinball ») à 10, 50 et 90 %. À 50 %, c'est la moitié de la perte
  L1 actuelle : la carte centrale est la médiane, comme aujourd'hui.
- **Entraînement** : identique au projet. Recadrages de 128 px, rotations et retournements, une date
  Sentinel-2 au hasard, AdamW (taux 10⁻³), OneCycle, 8 000 pas, lots de 32, bfloat16.
- **Prédiction** : comme le projet. Fenêtres de 256 px qui se chevauchent, moyenne sur les dates et 4
  retournements. Pour les quantiles, on fait la moyenne des quantiles prédits pour chaque variante.
- **Ensemble de 5 graines** : chaque graine est vue comme une distribution, approchée par 19 quantiles
  interpolés entre q10, q50 et q90. On prend les quantiles 10, 50 et 90 % du mélange des 5. L'intervalle
  s'élargit donc là où les graines divergent.
- **Calibration conforme**, borne par borne :
  - `c_bas` est le quantile à 90 % de `q10 − label` ;
  - `c_haut` est le quantile à 90 % de `label − q90` ;
  - les cartes livrées sont `q10 − c_bas` et `q90 + c_haut`, ramenées dans [0, 1].
  - **En validation croisée**, les constantes de chaque pli viennent des 4 autres plis, ce qui donne une
    couverture honnête.
  - **Pour les modèles finaux**, les constantes viennent des 5 plis.
  - **Ajout après la première évaluation** : une paire de constantes par tranche de 0,1 de la carte centrale,
    au lieu d'une seule pour toute la carte (voir Résultats). La règle de calcul est la même dans chaque tranche.

### Étape 3 : encodeur pré-entraîné sur Sentinel-2

Même U-Net à trois quantiles, avec un encodeur ResNet-50 pré-entraîné par apprentissage auto-supervisé
(MoCo) sur les 13 bandes Sentinel-2 de SSL4EO-S12 (poids `torchgeo/resnet50_sentinel2_all_moco`) :

- **Bandes** : nos 12 bandes sont dans le même ordre, sans B10. Le canal de B10 est retiré de la première
  couche (`02_poids.py`).
- **Normalisation** : le pré-entraînement a vu des réflectances brutes (comptes / 10 000). On lui donne nos
  réflectances telles quelles, sans la normalisation par bande de l'autre configuration.
- **Limite** : la comparaison mêle deux changements, le pré-entraînement et la taille (ResNet-50 contre
  ResNet-34). Les poids SSL4EO publiés n'existent pas en ResNet-34.

**Règle de choix, fixée d'avance** : le ResNet-50 SSL4EO est retenu si son score quantile hors pli (ensemble
de 5 graines, avant calibration) est meilleur que celui du ResNet-34 ImageNet dans **au moins 4 plis sur 5**.
Sinon, on garde le ResNet-34, plus léger.

### Configurations

| Configuration | Encodeur | Pré-entraînement | Sorties et perte | Entraînements |
|---|---|---|---|---|
| `l1_r34_imagenet` | ResNet-34 | ImageNet | 1 sortie, L1 (modèle actuel) | 5 plis × 5 graines = 25 |
| `q_r34_imagenet` | ResNet-34 | ImageNet | q10, q50, q90, perte quantile | 25 + 5 finaux = 30 |
| `q_r50_ssl4eo` | ResNet-50 | SSL4EO-S12 (Sentinel-2) | q10, q50, q90, perte quantile | 25 + 5 finaux = 30 |

Les modèles finaux sont entraînés sur les 49 sites et prédisent les sites externes.

### Mesures

- **Estimation centrale** (q50, ou moyenne des graines pour L1) : MAE, RMSE, biais, R².
- **Score quantile** : moyenne des pertes « pinball » à 10, 50 et 90 %. Plus bas est meilleur ; il récompense
  à la fois la justesse et des intervalles étroits.
- **Couverture** : part des labels sous la carte prudente (cible 10 %), au-dessus de la carte optimiste
  (cible 10 %), et dans l'intervalle (cible 80 %). On la calcule aussi par tranche de couvert et par site.
- **Largeur** moyenne de l'intervalle, et lien entre largeur et erreur par site (corrélation de rang).

Les résultats sont donnés pour l'ensemble de 5 graines et pour un modèle seul (moyenne des graines), avant et
après calibration.

## Résultats (17 novembre 2026)

Les 85 entraînements ont abouti. Les tableaux complets sont dans `resultats/` : `cv_*.csv`, `externe_*.csv`,
`couverture_tranches.csv`, `references_*.csv` et `resume.json`. Rapport en ligne (en anglais), avec un visualiseur
des cartes : https://claude.ai/artifact/UmtqSM7rsjjM6H5Tqihxz4.

**Deux changements après la première lecture des résultats**, sans réentraînement :
1. **Correction d'une erreur.** Le calcul des quantiles du mélange des graines utilisait la méthode linéaire de
   `np.quantile`. Elle resserrait l'intervalle d'environ 5 % de chaque demi-largeur : une graine seule ne redonnait
   pas son propre q10. La méthode est maintenant « weibull », qui correspond aux niveaux k/20.
2. **Calibration par tranche.** Avec une seule paire de constantes, la couverture de 80 % n'était qu'une moyenne :
   89 % pour les cases prédites près de 0 ou de 1, 68 % pour le couvert intermédiaire. Les constantes sont
   maintenant calculées par tranche de 0,1 de la carte centrale, toujours hors pli en validation croisée.

Le choix de l'encodeur, fixé d'avance sur le score quantile brut, ne dépend d'aucun de ces deux changements.

### Validation croisée (49 sites, 5 plis par longitude)

| Modèle | MAE | R² | Score quantile | Sous q10 | Au-dessus de q90 | Dans l'intervalle | Largeur |
|---|---|---|---|---|---|---|---|
| L1 ResNet-34, 5 graines (modèle actuel) | 0,0892 | 0,850 | — | — | — | — | — |
| Quantiles ResNet-34, 5 graines, brut | 0,0894 | 0,850 | 0,0285 | 14,2 % | 17,0 % | 68,8 % | 0,244 |
| Quantiles ResNet-34, 5 graines, calibration unique | 0,0894 | 0,850 | 0,0284 | 10,1 % | 10,0 % | 79,9 % | 0,257 |
| **Quantiles ResNet-34, 5 graines, calibration par tranche (modèle final)** | **0,0894** | **0,850** | **0,0281** | **10,1 %** | **9,1 %** | **80,8 %** | **0,279** |
| Quantiles ResNet-34, un seul modèle, brut | 0,0907 | 0,842 | 0,0291 | 13,3 % | 14,5 % | 72,2 % | 0,248 |
| Quantiles ResNet-50 SSL4EO, 5 graines, brut | 0,0908 | 0,849 | 0,0289 | 13,6 % | 19,1 % | 67,4 % | 0,242 |

- **Les trois quantiles ne coûtent rien à la carte centrale** : la MAE est de 0,0894, contre 0,0892 pour le
  modèle L1 actuel.
- **L'ensemble de 5 graines aide un peu** : la MAE passe de 0,0907 à 0,0894, et le score quantile de 0,0291 à
  0,0285.
- **Les intervalles bruts sont trop étroits** : 69 % de couverture au lieu de 80 %. Une grande partie des manques
  vient de labels à 0 ou à 1, que les bornes sigmoïdes n'atteignent jamais tout à fait.

### Encodeur pré-entraîné sur Sentinel-2 (étape 3)

**Le ResNet-50 SSL4EO n'est pas retenu.** Il n'est meilleur que dans 3 plis sur 5 (plis 1, 2 et 4), alors que
la règle fixée d'avance en exigeait 4. Globalement, il est même un peu moins bon : score quantile 0,0289
contre 0,0285, MAE 0,0908 contre 0,0894.

Le test externe confirme ce choix. Sur les 15 nouveaux sites ontariens, sa MAE est de 0,0946 contre 0,0875
pour le ResNet-34.

### Calibration par tranche de la carte centrale

Couverture de l'intervalle 10–90 % :

| Cases (selon la carte centrale) | Validation croisée, constante unique | Validation croisée, par tranche | 15 sites externes, constante unique | 15 sites externes, par tranche |
|---|---|---|---|---|
| Couvert intermédiaire (q50 entre 0,1 et 0,9) | 68 % | **79 %** | 76 % | **85 %** |
| Couvert extrême (q50 sous 0,1 ou dès 0,9) | 89 % | 82 % | 90 % | 83 % |

Les cases intermédiaires sont 45 % des cases en validation croisée et 52 % aux sites externes.

- **En validation croisée**, chaque tranche atteint 10 % sous la carte prudente et 10 % au-dessus de la carte
  optimiste. Les constantes de chaque pli viennent des 4 autres.
- **Les corrections** vont de presque 0 pour les tranches extrêmes à +0,12 sous la carte prudente vers q50 = 0,45.
  Elles sont dans `resume.json`.
- **Aux 15 sites externes**, la couverture d'ensemble est de 84 %. Pour q50 entre 0,1 et 0,5, les erreurs
  penchent d'un côté : 2 à 4 % des labels sont sous la carte prudente, 14 à 17 % au-dessus de la carte
  optimiste. Le modèle y sous-estime davantage le couvert partiel qu'aux sites du projet.

### Test externe (modèles finaux ResNet-34, constantes par tranche des 49 sites)

| Groupe | Sites | MAE | Sous q10 | Au-dessus de q90 | Dans l'intervalle | Largeur |
|---|---|---|---|---|---|---|
| Nouveaux sites ontariens | 15 | 0,0875 | 8,1 % | 7,6 % | 84,2 % | 0,302 |
| NEON A (forêts proches de l'Ontario) | 3 | 0,0720 | 10,2 % | 7,1 % | 82,7 % | 0,219 |
| NEON B (hors distribution) | 4 | 0,1349 | 14,3 % | 8,9 % | 76,8 % | 0,345 |

- **Sur les sites jamais vus, la couverture tient** : 84 % en Ontario, et chaque site est entre 75 % et 90 %.
- **L'intervalle s'élargit là où le modèle se trompe plus** : la corrélation de rang entre la largeur moyenne et
  la MAE des 15 sites ontariens est de 0,95.
- **Hors distribution, elle baisse** : 77 % dans le groupe NEON B, et 74 % à OSBS (savane de pins en Floride).

### Comparaison aux cartes existantes (`06_references.py`)

Meta/WRI CHM v2, SCANFI v2 et NTEMS 2022 sont mis sur la grille des labels avec `scripts/benchmarks.py`.
- **Version recalibrée** : comme dans le projet, une régression isotone vers le label, apprise sans les sites
  évalués (4 autres plis, ou les 49 sites pour les nouveaux sites).
- **Cases comparées** : seules celles où toutes les cartes sont valides. MAE à 10 m :

| Carte | Validation croisée (49 sites) | 15 nouveaux sites |
|---|---|---|
| **Modèle final, carte centrale** | **0,089** | **0,087** |
| Modèle actuel (L1), réentraîné dans les plis / modèle d'origine (3 graines) | 0,089 | 0,089 |
| Meta/WRI CHM v2, publié / recalibré | 0,148 / 0,154 | 0,191 / 0,164 |
| SCANFI v2, publié / recalibré | 0,219 / 0,156 | 0,212 / 0,155 |
| NTEMS 2022, publié / recalibré | 0,209 / 0,192 | 0,215 / 0,174 |

- **Sur les 15 nouveaux sites**, l'erreur est 44 % plus basse que celle de la meilleure carte existante (SCANFI
  recalibré).
- **Le modèle final a l'erreur la plus basse sur chacun des 15 sites**, contre chaque produit dans ses deux
  versions.
- **Limite** : le label vient de la même source lidar que celle sur laquelle le modèle a appris, ce qui l'avantage.

## Étapes

| Script | Rôle | Sortie |
|---|---|---|
| `01_plis.py` | 5 plis de validation croisée | `plis.csv` |
| `02_poids.py` | Poids SSL4EO adaptés à 12 bandes | `…/poids/resnet50_ssl4eo_12bandes.pth` |
| `03_entrainer.py` | Un entraînement (configuration, pli, graine) et ses prédictions | `…/predictions/`, `…/modeles/` (finaux) |
| `04_evaluer.py` | Ensembles, calibration, choix de l'encodeur, test externe, figures | `resultats/` |
| `05_cartes.py` | Cartes prudente, centrale et optimiste (GeoTIFF à 3 bandes) des sites externes et, hors pli, des 49 sites | `…/cartes/` |
| `06_references.py` | Comparaison à Meta/WRI CHM v2, SCANFI v2 et NTEMS 2022, publiés et recalibrés | `resultats/references_*.csv`, `…/references/` |

Lancement sur Slurm : `bash modeles-probabilistes/lancer.sh` (options `--etape N`, `--depuis N`, `--dry-run`).

- **Tableaux de tâches** : un par configuration, avec au plus 3 tâches GPU en même temps par configuration.
- **Reprise** : un entraînement déjà fait (fichier `FAIT`) est sauté. Relancer l'étape 3 ne refait que ce
  qui manque.

## Détails de mise en œuvre

- **Quota** : le volume de travail est presque plein (environ 97 Go sur 100, dont une vingtaine de Go hors
  de ce projet). Pour tenir dans la place restante :
  - les prédictions sont enregistrées en entiers 16 bits (couvert × 10 000, précision 0,0001) ;
  - les modèles de la validation croisée ne sont pas gardés, seulement leurs prédictions ;
  - les 10 modèles finaux sont gardés en demi-précision.

  Au total, environ 1,2 Go.
- **Durée mesurée** : environ 6 minutes par entraînement ResNet-34 et 8 minutes par ResNet-50, soit environ
  10 heures de GPU au total. L'évaluation prend 2 minutes sur CPU.
- **Test à petite échelle** : `PROBA_TRAVAIL` et `PROBA_SORTIES` redirigent les sorties. `03_entrainer.py`
  accepte `--pas` et `--max-sites`. Toute la chaîne a été essayée ainsi (4 pas, 2 sites) avant le lancement.
