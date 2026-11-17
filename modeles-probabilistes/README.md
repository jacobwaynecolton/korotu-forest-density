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

## Étapes

| Script | Rôle | Sortie |
|---|---|---|
| `01_plis.py` | 5 plis de validation croisée | `plis.csv` |
| `02_poids.py` | Poids SSL4EO adaptés à 12 bandes | `…/poids/resnet50_ssl4eo_12bandes.pth` |
| `03_entrainer.py` | Un entraînement (configuration, pli, graine) et ses prédictions | `…/predictions/`, `…/modeles/` (finaux) |
| `04_evaluer.py` | Ensembles, calibration, choix de l'encodeur, test externe, figures | `resultats/` |

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
- **Durée** : environ 7 minutes par entraînement ResNet-34 et 15 à 20 minutes par ResNet-50, soit 15 à 20
  heures de GPU, réparties sur plusieurs GPU en parallèle.
- **Test à petite échelle** : `PROBA_TRAVAIL` et `PROBA_SORTIES` redirigent les sorties. `03_entrainer.py`
  accepte `--pas` et `--max-sites`. Toute la chaîne a été essayée ainsi (4 pas, 2 sites) avant le lancement.
