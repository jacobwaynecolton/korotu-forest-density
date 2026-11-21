# Test de la fenêtre saisonnière des images Sentinel-2

**Question** : la sélection actuelle (`scripts/make_stacks.py`) garde des scènes du 1er mai au 15 octobre, alors que
les vols lidar sont tous entre le 1er juin et le 15 septembre. Les scènes de début ou de fin de saison, où le
feuillage des feuillus est incomplet, donnent-elles plus d'erreur ?

**Méthode** (`test_par_date.py`, sans réentraînement) :
- **Prédiction** : chaque image est prédite seule, uniquement par des modèles qui n'ont jamais vu le site :
  - les 5 modèles finaux de `modeles-probabilistes/` sur les 15 nouveaux sites ontariens ;
  - les 3 modèles d'origine du projet sur ses 6 sites de validation et de test ;
  - les 5 modèles finaux sur les 7 sites NEON, à titre descriptif.
- **Saisons** : « début » avant le 1er juin, « cœur » du 1er juin au 15 septembre, « fin » après le 15 septembre.
- **Comparaisons**, fixées avant le lancement :
  - chaque scène hors cœur contre la moyenne des scènes du cœur du même site ;
  - pour ces sites, MAE de la moyenne sur toutes les dates contre la moyenne sur les seules dates du cœur.

## Résultats (21 novembre 2026)

En Ontario, 12 scènes hors cœur (7 en début, 5 en fin) sur 11 sites :
- **Erreur** : 11 scènes sur 12 font plus d'erreur que les scènes du cœur du même site. L'écart de MAE est de
  +0,028 en médiane et +0,046 en moyenne.
- **Mai est plus touché** : écart médian de +0,036, contre +0,028 en fin de saison.
- **Biais** : le modèle sous-estime le couvert, de −0,086 en moyenne sur les cases de forêt (label ≥ 0,5).
- **Distance au vol** : elle n'explique pas l'écart. Parmi les scènes du cœur, celles à 25 jours ou plus du vol ne
  font que +0,003 de plus que les autres.
- **Pire cas** : le site de test `Z15_E0415_N5390_2020`, au 12 mai 2020. Le NDVI de la forêt dense y vaut 0,37,
  contre 0,86 à 0,90 en juin : les feuilles ne sont pas sorties. La MAE est supérieure de 0,24, et le biais en
  forêt de −0,64.
- **NEON** : les 2 scènes d'octobre font aussi plus d'erreur (+0,05 et +0,03).

**Effet sur les cartes moyennées sur les dates** : ne garder que les dates du cœur améliore 8 sites sur 11, de
0,006 en moyenne. Au site `Z15_E0415_N5390_2020`, la MAE passe de 0,134 à 0,110.

**Conclusion** : restreindre la fenêtre au 1er juin – 15 septembre, la même que celle des vols lidar.
- Aucun site du projet n'est perdu : chacun a au moins une scène dans cette fenêtre. Seules 26 scènes sur 181
  disparaissent.
- L'effet sur l'entraînement, qui perdrait des exemples contradictoires (label en feuilles, image sans
  feuilles), n'est pas mesuré ici.

**Limite** : 12 scènes hors cœur seulement. Les modèles testés ont eux-mêmes été entraînés avec des scènes hors
cœur.

Sorties : `resultats/scenes.csv` (une ligne par image), `resultats/comparaison_scenes.csv`,
`resultats/comparaison_sites.csv`, `resultats/resume.json`, `resultats/ecart_par_date.png`.
