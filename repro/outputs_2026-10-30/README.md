# Sorties de la reproduction du 30 octobre 2026

Le pipeline de la première version (`repro/lancer_tout.sh`) a été relancé en entier sur le cluster. Ce dossier garde
les fichiers qu'il a réécrits, aux mêmes chemins relatifs que dans le dépôt ; les fichiers d'origine de Jacob restent à
leur place, inchangés.

`comparaison.txt` résume l'écart avec les résultats d'origine : labels identiques (écart max 5,6e-17), 181 images,
MAE de l'ensemble 0,1020 en validation (0,1034 d'origine) et 0,1077 en test (0,1063), cartes de référence identiques,
10 vérifications sur 10 réussies. Les écarts sur le modèle viennent du non-déterminisme de l'entraînement sur GPU.
