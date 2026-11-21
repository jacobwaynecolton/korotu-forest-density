# Audit du décalage BOA (−0,1) des images Sentinel-2

**Question** : la fonction `boa_offset_applied` de `scripts/make_pairs.py` (trois branches : drapeau du catalogue,
absence de décalage, test des pixels) est-elle utile en pratique ?

**Méthode** : pour les 254 scènes utilisées (181 du projet, 51 des nouveaux sites ontariens, 22 NEON), on relève
dans le catalogue Earth Search (`sentinel-2-l2a`) la version de traitement, le drapeau
`earthsearch:boa_offset_applied` et le décalage des métadonnées. Le test des pixels n'est fait que si les deux
premiers ne tranchent pas, comme dans la fonction.

## Résultats (21 novembre 2026)

| Branche qui décide | Scènes | Décision |
|---|---|---|
| Drapeau du catalogue vrai (versions 04.00 à 05.11) | 195 | décalage déjà retiré |
| Aucun décalage dans les métadonnées (versions 02.12 à 03.01) | 57 | rien à retirer |
| Test des pixels | 2 | décalage déjà retiré |

Les 2 scènes du test des pixels sont celles du site `Z16_E0420_N5540_2022` (version 04.00, drapeau faux alors que
les pixels sont déjà corrigés), qui avaient causé le bug du double décalage.

**Conclusion** : sur les 254 scènes, le décalage n'est jamais ajouté. Avec cette collection, la conversion se
résume à réflectance = DN × 0,0001. La fonction peut devenir cette seule règle, plus une alarme qui arrête le
pipeline si une scène semble encore contenir le décalage (1er percentile du bleu ≥ 1 000 en DN).

**Réserve** : la collection retraitée `sentinel-2-c1-l2a` d'Earth Search suit une autre convention (version 05.xx
partout, décalage −0,1 dans les métadonnées, pas de drapeau). À la tuile testée, l'année 2022 y manquait. Un
changement de collection demanderait un inventaire, et sa propre règle.
