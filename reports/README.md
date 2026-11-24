# Sources des rapports en ligne

- `modele-final/` : rapport « Final Canopy Model » (https://claude.ai/artifact/NMmhueNPeL5DC4KdupuHCN). `preparer.py`
  produit `data.json` et les images des 39 sites de test ; la page est `body.html`, où `__STYLE__`, `__HELPERS__` et
  `__VIEWER_HEAD__` sont remplacés par les trois autres fichiers.
- `modeles-probabilistes/` : rapport « Probabilistic Canopy Maps » (https://claude.ai/artifact/UmtqSM7rsjjM6H5Tqihxz4).
  `preparer.py` produit `donnees.json` et les images ; `__DONNEES__` dans `template.html` reçoit le JSON.

Les images et les données ne sont pas dans le dépôt (plusieurs dizaines de Mo) : relancer `preparer.py`.
