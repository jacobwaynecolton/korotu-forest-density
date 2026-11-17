"""Étape 4 : labels à 10 m sur la grille Sentinel-2, avec le script de l'étude ontarienne (../model-verification/03_labels.py).

Ce script fixe MODELE_TRAVAIL (ce dossier) et MODELE_SORTIES (données lourdes américaines) avant d'exécuter
celui de l'Ontario, qui travaille alors dans nos dossiers. Une valeur déjà définie est respectée (tests).
Usage : python -I independant-verification/model-verification-USA/04_labels.py
"""
import os
import runpy
from pathlib import Path

ICI = Path(__file__).resolve().parent
os.environ.setdefault("MODELE_TRAVAIL", str(ICI))
os.environ.setdefault("MODELE_SORTIES", "/w/20261/tsalin/MLOps/independant_verification/modele_usa")
runpy.run_path(str(ICI.parent / "model-verification" / "03_labels.py"), run_name="__main__")
