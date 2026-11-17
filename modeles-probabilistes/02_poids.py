"""Étape 2 : télécharger les poids SSL4EO-S12 (ResNet-50, MoCo, 13 bandes Sentinel-2) et les adapter à nos 12 bandes.

Nos images ont les bandes de SSL4EO dans le même ordre, sans B10 (absente des produits L2A) : on retire le canal
de B10 de la première couche. Les autres couches sont reprises telles quelles ; la couche de classification
finale (fc) n'est pas utilisée par le U-Net.
Sortie : SORTIES/poids/resnet50_ssl4eo_12bandes.pth
Usage : python -I modeles-probabilistes/02_poids.py
"""
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commun import POIDS, POIDS_SSL4EO  # noqa: E402

B10 = 10   # position de B10 dans l'ordre B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12


def main():
    POIDS.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(*POIDS_SSL4EO, cache_dir=POIDS / "hf")
    sd = torch.load(source, map_location="cpu", weights_only=True)
    w = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    c1 = w["conv1.weight"]
    assert c1.shape[1] == 13, c1.shape
    w["conv1.weight"] = torch.cat([c1[:, :B10], c1[:, B10 + 1:]], dim=1)
    torch.save(w, POIDS / "resnet50_ssl4eo_12bandes.pth")
    print(f"{len(w)} tenseurs, première couche {tuple(w['conv1.weight'].shape)} -> {POIDS / 'resnet50_ssl4eo_12bandes.pth'}")


if __name__ == "__main__":
    main()
