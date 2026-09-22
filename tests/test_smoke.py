"""Test de fumée : instancie les modèles multimodaux et fait un passage avant sur des tenseurs aléatoires.

Aucune donnée GeoLifeCLEF ni GPU n'est nécessaire. Les poids ImageNet ne sont pas téléchargés :
les constructeurs torchvision sont appelés avec ``weights=None`` (seule l'architecture est testée).

Dimensions reprises du code et des sorties réelles de l'équipe :
- image Sentinel-2 : 4 canaux (R, G, B, NIR), 64 x 64 pixels (``yasmina/train.py``) ;
- série Landsat : 21 pas de temps x 24 variables (6 bandes x 4 saisons, ``yasmina/train.py``) ;
- série bioclimatique : 12 pas de temps x 76 variables (``yasmina/train.py``) ;
- env_dim = 64, aux_dim = 57, n_species = 5016 (``results/validation/*.json``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torchvision.models as tvm

ROOT = Path(__file__).resolve().parents[1]
BATCH = 2
ENV_DIM, AUX_DIM, N_SPECIES = 64, 57, 5016
LANDSAT_SHAPE = (21, 24)
CLIMATE_SHAPE = (12, 76)
IMAGE_SHAPE = (4, 64, 64)


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_pretrained_download(monkeypatch):
    """Remplace les constructeurs torchvision pour ignorer les poids pré-entraînés."""
    for builder in ("efficientnet_b3", "resnet50", "convnext_tiny"):
        original = getattr(tvm, builder)
        monkeypatch.setattr(tvm, builder, lambda *a, _f=original, **k: _f(weights=None))
    torch.manual_seed(0)


def _random_inputs():
    return dict(
        env_features=torch.randn(BATCH, ENV_DIM),
        aux_features=torch.randn(BATCH, AUX_DIM),
        landsat_series=torch.randn(BATCH, *LANDSAT_SHAPE),
        climate_series=torch.randn(BATCH, *CLIMATE_SHAPE),
        image=torch.randn(BATCH, *IMAGE_SHAPE),
    )


def test_dimensions_match_real_training_summaries():
    for path in (ROOT / "results" / "validation").glob("*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert (summary["env_dim"], summary["aux_dim"], summary["n_species"]) == (ENV_DIM, AUX_DIM, N_SPECIES)


@pytest.mark.parametrize("use_image_branch", [False, True])
def test_yasmina_model_forward(use_image_branch):
    module = _load_module("yasmina_model", "yasmina/model.py")
    model = module.MultimodalPlantModel(
        n_species=N_SPECIES, env_dim=ENV_DIM, aux_dim=AUX_DIM, use_image_branch=use_image_branch
    ).eval()
    inputs = _random_inputs()
    if not use_image_branch:
        inputs["image"] = None
    with torch.no_grad():
        logits = model(**inputs)
    assert logits.shape == (BATCH, N_SPECIES)
    assert torch.isfinite(logits).all()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"yasmina/model.py image={use_image_branch} -> sortie {tuple(logits.shape)}, {n_params:,} paramètres")


@pytest.mark.parametrize("backbone", ["efficientnet_b3", "resnet50", "convnext_tiny"])
def test_v6_model_forward(backbone):
    module = _load_module("v6_model", "Adrian/v6_package/model.py")
    model = module.MultimodalPlantModel(
        n_species=N_SPECIES, env_dim=ENV_DIM, aux_dim=AUX_DIM, use_image_branch=True, image_backbone=backbone
    ).eval()
    with torch.no_grad():
        logits = model(**_random_inputs())
    assert logits.shape == (BATCH, N_SPECIES)
    assert torch.isfinite(logits).all()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Adrian/v6_package/model.py {backbone} -> sortie {tuple(logits.shape)}, {n_params:,} paramètres")
    del model
