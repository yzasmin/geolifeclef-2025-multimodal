from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "data_root": os.environ.get("GLC_DATA_DIR", "data"),
        "cache_dir": os.environ.get("GLC_CACHE_DIR", ".cache"),
        "num_workers": 2,
        "spatial_grid_size": 1.0,
        "min_species_per_sample": 1,
        "max_species_per_sample": 50,
    },
    "model": {
        "embed_dim": 256,
        "attn_heads": 4,
        "dropout": 0.1,
        "patch_encoder": {
            "channels": [32, 64, 128],
            "token_dim": 128,
        },
        "temporal_encoder": {
            "hidden_dim": 128,
            "kernel_size": 5,
            "num_layers": 3,
            "bioclim_num_vars": 4,
        },
        "tabular_encoder": {
            "hidden_dims": [256, 128],
            "dropout": 0.1,
        },
    },
    "training": {
        "batch_size": 64,
        "epochs": 30,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "amp": True,
        "grad_clip": 1.0,
        "patience": 5,
        "folds": 5,
        "gamma_pos": 0.0,
        "gamma_neg": 2.0,
        "loss_clip": 0.05,
        "set_size_weight": 0.2,
        "max_train_samples": 600000,
        "max_val_samples": 75000,
        "max_po_samples": 400000,
    },
    "student_refine": {
        "enabled": True,
        "epochs": 3,
        "patience": 2,
        "lr": 1e-4,
    },
    "pseudo_label": {
        "t_pos": 0.5,
        "t_neg": 0.03,
        "max_negatives_per_sample": 64,
        "batch_size": 256,
    },
    "sampling": {
        "enabled": True,
        "pa_po_ratio": 2.0,
        "min_weight": 0.2,
        "max_weight": 5.0,
        "po_tabular_only_factor": 0.2,
    },
    "inference": {
        "default_threshold": 0.2,
        "set_size_alpha": 1.0,
        "max_k_calibrated_fallback": 30,
    },
    "output": {
        "artifacts_dir": "artifacts",
        "log_interval": 50,
    },
}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return cfg
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("Config file must define a mapping")
    return _deep_merge(cfg, user_cfg)
