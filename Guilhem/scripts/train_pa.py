#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.checkpoints import save_checkpoint
from sdm.config import load_config
from sdm.data.collate import collate_sdm
from sdm.data.datasets import PADataset, fit_normalizer
from sdm.data.features import MultiModalFeatureStore
from sdm.data.metadata import build_species_maps, load_pa_metadata
from sdm.data.splits import make_spatial_folds
from sdm.train import build_model, train_model
from sdm.utils import ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train teacher model on PA data")
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="artifacts/train_pa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["data_root"] = args.data_root

    set_seed(int(cfg["seed"]))

    data_root = Path(cfg["data"]["data_root"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = ensure_dir(args.output_dir)

    pa_meta = load_pa_metadata(data_root)
    if pa_meta.lat_col is None or pa_meta.lon_col is None:
        raise ValueError("PA metadata must contain latitude and longitude for spatial block CV")

    species_to_index, index_to_species = build_species_maps(pa_meta.species_ids)
    survey_ids = pa_meta.survey_df[pa_meta.survey_id_col].astype(int).to_numpy()

    folds = make_spatial_folds(
        survey_df=pa_meta.survey_df,
        survey_id_col=pa_meta.survey_id_col,
        lat_col=pa_meta.lat_col,
        lon_col=pa_meta.lon_col,
        n_folds=int(cfg["training"]["folds"]),
        grid_size=float(cfg["data"]["spatial_grid_size"]),
    )
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"Fold index {args.fold} out of range [0,{len(folds)-1}]")

    split = folds[args.fold]
    train_ids = split.train_ids
    val_ids = split.val_ids

    max_train_samples = cfg["training"].get("max_train_samples")
    max_val_samples = cfg["training"].get("max_val_samples")
    if max_train_samples:
        train_ids = train_ids[: int(max_train_samples)]
    if max_val_samples:
        val_ids = val_ids[: int(max_val_samples)]

    feature_store = MultiModalFeatureStore(
        data_root=data_root,
        cache_dir=cache_dir,
        bioclim_num_vars=int(cfg["model"]["temporal_encoder"]["bioclim_num_vars"]),
    )

    normalizer = fit_normalizer(feature_store, split="pa_train", survey_ids=train_ids)

    train_ds = PADataset(
        survey_ids=train_ids,
        labels=pa_meta.labels,
        species_to_index=species_to_index,
        split="pa_train",
        feature_store=feature_store,
        normalizer=normalizer,
    )
    val_ds = PADataset(
        survey_ids=val_ids,
        labels=pa_meta.labels,
        species_to_index=species_to_index,
        split="pa_train",
        feature_store=feature_store,
        normalizer=normalizer,
    )

    num_workers = min(4, int(cfg["data"]["num_workers"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sdm,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sdm,
    )

    dims = feature_store.infer_dims("pa_train", train_ids[:1])
    landsat_channels = int(dims["landsat"][0])
    bioclim_channels = int(dims["bioclim_ts"][0])
    tabular_dim = int(dims["tabular"][0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        num_species=len(species_to_index),
        landsat_channels=landsat_channels,
        bioclim_channels=bioclim_channels,
        tabular_dim=tabular_dim,
        model_cfg=cfg["model"],
        device=device,
    )

    training_cfg = dict(cfg["training"])
    training_cfg["log_interval"] = int(cfg["output"]["log_interval"])

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        min_k=int(cfg["data"]["min_species_per_sample"]),
        max_k=int(cfg["data"]["max_species_per_sample"]),
        device=device,
    )

    ckpt_path = output_dir / f"teacher_fold{args.fold}.pt"
    save_checkpoint(
        ckpt_path,
        {
            "state_dict": result.best_state_dict,
            "config": cfg,
            "fold": int(args.fold),
            "species_to_index": species_to_index,
            "index_to_species": index_to_species,
            "normalizer": {
                "patch_mean": normalizer.patch_mean,
                "patch_std": normalizer.patch_std,
                "landsat_mean": normalizer.landsat_mean,
                "landsat_std": normalizer.landsat_std,
                "landsat_seq_len": normalizer.landsat_seq_len,
                "bioclim_mean": normalizer.bioclim_mean,
                "bioclim_std": normalizer.bioclim_std,
                "bioclim_seq_len": normalizer.bioclim_seq_len,
                "tabular_mean": normalizer.tabular_mean,
                "tabular_std": normalizer.tabular_std,
            },
            "input_dims": {
                "landsat_channels": landsat_channels,
                "bioclim_channels": bioclim_channels,
                "tabular_dim": tabular_dim,
            },
            "metrics": result.best_metrics,
        },
    )

    metrics_path = output_dir / f"teacher_fold{args.fold}_metrics.json"
    write_json(result.best_metrics, metrics_path)

    print(f"Saved teacher checkpoint: {ckpt_path}")
    print(f"Best metrics: {json.dumps(result.best_metrics, indent=2)}")


if __name__ == "__main__":
    main()
