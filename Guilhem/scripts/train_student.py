#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.checkpoints import load_checkpoint, save_checkpoint
from sdm.config import load_config
from sdm.data.collate import collate_sdm
from sdm.data.datasets import PADataset, POPseudoDataset, load_pseudo_records
from sdm.data.features import MultiModalFeatureStore
from sdm.data.metadata import load_pa_metadata, load_po_metadata
from sdm.data.splits import make_spatial_folds
from sdm.imbalance import build_pa_po_sample_weights
from sdm.runtime import normalizer_from_checkpoint
from sdm.train import build_model, train_model
from sdm.utils import ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train student model on PA + pseudo-labeled PO")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--pseudo-labels", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts/train_student")
    parser.add_argument("--disable-imbalance-sampler", action="store_true")
    parser.add_argument("--pa-po-ratio", type=float, default=None)
    parser.add_argument("--max-po-samples", type=int, default=None)
    parser.add_argument("--disable-pa-refine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    teacher_ckpt = load_checkpoint(args.teacher_checkpoint, device="cpu")
    cfg = teacher_ckpt.get("config") or load_config(args.config)
    cfg["data"]["data_root"] = args.data_root

    set_seed(int(cfg["seed"]))

    data_root = Path(cfg["data"]["data_root"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = ensure_dir(args.output_dir)

    species_to_index = {int(k): int(v) for k, v in teacher_ckpt["species_to_index"].items()}
    index_to_species = {int(k): int(v) for k, v in teacher_ckpt["index_to_species"].items()}
    normalizer = normalizer_from_checkpoint(teacher_ckpt["normalizer"])

    pa_meta = load_pa_metadata(data_root)
    if pa_meta.lat_col is None or pa_meta.lon_col is None:
        raise ValueError("PA metadata must contain latitude and longitude for spatial block CV")

    folds = make_spatial_folds(
        survey_df=pa_meta.survey_df,
        survey_id_col=pa_meta.survey_id_col,
        lat_col=pa_meta.lat_col,
        lon_col=pa_meta.lon_col,
        n_folds=int(cfg["training"]["folds"]),
        grid_size=float(cfg["data"]["spatial_grid_size"]),
    )
    fold_idx = int(teacher_ckpt.get("fold", 0))
    split = folds[fold_idx]

    pa_train_ids = [int(x) for x in split.train_ids]
    pa_val_ids = [int(x) for x in split.val_ids]

    max_train_samples = cfg["training"].get("max_train_samples")
    max_val_samples = cfg["training"].get("max_val_samples")

    if max_train_samples:
        pa_train_ids = pa_train_ids[: int(max_train_samples)]
    if max_val_samples:
        pa_val_ids = pa_val_ids[: int(max_val_samples)]

    pseudo_records = load_pseudo_records(args.pseudo_labels)
    po_train_ids = sorted(int(x) for x in pseudo_records.keys())
    max_po_samples = args.max_po_samples
    if max_po_samples is None:
        max_po_samples = cfg["training"].get("max_po_samples")
    if max_po_samples is None:
        max_po_samples = max_train_samples
    if max_po_samples:
        po_train_ids = po_train_ids[: int(max_po_samples)]

    po_meta = load_po_metadata(data_root, species_to_index=species_to_index)

    feature_store = MultiModalFeatureStore(
        data_root=data_root,
        cache_dir=cache_dir,
        bioclim_num_vars=int(cfg["model"]["temporal_encoder"]["bioclim_num_vars"]),
    )

    pa_train_ds = PADataset(
        survey_ids=pa_train_ids,
        labels=pa_meta.labels,
        species_to_index=species_to_index,
        split="pa_train",
        feature_store=feature_store,
        normalizer=normalizer,
    )
    po_train_ds = POPseudoDataset(
        survey_ids=po_train_ids,
        pseudo_records=pseudo_records,
        n_species=len(species_to_index),
        split="po_train",
        feature_store=feature_store,
        normalizer=normalizer,
    )
    val_ds = PADataset(
        survey_ids=pa_val_ids,
        labels=pa_meta.labels,
        species_to_index=species_to_index,
        split="pa_train",
        feature_store=feature_store,
        normalizer=normalizer,
    )

    train_ds = ConcatDataset([pa_train_ds, po_train_ds])

    num_workers = min(4, int(cfg["data"]["num_workers"]))
    sampling_cfg = cfg.get("sampling", {})
    sampling_enabled = bool(sampling_cfg.get("enabled", True)) and (not args.disable_imbalance_sampler)
    pa_po_ratio = float(args.pa_po_ratio) if args.pa_po_ratio is not None else float(sampling_cfg.get("pa_po_ratio", 2.0))
    weight_min = float(sampling_cfg.get("min_weight", 0.2))
    weight_max = float(sampling_cfg.get("max_weight", 5.0))
    po_tabular_only_factor = float(sampling_cfg.get("po_tabular_only_factor", 0.2))

    po_modality_factors = np.ones((len(po_train_ids),), dtype=np.float64)
    po_has_patch = 0
    po_has_landsat = 0
    po_has_bioclim = 0
    po_has_non_tabular = 0
    for i, sid in enumerate(po_train_ids):
        avail = feature_store.availability(int(sid), split="po_train")
        if avail["has_patch"]:
            po_has_patch += 1
        if avail["has_landsat"]:
            po_has_landsat += 1
        if avail["has_bioclim"]:
            po_has_bioclim += 1
        if avail["has_non_tabular"]:
            po_has_non_tabular += 1
        else:
            po_modality_factors[i] = po_tabular_only_factor

    sampler = None
    if sampling_enabled:
        imbalance = build_pa_po_sample_weights(
            pa_train_ids=pa_train_ids,
            po_train_ids=po_train_ids,
            pa_labels=pa_meta.labels,
            pseudo_records=pseudo_records,
            species_to_index=species_to_index,
            pa_survey_df=pa_meta.survey_df,
            pa_survey_id_col=pa_meta.survey_id_col,
            pa_lat_col=pa_meta.lat_col,
            pa_lon_col=pa_meta.lon_col,
            po_survey_df=po_meta.survey_df,
            po_survey_id_col=po_meta.survey_id_col,
            po_lat_col=po_meta.lat_col,
            po_lon_col=po_meta.lon_col,
            grid_size=float(cfg["data"]["spatial_grid_size"]),
            target_pa_ratio=pa_po_ratio,
            min_weight=weight_min,
            max_weight=weight_max,
        )
        po_weights = imbalance.po_weights * po_modality_factors
        train_weights = np.concatenate([imbalance.pa_weights, po_weights], axis=0)
        mean_w = float(np.mean(train_weights))
        if mean_w > 0:
            train_weights = train_weights / mean_w
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=int(train_weights.shape[0]),
            replacement=True,
        )

        imbalance_report = dict(imbalance.report)
        imbalance_report["fold"] = int(fold_idx)
        imbalance_report["sampling_enabled"] = True
        imbalance_report["pa_po_ratio_used"] = float(pa_po_ratio)
        imbalance_report["po_tabular_only_factor"] = float(po_tabular_only_factor)
        imbalance_report["max_po_samples_used"] = int(len(po_train_ids))
        imbalance_report["po_non_tabular_rate"] = float(po_has_non_tabular / max(1, len(po_train_ids)))
        imbalance_report["po_patch_rate"] = float(po_has_patch / max(1, len(po_train_ids)))
        imbalance_report["po_landsat_rate"] = float(po_has_landsat / max(1, len(po_train_ids)))
        imbalance_report["po_bioclim_rate"] = float(po_has_bioclim / max(1, len(po_train_ids)))
    else:
        imbalance_report = {
            "fold": int(fold_idx),
            "sampling_enabled": False,
            "reason": "disabled by CLI or config",
            "max_po_samples_used": int(len(po_train_ids)),
            "po_non_tabular_rate": float(po_has_non_tabular / max(1, len(po_train_ids))),
            "po_patch_rate": float(po_has_patch / max(1, len(po_train_ids))),
            "po_landsat_rate": float(po_has_landsat / max(1, len(po_train_ids))),
            "po_bioclim_rate": float(po_has_bioclim / max(1, len(po_train_ids))),
        }

    imbalance_path = output_dir / f"imbalance_report_fold{fold_idx}.json"
    write_json(imbalance_report, imbalance_path)
    print(f"Saved imbalance report: {imbalance_path}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sdm,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sdm,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        num_species=len(species_to_index),
        landsat_channels=int(teacher_ckpt["input_dims"]["landsat_channels"]),
        bioclim_channels=int(teacher_ckpt["input_dims"]["bioclim_channels"]),
        tabular_dim=int(teacher_ckpt["input_dims"]["tabular_dim"]),
        model_cfg=cfg["model"],
        device=device,
    )
    model.load_state_dict(teacher_ckpt["state_dict"], strict=False)

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

    final_result = result
    selected_phase = "mixed_pa_po"
    refine_cfg = cfg.get("student_refine", {})
    refine_enabled = bool(refine_cfg.get("enabled", True)) and (not args.disable_pa_refine)
    if refine_enabled:
        refine_training_cfg = dict(training_cfg)
        refine_training_cfg["epochs"] = int(refine_cfg.get("epochs", 3))
        refine_training_cfg["patience"] = int(refine_cfg.get("patience", 2))
        refine_training_cfg["lr"] = float(refine_cfg.get("lr", float(training_cfg["lr"]) * 0.5))
        refine_batch_size = int(refine_cfg.get("batch_size", cfg["training"]["batch_size"]))

        pa_refine_loader = DataLoader(
            pa_train_ds,
            batch_size=refine_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_sdm,
        )

        model.load_state_dict(result.best_state_dict)
        refine_result = train_model(
            model=model,
            train_loader=pa_refine_loader,
            val_loader=val_loader,
            training_cfg=refine_training_cfg,
            min_k=int(cfg["data"]["min_species_per_sample"]),
            max_k=int(cfg["data"]["max_species_per_sample"]),
            device=device,
        )
        if refine_result.best_metrics.get("f1_samples", -1.0) >= result.best_metrics.get("f1_samples", -1.0):
            final_result = refine_result
            selected_phase = "pa_only_refine"
        else:
            selected_phase = "mixed_pa_po_kept_after_refine"

    ckpt_path = output_dir / f"student_fold{fold_idx}.pt"
    save_checkpoint(
        ckpt_path,
        {
            "state_dict": final_result.best_state_dict,
            "config": cfg,
            "fold": int(fold_idx),
            "species_to_index": species_to_index,
            "index_to_species": index_to_species,
            "normalizer": teacher_ckpt["normalizer"],
            "input_dims": teacher_ckpt["input_dims"],
            "metrics": final_result.best_metrics,
            "selected_phase": selected_phase,
        },
    )

    metrics_path = output_dir / f"student_fold{fold_idx}_metrics.json"
    payload_metrics = {
        "selected_phase": selected_phase,
        "mixed_pa_po": result.best_metrics,
        "selected": final_result.best_metrics,
    }
    write_json(payload_metrics, metrics_path)

    print(f"Saved student checkpoint: {ckpt_path}")
    print(f"Best metrics: {json.dumps(payload_metrics, indent=2)}")


if __name__ == "__main__":
    main()
