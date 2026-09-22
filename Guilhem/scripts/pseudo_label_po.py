#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.checkpoints import load_checkpoint
from sdm.config import load_config
from sdm.data.collate import collate_sdm
from sdm.data.datasets import TestDataset
from sdm.data.features import MultiModalFeatureStore
from sdm.data.metadata import load_po_metadata
from sdm.pseudo_label import build_pseudo_label_records
from sdm.runtime_overrides import deep_merge_dicts, resolve_int_with_source
from sdm.runtime import normalizer_from_checkpoint
from sdm.train import build_model
from sdm.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate masked pseudo labels for PO data")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="artifacts/pseudo/pseudo_labels.pt")
    parser.add_argument("--t-pos", type=float, default=None)
    parser.add_argument("--t-neg", type=float, default=None)
    parser.add_argument("--max-negatives-per-sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stats-json", type=str, default=None)
    return parser.parse_args()
def _resolve_cfg(teacher_ckpt: dict, config_path: str | None, data_root: str) -> dict:
    # Priority baseline: YAML/defaults, then checkpoint config override.
    yaml_cfg = load_config(config_path)
    ckpt_cfg = teacher_ckpt.get("config") or {}
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}
    cfg = deep_merge_dicts(yaml_cfg, ckpt_cfg)
    cfg["data"]["data_root"] = data_root
    return cfg


def main() -> None:
    args = parse_args()

    print("Loading teacher checkpoint...", flush=True)
    teacher_ckpt = load_checkpoint(args.teacher_checkpoint, device="cpu")
    cfg = _resolve_cfg(teacher_ckpt=teacher_ckpt, config_path=args.config, data_root=args.data_root)

    species_to_index = {int(k): int(v) for k, v in teacher_ckpt["species_to_index"].items()}
    normalizer = normalizer_from_checkpoint(teacher_ckpt["normalizer"])

    data_root = Path(cfg["data"]["data_root"])
    cache_dir = Path(cfg["data"]["cache_dir"])

    print("Loading PO metadata...", flush=True)
    po_meta = load_po_metadata(data_root, species_to_index=species_to_index)
    survey_ids = po_meta.survey_df[po_meta.survey_id_col].astype(int).to_numpy()
    print(f"PO surveys: {len(survey_ids)}", flush=True)

    feature_store = MultiModalFeatureStore(
        data_root=data_root,
        cache_dir=cache_dir,
        bioclim_num_vars=int(cfg["model"]["temporal_encoder"]["bioclim_num_vars"]),
    )

    po_ds = TestDataset(
        survey_ids=survey_ids,
        split="po_train",
        feature_store=feature_store,
        normalizer=normalizer,
    )

    ckpt_cfg = teacher_ckpt.get("config") if isinstance(teacher_ckpt.get("config"), dict) else {}
    ckpt_pseudo = ckpt_cfg.get("pseudo_label", {}) if isinstance(ckpt_cfg, dict) else {}
    ckpt_data = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else {}

    default_batch_size = int(cfg["pseudo_label"]["batch_size"])
    default_num_workers = int(cfg["data"]["num_workers"])
    batch_size, batch_source = resolve_int_with_source(
        cli_value=args.batch_size,
        ckpt_value=ckpt_pseudo.get("batch_size") if isinstance(ckpt_pseudo, dict) else None,
        default_value=default_batch_size,
    )
    num_workers_raw, num_workers_source = resolve_int_with_source(
        cli_value=args.num_workers,
        ckpt_value=ckpt_data.get("num_workers") if isinstance(ckpt_data, dict) else None,
        default_value=default_num_workers,
    )
    num_workers = min(4, int(num_workers_raw))
    if num_workers != num_workers_raw:
        num_workers_source = f"{num_workers_source}_clipped_to_4"

    po_loader = DataLoader(
        po_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_sdm,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building model on device={device}...", flush=True)
    model = build_model(
        num_species=len(species_to_index),
        landsat_channels=int(teacher_ckpt["input_dims"]["landsat_channels"]),
        bioclim_channels=int(teacher_ckpt["input_dims"]["bioclim_channels"]),
        tabular_dim=int(teacher_ckpt["input_dims"]["tabular_dim"]),
        model_cfg=cfg["model"],
        device=device,
    )
    model.load_state_dict(teacher_ckpt["state_dict"])
    model.eval()

    t_pos = float(args.t_pos) if args.t_pos is not None else float(cfg["pseudo_label"]["t_pos"])
    t_neg = float(args.t_neg) if args.t_neg is not None else float(cfg["pseudo_label"]["t_neg"])
    max_negatives_per_sample = (
        int(args.max_negatives_per_sample)
        if args.max_negatives_per_sample is not None
        else int(cfg["pseudo_label"]["max_negatives_per_sample"])
    )

    expected_steps = int(math.ceil(len(survey_ids) / max(1, int(batch_size))))
    print(
        (
            "Pseudo-label runtime config | "
            f"PO samples={len(survey_ids)} | "
            f"batch_size={batch_size} (source={batch_source}) | "
            f"num_workers={num_workers} (source={num_workers_source}) | "
            f"expected_steps={expected_steps}"
        ),
        flush=True,
    )

    print("Starting streaming pseudo-label inference...", flush=True)
    records: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(po_loader, desc="pseudo-label", total=len(po_loader), leave=True):
            survey_ids_batch = [int(x) for x in batch["survey_id"]]
            moved_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(device, non_blocking=True)
                else:
                    moved_batch[key] = value

            outputs = model(
                patch=moved_batch["patch"],
                landsat=moved_batch["landsat"],
                bioclim_ts=moved_batch["bioclim_ts"],
                tabular=moved_batch["tabular"],
            )
            probs = torch.sigmoid(outputs["logits"]).detach().cpu()

            batch_records = build_pseudo_label_records(
                survey_ids=survey_ids_batch,
                probs=probs,
                observed_labels=po_meta.labels,
                species_to_index=species_to_index,
                t_pos=t_pos,
                t_neg=t_neg,
                max_negatives_per_sample=max_negatives_per_sample,
            )
            records.extend(batch_records)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    torch.save(records, output_path)
    print(f"Saved pseudo labels: {output_path}", flush=True)
    print(f"Records: {len(records)}", flush=True)

    n_species = len(species_to_index)
    pos_counts = [len(row["positive_idx"]) for row in records]
    neg_counts = [len(row["negative_idx"]) for row in records]
    ignore_counts = [max(0, n_species - p - n) for p, n in zip(pos_counts, neg_counts)]

    stats = {
        "fold": int(teacher_ckpt.get("fold", 0)),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "output": str(output_path),
        "num_records": int(len(records)),
        "num_species": int(n_species),
        "t_pos": float(t_pos),
        "t_neg": float(t_neg),
        "max_negatives_per_sample": int(max_negatives_per_sample),
        "effective_batch_size": int(batch_size),
        "effective_num_workers": int(num_workers),
        "batch_size_source": batch_source,
        "num_workers_source": num_workers_source,
        "expected_steps": int(expected_steps),
        "mean_positive_labels": float(sum(pos_counts) / max(1, len(pos_counts))),
        "mean_negative_labels": float(sum(neg_counts) / max(1, len(neg_counts))),
        "mean_ignored_labels": float(sum(ignore_counts) / max(1, len(ignore_counts))),
        "positive_rate": float(sum(pos_counts) / max(1, len(records) * n_species)),
        "negative_rate": float(sum(neg_counts) / max(1, len(records) * n_species)),
        "ignored_rate": float(sum(ignore_counts) / max(1, len(records) * n_species)),
    }

    if args.stats_json:
        stats_path = Path(args.stats_json)
    else:
        fold_idx = int(teacher_ckpt.get("fold", 0))
        stats_path = output_path.parent / f"pseudo_labels_fold{fold_idx}_stats.json"
    ensure_dir(stats_path.parent)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f"Saved pseudo-label stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
