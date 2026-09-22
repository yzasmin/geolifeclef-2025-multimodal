#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.checkpoints import load_checkpoint
from sdm.data.collate import collate_sdm
from sdm.data.datasets import TestDataset
from sdm.data.features import MultiModalFeatureStore
from sdm.data.metadata import load_pa_test_metadata
from sdm.inference import decode_topk_predictions, validate_submission, write_submission
from sdm.runtime import normalizer_from_checkpoint
from sdm.train import build_model, predict_scores
from sdm.utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PA-test and generate Kaggle submission")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--output-csv", type=str, default="artifacts/submission.csv")
    parser.add_argument("--calibration-json", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    cfg = ckpt["config"]
    cfg["data"]["data_root"] = args.data_root

    data_root = Path(cfg["data"]["data_root"])
    cache_dir = Path(cfg["data"]["cache_dir"])

    normalizer = normalizer_from_checkpoint(ckpt["normalizer"])
    index_to_species = {int(k): int(v) for k, v in ckpt["index_to_species"].items()}

    alpha = float(cfg.get("inference", {}).get("set_size_alpha", 1.0))
    max_k = int(cfg.get("inference", {}).get("max_k_calibrated_fallback", 30))
    if args.calibration_json:
        calib = read_json(args.calibration_json)
        alpha = float(calib.get("alpha", alpha))
        max_k = int(calib.get("max_k", max_k))
    if args.alpha is not None:
        alpha = float(args.alpha)
    if args.max_k is not None:
        max_k = int(args.max_k)

    test_meta = load_pa_test_metadata(data_root)
    survey_ids = test_meta.survey_df[test_meta.survey_id_col].astype(int).to_numpy()

    feature_store = MultiModalFeatureStore(
        data_root=data_root,
        cache_dir=cache_dir,
        bioclim_num_vars=int(cfg["model"]["temporal_encoder"]["bioclim_num_vars"]),
    )

    test_ds = TestDataset(
        survey_ids=survey_ids,
        split="pa_test",
        feature_store=feature_store,
        normalizer=normalizer,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["pseudo_label"]["batch_size"]),
        shuffle=False,
        num_workers=min(4, int(cfg["data"]["num_workers"])),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sdm,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    species_to_index = {int(k): int(v) for k, v in ckpt["species_to_index"].items()}
    model = build_model(
        num_species=len(species_to_index),
        landsat_channels=int(ckpt["input_dims"]["landsat_channels"]),
        bioclim_channels=int(ckpt["input_dims"]["bioclim_channels"]),
        tabular_dim=int(ckpt["input_dims"]["tabular_dim"]),
        model_cfg=cfg["model"],
        device=device,
    )
    model.load_state_dict(ckpt["state_dict"])

    outputs = predict_scores(model, test_loader, device=device)
    predictions = decode_topk_predictions(
        probs=outputs["probs"],
        set_sizes=outputs["set_sizes"],
        index_to_species=index_to_species,
        min_k=int(cfg["data"]["min_species_per_sample"]),
        max_k=max_k,
        alpha=alpha,
    )

    output_csv = write_submission(
        survey_ids=outputs["survey_ids"],
        predictions=predictions,
        output_csv=args.output_csv,
    )

    is_valid, errors = validate_submission(output_csv, valid_species_ids=set(species_to_index.keys()))
    if not is_valid:
        raise RuntimeError("Submission validation failed:\n" + "\n".join(errors[:20]))

    print(f"Saved submission: {output_csv}")
    print(f"Inference calibration used: alpha={alpha:.4f}, max_k={max_k}")


if __name__ == "__main__":
    main()
