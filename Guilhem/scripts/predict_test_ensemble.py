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
    parser = argparse.ArgumentParser(description="Predict PA-test with checkpoint ensemble")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--output-csv", type=str, default="artifacts/submission_ensemble.csv")
    parser.add_argument("--calibration-json", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_probs = []
    all_set_sizes = []
    ref_survey_ids = None
    ref_species_to_index = None
    ref_index_to_species = None
    ref_cfg = None

    for ckpt_path in args.checkpoints:
        ckpt = load_checkpoint(ckpt_path, device="cpu")
        cfg = ckpt["config"]
        cfg["data"]["data_root"] = args.data_root
        data_root = Path(cfg["data"]["data_root"])
        cache_dir = Path(cfg["data"]["cache_dir"])

        species_to_index = {int(k): int(v) for k, v in ckpt["species_to_index"].items()}
        index_to_species = {int(k): int(v) for k, v in ckpt["index_to_species"].items()}

        if ref_species_to_index is None:
            ref_species_to_index = species_to_index
            ref_index_to_species = index_to_species
            ref_cfg = cfg
        elif species_to_index != ref_species_to_index:
            raise ValueError("All ensemble checkpoints must share identical species mapping")

        normalizer = normalizer_from_checkpoint(ckpt["normalizer"])

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

        if ref_survey_ids is None:
            ref_survey_ids = outputs["survey_ids"]
        elif outputs["survey_ids"] != ref_survey_ids:
            raise ValueError("Survey order mismatch across ensemble checkpoints")

        all_probs.append(outputs["probs"])
        all_set_sizes.append(outputs["set_sizes"])

        print(f"Loaded test predictions from: {ckpt_path}")

    if ref_cfg is None or ref_survey_ids is None or ref_index_to_species is None or ref_species_to_index is None:
        raise RuntimeError("No checkpoint predictions were produced")

    probs_mean = torch.stack(all_probs, dim=0).mean(dim=0)
    set_size_mean = torch.stack(all_set_sizes, dim=0).mean(dim=0)

    alpha = float(ref_cfg.get("inference", {}).get("set_size_alpha", 1.0))
    max_k = int(ref_cfg.get("inference", {}).get("max_k_calibrated_fallback", 30))
    if args.calibration_json:
        calib = read_json(args.calibration_json)
        alpha = float(calib.get("alpha", alpha))
        max_k = int(calib.get("max_k", max_k))
    if args.alpha is not None:
        alpha = float(args.alpha)
    if args.max_k is not None:
        max_k = int(args.max_k)

    predictions = decode_topk_predictions(
        probs=probs_mean,
        set_sizes=set_size_mean,
        index_to_species=ref_index_to_species,
        min_k=int(ref_cfg["data"]["min_species_per_sample"]),
        max_k=max_k,
        alpha=alpha,
    )

    output_csv = write_submission(
        survey_ids=ref_survey_ids,
        predictions=predictions,
        output_csv=args.output_csv,
    )

    is_valid, errors = validate_submission(output_csv, valid_species_ids=set(ref_species_to_index.keys()))
    if not is_valid:
        raise RuntimeError("Submission validation failed:\n" + "\n".join(errors[:20]))

    print(f"Saved ensemble submission: {output_csv}")
    print(f"Ensemble size: {len(args.checkpoints)} | alpha={alpha:.4f} | max_k={max_k}")


if __name__ == "__main__":
    main()
