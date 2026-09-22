#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdm.calibration import evaluate_calibration_grid
from sdm.checkpoints import load_checkpoint
from sdm.data.collate import collate_sdm
from sdm.data.datasets import PADataset
from sdm.data.features import MultiModalFeatureStore
from sdm.data.metadata import load_pa_metadata
from sdm.data.splits import make_spatial_folds
from sdm.runtime import normalizer_from_checkpoint
from sdm.train import build_model
from sdm.utils import write_json


def parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in vals]


def parse_int_list(raw: str) -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in vals]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate set-size alpha/max_k on OOF validation")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--output-json", type=str, default="artifacts/calibration/calibration.json")
    parser.add_argument("--alphas", type=str, default="0.55,0.65,0.75,0.85,0.95")
    parser.add_argument("--max-k-grid", type=str, default="20,25,30")
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def collect_val_outputs(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    model.eval()
    y_true = []
    probs = []
    set_sizes = []

    for batch in tqdm(loader, desc="collect-val", leave=False):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value

        outputs = model(
            patch=moved["patch"],
            landsat=moved["landsat"],
            bioclim_ts=moved["bioclim_ts"],
            tabular=moved["tabular"],
        )
        y_true.append(batch["labels"].detach().cpu().numpy())
        probs.append(torch.sigmoid(outputs["logits"]).detach().cpu())
        set_sizes.append(outputs["set_size"].detach().cpu())

    return np.concatenate(y_true, axis=0), torch.cat(probs, dim=0), torch.cat(set_sizes, dim=0)


def main() -> None:
    args = parse_args()

    alphas = parse_float_list(args.alphas)
    max_k_grid = parse_int_list(args.max_k_grid)

    all_y_true = []
    all_probs = []
    all_set_sizes = []

    for ckpt_path in args.checkpoints:
        ckpt = load_checkpoint(ckpt_path, device="cpu")
        cfg = ckpt["config"]
        cfg["data"]["data_root"] = args.data_root

        data_root = Path(cfg["data"]["data_root"])
        cache_dir = Path(cfg["data"]["cache_dir"])

        species_to_index = {int(k): int(v) for k, v in ckpt["species_to_index"].items()}
        pa_meta = load_pa_metadata(data_root)

        folds = make_spatial_folds(
            survey_df=pa_meta.survey_df,
            survey_id_col=pa_meta.survey_id_col,
            lat_col=pa_meta.lat_col,
            lon_col=pa_meta.lon_col,
            n_folds=int(cfg["training"]["folds"]),
            grid_size=float(cfg["data"]["spatial_grid_size"]),
        )

        fold_idx = int(ckpt.get("fold", 0))
        val_ids = [int(x) for x in folds[fold_idx].val_ids]
        max_val_samples = cfg["training"].get("max_val_samples")
        if max_val_samples:
            val_ids = val_ids[: int(max_val_samples)]

        normalizer = normalizer_from_checkpoint(ckpt["normalizer"])
        feature_store = MultiModalFeatureStore(
            data_root=data_root,
            cache_dir=cache_dir,
            bioclim_num_vars=int(cfg["model"]["temporal_encoder"]["bioclim_num_vars"]),
        )

        val_ds = PADataset(
            survey_ids=val_ids,
            labels=pa_meta.labels,
            species_to_index=species_to_index,
            split="pa_train",
            feature_store=feature_store,
            normalizer=normalizer,
        )

        batch_size = args.batch_size if args.batch_size > 0 else int(cfg["training"]["batch_size"])
        loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, int(cfg["data"]["num_workers"])),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_sdm,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(
            num_species=len(species_to_index),
            landsat_channels=int(ckpt["input_dims"]["landsat_channels"]),
            bioclim_channels=int(ckpt["input_dims"]["bioclim_channels"]),
            tabular_dim=int(ckpt["input_dims"]["tabular_dim"]),
            model_cfg=cfg["model"],
            device=device,
        )
        model.load_state_dict(ckpt["state_dict"])

        y_true, probs, set_sizes = collect_val_outputs(model, loader, device=device)
        all_y_true.append(y_true)
        all_probs.append(probs)
        all_set_sizes.append(set_sizes)

        print(f"Loaded OOF val predictions from {ckpt_path} (fold={fold_idx}, n={len(val_ids)})")

    oof_y_true = np.concatenate(all_y_true, axis=0)
    oof_probs = torch.cat(all_probs, dim=0)
    oof_set_sizes = torch.cat(all_set_sizes, dim=0)

    result = evaluate_calibration_grid(
        y_true=oof_y_true,
        probs=oof_probs,
        set_sizes=oof_set_sizes,
        min_k=int(args.min_k),
        alphas=alphas,
        max_k_grid=max_k_grid,
    )

    payload = {
        "alpha": float(result.alpha),
        "max_k": int(result.max_k),
        "min_k": int(args.min_k),
        "metrics": result.metrics,
        "grid": result.grid,
        "n_oof_samples": int(oof_y_true.shape[0]),
        "checkpoints": [str(x) for x in args.checkpoints],
    }
    write_json(payload, args.output_json)

    print(f"Saved calibration: {args.output_json}")
    print(f"Selected alpha={result.alpha:.4f}, max_k={result.max_k}")
    print(f"OOF metrics: {result.metrics}")


if __name__ == "__main__":
    main()
