#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from sdm.config import load_config
from sdm.data.features import MultiModalFeatureStore
from sdm.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit per-split modality coverage")
    parser.add_argument("--data-root", type=str, default=os.environ.get("GLC_DATA_DIR", "data"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--po-sample-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _unique_survey_ids(csv_path: Path) -> pd.Series:
    frame = pd.read_csv(csv_path, usecols=["surveyId"])
    return frame["surveyId"].astype("int64").drop_duplicates()


def _coverage_for_ids(feature_store: MultiModalFeatureStore, split: str, survey_ids: list[int]) -> dict[str, float]:
    has_patch = 0
    has_landsat = 0
    has_bioclim = 0
    has_tabular = 0
    has_non_tabular = 0

    for sid in survey_ids:
        avail = feature_store.availability(int(sid), split=split)
        has_patch += int(avail["has_patch"])
        has_landsat += int(avail["has_landsat"])
        has_bioclim += int(avail["has_bioclim"])
        has_tabular += int(avail["has_tabular"])
        has_non_tabular += int(avail["has_non_tabular"])

    n = max(1, len(survey_ids))
    return {
        "n_samples": int(len(survey_ids)),
        "patch_rate": float(has_patch / n),
        "landsat_rate": float(has_landsat / n),
        "bioclim_rate": float(has_bioclim / n),
        "tabular_rate": float(has_tabular / n),
        "non_tabular_rate": float(has_non_tabular / n),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["data_root"] = args.data_root

    data_root = Path(cfg["data"]["data_root"])
    cache_dir = Path(cfg["data"]["cache_dir"])

    feature_store = MultiModalFeatureStore(
        data_root=data_root,
        cache_dir=cache_dir,
        bioclim_num_vars=int(cfg["model"]["temporal_encoder"]["bioclim_num_vars"]),
    )

    pa_train_ids = _unique_survey_ids(data_root / "GLC25_PA_metadata_train.csv")
    pa_test_ids = _unique_survey_ids(data_root / "GLC25_PA_metadata_test.csv")
    po_path = data_root / "GLC25_PO_metadata_train.csv"
    if not po_path.exists():
        po_path = data_root / "GLC25_P0_metadata_train.csv"
    po_ids = _unique_survey_ids(po_path)

    po_sample_size = min(int(args.po_sample_size), len(po_ids))
    po_sample = po_ids.sample(n=po_sample_size, random_state=int(args.seed)).tolist()

    report = {
        "data_root": str(data_root),
        "counts": {
            "pa_train_unique": int(len(pa_train_ids)),
            "pa_test_unique": int(len(pa_test_ids)),
            "po_train_unique": int(len(po_ids)),
            "po_sample_size": int(po_sample_size),
        },
        "coverage": {
            "pa_train": _coverage_for_ids(feature_store, "pa_train", pa_train_ids.tolist()),
            "pa_test": _coverage_for_ids(feature_store, "pa_test", pa_test_ids.tolist()),
            "po_train_sample": _coverage_for_ids(feature_store, "po_train", po_sample),
        },
    }

    out_path = Path(args.output_json)
    ensure_dir(out_path.parent)
    write_json(report, out_path)
    print(f"Saved modality coverage audit: {out_path}")
    print(report["coverage"])


if __name__ == "__main__":
    main()

