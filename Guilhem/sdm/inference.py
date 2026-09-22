from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

from sdm.metrics import topk_predictions
from sdm.utils import ensure_dir


def decode_topk_predictions(
    probs: torch.Tensor,
    set_sizes: torch.Tensor,
    index_to_species: Dict[int, int],
    min_k: int,
    max_k: int,
    alpha: float = 1.0,
) -> list[list[int]]:
    pred_bin = topk_predictions(
        probs,
        set_sizes,
        min_k=min_k,
        max_k=max_k,
        alpha=alpha,
    ).numpy()
    decoded: list[list[int]] = []
    for row in pred_bin:
        idx = np.where(row > 0.5)[0].tolist()
        species = sorted(index_to_species[i] for i in idx if i in index_to_species)
        if not species and index_to_species:
            species = [index_to_species[int(np.argmax(row))]]
        decoded.append(species)
    return decoded


def write_submission(
    survey_ids: Iterable[int],
    predictions: list[list[int]],
    output_csv: str | Path,
) -> Path:
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)

    rows = []
    for sid, species in zip(survey_ids, predictions):
        species_sorted = sorted(set(int(s) for s in species))
        pred_str = " ".join(str(s) for s in species_sorted)
        rows.append({"surveyId": int(sid), "predictions": pred_str})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    return output_csv


def validate_submission(
    submission_csv: str | Path,
    valid_species_ids: set[int] | None = None,
) -> tuple[bool, list[str]]:
    frame = pd.read_csv(submission_csv)
    errors: list[str] = []

    if list(frame.columns) != ["surveyId", "predictions"]:
        errors.append("Columns must be exactly: surveyId,predictions")

    if frame["surveyId"].duplicated().any():
        errors.append("surveyId contains duplicates")

    for _, row in frame.iterrows():
        pred_text = str(row["predictions"]).strip()
        if not pred_text:
            errors.append(f"Empty predictions for surveyId={row['surveyId']}")
            continue

        try:
            species = [int(x) for x in pred_text.split()]
        except Exception:
            errors.append(f"Non-integer species id list at surveyId={row['surveyId']}")
            continue

        if species != sorted(species):
            errors.append(f"Species list is not sorted for surveyId={row['surveyId']}")

        if valid_species_ids is not None and any(s not in valid_species_ids for s in species):
            errors.append(f"Unknown species id in predictions for surveyId={row['surveyId']}")

    return len(errors) == 0, errors
