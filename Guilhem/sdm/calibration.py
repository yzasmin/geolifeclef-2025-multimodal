from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sdm.metrics import micro_precision_recall, sample_f1_score, topk_predictions


@dataclass
class CalibrationResult:
    alpha: float
    max_k: int
    metrics: dict[str, float]
    grid: list[dict[str, Any]]


def evaluate_calibration_grid(
    y_true: np.ndarray,
    probs: torch.Tensor,
    set_sizes: torch.Tensor,
    min_k: int,
    alphas: list[float],
    max_k_grid: list[int],
) -> CalibrationResult:
    best: CalibrationResult | None = None
    grid_rows: list[dict[str, Any]] = []

    for alpha in alphas:
        for max_k in max_k_grid:
            y_pred = topk_predictions(
                probs,
                set_sizes,
                min_k=min_k,
                max_k=max_k,
                alpha=alpha,
            ).numpy()
            f1_samples = float(sample_f1_score(y_true, y_pred))
            precision, recall = micro_precision_recall(y_true, y_pred)
            row = {
                "alpha": float(alpha),
                "max_k": int(max_k),
                "f1_samples": f1_samples,
                "precision_micro": float(precision),
                "recall_micro": float(recall),
            }
            grid_rows.append(row)
            if best is None:
                best = CalibrationResult(
                    alpha=float(alpha),
                    max_k=int(max_k),
                    metrics={
                        "f1_samples": f1_samples,
                        "precision_micro": float(precision),
                        "recall_micro": float(recall),
                    },
                    grid=[],
                )
                continue

            best_f1 = best.metrics["f1_samples"]
            best_precision = best.metrics["precision_micro"]
            if (f1_samples > best_f1) or (
                abs(f1_samples - best_f1) < 1e-12 and float(precision) > best_precision
            ):
                best = CalibrationResult(
                    alpha=float(alpha),
                    max_k=int(max_k),
                    metrics={
                        "f1_samples": f1_samples,
                        "precision_micro": float(precision),
                        "recall_micro": float(recall),
                    },
                    grid=[],
                )

    if best is None:
        raise ValueError("Calibration grid is empty")
    best.grid = grid_rows
    return best

