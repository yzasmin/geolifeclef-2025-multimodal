from __future__ import annotations

import numpy as np
import torch


def calibrated_set_sizes(
    set_sizes: torch.Tensor,
    min_k: int = 1,
    max_k: int = 50,
    alpha: float = 1.0,
) -> np.ndarray:
    k_np = np.rint(alpha * set_sizes.detach().cpu().numpy()).astype(int)
    return np.clip(k_np, min_k, max_k)


def topk_predictions(
    probs: torch.Tensor,
    set_sizes: torch.Tensor,
    min_k: int = 1,
    max_k: int = 50,
    alpha: float = 1.0,
) -> torch.Tensor:
    probs_np = probs.detach().cpu().numpy()
    k_np = calibrated_set_sizes(set_sizes, min_k=min_k, max_k=max_k, alpha=alpha)

    out = np.zeros_like(probs_np, dtype=np.float32)
    for i in range(probs_np.shape[0]):
        k = int(k_np[i])
        if k <= 0:
            continue
        idx = np.argpartition(-probs_np[i], kth=min(k, probs_np.shape[1] - 1))[:k]
        out[i, idx] = 1.0
    return torch.from_numpy(out)


def sample_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    tp = (y_true * y_pred).sum(axis=1)
    fp = ((1.0 - y_true) * y_pred).sum(axis=1)
    fn = (y_true * (1.0 - y_pred)).sum(axis=1)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(np.mean(f1))


def micro_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    eps = 1e-8
    tp = float((y_true * y_pred).sum())
    fp = float(((1.0 - y_true) * y_pred).sum())
    fn = float((y_true * (1.0 - y_pred)).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision, recall
