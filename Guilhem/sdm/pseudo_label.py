from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch


def build_pseudo_label_records(
    survey_ids: Iterable[int],
    probs: torch.Tensor,
    observed_labels: Dict[int, List[int]],
    species_to_index: Dict[int, int],
    t_pos: float,
    t_neg: float,
    max_negatives_per_sample: int,
) -> list[dict]:
    probs_np = probs.detach().cpu().numpy()
    survey_ids = [int(s) for s in survey_ids]

    records: list[dict] = []
    rng = np.random.default_rng(42)

    for row_idx, sid in enumerate(survey_ids):
        p = probs_np[row_idx]
        pos_idx = set(np.where(p >= t_pos)[0].astype(int).tolist())

        observed = observed_labels.get(sid, [])
        for sp in observed:
            sp_idx = species_to_index.get(int(sp))
            if sp_idx is not None:
                pos_idx.add(sp_idx)

        neg_candidates = np.where(p <= t_neg)[0].astype(int)
        neg_candidates = np.array([i for i in neg_candidates.tolist() if i not in pos_idx], dtype=int)
        if neg_candidates.size > max_negatives_per_sample:
            neg_candidates = rng.choice(neg_candidates, size=max_negatives_per_sample, replace=False)

        record = {
            "survey_id": sid,
            "positive_idx": sorted(int(i) for i in pos_idx),
            "negative_idx": sorted(int(i) for i in neg_candidates.tolist()),
        }
        records.append(record)

    return records
