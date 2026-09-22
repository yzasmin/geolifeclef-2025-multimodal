from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ImbalanceWeights:
    pa_weights: np.ndarray
    po_weights: np.ndarray
    report: dict[str, Any]


def _build_geo_lookup(
    survey_df: pd.DataFrame,
    survey_id_col: str,
    lat_col: str | None,
    lon_col: str | None,
    grid_size: float,
) -> dict[int, tuple[int, int]]:
    if lat_col is None or lon_col is None:
        return {}
    work = survey_df[[survey_id_col, lat_col, lon_col]].dropna().copy()
    if work.empty:
        return {}
    work[survey_id_col] = work[survey_id_col].astype(int)
    lat_blocks = np.floor(work[lat_col].astype(float) / float(grid_size)).astype(int)
    lon_blocks = np.floor(work[lon_col].astype(float) / float(grid_size)).astype(int)
    return {
        int(sid): (int(lb), int(lob))
        for sid, lb, lob in zip(work[survey_id_col].tolist(), lat_blocks.tolist(), lon_blocks.tolist())
    }


def build_pa_po_sample_weights(
    pa_train_ids: list[int],
    po_train_ids: list[int],
    pa_labels: dict[int, list[int]],
    pseudo_records: dict[int, Any],
    species_to_index: dict[int, int],
    pa_survey_df: pd.DataFrame,
    pa_survey_id_col: str,
    pa_lat_col: str | None,
    pa_lon_col: str | None,
    po_survey_df: pd.DataFrame,
    po_survey_id_col: str,
    po_lat_col: str | None,
    po_lon_col: str | None,
    grid_size: float,
    target_pa_ratio: float = 2.0,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
) -> ImbalanceWeights:
    pa_ids = [int(x) for x in pa_train_ids]
    po_ids = [int(x) for x in po_train_ids]

    pa_geo = _build_geo_lookup(pa_survey_df, pa_survey_id_col, pa_lat_col, pa_lon_col, grid_size)
    po_geo = _build_geo_lookup(po_survey_df, po_survey_id_col, po_lat_col, po_lon_col, grid_size)

    block_counter: Counter[tuple[int, int]] = Counter()
    for sid in pa_ids:
        block = pa_geo.get(sid)
        if block is not None:
            block_counter[block] += 1
    for sid in po_ids:
        block = po_geo.get(sid)
        if block is not None:
            block_counter[block] += 1

    n_species = len(species_to_index)
    sp_freq = np.zeros((n_species,), dtype=np.float64)

    for sid in pa_ids:
        for sp in pa_labels.get(sid, []):
            idx = species_to_index.get(int(sp))
            if idx is not None:
                sp_freq[idx] += 1.0
    for sid in po_ids:
        rec = pseudo_records.get(sid)
        if rec is None:
            continue
        for idx in getattr(rec, "positive_idx", []):
            if 0 <= int(idx) < n_species:
                sp_freq[int(idx)] += 1.0

    sp_freq = np.maximum(sp_freq, 1.0)
    sp_rarity = 1.0 / np.sqrt(sp_freq)

    def _sample_weight(sid: int, domain: str) -> float:
        geo_map = pa_geo if domain == "pa" else po_geo
        block = geo_map.get(sid)
        if block is None:
            w_geo = 1.0
        else:
            w_geo = 1.0 / np.sqrt(float(max(1, block_counter.get(block, 1))))

        if domain == "pa":
            pos_idx = [species_to_index[int(sp)] for sp in pa_labels.get(sid, []) if int(sp) in species_to_index]
        else:
            rec = pseudo_records.get(sid)
            pos_idx = [int(i) for i in getattr(rec, "positive_idx", []) if 0 <= int(i) < n_species] if rec else []

        if pos_idx:
            w_species = float(np.mean(sp_rarity[pos_idx]))
        else:
            w_species = 1.0
        return float(w_geo * w_species)

    pa_base = np.array([_sample_weight(sid, "pa") for sid in pa_ids], dtype=np.float64)
    po_base = np.array([_sample_weight(sid, "po") for sid in po_ids], dtype=np.float64)

    pa_sum = float(np.sum(pa_base)) if pa_base.size else 0.0
    po_sum = float(np.sum(po_base)) if po_base.size else 0.0

    pa_scale = 1.0
    po_scale = 1.0
    if pa_sum > 0 and po_sum > 0:
        target_pa_mass = float(target_pa_ratio) / (float(target_pa_ratio) + 1.0)
        target_po_mass = 1.0 / (float(target_pa_ratio) + 1.0)
        total_mass = pa_sum + po_sum
        cur_pa_mass = pa_sum / total_mass
        cur_po_mass = po_sum / total_mass
        pa_scale = target_pa_mass / max(cur_pa_mass, 1e-12)
        po_scale = target_po_mass / max(cur_po_mass, 1e-12)

    pa_w = pa_base * pa_scale if pa_base.size else pa_base
    po_w = po_base * po_scale if po_base.size else po_base

    if pa_w.size:
        pa_w = np.clip(pa_w, min_weight, max_weight)
    if po_w.size:
        po_w = np.clip(po_w, min_weight, max_weight)

    all_w = np.concatenate([pa_w, po_w]) if (pa_w.size or po_w.size) else np.array([1.0], dtype=np.float64)
    mean_w = float(np.mean(all_w))
    if mean_w > 0:
        if pa_w.size:
            pa_w = pa_w / mean_w
        if po_w.size:
            po_w = po_w / mean_w

    pa_mass = float(np.sum(pa_w)) if pa_w.size else 0.0
    po_mass = float(np.sum(po_w)) if po_w.size else 0.0
    observed_ratio = pa_mass / max(po_mass, 1e-12) if po_mass > 0 else float("inf")

    nonzero_sp = sp_freq[sp_freq > 1.0]
    report = {
        "target_pa_to_po_ratio": float(target_pa_ratio),
        "observed_weight_mass_ratio_pa_to_po": float(observed_ratio),
        "n_pa_samples": int(len(pa_ids)),
        "n_po_samples": int(len(po_ids)),
        "n_spatial_blocks": int(len(block_counter)),
        "species_nonzero_count": int(np.sum(sp_freq > 1.0)),
        "species_freq_p50": float(np.percentile(nonzero_sp, 50)) if nonzero_sp.size else 0.0,
        "species_freq_p90": float(np.percentile(nonzero_sp, 90)) if nonzero_sp.size else 0.0,
        "pa_weight_mean": float(np.mean(pa_w)) if pa_w.size else 0.0,
        "po_weight_mean": float(np.mean(po_w)) if po_w.size else 0.0,
        "pa_weight_min": float(np.min(pa_w)) if pa_w.size else 0.0,
        "po_weight_min": float(np.min(po_w)) if po_w.size else 0.0,
        "pa_weight_max": float(np.max(pa_w)) if pa_w.size else 0.0,
        "po_weight_max": float(np.max(po_w)) if po_w.size else 0.0,
    }
    return ImbalanceWeights(pa_weights=pa_w.astype(np.float64), po_weights=po_w.astype(np.float64), report=report)

