from __future__ import annotations

from typing import Any

import numpy as np

from sdm.data.datasets import FeatureNormalizer


def normalizer_from_checkpoint(blob: dict[str, Any]) -> FeatureNormalizer:
    return FeatureNormalizer(
        patch_mean=np.asarray(blob["patch_mean"], dtype=np.float32),
        patch_std=np.asarray(blob["patch_std"], dtype=np.float32),
        landsat_mean=np.asarray(blob["landsat_mean"], dtype=np.float32),
        landsat_std=np.asarray(blob["landsat_std"], dtype=np.float32),
        landsat_seq_len=int(blob.get("landsat_seq_len", 1)),
        bioclim_mean=np.asarray(blob["bioclim_mean"], dtype=np.float32),
        bioclim_std=np.asarray(blob["bioclim_std"], dtype=np.float32),
        bioclim_seq_len=int(blob.get("bioclim_seq_len", 1)),
        tabular_mean=np.asarray(blob["tabular_mean"], dtype=np.float32),
        tabular_std=np.asarray(blob["tabular_std"], dtype=np.float32),
    )
