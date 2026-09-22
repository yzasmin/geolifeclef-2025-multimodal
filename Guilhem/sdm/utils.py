from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


SURVEY_ID_CANDIDATES = ["surveyId", "survey_id", "surveyID", "id"]
SPECIES_ID_CANDIDATES = ["spId", "speciesId", "species_id", "taxonId"]
PRESENCE_CANDIDATES = ["presence", "is_present", "label", "occurrence", "target"]
LAT_CANDIDATES = ["lat", "latitude", "decimalLatitude"]
LON_CANDIDATES = ["lon", "lng", "longitude", "decimalLongitude"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_first_column(columns: list[str], candidates: list[str]) -> str | None:
    colmap = {c.lower(): c for c in columns}
    for cand in candidates:
        match = colmap.get(cand.lower())
        if match is not None:
            return match
    return None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
