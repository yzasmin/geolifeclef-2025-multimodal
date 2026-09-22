from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sdm.utils import ensure_dir


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)
