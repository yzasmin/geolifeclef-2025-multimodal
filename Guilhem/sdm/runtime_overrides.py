from __future__ import annotations

from typing import Any


def deep_merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def resolve_int_with_source(
    cli_value: int | None,
    ckpt_value: Any,
    default_value: int,
) -> tuple[int, str]:
    if cli_value is not None:
        return int(cli_value), "cli"
    if ckpt_value is not None:
        return int(ckpt_value), "checkpoint"
    return int(default_value), "default"

