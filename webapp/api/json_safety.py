"""Helpers for returning strict JSON from parsed COMTRADE data."""

from __future__ import annotations

import math
from typing import Any


def replace_non_finite_numbers(value: Any, replacement: float = 0.0) -> Any:
    """Recursively replace NaN/Infinity floats with a JSON-safe number."""
    if isinstance(value, float):
        return value if math.isfinite(value) else replacement

    if isinstance(value, list):
        return [replace_non_finite_numbers(item, replacement) for item in value]

    if isinstance(value, tuple):
        return tuple(replace_non_finite_numbers(item, replacement) for item in value)

    if isinstance(value, dict):
        return {key: replace_non_finite_numbers(item, replacement) for key, item in value.items()}

    return value
