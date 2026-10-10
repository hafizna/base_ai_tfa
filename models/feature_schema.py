"""Identical deterministic feature encoding for training and inference."""
import math

import numpy as np
import pandas as pd

FEATURE_VERSION = "v2.shared-2026-10"
FEATURE_COLS = [
    "fault_duration_ms", "fault_count", "peak_fault_current_a", "di_dt_max",
    "i0_i1_ratio", "thd_percent", "inception_angle_degrees", "voltage_sag_depth_pu",
    "voltage_phase_ratio_spread_pu", "healthy_phase_voltage_ratio", "v2_v1_ratio",
    "voltage_thd_max_percent", "reclose_enc", "is_ground_enc", "trip_type_enc",
    "phase_count", "zone_enc",
]


def encode_reclose(value):
    text = str(value).lower()
    return 1.0 if text == "true" else (0.0 if text == "false" else 0.5)


def encode_trip_type(value):
    text = str(value).lower()
    return 1 if "single" in text or "1" in text else (2 if "three" in text or "3" in text else 0)


def encode_zone(value):
    text = str(value).upper()
    return next((i for i in (1, 2, 3) if f"Z{i}" in text), 0)


def parse_phase_count(value):
    text = str(value)
    return text.count("+") + 1 if text and text != "nan" else 1


def _number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def build_feature_frame(rows, feature_cols=None):
    """Zero-fill missing numerics in both paths; never fit global medians on CV data."""
    columns = feature_cols or FEATURE_COLS
    encoded = []
    for row in rows:
        values = {col: _number(row.get(col)) for col in columns}
        values.update(
            fault_count=_number(row.get("fault_count"), 1.0),
            peak_fault_current_a=np.log1p(max(_number(row.get("peak_fault_current_a")), 0)),
            di_dt_max=np.log1p(max(_number(row.get("di_dt_max")), 0)),
            reclose_enc=encode_reclose(row.get("reclose_successful")),
            is_ground_enc=int(str(row.get("is_ground_fault")).lower() == "true"),
            trip_type_enc=encode_trip_type(row.get("trip_type")),
            phase_count=parse_phase_count(row.get("faulted_phases")),
            zone_enc=encode_zone(row.get("zone_operated")),
        )
        encoded.append([values[col] for col in columns])
    return pd.DataFrame(encoded, columns=columns, dtype=float)
