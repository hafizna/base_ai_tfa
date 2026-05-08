"""Relay 50/51 (OCR) and REF/GFR/SBEF — overcurrent characteristic overlay."""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import OvercurrentAnalysisRequest, OvercurrentResponse, OvercurrentPoint
from ..storage import load_analysis

router = APIRouter(prefix="/api/analyze/ocr", tags=["relay-ocr"])

# IEC 60255-151 overcurrent curve constants
IEC_CURVES = {
    "NI":  {"k": 0.14,  "alpha": 0.02},   # Normal Inverse
    "VI":  {"k": 13.5,  "alpha": 1.0},    # Very Inverse
    "EI":  {"k": 80.0,  "alpha": 2.0},    # Extremely Inverse
    "LTI": {"k": 120.0, "alpha": 1.0},    # Long Time Inverse
}


def _trip_time(i_ratio: float, tms: float, curve: str) -> Optional[float]:
    """IEC time-overcurrent formula: t = TMS * k / ((I/Is)^alpha - 1)"""
    if i_ratio <= 1.0:
        return None
    c = IEC_CURVES.get(curve, IEC_CURVES["NI"])
    denom = (i_ratio ** c["alpha"]) - 1.0
    if denom <= 0:
        return None
    return tms * c["k"] / denom


def _max_sliding_rms(samples: np.ndarray, window: int) -> float:
    if samples.size == 0:
        return 0.0
    window = max(1, min(window, samples.size))
    squared = samples.astype(float) ** 2
    csum = np.concatenate(([0.0], np.cumsum(squared)))
    means = (csum[window:] - csum[:-window]) / window
    if means.size == 0:
        return float(np.sqrt(np.mean(squared)))
    return float(np.sqrt(np.max(means)))


def _find_max_current(channels, time, frequency: float = 50.0):
    """Find maximum one-cycle RMS current from available phase channels."""
    phase_candidates = [
        ["IA", "IL1", "I1"],
        ["IB", "IL2", "I2"],
        ["IC", "IL3", "I3"],
        ["IN", "I0", "IE"],
    ]
    time_arr = np.array(time, dtype=float)
    if len(time_arr) > 1 and frequency > 0:
        sr = 1.0 / max(float(time_arr[1] - time_arr[0]), 1e-9)
        window = max(4, int(sr / frequency))
    else:
        window = 16
    max_rms = 0.0
    for candidates in phase_candidates:
        for ch in channels:
            if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
                samples = np.array(ch["samples"], dtype=float)
                rms = _max_sliding_rms(samples, window)
                if rms > max_rms:
                    max_rms = rms
    return max_rms


def _build_curve_points(curve_type: str, tms: float, measured_ratio: Optional[float] = None) -> list[OvercurrentPoint]:
    max_ratio = 20.0
    if measured_ratio is not None and np.isfinite(measured_ratio):
        max_ratio = max(max_ratio, min(float(measured_ratio) * 1.25, 200.0))

    near_pickup = np.geomspace(1.01, 2.0, 90)
    high_current = np.geomspace(2.01, max_ratio, 190)
    multipliers = np.unique(np.concatenate([near_pickup, high_current]))

    curve_points = []
    for mult in multipliers:
        t = _trip_time(float(mult), tms, curve_type)
        if t is not None and t < 100:
            curve_points.append(OvercurrentPoint(current_ratio=float(mult), trip_time_s=float(t)))
    return curve_points


@router.post("/characteristic", response_model=OvercurrentResponse)
async def overcurrent_characteristic(body: OvercurrentAnalysisRequest):
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")

    channels = payload["analog_channels"]
    time = payload["time"]
    frequency = float(payload.get("frequency") or 50.0)

    measured_a = _find_max_current(channels, time, frequency)

    # Find where the measured current intersects the trip curve
    intersection_ratio: Optional[float] = None
    measured_trip: Optional[float] = None
    if body.is_pickup_a > 0 and measured_a > 0:
        ratio = measured_a / body.is_pickup_a
        intersection_ratio = float(ratio)
        measured_trip = _trip_time(ratio, body.tms, body.curve_type)

    curve_points = _build_curve_points(body.curve_type, body.tms, intersection_ratio)

    return OvercurrentResponse(
        curve_points=curve_points,
        measured_current_a=measured_a,
        measured_trip_time_s=measured_trip,
        intersection_ratio=intersection_ratio,
    )
