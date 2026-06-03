"""Relay 50/51 (OCR) and REF/GFR/SBEF — overcurrent characteristic overlay."""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    OvercurrentAnalysisRequest, OvercurrentResponse, OvercurrentPoint,
    TccRequest, TccResponse, TccStage, TccCurveLine, TccFaultPoint,
)
from ..storage import load_analysis

router = APIRouter(prefix="/api/analyze/ocr", tags=["relay-ocr"])

# IEC 60255-151 overcurrent curve constants
IEC_CURVES = {
    "NI":  {"k": 0.14,  "alpha": 0.02},   # Normal Inverse
    "VI":  {"k": 13.5,  "alpha": 1.0},    # Very Inverse
    "EI":  {"k": 80.0,  "alpha": 2.0},    # Extremely Inverse
    "LTI": {"k": 120.0, "alpha": 1.0},    # Long Time Inverse
}

CURVE_NAMES = {
    "NI": "IEC Normal Inverse",
    "VI": "IEC Very Inverse",
    "EI": "IEC Extremely Inverse",
    "LTI": "IEC Long Time Inverse",
    "DT": "Definite Time / INST",
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


def _stage_trip_time(current_a: float, stage: TccStage) -> Optional[float]:
    """Operating time of one stage for an absolute fault current. None = below pickup."""
    if stage.is_pickup_a <= 0 or current_a < stage.is_pickup_a:
        return None
    if stage.curve_type == "DT":
        # Definite time / instant: constant operating time once picked up.
        return max(0.0, float(stage.definite_time_s))
    ratio = current_a / stage.is_pickup_a
    return _trip_time(ratio, stage.tms, stage.curve_type)


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


# --- Multi-stage TCC (absolute Ampere axis) -------------------------------

# Channel candidate sets per mode. Phase = A/B/C; EF = residual/neutral.
_PHASE_SETS = [
    ("A", ["IA", "IL1", "I1"]),
    ("B", ["IB", "IL2", "I2"]),
    ("C", ["IC", "IL3", "I3"]),
]
_EF_SETS = [
    ("N/EF", ["IN", "I0", "IE", "IG", "3I0"]),
]


def _rms_window(time, frequency: float) -> int:
    time_arr = np.array(time, dtype=float)
    if len(time_arr) > 1 and frequency > 0:
        sr = 1.0 / max(float(time_arr[1] - time_arr[0]), 1e-9)
        return max(4, int(sr / frequency))
    return 16


def _max_rms_for_set(channels, candidates, window: int) -> float:
    max_rms = 0.0
    for ch in channels:
        if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
            samples = np.array(ch["samples"], dtype=float)
            rms = _max_sliding_rms(samples, window)
            if rms > max_rms:
                max_rms = rms
    return max_rms


def _build_stage_curve(stage: TccStage, i_max_a: float) -> TccCurveLine:
    """Sample one stage across the plotted Ampere range."""
    pickup = max(stage.is_pickup_a, 1e-6)
    i_hi = max(i_max_a * 1.3, pickup * 25.0)
    currents: list[float] = []
    times: list[float] = []

    if stage.curve_type == "DT":
        # Flat line from pickup to top of range at the definite time.
        t = max(0.0, float(stage.definite_time_s))
        currents = [pickup, i_hi]
        times = [t if t > 0 else 0.001, t if t > 0 else 0.001]
    else:
        grid = np.geomspace(pickup * 1.001, i_hi, 220)
        for c in grid:
            t = _stage_trip_time(float(c), stage)
            if t is not None and 0 < t < 1000:
                currents.append(float(c))
                times.append(float(t))

    return TccCurveLine(
        label=stage.label,
        curve_type=stage.curve_type,
        is_pickup_a=stage.is_pickup_a,
        currents_a=currents,
        trip_times_s=times,
    )


def _evaluate_fault(channel_label: str, current_a: float, stages: list[TccStage]) -> TccFaultPoint:
    """Pick the stage that trips first (fastest operating time) for this fault current."""
    best_stage: Optional[TccStage] = None
    best_t: Optional[float] = None
    for st in stages:
        t = _stage_trip_time(current_a, st)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_stage = st

    if best_stage is None:
        return TccFaultPoint(
            channel_label=channel_label, current_a=current_a,
            winning_stage_label=None, winning_curve_type=None,
            trip_time_s=None, multiple_of_pickup=None, is_moment=False,
        )

    mult = current_a / best_stage.is_pickup_a if best_stage.is_pickup_a > 0 else None
    return TccFaultPoint(
        channel_label=channel_label,
        current_a=current_a,
        winning_stage_label=best_stage.label,
        winning_curve_type=best_stage.curve_type,
        trip_time_s=best_t,
        multiple_of_pickup=mult,
        is_moment=(best_stage.curve_type == "DT"),
    )


def _build_assessment(mode: str, domain: str, points: list[TccFaultPoint], stages: list[TccStage]) -> str:
    """Descriptive evaluation text — kept and enriched with moment-overcurrent context."""
    elem = "elemen GFR/EF (gangguan tanah)" if mode == "ef" else "elemen OCR fasa"
    active = [p for p in points if p.current_a > 0]
    if not active:
        return f"Assessment: tidak ada arus terukur pada {elem}. Periksa pemetaan channel atau setting."

    operating = [p for p in active if p.winning_stage_label is not None]
    if not operating:
        hi = max(active, key=lambda p: p.current_a)
        return (
            f"Assessment: arus maksimum {hi.current_a:.0f} A masih di bawah pickup semua stage. "
            f"Dengan setting ini {elem} diasumsikan tidak seharusnya trip. "
            "Cocokkan dengan SOE/trip aktual untuk konfirmasi."
        )

    moment = [p for p in operating if p.is_moment]
    inverse = [p for p in operating if not p.is_moment]
    lines: list[str] = []

    if moment:
        # The headline case the user wants explained clearly.
        m = max(moment, key=lambda p: p.current_a)
        mult_txt = f"{m.multiple_of_pickup:.1f}× pickup" if m.multiple_of_pickup else ""
        t_txt = f"{m.trip_time_s*1000:.0f} ms" if m.trip_time_s is not None else "instan"
        phases = ", ".join(sorted({p.channel_label for p in moment}))
        guidance = (
            "Pada trafo, elemen moment biasanya diset tunda ~300–700 ms untuk koordinasi dengan sisi hilir."
            if domain == "trafo"
            else "Pada penghantar, elemen moment umumnya bekerja ~80–300 ms (high-set / instant)."
        )
        lines.append(
            f"Assessment — GANGGUAN MOMENT OVERCURRENT: fasa {phases} mencapai {m.current_a:.0f} A ({mult_txt}), "
            f"melampaui pickup stage moment ({m.winning_stage_label}). "
            f"Karena arus jauh di atas high-set, relay tidak menunggu kurva inverse yang lambat — "
            f"elemen moment ambil alih dan trip ~{t_txt}. {guidance} "
            "Artinya gangguan berarus besar (dekat/berat); verifikasi waktu trip aktual di SOE ≈ waktu moment ini."
        )
    if inverse:
        iv = max(inverse, key=lambda p: p.current_a)
        mult_txt = f"{iv.multiple_of_pickup:.1f}× pickup" if iv.multiple_of_pickup else ""
        t_txt = f"{iv.trip_time_s:.2f} s" if iv.trip_time_s is not None else "?"
        phases = ", ".join(sorted({p.channel_label for p in inverse}))
        lines.append(
            f"Pada elemen inverse: fasa {phases} mencapai {mult_txt}, diperkirakan trip via stage "
            f"{iv.winning_stage_label} dalam ~{t_txt} (time-graded, bukan moment)."
        )

    return " ".join(lines)


@router.post("/tcc", response_model=TccResponse)
async def tcc_multistage(body: TccRequest):
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    if not body.stages:
        raise HTTPException(status_code=400, detail="At least one stage is required.")

    channels = payload["analog_channels"]
    time = payload["time"]
    frequency = float(payload.get("frequency") or 50.0)
    window = _rms_window(time, frequency)

    sets = _EF_SETS if body.mode == "ef" else _PHASE_SETS
    fault_points: list[TccFaultPoint] = []
    i_max_a = 0.0
    for label, candidates in sets:
        current = _max_rms_for_set(channels, candidates, window)
        if current > i_max_a:
            i_max_a = current
        fault_points.append(_evaluate_fault(label, current, body.stages))

    curves = [_build_stage_curve(st, i_max_a) for st in body.stages]
    assessment = _build_assessment(body.mode, body.domain, fault_points, body.stages)

    return TccResponse(
        mode=body.mode,
        domain=body.domain,
        curves=curves,
        fault_points=fault_points,
        assessment=assessment,
    )
