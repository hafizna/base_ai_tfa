"""Relay 87L (Differential Line) - diff/restraint plot + AI analysis."""

import asyncio
import sys
from functools import partial
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    AIFaultResult,
    DiffRestraintAnalysisRequest,
    DiffRestraintResponse,
    DiffRestraintSample,
    TripMarker,
    PhaseClassification,
)
from ..storage import load_analysis
from ..ml_predict import run_ml_prediction

router = APIRouter(prefix="/api/analyze/87l", tags=["relay-87l"])

PHASE_CURRENT_MAP = {
    "L1": ["IA", "IL1", "I1"],
    "L2": ["IB", "IL2", "I2"],
    "L3": ["IC", "IL3", "I3"],
}


def _find_ch(channels, candidates):
    for ch in channels:
        if ch["canonical_name"] in candidates or ch["name"].upper() in candidates:
            return np.array(ch["samples"])
    return None


def _compute_diff_restraint(comtrade_data: dict, params: dict) -> dict:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])
    samples = []
    operated_phases = []

    idiff_pickup = params["idiff_pickup"]
    idiff_fast = params["idiff_fast"]
    slope1 = params["slope1"]
    intersection1 = params["intersection1"]
    slope2 = params["slope2"]
    intersection2 = params["intersection2"]

    for phase, candidates in PHASE_CURRENT_MAP.items():
        i = _find_ch(channels, candidates)
        if i is None:
            continue

        sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
        freq = comtrade_data.get("frequency", 50.0)
        win = max(1, int(sr / freq))

        for k in range(0, len(time), max(1, win // 4)):
            s = max(0, k - win + 1)
            i_w = i[s : k + 1]
            i_rms = float(np.sqrt(np.mean(i_w**2))) if len(i_w) > 0 else 0.0
            samples.append(
                {
                    "t": float(time[k]),
                    "i_diff": abs(i_rms),
                    "i_rest": abs(i_rms),
                    "phase": phase,
                }
            )

        phase_operated = False
        for sample in samples:
            if sample["phase"] != phase:
                continue
            threshold = _characteristic_threshold(
                sample["i_rest"],
                idiff_pickup,
                slope1,
                intersection1,
                slope2,
                intersection2,
            )
            if sample["i_diff"] >= threshold:
                phase_operated = True
                break
        if phase_operated:
            operated_phases.append(phase)

    if any(sample["i_diff"] >= idiff_fast for sample in samples):
        status = "IDIFF_FAST_OPERATED"
    elif operated_phases:
        status = "IDIFF_OPERATED"
    else:
        status = "NOT_OPERATED"

    return {
        "samples": samples,
        "operated_status": status,
        "operated_phases": operated_phases,
        "trip_markers": _detect_trip_markers(comtrade_data, samples),
        "phase_classification": _classify_phases(samples, params),
    }


def _characteristic_threshold(i_rest, pickup, slope1, int1, slope2, int2):
    """Return the I-DIFF operate threshold for a given I_rest value."""
    if i_rest <= int1:
        return pickup
    if i_rest <= int2:
        return pickup + slope1 * (i_rest - int1)
    return pickup + slope1 * (int2 - int1) + slope2 * (i_rest - int2)


# --- Trip detection + per-phase classification (shared by 87L and 87T) -----

# Digital channel name patterns that indicate a trip assertion, mapped to a kind.
# Checked in order; first match wins. Phase suffix (A/B/C/L1..) parsed separately.
_TRIP_PATTERNS = [
    ("DIFF_FAST", ["DIFF>>", "IDIFF>>", "87-2", "I-DIFF>>", "DIFF FAST", "UNRESTRAINED"]),
    # "DIF" (single-F) and "DIFG" cover many DFR vendors (e.g. DIF-A_TRIP, DIFG_TRIP).
    ("DIFF",      ["DIFF>", "IDIFF>", "87-1", "I-DIFF>", "87T TRIP", "87L TRIP",
                   "DIFF PICKUP", "DIFF TRIP", "87 TRIP", "DIF-", "DIF_", "DIFG", "DIF "]),
    ("RELAY_TRIP", ["TRIP", "GEN TRIP", "RELAY TRIP", "PMT", "GENERAL TRIP"]),
]


def _phase_of_name(name_upper: str) -> "str | None":
    """Extract a phase label (L1/L2/L3) from a digital channel name, if present.

    Only matches explicit phase markers (PH A, L1, _R, "PHASE B"), not an
    incidental trailing letter — e.g. "PMT BUKA" must NOT be read as phase A.
    """
    import re
    # Trailing boundary: token must not be followed by another alphanumeric,
    # but "_" / "-" / end-of-string are valid separators (so "DIF-A_TRIP" -> A).
    end = r"(?![A-Z0-9])"
    for pnum, letters in (("L1", ("A", "1", "R")), ("L2", ("B", "2", "S")), ("L3", ("C", "3", "T"))):
        for tok in letters:
            if re.search(rf"(?:PHASE|PH|L|_|-)\s*{tok}{end}", name_upper):
                return pnum
            if re.search(rf"\b{tok}{end}", name_upper) and tok not in ("R", "S", "T"):
                # bare A/B/C/1/2/3 token (e.g. "DIFF> A"); skip R/S/T to avoid
                # false hits on common words. Phase letters R/S/T require a prefix.
                return pnum
    return None


def _detect_trip_markers(comtrade_data: dict, samples: list) -> list:
    """Find first assertion of each trip-type digital channel and map to a sample point."""
    status_channels = comtrade_data.get("status_channels", [])
    time = comtrade_data.get("time", [])
    if not status_channels or not time or not samples:
        return []
    import numpy as _np
    time_arr = _np.asarray(time, dtype=float)
    markers = []
    seen_kinds = set()

    for ch in status_channels:
        name_up = str(ch.get("name", "")).upper()
        kind = None
        for k, pats in _TRIP_PATTERNS:
            if any(p in name_up for p in pats):
                kind = k
                break
        if kind is None:
            continue
        samp = _np.asarray(ch.get("samples", []), dtype=float)
        if samp.size == 0:
            continue
        asserted = _np.nonzero(samp > 0.5)[0]
        if asserted.size == 0:
            continue
        idx = int(asserted[0])
        if idx >= time_arr.size:
            continue
        t_trip = float(time_arr[idx])
        phase = _phase_of_name(name_up)

        # Map onto the nearest operating-point sample (optionally phase-matched).
        candidates = [s for s in samples if phase is None or s["phase"] == phase] or samples
        nearest = min(candidates, key=lambda s: abs(s["t"] - t_trip))

        dedup_key = (kind, phase)
        if dedup_key in seen_kinds:
            continue
        seen_kinds.add(dedup_key)
        markers.append({
            "kind": kind,
            "channel_name": str(ch.get("name", "")),
            "t": t_trip,
            "phase": phase,
            "i_diff": float(nearest["i_diff"]),
            "i_rest": float(nearest["i_rest"]),
        })
    return markers


def _classify_phases(samples: list, params: dict) -> list:
    """Per-phase verdict + max operating stats, evidence-based on the trajectory."""
    pickup = params["idiff_pickup"]
    fast = params["idiff_fast"]
    s1, i1, s2, i2 = params["slope1"], params["intersection1"], params["slope2"], params["intersection2"]
    out = []
    for phase in ["L1", "L2", "L3"]:
        pts = [s for s in samples if s["phase"] == phase]
        if not pts:
            continue
        max_idiff = max(s["i_diff"] for s in pts)
        max_irest = max(s["i_rest"] for s in pts)
        # Ratio of operating point to its threshold along the trajectory.
        max_ratio = 0.0
        for s in pts:
            thr = _characteristic_threshold(s["i_rest"], pickup, s1, i1, s2, i2)
            if thr > 0:
                max_ratio = max(max_ratio, s["i_diff"] / thr)

        if max_idiff >= fast:
            verdict, conf = "Internal Fault", "high"
        elif max_ratio >= 1.0:
            # Inside operate region. High restraint with diff just above slope often = through-fault leakage.
            verdict = "Internal Fault" if max_ratio >= 1.3 else "Through Fault"
            conf = "high" if max_ratio >= 1.5 else "medium"
        else:
            verdict, conf = "Not Operated", "high" if max_idiff < pickup else "medium"

        out.append({
            "phase": phase,
            "verdict": verdict,
            "confidence": conf,
            "max_idiff": round(max_idiff, 3),
            "max_irest": round(max_irest, 3),
            "max_ratio": round(max_ratio, 3),
        })
    return out


def _load_analysis_or_404(analysis_id: str) -> dict:
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    return payload


@router.post("/diff-restraint", response_model=DiffRestraintResponse)
async def diff_restraint(body: DiffRestraintAnalysisRequest):
    payload = _load_analysis_or_404(body.analysis_id)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(_compute_diff_restraint, payload, body.params.model_dump()),
    )
    return DiffRestraintResponse(
        samples=[DiffRestraintSample(**sample) for sample in result["samples"]],
        params=body.params,
        operated_status=result["operated_status"],
        operated_phases=result["operated_phases"],
        trip_markers=[TripMarker(**m) for m in result.get("trip_markers", [])],
        phase_classification=[PhaseClassification(**c) for c in result.get("phase_classification", [])],
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis_87l(body: DiffRestraintAnalysisRequest):
    """Run LightGBM fault cause analysis for relay 87L (differential line)."""
    payload = _load_analysis_or_404(body.analysis_id)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ml_prediction, payload, "87L")
    return AIFaultResult(**result)
