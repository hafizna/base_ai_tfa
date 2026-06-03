"""Relay 87T (Differential Transformer) — diff/restraint plot + operated status."""

import sys
import asyncio
from pathlib import Path
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    DiffRestraintAnalysisRequest, DiffRestraintResponse, DiffRestraintSample,
    TripMarker, PhaseClassification,
)
from ..storage import load_analysis
from .relay_87l import (
    _characteristic_threshold, _detect_trip_markers, _classify_phases, _relay_diff_phases,
)

router = APIRouter(prefix="/api/analyze/87t", tags=["relay-87t"])

import re

# Phase channel candidates — checked against ch["canonical_name"] and ch["name"].upper()
HV_CHANNELS = {
    "L1": ["IA_HV", "IHA", "IA1", "HVS.IA"],
    "L2": ["IB_HV", "IHB", "IB1", "HVS.IB"],
    "L3": ["IC_HV", "IHC", "IC1", "HVS.IC"],
}
MV_CHANNELS = {
    "L1": ["MVS.IA", "IA_MV", "IMA"],
    "L2": ["MVS.IB", "IB_MV", "IMB"],
    "L3": ["MVS.IC", "IC_MV", "IMC"],
}
LV_CHANNELS = {
    "L1": ["IA_LV", "ILA", "IA2", "LVS.IA"],
    "L2": ["IB_LV", "ILB", "IB2", "LVS.IB"],
    "L3": ["IC_LV", "ILC", "IC2", "LVS.IC"],
}
# Relay-computed differential channels (already in pu, SIPROTEC 5 convention)
RELAY_DIFF_CHANNELS = {
    "L1": ["87T.IDA"],
    "L2": ["87T.IDB"],
    "L3": ["87T.IDC"],
}

# Phase index for Siemens-style "iL1-S1" / "IL2-S2" naming. L1=A, L2=B, L3=C.
PHASE_NUM = {"L1": "1", "L2": "2", "L3": "3"}


def _name_matches_side(name_upper: str, phase: str, side: int) -> bool:
    """Return True when channel name encodes the phase and winding side.

    Recognizes Siemens 7UT612 (e.g. "IL1-S1"), Siemens 7UT8x ("HVS.IA"),
    and the legacy "_HV" / "_LV" suffix conventions.
    """
    pnum = PHASE_NUM[phase]
    pletter = {"L1": "A", "L2": "B", "L3": "C"}[phase]
    side_keyword = {1: "HVS?", 2: "LVS?", 3: "TVS?"}[side]
    # Siemens 7UT612: "IL1-S1", "ILA-S2"
    if re.search(rf"\bI?L?{pnum}\b.*-S{side}\b", name_upper):
        return True
    if re.search(rf"\bI{pletter}\b.*-S{side}\b", name_upper):
        return True
    # 7UT8x dotted: "HVS.IA"
    if re.search(rf"\b{side_keyword}\.\s*I{pletter}\b", name_upper):
        return True
    # Generic _HV / _LV suffix
    suffix_word = {1: "HV", 2: "LV", 3: "TV"}[side]
    if re.search(rf"\bI{pletter}_?{suffix_word}\b", name_upper):
        return True
    return False


def _find_ch(channels, candidates):
    for ch in channels:
        if ch.get("canonical_name") in candidates or ch["name"].upper() in candidates:
            return np.array(ch["samples"], dtype=float)
    return None


def _find_winding_ch(channels, phase: str, side: int):
    """Locate the per-phase current of a given transformer winding side."""
    for ch in channels:
        if _name_matches_side(ch["name"].upper(), phase, side):
            return np.array(ch["samples"], dtype=float)
    return None


def _rms_window(arr, start, length):
    seg = arr[start: start + length]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0


def _compute_87t(comtrade_data: dict, params: dict) -> dict:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])

    idiff_pickup  = params["idiff_pickup"]
    idiff_fast    = params["idiff_fast"]
    slope1        = params["slope1"]
    intersection1 = params["intersection1"]
    slope2        = params["slope2"]
    intersection2 = params["intersection2"]

    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
    freq = comtrade_data.get("frequency", 50.0)
    win = max(1, int(sr / freq))
    step = max(1, win // 4)

    # Detect whether relay has pre-computed differential channels (SIPROTEC 5 etc.)
    relay_diff_available = all(
        _find_ch(channels, RELAY_DIFF_CHANNELS[ph]) is not None for ph in ["L1", "L2", "L3"]
    )

    # A transformer differential needs at least two windings (HV + LV/MV) on the
    # same relay. If only one winding side is present the diff is meaningless;
    # flag it so the UI defers to the relay's own trip signals instead.
    def _side_present(name_map, side):
        return any(
            _find_ch(channels, name_map[ph]) is not None or _find_winding_ch(channels, ph, side) is not None
            for ph in ["L1", "L2", "L3"]
        )
    sides_present = sum([
        _side_present(HV_CHANNELS, 1),
        _side_present(LV_CHANNELS, 2),
        _side_present(MV_CHANNELS, 3),
    ])
    diff_mode = "TWO_TERMINAL" if (relay_diff_available or sides_present >= 2) else "LOCAL_ONLY"

    samples = []
    operated_phases = []

    for phase in ["L1", "L2", "L3"]:
        i_hv = _find_ch(channels, HV_CHANNELS[phase])
        if i_hv is None:
            i_hv = _find_winding_ch(channels, phase, 1)
        i_mv = _find_ch(channels, MV_CHANNELS[phase])
        if i_mv is None:
            i_mv = _find_winding_ch(channels, phase, 3)
        i_lv = _find_ch(channels, LV_CHANNELS[phase])
        if i_lv is None:
            i_lv = _find_winding_ch(channels, phase, 2)
        i_diff_ch = _find_ch(channels, RELAY_DIFF_CHANNELS[phase])

        # Fallback: try generic phase channel names (single-bay recordings)
        if i_hv is None:
            from .relay_87l import PHASE_CURRENT_MAP
            i_hv = _find_ch(channels, PHASE_CURRENT_MAP[phase])
        if i_lv is None:
            i_lv = np.zeros(len(time))
        if i_mv is None:
            i_mv = np.zeros(len(time))

        if i_hv is None and i_diff_ch is None:
            continue

        # Estimate rated current (In) for pu normalisation of winding currents.
        # If relay-computed diff exists (pu) we back-calculate In from pre-fault window
        # where diff ≈ magnetising current and restraint ≈ load current.
        # Fallback: use the peak of the HVS current as a rough In estimate.
        if i_hv is not None:
            pre_end = min(2 * win, len(i_hv) // 4)
            in_est = max(float(np.sqrt(np.mean(i_hv[:pre_end] ** 2))) if pre_end > 0 else 0, 1.0)
            # Scale up: pre-fault is load, In ≥ load. Use peak as upper bound.
            peak_hv = float(np.max(np.abs(i_hv))) if len(i_hv) > 0 else 1.0
            in_est = max(in_est, peak_hv / 10.0, 1.0)  # In is typically > 10% of fault peak
        else:
            in_est = 1.0

        phase_samples = []
        for k in range(0, len(time), step):
            s = max(0, k - win + 1)

            if relay_diff_available and i_diff_ch is not None:
                # Use relay-computed differential directly (pu)
                i_diff = _rms_window(i_diff_ch, s, k - s + 1)
            else:
                # Compute from winding currents (A) then normalise to pu
                hv_rms = _rms_window(i_hv, s, k - s + 1) if i_hv is not None else 0.0
                mv_rms = _rms_window(i_mv, s, k - s + 1)
                lv_rms = _rms_window(i_lv, s, k - s + 1)
                i_diff = abs(hv_rms - mv_rms - lv_rms) / in_est

            # Restraint from winding RMS values (pu)
            hv_rms_r = _rms_window(i_hv, s, k - s + 1) / in_est if i_hv is not None else 0.0
            mv_rms_r = _rms_window(i_mv, s, k - s + 1) / in_est
            lv_rms_r = _rms_window(i_lv, s, k - s + 1) / in_est
            i_rest = (abs(hv_rms_r) + abs(mv_rms_r) + abs(lv_rms_r)) / 2.0

            phase_samples.append({
                "t": float(time[k]),
                "i_diff": round(i_diff, 4),
                "i_rest": round(i_rest, 4),
                "phase": phase,
            })

        samples.extend(phase_samples)

        phase_operated = any(
            s["i_diff"] >= _characteristic_threshold(
                s["i_rest"], idiff_pickup, slope1, intersection1, slope2, intersection2
            )
            for s in phase_samples
        )
        if phase_operated:
            operated_phases.append(phase)

    if any(s["i_diff"] >= idiff_fast for s in samples):
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
        # Genuine differential only when ≥2 winding sides are present.
        "diff_data_mode": diff_mode,
        "relay_diff_phases": _relay_diff_phases(comtrade_data),
    }


@router.post("/diff-restraint", response_model=DiffRestraintResponse)
async def diff_restraint_87t(body: DiffRestraintAnalysisRequest):
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(_compute_87t, payload, body.params.model_dump()),
    )
    return DiffRestraintResponse(
        samples=[DiffRestraintSample(**s) for s in result["samples"]],
        params=body.params,
        operated_status=result["operated_status"],
        operated_phases=result["operated_phases"],
        trip_markers=[TripMarker(**m) for m in result.get("trip_markers", [])],
        phase_classification=[PhaseClassification(**c) for c in result.get("phase_classification", [])],
        diff_data_mode=result.get("diff_data_mode", "TWO_TERMINAL"),
        relay_diff_phases=result.get("relay_diff_phases", []),
    )
