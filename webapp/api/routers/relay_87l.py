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

    return {"samples": samples, "operated_status": status, "operated_phases": operated_phases}


def _characteristic_threshold(i_rest, pickup, slope1, int1, slope2, int2):
    """Return the I-DIFF operate threshold for a given I_rest value."""
    if i_rest <= int1:
        return pickup
    if i_rest <= int2:
        return pickup + slope1 * (i_rest - int1)
    return pickup + slope1 * (int2 - int1) + slope2 * (i_rest - int2)


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
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis_87l(body: DiffRestraintAnalysisRequest):
    """Run LightGBM fault cause analysis for relay 87L (differential line)."""
    payload = _load_analysis_or_404(body.analysis_id)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ml_prediction, payload, "87L")
    return AIFaultResult(**result)
