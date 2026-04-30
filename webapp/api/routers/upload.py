"""Upload router - parses .cfg + .dat pair and returns structured JSON."""

import asyncio
import sys
import tempfile
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.comtrade_parser import ComtradeRecord, parse_comtrade
from core.protection_router import ProtectionType, determine_protection
from ..json_safety import replace_non_finite_numbers
from ..schemas import AnalysisCreatedResponse, AnalysisSummaryOut, ComtradeOut, RecalcByIdRequest
from ..storage import load_analysis, save_analysis, update_analysis

_PROTECTION_TO_RELAY: dict[ProtectionType, str] = {
    ProtectionType.DISTANCE: "21",
    ProtectionType.DIFFERENTIAL: "87L",
    ProtectionType.TRANSFORMER_DIFF: "87T",
    ProtectionType.OVERCURRENT: "OCR",
}

router = APIRouter(prefix="/api", tags=["upload"])


def _record_to_out(record: ComtradeRecord) -> dict:
    payload = {
        "station_name": record.station_name,
        "rec_dev_id": record.rec_dev_id,
        "rev_year": record.rev_year,
        "sampling_rates": record.sampling_rates,
        "trigger_time": record.trigger_time,
        "total_samples": record.total_samples,
        "frequency": record.frequency,
        "time": record.time.tolist(),
        "analog_channels": [
            {
                "id": ch.id,
                "name": ch.name,
                "canonical_name": ch.canonical_name,
                "unit": ch.unit,
                "phase": ch.phase,
                "measurement": ch.measurement,
                "ct_primary": ch.ct_primary,
                "ct_secondary": ch.ct_secondary,
                "pors": ch.pors,
                "samples": ch.samples.tolist(),
            }
            for ch in record.analog_channels
        ],
        "status_channels": [
            {
                "id": ch.id,
                "name": ch.name,
                "samples": ch.samples.tolist(),
            }
            for ch in record.status_channels
        ],
        "warnings": record.warnings,
    }
    return replace_non_finite_numbers(payload)


def _status_transition_count(samples: list[int]) -> int:
    transitions = 0
    for idx in range(1, len(samples)):
        if samples[idx] != samples[idx - 1]:
            transitions += 1
    return transitions


def _analysis_to_summary(analysis_id: str, payload: dict) -> AnalysisSummaryOut:
    time_values = payload.get("time") or []
    duration_ms = 0.0
    if len(time_values) > 1:
        duration_ms = float((time_values[-1] - time_values[0]) * 1000.0)

    analog_channels = [
        {
            "id": ch["id"],
            "name": ch["name"],
            "canonical_name": ch["canonical_name"],
            "unit": ch["unit"],
            "phase": ch["phase"],
            "measurement": ch["measurement"],
            "ct_primary": ch["ct_primary"],
            "ct_secondary": ch["ct_secondary"],
            "pors": ch["pors"],
        }
        for ch in payload.get("analog_channels", [])
    ]

    status_channels = [
        {
            "id": ch["id"],
            "name": ch["name"],
            "sample_count": len(ch.get("samples", [])),
            "on_count": int(sum(ch.get("samples", []))),
            "transition_count": _status_transition_count(ch.get("samples", [])),
        }
        for ch in payload.get("status_channels", [])
    ]

    return AnalysisSummaryOut(
        analysis_id=analysis_id,
        station_name=payload.get("station_name", ""),
        rec_dev_id=payload.get("rec_dev_id", ""),
        rev_year=payload.get("rev_year", ""),
        sampling_rates=payload.get("sampling_rates", []),
        trigger_time=payload.get("trigger_time", 0.0),
        total_samples=payload.get("total_samples", 0),
        frequency=payload.get("frequency", 0.0),
        duration_ms=duration_ms,
        analog_channels=analog_channels,
        status_channels=status_channels,
        warnings=payload.get("warnings", []),
    )


@router.post("/upload", response_model=AnalysisCreatedResponse)
async def upload_comtrade(
    cfg_file: UploadFile = File(...),
    dat_file: UploadFile = File(...),
):
    """Parse a .cfg + .dat COMTRADE pair and create a backend analysis session."""
    cfg_bytes = await cfg_file.read()
    dat_bytes = await dat_file.read()

    with tempfile.TemporaryDirectory(prefix="dfr_upload_") as tmp_dir:
        tmp = Path(tmp_dir)

        cfg_name = Path(cfg_file.filename or "record.cfg").name
        dat_name = Path(dat_file.filename or "record.dat").name

        cfg_path = tmp / cfg_name
        dat_path = tmp / dat_name

        cfg_path.write_bytes(cfg_bytes)
        dat_path.write_bytes(dat_bytes)

        loop = asyncio.get_event_loop()
        record = await loop.run_in_executor(
            None,
            partial(parse_comtrade, str(cfg_path), str(dat_path)),
        )

    if record is None:
        raise HTTPException(
            status_code=422,
            detail="Could not parse COMTRADE files. Check that the .cfg and .dat pair is valid.",
        )

    payload = _record_to_out(record)
    analysis_id = save_analysis(payload)

    suggested_relay_type: str | None = None
    detection_confidence: float | None = None
    try:
        event = determine_protection(record)
        mapped = _PROTECTION_TO_RELAY.get(event.primary_protection)
        if mapped and event.confidence >= 0.7:
            suggested_relay_type = mapped
            detection_confidence = event.confidence
    except Exception:
        pass

    return AnalysisCreatedResponse(
        analysis_id=analysis_id,
        station_name=payload["station_name"],
        rec_dev_id=payload["rec_dev_id"],
        total_samples=payload["total_samples"],
        analog_channel_count=len(payload["analog_channels"]),
        status_channel_count=len(payload["status_channels"]),
        suggested_relay_type=suggested_relay_type,
        detection_confidence=detection_confidence,
    )


@router.get("/analysis/{analysis_id}", response_model=ComtradeOut)
async def get_analysis(analysis_id: str):
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    return replace_non_finite_numbers(payload)


@router.get("/analysis/{analysis_id}/summary", response_model=AnalysisSummaryOut)
async def get_analysis_summary(analysis_id: str):
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    return _analysis_to_summary(analysis_id, payload)


@router.post("/recalculate-ratio")
async def recalculate_ratio(body: RecalcByIdRequest):
    """Apply per-channel CT/VT ratio overrides and return recalculated samples."""
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")

    ratio_map = {r.channel_id: (r.primary, r.secondary) for r in body.ratios}
    updated_channels = []
    for channel in payload["analog_channels"]:
        channel_id = channel["id"]
        if channel_id in ratio_map:
            pri, sec = ratio_map[channel_id]
            factor = pri / sec if sec != 0 else 1.0
            orig_secondary = channel["ct_secondary"]
            orig_primary = channel["ct_primary"]
            orig_factor = orig_primary / orig_secondary if orig_secondary != 0 else 1.0
            new_samples = [sample / orig_factor * factor for sample in channel["samples"]]
            updated_channels.append(
                {
                    **channel,
                    "samples": new_samples,
                    "ct_primary": pri,
                    "ct_secondary": sec,
                }
            )
        else:
            updated_channels.append(channel)

    updated_payload = replace_non_finite_numbers({**payload, "analog_channels": updated_channels})
    update_analysis(body.analysis_id, updated_payload)
    return updated_payload
