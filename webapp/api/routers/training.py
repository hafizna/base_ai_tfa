"""Training-data retention and feedback endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import training_retention
from ..record_analysis import build_record_analysis
from ..storage import load_analysis

router = APIRouter(prefix="/api/training", tags=["training-retention"])

# Ground-truth source / confidence enums (Stage 0). Kept as plain strings
# (not a pydantic Enum) so unrecognised/legacy values never hard-fail
# submission — the frontend enforces the option list.
GROUND_TRUTH_SOURCES = {
    "RELAY_EVENT_REPORT",
    "OPERATOR_SOE",
    "REMOTE_END_COMTRADE",
    "DFR_RECORD",
    "FIELD_INSPECTION",
    "LIGHTNING_DETECTION",
    "PATROL_REPORT",
    "PROTECTION_ENGINEER_REVIEW",
    "UNCONFIRMED_ASSUMPTION",
    "OTHER",
}
GROUND_TRUTH_CONFIDENCE_LEVELS = {"CONFIRMED", "PROBABLE", "POSSIBLE", "UNKNOWN"}


class TrainingFeedbackRequest(BaseModel):
    analysis_id: str
    relay_type: str

    # --- Legacy fields (Stage -1). Kept for backward compatibility with
    # existing feedback rows and any caller not yet using the granular form. ---
    ai_correct: Optional[bool] = None
    actual_label: str = ""
    fault_type: str = ""
    include_for_training: bool = True
    operator: str = ""
    notes: str = ""
    ai_prediction: Optional[dict[str, Any]] = None

    # --- Stage 0: per-layer ground-truth correction fields. All optional so
    # existing callers submitting only the legacy fields keep working. ---
    parsing_correct: Optional[bool] = None
    channel_mapping_correct: Optional[bool] = None

    inception_correct: Optional[bool] = None
    corrected_inception_time_ms: Optional[float] = None

    clearing_correct: Optional[bool] = None
    corrected_clearing_time_ms: Optional[float] = None

    faulted_phases_correct: Optional[bool] = None
    actual_faulted_phases: Optional[list[str]] = None

    fault_type_correct: Optional[bool] = None
    actual_fault_type: str = ""

    zone_correct: Optional[bool] = None
    actual_zone: str = ""

    trip_type_correct: Optional[bool] = None
    actual_trip_type: str = ""

    reclose_correct: Optional[bool] = None
    actual_reclose_outcome: str = ""

    event_segmentation_correct: Optional[bool] = None
    actual_episode_count: Optional[int] = None

    protection_interpretation_correct: Optional[bool] = None
    actual_event_class: str = ""

    cause_correct: Optional[bool] = None
    actual_cause: str = ""

    ground_truth_source: list[str] = Field(default_factory=list)
    ground_truth_confidence: str = "UNKNOWN"


class TrainingClearRequest(BaseModel):
    confirm: str = Field(..., description="Must be CLEAR to delete retained training data.")


def _require_admin_token(token: str | None) -> None:
    if not training_retention.admin_token_configured():
        raise HTTPException(
            status_code=503,
            detail="TRAINING_ADMIN_TOKEN is not configured on the server.",
        )
    if not training_retention.verify_admin_token(token):
        raise HTTPException(status_code=401, detail="Invalid training admin token.")


@router.get("/status")
async def training_status():
    return training_retention.get_training_status()


@router.post("/feedback")
async def submit_training_feedback(
    body: TrainingFeedbackRequest,
    x_training_admin_token: str | None = Header(default=None),
):
    _require_admin_token(x_training_admin_token)
    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()

    invalid_sources = set(payload.get("ground_truth_source") or []) - GROUND_TRUTH_SOURCES
    if invalid_sources:
        raise HTTPException(status_code=422, detail=f"Unknown ground_truth_source value(s): {sorted(invalid_sources)}")
    if payload.get("ground_truth_confidence") not in GROUND_TRUTH_CONFIDENCE_LEVELS:
        raise HTTPException(status_code=422, detail="ground_truth_confidence must be one of CONFIRMED, PROBABLE, POSSIBLE, UNKNOWN")

    # Snapshot the canonical analysis at feedback time so a correction stays
    # auditable even after the model or canonical-timing logic changes later.
    analysis_snapshot = None
    try:
        stored = load_analysis(payload["analysis_id"])
        if stored is not None:
            analysis_snapshot = build_record_analysis(payload["analysis_id"], stored).to_dict()
    except Exception:
        analysis_snapshot = None
    payload["canonical_analysis_snapshot"] = analysis_snapshot

    row = training_retention.append_feedback(payload)
    return {"status": "ok", "feedback": row}


@router.get("/export")
async def export_training_archive(
    background_tasks: BackgroundTasks,
    x_training_admin_token: str | None = Header(default=None),
):
    _require_admin_token(x_training_admin_token)
    zip_path = training_retention.build_training_archive()
    background_tasks.add_task(Path(zip_path).unlink, missing_ok=True)
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename="base-ai-tfa-training-data.zip",
    )


@router.post("/clear")
async def clear_training_archive(
    body: TrainingClearRequest,
    x_training_admin_token: str | None = Header(default=None),
):
    _require_admin_token(x_training_admin_token)
    if body.confirm != "CLEAR":
        raise HTTPException(status_code=400, detail="Set confirm to CLEAR to delete retained data.")
    result = training_retention.clear_training_archive()
    return {"status": "ok", **result}
