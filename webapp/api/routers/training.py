"""Training-data retention and feedback endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import training_retention

router = APIRouter(prefix="/api/training", tags=["training-retention"])


class TrainingFeedbackRequest(BaseModel):
    analysis_id: str
    relay_type: str
    ai_correct: Optional[bool] = None
    actual_label: str = ""
    fault_type: str = ""
    include_for_training: bool = True
    operator: str = ""
    notes: str = ""
    ai_prediction: Optional[dict[str, Any]] = None


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
