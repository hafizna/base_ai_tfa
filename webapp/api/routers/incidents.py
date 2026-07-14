"""Incident API — Stage 1 + Stage 2."""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..incidents import service as incident_service
from ..incidents.batch_upload import UploadedFile, run_batch_upload
from ..incidents.service import IncidentServiceError

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


def _handle(exc: IncidentServiceError):
    raise HTTPException(status_code=exc.status_code, detail=exc.message)


def _multi_comtrade_enabled() -> bool:
    """Re-read the env var per-request (not cached at import time) so tests
    can toggle it via monkeypatch without reloading this module."""
    return os.getenv("MULTI_COMTRADE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def _require_multi_comtrade_enabled() -> None:
    if not _multi_comtrade_enabled():
        raise HTTPException(
            status_code=403,
            detail="Multi-COMTRADE incident reconstruction (Stage 2) is disabled on this server.",
        )


# --- Schemas -----------------------------------------------------------------

class IncidentCreateRequest(BaseModel):
    title: str = "Untitled incident"
    station_name: Optional[str] = None
    bay_name: Optional[str] = None
    asset_id: Optional[str] = None
    asset_name: Optional[str] = None
    asset_type: Optional[str] = None
    voltage_level_kv: Optional[float] = None
    protection_family: Optional[str] = None
    operator_notes: Optional[str] = None


class IncidentPatchRequest(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    station_name: Optional[str] = None
    bay_name: Optional[str] = None
    asset_id: Optional[str] = None
    asset_name: Optional[str] = None
    asset_type: Optional[str] = None
    voltage_level_kv: Optional[float] = None
    protection_family: Optional[str] = None
    clock_assessment: Optional[str] = None
    clock_assessment_reason: Optional[str] = None
    operator_notes: Optional[str] = None
    incident_start_iso: Optional[str] = None
    incident_end_iso: Optional[str] = None


class AttachRecordRequest(BaseModel):
    analysis_id: str
    attachment_role: str = "UNKNOWN"
    bay_name: Optional[str] = None
    relay_id: Optional[str] = None
    relay_model: Optional[str] = None
    protection_type: Optional[str] = None
    source_filename: Optional[str] = None
    override_warnings: bool = False
    operator_notes: Optional[str] = None


class ReorderRecordsRequest(BaseModel):
    incident_record_ids: list[str] = Field(default_factory=list)


class EvidenceCreateRequest(BaseModel):
    evidence_type: str
    source: str = ""
    description: str = ""
    value: Any = None
    confidence: str = "UNKNOWN"
    observed_at_iso: Optional[str] = None
    attachment_name: Optional[str] = None
    created_by: Optional[str] = None


class FeedbackRequest(BaseModel):
    operator: Optional[str] = None
    record_grouping_correct: Optional[bool] = None
    actual_record_count: Optional[int] = None
    record_order_correct: Optional[bool] = None
    corrected_record_order: Optional[list[str]] = None
    incident_start_correct: Optional[bool] = None
    corrected_incident_start_iso: Optional[str] = None
    incident_end_correct: Optional[bool] = None
    corrected_incident_end_iso: Optional[str] = None
    clock_assessment_correct: Optional[bool] = None
    actual_clock_assessment: Optional[str] = None
    incident_interpretation_correct: Optional[bool] = None
    actual_incident_class: Optional[str] = None
    cause_correct: Optional[bool] = None
    actual_root_cause: Optional[str] = None
    ground_truth_sources: list[str] = Field(default_factory=list)
    ground_truth_confidence: str = "UNKNOWN"
    include_for_future_analysis: bool = True
    notes: Optional[str] = None

    # --- Stage 2 reconstruction correction fields ---
    same_bay_correct: Optional[bool] = None
    relationships_correct: Optional[bool] = None
    corrected_relationships: list[dict[str, Any]] = Field(default_factory=list)
    episode_grouping_correct: Optional[bool] = None
    corrected_episode_groups: list[list[str]] = Field(default_factory=list)
    evolving_fault_correct: Optional[bool] = None
    root_cause: Optional[str] = None


class ReconstructRequest(BaseModel):
    same_bay_override_reason: Optional[str] = None
    same_bay_override_operator: Optional[str] = None


class RelationshipOverrideRequest(BaseModel):
    corrected_relationship: str
    operator: Optional[str] = None
    reason: Optional[str] = None


# --- Incident CRUD ------------------------------------------------------------

@router.post("")
async def create_incident(body: IncidentCreateRequest):
    try:
        incident = incident_service.create_incident(**body.model_dump())
    except IncidentServiceError as exc:
        _handle(exc)
    return incident_service.to_response(incident, [])


@router.get("")
async def list_incidents(status: Optional[str] = None, station_name: Optional[str] = None):
    incidents = incident_service.list_incidents(status=status, station_name=station_name)
    out = []
    for incident in incidents:
        records = incident_service.list_records(incident.incident_id)
        out.append(incident_service.to_response(incident, records))
    return out


@router.get("/{incident_id}")
async def get_incident(incident_id: str):
    try:
        incident = incident_service.get_incident(incident_id)
        records = incident_service.list_records(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return incident_service.to_response(incident, records)


@router.patch("/{incident_id}")
async def patch_incident(incident_id: str, body: IncidentPatchRequest):
    patch = {k: v for k, v in body.model_dump().items() if v is not None or k in body.model_fields_set}
    try:
        incident = incident_service.update_incident(incident_id, patch)
        records = incident_service.list_records(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return incident_service.to_response(incident, records)


@router.delete("/{incident_id}")
async def delete_incident(incident_id: str):
    try:
        incident_service.get_incident(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    incident_service.delete_incident(incident_id)
    return {"status": "ok"}


# --- Record management ---------------------------------------------------------

@router.post("/{incident_id}/records")
async def attach_record(incident_id: str, body: AttachRecordRequest):
    fields = body.model_dump()
    fields.pop("analysis_id")
    try:
        record = incident_service.attach_record(incident_id, analysis_id=body.analysis_id, **fields)
    except IncidentServiceError as exc:
        _handle(exc)
    return record.to_dict()


@router.get("/{incident_id}/records")
async def list_records(incident_id: str):
    try:
        records = incident_service.list_records(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [r.to_dict() for r in records]


@router.delete("/{incident_id}/records/{incident_record_id}")
async def detach_record(incident_id: str, incident_record_id: str):
    try:
        incident_service.detach_record(incident_id, incident_record_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return {"status": "ok"}


@router.patch("/{incident_id}/records/order")
async def reorder_records(incident_id: str, body: ReorderRecordsRequest):
    try:
        records = incident_service.reorder_records(incident_id, body.incident_record_ids)
    except IncidentServiceError as exc:
        _handle(exc)
    return [r.to_dict() for r in records]


@router.post("/{incident_id}/refresh-snapshots")
async def refresh_snapshots(incident_id: str):
    try:
        records = incident_service.refresh_snapshots(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [r.to_dict() for r in records]


# --- Evidence -----------------------------------------------------------------

@router.post("/{incident_id}/evidence")
async def add_evidence(incident_id: str, body: EvidenceCreateRequest):
    try:
        evidence = incident_service.add_evidence(incident_id, **body.model_dump())
    except IncidentServiceError as exc:
        _handle(exc)
    return evidence.to_dict()


@router.get("/{incident_id}/evidence")
async def list_evidence(incident_id: str):
    try:
        items = incident_service.list_evidence(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [e.to_dict() for e in items]


@router.delete("/{incident_id}/evidence/{evidence_id}")
async def remove_evidence(incident_id: str, evidence_id: str):
    try:
        incident_service.get_incident(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    incident_service.remove_evidence(incident_id, evidence_id)
    return {"status": "ok"}


# --- Feedback ------------------------------------------------------------------

@router.post("/{incident_id}/feedback")
async def submit_feedback(incident_id: str, body: FeedbackRequest):
    try:
        feedback = incident_service.save_feedback(incident_id, body.model_dump())
    except IncidentServiceError as exc:
        _handle(exc)
    return feedback.to_dict()


@router.get("/{incident_id}/feedback")
async def get_feedback(incident_id: str):
    try:
        items = incident_service.list_feedback(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [f.to_dict() for f in items]


# --- Stage 2: batch upload -----------------------------------------------------

@router.post("/{incident_id}/upload-records")
async def upload_records(
    incident_id: str,
    files: list[UploadFile] = File(...),
    partial_success: bool = False,
    attachment_role: str = "UNKNOWN",
    bay_name: Optional[str] = None,
    protection_type: Optional[str] = None,
    override_warnings: bool = False,
):
    """Batch-upload multiple COMTRADE records (.cfg+.dat pairs and/or .cff
    files) and attach each successfully parsed record to this incident.

    Sequential processing keeps memory bounded (one record's waveform at a
    time). Default mode is atomic — see ``run_batch_upload`` docstring.
    ``override_warnings`` mirrors the single-record attach endpoint: pass
    true to attach records despite a station-name mismatch against the
    incident's own station metadata.
    """
    _require_multi_comtrade_enabled()
    try:
        incident_service.get_incident(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)

    uploaded = [UploadedFile(filename=f.filename or "upload.bin", content_type=f.content_type, data=await f.read()) for f in files]

    result = run_batch_upload(
        incident_id,
        uploaded,
        partial_success=partial_success,
        attachment_role=attachment_role,
        bay_name=bay_name,
        protection_type=protection_type,
        override_warnings=override_warnings,
    )
    return {
        "incident_id": result.incident_id,
        "records_created": [
            {
                "analysis_id": r.analysis_id,
                "incident_record_id": r.incident_record_id,
                "source_files": r.source_files,
                "status": r.status,
                "error": r.error,
            }
            for r in result.records_created
        ],
        "errors": result.errors,
        "reconstruction_status": result.reconstruction_status,
    }


# --- Stage 2: reconstruction -----------------------------------------------------

@router.post("/{incident_id}/reconstruct")
async def reconstruct(incident_id: str, body: ReconstructRequest):
    _require_multi_comtrade_enabled()
    try:
        reconstruction = incident_service.reconstruct(
            incident_id,
            same_bay_override_reason=body.same_bay_override_reason,
            same_bay_override_operator=body.same_bay_override_operator,
        )
        timeline = incident_service.get_timeline(incident_id)
        relationships = incident_service.get_relationships(incident_id)
        episodes = incident_service.get_episodes(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return incident_service.to_reconstruction_response(reconstruction, timeline, relationships, episodes)


@router.get("/{incident_id}/reconstruction")
async def get_reconstruction(incident_id: str, reconstruction_id: Optional[str] = None):
    _require_multi_comtrade_enabled()
    try:
        reconstruction = incident_service.get_reconstruction(incident_id, reconstruction_id)
        timeline = incident_service.get_timeline(incident_id)
        relationships = incident_service.get_relationships(incident_id)
        episodes = incident_service.get_episodes(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return incident_service.to_reconstruction_response(reconstruction, timeline, relationships, episodes)


@router.get("/{incident_id}/reconstructions")
async def list_reconstructions(incident_id: str):
    _require_multi_comtrade_enabled()
    try:
        versions = incident_service.list_reconstructions(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [v.to_dict() for v in versions]


@router.get("/{incident_id}/timeline")
async def get_timeline(incident_id: str):
    _require_multi_comtrade_enabled()
    try:
        events = incident_service.get_timeline(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [e.to_dict() for e in events]


@router.get("/{incident_id}/relationships")
async def get_relationships(incident_id: str):
    _require_multi_comtrade_enabled()
    try:
        relationships = incident_service.get_relationships(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [r.to_dict() for r in relationships]


@router.post("/{incident_id}/relationships/{relationship_id}/override")
async def override_relationship(incident_id: str, relationship_id: str, body: RelationshipOverrideRequest):
    _require_multi_comtrade_enabled()
    try:
        relationship = incident_service.override_relationship(
            incident_id,
            relationship_id,
            corrected_relationship=body.corrected_relationship,
            operator=body.operator,
            reason=body.reason,
        )
    except IncidentServiceError as exc:
        _handle(exc)
    return relationship.to_dict()


@router.get("/{incident_id}/episodes")
async def get_episodes(incident_id: str):
    _require_multi_comtrade_enabled()
    try:
        episodes = incident_service.get_episodes(incident_id)
    except IncidentServiceError as exc:
        _handle(exc)
    return [e.to_dict() for e in episodes]
