"""Incident service — Stage 1 + Stage 2.

Business logic for creating incidents, attaching/detaching canonical
single-record analyses, manual ordering, evidence, and feedback (Stage 1),
plus same-bay multi-record reconstruction: alignment, canonical timeline,
pairwise relationships, fault episodes, and narrative (Stage 2). Routers
should call into this module rather than talking to storage,
``record_analysis``, or the reconstruction engine modules directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ..record_analysis import RecordAnalysis, build_record_analysis
from ..storage import load_analysis
from . import storage as incident_storage
from .models import (
    ASSET_TYPES,
    CLOCK_ASSESSMENTS,
    EVIDENCE_TYPES,
    INCIDENT_ENGINE_VERSION,
    INCIDENT_STATUSES,
    PROTECTION_FAMILIES,
    RECORD_ATTACHMENT_ROLES,
    RELATIONSHIP_TYPES,
    SCHEMA_VERSION,
    FaultEpisode,
    Incident,
    IncidentEvidence,
    IncidentFeedback,
    IncidentRecord,
    IncidentTimelineEvent,
    Reconstruction,
    RecordRelationship,
)
from .reconstruction import run_reconstruction as _run_reconstruction


class IncidentServiceError(Exception):
    """Raised for domain-level validation failures; routers map this to 4xx."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_analysis(analysis_id: str) -> RecordAnalysis:
    payload = load_analysis(analysis_id)
    if payload is None:
        raise IncidentServiceError(f"analysis_id '{analysis_id}' not found or expired.", status_code=404)
    return build_record_analysis(analysis_id, payload)


# --- Incident CRUD ------------------------------------------------------------

def create_incident(
    *,
    title: str,
    station_name: Optional[str] = None,
    bay_name: Optional[str] = None,
    asset_id: Optional[str] = None,
    asset_name: Optional[str] = None,
    asset_type: Optional[str] = None,
    voltage_level_kv: Optional[float] = None,
    protection_family: Optional[str] = None,
    operator_notes: Optional[str] = None,
) -> Incident:
    if asset_type is not None and asset_type not in ASSET_TYPES:
        raise IncidentServiceError(f"Unknown asset_type '{asset_type}'.")
    if protection_family is not None and protection_family not in PROTECTION_FAMILIES:
        raise IncidentServiceError(f"Unknown protection_family '{protection_family}'.")

    now = _now_iso()
    incident = Incident(
        incident_id=incident_storage.new_id(),
        title=title or "Untitled incident",
        status="DRAFT",
        station_name=station_name,
        bay_name=bay_name,
        asset_id=asset_id,
        asset_name=asset_name,
        asset_type=asset_type,
        voltage_level_kv=voltage_level_kv,
        protection_family=protection_family,
        clock_assessment="UNKNOWN",
        operator_notes=operator_notes,
        created_at=now,
        updated_at=now,
        schema_version=SCHEMA_VERSION,
    )
    _recompute_derived(incident, [])
    return incident_storage.create_incident(incident)


def get_incident(incident_id: str) -> Incident:
    incident = incident_storage.get_incident(incident_id)
    if incident is None:
        raise IncidentServiceError(f"Incident '{incident_id}' not found.", status_code=404)
    return incident


def list_incidents(*, status: Optional[str] = None, station_name: Optional[str] = None) -> list[Incident]:
    incidents = incident_storage.list_incidents()
    if status is not None:
        incidents = [i for i in incidents if i.status == status]
    else:
        # Default view hides archived incidents; pass status=ARCHIVED explicitly to see them.
        incidents = [i for i in incidents if i.status != "ARCHIVED"]
    if station_name is not None:
        incidents = [i for i in incidents if i.station_name == station_name]
    return incidents


def update_incident(incident_id: str, patch: dict[str, Any]) -> Incident:
    incident = get_incident(incident_id)

    if "status" in patch and patch["status"] is not None:
        if patch["status"] not in INCIDENT_STATUSES:
            raise IncidentServiceError(f"Unknown status '{patch['status']}'.")
        incident.status = patch["status"]
    if "title" in patch and patch["title"] is not None:
        incident.title = patch["title"]
    if "station_name" in patch:
        incident.station_name = patch["station_name"]
    if "bay_name" in patch:
        incident.bay_name = patch["bay_name"]
    if "asset_id" in patch:
        incident.asset_id = patch["asset_id"]
    if "asset_name" in patch:
        incident.asset_name = patch["asset_name"]
    if "asset_type" in patch and patch["asset_type"] is not None:
        if patch["asset_type"] not in ASSET_TYPES:
            raise IncidentServiceError(f"Unknown asset_type '{patch['asset_type']}'.")
        incident.asset_type = patch["asset_type"]
    if "voltage_level_kv" in patch:
        incident.voltage_level_kv = patch["voltage_level_kv"]
    if "protection_family" in patch and patch["protection_family"] is not None:
        if patch["protection_family"] not in PROTECTION_FAMILIES:
            raise IncidentServiceError(f"Unknown protection_family '{patch['protection_family']}'.")
        incident.protection_family = patch["protection_family"]
    if "clock_assessment" in patch and patch["clock_assessment"] is not None:
        if patch["clock_assessment"] not in CLOCK_ASSESSMENTS:
            raise IncidentServiceError(f"Unknown clock_assessment '{patch['clock_assessment']}'.")
        incident.clock_assessment = patch["clock_assessment"]
    if "clock_assessment_reason" in patch:
        incident.clock_assessment_reason = patch["clock_assessment_reason"]
    if "operator_notes" in patch:
        incident.operator_notes = patch["operator_notes"]
    if "incident_start_iso" in patch:
        incident.incident_start_iso = patch["incident_start_iso"]
    if "incident_end_iso" in patch:
        incident.incident_end_iso = patch["incident_end_iso"]

    incident.updated_at = _now_iso()
    return incident_storage.update_incident(incident)


def delete_incident(incident_id: str) -> None:
    incident_storage.delete_incident(incident_id)


# --- Record attachment ---------------------------------------------------------

def _attachment_warnings(incident: Incident, analysis: RecordAnalysis, override: bool) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    record_station = analysis.source_metadata.get("station_name") or None

    if not record_station:
        warnings.append({
            "type": "MISSING_STATION_METADATA",
            "description": "This record has no station name in its COMTRADE header.",
            "requires_review": True,
        })
    elif incident.station_name and record_station != incident.station_name:
        warnings.append({
            "type": "STATION_MISMATCH",
            "incident_station": incident.station_name,
            "record_station": record_station,
            "requires_review": True,
            "overridden": override,
        })

    if not analysis.source_metadata.get("rec_dev_id"):
        warnings.append({
            "type": "MISSING_RELAY_METADATA",
            "description": "This record has no relay/device id in its COMTRADE header.",
            "requires_review": True,
        })

    if not analysis.source_metadata.get("trigger_time_iso") and not analysis.source_metadata.get("start_time_iso"):
        warnings.append({
            "type": "NO_ABSOLUTE_TIME",
            "description": "This record has no absolute (wall-clock) timestamp; ordering will rely on manual/upload order.",
            "requires_review": False,
        })

    return warnings


def attach_record(
    incident_id: str,
    *,
    analysis_id: str,
    attachment_role: str = "UNKNOWN",
    bay_name: Optional[str] = None,
    relay_id: Optional[str] = None,
    relay_model: Optional[str] = None,
    protection_type: Optional[str] = None,
    source_filename: Optional[str] = None,
    override_warnings: bool = False,
    operator_notes: Optional[str] = None,
) -> IncidentRecord:
    incident = get_incident(incident_id)

    if attachment_role not in RECORD_ATTACHMENT_ROLES:
        raise IncidentServiceError(f"Unknown attachment_role '{attachment_role}'.")

    existing = incident_storage.list_incident_records(incident_id)
    if any(r.analysis_id == analysis_id for r in existing):
        raise IncidentServiceError(
            f"analysis_id '{analysis_id}' is already attached to incident '{incident_id}'.",
            status_code=409,
        )

    analysis = _canonical_analysis(analysis_id)
    warnings = _attachment_warnings(incident, analysis, override_warnings)
    blocking = [w for w in warnings if w.get("requires_review") and w["type"] == "STATION_MISMATCH" and not override_warnings]
    if blocking:
        raise IncidentServiceError(
            "Station mismatch between incident and record. Pass override_warnings=true to attach anyway.",
            status_code=409,
        )

    now = _now_iso()
    record = IncidentRecord(
        incident_record_id=incident_storage.new_id(),
        incident_id=incident_id,
        analysis_id=analysis_id,
        source_filename=source_filename,
        station_name=analysis.source_metadata.get("station_name") or None,
        bay_name=bay_name,
        relay_id=relay_id or analysis.source_metadata.get("rec_dev_id") or None,
        relay_model=relay_model,
        protection_type=protection_type,
        record_start_iso=analysis.source_metadata.get("start_time_iso"),
        trigger_time_iso=analysis.source_metadata.get("trigger_time_iso"),
        trigger_offset_s=analysis.source_metadata.get("trigger_offset_s"),
        sequence_index=len(existing),
        manual_order=None,
        order_source="UPLOAD_ORDER",
        attachment_role=attachment_role,
        inclusion_status="INCLUDED",
        canonical_snapshot=analysis.to_dict(),
        attachment_warnings=warnings,
        created_at=now,
    )
    incident_storage.create_incident_record(record)

    incident.record_ids = [r.analysis_id for r in existing] + [analysis_id]
    if bay_name and not incident.bay_name:
        incident.bay_name = bay_name
    if operator_notes:
        incident.operator_notes = ((incident.operator_notes or "") + f"\n[attach {analysis_id}] {operator_notes}").strip()
    incident.updated_at = now
    _recompute_derived(incident, incident_storage.list_incident_records(incident_id))
    incident_storage.update_incident(incident)

    return record


def detach_record(incident_id: str, incident_record_id: str) -> None:
    incident = get_incident(incident_id)
    record = incident_storage.get_incident_record(incident_record_id)
    if record is None or record.incident_id != incident_id:
        raise IncidentServiceError(f"Incident record '{incident_record_id}' not found on incident '{incident_id}'.", status_code=404)

    incident_storage.delete_incident_record(incident_record_id)
    remaining = incident_storage.list_incident_records(incident_id)
    incident.record_ids = [r.analysis_id for r in remaining]
    incident.updated_at = _now_iso()
    _recompute_derived(incident, remaining)
    incident_storage.update_incident(incident)


def list_records(incident_id: str) -> list[IncidentRecord]:
    get_incident(incident_id)  # 404 if missing
    return incident_storage.list_incident_records(incident_id)


def reorder_records(incident_id: str, ordered_incident_record_ids: list[str]) -> list[IncidentRecord]:
    """Persist an explicit manual order. Does not silently override an
    existing manual order elsewhere — this IS the explicit override."""
    incident = get_incident(incident_id)
    records = {r.incident_record_id: r for r in incident_storage.list_incident_records(incident_id)}

    missing = [rid for rid in ordered_incident_record_ids if rid not in records]
    if missing:
        raise IncidentServiceError(f"Unknown incident_record_id(s) for reorder: {missing}", status_code=404)
    if len(ordered_incident_record_ids) != len(records):
        raise IncidentServiceError("Reorder must include every attached incident_record_id exactly once.")

    for order, rid in enumerate(ordered_incident_record_ids):
        rec = records[rid]
        rec.manual_order = order
        rec.order_source = "MANUAL"
        incident_storage.update_incident_record(rec)

    updated = incident_storage.list_incident_records(incident_id)
    incident.updated_at = _now_iso()
    _recompute_derived(incident, updated)
    incident_storage.update_incident(incident)
    return updated


def refresh_snapshots(incident_id: str) -> list[IncidentRecord]:
    """Re-fetch canonical analysis for every attached record and store a new
    snapshot. Only runs on explicit request; keeps prior snapshot data inside
    the updated record (no feedback is touched or deleted)."""
    incident = get_incident(incident_id)
    records = incident_storage.list_incident_records(incident_id)
    refreshed = []
    for rec in records:
        try:
            analysis = _canonical_analysis(rec.analysis_id)
        except IncidentServiceError:
            refreshed.append(rec)
            continue
        rec.canonical_snapshot = analysis.to_dict()
        rec.attachment_warnings = _attachment_warnings(incident, analysis, override=True)
        rec.station_name = analysis.source_metadata.get("station_name") or rec.station_name
        rec.record_start_iso = analysis.source_metadata.get("start_time_iso")
        rec.trigger_time_iso = analysis.source_metadata.get("trigger_time_iso")
        rec.trigger_offset_s = analysis.source_metadata.get("trigger_offset_s")
        incident_storage.update_incident_record(rec)
        refreshed.append(rec)

    incident.updated_at = _now_iso()
    _recompute_derived(incident, refreshed)
    incident_storage.update_incident(incident)
    return refreshed


# --- Evidence -----------------------------------------------------------------

def add_evidence(
    incident_id: str,
    *,
    evidence_type: str,
    source: str = "",
    description: str = "",
    value: Any = None,
    confidence: str = "UNKNOWN",
    observed_at_iso: Optional[str] = None,
    attachment_name: Optional[str] = None,
    created_by: Optional[str] = None,
) -> IncidentEvidence:
    incident = get_incident(incident_id)
    if evidence_type not in EVIDENCE_TYPES:
        raise IncidentServiceError(f"Unknown evidence_type '{evidence_type}'.")

    now = _now_iso()
    evidence = IncidentEvidence(
        evidence_id=incident_storage.new_id(),
        incident_id=incident_id,
        evidence_type=evidence_type,
        source=source,
        description=description,
        value=value,
        confidence=confidence,
        observed_at_iso=observed_at_iso,
        attachment_name=attachment_name,
        created_by=created_by,
        created_at=now,
    )
    incident_storage.create_incident_evidence(evidence)

    incident.evidence_ids = [e.evidence_id for e in incident_storage.list_incident_evidence(incident_id)]
    incident.updated_at = now
    incident_storage.update_incident(incident)
    return evidence


def list_evidence(incident_id: str) -> list[IncidentEvidence]:
    get_incident(incident_id)
    return incident_storage.list_incident_evidence(incident_id)


def remove_evidence(incident_id: str, evidence_id: str) -> None:
    incident = get_incident(incident_id)
    incident_storage.delete_incident_evidence(evidence_id)
    incident.evidence_ids = [e.evidence_id for e in incident_storage.list_incident_evidence(incident_id)]
    incident.updated_at = _now_iso()
    incident_storage.update_incident(incident)


# --- Feedback ------------------------------------------------------------------

def save_feedback(incident_id: str, fields: dict[str, Any]) -> IncidentFeedback:
    incident = get_incident(incident_id)
    now = _now_iso()
    feedback = IncidentFeedback(
        feedback_id=incident_storage.new_id(),
        incident_id=incident_id,
        operator=fields.get("operator"),
        record_grouping_correct=fields.get("record_grouping_correct"),
        actual_record_count=fields.get("actual_record_count"),
        record_order_correct=fields.get("record_order_correct"),
        corrected_record_order=fields.get("corrected_record_order"),
        incident_start_correct=fields.get("incident_start_correct"),
        corrected_incident_start_iso=fields.get("corrected_incident_start_iso"),
        incident_end_correct=fields.get("incident_end_correct"),
        corrected_incident_end_iso=fields.get("corrected_incident_end_iso"),
        clock_assessment_correct=fields.get("clock_assessment_correct"),
        actual_clock_assessment=fields.get("actual_clock_assessment"),
        incident_interpretation_correct=fields.get("incident_interpretation_correct"),
        actual_incident_class=fields.get("actual_incident_class"),
        cause_correct=fields.get("cause_correct"),
        actual_root_cause=fields.get("actual_root_cause"),
        ground_truth_sources=list(fields.get("ground_truth_sources") or []),
        ground_truth_confidence=fields.get("ground_truth_confidence", "UNKNOWN"),
        include_for_future_analysis=fields.get("include_for_future_analysis", True),
        notes=fields.get("notes"),
        same_bay_correct=fields.get("same_bay_correct"),
        relationships_correct=fields.get("relationships_correct"),
        corrected_relationships=list(fields.get("corrected_relationships") or []),
        episode_grouping_correct=fields.get("episode_grouping_correct"),
        corrected_episode_groups=list(fields.get("corrected_episode_groups") or []),
        actual_episode_count=fields.get("actual_episode_count"),
        evolving_fault_correct=fields.get("evolving_fault_correct"),
        root_cause=fields.get("root_cause"),
        incident_snapshot=to_response(incident, incident_storage.list_incident_records(incident_id)),
        reconstruction_snapshot=_reconstruction_snapshot(incident_id),
        created_at=now,
    )
    return incident_storage.create_incident_feedback(feedback)


def _reconstruction_snapshot(incident_id: str) -> dict[str, Any]:
    """Best-effort: an incident may not have been reconstructed yet, in
    which case feedback is still saved with an empty reconstruction_snapshot
    rather than failing."""
    reconstruction = incident_storage.get_latest_reconstruction(incident_id)
    if reconstruction is None:
        return {}
    return to_reconstruction_response(
        reconstruction,
        incident_storage.list_timeline_events(incident_id),
        incident_storage.list_relationships(incident_id),
        incident_storage.list_episodes(incident_id),
    )


def list_feedback(incident_id: str) -> list[IncidentFeedback]:
    get_incident(incident_id)
    return incident_storage.list_incident_feedback(incident_id)


# --- Derived summary / interpretation ------------------------------------------

def _reclose_outcome(record: IncidentRecord) -> Optional[str]:
    """Map the canonical ``{time, success: bool}`` reclose event shape (see
    ``core.fault_detector``) to the ``"successful"/"failed"/None`` vocabulary
    used in incident-level summaries."""
    reclose_events = (record.canonical_snapshot or {}).get("observed_facts", {}).get("reclose_events", [])
    if not reclose_events or not isinstance(reclose_events[-1], dict):
        return None
    success = reclose_events[-1].get("success")
    if success is True:
        return "successful"
    if success is False:
        return "failed"
    return None


def _record_order_warning(records: list[IncidentRecord]) -> Optional[dict[str, Any]]:
    manually_ordered = [r for r in records if r.manual_order is not None]
    if len(manually_ordered) < 2:
        return None

    timed = [(r, r.trigger_time_iso or r.record_start_iso) for r in manually_ordered if (r.trigger_time_iso or r.record_start_iso)]
    if len(timed) < 2:
        return None

    manual_sorted = sorted(manually_ordered, key=lambda r: r.manual_order)
    time_sorted = sorted(timed, key=lambda pair: pair[1])
    manual_ids = [r.incident_record_id for r in manual_sorted if r.incident_record_id in {t[0].incident_record_id for t in timed}]
    time_ids = [pair[0].incident_record_id for pair in time_sorted]

    if manual_ids != time_ids:
        return {
            "type": "MANUAL_ORDER_DIFFERS_FROM_TIMESTAMP_ORDER",
            "description": "Manual record order does not match absolute-timestamp order. Review before relying on sequence.",
            "requires_review": True,
        }
    return None


def _recompute_derived(incident: Incident, records: list[IncidentRecord]) -> None:
    """Stage 1 summary: defensive facts only — no relationship inference."""
    included = [r for r in records if r.inclusion_status != "EXCLUDED"]

    records_with_time = [r for r in included if r.trigger_time_iso or r.record_start_iso]
    times = sorted(r.trigger_time_iso or r.record_start_iso for r in records_with_time)
    if times and not incident.incident_start_iso:
        incident.incident_start_iso = times[0]
    if times and not incident.incident_end_iso:
        incident.incident_end_iso = times[-1]

    protection_types = sorted({r.protection_type for r in included if r.protection_type})
    faulted_phase_sets = [
        r.canonical_snapshot.get("observed_facts", {}).get("faulted_phases", [])
        for r in included
    ]
    reclose_outcomes = [_reclose_outcome(r) for r in included]

    cause_hypotheses_per_record = []
    for r in included:
        hyps = r.canonical_snapshot.get("cause_hypotheses") or []
        top = hyps[0] if hyps else None
        cause_hypotheses_per_record.append({
            "analysis_id": r.analysis_id,
            "top_hypothesis": top.get("cause") if isinstance(top, dict) else None,
            "confidence": top.get("confidence") if isinstance(top, dict) else None,
            "scope": "RECORD_LOCAL_SIGNATURE",
        })

    incident.observed_summary = {
        "record_count": len(included),
        "records_with_absolute_time": len(records_with_time),
        "records_without_absolute_time": len(included) - len(records_with_time),
        "protection_types": protection_types,
        "faulted_phase_sets": faulted_phase_sets,
        "reclose_outcomes": reclose_outcomes,
        "cause_hypotheses_per_record": cause_hypotheses_per_record,
    }

    if len(included) == 0:
        incident.incident_interpretation = {
            "summary": "No records attached yet.",
            "scope": "INCIDENT_RECORD_COLLECTION_SUMMARY",
        }
    elif len(included) == 1:
        incident.incident_interpretation = {
            "summary": "Single record attached. Interpretation matches the record-level canonical analysis.",
            "scope": "INCIDENT_RECORD_COLLECTION_SUMMARY",
        }
    else:
        incident.incident_interpretation = {
            "summary": "Multiple records are attached. Automated inter-record reconstruction has not yet been performed.",
            "scope": "INCIDENT_RECORD_COLLECTION_SUMMARY",
        }

    missing: list[dict[str, Any]] = []
    if not incident.bay_name:
        missing.append({"type": "BAY_NAME_UNDETERMINED", "description": "Bay name has not been set for this incident."})
    if records_with_time != included:
        missing.append({"type": "RECORDS_MISSING_ABSOLUTE_TIME", "description": "One or more attached records lack an absolute timestamp."})
    if not any(r.attachment_role == "REMOTE_END" for r in included) and len(included) > 0:
        missing.append({"type": "REMOTE_END_UNAVAILABLE", "description": "No remote-end record is attached."})
    order_warning = _record_order_warning(included)
    if order_warning:
        missing.append({"type": "RECORD_ORDER_REQUIRES_REVIEW", "description": order_warning["description"]})

    incident.missing_evidence = missing


def _snapshot_versions(records: list[IncidentRecord]) -> list[dict[str, Any]]:
    versions = []
    for r in records:
        provenance = r.canonical_snapshot.get("provenance", {}) if r.canonical_snapshot else {}
        versions.append({
            "analysis_id": r.analysis_id,
            "canonical_schema_version": provenance.get("schema_version"),
            "model_version": provenance.get("model_version"),
            "feature_version": provenance.get("feature_version"),
            "timing_source": provenance.get("timing_source"),
        })
    return versions


def to_response(incident: Incident, records: list[IncidentRecord]) -> dict[str, Any]:
    data = incident.to_dict()
    data["records"] = [r.to_dict() for r in records]
    data["provenance"] = {
        "schema_version": incident.schema_version,
        "incident_engine_version": INCIDENT_ENGINE_VERSION,
        "record_snapshot_versions": _snapshot_versions(records),
        "created_at": incident.created_at,
        "updated_at": incident.updated_at,
    }
    return data


# --- Stage 2: reconstruction --------------------------------------------------

def reconstruct(
    incident_id: str,
    *,
    same_bay_override_reason: Optional[str] = None,
    same_bay_override_operator: Optional[str] = None,
) -> Reconstruction:
    """Run the Stage 2 reconstruction engine for this incident and persist a
    new versioned result. Never deletes prior reconstruction history."""
    incident = get_incident(incident_id)
    records = incident_storage.list_incident_records(incident_id)
    if not records:
        raise IncidentServiceError("Cannot reconstruct an incident with no attached records.", status_code=400)

    reconstruction, timeline_events, relationships, episodes = _run_reconstruction(
        incident,
        records,
        same_bay_override_reason=same_bay_override_reason,
        same_bay_override_operator=same_bay_override_operator,
    )

    incident_storage.replace_timeline_events(incident_id, timeline_events)
    incident_storage.replace_relationships(incident_id, relationships)
    incident_storage.replace_episodes(incident_id, episodes)
    incident_storage.create_reconstruction(reconstruction)

    incident.updated_at = _now_iso()
    incident_storage.update_incident(incident)

    return reconstruction


def get_reconstruction(incident_id: str, reconstruction_id: Optional[str] = None) -> Reconstruction:
    get_incident(incident_id)
    if reconstruction_id:
        reconstruction = incident_storage.get_reconstruction(reconstruction_id)
        if reconstruction is None or reconstruction.incident_id != incident_id:
            raise IncidentServiceError(f"Reconstruction '{reconstruction_id}' not found on incident '{incident_id}'.", status_code=404)
        return reconstruction

    reconstruction = incident_storage.get_latest_reconstruction(incident_id)
    if reconstruction is None:
        raise IncidentServiceError(f"Incident '{incident_id}' has not been reconstructed yet.", status_code=404)
    return reconstruction


def list_reconstructions(incident_id: str) -> list[Reconstruction]:
    get_incident(incident_id)
    return incident_storage.list_reconstructions(incident_id)


def get_timeline(incident_id: str) -> list[IncidentTimelineEvent]:
    get_incident(incident_id)
    return incident_storage.list_timeline_events(incident_id)


def get_relationships(incident_id: str) -> list[RecordRelationship]:
    get_incident(incident_id)
    return incident_storage.list_relationships(incident_id)


def get_episodes(incident_id: str) -> list[FaultEpisode]:
    get_incident(incident_id)
    return incident_storage.list_episodes(incident_id)


def override_relationship(
    incident_id: str,
    relationship_id: str,
    *,
    corrected_relationship: str,
    operator: Optional[str],
    reason: Optional[str],
) -> RecordRelationship:
    get_incident(incident_id)
    relationship = incident_storage.get_relationship(relationship_id)
    if relationship is None or relationship.incident_id != incident_id:
        raise IncidentServiceError(f"Relationship '{relationship_id}' not found on incident '{incident_id}'.", status_code=404)
    if corrected_relationship not in RELATIONSHIP_TYPES:
        raise IncidentServiceError(f"Unknown relationship_type '{corrected_relationship}'.")

    latest = incident_storage.get_latest_reconstruction(incident_id)

    relationship.override_previous_type = relationship.relationship_type
    relationship.relationship_type = corrected_relationship
    relationship.overridden = True
    relationship.override_operator = operator
    relationship.override_reason = reason
    relationship.override_at_iso = _now_iso()
    relationship.reconstruction_version = latest.reconstruction_id if latest else None

    return incident_storage.update_relationship(relationship)


def to_reconstruction_response(
    reconstruction: Reconstruction,
    timeline_events: list[IncidentTimelineEvent],
    relationships: list[RecordRelationship],
    episodes: list[FaultEpisode],
) -> dict[str, Any]:
    data = reconstruction.to_dict()
    data["timeline"] = [e.to_dict() for e in timeline_events]
    data["relationships"] = [r.to_dict() for r in relationships]
    data["episodes"] = [e.to_dict() for e in episodes]
    return data
