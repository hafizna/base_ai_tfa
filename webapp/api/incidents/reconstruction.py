"""Incident reconstruction engine — Stage 2.

Orchestrates the same-bay multi-record reconstruction pipeline:

    incident records
          -> same-bay assessment
          -> clock/alignment assessment
          -> canonical timeline events
          -> pairwise relationships
          -> duplicate/overlap grouping into fault episodes
          -> episode sequence interpretation
          -> incident summary and hypotheses (+ deterministic narrative)

Deterministic and auditable: every conclusion carries evidence, confidence,
assumptions, contradictory evidence, missing evidence, and provenance. Does
not retrain or replace LightGBM — cause hypotheses are read per-record via
``webapp.api.ml_predict.run_ml_prediction`` (never averaged/normalized across
records, see ``_physical_cause_evidence``) and are always labeled
``RECORD_LOCAL_SIGNATURE`` scope.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .. import ml_predict
from ..storage import load_analysis
from . import storage as incident_storage
from .alignment import assess_alignment
from .episodes import group_episodes
from .models import (
    RECONSTRUCTION_ENGINE_VERSION,
    RECONSTRUCTION_SCHEMA_VERSION,
    FaultEpisode,
    Incident,
    IncidentRecord,
    Reconstruction,
    RecordRelationship,
)
from .narrative import build_narrative
from .relationships import build_relationships
from .same_bay import assess_same_bay
from .timeline import build_timeline


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _record_ml_result(record: IncidentRecord) -> dict[str, Any]:
    """Best-effort full ``run_ml_prediction`` result for one record, computed
    fresh (never averaged with other records — see module docstring).
    Returns ``{}`` rather than raising if the stored analysis has expired or
    the model import fails; reconstruction must not fail because one
    record's ML call did."""
    payload = load_analysis(record.analysis_id)
    if payload is None:
        return {}
    relay_type = (record.protection_type or "21").upper()
    try:
        return ml_predict.run_ml_prediction(payload, relay_type)
    except Exception:
        return {}


def _physical_cause_evidence(records: list[IncidentRecord]) -> dict[str, Any]:
    """Per-record top cause hypothesis plus a qualitative consistency label.
    Deliberately does NOT average or otherwise combine per-record
    probabilities into one incident-level probability (spec section 11) —
    duplicate records aren't independent evidence, and different episodes
    may have different mechanisms entirely.

    Each record entry carries the exact audit trail used at reconstruction
    time (model_version, feature_version, calibration method, timing
    source, raw/calibrated probabilities, applied confidence caps) so the
    stored ``Reconstruction`` snapshot remains the source of truth even if
    the live model or a record's analysis session later changes.
    """
    record_entries = []
    top_causes = []
    for record in records:
        result = _record_ml_result(record)
        ranking = result.get("cause_ranking") or []
        top = ranking[0] if ranking else None
        cause = top.get("cause") if isinstance(top, dict) else None
        confidence = top.get("confidence") if isinstance(top, dict) else None
        meta = result.get("meta") or {}
        record_entries.append({
            "analysis_id": record.analysis_id,
            "incident_record_id": record.incident_record_id,
            "top_hypothesis": cause,
            "confidence": confidence,
            "cause_ranking": ranking,
            "model_version": meta.get("model_version"),
            "feature_version": meta.get("feature_version"),
            "calibration_method": meta.get("calibration_method_used") or (meta.get("calibration") or {}).get("method"),
            "timing_source": meta.get("timing_source"),
            "timing_confidence": meta.get("timing_confidence"),
            "raw_probabilities": result.get("raw_probabilities"),
            "calibrated_probabilities": result.get("calibrated_probabilities"),
            "applied_caps": result.get("applied_caps") or [],
        })
        if cause:
            top_causes.append(cause)

    if not record_entries:
        consistency = "INSUFFICIENT"
    elif not top_causes:
        consistency = "INSUFFICIENT"
    else:
        distinct = set(top_causes)
        if len(distinct) == 1:
            consistency = "CONSISTENT" if len(top_causes) == len(record_entries) else "MOSTLY_CONSISTENT"
        elif len(distinct) <= max(1, len(top_causes) // 2):
            consistency = "MOSTLY_CONSISTENT"
        else:
            consistency = "MIXED"

    return {
        "scope": "RECORD_LOCAL_SIGNATURES",
        "records": record_entries,
        "consistency": consistency,
        "incident_root_cause": "UNCONFIRMED",
    }


def _observed_incident_facts(records: list[IncidentRecord], episodes: list[FaultEpisode]) -> dict[str, Any]:
    times = []
    for r in records:
        t = _parse_iso(r.trigger_time_iso or r.record_start_iso)
        if t is not None:
            times.append(t)
    duration_ms = None
    if len(times) >= 2:
        duration_ms = (max(times) - min(times)).total_seconds() * 1000.0

    return {
        "record_count": len(records),
        "episode_count": len(episodes),
        "incident_duration_ms": duration_ms,
        "phase_sequence": [e.faulted_phases for e in episodes],
        "reclose_sequence": [e.reclose_outcome for e in episodes],
    }


def _protection_sequence_interpretation(episodes: list[FaultEpisode], relationships: list[RecordRelationship]) -> dict[str, Any]:
    if not episodes:
        return {"event_class": "NO_EPISODES", "summary": "No fault episodes could be reconstructed."}
    if len(episodes) == 1:
        ep = episodes[0]
        if ep.reclose_outcome == "failed":
            return {"event_class": "SINGLE_FAULT_FAILED_RECLOSE", "summary": "A single fault episode was followed by a failed reclose attempt."}
        if ep.reclose_outcome == "successful":
            return {"event_class": "SINGLE_TRANSIENT_FAULT", "summary": "A single transient fault episode was cleared with a successful reclose."}
        return {"event_class": "SINGLE_FAULT_EPISODE", "summary": "A single fault episode was reconstructed from the attached records."}

    relation_types = [e.relationship_to_previous for e in episodes[1:]]
    last = episodes[-1]

    if any(t == "POSSIBLE_EVOLVING_FAULT" for t in relation_types):
        if last.reclose_outcome == "failed":
            return {
                "event_class": "REPEATED_FAULT_WITH_FINAL_FAILED_RECLOSE",
                "summary": (
                    f"{len(episodes) - 1} earlier transient episode(s) were followed by a fault that may have evolved "
                    "into a more severe condition, ending in a failed reclose."
                ),
            }
        return {
            "event_class": "POSSIBLE_EVOLVING_FAULT_SEQUENCE",
            "summary": "The fault signature appears to evolve across episodes; treat as provisional pending further evidence.",
        }

    if all(t == "REPEATED_FAULT" for t in relation_types):
        return {
            "event_class": "REPEATED_INDEPENDENT_FAULTS",
            "summary": f"{len(episodes)} episodes with a similar fault signature were separated by clear intervals, consistent with repeated independent faults.",
        }

    if all(t in ("RECLOSE_SEQUENCE",) for t in relation_types):
        return {
            "event_class": "TRANSIENT_FAULT_WITH_RECLOSE_SEQUENCE",
            "summary": "A fault episode was followed by a captured breaker reclose sequence.",
        }

    return {
        "event_class": "MULTIPLE_EPISODES_MIXED_RELATIONSHIP",
        "summary": f"{len(episodes)} fault episodes were reconstructed with mixed relationships between them; see per-episode relationship_to_previous for detail.",
    }


def _incident_hypotheses(episodes: list[FaultEpisode], relationships: list[RecordRelationship]) -> list[dict[str, Any]]:
    hypotheses = []
    for rel in relationships:
        if rel.relationship_type != "POSSIBLE_EVOLVING_FAULT":
            continue
        evidence_for = [e.get("description") or e.get("type") for e in rel.evidence_for]
        evidence_against = [e.get("description") or e.get("type") for e in rel.evidence_against]
        hypotheses.append({
            "hypothesis": "POSSIBLE_EVOLVING_FAULT",
            "confidence": rel.confidence,
            "evidence_for": [e for e in evidence_for if e],
            "evidence_against": [e for e in evidence_against if e],
        })
    return hypotheses


def run_reconstruction(
    incident: Incident,
    records: list[IncidentRecord],
    *,
    same_bay_override_reason: Optional[str] = None,
    same_bay_override_operator: Optional[str] = None,
) -> tuple[Reconstruction, list, list[RecordRelationship], list[FaultEpisode]]:
    """Run the full Stage 2 reconstruction pipeline for one incident.

    Returns (reconstruction, timeline_events, relationships, episodes) — the
    caller (service layer) is responsible for persisting them and linking
    IDs, and for reconstruction version bookkeeping.
    """
    reconstruction_id = incident_storage.new_id()
    id_counter = {"n": 0}

    def new_id() -> str:
        id_counter["n"] += 1
        return f"{reconstruction_id}-{id_counter['n']}"

    same_bay = assess_same_bay(
        incident.station_name,
        incident.bay_name,
        records,
        override_reason=same_bay_override_reason,
        override_operator=same_bay_override_operator,
        override_at_iso=_now_iso() if same_bay_override_reason or same_bay_override_operator else None,
    )

    alignment = assess_alignment(records)
    timeline_events = build_timeline(incident.incident_id, records, alignment, new_id)
    relationships = build_relationships(incident.incident_id, records, alignment, new_id)

    # Computed once, before episode grouping, so episodes and the incident-level
    # physical_cause_evidence report exactly the same per-record ML call
    # result (same model_version/probabilities) rather than invoking
    # LightGBM a second time per episode.
    physical_cause = _physical_cause_evidence(records)
    cause_lookup = {e["incident_record_id"]: e for e in physical_cause["records"]}

    episodes = group_episodes(incident.incident_id, records, relationships, alignment.record_order, new_id, record_cause_lookup=cause_lookup)

    observed_facts = _observed_incident_facts(records, episodes)
    interpretation = _protection_sequence_interpretation(episodes, relationships)
    hypotheses = _incident_hypotheses(episodes, relationships)

    narrative = build_narrative(
        episodes,
        observed_facts.get("incident_duration_ms"),
        same_bay.status,
        physical_cause.get("consistency", "INSUFFICIENT"),
        hypotheses,
    )

    prior = incident_storage.get_latest_reconstruction(incident.incident_id)

    reconstruction = Reconstruction(
        reconstruction_id=reconstruction_id,
        incident_id=incident.incident_id,
        engine_version=RECONSTRUCTION_ENGINE_VERSION,
        schema_version=RECONSTRUCTION_SCHEMA_VERSION,
        same_bay_status=same_bay.status,
        same_bay_evidence=same_bay.evidence,
        same_bay_override=same_bay.override,
        alignment=alignment.to_dict(),
        timeline_event_ids=[e.timeline_event_id for e in timeline_events],
        relationship_ids=[r.relationship_id for r in relationships],
        episode_ids=[e.episode_id for e in episodes],
        observed_incident_facts=observed_facts,
        protection_sequence_interpretation=interpretation,
        incident_hypotheses=hypotheses,
        physical_cause_evidence=physical_cause,
        narrative=narrative,
        record_snapshot_versions=_snapshot_versions(records),
        is_latest=True,
        supersedes=prior.reconstruction_id if prior else None,
        created_at=_now_iso(),
    )

    return reconstruction, timeline_events, relationships, episodes


def _snapshot_versions(records: list[IncidentRecord]) -> list[dict[str, Any]]:
    versions = []
    for r in records:
        provenance = (r.canonical_snapshot or {}).get("provenance", {})
        versions.append({
            "analysis_id": r.analysis_id,
            "incident_record_id": r.incident_record_id,
            "canonical_schema_version": provenance.get("schema_version"),
            "model_version": provenance.get("model_version"),
            "feature_version": provenance.get("feature_version"),
            "timing_source": provenance.get("timing_source"),
        })
    return versions
