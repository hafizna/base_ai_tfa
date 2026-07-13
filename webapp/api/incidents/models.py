"""Incident domain model — Stage 1 + Stage 2.

Separates record-level analysis (Stage 0 ``RecordAnalysis`` — what one
COMTRADE shows) from incident-level analysis (which records belong together,
their manual order, site/asset metadata, evidence, and operator feedback —
Stage 1), and from same-bay multi-record reconstruction (alignment, canonical
timeline, pairwise relationships, fault episodes, and narrative — Stage 2).

Stage 2 adds relationship classification, episode grouping, and reconstruction
versioning, but every conclusion still traces back to Stage 0 canonical
timing/observed facts. Enums here are versioned and additive so future stages
can extend them without a breaking migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "stage2"
INCIDENT_ENGINE_VERSION = "stage1.0"
RECONSTRUCTION_ENGINE_VERSION = "stage2.0"
RECONSTRUCTION_SCHEMA_VERSION = "stage2"


# --- Enums (plain string constants, not pydantic/Enum classes, so legacy or
# unrecognised values never hard-fail a read — see training_retention.py for
# the same convention used with ground-truth vocab). ---

INCIDENT_STATUSES = {
    "DRAFT",
    "OPEN",
    "UNDER_REVIEW",
    "CONFIRMED",
    "CLOSED",
    "ARCHIVED",
}

ASSET_TYPES = {
    "TRANSMISSION_LINE",
    "TRANSFORMER",
    "BUSBAR",
    "FEEDER",
    "REACTOR",
    "CAPACITOR",
    "OTHER",
    "UNKNOWN",
}

PROTECTION_FAMILIES = {
    "DISTANCE",
    "LINE_DIFFERENTIAL",
    "TRANSFORMER_DIFFERENTIAL",
    "OVERCURRENT",
    "REF",
    "SBEF",
    "MIXED",
    "UNKNOWN",
}

CLOCK_ASSESSMENTS = {
    "SYNCHRONIZED",
    "LIKELY_SYNCHRONIZED",
    "ORDER_ONLY",
    "UNTRUSTED",
    "UNKNOWN",
}

RECORD_ATTACHMENT_ROLES = {
    "PRIMARY",
    "SUPPORTING",
    "REMOTE_END",
    "BACKUP_RELAY",
    "DFR_EXTERNAL",
    "OTHER",
    "UNKNOWN",
}

INCLUSION_STATUSES = {
    "INCLUDED",
    "EXCLUDED",
    "PENDING_REVIEW",
}

ORDER_SOURCES = {
    "ABSOLUTE_TIME",
    "MANUAL",
    "UPLOAD_ORDER",
    "UNKNOWN",
}

EVIDENCE_TYPES = {
    "COMTRADE_RECORD",
    "REMOTE_END_COMTRADE",
    "RELAY_EVENT_REPORT",
    "OPERATOR_SOE",
    "FIELD_INSPECTION",
    "PATROL_REPORT",
    "LIGHTNING_DETECTION",
    "PROTECTION_ENGINEER_NOTE",
    "PHOTO",
    "OTHER",
}

EVIDENCE_CONFIDENCE_LEVELS = {"CONFIRMED", "PROBABLE", "POSSIBLE", "UNKNOWN"}

# --- Stage 2 enums -----------------------------------------------------------

SAME_BAY_STATUSES = {
    "CONFIRMED_SAME_BAY",
    "LIKELY_SAME_BAY",
    "MISMATCH_REQUIRES_REVIEW",
    "UNKNOWN",
}

ALIGNMENT_STATUSES = {
    "ALIGNED",
    "LIKELY_ALIGNED",
    "ORDER_ONLY",
    "MANUAL_ORDER",
    "UNTRUSTED",
    "INSUFFICIENT_DATA",
}

TIMELINE_EVENT_TYPES = {
    "RECORD_START",
    "RECORD_TRIGGER",
    "FAULT_INCEPTION",
    "PROTECTION_PICKUP",
    "ZONE_OPERATE",
    "TRIP_COMMAND",
    "BREAKER_OPEN",
    "FAULT_CLEARING",
    "RECLOSE_START",
    "RECLOSE_SUCCESS",
    "RECLOSE_FAILED",
    "REFAULT",
    "RECORD_END",
    "DATA_GAP",
    "CLOCK_WARNING",
    "MANUAL_ANNOTATION",
}

RELATIONSHIP_TYPES = {
    "DUPLICATE_TRIGGER",
    "OVERLAPPING_CAPTURE",
    "CONTINUATION",
    "RECLOSE_SEQUENCE",
    "NEW_FAULT_EPISODE",
    "REPEATED_FAULT",
    "POSSIBLE_EVOLVING_FAULT",
    "UNRELATED",
    "UNCERTAIN",
}

CONSISTENCY_LEVELS = {
    "CONSISTENT",
    "MOSTLY_CONSISTENT",
    "MIXED",
    "CONTRADICTORY",
    "INSUFFICIENT",
}


@dataclass
class Incident:
    """One incident: a manually-curated grouping of one or more COMTRADE records."""

    incident_id: str
    title: str
    status: str = "DRAFT"

    station_name: Optional[str] = None
    bay_name: Optional[str] = None
    asset_id: Optional[str] = None
    asset_name: Optional[str] = None
    asset_type: Optional[str] = None
    voltage_level_kv: Optional[float] = None
    protection_family: Optional[str] = None

    incident_start_iso: Optional[str] = None
    incident_end_iso: Optional[str] = None

    clock_assessment: str = "UNKNOWN"
    clock_assessment_reason: Optional[str] = None

    record_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    observed_summary: dict[str, Any] = field(default_factory=dict)
    incident_interpretation: dict[str, Any] = field(default_factory=dict)
    incident_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)

    operator_notes: Optional[str] = None

    created_at: str = ""
    updated_at: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "status": self.status,
            "station_name": self.station_name,
            "bay_name": self.bay_name,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "asset_type": self.asset_type,
            "voltage_level_kv": self.voltage_level_kv,
            "protection_family": self.protection_family,
            "incident_start_iso": self.incident_start_iso,
            "incident_end_iso": self.incident_end_iso,
            "clock_assessment": self.clock_assessment,
            "clock_assessment_reason": self.clock_assessment_reason,
            "record_ids": list(self.record_ids),
            "evidence_ids": list(self.evidence_ids),
            "observed_summary": self.observed_summary,
            "incident_interpretation": self.incident_interpretation,
            "incident_hypotheses": self.incident_hypotheses,
            "missing_evidence": self.missing_evidence,
            "operator_notes": self.operator_notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Incident":
        """Reconstruct from persisted JSON, applying safe defaults for any
        field added after a given incident was first written."""
        return Incident(
            incident_id=data["incident_id"],
            title=data.get("title", ""),
            status=data.get("status", "DRAFT"),
            station_name=data.get("station_name"),
            bay_name=data.get("bay_name"),
            asset_id=data.get("asset_id"),
            asset_name=data.get("asset_name"),
            asset_type=data.get("asset_type"),
            voltage_level_kv=data.get("voltage_level_kv"),
            protection_family=data.get("protection_family"),
            incident_start_iso=data.get("incident_start_iso"),
            incident_end_iso=data.get("incident_end_iso"),
            clock_assessment=data.get("clock_assessment", "UNKNOWN"),
            clock_assessment_reason=data.get("clock_assessment_reason"),
            record_ids=list(data.get("record_ids") or []),
            evidence_ids=list(data.get("evidence_ids") or []),
            observed_summary=data.get("observed_summary") or {},
            incident_interpretation=data.get("incident_interpretation") or {},
            incident_hypotheses=list(data.get("incident_hypotheses") or []),
            missing_evidence=list(data.get("missing_evidence") or []),
            operator_notes=data.get("operator_notes"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class IncidentRecord:
    """One COMTRADE record (Stage 0 analysis) attached to an incident."""

    incident_record_id: str
    incident_id: str
    analysis_id: str

    source_filename: Optional[str] = None
    station_name: Optional[str] = None
    bay_name: Optional[str] = None
    relay_id: Optional[str] = None
    relay_model: Optional[str] = None
    protection_type: Optional[str] = None

    record_start_iso: Optional[str] = None
    trigger_time_iso: Optional[str] = None
    trigger_offset_s: Optional[float] = None

    sequence_index: int = 0
    manual_order: Optional[int] = None
    order_source: str = "UPLOAD_ORDER"

    attachment_role: str = "UNKNOWN"
    inclusion_status: str = "INCLUDED"
    exclusion_reason: Optional[str] = None

    canonical_snapshot: dict[str, Any] = field(default_factory=dict)
    attachment_warnings: list[dict[str, Any]] = field(default_factory=list)

    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_record_id": self.incident_record_id,
            "incident_id": self.incident_id,
            "analysis_id": self.analysis_id,
            "source_filename": self.source_filename,
            "station_name": self.station_name,
            "bay_name": self.bay_name,
            "relay_id": self.relay_id,
            "relay_model": self.relay_model,
            "protection_type": self.protection_type,
            "record_start_iso": self.record_start_iso,
            "trigger_time_iso": self.trigger_time_iso,
            "trigger_offset_s": self.trigger_offset_s,
            "sequence_index": self.sequence_index,
            "manual_order": self.manual_order,
            "order_source": self.order_source,
            "attachment_role": self.attachment_role,
            "inclusion_status": self.inclusion_status,
            "exclusion_reason": self.exclusion_reason,
            "canonical_snapshot": self.canonical_snapshot,
            "attachment_warnings": self.attachment_warnings,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "IncidentRecord":
        return IncidentRecord(
            incident_record_id=data["incident_record_id"],
            incident_id=data["incident_id"],
            analysis_id=data["analysis_id"],
            source_filename=data.get("source_filename"),
            station_name=data.get("station_name"),
            bay_name=data.get("bay_name"),
            relay_id=data.get("relay_id"),
            relay_model=data.get("relay_model"),
            protection_type=data.get("protection_type"),
            record_start_iso=data.get("record_start_iso"),
            trigger_time_iso=data.get("trigger_time_iso"),
            trigger_offset_s=data.get("trigger_offset_s"),
            sequence_index=data.get("sequence_index", 0),
            manual_order=data.get("manual_order"),
            order_source=data.get("order_source", "UPLOAD_ORDER"),
            attachment_role=data.get("attachment_role", "UNKNOWN"),
            inclusion_status=data.get("inclusion_status", "INCLUDED"),
            exclusion_reason=data.get("exclusion_reason"),
            canonical_snapshot=data.get("canonical_snapshot") or {},
            attachment_warnings=list(data.get("attachment_warnings") or []),
            created_at=data.get("created_at", ""),
        )


@dataclass
class IncidentEvidence:
    evidence_id: str
    incident_id: str
    evidence_type: str
    source: str = ""
    description: str = ""
    value: Any = None
    confidence: str = "UNKNOWN"
    observed_at_iso: Optional[str] = None
    attachment_name: Optional[str] = None
    created_by: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "incident_id": self.incident_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "description": self.description,
            "value": self.value,
            "confidence": self.confidence,
            "observed_at_iso": self.observed_at_iso,
            "attachment_name": self.attachment_name,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "IncidentEvidence":
        return IncidentEvidence(
            evidence_id=data["evidence_id"],
            incident_id=data["incident_id"],
            evidence_type=data.get("evidence_type", "OTHER"),
            source=data.get("source", ""),
            description=data.get("description", ""),
            value=data.get("value"),
            confidence=data.get("confidence", "UNKNOWN"),
            observed_at_iso=data.get("observed_at_iso"),
            attachment_name=data.get("attachment_name"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at", ""),
        )


@dataclass
class IncidentFeedback:
    feedback_id: str
    incident_id: str
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

    ground_truth_sources: list[str] = field(default_factory=list)
    ground_truth_confidence: str = "UNKNOWN"

    include_for_future_analysis: bool = True
    notes: Optional[str] = None

    # --- Stage 2: reconstruction correction fields. All optional so Stage 1
    # feedback payloads keep working unchanged. ---
    same_bay_correct: Optional[bool] = None

    relationships_correct: Optional[bool] = None
    corrected_relationships: list[dict[str, Any]] = field(default_factory=list)

    episode_grouping_correct: Optional[bool] = None
    corrected_episode_groups: list[list[str]] = field(default_factory=list)
    actual_episode_count: Optional[int] = None

    evolving_fault_correct: Optional[bool] = None

    root_cause: Optional[str] = None

    incident_snapshot: dict[str, Any] = field(default_factory=dict)
    reconstruction_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "incident_id": self.incident_id,
            "operator": self.operator,
            "record_grouping_correct": self.record_grouping_correct,
            "actual_record_count": self.actual_record_count,
            "record_order_correct": self.record_order_correct,
            "corrected_record_order": self.corrected_record_order,
            "incident_start_correct": self.incident_start_correct,
            "corrected_incident_start_iso": self.corrected_incident_start_iso,
            "incident_end_correct": self.incident_end_correct,
            "corrected_incident_end_iso": self.corrected_incident_end_iso,
            "same_bay_correct": self.same_bay_correct,
            "relationships_correct": self.relationships_correct,
            "corrected_relationships": self.corrected_relationships,
            "episode_grouping_correct": self.episode_grouping_correct,
            "corrected_episode_groups": self.corrected_episode_groups,
            "actual_episode_count": self.actual_episode_count,
            "evolving_fault_correct": self.evolving_fault_correct,
            "root_cause": self.root_cause,
            "reconstruction_snapshot": self.reconstruction_snapshot,
            "clock_assessment_correct": self.clock_assessment_correct,
            "actual_clock_assessment": self.actual_clock_assessment,
            "incident_interpretation_correct": self.incident_interpretation_correct,
            "actual_incident_class": self.actual_incident_class,
            "cause_correct": self.cause_correct,
            "actual_root_cause": self.actual_root_cause,
            "ground_truth_sources": self.ground_truth_sources,
            "ground_truth_confidence": self.ground_truth_confidence,
            "include_for_future_analysis": self.include_for_future_analysis,
            "notes": self.notes,
            "incident_snapshot": self.incident_snapshot,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "IncidentFeedback":
        return IncidentFeedback(
            feedback_id=data["feedback_id"],
            incident_id=data["incident_id"],
            operator=data.get("operator"),
            record_grouping_correct=data.get("record_grouping_correct"),
            actual_record_count=data.get("actual_record_count"),
            record_order_correct=data.get("record_order_correct"),
            corrected_record_order=data.get("corrected_record_order"),
            incident_start_correct=data.get("incident_start_correct"),
            corrected_incident_start_iso=data.get("corrected_incident_start_iso"),
            incident_end_correct=data.get("incident_end_correct"),
            corrected_incident_end_iso=data.get("corrected_incident_end_iso"),
            clock_assessment_correct=data.get("clock_assessment_correct"),
            actual_clock_assessment=data.get("actual_clock_assessment"),
            incident_interpretation_correct=data.get("incident_interpretation_correct"),
            actual_incident_class=data.get("actual_incident_class"),
            cause_correct=data.get("cause_correct"),
            actual_root_cause=data.get("actual_root_cause"),
            ground_truth_sources=list(data.get("ground_truth_sources") or []),
            ground_truth_confidence=data.get("ground_truth_confidence", "UNKNOWN"),
            include_for_future_analysis=data.get("include_for_future_analysis", True),
            notes=data.get("notes"),
            same_bay_correct=data.get("same_bay_correct"),
            relationships_correct=data.get("relationships_correct"),
            corrected_relationships=list(data.get("corrected_relationships") or []),
            episode_grouping_correct=data.get("episode_grouping_correct"),
            corrected_episode_groups=list(data.get("corrected_episode_groups") or []),
            actual_episode_count=data.get("actual_episode_count"),
            evolving_fault_correct=data.get("evolving_fault_correct"),
            root_cause=data.get("root_cause"),
            incident_snapshot=data.get("incident_snapshot") or {},
            reconstruction_snapshot=data.get("reconstruction_snapshot") or {},
            created_at=data.get("created_at", ""),
        )


# --- Stage 2: same-bay multi-record reconstruction ---------------------------

@dataclass
class AlignmentAssessment:
    """Clock/ordering trust assessment across an incident's records.

    Never performs speculative clock correction — if precise alignment
    cannot be trusted, ``status`` must be ``ORDER_ONLY`` (or worse) and
    ``pairwise_gaps_ms`` must not assert a false-precision millisecond gap.
    """

    status: str = "INSUFFICIENT_DATA"
    confidence: float = 0.0
    order_source: str = "UNKNOWN"
    record_order: list[str] = field(default_factory=list)
    pairwise_gaps_ms: list[dict[str, Any]] = field(default_factory=list)
    overlap_groups: list[list[str]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "order_source": self.order_source,
            "record_order": self.record_order,
            "pairwise_gaps_ms": self.pairwise_gaps_ms,
            "overlap_groups": self.overlap_groups,
            "warnings": self.warnings,
            "assumptions": self.assumptions,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AlignmentAssessment":
        return AlignmentAssessment(
            status=data.get("status", "INSUFFICIENT_DATA"),
            confidence=data.get("confidence", 0.0),
            order_source=data.get("order_source", "UNKNOWN"),
            record_order=list(data.get("record_order") or []),
            pairwise_gaps_ms=list(data.get("pairwise_gaps_ms") or []),
            overlap_groups=list(data.get("overlap_groups") or []),
            warnings=list(data.get("warnings") or []),
            assumptions=list(data.get("assumptions") or []),
        )


@dataclass
class IncidentTimelineEvent:
    timeline_event_id: str
    incident_id: str
    incident_record_id: Optional[str] = None
    episode_id: Optional[str] = None

    event_type: str = "MANUAL_ANNOTATION"
    absolute_time_iso: Optional[str] = None
    relative_incident_ms: Optional[float] = None
    relative_record_ms: Optional[float] = None

    source: str = ""
    label: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline_event_id": self.timeline_event_id,
            "incident_id": self.incident_id,
            "incident_record_id": self.incident_record_id,
            "episode_id": self.episode_id,
            "event_type": self.event_type,
            "absolute_time_iso": self.absolute_time_iso,
            "relative_incident_ms": self.relative_incident_ms,
            "relative_record_ms": self.relative_record_ms,
            "source": self.source,
            "label": self.label,
            "details": self.details,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "IncidentTimelineEvent":
        return IncidentTimelineEvent(
            timeline_event_id=data["timeline_event_id"],
            incident_id=data["incident_id"],
            incident_record_id=data.get("incident_record_id"),
            episode_id=data.get("episode_id"),
            event_type=data.get("event_type", "MANUAL_ANNOTATION"),
            absolute_time_iso=data.get("absolute_time_iso"),
            relative_incident_ms=data.get("relative_incident_ms"),
            relative_record_ms=data.get("relative_record_ms"),
            source=data.get("source", ""),
            label=data.get("label", ""),
            details=data.get("details") or {},
            confidence=data.get("confidence", 0.0),
            provenance=data.get("provenance") or {},
        )


@dataclass
class RecordRelationship:
    relationship_id: str
    incident_id: str
    left_record_id: str
    right_record_id: str
    relationship_type: str = "UNCERTAIN"
    confidence: float = 0.0
    evidence_for: list[dict[str, Any]] = field(default_factory=list)
    evidence_against: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Manual override provenance (section 13 / 7).
    overridden: bool = False
    override_operator: Optional[str] = None
    override_reason: Optional[str] = None
    override_previous_type: Optional[str] = None
    override_at_iso: Optional[str] = None
    reconstruction_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "incident_id": self.incident_id,
            "left_record_id": self.left_record_id,
            "right_record_id": self.right_record_id,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "overridden": self.overridden,
            "override_operator": self.override_operator,
            "override_reason": self.override_reason,
            "override_previous_type": self.override_previous_type,
            "override_at_iso": self.override_at_iso,
            "reconstruction_version": self.reconstruction_version,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RecordRelationship":
        return RecordRelationship(
            relationship_id=data["relationship_id"],
            incident_id=data["incident_id"],
            left_record_id=data["left_record_id"],
            right_record_id=data["right_record_id"],
            relationship_type=data.get("relationship_type", "UNCERTAIN"),
            confidence=data.get("confidence", 0.0),
            evidence_for=list(data.get("evidence_for") or []),
            evidence_against=list(data.get("evidence_against") or []),
            assumptions=list(data.get("assumptions") or []),
            warnings=list(data.get("warnings") or []),
            metrics=data.get("metrics") or {},
            overridden=data.get("overridden", False),
            override_operator=data.get("override_operator"),
            override_reason=data.get("override_reason"),
            override_previous_type=data.get("override_previous_type"),
            override_at_iso=data.get("override_at_iso"),
            reconstruction_version=data.get("reconstruction_version"),
        )


@dataclass
class FaultEpisode:
    episode_id: str
    incident_id: str
    member_record_ids: list[str] = field(default_factory=list)

    episode_index: int = 0

    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    duration_ms: Optional[float] = None

    faulted_phases: list[str] = field(default_factory=list)
    fault_type: Optional[str] = None

    zone_operations: list[str] = field(default_factory=list)
    trip_types: list[str] = field(default_factory=list)
    reclose_outcome: Optional[str] = None

    electrical_summary: dict[str, Any] = field(default_factory=dict)
    local_cause_hypotheses: list[dict[str, Any]] = field(default_factory=list)

    relationship_to_previous: Optional[str] = None
    confidence: float = 0.0

    observed_facts: dict[str, Any] = field(default_factory=dict)
    interpretation: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "incident_id": self.incident_id,
            "member_record_ids": self.member_record_ids,
            "episode_index": self.episode_index,
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "duration_ms": self.duration_ms,
            "faulted_phases": self.faulted_phases,
            "fault_type": self.fault_type,
            "zone_operations": self.zone_operations,
            "trip_types": self.trip_types,
            "reclose_outcome": self.reclose_outcome,
            "electrical_summary": self.electrical_summary,
            "local_cause_hypotheses": self.local_cause_hypotheses,
            "relationship_to_previous": self.relationship_to_previous,
            "confidence": self.confidence,
            "observed_facts": self.observed_facts,
            "interpretation": self.interpretation,
            "missing_evidence": self.missing_evidence,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FaultEpisode":
        return FaultEpisode(
            episode_id=data["episode_id"],
            incident_id=data["incident_id"],
            member_record_ids=list(data.get("member_record_ids") or []),
            episode_index=data.get("episode_index", 0),
            start_iso=data.get("start_iso"),
            end_iso=data.get("end_iso"),
            duration_ms=data.get("duration_ms"),
            faulted_phases=list(data.get("faulted_phases") or []),
            fault_type=data.get("fault_type"),
            zone_operations=list(data.get("zone_operations") or []),
            trip_types=list(data.get("trip_types") or []),
            reclose_outcome=data.get("reclose_outcome"),
            electrical_summary=data.get("electrical_summary") or {},
            local_cause_hypotheses=list(data.get("local_cause_hypotheses") or []),
            relationship_to_previous=data.get("relationship_to_previous"),
            confidence=data.get("confidence", 0.0),
            observed_facts=data.get("observed_facts") or {},
            interpretation=data.get("interpretation") or {},
            missing_evidence=list(data.get("missing_evidence") or []),
            provenance=data.get("provenance") or {},
        )


@dataclass
class Reconstruction:
    """One versioned run of the Stage 2 reconstruction engine for an incident.

    Re-running reconstruction never deletes a prior version — ``supersedes``
    links to the previous reconstruction_id, and callers can still fetch old
    versions for audit even though the latest is served by default.
    """

    reconstruction_id: str
    incident_id: str
    engine_version: str = RECONSTRUCTION_ENGINE_VERSION
    schema_version: str = RECONSTRUCTION_SCHEMA_VERSION

    same_bay_status: str = "UNKNOWN"
    same_bay_evidence: list[dict[str, Any]] = field(default_factory=list)
    same_bay_override: Optional[dict[str, Any]] = None

    alignment: dict[str, Any] = field(default_factory=dict)
    timeline_event_ids: list[str] = field(default_factory=list)
    relationship_ids: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)

    observed_incident_facts: dict[str, Any] = field(default_factory=dict)
    protection_sequence_interpretation: dict[str, Any] = field(default_factory=dict)
    incident_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    physical_cause_evidence: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""

    record_snapshot_versions: list[dict[str, Any]] = field(default_factory=list)
    is_latest: bool = True
    supersedes: Optional[str] = None

    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconstruction_id": self.reconstruction_id,
            "incident_id": self.incident_id,
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "same_bay_status": self.same_bay_status,
            "same_bay_evidence": self.same_bay_evidence,
            "same_bay_override": self.same_bay_override,
            "alignment": self.alignment,
            "timeline_event_ids": self.timeline_event_ids,
            "relationship_ids": self.relationship_ids,
            "episode_ids": self.episode_ids,
            "observed_incident_facts": self.observed_incident_facts,
            "protection_sequence_interpretation": self.protection_sequence_interpretation,
            "incident_hypotheses": self.incident_hypotheses,
            "physical_cause_evidence": self.physical_cause_evidence,
            "narrative": self.narrative,
            "record_snapshot_versions": self.record_snapshot_versions,
            "is_latest": self.is_latest,
            "supersedes": self.supersedes,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Reconstruction":
        return Reconstruction(
            reconstruction_id=data["reconstruction_id"],
            incident_id=data["incident_id"],
            engine_version=data.get("engine_version", RECONSTRUCTION_ENGINE_VERSION),
            schema_version=data.get("schema_version", RECONSTRUCTION_SCHEMA_VERSION),
            same_bay_status=data.get("same_bay_status", "UNKNOWN"),
            same_bay_evidence=list(data.get("same_bay_evidence") or []),
            same_bay_override=data.get("same_bay_override"),
            alignment=data.get("alignment") or {},
            timeline_event_ids=list(data.get("timeline_event_ids") or []),
            relationship_ids=list(data.get("relationship_ids") or []),
            episode_ids=list(data.get("episode_ids") or []),
            observed_incident_facts=data.get("observed_incident_facts") or {},
            protection_sequence_interpretation=data.get("protection_sequence_interpretation") or {},
            incident_hypotheses=list(data.get("incident_hypotheses") or []),
            physical_cause_evidence=data.get("physical_cause_evidence") or {},
            narrative=data.get("narrative", ""),
            record_snapshot_versions=list(data.get("record_snapshot_versions") or []),
            is_latest=data.get("is_latest", True),
            supersedes=data.get("supersedes"),
            created_at=data.get("created_at", ""),
        )
