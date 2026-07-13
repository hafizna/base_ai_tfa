"""Canonical incident timeline builder — Stage 2.

Builds ``IncidentTimelineEvent`` rows purely from Stage 0 canonical data
already captured in each ``IncidentRecord.canonical_snapshot`` (the
``EventWindow``, observed facts, and reclose events built in
``core.event_analysis`` / ``webapp.api.record_analysis``). This module does
not run any new fault/inception detector — it only translates canonical
facts already computed into incident-relative timeline events.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from .models import AlignmentAssessment, IncidentRecord, IncidentTimelineEvent


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _incident_anchor(records: list[IncidentRecord], alignment: AlignmentAssessment) -> Optional[datetime]:
    """The zero point for ``relative_incident_ms``. Only meaningful when at
    least one record has a trusted absolute timestamp."""
    if alignment.status in ("UNTRUSTED", "INSUFFICIENT_DATA"):
        return None
    times = [t for t in (_parse_iso(r.trigger_time_iso or r.record_start_iso) for r in records) if t is not None]
    if not times:
        return None
    return min(times)


def _record_absolute_start(record: IncidentRecord) -> Optional[datetime]:
    return _parse_iso(record.trigger_time_iso) or _parse_iso(record.record_start_iso)


def _new_event(
    new_id_fn,
    incident_id: str,
    *,
    incident_record_id: Optional[str],
    event_type: str,
    absolute_time: Optional[datetime],
    anchor: Optional[datetime],
    record_ms: Optional[float],
    source: str,
    label: str,
    details: dict[str, Any],
    confidence: float,
    provenance: dict[str, Any],
) -> IncidentTimelineEvent:
    relative_incident_ms = None
    if absolute_time is not None and anchor is not None:
        relative_incident_ms = round((absolute_time - anchor).total_seconds() * 1000.0, 1)

    return IncidentTimelineEvent(
        timeline_event_id=new_id_fn(),
        incident_id=incident_id,
        incident_record_id=incident_record_id,
        event_type=event_type,
        absolute_time_iso=absolute_time.isoformat() if absolute_time is not None else None,
        relative_incident_ms=relative_incident_ms,
        relative_record_ms=record_ms,
        source=source,
        label=label,
        details=details,
        confidence=confidence,
        provenance=provenance,
    )


def build_timeline(
    incident_id: str,
    records: list[IncidentRecord],
    alignment: AlignmentAssessment,
    new_id_fn,
) -> list[IncidentTimelineEvent]:
    """Build the canonical timeline for an incident.

    ``new_id_fn`` is injected (rather than importing incident_storage.new_id
    directly) so this module stays a pure function of its inputs and is easy
    to unit test deterministically.
    """
    anchor = _incident_anchor(records, alignment)
    events: list[IncidentTimelineEvent] = []

    order_index = {rid: i for i, rid in enumerate(alignment.record_order)}
    ordered_records = sorted(records, key=lambda r: order_index.get(r.incident_record_id, r.sequence_index))

    for record in ordered_records:
        snapshot = record.canonical_snapshot or {}
        window = snapshot.get("event_window") or {}
        observed = snapshot.get("observed_facts") or {}
        provenance = {
            "analysis_id": record.analysis_id,
            "incident_record_id": record.incident_record_id,
            "canonical_schema_version": (snapshot.get("provenance") or {}).get("schema_version"),
            "timing_source": window.get("method") or window.get("timing_source"),
        }

        abs_start = _record_absolute_start(record)
        record_start_ms = window.get("record_start_ms", 0.0) or 0.0

        def to_abs(ms: Optional[float]) -> Optional[datetime]:
            if ms is None or abs_start is None:
                return None
            return abs_start + timedelta(milliseconds=(ms - (window.get("trigger_time_ms") or 0.0)))

        events.append(_new_event(
            new_id_fn, incident_id,
            incident_record_id=record.incident_record_id,
            event_type="RECORD_START",
            absolute_time=abs_start,
            anchor=anchor,
            record_ms=record_start_ms,
            source="canonical_event_window",
            label=f"Record start ({record.source_filename or record.analysis_id})",
            details={"analysis_id": record.analysis_id},
            confidence=1.0 if abs_start else 0.0,
            provenance=provenance,
        ))

        trigger_ms = window.get("trigger_time_ms")
        if trigger_ms is not None:
            events.append(_new_event(
                new_id_fn, incident_id,
                incident_record_id=record.incident_record_id,
                event_type="RECORD_TRIGGER",
                absolute_time=abs_start,
                anchor=anchor,
                record_ms=trigger_ms,
                source="canonical_event_window",
                label="Recording trigger",
                details={},
                confidence=1.0 if abs_start else 0.3,
                provenance=provenance,
            ))

        inception_ms = window.get("inception_time_ms")
        if inception_ms is not None:
            events.append(_new_event(
                new_id_fn, incident_id,
                incident_record_id=record.incident_record_id,
                event_type="FAULT_INCEPTION",
                absolute_time=to_abs(inception_ms),
                anchor=anchor,
                record_ms=inception_ms,
                source="canonical_event_window",
                label=f"Fault inception ({'+'.join(observed.get('faulted_phases') or window.get('faulted_phases') or []) or 'unknown phase'})",
                details={"faulted_phases": observed.get("faulted_phases") or window.get("faulted_phases") or []},
                confidence=float(window.get("confidence", 0.0) or 0.0),
                provenance=provenance,
            ))

        clearing_ms = window.get("clearing_time_ms")
        if clearing_ms is not None:
            events.append(_new_event(
                new_id_fn, incident_id,
                incident_record_id=record.incident_record_id,
                event_type="FAULT_CLEARING",
                absolute_time=to_abs(clearing_ms),
                anchor=anchor,
                record_ms=clearing_ms,
                source="canonical_event_window",
                label="Fault clearing",
                details={"fault_duration_ms": window.get("fault_duration_ms")},
                confidence=float(window.get("confidence", 0.0) or 0.0),
                provenance=provenance,
            ))

        for idx, reclose in enumerate(observed.get("reclose_events") or window.get("reclose_events") or []):
            if not isinstance(reclose, dict):
                continue
            reclose_time_s = reclose.get("time")
            reclose_ms = float(reclose_time_s) * 1000.0 if reclose_time_s is not None else None
            success = reclose.get("success")
            event_type = "RECLOSE_SUCCESS" if success is True else ("RECLOSE_FAILED" if success is False else "RECLOSE_START")
            events.append(_new_event(
                new_id_fn, incident_id,
                incident_record_id=record.incident_record_id,
                event_type=event_type,
                absolute_time=to_abs(reclose_ms) if reclose_ms is not None else None,
                anchor=anchor,
                record_ms=reclose_ms,
                source="canonical_event_window",
                label=f"Reclose attempt #{idx + 1}" + (" (successful)" if success is True else " (failed)" if success is False else ""),
                details={"success": success},
                confidence=0.7,
                provenance=provenance,
            ))

        for w in snapshot.get("data_quality", {}).get("timing_warnings", []) or []:
            events.append(_new_event(
                new_id_fn, incident_id,
                incident_record_id=record.incident_record_id,
                event_type="CLOCK_WARNING",
                absolute_time=None,
                anchor=anchor,
                record_ms=None,
                source="canonical_data_quality",
                label=str(w),
                details={},
                confidence=0.5,
                provenance=provenance,
            ))

    # Gaps between consecutive records (chronological, not episode-based).
    for gap in alignment.pairwise_gaps_ms:
        if gap.get("gap_ms", 0) <= 0:
            continue
        events.append(_new_event(
            new_id_fn, incident_id,
            incident_record_id=None,
            event_type="DATA_GAP",
            absolute_time=None,
            anchor=anchor,
            record_ms=None,
            source="alignment_assessment",
            label=f"Gap of {gap['gap_ms'] / 1000.0:.1f}s between records",
            details=gap,
            confidence=1.0 if gap.get("precise") else 0.4,
            provenance={"kind": "inter_record_gap"},
        ))

    events.sort(key=lambda e: (e.relative_incident_ms if e.relative_incident_ms is not None else float("inf")))
    return events
