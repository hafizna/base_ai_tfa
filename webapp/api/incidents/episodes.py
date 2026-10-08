"""Fault episode grouping — Stage 2.

Groups attached records into incident-aware ``FaultEpisode`` objects using
the pairwise ``RecordRelationship`` classifications already produced by
``webapp.api.incidents.relationships``. Grouping rules:

  - ``DUPLICATE_TRIGGER`` and ``CONTINUATION`` and ``RECLOSE_SEQUENCE`` merge
    two adjacent records into the SAME episode (they describe one electrical
    event/protection sequence, captured by one or more records).
  - ``OVERLAPPING_CAPTURE`` also merges (same event window, partial overlap).
  - ``REFAULT_AFTER_RECLOSE``, ``NEW_FAULT_EPISODE``, ``REPEATED_FAULT``,
    ``POSSIBLE_EVOLVING_FAULT``, ``UNRELATED``, and ``UNCERTAIN`` start a new
    episode. A re-fault after a reclose is a new fault inception (its own
    cause evidence), even though it belongs to the same incident sequence.

This means duplicate captures are never counted as separate episodes (spec
section 8/10), while a possibly-evolving fault still gets its own episode
(its relationship to the previous episode is recorded, not merged away).
"""

from __future__ import annotations

from typing import Optional

from .models import FaultEpisode, IncidentRecord, RecordRelationship

_MERGE_TYPES = {"DUPLICATE_TRIGGER", "CONTINUATION", "RECLOSE_SEQUENCE", "OVERLAPPING_CAPTURE"}


def _phases(record: IncidentRecord) -> list[str]:
    snapshot = record.canonical_snapshot or {}
    observed = snapshot.get("observed_facts") or {}
    return list(observed.get("faulted_phases") or (snapshot.get("event_window") or {}).get("faulted_phases") or [])


def _fault_type_from_phases(phases: list[str]) -> Optional[str]:
    n = len(set(phases))
    if n == 0:
        return None
    if n >= 3:
        return "3PH"
    if n == 2:
        return "LL_OR_DLG"
    return "SLG"


def _reclose_outcome(record: IncidentRecord) -> Optional[str]:
    snapshot = record.canonical_snapshot or {}
    observed = snapshot.get("observed_facts") or {}
    # Same rule as relationships._reclose_events: an unverified waveform-only
    # "current came back" reading is not evidence of a reclose.
    events = [
        e for e in (observed.get("reclose_events") or [])
        if isinstance(e, dict) and e.get("cb_open_verified", True) is not False
    ]
    if not events:
        return None
    success = events[-1].get("success")
    if success is True:
        return "successful"
    if success is False:
        return "failed"
    return None


def _record_time_iso(record: IncidentRecord) -> Optional[str]:
    return record.trigger_time_iso or record.record_start_iso


def group_episodes(
    incident_id: str,
    records: list[IncidentRecord],
    relationships: list[RecordRelationship],
    record_order: list[str],
    new_id_fn,
    record_cause_lookup: Optional[dict[str, dict]] = None,
) -> list[FaultEpisode]:
    """Group records into episodes.

    ``record_cause_lookup`` maps ``incident_record_id`` -> the per-record
    cause-evidence entry already computed once by
    ``webapp.api.incidents.reconstruction._physical_cause_evidence`` (model
    version, feature version, calibration, timing source, raw/calibrated
    probabilities, applied caps). Stage 0's ``RecordAnalysis.cause_hypotheses``
    is currently always empty (never wired to the LightGBM call), so episode
    cards source cause hypotheses from this shared lookup instead of
    re-running inference a second time per episode.
    """
    by_id = {r.incident_record_id: r for r in records}
    ordered = [by_id[rid] for rid in record_order if rid in by_id]
    # Any record missing from record_order (shouldn't happen) is appended so
    # nothing is silently dropped from episode membership.
    missing = [r for r in records if r.incident_record_id not in record_order]
    ordered.extend(missing)

    rel_by_pair = {(r.left_record_id, r.right_record_id): r for r in relationships}

    groups: list[list[IncidentRecord]] = []
    relationship_to_previous_group: list[Optional[str]] = []
    # The relationship that opened each group, and the merge relationships
    # inside it — read below for dead time / refault timing / outcome fixes.
    boundary_relationships: list[Optional[RecordRelationship]] = []
    internal_relationships: list[list[RecordRelationship]] = []
    current_group: list[IncidentRecord] = []
    current_internal: list[RecordRelationship] = []

    for i, record in enumerate(ordered):
        if not current_group:
            current_group = [record]
            relationship_to_previous_group.append(None)
            boundary_relationships.append(None)
            continue

        prev_record = ordered[i - 1]
        rel = rel_by_pair.get((prev_record.incident_record_id, record.incident_record_id))
        rel_type = rel.relationship_type if rel else "UNCERTAIN"

        if rel_type in _MERGE_TYPES:
            current_group.append(record)
            if rel is not None:
                current_internal.append(rel)
        else:
            groups.append(current_group)
            internal_relationships.append(current_internal)
            current_group = [record]
            current_internal = []
            relationship_to_previous_group.append(rel_type)
            boundary_relationships.append(rel)

    if current_group:
        groups.append(current_group)
        internal_relationships.append(current_internal)

    episodes: list[FaultEpisode] = []
    for idx, group in enumerate(groups):
        member_ids = [r.incident_record_id for r in group]
        times = sorted(t for t in (_record_time_iso(r) for r in group) if t)
        start_iso = times[0] if times else None
        end_iso = times[-1] if times else None

        durations = [
            (r.canonical_snapshot or {}).get("event_window", {}).get("fault_duration_ms")
            for r in group
        ]
        durations = [d for d in durations if d is not None]
        duration_ms = max(durations) if durations else None

        all_phases: list[str] = []
        for r in group:
            for p in _phases(r):
                if p not in all_phases:
                    all_phases.append(p)

        reclose_outcomes = [o for o in (_reclose_outcome(r) for r in group) if o is not None]
        reclose_outcome = reclose_outcomes[-1] if reclose_outcomes else None
        internal_metrics = [rel.metrics for rel in internal_relationships[idx] if isinstance(rel.metrics, dict)]
        if any(m.get("reclose_outcome_correction") == "failed" for m in internal_metrics):
            # The next record shows the fault already present as the breaker
            # closed: whatever that reclose looked like on its own, it failed.
            reclose_outcome = "failed"
        dead_times = [m["dead_time_s"] for m in internal_metrics if m.get("dead_time_s") is not None]
        sequence_facts: dict = {}
        if dead_times:
            sequence_facts["reclose_dead_time_s"] = dead_times[-1]
        opened_by = boundary_relationships[idx]
        if opened_by is not None and opened_by.relationship_type == "REFAULT_AFTER_RECLOSE":
            sequence_facts["seconds_after_previous_reclose"] = (opened_by.metrics or {}).get("seconds_after_reclose")
        closed_by = boundary_relationships[idx + 1] if idx + 1 < len(boundary_relationships) else None
        if closed_by is not None and closed_by.relationship_type == "REFAULT_AFTER_RECLOSE":
            # This episode's reclose succeeded but did not hold.
            sequence_facts["refault_after_reclose_s"] = (closed_by.metrics or {}).get("seconds_after_reclose")

        local_cause_hypotheses = []
        for r in group:
            cause_entry = (record_cause_lookup or {}).get(r.incident_record_id)
            if cause_entry:
                local_cause_hypotheses.append({
                    "analysis_id": r.analysis_id,
                    "top_hypothesis": cause_entry.get("top_hypothesis"),
                    "confidence": cause_entry.get("confidence"),
                    "cause_ranking": cause_entry.get("cause_ranking") or [],
                    "model_version": cause_entry.get("model_version"),
                    "timing_source": cause_entry.get("timing_source"),
                    "scope": "RECORD_LOCAL_SIGNATURE",
                    # "inception" (independently faulted waveform, real cause
                    # evidence) vs "aftermath" (this record only captures a
                    # reclose/continuation/duplicate of a preceding record's
                    # event — its own classifier reading is preserved here
                    # but must not be read as a second, disagreeing cause).
                    # See reconstruction.py::_evidence_roles.
                    "evidence_role": cause_entry.get("evidence_role", "inception"),
                    "fault_type": cause_entry.get("fault_type"),
                    "requires_review": cause_entry.get("requires_review", False),
                    "skip_reason": cause_entry.get("skip_reason"),
                })
            else:
                hyps = (r.canonical_snapshot or {}).get("cause_hypotheses") or []
                top = hyps[0] if hyps else None
                local_cause_hypotheses.append({
                    "analysis_id": r.analysis_id,
                    "top_hypothesis": top.get("cause") if isinstance(top, dict) else None,
                    "confidence": top.get("confidence") if isinstance(top, dict) else None,
                    "scope": "RECORD_LOCAL_SIGNATURE",
                })

        event_classes = {(r.canonical_snapshot or {}).get("protection_interpretation", {}).get("event_class") for r in group}
        missing_evidence = []
        if len(group) > 1:
            sequential = any(rel.relationship_type == "RECLOSE_SEQUENCE" for rel in internal_relationships[idx])
            missing_evidence.append({
                "type": "MULTIPLE_RECORDS_ONE_EPISODE",
                "description": (
                    f"This episode spans {len(group)} records ({', '.join(member_ids)}): the fault and its "
                    "trip/reclose sequence captured in separate files. Only the record holding the fault "
                    "inception is cause evidence."
                    if sequential else
                    f"This episode is backed by {len(group)} records ({', '.join(member_ids)}); treat as one "
                    "electrical event captured redundantly, not independent evidence."
                ),
            })
        if not any(_record_time_iso(r) for r in group):
            missing_evidence.append({"type": "NO_ABSOLUTE_TIME", "description": "No member record has an absolute timestamp for this episode."})
        # Surface Stage 0's NO_PROTECTION_OPERATION flag (see record_analysis.py)
        # at episode level: a fault was seen on the waveform but no trip/reclose
        # element ever asserted in ANY member record. Don't let this silently
        # disappear into per-record missing_evidence that only the single-record
        # view would show.
        no_protection_records = [
            r.incident_record_id for r in group
            if any(
                (item.get("type") == "NO_PROTECTION_OPERATION")
                for item in ((r.canonical_snapshot or {}).get("missing_evidence") or [])
            )
        ]
        if no_protection_records:
            missing_evidence.append({
                "type": "NO_PROTECTION_OPERATION",
                "description": (
                    "A fault was detected from the analog waveform in "
                    f"{'this record' if len(no_protection_records) == 1 else f'{len(no_protection_records)} member records'} "
                    "but no trip/reclose element ever asserted - the cause hypothesis below is derived from raw "
                    "waveform signature alone, not confirmed by any protection operation."
                ),
                "requires_review": True,
            })

        confidence = min((r.canonical_snapshot or {}).get("event_window", {}).get("confidence", 0.0) or 0.0 for r in group) if group else 0.0

        episode = FaultEpisode(
            episode_id=new_id_fn(),
            incident_id=incident_id,
            member_record_ids=member_ids,
            episode_index=idx,
            start_iso=start_iso,
            end_iso=end_iso,
            duration_ms=duration_ms,
            faulted_phases=all_phases,
            fault_type=_fault_type_from_phases(all_phases),
            zone_operations=[],
            trip_types=[],
            reclose_outcome=reclose_outcome,
            electrical_summary={},
            local_cause_hypotheses=local_cause_hypotheses,
            relationship_to_previous=relationship_to_previous_group[idx] if idx < len(relationship_to_previous_group) else None,
            confidence=float(confidence),
            observed_facts={
                "member_record_ids": member_ids,
                "faulted_phases": all_phases,
                "reclose_outcome": reclose_outcome,
                **sequence_facts,
            },
            interpretation={
                "event_classes": sorted(c for c in event_classes if c),
            },
            missing_evidence=missing_evidence,
            provenance={
                "member_analysis_ids": [r.analysis_id for r in group],
            },
        )
        episodes.append(episode)

    return episodes
