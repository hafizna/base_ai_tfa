"""Fault episode grouping — Stage 2.

Groups attached records into incident-aware ``FaultEpisode`` objects using
the pairwise ``RecordRelationship`` classifications already produced by
``webapp.api.incidents.relationships``. Grouping rules:

  - ``DUPLICATE_TRIGGER`` and ``CONTINUATION`` and ``RECLOSE_SEQUENCE`` merge
    two adjacent records into the SAME episode (they describe one electrical
    event/protection sequence, captured by one or more records).
  - ``OVERLAPPING_CAPTURE`` also merges (same event window, partial overlap).
  - ``NEW_FAULT_EPISODE``, ``REPEATED_FAULT``, ``POSSIBLE_EVOLVING_FAULT``,
    ``UNRELATED``, and ``UNCERTAIN`` start a new episode.

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
    events = [e for e in (observed.get("reclose_events") or []) if isinstance(e, dict)]
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
    current_group: list[IncidentRecord] = []
    pending_relationship_label: Optional[str] = None

    for i, record in enumerate(ordered):
        if not current_group:
            current_group = [record]
            relationship_to_previous_group.append(None)
            continue

        prev_record = ordered[i - 1]
        rel = rel_by_pair.get((prev_record.incident_record_id, record.incident_record_id))
        rel_type = rel.relationship_type if rel else "UNCERTAIN"

        if rel_type in _MERGE_TYPES:
            current_group.append(record)
        else:
            groups.append(current_group)
            current_group = [record]
            relationship_to_previous_group.append(rel_type)

    if current_group:
        groups.append(current_group)

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
            missing_evidence.append({
                "type": "MULTIPLE_RECORDS_ONE_EPISODE",
                "description": f"This episode is backed by {len(group)} records ({', '.join(member_ids)}); treat as one electrical event captured redundantly, not independent evidence.",
            })
        if not any(_record_time_iso(r) for r in group):
            missing_evidence.append({"type": "NO_ABSOLUTE_TIME", "description": "No member record has an absolute timestamp for this episode."})

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
