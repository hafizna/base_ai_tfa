"""Fault episode grouping — Stage 2.

Groups attached records into incident-aware ``FaultEpisode`` objects using
the pairwise ``RecordRelationship`` classifications already produced by
``webapp.api.incidents.relationships``. Each record follows the relationship
that ties it to an earlier record (another recorder's capture of the same
event first, then its own recorder's previous record). Grouping rules:

  - ``DUPLICATE_TRIGGER`` and ``CONTINUATION`` and ``RECLOSE_SEQUENCE`` merge
    the record into the SAME episode as that earlier record (they describe
    one electrical event/protection sequence, captured by one or more records).
  - ``OVERLAPPING_CAPTURE`` also merges (same event window, partial overlap),
    and so does ``REMOTE_END_CAPTURE`` (the other line end's recording).
  - ``REFAULT_AFTER_RECLOSE``, ``NEW_FAULT_EPISODE``, ``REPEATED_FAULT``,
    ``POSSIBLE_EVOLVING_FAULT``, ``UNRELATED``, and ``UNCERTAIN`` start a new
    episode. A re-fault after a reclose is a new fault inception (its own
    cause evidence), even though it belongs to the same incident sequence.

This means duplicate captures are never counted as separate episodes (spec
section 8/10), while a possibly-evolving fault still gets its own episode
(its relationship to the previous episode is recorded, not merged away).

An episode's facts (phases, duration, reclose outcome, dead time) are read
from one end — the reference recorder's records — and what other recorders
saw of the same episode is summarised separately in
``observed_facts["other_recorders"]``: a weak-infeed far end reading other
phases, or its own breaker's dead time, must not overwrite this end's.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import FaultEpisode, IncidentRecord, RecordRelationship
from .time_axis import TimeAxis, build_time_axis, fault_phases, same_station, station_of

_MERGE_TYPES = {"DUPLICATE_TRIGGER", "CONTINUATION", "RECLOSE_SEQUENCE", "OVERLAPPING_CAPTURE", "REMOTE_END_CAPTURE"}
# Which incoming relationship decides a record's episode when it has several
# (see relationships.build_relationships): a tie to another recorder's
# capture of the same event first, then its own recorder's sequence.
_LINK_PRIORITY = {"other_recorder": 0, "same_recorder": 1, "nearest_record": 2}


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


def _other_recorders(
    main: list[IncidentRecord],
    others: list[IncidentRecord],
    relationships: list[RecordRelationship],
    axis: TimeAxis,
) -> list[dict[str, Any]]:
    """What each other recorder in the episode saw of it, read on its own
    terms: its fault phases and clearing time, its breaker's reclose."""
    main_station = station_of(main[0]) if main else ""
    lanes: dict[str, list[IncidentRecord]] = {}
    for record in others:
        lanes.setdefault(axis.lane(record), []).append(record)

    summaries = []
    for lane, members in lanes.items():
        ids = {r.incident_record_id for r in members}
        ties = [rel for rel in relationships if rel.right_record_id in ids and (rel.metrics or {}).get("link") == "other_recorder"]
        own = [rel for rel in relationships if rel.left_record_id in ids and rel.right_record_id in ids]
        fault = next((r for r in members if fault_phases(r) and _has_fault(r)), None)
        trace = ((fault.canonical_snapshot or {}).get("analog_trace") or {}).get("summary") or {} if fault else {}
        outcomes = [o for o in (_reclose_outcome(r) for r in members) if o is not None]
        dead_times = [rel.metrics["dead_time_s"] for rel in own if (rel.metrics or {}).get("dead_time_s") is not None]
        tie = next((rel for rel in ties if fault is not None and rel.right_record_id == fault.incident_record_id), None)
        placement = axis.placement(members[0])
        summaries.append({
            "recorder": lane,
            "station": station_of(members[0]),
            "same_station": same_station(station_of(members[0]), main_station) or not (station_of(members[0]) and main_station),
            "member_record_ids": [r.incident_record_id for r in members],
            "relationship_types": sorted({rel.relationship_type for rel in ties}),
            "fault_record_id": fault.incident_record_id if fault else None,
            "faulted_phases": sorted(fault_phases(fault)) if fault else [],
            "fct_ms": trace.get("fct_ms"),
            "fault_start_difference_ms": (tie.metrics or {}).get("fault_start_difference_ms") if tie else None,
            "reclose_outcome": outcomes[-1] if outcomes else None,
            "reclose_dead_time_s": dead_times[-1] if dead_times else None,
            "clock": {
                "method": placement.method if placement else None,
                "zone_offset_h": placement.zone_offset_h if placement else None,
                "clock_offset_ms": placement.clock_offset_ms if placement else None,
            },
        })
    return summaries


def _has_fault(record: IncidentRecord) -> bool:
    snapshot = record.canonical_snapshot or {}
    if (snapshot.get("protection_interpretation") or {}).get("event_class") in ("NO_FAULT_TRIGGER", "RECLOSE_CAPTURE"):
        return False
    summary = (snapshot.get("analog_trace") or {}).get("summary") or {}
    return summary.get("fault_start_ms") is not None or (snapshot.get("event_window") or {}).get("inception_time_ms") is not None


def group_episodes(
    incident_id: str,
    records: list[IncidentRecord],
    relationships: list[RecordRelationship],
    record_order: list[str],
    new_id_fn,
    record_cause_lookup: Optional[dict[str, dict]] = None,
    axis: Optional[TimeAxis] = None,
) -> list[FaultEpisode]:
    """Group records into episodes. An episode's ``start_iso``/``end_iso`` are
    its first and last member trigger on the incident time axis ``axis``.

    ``record_cause_lookup`` maps ``incident_record_id`` -> the per-record
    cause-evidence entry already computed once by
    ``webapp.api.incidents.reconstruction._physical_cause_evidence`` (model
    version, feature version, calibration, timing source, raw/calibrated
    probabilities, applied caps). Stage 0's ``RecordAnalysis.cause_hypotheses``
    is currently always empty (never wired to the LightGBM call), so episode
    cards source cause hypotheses from this shared lookup instead of
    re-running inference a second time per episode.
    """
    axis = axis or build_time_axis(records)

    def _record_time_iso(record: IncidentRecord) -> Optional[str]:
        trigger = axis.trigger(record)
        return trigger.isoformat() if trigger is not None else None

    by_id = {r.incident_record_id: r for r in records}
    ordered = [by_id[rid] for rid in record_order if rid in by_id]
    # Any record missing from record_order (shouldn't happen) is appended so
    # nothing is silently dropped from episode membership.
    missing = [r for r in records if r.incident_record_id not in record_order]
    ordered.extend(missing)
    primary = axis.reference_group or (axis.lane(ordered[0]) if ordered else None)

    incoming: dict[str, list[RecordRelationship]] = {}
    for rel in relationships:
        incoming.setdefault(rel.right_record_id, []).append(rel)

    def tie(record: IncidentRecord) -> Optional[RecordRelationship]:
        # A record tied to another recorder's capture of the same event joins
        # that event; otherwise its own recorder's sequence decides.
        rels = incoming.get(record.incident_record_id) or []
        return min(rels, key=lambda r: _LINK_PRIORITY.get((r.metrics or {}).get("link"), 1)) if rels else None

    # Each group: its records, and the relationship that opened it (read
    # below for the relationship to the previous episode and refault timing).
    found: list[dict[str, Any]] = []
    found_of: dict[str, dict[str, Any]] = {}
    pending = list(ordered)
    while pending:
        waiting = []
        for record in pending:
            rel = tie(record)
            if rel is not None and rel.relationship_type in _MERGE_TYPES and rel.left_record_id in by_id:
                target = found_of.get(rel.left_record_id)
                if target is None:
                    # The far end can start recording before this end does:
                    # wait for the record it is tied to.
                    waiting.append(record)
                    continue
                target["members"].append(record)
                found_of[record.incident_record_id] = target
                continue
            group = {"members": [record], "opened_by": rel}
            found.append(group)
            found_of[record.incident_record_id] = group
        if len(waiting) == len(pending):  # nothing resolved: never loop forever
            for record in waiting:
                group = {"members": [record], "opened_by": tie(record)}
                found.append(group)
                found_of[record.incident_record_id] = group
            break
        pending = waiting

    def lane_of(record: IncidentRecord) -> str:
        return axis.lane(record)

    order_index = {r.incident_record_id: i for i, r in enumerate(ordered)}

    def first_index(group: dict[str, Any]) -> int:
        members = group["members"]
        main = [r for r in members if lane_of(r) == primary] or members
        return min(order_index[r.incident_record_id] for r in main)

    found.sort(key=first_index)
    groups = [sorted(g["members"], key=lambda r: order_index[r.incident_record_id]) for g in found]
    boundary_relationships: list[Optional[RecordRelationship]] = [g["opened_by"] for g in found]
    group_of = {r.incident_record_id: idx for idx, members in enumerate(groups) for r in members}

    episodes: list[FaultEpisode] = []
    for idx, group in enumerate(groups):
        # The episode is read from one end: the reference recorder's records
        # when it has any, else the recorder of its first record. Records from
        # other recorders (the far end, a second device in the bay) are listed
        # after them and summarised separately — their phases, breaker and
        # timing belong to their own end.
        main_lane = primary if any(lane_of(r) == primary for r in group) else lane_of(group[0])
        main = [r for r in group if lane_of(r) == main_lane]
        others = [r for r in group if lane_of(r) != main_lane]
        group = main + others
        member_ids = [r.incident_record_id for r in group]
        main_ids = {r.incident_record_id for r in main}
        inside = [
            rel for rel in relationships
            if rel.left_record_id in main_ids and rel.right_record_id in main_ids
        ]

        triggers = [t for t in (axis.trigger(r) for r in main) if t is not None]
        try:
            triggers.sort()
        except TypeError:  # one timestamp timezone-aware, the other naive
            triggers.sort(key=str)
        start_iso = triggers[0].isoformat() if triggers else None
        end_iso = triggers[-1].isoformat() if triggers else None

        durations = [
            (r.canonical_snapshot or {}).get("event_window", {}).get("fault_duration_ms")
            for r in main
        ]
        durations = [d for d in durations if d is not None]
        duration_ms = max(durations) if durations else None

        all_phases: list[str] = []
        for r in main:
            for p in _phases(r):
                if p not in all_phases:
                    all_phases.append(p)

        reclose_outcomes = [o for o in (_reclose_outcome(r) for r in main) if o is not None]
        reclose_outcome = reclose_outcomes[-1] if reclose_outcomes else None
        internal_metrics = [rel.metrics for rel in inside if isinstance(rel.metrics, dict)]
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
        closed_by = next(
            (
                rel for rel in relationships
                if rel.relationship_type == "REFAULT_AFTER_RECLOSE"
                and rel.left_record_id in main_ids
                and group_of.get(rel.right_record_id, idx) > idx
            ),
            None,
        )
        if closed_by is not None:
            # This episode's reclose succeeded but did not hold.
            sequence_facts["refault_after_reclose_s"] = (closed_by.metrics or {}).get("seconds_after_reclose")
        other_recorders = _other_recorders(main, others, relationships, axis)
        if other_recorders:
            sequence_facts["other_recorders"] = other_recorders

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

        event_classes = {(r.canonical_snapshot or {}).get("protection_interpretation", {}).get("event_class") for r in main}
        missing_evidence = []
        if len(group) > 1:
            sequential = any(rel.relationship_type == "RECLOSE_SEQUENCE" for rel in inside)
            other_end = any(not o["same_station"] for o in other_recorders)
            if sequential and other_end:
                spans = "the fault and its trip/reclose sequence captured in separate files, and the other line end's recordings of them"
            elif sequential:
                spans = "the fault and its trip/reclose sequence captured in separate files"
            else:
                spans = "the fault as recorded at both line ends"
            missing_evidence.append({
                "type": "MULTIPLE_RECORDS_ONE_EPISODE",
                "description": (
                    f"This episode spans {len(group)} records ({', '.join(member_ids)}): {spans}. Only the record "
                    "holding the fault inception is cause evidence."
                    if sequential or other_end else
                    f"This episode is backed by {len(group)} records ({', '.join(member_ids)}); treat as one "
                    "electrical event captured redundantly, not independent evidence."
                ),
            })
        if not any(_record_time_iso(r) for r in group):
            missing_evidence.append({"type": "NO_ABSOLUTE_TIME", "description": "No member record has an absolute timestamp for this episode."})
        disagreeing = [
            o for o in other_recorders
            if not o["same_station"] and o["faulted_phases"] and all_phases and set(o["faulted_phases"]) != set(all_phases)
        ]
        if disagreeing:
            # One fault has one set of phases: two ends reading different ones
            # means one end's reading is off (often the weak-infeed end).
            missing_evidence.append({
                "type": "ENDS_DISAGREE_ON_FAULTED_PHASES",
                "description": (
                    f"This end reads the fault on {'-'.join(sorted(all_phases))}, "
                    + "; ".join(f"{o['station']} on {'-'.join(o['faulted_phases'])}" for o in disagreeing)
                    + ". Review which end's phase reading holds."
                ),
                "requires_review": True,
            })
        # Surface Stage 0's NO_PROTECTION_OPERATION flag (see record_analysis.py)
        # at episode level: a fault was seen on the waveform but no trip/reclose
        # element ever asserted in ANY member record. Don't let this silently
        # disappear into per-record missing_evidence that only the single-record
        # view would show.
        no_protection_records = [
            r.incident_record_id for r in main
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

        confidence = min((r.canonical_snapshot or {}).get("event_window", {}).get("confidence", 0.0) or 0.0 for r in main) if main else 0.0
        if idx == 0:
            relationship_to_previous = None
        else:
            relationship_to_previous = opened_by.relationship_type if opened_by is not None else "UNCERTAIN"

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
            relationship_to_previous=relationship_to_previous,
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
                "recorder": main_lane,
            },
        )
        episodes.append(episode)

    return episodes
