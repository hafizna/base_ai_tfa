"""Clock and alignment assessment — Stage 2.

Decides how much to trust absolute-time ordering across an incident's
records. Never performs speculative clock correction: if precise millisecond
alignment can't be trusted, the assessment degrades to ``ORDER_ONLY`` (or
worse) and pairwise gaps are reported with an explicit precision caveat
rather than a false-confidence number.

Input is the list of ``IncidentRecord`` already attached to an incident
(Stage 1); timing values come straight from each record's Stage 0 canonical
snapshot (``source_metadata`` / ``event_window``) — this module does not
recompute or re-detect timing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .models import AlignmentAssessment
from .models import IncidentRecord


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _record_timestamp(record: IncidentRecord) -> Optional[datetime]:
    """Prefer the fault-relevant trigger time; fall back to record start."""
    return _parse_iso(record.trigger_time_iso) or _parse_iso(record.record_start_iso)


def _record_duration_s(record: IncidentRecord) -> Optional[float]:
    snapshot = record.canonical_snapshot or {}
    window = snapshot.get("event_window") or {}
    total_samples = (snapshot.get("source_metadata") or {}).get("total_samples")
    frequency = (snapshot.get("source_metadata") or {}).get("frequency")
    # Best-effort: prefer explicit record_start->clearing span if available,
    # otherwise fall back to sample_count / sampling assumptions being absent
    # entirely rather than guessed — duration is only used for overlap
    # detection, and an unknown duration simply skips overlap evidence for
    # that record instead of fabricating one.
    clearing_ms = window.get("clearing_time_ms")
    if clearing_ms is not None:
        return float(clearing_ms) / 1000.0
    return None


def _has_timezone_info(record: IncidentRecord) -> bool:
    meta = (record.canonical_snapshot or {}).get("source_metadata") or {}
    return bool(meta.get("time_code") or meta.get("local_code"))


def _clock_quality(record: IncidentRecord) -> Optional[str]:
    meta = (record.canonical_snapshot or {}).get("source_metadata") or {}
    return meta.get("clock_quality")


def assess_alignment(records: list[IncidentRecord]) -> AlignmentAssessment:
    warnings: list[dict] = []
    assumptions: list[str] = []

    if not records:
        return AlignmentAssessment(status="INSUFFICIENT_DATA", confidence=0.0, order_source="UNKNOWN")

    if len(records) == 1:
        return AlignmentAssessment(
            status="ALIGNED",
            confidence=1.0,
            order_source="ABSOLUTE_TIME" if _record_timestamp(records[0]) else "UPLOAD_ORDER",
            record_order=[records[0].incident_record_id],
        )

    manually_ordered = [r for r in records if r.manual_order is not None]
    if len(manually_ordered) == len(records):
        timed = [r for r in manually_ordered if _record_timestamp(r) is not None]
        if len(timed) >= 2:
            manual_seq = sorted(timed, key=lambda r: r.manual_order)
            time_seq = sorted(timed, key=lambda r: _record_timestamp(r))
            if [r.incident_record_id for r in manual_seq] != [r.incident_record_id for r in time_seq]:
                return AlignmentAssessment(
                    status="UNTRUSTED",
                    confidence=0.15,
                    order_source="MANUAL",
                    record_order=[r.incident_record_id for r in manual_seq],
                    warnings=[{
                        "type": "MANUAL_ORDER_CONTRADICTS_TIMESTAMPS",
                        "description": "Manual record order disagrees with absolute-timestamp order for records that have both.",
                        "requires_review": True,
                    }],
                    assumptions=["Manual order was kept as-is despite contradicting timestamps; review required before trusting sequence."],
                )

        order = sorted(records, key=lambda r: r.manual_order)
        return AlignmentAssessment(
            status="MANUAL_ORDER",
            confidence=0.5,
            order_source="MANUAL",
            record_order=[r.incident_record_id for r in order],
            warnings=[{
                "type": "MANUAL_ORDER_NOT_VERIFIED_BY_CLOCK",
                "description": "Record order was set manually; it has not been cross-checked against absolute timestamps.",
            }],
            assumptions=["Manual order is taken as given; no clock-based verification was possible or was skipped by using manual order as the source of truth."],
        )

    timestamps = {r.incident_record_id: _record_timestamp(r) for r in records}
    with_time = [r for r in records if timestamps[r.incident_record_id] is not None]
    without_time = [r for r in records if timestamps[r.incident_record_id] is None]

    if not with_time:
        # No absolute time anywhere: fall back to manual order if partially
        # set, else upload/sequence order — but never claim ALIGNED.
        if manually_ordered:
            order = sorted(records, key=lambda r: (r.manual_order is None, r.manual_order if r.manual_order is not None else r.sequence_index))
            order_source = "MANUAL"
        else:
            order = sorted(records, key=lambda r: r.sequence_index)
            order_source = "UPLOAD_ORDER"
        return AlignmentAssessment(
            status="ORDER_ONLY",
            confidence=0.2,
            order_source=order_source,
            record_order=[r.incident_record_id for r in order],
            warnings=[{
                "type": "NO_ABSOLUTE_TIME_AVAILABLE",
                "description": "No record in this incident carries an absolute (wall-clock) timestamp. Ordering reflects upload/manual order only.",
            }],
            assumptions=["Record order does not reflect verified chronology; treat sequence as provisional."],
        )

    if without_time:
        warnings.append({
            "type": "PARTIAL_ABSOLUTE_TIME",
            "description": f"{len(without_time)} of {len(records)} records lack an absolute timestamp; their position is inferred from upload/manual order only.",
        })

    # Duplicate timestamps: cannot distinguish order from clock alone.
    time_values = [timestamps[r.incident_record_id] for r in with_time]
    duplicate_times = len(set(time_values)) < len(time_values)
    if duplicate_times:
        warnings.append({
            "type": "DUPLICATE_TIMESTAMPS",
            "description": "Two or more records report the identical absolute timestamp — cannot be used alone to establish order.",
        })

    # Impossible order: manual order (if partially set) directly contradicts
    # timestamp order among the records that have both.
    manual_conflict = False
    if manually_ordered:
        commonly_timed_and_ordered = [r for r in manually_ordered if timestamps[r.incident_record_id] is not None]
        if len(commonly_timed_and_ordered) >= 2:
            manual_seq = sorted(commonly_timed_and_ordered, key=lambda r: r.manual_order)
            time_seq = sorted(commonly_timed_and_ordered, key=lambda r: timestamps[r.incident_record_id])
            if [r.incident_record_id for r in manual_seq] != [r.incident_record_id for r in time_seq]:
                manual_conflict = True
                warnings.append({
                    "type": "MANUAL_ORDER_CONTRADICTS_TIMESTAMPS",
                    "description": "Manual record order disagrees with absolute-timestamp order for records that have both.",
                    "requires_review": True,
                })

    clock_qualities = {_clock_quality(r) for r in with_time if _clock_quality(r)}
    tz_aware = all(_has_timezone_info(r) for r in with_time)
    poor_clock_quality = any(q and q.upper() not in ("GOOD", "LOCKED", "SYNC", "VALID") for q in clock_qualities)

    # datetime comparison requires consistent tz-awareness; normalize by
    # falling back to string comparison only when mixing occurs.
    try:
        order = sorted(records, key=lambda r: (timestamps[r.incident_record_id] is None, timestamps[r.incident_record_id]))
    except TypeError:
        order = sorted(records, key=lambda r: (timestamps[r.incident_record_id] is None, str(timestamps[r.incident_record_id])))
        warnings.append({
            "type": "MIXED_TIMEZONE_AWARENESS",
            "description": "Some timestamps are timezone-aware and others are not; comparison used string ordering as a fallback.",
        })

    record_order = [r.incident_record_id for r in order]

    # Compute pairwise gaps only between consecutive, both-timed records.
    pairwise_gaps_ms: list[dict] = []
    overlap_groups: list[list[str]] = []
    ordered_with_time = [r for r in order if timestamps[r.incident_record_id] is not None]
    for i in range(len(ordered_with_time) - 1):
        left, right = ordered_with_time[i], ordered_with_time[i + 1]
        t_left, t_right = timestamps[left.incident_record_id], timestamps[right.incident_record_id]
        try:
            gap_s = (t_right - t_left).total_seconds()
        except TypeError:
            continue
        entry = {
            "left_incident_record_id": left.incident_record_id,
            "right_incident_record_id": right.incident_record_id,
            "gap_ms": round(gap_s * 1000.0, 1),
            "precise": tz_aware and not poor_clock_quality and not duplicate_times,
        }
        pairwise_gaps_ms.append(entry)

        left_duration_s = _record_duration_s(left)
        if left_duration_s is not None and gap_s < left_duration_s:
            overlap_groups.append([left.incident_record_id, right.incident_record_id])
        elif gap_s < 0:
            overlap_groups.append([left.incident_record_id, right.incident_record_id])

    if overlap_groups:
        warnings.append({
            "type": "OVERLAPPING_TIME_RANGES",
            "description": "One or more record pairs have overlapping absolute time ranges.",
        })

    # Decide status.
    if manual_conflict:
        status = "UNTRUSTED"
        confidence = 0.15
        assumptions.append("Manual order was kept as-is despite contradicting timestamps; review required before trusting sequence.")
    elif duplicate_times or without_time:
        status = "ORDER_ONLY"
        confidence = 0.4
    elif not tz_aware:
        status = "LIKELY_ALIGNED"
        confidence = 0.6
        assumptions.append("Timezone metadata is missing on at least one record; absolute-time comparison assumes all records share the same clock reference.")
    elif poor_clock_quality:
        status = "LIKELY_ALIGNED"
        confidence = 0.55
        assumptions.append("At least one record reports non-nominal clock quality; millisecond-level gaps are indicative, not exact.")
    else:
        status = "ALIGNED"
        confidence = 0.9

    return AlignmentAssessment(
        status=status,
        confidence=confidence,
        order_source="ABSOLUTE_TIME",
        record_order=record_order,
        pairwise_gaps_ms=pairwise_gaps_ms,
        overlap_groups=overlap_groups,
        warnings=warnings,
        assumptions=assumptions,
    )
