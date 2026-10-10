"""Pairwise record relationship engine — Stage 2.

For each relevant pair of attached records, classifies the relationship
between them (duplicate capture, overlap, continuation, reclose sequence,
new/repeated/possibly-evolving fault episode, unrelated, or uncertain) using
only auditable, bounded evidence:

  - overlapping absolute time ranges (from the alignment assessment);
  - station/bay/relay metadata already on the ``IncidentRecord``;
  - canonical observed facts (faulted phases, reclose outcome, duration)
    already computed in Stage 0's ``RecordAnalysis``;
  - bounded waveform similarity metrics computed ONLY over the overlapping
    time window and only for compatible (same canonical phase) channels —
    never a full-record dynamic-time-warping comparison.

Every relationship decision records its evidence-for, evidence-against,
assumptions, and the raw metrics used, so it can be audited or manually
overridden (see ``webapp.api.incidents.service``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from core.line_selection import scope_payload
from ..storage import load_analysis
from .models import AlignmentAssessment, IncidentRecord, RecordRelationship
from .time_axis import TimeAxis, build_time_axis, fault_phases, fault_start_s, same_station, span_s, station_of

# Similarity thresholds. Kept as module-level constants (not tunable per
# request) so relationship classification stays deterministic and auditable
# across reconstruction runs.
DUPLICATE_CORRELATION_THRESHOLD = 0.92
DUPLICATE_RMS_RATIO_THRESHOLD = 0.15   # normalized RMS diff below this -> near-identical
OVERLAP_MIN_SECONDS = 0.0              # any positive overlap counts
CONTINUATION_GAP_MS = 2000.0           # record starts within this long of previous record's clearing/reclose tail
RECLOSE_GAP_MS = 5000.0                # generous window after a reclose attempt
REPEATED_FAULT_MAX_GAP_S = 3600.0      # up to 1 hour still considered "repeated" rather than unrelated
SAME_PHASE_SET_BONUS = 0.15
# A reclose captured in its own file (the DFR re-triggers when the breaker
# closes, so the record starts in dead time) is linked to the fault record
# before it when the dead time between them is plausible. Auto-reclose dead
# times are well under a minute; longer gaps are still the same line being
# re-energized, but read as a likely manual close.
MAX_RECLOSE_DEAD_TIME_S = 3600.0
AUTO_RECLOSE_DEAD_TIME_MAX_S = 60.0
# A new fault this soon after a successful reclose means the reclose did not
# hold — well inside the reclaim window of typical auto-reclose schemes.
REFAULT_AFTER_RECLOSE_MAX_S = 60.0
# A fault starting this close to the reclose instant was already there when
# the breaker closed: a failed reclose (switch-on-to-fault), not a new fault.
RECLOSE_ONTO_FAULT_MAX_S = 0.5
# A record from the other line end belongs with the local record its
# recording overlaps — or, failing that, the one it starts within this long
# of (each end's breaker recloses on its own timer, so the two reclose
# captures need not overlap).
REMOTE_END_WINDOW_S = AUTO_RECLOSE_DEAD_TIME_MAX_S

_NO_INCEPTION_METHODS = {"dead_time_recording", "trigger_fallback", "no_fault_evidence", "insufficient_data"}


def _load_line_payload(analysis_id: str) -> Optional[dict]:
    """Stored payload restricted to the record's disturbed line, so a DFR
    file carrying two lines is compared on the line that actually faulted
    (channels are keyed by canonical name below, which would otherwise let
    one line's IA silently overwrite the other's)."""
    payload = load_analysis(analysis_id)
    return scope_payload(payload) if payload is not None else None


def _phases(record: IncidentRecord) -> set[str]:
    snapshot = record.canonical_snapshot or {}
    observed = snapshot.get("observed_facts") or {}
    return set(observed.get("faulted_phases") or (snapshot.get("event_window") or {}).get("faulted_phases") or [])


def _duration_ms(record: IncidentRecord) -> Optional[float]:
    snapshot = record.canonical_snapshot or {}
    window = snapshot.get("event_window") or {}
    return window.get("fault_duration_ms")


def _clearing_s(record: IncidentRecord) -> Optional[float]:
    snapshot = record.canonical_snapshot or {}
    window = snapshot.get("event_window") or {}
    clearing_ms = window.get("clearing_time_ms")
    return clearing_ms / 1000.0 if clearing_ms is not None else None


def _reclose_events(record: IncidentRecord) -> list[dict]:
    """Reclose events backed by evidence the breaker really was open: status
    channels, a record that starts in dead time, or a waveform reading with a
    verified V-and-I-both-zero window. A waveform-only "current came back"
    reading (``cb_open_verified`` False) is also what a self-clearing fault or
    a remote-end clearing misread produces, so it is not used to link records
    or to report a reclose outcome."""
    snapshot = record.canonical_snapshot or {}
    observed = snapshot.get("observed_facts") or {}
    return [
        e for e in (observed.get("reclose_events") or [])
        if isinstance(e, dict) and e.get("cb_open_verified", True) is not False
    ]


def _reclose_outcome(record: IncidentRecord) -> Optional[bool]:
    events = _reclose_events(record)
    return events[-1].get("success") if events else None


def _window(record: IncidentRecord) -> dict:
    return (record.canonical_snapshot or {}).get("event_window") or {}


def _is_reclose_capture(record: IncidentRecord) -> bool:
    """Record starts during breaker dead time: it holds the reclose of a fault
    recorded earlier, not a fault of its own."""
    return _window(record).get("method") == "dead_time_recording"


def _has_fault_inception(record: IncidentRecord) -> bool:
    window = _window(record)
    return (
        window.get("inception_time_ms") is not None
        and window.get("method") not in _NO_INCEPTION_METHODS
        and not _is_no_fault(record)
    )


def _seconds_between(earlier: Optional[datetime], later: Optional[datetime]) -> Optional[float]:
    if earlier is None or later is None:
        return None
    try:
        return (later - earlier).total_seconds()
    except TypeError:  # one timestamp timezone-aware, the other naive
        return None


def _event_class(record: IncidentRecord) -> Optional[str]:
    snapshot = record.canonical_snapshot or {}
    return (snapshot.get("protection_interpretation") or {}).get("event_class")


def _is_no_fault(record: IncidentRecord) -> bool:
    return _event_class(record) == "NO_FAULT_TRIGGER"


def _same_relay(left: IncidentRecord, right: IncidentRecord) -> Optional[bool]:
    if left.relay_id and right.relay_id:
        return left.relay_id == right.relay_id
    return None


def _waveform_similarity(left: IncidentRecord, right: IncidentRecord, axis: TimeAxis) -> dict[str, Any]:
    """Bounded waveform similarity computed ONLY over the overlapping
    absolute-time window and only for matching canonical phase-current
    channels. Returns an empty/low-confidence result if either record lacks
    absolute time, the ranges don't overlap, or channels can't be paired —
    never raises, never falls back to a full-record comparison.

    Both records are indexed from their first sample, placed on the incident
    time axis — not from the trigger, which sits a different pre-trigger
    length into each record."""
    result: dict[str, Any] = {"computed": False, "reason": None}

    t_left = axis.start(left)
    t_right = axis.start(right)
    if t_left is None or t_right is None:
        result["reason"] = "missing_absolute_time"
        return result

    left_payload = _load_line_payload(left.analysis_id)
    right_payload = _load_line_payload(right.analysis_id)
    if left_payload is None or right_payload is None:
        result["reason"] = "analysis_expired_or_missing"
        return result

    try:
        left_time = np.asarray(left_payload.get("time") or [], dtype=float)
        right_time = np.asarray(right_payload.get("time") or [], dtype=float)
        if len(left_time) < 2 or len(right_time) < 2:
            result["reason"] = "insufficient_samples"
            return result

        left_abs_start = t_left
        right_abs_start = t_right
        left_abs = [left_abs_start + _seconds_delta(s - left_time[0]) for s in (left_time[0], left_time[-1])]
        right_abs = [right_abs_start + _seconds_delta(s - right_time[0]) for s in (right_time[0], right_time[-1])]

        try:
            overlap_start = max(left_abs[0], right_abs[0])
            overlap_end = min(left_abs[1], right_abs[1])
        except TypeError:  # one timestamp timezone-aware, the other naive
            result["reason"] = "missing_absolute_time"
            return result
        if overlap_end <= overlap_start:
            result["reason"] = "no_time_overlap"
            return result

        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        result["overlap_seconds"] = round(overlap_seconds, 3)

        left_channels = {c.get("canonical_name"): c for c in left_payload.get("analog_channels", []) if c.get("measurement") == "current"}
        right_channels = {c.get("canonical_name"): c for c in right_payload.get("analog_channels", []) if c.get("measurement") == "current"}
        common = sorted(set(left_channels) & set(right_channels))
        if not common:
            result["reason"] = "no_compatible_channels"
            return result

        correlations = []
        rms_ratios = []
        for canon in common:
            l_start_idx = _index_for_time(left_time, (overlap_start - left_abs_start).total_seconds())
            l_end_idx = _index_for_time(left_time, (overlap_end - left_abs_start).total_seconds())
            r_start_idx = _index_for_time(right_time, (overlap_start - right_abs_start).total_seconds())
            r_end_idx = _index_for_time(right_time, (overlap_end - right_abs_start).total_seconds())

            l_samples = np.asarray(left_channels[canon].get("samples") or [], dtype=float)[l_start_idx:l_end_idx]
            r_samples = np.asarray(right_channels[canon].get("samples") or [], dtype=float)[r_start_idx:r_end_idx]
            n = min(len(l_samples), len(r_samples))
            if n < 8:
                continue
            l_samples = l_samples[:n]
            r_samples = r_samples[:n]

            if np.std(l_samples) > 1e-9 and np.std(r_samples) > 1e-9:
                corr = float(np.corrcoef(l_samples, r_samples)[0, 1])
                if np.isfinite(corr):
                    correlations.append(corr)

            rms_l = float(np.sqrt(np.mean(l_samples ** 2)))
            rms_r = float(np.sqrt(np.mean(r_samples ** 2)))
            denom = max(rms_l, rms_r, 1e-9)
            rms_ratios.append(abs(rms_l - rms_r) / denom)

        if not correlations and not rms_ratios:
            result["reason"] = "insufficient_overlap_samples"
            return result

        result["computed"] = True
        result["channels_compared"] = common
        result["mean_correlation"] = round(float(np.mean(correlations)), 4) if correlations else None
        result["mean_rms_relative_diff"] = round(float(np.mean(rms_ratios)), 4) if rms_ratios else None
        return result
    except Exception as exc:  # pragma: no cover - defensive; similarity must never crash reconstruction
        result["reason"] = f"error: {exc}"
        return result


def _seconds_delta(seconds: float) -> timedelta:
    return timedelta(seconds=float(seconds))


def _index_for_time(time_arr: np.ndarray, target_offset_s: float) -> int:
    idx = int(np.searchsorted(time_arr - time_arr[0], target_offset_s))
    return max(0, min(idx, len(time_arr)))


def _digital_sequence_similarity(left: IncidentRecord, right: IncidentRecord) -> Optional[float]:
    """Compare the set of asserted digital/status channel names as a coarse,
    cheap proxy for "did the same protection elements operate". Not a
    time-aligned bitwise comparison — that would require the same overlap
    machinery as waveform similarity and Stage 2 keeps this metric coarse."""
    left_payload = _load_line_payload(left.analysis_id)
    right_payload = _load_line_payload(right.analysis_id)
    if left_payload is None or right_payload is None:
        return None

    def asserted_names(payload: dict) -> set[str]:
        names = set()
        for ch in payload.get("status_channels", []):
            samples = ch.get("samples") or []
            if any(samples):
                names.add((ch.get("name") or "").strip().upper())
        return names

    left_names = asserted_names(left_payload)
    right_names = asserted_names(right_payload)
    if not left_names and not right_names:
        return None
    union = left_names | right_names
    if not union:
        return None
    return len(left_names & right_names) / len(union)


def classify_pair(
    left: IncidentRecord,
    right: IncidentRecord,
    alignment: AlignmentAssessment,
    new_id_fn,
    incident_id: str,
    prior_fault: Optional[IncidentRecord] = None,
    axis: Optional[TimeAxis] = None,
) -> RecordRelationship:
    """Classify the relationship between two attached incident records.

    ``left``/``right`` are assumed already in chronological (or best-known)
    order per ``alignment.record_order``. ``prior_fault`` is the latest
    record before ``right`` that contains a fault inception (``left`` itself,
    or an earlier one when ``left`` only captured a reclose) — what a new
    fault's phases are compared against. Times are read on the incident time
    axis ``axis`` (built from the pair alone when not given).
    """
    evidence_for: list[dict] = []
    evidence_against: list[dict] = []
    assumptions: list[str] = []
    warnings: list[dict] = []
    metrics: dict[str, Any] = {}
    axis = axis or build_time_axis([left, right])

    def _abs_at(record: IncidentRecord, t_s: Optional[float]) -> Optional[datetime]:
        return axis.absolute(record, t_s)

    gap_s = _seconds_between(axis.trigger(left), axis.trigger(right))
    # Where the right record starts on the left record's own time axis (s
    # after its first sample), comparable with the left record's event times.
    right_start_on_left_s = _seconds_between(axis.start(left), axis.start(right))
    if gap_s is not None:
        metrics["gap_seconds"] = round(gap_s, 3)
        left_span_s = span_s(left)
        if left_span_s is not None and right_start_on_left_s is not None:
            metrics["data_gap_seconds"] = round(right_start_on_left_s - left_span_s, 3)
    else:
        warnings.append({"type": "NO_ABSOLUTE_TIME", "description": "At least one record lacks absolute time; relationship relies on order and signature only."})
        assumptions.append("Temporal gap is unknown; classification relies on record order and fault-signature evidence only.")

    left_phases = _phases(left)
    right_phases = _phases(right)
    same_phases = bool(left_phases) and left_phases == right_phases
    phases_progressed = bool(left_phases) and bool(right_phases) and left_phases < right_phases

    left_no_fault = _is_no_fault(left)
    right_no_fault = _is_no_fault(right)

    similarity = _waveform_similarity(left, right, axis)
    metrics["waveform_similarity"] = similarity
    digital_sim = _digital_sequence_similarity(left, right)
    if digital_sim is not None:
        metrics["digital_sequence_similarity"] = round(digital_sim, 3)

    overlapping = gap_s is not None and similarity.get("computed") and similarity.get("overlap_seconds", 0) > OVERLAP_MIN_SECONDS

    # --- DUPLICATE_TRIGGER: high waveform overlap correlation + same phases,
    # optionally different relay/device (e.g. distance relay + external DFR).
    if overlapping:
        corr = similarity.get("mean_correlation")
        rms_diff = similarity.get("mean_rms_relative_diff")
        if corr is not None and corr >= DUPLICATE_CORRELATION_THRESHOLD and (rms_diff is None or rms_diff <= DUPLICATE_RMS_RATIO_THRESHOLD):
            evidence_for.append({"type": "HIGH_WAVEFORM_CORRELATION", "value": corr})
            if rms_diff is not None:
                evidence_for.append({"type": "LOW_RMS_DIFFERENCE", "value": rms_diff})
            if same_phases:
                evidence_for.append({"type": "SAME_FAULTED_PHASES", "value": sorted(left_phases)})
            same_relay = _same_relay(left, right)
            if same_relay is False:
                evidence_for.append({"type": "DIFFERENT_RELAY_OR_DFR", "description": "Records come from different relay/device ids, consistent with two devices capturing the same electrical event."})
            return _build(new_id_fn, incident_id, left, right, "DUPLICATE_TRIGGER", 0.85, evidence_for, evidence_against, assumptions, warnings, metrics)

        # --- OVERLAPPING_CAPTURE: overlap exists but not identical enough.
        evidence_for.append({"type": "TIME_RANGE_OVERLAP", "value": similarity.get("overlap_seconds")})
        if corr is not None:
            evidence_against.append({"type": "MODERATE_WAVEFORM_CORRELATION", "value": corr, "description": "Overlap present but correlation below the duplicate threshold."})
        return _build(new_id_fn, incident_id, left, right, "OVERLAPPING_CAPTURE", 0.55, evidence_for, evidence_against, assumptions, warnings, metrics)

    # --- RECLOSE_SEQUENCE, reclose in its own record: the DFR re-triggered
    # when the breaker closed, so the right record starts in dead time and
    # holds only the reclose of the fault the left record tripped for. Checked
    # before the no-fault rule — a line re-energized from the far end can
    # show no current step at this end, yet it is part of the sequence.
    if _is_reclose_capture(right) and _has_fault_inception(left):
        reclose = (_reclose_events(right) or [{}])[-1]
        left_window = _window(left)
        trip_ms = left_window.get("clearing_time_ms")
        if trip_ms is None:
            trip_ms = left_window.get("inception_time_ms")
        dead_time_s = _seconds_between(
            _abs_at(left, trip_ms / 1000.0 if trip_ms is not None else None),
            _abs_at(right, reclose.get("time")),
        )
        capture_evidence = [{
            "type": "RIGHT_RECORD_STARTS_IN_DEAD_TIME",
            "description": "The right record starts with the breaker open and captures its reclose; the fault itself is in the left record.",
        }]
        if reclose.get("success") is not None:
            capture_evidence.append({"type": "RECLOSE_OUTCOME", "value": "successful" if reclose["success"] else "failed"})
        if dead_time_s is None and gap_s is None:
            evidence_for.extend(capture_evidence)
            assumptions.append("No absolute time: linked by record order and by the right record starting with the breaker open.")
            return _build(new_id_fn, incident_id, left, right, "RECLOSE_SEQUENCE", 0.5, evidence_for, evidence_against, assumptions, warnings, metrics)
        if dead_time_s is not None and 0.0 < dead_time_s <= MAX_RECLOSE_DEAD_TIME_S:
            metrics["dead_time_s"] = round(dead_time_s, 3)
            evidence_for.extend(capture_evidence)
            evidence_for.append({"type": "DEAD_TIME_S", "value": round(dead_time_s, 3)})
            confidence = 0.85
            if dead_time_s > AUTO_RECLOSE_DEAD_TIME_MAX_S:
                assumptions.append(
                    f"A {dead_time_s:.0f} s dead time is longer than a typical auto-reclose; this is more likely "
                    "a manual re-energization of the line."
                )
                confidence = 0.6
            return _build(new_id_fn, incident_id, left, right, "RECLOSE_SEQUENCE", confidence, evidence_for, evidence_against, assumptions, warnings, metrics)
        # Implausible dead time (negative / too long): let the rules below decide.

    # --- Fault after a successful reclose (left captured the reclose, either
    # inside its own fault record or as a dead-time record).
    left_reclose = _reclose_events(left)
    if (
        left_reclose
        and left_reclose[-1].get("success") is True
        and _has_fault_inception(right)
        and not _is_reclose_capture(right)
    ):
        since_reclose_s = _seconds_between(
            _abs_at(left, left_reclose[-1].get("time")),
            _abs_at(right, (_window(right).get("inception_time_ms") or 0.0) / 1000.0),
        )
        if since_reclose_s is not None and 0.0 <= since_reclose_s <= RECLOSE_ONTO_FAULT_MAX_S:
            # The fault was still there when the breaker closed: one fault,
            # a failed reclose — not a second fault.
            metrics["seconds_after_reclose"] = round(since_reclose_s, 3)
            metrics["reclose_outcome_correction"] = "failed"
            evidence_for.append({
                "type": "FAULT_PRESENT_AT_RECLOSE", "value": round(since_reclose_s, 3),
                "description": "Fault current appears as the breaker closes: the reclose closed onto the still-present fault.",
            })
            return _build(new_id_fn, incident_id, left, right, "RECLOSE_SEQUENCE", 0.75, evidence_for, evidence_against, assumptions, warnings, metrics)
        if since_reclose_s is not None and RECLOSE_ONTO_FAULT_MAX_S < since_reclose_s <= REFAULT_AFTER_RECLOSE_MAX_S:
            metrics["seconds_after_reclose"] = round(since_reclose_s, 3)
            evidence_for.append({
                "type": "NEW_FAULT_AFTER_SUCCESSFUL_RECLOSE", "value": round(since_reclose_s, 3),
                "description": f"The line faulted again {since_reclose_s:.1f} s after a successful reclose — the reclose did not hold.",
            })
            reference_phases = left_phases or (_phases(prior_fault) if prior_fault is not None else set())
            if reference_phases and right_phases:
                if reference_phases == right_phases:
                    evidence_for.append({"type": "SAME_FAULTED_PHASES_AS_RECLOSED_FAULT", "value": sorted(right_phases)})
                elif reference_phases < right_phases:
                    evidence_for.append({"type": "FAULT_PHASE_PROGRESSED", "from": sorted(reference_phases), "to": sorted(right_phases)})
                else:
                    evidence_against.append({"type": "DIFFERENT_FAULTED_PHASES", "previous": sorted(reference_phases), "current": sorted(right_phases)})
            return _build(new_id_fn, incident_id, left, right, "REFAULT_AFTER_RECLOSE", 0.8, evidence_for, evidence_against, assumptions, warnings, metrics)

    if left_no_fault or right_no_fault:
        evidence_against.append({"type": "NO_FAULT_RECORD_IN_PAIR", "description": "One record has no fault signature (no-fault trigger); no meaningful electrical relationship to classify."})
        return _build(new_id_fn, incident_id, left, right, "UNRELATED", 0.5, evidence_for, evidence_against, assumptions, warnings, metrics)

    if gap_s is None:
        # No timing evidence at all — cannot place in sequence confidently.
        return _build(new_id_fn, incident_id, left, right, "UNCERTAIN", 0.3, evidence_for, evidence_against, assumptions, warnings, metrics)

    if gap_s < 0:
        warnings.append({"type": "NEGATIVE_GAP", "description": "Right record's timestamp precedes the left record's in the assumed order.", "requires_review": True})
        return _build(new_id_fn, incident_id, left, right, "UNCERTAIN", 0.25, evidence_for, evidence_against, assumptions, warnings, metrics)

    gap_ms = gap_s * 1000.0
    left_clearing_s = _clearing_s(left)
    left_reclose_events = _reclose_events(left)
    left_last_reclose_s = left_reclose_events[-1].get("time") if left_reclose_events else None
    left_reclose_outcome = _reclose_outcome(left)

    # --- RECLOSE_SEQUENCE: right record starts shortly after left's reclose
    # attempt and itself shows a reclose-related signature (trip-on-reclose,
    # failed reclose, or refault right after reclose).
    if left_last_reclose_s is not None and gap_ms <= RECLOSE_GAP_MS:
        evidence_for.append({"type": "STARTS_SHORTLY_AFTER_RECLOSE_ATTEMPT", "value": gap_ms})
        if right.canonical_snapshot and _reclose_events(right):
            evidence_for.append({"type": "RIGHT_RECORD_ALSO_SHOWS_RECLOSE_ACTIVITY", "value": True})
        if left_reclose_outcome is False:
            evidence_for.append({"type": "LEFT_RECLOSE_FAILED", "description": "Left record's reclose attempt did not succeed; right record likely captures the resulting trip/lockout."})
        return _build(new_id_fn, incident_id, left, right, "RECLOSE_SEQUENCE", 0.7, evidence_for, evidence_against, assumptions, warnings, metrics)

    # --- CONTINUATION: right record starts before left's fault/dead-time
    # sequence has fully concluded (i.e. within the fault+reclose window).
    left_sequence_end_s = None
    if left_last_reclose_s is not None:
        left_sequence_end_s = left_last_reclose_s
    elif left_clearing_s is not None:
        left_sequence_end_s = left_clearing_s
    right_start_s = right_start_on_left_s if right_start_on_left_s is not None else gap_s
    if left_sequence_end_s is not None and right_start_s <= (left_sequence_end_s + CONTINUATION_GAP_MS / 1000.0):
        evidence_for.append({"type": "STARTS_BEFORE_PRIOR_SEQUENCE_CONCLUDED", "value": gap_ms})
        assumptions.append("Right record's start falls within the left record's fault/reclose sequence window; treated as a continuation rather than a fully independent new episode.")
        return _build(new_id_fn, incident_id, left, right, "CONTINUATION", 0.6, evidence_for, evidence_against, assumptions, warnings, metrics)

    # --- POSSIBLE_EVOLVING_FAULT: phase set progressed AND left ended in a
    # failed reclose or the fault duration/character suggests a permanent
    # aftermath directly following a transient. Conservative naming per spec.
    if phases_progressed and gap_s <= REPEATED_FAULT_MAX_GAP_S:
        evidence_for.append({"type": "FAULT_PHASE_PROGRESSED", "from": sorted(left_phases), "to": sorted(right_phases)})
        if left_reclose_outcome is False:
            evidence_for.append({"type": "PRECEDED_BY_FAILED_RECLOSE", "description": "The earlier episode's reclose attempt failed."})
        if gap_s > 60:
            evidence_against.append({"type": "SIGNIFICANT_TIME_GAP", "value": gap_s, "description": "Second episode occurred a while after the first; plausible but not proven continuity."})
        confidence = 0.74 if left_reclose_outcome is False else 0.55
        return _build(new_id_fn, incident_id, left, right, "POSSIBLE_EVOLVING_FAULT", confidence, evidence_for, evidence_against, assumptions, warnings, metrics)

    # --- REPEATED_FAULT: same phase set, separated by a clear interval
    # (past any reclose/dead-time window), still within a plausible window.
    if same_phases and gap_s <= REPEATED_FAULT_MAX_GAP_S:
        evidence_for.append({"type": "SAME_FAULTED_PHASES", "value": sorted(left_phases)})
        evidence_for.append({"type": "SEPARATED_BY_CLEAR_INTERVAL", "value": gap_s})
        return _build(new_id_fn, incident_id, left, right, "REPEATED_FAULT", 0.65, evidence_for, evidence_against, assumptions, warnings, metrics)

    # --- NEW_FAULT_EPISODE: clear separation, no strong signature link.
    if gap_s > REPEATED_FAULT_MAX_GAP_S:
        evidence_for.append({"type": "LARGE_TIME_SEPARATION", "value": gap_s})
        return _build(new_id_fn, incident_id, left, right, "NEW_FAULT_EPISODE", 0.5, evidence_for, evidence_against, assumptions, warnings, metrics)

    if left_phases and right_phases and not same_phases and not phases_progressed:
        evidence_against.append({"type": "DIFFERENT_FAULT_SIGNATURE", "left": sorted(left_phases), "right": sorted(right_phases)})
        return _build(new_id_fn, incident_id, left, right, "NEW_FAULT_EPISODE", 0.45, evidence_for, evidence_against, assumptions, warnings, metrics)

    return _build(new_id_fn, incident_id, left, right, "UNCERTAIN", 0.3, evidence_for, evidence_against, assumptions, warnings, metrics)


def _build(
    new_id_fn,
    incident_id: str,
    left: IncidentRecord,
    right: IncidentRecord,
    relationship_type: str,
    confidence: float,
    evidence_for: list[dict],
    evidence_against: list[dict],
    assumptions: list[str],
    warnings: list[dict],
    metrics: dict[str, Any],
) -> RecordRelationship:
    return RecordRelationship(
        relationship_id=new_id_fn(),
        incident_id=incident_id,
        left_record_id=left.incident_record_id,
        right_record_id=right.incident_record_id,
        relationship_type=relationship_type,
        confidence=confidence,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        assumptions=assumptions,
        warnings=warnings,
        metrics=metrics,
    )


def _overlap_s(axis: TimeAxis, a: IncidentRecord, b: IncidentRecord) -> Optional[float]:
    """How long two recordings overlap on the incident clock (s); negative:
    how far apart they are."""
    a0, a1, b0, b1 = axis.start(a), axis.end(a), axis.start(b), axis.end(b)
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return None
    try:
        return (min(a1, b1) - max(a0, b0)).total_seconds()
    except TypeError:  # one timestamp timezone-aware, the other naive
        return None


def _local_counterpart(
    record: IncidentRecord, local_records: list[IncidentRecord], axis: TimeAxis
) -> tuple[Optional[IncidentRecord], Optional[float]]:
    """The reference recorder's record that ``record`` overlaps most — or,
    with no overlap, the nearest one — and that overlap (negative: gap)."""
    best, best_overlap = None, None
    for local in local_records:
        overlap = _overlap_s(axis, local, record)
        if overlap is not None and (best_overlap is None or overlap > best_overlap):
            best, best_overlap = local, overlap
    return best, best_overlap


def _fault_instant(axis: TimeAxis, record: IncidentRecord) -> Optional[datetime]:
    return axis.absolute(record, fault_start_s(record))


def classify_remote_end(
    local: IncidentRecord,
    remote: IncidentRecord,
    axis: TimeAxis,
    new_id_fn,
    incident_id: str,
    overlap_s: float,
) -> RecordRelationship:
    """``remote`` is the other line end's recording of the event ``local``
    captured: a recorder at another substation, running over (or within
    ``REMOTE_END_WINDOW_S`` of) the local recording on the incident clock.
    Not independent evidence of a second event, and not this end's breaker
    sequence — its trip, dead time and reclose belong to the other end."""
    evidence_for: list[dict] = []
    evidence_against: list[dict] = []
    assumptions: list[str] = []
    warnings: list[dict] = []
    metrics: dict[str, Any] = {"link": "other_recorder"}

    evidence_for.append({
        "type": "OTHER_SUBSTATION", "local": station_of(local), "remote": station_of(remote),
        "description": "Recorded at the other end of the line.",
    })
    if overlap_s >= 0:
        metrics["overlap_seconds"] = round(overlap_s, 3)
        evidence_for.append({"type": "TIME_RANGE_OVERLAP", "value": round(overlap_s, 3)})
    else:
        metrics["separation_seconds"] = round(-overlap_s, 3)
        evidence_for.append({
            "type": "WITHIN_RECLOSE_WINDOW", "value": round(-overlap_s, 3),
            "description": f"The recordings do not overlap; they are {-overlap_s:.1f} s apart.",
        })

    placement = axis.placement(remote)
    confidence = 0.6
    if placement is not None and placement.method == "fault_aligned":
        metrics["clock_offset_ms"] = placement.clock_offset_ms
        metrics["zone_offset_h"] = placement.zone_offset_h
        evidence_for.append({
            "type": "CLOCK_LINED_UP_ON_SHARED_FAULT",
            "description": "The other end's clock was lined up with this end's on a fault both recorded.",
        })
        confidence = 0.85
    elif placement is not None and placement.method == "own_clock_unverified":
        assumptions.append(
            "The other end's clock could not be checked against a shared fault; the pairing rests on its own timestamps."
        )
        confidence = 0.5

    local_fault, remote_fault = _fault_instant(axis, local), _fault_instant(axis, remote)
    if local_fault is not None and remote_fault is not None:
        difference_s = _seconds_between(local_fault, remote_fault)
        if difference_s is not None:
            metrics["fault_start_difference_ms"] = round(difference_s * 1000.0, 1)
        local_phases, remote_phases = fault_phases(local), fault_phases(remote)
        if local_phases and remote_phases:
            if local_phases & remote_phases:
                evidence_for.append({"type": "SAME_FAULTED_PHASES_AT_BOTH_ENDS", "local": sorted(local_phases), "remote": sorted(remote_phases)})
            else:
                evidence_against.append({
                    "type": "DIFFERENT_FAULTED_PHASES", "local": sorted(local_phases), "remote": sorted(remote_phases),
                    "description": "The two ends read different faulted phases.",
                })
    return _build(new_id_fn, incident_id, local, remote, "REMOTE_END_CAPTURE", confidence,
                  evidence_for, evidence_against, assumptions, warnings, metrics)


def primary_lane(axis: TimeAxis, ordered: list[IncidentRecord]) -> Optional[str]:
    """The reference recorder: the incident's story is told from its end."""
    if axis.reference_group is not None:
        return axis.reference_group
    return axis.lane(ordered[0]) if ordered else None


def build_relationships(
    incident_id: str,
    records: list[IncidentRecord],
    alignment: AlignmentAssessment,
    new_id_fn,
    axis: Optional[TimeAxis] = None,
) -> list[RecordRelationship]:
    """Classify how each record relates to the records before it, per
    recorder ("lane") on the incident time axis:

    - within one recorder, consecutive records in time order — a fault file,
      the dead-time file of its reclose, a re-fault file (``link``
      ``"same_recorder"``);
    - a record from another recorder is tied to the reference recorder's
      record it overlaps: the other line end's capture of the same event
      (``REMOTE_END_CAPTURE``), or, at the same substation, a second device's
      capture, compared as before (duplicate / overlapping capture). From
      another substation, the nearest reference record within
      ``REMOTE_END_WINDOW_S`` also counts (``link`` ``"other_recorder"``);
    - the first record of another recorder with nothing of the reference
      recorder around it is compared with the record just before it, as all
      records used to be (``link`` ``"nearest_record"``).

    Only those pairs are compared — O(n) rather than O(n^2)."""
    axis = axis or build_time_axis(records)
    order_index = {rid: i for i, rid in enumerate(alignment.record_order)}
    ordered = sorted(records, key=lambda r: order_index.get(r.incident_record_id, r.sequence_index))
    primary = primary_lane(axis, ordered)
    local_records = [r for r in ordered if axis.lane(r) == primary]

    def classify(left: IncidentRecord, right: IncidentRecord, link: str, prior_fault=None) -> RecordRelationship:
        rel = classify_pair(left, right, alignment, new_id_fn, incident_id, prior_fault=prior_fault, axis=axis)
        rel.metrics["link"] = link
        return rel

    relationships: list[RecordRelationship] = []
    previous_in_lane: dict[str, IncidentRecord] = {}
    prior_fault_in_lane: dict[str, IncidentRecord] = {}
    for i, record in enumerate(ordered):
        lane = axis.lane(record)
        previous = previous_in_lane.get(lane)
        if previous is not None:
            if _has_fault_inception(previous):
                prior_fault_in_lane[lane] = previous
            relationships.append(classify(previous, record, "same_recorder", prior_fault_in_lane.get(lane)))
        previous_in_lane[lane] = record
        if lane == primary:
            continue

        counterpart, overlap = _local_counterpart(record, local_records, axis)
        same_bay = counterpart is not None and (
            same_station(station_of(counterpart), station_of(record))
            or not (station_of(counterpart) and station_of(record))
        )
        if counterpart is not None and overlap is not None and overlap >= 0 and same_bay:
            relationships.append(classify(counterpart, record, "other_recorder"))
        elif counterpart is not None and overlap is not None and not same_bay and overlap >= -REMOTE_END_WINDOW_S:
            relationships.append(classify_remote_end(counterpart, record, axis, new_id_fn, incident_id, overlap))
        elif previous is None and i > 0:
            relationships.append(classify(ordered[i - 1], record, "nearest_record"))
    return relationships
