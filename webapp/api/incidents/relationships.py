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

_NO_INCEPTION_METHODS = {"dead_time_recording", "trigger_fallback", "no_fault_evidence", "insufficient_data"}


def _load_line_payload(analysis_id: str) -> Optional[dict]:
    """Stored payload restricted to the record's disturbed line, so a DFR
    file carrying two lines is compared on the line that actually faulted
    (channels are keyed by canonical name below, which would otherwise let
    one line's IA silently overwrite the other's)."""
    payload = load_analysis(analysis_id)
    return scope_payload(payload) if payload is not None else None


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _record_time(record: IncidentRecord) -> Optional[datetime]:
    return _parse_iso(record.trigger_time_iso) or _parse_iso(record.record_start_iso)


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


def _abs_at(record: IncidentRecord, t_s: Optional[float]) -> Optional[datetime]:
    """Wall-clock time of an instant on the record's own time axis (seconds;
    the axis starts at the first sample, the trigger sits at
    ``trigger_offset_s`` on it)."""
    if t_s is None:
        return None
    window = _window(record)
    start = _parse_iso(record.record_start_iso)
    if start is not None:
        return start + timedelta(seconds=float(t_s) - float(window.get("record_start_ms") or 0.0) / 1000.0)
    trigger = _parse_iso(record.trigger_time_iso)
    if trigger is not None:
        offset_s = record.trigger_offset_s
        if offset_s is None:
            offset_s = float(window.get("trigger_time_ms") or 0.0) / 1000.0
        return trigger + timedelta(seconds=float(t_s) - float(offset_s))
    return None


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


def _waveform_similarity(left: IncidentRecord, right: IncidentRecord) -> dict[str, Any]:
    """Bounded waveform similarity computed ONLY over the overlapping
    absolute-time window and only for matching canonical phase-current
    channels. Returns an empty/low-confidence result if either record lacks
    absolute time, the ranges don't overlap, or channels can't be paired —
    never raises, never falls back to a full-record comparison."""
    result: dict[str, Any] = {"computed": False, "reason": None}

    t_left = _record_time(left)
    t_right = _record_time(right)
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

        overlap_start = max(left_abs[0], right_abs[0])
        overlap_end = min(left_abs[1], right_abs[1])
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
) -> RecordRelationship:
    """Classify the relationship between two attached incident records.

    ``left``/``right`` are assumed already in chronological (or best-known)
    order per ``alignment.record_order``. ``prior_fault`` is the latest
    record before ``right`` that contains a fault inception (``left`` itself,
    or an earlier one when ``left`` only captured a reclose) — what a new
    fault's phases are compared against.
    """
    evidence_for: list[dict] = []
    evidence_against: list[dict] = []
    assumptions: list[str] = []
    warnings: list[dict] = []
    metrics: dict[str, Any] = {}

    t_left = _record_time(left)
    t_right = _record_time(right)
    gap_s = None
    if t_left is not None and t_right is not None:
        gap_s = (t_right - t_left).total_seconds()
        metrics["gap_seconds"] = round(gap_s, 3)
    else:
        warnings.append({"type": "NO_ABSOLUTE_TIME", "description": "At least one record lacks absolute time; relationship relies on order and signature only."})
        assumptions.append("Temporal gap is unknown; classification relies on record order and fault-signature evidence only.")

    left_phases = _phases(left)
    right_phases = _phases(right)
    same_phases = bool(left_phases) and left_phases == right_phases
    phases_progressed = bool(left_phases) and bool(right_phases) and left_phases < right_phases

    left_no_fault = _is_no_fault(left)
    right_no_fault = _is_no_fault(right)

    similarity = _waveform_similarity(left, right)
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
    if left_sequence_end_s is not None and gap_s <= (left_sequence_end_s + CONTINUATION_GAP_MS / 1000.0):
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


def build_relationships(
    incident_id: str,
    records: list[IncidentRecord],
    alignment: AlignmentAssessment,
    new_id_fn,
) -> list[RecordRelationship]:
    """Classify relationships for consecutive record pairs in chronological
    (or best-known) order. Only adjacent pairs are compared by default —
    O(n) rather than O(n^2) — since Stage 2 targets same-bay incidents where
    the interesting relationships are almost always between neighbors in
    sequence. Non-adjacent duplicate/overlap pairs would require full
    pairwise comparison, which is explicitly out of scope for a same-bay
    reconstruction pass and left for manual relationship inspection."""
    order_index = {rid: i for i, rid in enumerate(alignment.record_order)}
    ordered = sorted(records, key=lambda r: order_index.get(r.incident_record_id, r.sequence_index))

    relationships: list[RecordRelationship] = []
    prior_fault: Optional[IncidentRecord] = None
    for i in range(len(ordered) - 1):
        left, right = ordered[i], ordered[i + 1]
        if _has_fault_inception(left):
            prior_fault = left
        relationships.append(classify_pair(left, right, alignment, new_id_fn, incident_id, prior_fault=prior_fault))
    return relationships
