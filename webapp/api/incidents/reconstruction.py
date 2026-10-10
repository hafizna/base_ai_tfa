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
from .episodes import _reclose_outcome as _record_reclose_outcome
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
from .time_axis import TimeAxis, build_time_axis
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


# run_ml_prediction's 17-feature extractor (webapp.api.ml_predict.extract_ml_features)
# is built exclusively around single-line three-phase IA/IB/IC/VA/VB/VC
# channels and the 7-class line-fault taxonomy (PETIR/LAYANG/POHON/HEWAN/
# BENDA_ASING/KONDUKTOR/PERALATAN) — it has no branch for transformer HV/LV/
# diff/restraint channels or any other protection family's channel layout.
# The single-record 87T workspace already knows this and deliberately shows
# NO AI verdict for transformer records (README: "Workspace 87T saat ini
# hanya menampilkan analisa rekaman COMTRADE, tanpa verdict AI") — a
# same-bay incident is explicitly allowed to mix protection families (a
# distance relay AND a transformer differential relay both tripping for one
# event is normal same-bay evidence, see same_bay.py's MIXED_PROTECTION_FAMILY),
# so this same restraint has to be applied per-record here too, or a 87T
# record silently gets a meaningless line-fault cause reading.
_LINE_FAULT_CLASSIFIER_PROTECTION_TYPES = {"21", "87L"}


def _record_ml_result(record: IncidentRecord) -> dict[str, Any]:
    """Best-effort full ``run_ml_prediction`` result for one record, computed
    fresh (never averaged with other records — see module docstring).
    Returns ``{}`` rather than raising if the stored analysis has expired or
    the model import fails; reconstruction must not fail because one
    record's ML call did. Also returns ``{}`` — with a distinguishable
    ``skip_reason`` — when the record's protection type isn't one the line-
    fault classifier is built for (see _LINE_FAULT_CLASSIFIER_PROTECTION_TYPES)."""
    relay_type = (record.protection_type or "21").upper()
    if relay_type not in _LINE_FAULT_CLASSIFIER_PROTECTION_TYPES:
        return {"skip_reason": "unsupported_protection_type", "protection_type": relay_type}

    payload = load_analysis(record.analysis_id)
    if payload is None:
        return {}
    try:
        return ml_predict.run_ml_prediction(payload, relay_type)
    except Exception:
        return {}


# Relationship types where the right record captures the AFTERMATH of the
# left record's event (a reclose attempt/outcome, or a continued capture of
# the same still-in-progress sequence) rather than a new, independently
# faulted waveform. A record reached only via one of these relationships is
# not treated as separate cause evidence — see _evidence_roles. (The other
# line end's recording, REMOTE_END_CAPTURE, gets a role of its own.)
_AFTERMATH_RELATIONSHIP_TYPES = {"RECLOSE_SEQUENCE", "CONTINUATION", "DUPLICATE_TRIGGER", "OVERLAPPING_CAPTURE"}


def _evidence_roles(
    records: list[IncidentRecord], relationships: list[RecordRelationship], record_order: list[str]
) -> dict[str, str]:
    """Classify each record as ``"inception"`` (captures an independently
    faulted waveform — its cause hypothesis is real evidence),
    ``"aftermath"`` (only captures the reclose/continuation/duplicate of an
    earlier record's event — its cause hypothesis reflects whatever the
    classifier saw in ITS OWN waveform, e.g. reclose inrush or CT transient,
    not a second independent cause, and must not be pitted against the
    inception record's reading), or ``"remote_end"`` (the other line end's
    recording of an event this end recorded — another view of the same
    fault, not a second one, and often a weak-infeed end whose waveform the
    classifier was not trained on).

    Chosen from the relationships that tie each record to an earlier one (the
    pairs ``relationships.build_relationships`` classified: its own
    recorder's previous record, another recorder's capture of the same
    event). A record with no such relationship — the first record overall —
    is always "inception"; only an explicit aftermath-type or remote-end tie
    demotes it.
    """
    incoming: dict[str, list[RecordRelationship]] = {}
    for rel in relationships:
        incoming.setdefault(rel.right_record_id, []).append(rel)

    roles: dict[str, str] = {}
    for record in records:
        types = {rel.relationship_type for rel in incoming.get(record.incident_record_id) or []}
        if "REMOTE_END_CAPTURE" in types:
            roles[record.incident_record_id] = "remote_end"
        elif types & _AFTERMATH_RELATIONSHIP_TYPES:
            roles[record.incident_record_id] = "aftermath"
        else:
            roles[record.incident_record_id] = "inception"
    return roles


def _physical_cause_evidence(
    records: list[IncidentRecord],
    relationships: list[RecordRelationship],
    record_order: list[str],
) -> dict[str, Any]:
    """Per-record top cause hypothesis plus a qualitative consistency label.
    Deliberately does NOT average or otherwise combine per-record
    probabilities into one incident-level probability (spec section 11) —
    duplicate records aren't independent evidence, and different episodes
    may have different mechanisms entirely. ``incident_root_cause`` stays
    ``"UNCONFIRMED"`` unconditionally — this function still never claims a
    confirmed root cause, it only decides which per-record readings count as
    evidence when judging whether they *agree*.

    Consistency is judged only across records with evidence_role
    "inception" (see ``_evidence_roles``): a record whose ONLY relationship
    to its predecessor is RECLOSE_SEQUENCE/CONTINUATION/DUPLICATE_TRIGGER/
    OVERLAPPING_CAPTURE captures the aftermath of that predecessor's fault,
    not an independently faulted waveform — running the classifier on it is
    still useful (its own reading is preserved and shown, e.g. to catch a
    refault-on-reclose), but a different top_hypothesis there must not, on
    its own, downgrade consistency to MIXED. Real precedent: a trip record
    (LAYANG, 89%) and its close/reclose record (PETIR, 79%, because reclose
    inrush has a different waveform shape than the original fault) previously
    reported MIXED/"signatures disagree" even though there was only ever one
    physical fault event — the close record's classifier run was never
    independent evidence about what caused the fault in the first place.

    Each record entry carries the exact audit trail used at reconstruction
    time (model_version, feature_version, calibration method, timing
    source, raw/calibrated probabilities, applied confidence caps) so the
    stored ``Reconstruction`` snapshot remains the source of truth even if
    the live model or a record's analysis session later changes.
    """
    roles = _evidence_roles(records, relationships, record_order)

    record_entries = []
    inception_causes = []
    for record in records:
        result = _record_ml_result(record)
        ranking = result.get("cause_ranking") or []
        top = ranking[0] if ranking else None
        cause = top.get("cause") if isinstance(top, dict) else None
        confidence = top.get("confidence") if isinstance(top, dict) else None
        meta = result.get("meta") or {}
        role = roles.get(record.incident_record_id, "inception")
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
            "evidence_role": role,
            "fault_type": result.get("fault_type"),
            # Set only when this record's protection type isn't one the line
            # -fault classifier is built for (e.g. "87T") — distinguishes
            # "no reading because unsupported" from "no reading because the
            # model/session failed" for the UI. See _record_ml_result.
            "skip_reason": result.get("skip_reason"),
        })
        if cause and role == "inception":
            inception_causes.append(cause)

    if not record_entries:
        consistency = "INSUFFICIENT"
    elif not inception_causes:
        # Every record was classified as "aftermath" (shouldn't normally
        # happen — the first record in chronological order is always
        # "inception" — but degrade honestly rather than divide by zero).
        consistency = "INSUFFICIENT"
    else:
        # CONSISTENT/MOSTLY_CONSISTENT/MIXED is judged purely among inception
        # records — an aftermath record's own reading never counts toward or
        # against agreement, regardless of how many aftermath records exist.
        distinct = set(inception_causes)
        if len(distinct) == 1:
            consistency = "CONSISTENT"
        elif len(distinct) <= max(1, len(inception_causes) // 2):
            consistency = "MOSTLY_CONSISTENT"
        else:
            consistency = "MIXED"

    return {
        "scope": "RECORD_LOCAL_SIGNATURES",
        "records": record_entries,
        "consistency": consistency,
        "incident_root_cause": "UNCONFIRMED",
    }


# A physical fault mechanism predicts a specific reclose outcome: a genuinely
# transient cause (lightning, kite contact, animal, foreign object — the
# object/arc is gone after the trip) should self-clear, so a successful
# reclose is the EXPECTED outcome, not independent confirmation of nothing.
# A permanent cause (conductor/tower damage, stuck equipment) should NOT
# self-clear, so a failed reclose is the expected outcome. This mirrors how
# a protection engineer reads the two files together: the reclose result is
# evidence for or against the inception record's cause hypothesis, even
# though it was never re-run through the classifier as competing evidence.
_TRANSIENT_EXPECTS_SUCCESS = True   # cause fault_type == "transient"
_PERMANENT_EXPECTS_FAILURE = True   # cause fault_type == "permanent"

# Confidence adjustment magnitude — deliberately small and symmetric (this is
# corroboration/contradiction from ONE downstream reclose outcome, not a
# second independent classifier vote) and never exceeds the same 92% hard
# ceiling ml_predict.py already enforces.
_RECLOSE_MATCH_BONUS = 0.05
_RECLOSE_MISMATCH_PENALTY = 0.10
_CONFIDENCE_CEILING = 0.92


def _apply_reclose_outcome_cross_validation(
    physical_cause: dict[str, Any],
    records: list[IncidentRecord],
    relationships: list[RecordRelationship],
    record_order: list[str],
    axis: Optional[TimeAxis] = None,
) -> dict[str, Any]:
    """Adjust each inception record's cause confidence based on whether the
    reclose outcome captured by its aftermath record(s) is physically
    consistent with the cause's fault_type (transient vs permanent) —
    recorded as an ``applied_caps`` entry named
    ``reclose_outcome_consistency`` / ``reclose_outcome_conflict`` on that
    record, exactly like ml_predict.py's existing caps, so the adjustment is
    auditable rather than silent.

    Must run BEFORE group_episodes (which consumes physical_cause's
    per-record confidence via cause_lookup) so episodes and the incident-
    level physical_cause_evidence agree on one final, already-adjusted
    confidence — never two numbers for the same record.

    Only touches "inception"-role records (the ones carrying real cause
    evidence — see _evidence_roles) and only when at least one of ITS OWN
    aftermath records (same grouping the RECLOSE_SEQUENCE/CONTINUATION/etc.
    relationship already establishes) has a reclose outcome to compare
    against. Never changes incident_root_cause (still UNCONFIRMED) or
    invents a new cause — purely reweights confidence in the SAME
    already-computed cause_ranking.
    """
    roles = _evidence_roles(records, relationships, record_order)
    order_index = {rid: i for i, rid in enumerate(record_order)}
    ordered = sorted(records, key=lambda r: order_index.get(r.incident_record_id, 10**9))
    lane_of = {r.incident_record_id: (axis.lane(r) if axis is not None else "") for r in records}
    # The relationship tying each record to an earlier record of the same
    # breaker: its own recorder's previous record, or a second device's
    # capture in the same bay (not the far end — that is the other breaker).
    previous_rel: dict[str, RecordRelationship] = {}
    for rel in relationships:
        if (rel.metrics or {}).get("link") == "other_recorder" and rel.relationship_type != "REMOTE_END_CAPTURE":
            previous_rel[rel.right_record_id] = rel
    for rel in relationships:
        if lane_of.get(rel.left_record_id) == lane_of.get(rel.right_record_id) and (rel.metrics or {}).get("link") != "other_recorder":
            previous_rel.setdefault(rel.right_record_id, rel)

    # Map each inception record -> the reclose outcome captured by the
    # following aftermath record(s) of the same breaker (the far end's
    # reclose is its own breaker's). A reclose that succeeded but was
    # followed by a REFAULT_AFTER_RECLOSE did not hold: recorded as
    # "refault_after_reclose", which is what the cause has to explain.
    reclose_outcome_by_inception: dict[str, str] = {}
    current_inception_by_lane: dict[str, str] = {}
    for record in ordered:
        rel = previous_rel.get(record.incident_record_id)
        lane = lane_of[rel.left_record_id] if rel is not None else lane_of[record.incident_record_id]
        current_inception_id = current_inception_by_lane.get(lane)
        if rel is not None and rel.relationship_type == "REFAULT_AFTER_RECLOSE" and current_inception_id is not None:
            reclose_outcome_by_inception[current_inception_id] = "refault_after_reclose"
        role = roles.get(record.incident_record_id, "inception")
        if role == "inception":
            current_inception_by_lane[lane] = record.incident_record_id
            continue
        if role == "remote_end" or current_inception_id is None:
            continue
        outcome = _record_reclose_outcome(record)
        if rel is not None and (rel.metrics or {}).get("reclose_outcome_correction") == "failed":
            outcome = "failed"
        if outcome is not None:
            # Last aftermath record's outcome wins if there are several
            # (e.g. a failed then successful second attempt).
            reclose_outcome_by_inception[current_inception_id] = outcome

    for entry in physical_cause["records"]:
        if entry.get("evidence_role") != "inception":
            continue
        fault_type = entry.get("fault_type")
        cause = entry.get("top_hypothesis")
        confidence = entry.get("confidence")
        if fault_type not in ("transient", "permanent") or cause is None or confidence is None:
            continue

        reclose_outcome = reclose_outcome_by_inception.get(entry["incident_record_id"])
        if reclose_outcome not in ("successful", "failed", "refault_after_reclose"):
            continue  # no reclose evidence to cross-validate against

        # A re-fault seconds after a successful reclose behaves like a
        # persisting cause: the transient reading is the one it contradicts.
        held = reclose_outcome == "successful"
        expected_success = fault_type == "transient"
        matches = held == expected_success

        cap_name = "reclose_outcome_consistency" if matches else "reclose_outcome_conflict"
        delta = _RECLOSE_MATCH_BONUS if matches else -_RECLOSE_MISMATCH_PENALTY
        new_confidence = round(min(_CONFIDENCE_CEILING, max(0.0, confidence + delta)), 3)
        if new_confidence == confidence:
            continue

        outcome_txt = (
            "the line faulted again shortly after a successful reclose"
            if reclose_outcome == "refault_after_reclose" else f"reclose outcome '{reclose_outcome}'"
        )
        reason = (
            f"Episode {outcome_txt} — "
            f"{'consistent with' if matches else 'inconsistent with'} a "
            f"{fault_type} cause ({cause}): "
            f"{'transient causes are expected to self-clear and stay cleared' if fault_type == 'transient' else 'permanent causes are expected to persist through reclose'}."
        )
        entry.setdefault("applied_caps", []).append({
            "name": cap_name, "before": confidence, "after": new_confidence, "reason": reason,
        })
        entry["confidence"] = new_confidence
        for candidate in entry.get("cause_ranking") or []:
            if candidate.get("cause") == cause:
                candidate["confidence"] = new_confidence
                break

        if not matches:
            entry["requires_review"] = True

    return physical_cause


def _observed_incident_facts(records: list[IncidentRecord], episodes: list[FaultEpisode], axis: TimeAxis) -> dict[str, Any]:
    """``incident_duration_ms``: first to last record trigger on the incident
    time axis."""
    times = [t for t in (axis.trigger(r) for r in records) if t is not None]
    duration_ms = None
    if len(times) >= 2:
        try:
            duration_ms = (max(times) - min(times)).total_seconds() * 1000.0
        except TypeError:  # one timestamp timezone-aware, the other naive
            duration_ms = None

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

    refaults = [e for e in episodes[1:] if e.relationship_to_previous == "REFAULT_AFTER_RECLOSE"]
    if refaults:
        delays = [(e.observed_facts or {}).get("seconds_after_previous_reclose") for e in refaults]
        delay_txt = ", ".join(f"{d:.1f} s" for d in delays if isinstance(d, (int, float))) or "seconds"
        if last.reclose_outcome == "successful":
            final_txt = "The last fault was reclosed successfully."
        elif last.reclose_outcome == "failed":
            final_txt = "The reclose after the last fault failed."
        else:
            final_txt = "No reclose of the last fault was captured in the attached records (the line may have locked out)."
        return {
            "event_class": "RECLOSE_THEN_REFAULT",
            "summary": (
                f"A fault was cleared and the line reclosed successfully, but it faulted again {delay_txt} after "
                f"the reclose — the reclose did not hold. {final_txt}"
            ),
        }

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
    hypotheses.extend(_pattern_based_cause_signals(episodes))
    hypotheses.extend(_refault_after_reclose_signals(episodes))
    return hypotheses


def _refault_after_reclose_signals(episodes: list[FaultEpisode]) -> list[dict[str, Any]]:
    """A fault that comes back seconds after a SUCCESSFUL reclose: the line was
    healthy when re-energized, then faulted again — so whatever caused the
    first fault did not go away at the trip. That is the textbook signature
    of something still near the conductor (vegetation, a foreign object, a
    sagging conductor), and the reading a single-record classifier cannot
    make: on the 21/08/2023 Bringin-Mojosongo #2 tree fault, each fault
    record on its own read as lightning. Same phases strengthen it (a fixed
    contact point reproduces the same fault); a second independent strike in
    the same storm stays possible and is stated, not ruled out."""
    signals: list[dict[str, Any]] = []
    for i in range(1, len(episodes)):
        prev, cur = episodes[i - 1], episodes[i]
        if cur.relationship_to_previous != "REFAULT_AFTER_RECLOSE":
            continue
        seconds = (cur.observed_facts or {}).get("seconds_after_previous_reclose")
        prev_phases, cur_phases = set(prev.faulted_phases or []), set(cur.faulted_phases or [])
        same = bool(prev_phases) and prev_phases == cur_phases
        when = f"{seconds:.1f} s" if isinstance(seconds, (int, float)) else "seconds"
        evidence_against = [
            "A second, independent lightning strike within seconds cannot be excluded from COMTRADE evidence "
            "alone — lightning-detection data, or both faults locating to the same point on the line, would settle it.",
        ]
        if prev_phases and cur_phases and not same:
            evidence_against.append(
                f"The faulted phases differ ({'-'.join(sorted(prev_phases))} then {'-'.join(sorted(cur_phases))})."
            )
        signals.append({
            "hypothesis": "REFAULT_AFTER_SUCCESSFUL_RECLOSE",
            "mechanism_signal": "CONSISTENT_WITH_PERSISTENT_PHYSICAL_CONTACT",
            "confidence": 0.6 if same else 0.45,
            "episode_indices": [prev.episode_index, cur.episode_index],
            "evidence_for": [
                f"The line faulted again {when} after a successful reclose"
                + (f", on the same phases ({'-'.join(sorted(cur_phases))})." if same else "."),
            ],
            "evidence_against": evidence_against,
            "description": (
                "The line was re-energized healthy and faulted again within seconds, so the cause of the first "
                "fault did not go away when the breaker tripped. That is more consistent with something still near "
                "the conductor (e.g. vegetation or a foreign object) than with a single lightning transient, which "
                "leaves nothing behind once the arc is extinguished. Not a confirmed cause."
            ),
        })
    return signals


# --- Pattern-based mechanism signals -----------------------------------------
#
# These are NOT a replacement for the per-record LightGBM cause_ranking (see
# _physical_cause_evidence) and never produce a "confirmed" cause. LightGBM
# reads one record's waveform in isolation and cannot see a multi-episode
# pattern; these rules read the opposite — the SHAPE of the incident across
# episodes (phase count trending up, an identical fault recurring, a failed
# reclose) — using textbook protection-engineering associations that a human
# reviewer would draw by eye from the episode table. Deliberately coarse
# (thresholds, not a learned model) and always phrased as "consistent with",
# carrying its own evidence_for/evidence_against, so it is exactly as
# auditable as every other reconstruction conclusion and never silently
# overrides or averages into the per-record cause_ranking.
#
# Mechanism vocabulary matches core/ml_predict cause labels (PETIR = lightning,
# BENDA_ASING = foreign object / vegetation, KONDUKTOR = conductor fault) so a
# reader can directly compare a pattern signal against the per-record
# cause_ranking candidates shown in physical_cause_evidence.

_ESCALATION_MAX_GAP_S = 30.0     # phase count trending up must happen quickly to read as "one worsening event"
_RECURRING_MAX_GAP_S = 3600.0    # matches REPEATED_FAULT_MAX_GAP_S in relationships.py
_TRANSIENT_MAX_DURATION_MS = 100.0


def _phase_count(episode: FaultEpisode) -> int:
    return len(set(episode.faulted_phases or []))


def _pattern_based_cause_signals(episodes: list[FaultEpisode]) -> list[dict[str, Any]]:
    if not episodes:
        return []

    signals: list[dict[str, Any]] = []

    # --- ESCALATING_PHASE_INVOLVEMENT: consecutive episodes where the
    # faulted-phase COUNT strictly increases within a short window (e.g. a
    # phase-to-phase fault followed shortly by a three-phase fault at the
    # same location). A single lightning strike is a near-instantaneous
    # transient — it does not typically re-manifest moments later as a
    # LARGER fault at the same spot. A worsening contact (a falling branch,
    # a foreign object settling further onto the conductors, vegetation
    # burning through) escalating over seconds fits this shape much better.
    for i in range(1, len(episodes)):
        prev, cur = episodes[i - 1], episodes[i]
        prev_n, cur_n = _phase_count(prev), _phase_count(cur)
        if prev_n == 0 or cur_n <= prev_n:
            continue
        prev_t = _parse_iso(prev.end_iso or prev.start_iso)
        cur_t = _parse_iso(cur.start_iso)
        if prev_t is None or cur_t is None:
            continue
        gap_s = (cur_t - prev_t).total_seconds()
        if gap_s < 0 or gap_s > _ESCALATION_MAX_GAP_S:
            continue
        signals.append({
            "hypothesis": "ESCALATING_PHASE_INVOLVEMENT",
            "mechanism_signal": "CONSISTENT_WITH_PHYSICAL_CONTACT",
            "confidence": 0.5,
            "episode_indices": [prev.episode_index, cur.episode_index],
            "evidence_for": [
                f"Faulted phases went from {sorted(set(prev.faulted_phases))} ({prev_n}-phase) to "
                f"{sorted(set(cur.faulted_phases))} ({cur_n}-phase) within {round(gap_s, 1)}s.",
            ],
            "evidence_against": [],
            "description": (
                "Fault severity escalated (more phases involved) within a short window. This pattern is "
                "more commonly associated with a worsening physical contact — e.g. vegetation or a foreign "
                "object — than with a single lightning transient, which does not usually re-escalate "
                "moments after clearing. Not a confirmed cause; corroborate with field inspection or "
                "lightning-detection network data."
            ),
        })

    # --- RECURRING_SAME_SIGNATURE: 2+ episodes with the identical faulted-phase
    # set and fault_type, separated by a clear interval, each with a
    # successful reclose (so the line kept re-energizing into the same
    # fault). Consistent with an intermittent contact (e.g. a branch
    # swinging into and out of clearance) rather than one-off transients,
    # though repeated independent lightning strikes are also possible and
    # cannot be excluded from this evidence alone.
    signature_groups: dict[tuple, list[FaultEpisode]] = {}
    for ep in episodes:
        if not ep.faulted_phases or ep.reclose_outcome != "successful":
            continue
        key = (tuple(sorted(set(ep.faulted_phases))), ep.fault_type)
        signature_groups.setdefault(key, []).append(ep)
    for (phases, fault_type), group in signature_groups.items():
        if len(group) < 2:
            continue
        times = sorted(t for t in (_parse_iso(e.start_iso) for e in group) if t)
        if len(times) < 2 or (times[-1] - times[0]).total_seconds() > _RECURRING_MAX_GAP_S:
            continue
        signals.append({
            "hypothesis": "RECURRING_SAME_SIGNATURE",
            "mechanism_signal": "CONSISTENT_WITH_INTERMITTENT_CONTACT_OR_REPEATED_STRIKES",
            "confidence": 0.4,
            "episode_indices": [e.episode_index for e in group],
            "evidence_for": [
                f"{len(group)} episodes share the same faulted phases {list(phases)} and fault type "
                f"({fault_type}), each followed by a successful reclose, over "
                f"{round((times[-1] - times[0]).total_seconds(), 1)}s.",
            ],
            "evidence_against": [
                "Repeated independent lightning strikes on the same phases cannot be ruled out from "
                "COMTRADE evidence alone.",
            ],
            "description": (
                "The same fault signature recurred multiple times with successful reclose each time. "
                "Consistent with an intermittent physical contact (e.g. vegetation swinging in and out of "
                "clearance), though repeated independent transient strikes remain possible. Not a confirmed "
                "cause."
            ),
        })

    # --- FAILED_RECLOSE_PERMANENT: any episode whose reclose attempt failed
    # indicates the fault was still present when the breaker re-energized —
    # i.e. a permanent condition, not a transient that had already cleared.
    # A permanent fault is inconsistent with lightning (which does not
    # persist) and consistent with a sustained physical obstruction (a
    # fallen tree/branch still in contact, permanent conductor damage).
    failed = [e for e in episodes if e.reclose_outcome == "failed"]
    if failed:
        signals.append({
            "hypothesis": "FAILED_RECLOSE_INDICATES_PERMANENT_FAULT",
            "mechanism_signal": "CONSISTENT_WITH_SUSTAINED_PHYSICAL_OBSTRUCTION",
            "confidence": 0.6,
            "episode_indices": [e.episode_index for e in failed],
            "evidence_for": [
                f"Episode {e.episode_index + 1} reclose failed — the fault was still present when the "
                "breaker re-energized, indicating a permanent (not transient) condition."
                for e in failed
            ],
            "evidence_against": [],
            "description": (
                "A failed reclose means the fault persisted through re-energization. This is inconsistent "
                "with a transient cause like lightning and consistent with a sustained physical obstruction "
                "(e.g. a fallen tree/branch still in contact with the conductor, or permanent damage). Not a "
                "confirmed cause."
            ),
        })

    # --- SINGLE_TRANSIENT_NO_RECURRENCE: the counter-signal. Exactly one
    # episode, very short duration, successful reclose, and (implicitly, by
    # not appearing above) no escalation or recurrence. This is the classic
    # transient-fault shape and is the pattern most consistent with a single
    # lightning strike or switching transient — included so the absence of
    # the other three signals is stated explicitly rather than left as
    # silence the reader has to infer.
    if (
        len(episodes) == 1
        and episodes[0].duration_ms is not None
        and episodes[0].duration_ms <= _TRANSIENT_MAX_DURATION_MS
        and episodes[0].reclose_outcome == "successful"
    ):
        signals.append({
            "hypothesis": "SINGLE_TRANSIENT_NO_RECURRENCE",
            "mechanism_signal": "CONSISTENT_WITH_TRANSIENT_STRIKE_OR_SWITCHING",
            "confidence": 0.4,
            "episode_indices": [episodes[0].episode_index],
            "evidence_for": [
                f"A single {episodes[0].duration_ms:.0f} ms fault episode cleared on the first reclose "
                "attempt, with no recurrence or escalation seen in the attached records.",
            ],
            "evidence_against": [],
            "description": (
                "A short, one-off fault that cleared on the first reclose is the classic transient-fault "
                "shape most consistent with a single lightning strike or switching transient. Not a "
                "confirmed cause, and does not rule out a physical cause that happened not to recur within "
                "the attached records."
            ),
        })

    # --- REPEATED_ESCALATING_SIGNATURE_AMBIGUOUS: whenever
    # ESCALATING_PHASE_INVOLVEMENT fired above (phase count went up within a
    # short window), state the two competing readings side by side with
    # their actual evidence weight instead of leaving the reader to weigh
    # "escalating fault" against "could just be lightning" themselves.
    # RECURRING_SAME_SIGNATURE firing too (the larger-phase fault repeating
    # identically) strengthens the physical-contact reading and is folded in
    # when present, but is NOT required — requiring it excluded cases like
    # an episode whose reclose was never verified (see PR #14's
    # cb_open_verified fix), where "repeated" can't be claimed even though
    # escalation is still clearly visible. Neither reading is preferred here
    # — this is deliberately NOT a tie-breaker, just an explicit statement
    # of what would make each one right.
    escalating = [s for s in signals if s["hypothesis"] == "ESCALATING_PHASE_INVOLVEMENT"]
    recurring = [s for s in signals if s["hypothesis"] == "RECURRING_SAME_SIGNATURE"]
    if escalating:
        gap_texts = [s["evidence_for"][0] for s in escalating]
        recur_texts = [s["evidence_for"][0] for s in recurring]
        involved_indices = sorted(set(
            idx for s in escalating + recurring for idx in s["episode_indices"]
        ))
        recurrence_clause = (
            "the resulting larger-phase fault then recurring with an identical signature"
            if recurring else
            "with no confirmed recurrence of an identical signature in the attached records"
        )
        physical_favor_recurrence = (
            "the repeated fault's phase set and fault_type being IDENTICAL across episodes (a fixed "
            "contact point reproduces the same signature; independent strikes on the same phases "
            "repeatedly is a coincidence each time), and "
            if recurring else ""
        )
        signals.append({
            "hypothesis": "REPEATED_ESCALATING_SIGNATURE_AMBIGUOUS",
            "mechanism_signal": "LIGHTNING_VS_PHYSICAL_CONTACT_UNRESOLVED",
            "confidence": None,  # deliberately unscored — this signal states a disagreement, not a reading
            "episode_indices": involved_indices,
            "evidence_for": gap_texts + recur_texts,
            "evidence_against": [],
            "description": (
                f"This sequence (phase count escalating, {recurrence_clause}) "
                "can be read two ways, and COMTRADE evidence alone cannot decide between them:\n"
                "(1) PHYSICAL CONTACT — a single worsening contact (e.g. a falling/burning branch, a "
                "foreign object) that first touched fewer phases, then settled onto more, and continues "
                f"to make contact each time the line re-energizes. Favored by: {physical_favor_recurrence}"
                "inter-episode gaps that are short enough to plausibly be one continuously unresolved "
                "contact rather than unrelated weather events.\n"
                "(2) SEPARATE LIGHTNING STRIKES — multiple strokes/strikes in the same storm cell, which "
                "commonly occur seconds apart at the same location and are not physically required to "
                "escalate or repeat identically. Favored by: no field/lightning-network evidence "
                "contradicts it, and a successful reclose after the escalated episode (where confirmed) is "
                "also consistent with that episode being an independently transient strike rather than a "
                "persistent obstruction.\n"
                "Resolving this needs external evidence this record set does not contain: lightning-"
                "detection network data for the incident time/location, or a field inspection report."
            ),
        })

    return signals


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

    # One time axis for every step below: recorder clocks lined up on a
    # shared fault, each record placed by its first sample, told from the
    # incident's own substation.
    axis = build_time_axis(records, home_station=incident.station_name)

    same_bay = assess_same_bay(
        incident.station_name,
        incident.bay_name,
        records,
        override_reason=same_bay_override_reason,
        override_operator=same_bay_override_operator,
        override_at_iso=_now_iso() if same_bay_override_reason or same_bay_override_operator else None,
        axis=axis,
    )

    alignment = assess_alignment(records, axis)
    timeline_events = build_timeline(incident.incident_id, records, alignment, new_id, axis)
    relationships = build_relationships(incident.incident_id, records, alignment, new_id, axis)

    # Computed once, before episode grouping, so episodes and the incident-level
    # physical_cause_evidence report exactly the same per-record ML call
    # result (same model_version/probabilities) rather than invoking
    # LightGBM a second time per episode.
    physical_cause = _physical_cause_evidence(records, relationships, alignment.record_order)

    # Cross-validate each inception record's cause against the reclose
    # outcome its own aftermath record(s) captured — MUST run before
    # group_episodes (below) reads cause_lookup, so episodes and the
    # incident-level physical_cause_evidence agree on one final,
    # already-adjusted confidence rather than reporting two numbers for the
    # same record.
    physical_cause = _apply_reclose_outcome_cross_validation(physical_cause, records, relationships, alignment.record_order, axis)
    cause_lookup = {e["incident_record_id"]: e for e in physical_cause["records"]}

    episodes = group_episodes(
        incident.incident_id, records, relationships, alignment.record_order, new_id,
        record_cause_lookup=cause_lookup, axis=axis,
    )

    observed_facts = _observed_incident_facts(records, episodes, axis)
    interpretation = _protection_sequence_interpretation(episodes, relationships)
    hypotheses = _incident_hypotheses(episodes, relationships)

    narrative = build_narrative(
        episodes,
        observed_facts.get("incident_duration_ms"),
        same_bay.status,
        physical_cause.get("consistency", "INSUFFICIENT"),
        hypotheses,
        physical_cause.get("records", []),
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
        alignment={**alignment.to_dict(), "time_axis": axis.to_dict()},
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
