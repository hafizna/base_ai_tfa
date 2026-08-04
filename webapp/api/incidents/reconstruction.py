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
    hypotheses.extend(_pattern_based_cause_signals(episodes))
    return hypotheses


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
