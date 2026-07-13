"""Deterministic incident narrative generator — Stage 2.

Builds a plain-language summary from the already-computed structured
reconstruction result (episodes, relationships, hypotheses) — not from an
LLM and not from free-form string templating disconnected from the data.
Every sentence traces back to a specific field on a ``FaultEpisode`` or the
incident-level hypothesis list, so the narrative can be audited by diffing
against the structured result.
"""

from __future__ import annotations

from typing import Optional

from .models import FaultEpisode

_PHASE_LABELS = {
    "SLG": "single-line-to-ground",
    "LL_OR_DLG": "phase-to-phase (or double-line-to-ground)",
    "3PH": "three-phase",
}

_RELATIONSHIP_PHRASES = {
    "DUPLICATE_TRIGGER": "was captured redundantly by more than one device",
    "OVERLAPPING_CAPTURE": "overlaps the previous record's capture window",
    "CONTINUATION": "continues directly from the previous record's fault/reclose sequence",
    "RECLOSE_SEQUENCE": "captures the breaker reclose sequence following the previous episode",
    "NEW_FAULT_EPISODE": "is a separate fault episode",
    "REPEATED_FAULT": "is a repeated fault with a similar signature to the previous episode",
    "POSSIBLE_EVOLVING_FAULT": "may represent the previous fault evolving into a different signature",
    "UNRELATED": "does not appear related to the previous episode",
    "UNCERTAIN": "has an uncertain relationship to the previous episode",
}


def _phase_text(phases: list[str]) -> str:
    if not phases:
        return "an unclassified fault"
    return "-".join(phases) + " fault"


def _episode_sentence(episode: FaultEpisode, index: int) -> str:
    phase_txt = _phase_text(episode.faulted_phases)
    fault_type_txt = _PHASE_LABELS.get(episode.fault_type or "", "")
    duration_txt = f", lasting {episode.duration_ms:.0f} ms" if episode.duration_ms else ""
    reclose_txt = ""
    if episode.reclose_outcome == "successful":
        reclose_txt = " Reclose was successful."
    elif episode.reclose_outcome == "failed":
        reclose_txt = " Reclose failed."

    relation_txt = ""
    if episode.relationship_to_previous:
        phrase = _RELATIONSHIP_PHRASES.get(episode.relationship_to_previous, "has an unspecified relationship to the previous episode")
        relation_txt = f" This episode {phrase}."

    member_txt = ""
    if len(episode.member_record_ids) > 1:
        member_txt = f" ({len(episode.member_record_ids)} records captured this episode.)"

    prefix = f"Episode {index + 1}"
    fault_desc = f"a {fault_type_txt} ({phase_txt})" if fault_type_txt else phase_txt
    return f"{prefix} was {fault_desc}{duration_txt}.{reclose_txt}{relation_txt}{member_txt}"


def build_narrative(
    episodes: list[FaultEpisode],
    incident_duration_ms: Optional[float],
    same_bay_status: str,
    consistency: str,
    incident_hypotheses: list[dict],
) -> str:
    """Compose the deterministic narrative. Pure function of its inputs so
    it is trivially testable and auditable — no hidden state, no model call."""
    if not episodes:
        return "No fault episodes could be reconstructed from the attached records."

    record_count = sum(len(e.member_record_ids) for e in episodes)
    duration_txt = ""
    if incident_duration_ms is not None:
        total_seconds = round(incident_duration_ms / 1000.0)
        minutes, remainder = divmod(total_seconds, 60)
        if minutes > 0:
            duration_txt = f" over {minutes} minute{'s' if minutes != 1 else ''} {remainder} second{'s' if remainder != 1 else ''}"
        else:
            duration_txt = f" over {remainder} seconds"

    bay_txt = {
        "CONFIRMED_SAME_BAY": "from the same bay",
        "LIKELY_SAME_BAY": "likely from the same bay",
        "MISMATCH_REQUIRES_REVIEW": "flagged for a possible bay mismatch and require review",
        "UNKNOWN": "with an unconfirmed bay relationship",
    }.get(same_bay_status, "with an unconfirmed bay relationship")

    lines = [
        f"{record_count} COMTRADE record{'s' if record_count != 1 else ''} {bay_txt} were reconstructed into "
        f"{len(episodes)} fault episode{'s' if len(episodes) != 1 else ''}{duration_txt}.",
        "",
    ]

    for idx, episode in enumerate(episodes):
        lines.append(_episode_sentence(episode, idx))

    lines.append("")

    if len(episodes) > 1:
        evolving = [h for h in incident_hypotheses if h.get("hypothesis") == "POSSIBLE_EVOLVING_FAULT"]
        repeated_only = [e for e in episodes if e.relationship_to_previous == "REPEATED_FAULT"]
        if evolving:
            lines.append(
                "The sequence is consistent with a repeated fault that may have evolved into a different or "
                "more severe condition. This is a possibility raised by the pattern of evidence, not a confirmed conclusion."
            )
        elif repeated_only:
            lines.append("The sequence is consistent with independently repeated faults of similar signature.")

    consistency_txt = {
        "CONSISTENT": "The per-record physical-cause signatures are consistent with each other.",
        "MOSTLY_CONSISTENT": "The per-record physical-cause signatures are mostly consistent, with minor disagreement.",
        "MIXED": "The per-record physical-cause signatures disagree; no single cause is dominant across records.",
        "CONTRADICTORY": "The per-record physical-cause signatures actively contradict each other.",
        "INSUFFICIENT": "There is not enough per-record physical-cause evidence to assess consistency.",
    }.get(consistency, "")
    if consistency_txt:
        lines.append(consistency_txt)

    lines.append(
        "The physical initiating cause remains unconfirmed from local COMTRADE evidence alone; "
        "external evidence (field inspection, lightning detection, protection engineer review) is needed to confirm it."
    )

    return "\n".join(line for line in lines if line is not None)
