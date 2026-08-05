"""Joined waveform view — Stage 2.

Places two or more attached records' actual analog waveform samples on one
shared, incident-relative time axis, with the real gap between records
preserved (never filled/interpolated/fabricated) — so a trip/dead-time/
reclose sequence captured as separate COMTRADE files can be read as one
continuous story instead of switching between panels.

This deliberately reuses the SAME absolute-time-from-relative-time
conversion already used by ``relationships.py::_waveform_similarity`` (and
the same anchor convention as ``timeline.py``), so a joined waveform's time
axis can never disagree with what the relationship/timeline views already
show for the same pair of records.

Trust gate: a pair is only joined with a *precise* gap when both records
carry absolute time. When one or both records lack it, the pair is still
returned (so the UI isn't empty), but flagged ``gap_precision: "unknown"``
and the join falls back to placing the right record immediately after the
left one's last sample — never claiming a duration we didn't measure. This
mirrors ``alignment.py``'s philosophy: degrade the confidence label rather
than invent a number.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np

from ..storage import load_analysis
from .models import FaultEpisode, IncidentRecord, RecordRelationship

# A gap longer than this is still joined (never refused), but flagged so the
# UI can render it as a visibly compressed/labeled break rather than an
# actual-scale blank stretch that would dwarf the fault waveforms either
# side of it.
LONG_GAP_DISPLAY_THRESHOLD_S = 2.0


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _record_time(record: IncidentRecord) -> Optional[datetime]:
    return _parse_iso(record.trigger_time_iso) or _parse_iso(record.record_start_iso)


def _relationship_for_pair(
    relationships: list[RecordRelationship], left_id: str, right_id: str
) -> Optional[RecordRelationship]:
    for rel in relationships:
        if rel.left_record_id == left_id and rel.right_record_id == right_id:
            return rel
    return None


def build_joined_waveform(
    episode: FaultEpisode,
    records_by_id: dict[str, IncidentRecord],
    relationships: list[RecordRelationship],
) -> dict[str, Any]:
    """Build a joined, incident-relative waveform view for one episode.

    Returns a dict (not a dataclass — this is a read/derive-only view, never
    persisted) shaped as:
      {
        "episode_id": ...,
        "can_join": bool,
        "reason": str | None,           # set when can_join is False
        "warnings": [...],
        "segments": [                    # one per record, in join order
          {
            "incident_record_id": ...,
            "source_filename": ...,
            "t_offset_s": float,         # this record's t=0 position on the
                                          # joined incident-relative axis
            "gap_precision": "measured" | "assumed_back_to_back" | None,  # gap BEFORE this segment (None for the first)
            "gap_seconds": float | None,
            "channels": {canonical_name: {"t": [...], "values": [...]}},
          },
          ...
        ],
        "gap_ranges": [                  # incident-relative [start,end] per gap, for UI shading
          {"start_s": float, "end_s": float, "precision": "measured"|"assumed_back_to_back"},
        ],
      }

    Never interpolates or fabricates samples inside a gap — ``gap_ranges``
    exists precisely so the UI can draw that stretch as an explicit "no
    data" break instead of a misleadingly continuous line.
    """
    member_ids = list(episode.member_record_ids)
    ordered = [records_by_id[rid] for rid in member_ids if rid in records_by_id]
    if len(ordered) < 1:
        return {"episode_id": episode.episode_id, "can_join": False, "reason": "no_records", "segments": [], "gap_ranges": [], "warnings": []}

    if len(ordered) == 1:
        return _build_single_record(episode, ordered[0])

    warnings: list[dict[str, Any]] = []
    payloads: dict[str, dict] = {}
    for rec in ordered:
        payload = load_analysis(rec.analysis_id)
        if payload is None:
            return {
                "episode_id": episode.episode_id,
                "can_join": False,
                "reason": "analysis_expired_or_missing",
                "segments": [],
                "gap_ranges": [],
                "warnings": [{"type": "MISSING_ANALYSIS", "incident_record_id": rec.incident_record_id}],
            }
        payloads[rec.incident_record_id] = payload

    # Anchor: first record's absolute time if available, else its samples
    # simply start at t_offset_s = 0 with everything after it placed
    # relative to measured/assumed gaps — there's no absolute reference to
    # convert to, but relative placement is still exact.
    t_offsets_s: list[float] = [0.0]
    gap_infos: list[dict[str, Any]] = [None]  # index 0 has no "gap before" it

    for i in range(1, len(ordered)):
        left, right = ordered[i - 1], ordered[i]
        rel = _relationship_for_pair(relationships, left.incident_record_id, right.incident_record_id)
        gap_s = rel.metrics.get("gap_seconds") if rel and isinstance(rel.metrics, dict) else None

        if gap_s is None:
            t_left = _record_time(left)
            t_right = _record_time(right)
            if t_left is not None and t_right is not None:
                gap_s = (t_right - t_left).total_seconds()

        left_time = np.asarray(payloads[left.incident_record_id].get("time") or [], dtype=float)
        left_span_s = float(left_time[-1] - left_time[0]) if len(left_time) >= 2 else 0.0

        if gap_s is not None and gap_s >= 0:
            precision = "measured"
        else:
            # No absolute-time evidence for this pair, or a nonsensical
            # (negative) gap — place the right record immediately after the
            # left one's last sample rather than guessing a duration.
            gap_s = 0.0
            precision = "assumed_back_to_back"
            warnings.append({
                "type": "GAP_NOT_MEASURED",
                "left_incident_record_id": left.incident_record_id,
                "right_incident_record_id": right.incident_record_id,
                "description": "No absolute-time evidence for this pair; placed back-to-back with no implied dead-time duration.",
            })

        t_offsets_s.append(t_offsets_s[-1] + left_span_s + gap_s)
        gap_infos.append({"gap_seconds": round(gap_s, 3), "precision": precision})

    segments: list[dict[str, Any]] = []
    gap_ranges: list[dict[str, Any]] = []
    for i, rec in enumerate(ordered):
        payload = payloads[rec.incident_record_id]
        time_arr = np.asarray(payload.get("time") or [], dtype=float)
        t0 = float(time_arr[0]) if len(time_arr) else 0.0
        rel_t = (time_arr - t0) + t_offsets_s[i]

        channels: dict[str, dict[str, list[float]]] = {}
        for ch in payload.get("analog_channels", []):
            canon = ch.get("canonical_name")
            if not canon:
                continue
            samples = ch.get("samples") or []
            if len(samples) != len(rel_t):
                continue
            channels[canon] = {"t": rel_t.tolist(), "values": list(samples)}

        gap_info = gap_infos[i]
        segments.append({
            "incident_record_id": rec.incident_record_id,
            "source_filename": rec.source_filename,
            "t_offset_s": round(t_offsets_s[i], 6),
            "gap_precision": gap_info["precision"] if gap_info else None,
            "gap_seconds": gap_info["gap_seconds"] if gap_info else None,
            "channels": channels,
        })

        if gap_info is not None and gap_info["gap_seconds"] > 0:
            gap_start = t_offsets_s[i] - gap_info["gap_seconds"]
            gap_ranges.append({
                "start_s": round(gap_start, 6),
                "end_s": round(t_offsets_s[i], 6),
                "precision": gap_info["precision"],
                "long_gap": gap_info["gap_seconds"] > LONG_GAP_DISPLAY_THRESHOLD_S,
            })

    return {
        "episode_id": episode.episode_id,
        "can_join": True,
        "reason": None,
        "segments": segments,
        "gap_ranges": gap_ranges,
        "warnings": warnings,
    }


def _build_single_record(episode: FaultEpisode, rec: IncidentRecord) -> dict[str, Any]:
    payload = load_analysis(rec.analysis_id)
    if payload is None:
        return {
            "episode_id": episode.episode_id,
            "can_join": False,
            "reason": "analysis_expired_or_missing",
            "segments": [],
            "gap_ranges": [],
            "warnings": [{"type": "MISSING_ANALYSIS", "incident_record_id": rec.incident_record_id}],
        }
    time_arr = np.asarray(payload.get("time") or [], dtype=float)
    t0 = float(time_arr[0]) if len(time_arr) else 0.0
    rel_t = (time_arr - t0)
    channels: dict[str, dict[str, list[float]]] = {}
    for ch in payload.get("analog_channels", []):
        canon = ch.get("canonical_name")
        if not canon:
            continue
        samples = ch.get("samples") or []
        if len(samples) != len(rel_t):
            continue
        channels[canon] = {"t": rel_t.tolist(), "values": list(samples)}

    return {
        "episode_id": episode.episode_id,
        "can_join": True,
        "reason": None,
        "segments": [{
            "incident_record_id": rec.incident_record_id,
            "source_filename": rec.source_filename,
            "t_offset_s": 0.0,
            "gap_precision": None,
            "gap_seconds": None,
            "channels": channels,
        }],
        "gap_ranges": [],
        "warnings": [],
    }
