"""Joined waveform view — Stage 2.

Places two or more attached records' actual analog waveform samples on one
shared, incident-relative time axis, with the real gap between records
preserved (never filled/interpolated/fabricated) — so a trip/dead-time/
reclose sequence captured as separate COMTRADE files can be read as one
continuous story instead of switching between panels.

Each record is placed by its first sample on the incident time axis
(``time_axis``) — the same placement the relationship and timeline views use
— so a joined waveform can never disagree with them about when a record ran.
The gap shown between two records is the stretch from the left record's last
sample to the right record's first.

The joined trace is one recorder's: the episode's own end, whose records
follow each other in time. Recordings of the same episode by other recorders
(the far line end, a second device in the bay) run alongside it, so they are
returned separately in ``other_lanes``, on the same time axis, rather than
drawn over it.

Trust gate: a record is only placed at a *measured* offset when it and the
episode's first record both carry absolute time. Otherwise it is still
returned (so the UI isn't empty), but flagged ``gap_precision:
"assumed_back_to_back"`` and placed immediately after the previous record's
last sample — never claiming a duration we didn't measure. This mirrors
``alignment.py``'s philosophy: degrade the confidence label rather than
invent a number.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np

from core.line_selection import scope_payload
from ..storage import load_analysis
from .models import FaultEpisode, IncidentRecord
from .time_axis import TimeAxis, build_time_axis, station_of

# A gap longer than this is still joined (never refused), but flagged so the
# UI can render it as a visibly compressed/labeled break rather than an
# actual-scale blank stretch that would dwarf the fault waveforms either
# side of it.
LONG_GAP_DISPLAY_THRESHOLD_S = 2.0


def _load_line_payload(analysis_id: str) -> Optional[dict]:
    """Stored payload restricted to the record's disturbed line — channels are
    keyed by canonical name below, so a DFR file carrying two lines would
    otherwise show whichever line's IA/VA happened to be listed last."""
    payload = load_analysis(analysis_id)
    return scope_payload(payload) if payload is not None else None


def _span_s(payload: dict) -> float:
    time = payload.get("time") or []
    return float(time[-1]) - float(time[0]) if len(time) >= 2 else 0.0


def _channels(payload: dict, t_offset_s: float) -> dict[str, dict[str, list[float]]]:
    time_arr = np.asarray(payload.get("time") or [], dtype=float)
    t0 = float(time_arr[0]) if len(time_arr) else 0.0
    rel_t = (time_arr - t0) + t_offset_s
    channels: dict[str, dict[str, list[float]]] = {}
    for ch in payload.get("analog_channels", []):
        canon = ch.get("canonical_name")
        if not canon:
            continue
        samples = ch.get("samples") or []
        if len(samples) != len(rel_t):
            continue
        channels[canon] = {"t": rel_t.tolist(), "values": list(samples)}
    return channels


def _offset_s(axis: TimeAxis, record: IncidentRecord, origin: Optional[datetime]) -> Optional[float]:
    start = axis.start(record)
    if origin is None or start is None:
        return None
    try:
        return (start - origin).total_seconds()
    except TypeError:  # one timestamp timezone-aware, the other naive
        return None


def _not_joined(episode: FaultEpisode, reason: str, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "can_join": False,
        "reason": reason,
        "segments": [],
        "gap_ranges": [],
        "other_lanes": [],
        "warnings": warnings,
    }


def build_joined_waveform(
    episode: FaultEpisode,
    records_by_id: dict[str, IncidentRecord],
    axis: Optional[TimeAxis] = None,
) -> dict[str, Any]:
    """Build a joined, incident-relative waveform view for one episode.

    Returns a dict (not a dataclass — this is a read/derive-only view, never
    persisted) shaped as:
      {
        "episode_id": ...,
        "can_join": bool,
        "reason": str | None,           # set when can_join is False
        "warnings": [...],
        "segments": [                    # the episode's own recorder, in join order
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
        "other_lanes": [                 # other recorders' recordings of the episode, same axis
          {"recorder": str, "station": str, "clock_method": str | None,
           "segments": [{"incident_record_id", "source_filename", "t_offset_s", "channels"}]},
        ],
      }

    Never interpolates or fabricates samples inside a gap — ``gap_ranges``
    exists precisely so the UI can draw that stretch as an explicit "no
    data" break instead of a misleadingly continuous line.
    """
    member_ids = list(episode.member_record_ids)
    ordered = [records_by_id[rid] for rid in member_ids if rid in records_by_id]
    if len(ordered) < 1:
        return _not_joined(episode, "no_records", [])

    # The episode's own recorder is its first member's (episodes list the
    # reference recorder's records first).
    axis = axis or build_time_axis(list(records_by_id.values()))
    main_lane = axis.lane(ordered[0])
    main = [r for r in ordered if axis.lane(r) == main_lane]
    others = [r for r in ordered if axis.lane(r) != main_lane]

    warnings: list[dict[str, Any]] = []
    payloads: dict[str, dict] = {}
    for rec in main:
        payload = _load_line_payload(rec.analysis_id)
        if payload is None:
            return _not_joined(episode, "analysis_expired_or_missing",
                               [{"type": "MISSING_ANALYSIS", "incident_record_id": rec.incident_record_id}])
        payloads[rec.incident_record_id] = payload

    # Anchor: the first record's first sample is t = 0; every record with an
    # absolute time is placed by its own first sample on the incident time
    # axis. Without one, a record goes right after the previous record's last
    # sample — relative placement inside each record is still exact.
    origin = axis.start(main[0])
    t_offsets_s: list[float] = [0.0]
    gap_infos: list[Optional[dict[str, Any]]] = [None]  # index 0 has no "gap before" it

    for i in range(1, len(main)):
        left, right = main[i - 1], main[i]
        left_end_s = t_offsets_s[-1] + _span_s(payloads[left.incident_record_id])
        offset_s = _offset_s(axis, right, origin)

        if offset_s is not None:
            precision = "measured"
            # Negative when the two recordings overlap: no gap to shade.
            gap_s = max(0.0, offset_s - left_end_s)
        else:
            # No absolute-time evidence for this record — place it
            # immediately after the previous one's last sample rather than
            # guessing a duration.
            offset_s = left_end_s
            gap_s = 0.0
            precision = "assumed_back_to_back"
            warnings.append({
                "type": "GAP_NOT_MEASURED",
                "left_incident_record_id": left.incident_record_id,
                "right_incident_record_id": right.incident_record_id,
                "description": "No absolute-time evidence for this pair; placed back-to-back with no implied dead-time duration.",
            })

        t_offsets_s.append(offset_s)
        gap_infos.append({"gap_seconds": round(gap_s, 3), "precision": precision})

    segments: list[dict[str, Any]] = []
    gap_ranges: list[dict[str, Any]] = []
    for i, rec in enumerate(main):
        gap_info = gap_infos[i]
        segments.append({
            "incident_record_id": rec.incident_record_id,
            "source_filename": rec.source_filename,
            "t_offset_s": round(t_offsets_s[i], 6),
            "gap_precision": gap_info["precision"] if gap_info else None,
            "gap_seconds": gap_info["gap_seconds"] if gap_info else None,
            "channels": _channels(payloads[rec.incident_record_id], t_offsets_s[i]),
        })

        if gap_info is not None and gap_info["gap_seconds"] > 0:
            gap_start = t_offsets_s[i] - gap_info["gap_seconds"]
            gap_ranges.append({
                "start_s": round(gap_start, 6),
                "end_s": round(t_offsets_s[i], 6),
                "precision": gap_info["precision"],
                "long_gap": gap_info["gap_seconds"] > LONG_GAP_DISPLAY_THRESHOLD_S,
            })

    other_lanes: list[dict[str, Any]] = []
    lanes: dict[str, list[IncidentRecord]] = {}
    for rec in others:
        lanes.setdefault(axis.lane(rec), []).append(rec)
    for lane, recs in lanes.items():
        lane_segments = []
        for rec in recs:
            offset_s = _offset_s(axis, rec, origin)
            payload = _load_line_payload(rec.analysis_id)
            if offset_s is None or payload is None:
                warnings.append({
                    "type": "OTHER_RECORDER_NOT_PLACED",
                    "incident_record_id": rec.incident_record_id,
                    "description": (
                        "This recording could not be placed beside the episode's records: "
                        + ("it has no absolute time." if offset_s is None else "its analysis has expired.")
                    ),
                })
                continue
            lane_segments.append({
                "incident_record_id": rec.incident_record_id,
                "source_filename": rec.source_filename,
                "t_offset_s": round(offset_s, 6),
                "channels": _channels(payload, offset_s),
            })
        if lane_segments:
            placement = axis.placement(recs[0])
            other_lanes.append({
                "recorder": lane,
                "station": station_of(recs[0]),
                "clock_method": placement.method if placement else None,
                "segments": lane_segments,
            })

    return {
        "episode_id": episode.episode_id,
        "can_join": True,
        "reason": None,
        "segments": segments,
        "gap_ranges": gap_ranges,
        "other_lanes": other_lanes,
        "warnings": warnings,
    }
