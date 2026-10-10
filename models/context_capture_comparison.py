"""Compare complete and partitioned captures of the same measured event.

Partitions are a regression experiment, not additional training examples.
Absolute sample timestamps are preserved; no waveform is interpolated.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np

from webapp.api.record_analysis import build_record_analysis
from webapp.api.incidents.models import IncidentRecord
from webapp.api.incidents.time_axis import build_time_axis
from webapp.api.incidents.relationships import build_relationships
from webapp.api.incidents.episodes import group_episodes
from webapp.api.incidents.alignment import assess_alignment
from webapp.api.incidents.timeline import build_timeline
from webapp.api.incidents.reconstruction import _protection_sequence_interpretation


def partition(payload, cuts_s):
    time = np.asarray(payload["time"])
    if not len(time) or cuts_s != sorted(set(cuts_s)) or any(cut <= time[0] or cut >= time[-1] for cut in cuts_s):
        raise ValueError("Cuts must be unique, increasing, and strictly inside the recording")
    edges = [0, *[int(np.searchsorted(time, cut)) for cut in cuts_s], len(time)]
    pieces = []
    start = datetime.fromisoformat(payload.get("start_time_iso") or "2026-01-01T00:00:00")
    for left, right in zip(edges, edges[1:]):
        piece = deepcopy(payload)
        offset = float(time[left] - time[0])
        piece["time"] = (time[left:right] - time[left]).tolist()
        piece["start_time_iso"] = (start + timedelta(seconds=offset)).isoformat()
        piece["trigger_time_iso"] = piece["start_time_iso"]
        piece["trigger_offset_s"] = piece["trigger_time"] = 0.0
        piece["total_samples"] = right - left
        for key in ("analog_channels", "status_channels"):
            for ch in piece.get(key, []):
                ch["samples"] = ch["samples"][left:right]
        pieces.append(piece)
    return pieces


def incident_view(payloads):
    records = []
    for index, payload in enumerate(payloads):
        analysis = build_record_analysis(str(index), payload)
        records.append(IncidentRecord(
            incident_record_id=str(index), incident_id="comparison", analysis_id=str(index),
            station_name=payload.get("station_name"), relay_id="same-recorder", protection_type="21",
            record_start_iso=payload.get("start_time_iso"), trigger_time_iso=payload.get("trigger_time_iso"),
            trigger_offset_s=payload.get("trigger_offset_s", 0), sequence_index=index,
            canonical_snapshot=analysis.to_dict()))
    axis = build_time_axis(records)
    alignment = assess_alignment(records, axis)
    ids = iter(str(i) for i in range(10000))
    relationships = build_relationships("comparison", records, alignment, lambda: next(ids), axis)
    episodes = group_episodes("comparison", records, relationships, alignment.record_order, lambda: next(ids), axis=axis)
    return {"records": [r.canonical_snapshot["event_window"] for r in records],
            "relationships": [r.relationship_type for r in relationships],
            "episodes": [e.to_dict() for e in episodes],
            "timeline": [event.to_dict() for event in build_timeline("comparison", records, alignment, lambda: next(ids), axis)],
            "interpretation": _protection_sequence_interpretation(episodes, relationships, records)}


def compare(payload, cuts_s):
    # A full recording without an absolute clock can be given an arbitrary
    # common origin here: all partition offsets remain exact by construction.
    payload = deepcopy(payload)
    payload.setdefault("start_time_iso", "2026-01-01T00:00:00")
    payload.setdefault("trigger_time_iso", payload["start_time_iso"])
    return {"complete": incident_view([payload]), "partitioned": incident_view(partition(payload, cuts_s)),
            "cuts_s": cuts_s, "purpose": "regression_only_not_training_samples"}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from core.comtrade_parser import parse_comtrade
    from webapp.api.record_payload import record_to_payload
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfg", type=Path)
    parser.add_argument("--cuts", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/training-runs/capture-comparison.json"))
    args = parser.parse_args()
    record = parse_comtrade(str(args.cfg))
    if record is None:
        raise SystemExit("COMTRADE reader could not parse this recording")
    result = compare(record_to_payload(record), args.cuts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key]["interpretation"] for key in ("complete", "partitioned")}, indent=2))
