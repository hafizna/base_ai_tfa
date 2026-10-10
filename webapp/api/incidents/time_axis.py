"""One time axis for all the records of an incident.

Each record's own time axis starts at its first sample, whose wall-clock time
is ``record_start_iso`` on the recorder's clock; the trigger sits
``trigger_offset_s`` into it. Placing records on one incident axis therefore
takes two steps:

- Records from one recorder (same station and device in the COMTRADE header)
  share its clock, so their own timestamps place them relative to each other:
  a fault file, the dead-time file of its reclose, and a re-fault file. The
  recorder of the first record attached at the incident's substation (else of
  the first record attached) is the incident's reference clock.
- Another recorder — the other line end, or a second device in the same bay —
  has its own clock, possibly in another time zone (a Qualitrol file stamped
  in UTC against a WIB relay record) or simply off. A fault starts at the same
  instant everywhere on the line, so that recorder is placed by lining up a
  fault both recorded, on compatible phases: the correction is the difference
  between the two fault starts, accepted only when the clocks agree to within
  a second once a whole time-zone offset is removed.

A recorder that shares no fault with the reference recorder keeps its own
clock, flagged as unverified, and so does a record that names no recorder.
Nothing here is guessed: a record without a timestamp stays unplaced.

The previous convention (``trigger_time_iso or record_start_iso`` as "the
record's time") mixed two anchors: the joined waveform added the left record's
span to a trigger-to-trigger gap, and the duplicate check compared samples
offset by the difference in pre-trigger length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from core.clock_offsets import MAX_CLOCK_REMAINDER_S, split_clock_offset

from .models import IncidentRecord

_NO_INCEPTION_METHODS = {"no_fault_evidence", "trigger_fallback", "insufficient_data", "dead_time_recording"}
# Two fault pairings describe the same clock offset when their corrections
# agree within a cycle (each recorder detects the inception within a few ms).
_CORRECTION_AGREEMENT_S = 0.02
# A recorder that shares no fault with the reference but reads at least this
# far from it is called out: likely a time-zone setting, though unconfirmed.
_UNVERIFIED_OFFSET_NOTE_S = 900.0


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _snapshot(record: IncidentRecord) -> dict[str, Any]:
    return record.canonical_snapshot or {}


def _axis_origin_s(record: IncidentRecord) -> float:
    """The record's own time-axis value at its first sample (normally 0)."""
    window = _snapshot(record).get("event_window") or {}
    return float(window.get("record_start_ms") or 0.0) / 1000.0


def own_start(record: IncidentRecord) -> Optional[datetime]:
    """Wall-clock time of the record's first sample on its recorder's clock."""
    start = _parse_iso(record.record_start_iso)
    if start is not None:
        return start
    trigger = _parse_iso(record.trigger_time_iso)
    if trigger is None:
        return None
    return trigger - timedelta(seconds=_trigger_offset_s(record))


def _trigger_offset_s(record: IncidentRecord) -> float:
    """Seconds from the record's first sample to its trigger."""
    if record.trigger_offset_s is not None:
        return float(record.trigger_offset_s)
    window = _snapshot(record).get("event_window") or {}
    return float(window.get("trigger_time_ms") or 0.0) / 1000.0 - _axis_origin_s(record)


def span_s(record: IncidentRecord) -> Optional[float]:
    """Record length in seconds, when the snapshot carries it."""
    meta = _snapshot(record).get("source_metadata") or {}
    value = meta.get("duration_s")
    return float(value) if value is not None else None


def fault_start_s(record: IncidentRecord) -> Optional[float]:
    """The record's first fault start on its own time axis (s): the analog
    trace's, else the event window's inception when it detected one. None for
    a record the no-fault gate rejected."""
    snapshot = _snapshot(record)
    if (snapshot.get("protection_interpretation") or {}).get("event_class") == "NO_FAULT_TRIGGER":
        return None
    summary = (snapshot.get("analog_trace") or {}).get("summary") or {}
    if summary.get("fault_start_ms") is not None:
        return float(summary["fault_start_ms"]) / 1000.0
    window = snapshot.get("event_window") or {}
    if window.get("inception_time_ms") is not None and window.get("method") not in _NO_INCEPTION_METHODS:
        return float(window["inception_time_ms"]) / 1000.0
    return None


def fault_phases(record: IncidentRecord) -> set[str]:
    """Phases of the record's first fault: the ones that carried fault
    current, else the ones whose voltage sagged (a weak-infeed end), else the
    event window's reading."""
    snapshot = _snapshot(record)
    summary = (snapshot.get("analog_trace") or {}).get("summary") or {}
    if summary.get("fault_start_ms") is not None:
        phases = summary.get("high_current_phases") or summary.get("sagged_phases") or []
        if phases:
            return set(phases)
    window = snapshot.get("event_window") or {}
    return set(window.get("faulted_phases") or [])


def _named(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text.upper() != "UNKNOWN":  # the parser's placeholder for an empty header field
            return text
    return ""


def station_of(record: IncidentRecord) -> str:
    """The substation named in the record's COMTRADE header ("" when none)."""
    meta = _snapshot(record).get("source_metadata") or {}
    return _named(meta.get("station_name"), record.station_name)


def same_station(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two station names denote one substation: case, spacing and a
    leading "GI"/"GITET"/"GIS" don't matter ("Bringin" is "GI BRINGIN")."""
    def key(name: Optional[str]) -> str:
        text = re.sub(r"\s+", " ", str(name or "").strip().upper())
        return re.sub(r"^(GITET|GIS|GI)\b\s*", "", text)
    return bool(key(a)) and key(a) == key(b)


def clock_group(record: IncidentRecord) -> Optional[str]:
    """Records from one recorder (the station and device named in the
    COMTRADE header) share its clock. None when the record names neither."""
    meta = _snapshot(record).get("source_metadata") or {}
    station = station_of(record)
    device = _named(meta.get("rec_dev_id"), record.relay_id)
    if not station and not device:
        return None
    return f"{station or '?'} | {device or '?'}"


@dataclass
class Placement:
    incident_record_id: str
    clock_group: str
    method: str                               # "reference_clock" | "fault_aligned" | "own_clock_unverified" | "unplaced"
    own_start: Optional[datetime]
    correction_s: Optional[float]             # added to the recorder clock to read the incident clock
    span_s: Optional[float]
    zone_offset_h: Optional[float] = None     # this recorder's clock minus the reference clock, whole time zones
    clock_offset_ms: Optional[float] = None   # this recorder's clock minus the reference clock beyond the zone, at the shared fault
    aligned_on: Optional[dict[str, Any]] = None

    @property
    def start(self) -> Optional[datetime]:
        if self.own_start is None or self.correction_s is None:
            return None
        return self.own_start + timedelta(seconds=self.correction_s)


@dataclass
class TimeAxis:
    placements: dict[str, Placement]
    reference_group: Optional[str]
    zero: Optional[datetime]
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def placement(self, record: IncidentRecord) -> Optional[Placement]:
        return self.placements.get(record.incident_record_id)

    def lane(self, record: IncidentRecord) -> str:
        """The recorder whose clock and sequence the record belongs to."""
        placement = self.placement(record)
        return placement.clock_group if placement else f"record {record.incident_record_id}"

    def start(self, record: IncidentRecord) -> Optional[datetime]:
        placement = self.placement(record)
        return placement.start if placement else None

    def end(self, record: IncidentRecord) -> Optional[datetime]:
        """The record's last sample on the incident clock."""
        placement = self.placement(record)
        if placement is None or placement.start is None or placement.span_s is None:
            return None
        return placement.start + timedelta(seconds=placement.span_s)

    def absolute(self, record: IncidentRecord, t_s: Optional[float]) -> Optional[datetime]:
        """Incident-clock time of an instant on the record's own time axis (s)."""
        start = self.start(record)
        if start is None or t_s is None:
            return None
        return start + timedelta(seconds=float(t_s) - _axis_origin_s(record))

    def trigger(self, record: IncidentRecord) -> Optional[datetime]:
        """The record's trigger on the incident clock: its own trigger
        timestamp when it carries one, else its first sample plus the
        trigger offset."""
        placement = self.placement(record)
        if placement is None or placement.correction_s is None:
            return None
        own_trigger = _parse_iso(record.trigger_time_iso)
        if own_trigger is not None:
            return own_trigger + timedelta(seconds=placement.correction_s)
        start = placement.start
        return start + timedelta(seconds=_trigger_offset_s(record)) if start is not None else None

    def offset_s(self, record: IncidentRecord) -> Optional[float]:
        """Seconds from the incident axis zero to the record's first sample."""
        start = self.start(record)
        if start is None or self.zero is None:
            return None
        return (start - self.zero).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        records = []
        for placement in self.placements.values():
            start = placement.start
            records.append({
                "incident_record_id": placement.incident_record_id,
                "clock_group": placement.clock_group,
                "method": placement.method,
                "correction_s": placement.correction_s,
                "zone_offset_h": placement.zone_offset_h,
                "clock_offset_ms": placement.clock_offset_ms,
                "aligned_on": placement.aligned_on,
                "start_iso": start.isoformat() if start else None,
                "offset_s": round((start - self.zero).total_seconds(), 6) if start and self.zero else None,
                "span_s": placement.span_s,
            })
        return {
            "reference_group": self.reference_group,
            "zero_iso": self.zero.isoformat() if self.zero else None,
            "records": records,
            "warnings": self.warnings,
        }


def _seconds(later: Optional[datetime], earlier: Optional[datetime]) -> Optional[float]:
    if later is None or earlier is None:
        return None
    try:
        return (later - earlier).total_seconds()
    except TypeError:  # one timestamp timezone-aware, the other naive
        return None


def _earliest(members: list[IncidentRecord], starts: dict[str, Optional[datetime]]) -> Optional[datetime]:
    timed = [starts[r.incident_record_id] for r in members if starts[r.incident_record_id] is not None]
    try:
        return min(timed) if timed else None
    except TypeError:
        return None


def _shared_fault_alignment(
    reference_faults: list[tuple[IncidentRecord, datetime]],
    candidates_faults: list[tuple[IncidentRecord, datetime]],
) -> Optional[dict[str, Any]]:
    """The clock correction that lines this recorder's faults up with the
    reference recorder's. A pairing counts only when the two faults are on
    compatible phases and the clocks then agree to within
    ``MAX_CLOCK_REMAINDER_S`` after a whole time-zone offset. Every pairing
    of the same two clocks gives the same correction, so when several
    pairings exist the correction most of them agree on wins — a wrong
    pairing (a later fault matched to an earlier one) stands alone."""
    candidates = []
    for ref_record, ref_fault in reference_faults:
        ref_phases = fault_phases(ref_record)
        for record, fault in candidates_faults:
            phases = fault_phases(record)
            if ref_phases and phases and not (ref_phases & phases):
                continue
            difference = _seconds(ref_fault, fault)
            if difference is None:
                continue
            remainder, zone = split_clock_offset(difference)
            if abs(remainder) > MAX_CLOCK_REMAINDER_S:
                continue
            candidates.append({
                "correction": difference, "remainder": remainder, "zone": zone,
                "reference_record_id": ref_record.incident_record_id,
                "record_id": record.incident_record_id,
            })
    if not candidates:
        return None

    def support(candidate: dict[str, Any]) -> int:
        return sum(1 for other in candidates if abs(other["correction"] - candidate["correction"]) <= _CORRECTION_AGREEMENT_S)

    best = max(candidates, key=lambda c: (support(c), -abs(c["remainder"])))
    best["agreeing_pairs"] = support(best)
    best["candidate_pairs"] = len(candidates)
    return best


def build_time_axis(records: list[IncidentRecord], home_station: Optional[str] = None) -> TimeAxis:
    """Place ``records`` on one incident clock. ``home_station`` (the
    incident's substation) picks the reference recorder: the one of the first
    record attached from that substation, else of the first record attached."""
    starts = {r.incident_record_id: own_start(r) for r in records}
    keys = {r.incident_record_id: clock_group(r) for r in records}
    # A record that names no recorder is a group of its own: nothing says
    # which clock it shares, so it is never shifted onto another one.
    labels = {rid: key or f"record {rid}" for rid, key in keys.items()}
    groups: dict[str, list[IncidentRecord]] = {}
    for record in records:
        groups.setdefault(labels[record.incident_record_id], []).append(record)

    timed = [r for r in records if starts[r.incident_record_id] is not None]
    placements: dict[str, Placement] = {}
    warnings: list[dict[str, Any]] = []
    if not timed:
        for record in records:
            placements[record.incident_record_id] = Placement(
                record.incident_record_id, labels[record.incident_record_id], "unplaced", None, None, span_s(record),
            )
        return TimeAxis(placements, None, None, warnings)

    # The reference clock — and the end the incident is told from: the
    # recorder of the first record attached at the incident's own substation.
    home = [r for r in timed if same_station(station_of(r), home_station)] if home_station else []
    reference_group = labels[min(home or timed, key=lambda r: r.sequence_index).incident_record_id]

    def fault_at(record: IncidentRecord) -> Optional[datetime]:
        start = starts[record.incident_record_id]
        t_fault = fault_start_s(record)
        if start is None or t_fault is None:
            return None
        return start + timedelta(seconds=t_fault - _axis_origin_s(record))

    def faults_of(members: list[IncidentRecord]) -> list[tuple[IncidentRecord, datetime]]:
        found = [(r, fault_at(r)) for r in members]
        return [(r, t) for r, t in found if t is not None]

    reference_faults = faults_of(groups[reference_group])

    for group, members in groups.items():
        group_timed = [r for r in members if starts[r.incident_record_id] is not None]
        if group == reference_group:
            for record in members:
                method = "reference_clock" if starts[record.incident_record_id] else "unplaced"
                placements[record.incident_record_id] = Placement(
                    record.incident_record_id, group, method, starts[record.incident_record_id],
                    0.0 if starts[record.incident_record_id] else None, span_s(record),
                )
            continue

        identified = any(keys[r.incident_record_id] is not None for r in members)
        # Line up a fault this recorder shares with the reference recorder.
        best = _shared_fault_alignment(reference_faults, faults_of(group_timed)) if identified else None

        for record in members:
            start = starts[record.incident_record_id]
            if start is None:
                placements[record.incident_record_id] = Placement(
                    record.incident_record_id, group, "unplaced", None, None, span_s(record),
                )
            elif best is not None:
                placements[record.incident_record_id] = Placement(
                    record.incident_record_id, group, "fault_aligned", start, best["correction"], span_s(record),
                    zone_offset_h=(-best["zone"] / 3600.0) if best["zone"] is not None else None,
                    clock_offset_ms=round(-best["remainder"] * 1000.0, 3),
                    aligned_on={
                        "reference_record_id": best["reference_record_id"],
                        "record_id": best["record_id"],
                        "agreeing_pairs": best["agreeing_pairs"],
                        "candidate_pairs": best["candidate_pairs"],
                    },
                )
            else:
                placements[record.incident_record_id] = Placement(
                    record.incident_record_id, group, "own_clock_unverified", start, 0.0, span_s(record),
                )
        if not group_timed:
            continue
        if not identified:
            warnings.append({
                "type": "CLOCK_UNIDENTIFIED",
                "clock_group": group,
                "description": (
                    "This record names no station or recording device, so it cannot be matched to a recorder "
                    "clock; its own timestamp is used as it is."
                ),
            })
        elif best is None:
            description = (
                f"Records from {group} share no detected fault with {reference_group}, so their clock "
                "could not be checked; their own timestamps are used as they are."
            )
            apart = _seconds(_earliest(group_timed, starts), _earliest(groups[reference_group], starts))
            if apart is not None and abs(apart) >= _UNVERIFIED_OFFSET_NOTE_S:
                description += (
                    f" They read {abs(apart) / 3600.0:.1f} h {'after' if apart > 0 else 'before'} the reference "
                    "recorder's records — check the recorders' time-zone settings."
                )
            warnings.append({"type": "CLOCK_NOT_VERIFIED", "clock_group": group, "description": description})
        elif best["candidate_pairs"] > best["agreeing_pairs"] and best["agreeing_pairs"] == 1:
            warnings.append({
                "type": "CLOCK_ALIGNMENT_AMBIGUOUS",
                "clock_group": group,
                "description": (
                    f"Records from {group} were lined up with {reference_group} on one shared fault, but other "
                    "fault pairings would give a different clock offset; review the record order."
                ),
                "requires_review": True,
            })
        if best is not None and best["zone"] is not None:
            warnings.append({
                "type": "TIME_ZONE_OFFSET_REMOVED",
                "clock_group": group,
                "zone_offset_h": -best["zone"] / 3600.0,
                "description": (
                    f"The clock of {group} reads {abs(best['zone']) / 3600.0:g} h "
                    f"{'behind' if best['zone'] > 0 else 'ahead of'} {reference_group}; aligned on the fault "
                    "both recorded."
                ),
            })

    placed = [p.start for p in placements.values() if p.start is not None]
    zero = None
    if placed:
        try:
            zero = min(placed)
        except TypeError:
            zero = None
            warnings.append({
                "type": "MIXED_TIMEZONE_AWARENESS",
                "description": "Some timestamps carry a time zone and others do not; the incident axis has no zero.",
            })
    return TimeAxis(placements, reference_group, zero, warnings)
