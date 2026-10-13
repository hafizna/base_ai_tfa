"""Run fault section estimation for one incident described by a case file.

A case file (JSON, see ``config/fault_section_cases``) says what the records
are, not what happened:

- ``title`` and ``station`` (the substation of the incident folder);
- ``recorders``: each relay/recorder with its ``folder`` and files — COMTRADE,
  MiCOM ``.evt``, and ``manual_events`` transcribed from a ``screenshot`` —
  one clock per recorder;
- the bay template: ``diameter`` (1½-breaker, ``diameter.build_diameter``) or
  ``transformer`` (``transformer.build_transformer``);
- optionally the mapping: ``currents`` (which channels measure which CT; a
  neutral CT is a single channel), ``breakers`` / ``zones`` /
  ``direction_signals`` (which status channels or event labels report a
  breaker open, a zone operating, a distance element deciding forward) and
  ``ct_ratings_a``. For a diameter, whatever mapping is left out is inferred
  from the records (``mapping``), each decision listed in the report; a
  transformer case still gives its mapping.

Clock offsets, episodes, intervals and every observation are derived from the
records; nothing about the fault location is written in the case.

Usage::

    python -m core.fault_section.case config/fault_section_cases/<case>.json --root <incident folder> [--json out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.comtrade_parser import ComtradeRecord, parse_comtrade
from core.micom_evt_parser import parse_micom_evt, signal_timelines

from .diameter import build_diameter
from .estimate import Estimate, estimate
from .mapping import Decision, Mapping, SourceRecorder, infer_mapping, phase_sets
from .transformer import build_transformer
from .observe import (
    ClockMatch,
    CurrentTrace,
    SignalTrace,
    breaker_changes,
    current_observations,
    dedupe,
    direction_observations,
    episodes,
    intervals,
    load_level,
    match_clock,
    onsets,
    thresholds,
    zone_observations,
)

_MAPPING_KEYS = ("currents", "breakers", "zones", "direction_signals")


@dataclass
class _Recorder:
    id: str
    label: str
    folder: str
    files: tuple[str, ...]
    weak: bool
    comtrade: list[ComtradeRecord] = field(default_factory=list)
    timelines: dict[str, list[tuple[datetime, bool]]] = field(default_factory=dict)
    timeline_span: Optional[tuple[datetime, datetime]] = None
    transcribed: set[str] = field(default_factory=set)      # signals read off a screenshot
    has_event_log: bool = False
    offset_s: float = 0.0
    alignment: str = ""


@dataclass
class EpisodeResult:
    start_s: float
    end_s: float
    estimate: Estimate


@dataclass
class CaseResult:
    title: str
    epoch: datetime
    recorders: list[_Recorder]
    loads_a: dict[str, float]
    limits_a: dict[str, tuple[float, float]]
    episodes: list[EpisodeResult]
    notes: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    mapping_choice: str = ""

    def clock(self, seconds: float) -> str:
        return (self.epoch + timedelta(seconds=seconds)).strftime("%H:%M:%S.%f")[:-3]

    def report_lines(self) -> list[str]:
        lines = [self.title]
        if self.decisions:
            lines += ["", "Pemetaan otomatis dari rekaman:"]
            for d in self.decisions:
                lines.append(f"  {'•' if d.certain else '?'} [{d.rule}] {d.subject} — {d.evidence}")
        if self.mapping_choice:
            lines.append(f"  ⇒ {self.mapping_choice}")
        lines += ["", "Jam rekaman (acuan = offset 0):"]
        for rec in self.recorders:
            lines.append(f"  {rec.label:<42} {rec.offset_s:+.3f} s  {rec.alignment}")
        lines += ["", "Ambang arus per CT (load → membawa arus gangguan / tanpa arus):"]
        for ct, (carries, none) in self.limits_a.items():
            lines.append(f"  {ct:<20} load {self.loads_a.get(ct, 0):7,.0f} A → > {carries:7,.0f} A / < {none:5,.0f} A")
        lines.extend(f"  ! {note}" for note in self.notes)
        for n, ep in enumerate(self.episodes, 1):
            lines.append("")
            lines.append(f"=== Episode {n}: {self.clock(ep.start_s)} – {self.clock(ep.end_s)} "
                         f"({(ep.end_s - ep.start_s) * 1000:.0f} ms)")
            for interval in ep.estimate.intervals:
                lines.append(f"  {interval.id:<3} {self.clock(interval.start_s)}–{self.clock(interval.end_s)}  {interval.label}")
            lines.append("")
            lines.extend(ep.estimate.report_lines())
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "epoch": self.epoch.isoformat(),
            "recorders": [{"id": r.id, "label": r.label, "offset_s": r.offset_s, "alignment": r.alignment}
                          for r in self.recorders],
            "mapping": [{"subject": d.subject, "rule": d.rule, "evidence": d.evidence, "certain": d.certain}
                        for d in self.decisions],
            "mapping_choice": self.mapping_choice,
            "limits_a": {ct: list(v) for ct, v in self.limits_a.items()},
            "notes": self.notes,
            "episodes": [{"start_s": e.start_s, "end_s": e.end_s, **e.estimate.to_dict()} for e in self.episodes],
        }


# --- loading ------------------------------------------------------------------------

def _record_start(rec: ComtradeRecord) -> datetime:
    if not rec.start_time_iso:
        raise ValueError(f"{Path(rec.cfg_path).name}: no absolute start time")
    return datetime.fromisoformat(rec.start_time_iso)


def _seconds(epoch: datetime, when: datetime) -> float:
    return (when - epoch).total_seconds()


def _record_time(epoch: datetime, record: ComtradeRecord) -> np.ndarray:
    return _seconds(epoch, _record_start(record)) + (record.time - record.time[0])


def _channel(rec: ComtradeRecord, name: str, analog: bool):
    for ch in rec.analog_channels if analog else rec.status_channels:
        if ch.name.strip() == name:
            return ch
    return None


_GENERIC_FOLDERS = {"DR", "DFR", "EVENTS", "EVENT", "COMTRADE", "SETTINGS", "PSL", "RECORDS"}


def _folder_label(folder: str) -> str:
    """A recorder's folder without the export subfolders ("…/DR", "…/Events")."""
    parts = [p for p in folder.replace("\\", "/").split("/") if p and p.upper() not in _GENERIC_FOLDERS]
    return " / ".join(p.strip(" _") for p in parts) or folder


def _load_recorders(case: dict[str, Any], root: Path) -> list[_Recorder]:
    recorders = []
    for spec in case["recorders"]:
        paths = list(spec.get("comtrade") or []) + list(spec.get("micom_evt") or [])
        folder = spec.get("folder") or (Path(paths[0]).parent.as_posix() if paths else spec["id"])
        files = tuple(Path(p).name for p in paths) + ((spec["screenshot"],) if spec.get("screenshot") else ())
        rec = _Recorder(spec["id"], spec.get("label") or _folder_label(folder), folder, files, bool(spec.get("weak")))
        for rel in spec.get("comtrade") or []:
            parsed = parse_comtrade(str(root / rel))
            if parsed is None or not len(parsed.time):
                raise ValueError(f"{rec.id}: cannot read {rel}")
            rec.comtrade.append(parsed)
        events = []
        for rel in spec.get("micom_evt") or []:
            events.extend(parse_micom_evt(root / rel))
        events.sort(key=lambda e: e.time)
        if events:
            rec.timelines = signal_timelines(events)
            rec.timeline_span = (events[0].time, events[-1].time)
            rec.has_event_log = True
        manual = spec.get("manual_events") or []
        if manual:
            # Each transcribed entry is a transition, so a signal's state
            # before its first entry is the opposite of that entry.
            times = [datetime.fromisoformat(e["time"]) for e in manual]
            first, last = min(times), max(times)
            if rec.timeline_span:
                first, last = min(first, rec.timeline_span[0]), max(last, rec.timeline_span[1])
            rec.timeline_span = (first, last)
            for event, when in sorted(zip(manual, times), key=lambda p: p[1]):
                timeline = rec.timelines.setdefault(event["signal"], [])
                if not timeline:
                    timeline.append((first, not bool(event["state"])))
                timeline.append((when, bool(event["state"])))
                rec.transcribed.add(event["signal"])
        recorders.append(rec)
    return recorders


def _sources(recorders: list[_Recorder], epoch: datetime) -> list[SourceRecorder]:
    """Every three-phase current set of every recorder, for ``infer_mapping``."""
    sources = []
    for rec in recorders:
        current_sets: dict[str, list[CurrentTrace]] = {}
        channels: dict[str, dict[str, str]] = {}
        voltages: dict[str, dict[str, str]] = {}
        ratings: dict[str, float] = {}
        for record in rec.comtrade:
            t = _record_time(epoch, record)
            for key, chans in phase_sets(c.name for c in record.analog_channels).items():
                if key.startswith("V:"):
                    voltages.setdefault(key, chans)
                    continue
                analog = [_channel(record, chans[p], analog=True) for p in "ABC"]
                current_sets.setdefault(key, []).append(CurrentTrace(
                    ct=key, source=f"{rec.label}: {'/'.join(chans[p] for p in 'ABC')}", recorder=rec.id, t=t,
                    phases=tuple(np.asarray(a.samples, float) for a in analog)))
                channels[key] = chans
                primary = max((a.ct_primary for a in analog), default=0.0)
                if primary > 10:
                    ratings[key] = primary
        signals = {c.name.strip() for record in rec.comtrade for c in record.status_channels} | set(rec.timelines)
        sources.append(SourceRecorder(
            id=rec.id, folder=rec.folder, files=rec.files,
            station_names=tuple(r.station_name for r in rec.comtrade),
            current_sets=current_sets, voltage_sets=voltages, current_channels=channels,
            signals=tuple(sorted(signals)), has_event_log=rec.has_event_log, ratings=ratings))
    return sources


def _current_traces(case: dict[str, Any], recorders: dict[str, _Recorder], epoch: datetime) -> list[CurrentTrace]:
    traces = []
    for spec in case.get("currents") or []:
        rec = recorders[spec["recorder"]]
        for record in rec.comtrade:
            chans = [_channel(record, name, analog=True) for name in spec["channels"]]
            if any(c is None for c in chans):
                continue
            volts = [_channel(record, name, analog=True) for name in spec.get("voltages") or []]
            volts = volts if volts and all(v is not None for v in volts) else []
            traces.append(CurrentTrace(
                ct=spec["ct"], recorder=rec.id, t=_record_time(epoch, record),
                source=f"{rec.label}: {'/'.join(spec['channels'])}",
                phases=tuple(np.asarray(c.samples, float) for c in chans),
                voltages=tuple(_volts(v) for v in volts), forward=spec.get("forward", "") if volts else ""))
    return traces


def _volts(channel) -> np.ndarray:
    samples = np.asarray(channel.samples, float)
    return samples * 1000.0 if channel.unit.lower() == "kv" else samples


def _combine_edges(series: list[list[tuple[float, bool]]], combine) -> list[tuple[float, bool]]:
    """Edges of ``combine(states)`` over several channels' edge lists."""
    states = [s[0][1] if s else False for s in series]
    merged = sorted((t, idx, v) for idx, s in enumerate(series) for t, v in s)
    out: list[tuple[float, bool]] = []
    for t, idx, v in merged:
        states[idx] = v
        value = combine(states)
        if not out or out[-1][1] != value:
            out.append((t, value))
    return out


def _signal_traces(case: dict[str, Any], recorders: dict[str, _Recorder], epoch: datetime) -> list[SignalTrace]:
    traces = []
    mappings = ([("breaker", m) for m in case.get("breakers") or []]
                + [("zone", m) for m in case.get("zones") or []]
                + [("direction", m) for m in case.get("direction_signals") or []])
    for kind, spec in mappings:
        rec = recorders[spec["recorder"]]
        subject = spec.get("breaker") or spec.get("zone") or spec.get("ct")
        open_when = bool(spec.get("open_when", True))

        def as_subject(states, _kind=kind, _open_when=open_when):
            # A breaker conducts until every pole reads open; a zone operates
            # when any of its operate signals is high.
            if _kind == "breaker":
                return all(s == _open_when for s in states)
            return any(states)

        common = dict(subject=subject, kind=kind, recorder=rec.id, weak=bool(spec.get("weak", rec.weak)),
                      value=spec.get("value", ""), forward=spec.get("forward", ""))
        if spec.get("never_operated"):
            if rec.timeline_span:
                span = (_seconds(epoch, rec.timeline_span[0]), _seconds(epoch, rec.timeline_span[1]))
                traces.append(SignalTrace(source=f"{rec.label}: tidak ada trip tercatat", edges=[(span[0], False)],
                                          span=span, **common))
            continue
        names = spec.get("channels") or []
        for record in rec.comtrade:
            chans = [c for c in (_channel(record, name, analog=False) for name in names) if c is not None]
            if not chans:
                continue
            t = _record_time(epoch, record)
            series = []
            for c in chans:
                x = np.asarray(c.samples).astype(bool)
                idx = np.flatnonzero(np.diff(x.astype(int))) + 1
                series.append([(float(t[0]), bool(x[0]))] + [(float(t[i]), bool(x[i])) for i in idx])
            traces.append(SignalTrace(source=f"{rec.label}: {'/'.join(c.name.strip() for c in chans)}",
                                      edges=_combine_edges(series, as_subject),
                                      span=(float(t[0]), float(t[-1])), **common))
        present = [name for name in names if name in rec.timelines]
        if present and rec.timeline_span:
            series = [[(_seconds(epoch, when), state) for when, state in rec.timelines[name]] for name in present]
            span = (_seconds(epoch, rec.timeline_span[0]), _seconds(epoch, rec.timeline_span[1]))
            transcribed = any(name in rec.transcribed for name in present)
            traces.append(SignalTrace(
                source=f"{rec.label}: {'/'.join(present)} ({'screenshot' if transcribed else 'event'})",
                edges=_combine_edges(series, as_subject), span=span,
                **{**common, "weak": common["weak"] or transcribed}))
    return traces


# --- alignment ---------------------------------------------------------------------

def _align(case: dict[str, Any], recorders: list[_Recorder], traces: list[CurrentTrace],
           signals: list[SignalTrace], limits: dict[str, tuple[float, float]]) -> None:
    by_id = {r.id: r for r in recorders}
    reference = case["reference_recorder"]
    own_onsets: dict[str, list[float]] = {}
    for trace in traces:
        own_onsets.setdefault(trace.recorder, []).extend(onsets(trace, limits[trace.ct][0]))
    own_onsets = {k: dedupe(v) for k, v in own_onsets.items()}
    ref_onsets = own_onsets.get(reference, [])
    by_id[reference].alignment = f"acuan ({len(ref_onsets)} awal gangguan)"
    for rec in recorders:
        if rec.id == reference or rec.id not in own_onsets:
            continue
        match: Optional[ClockMatch] = match_clock(ref_onsets, own_onsets[rec.id])
        if match is None or match.anchors == 0:
            rec.alignment = "tidak bisa diselaraskan (tidak ada awal gangguan yang cocok)"
            continue
        rec.offset_s = match.offset_s
        rec.alignment = (f"awal gangguan: {match.anchors}/{match.candidates} cocok"
                         + (" (satu jangkar, diambil yang terdekat)" if match.anchors == 1 else ""))
    # Recorders without current line up on a breaker opening an aligned record shows.
    aligned_edges: dict[str, list[float]] = {}
    for sig in signals:
        rec = by_id[sig.recorder]
        if sig.kind == "breaker" and (rec.id == reference or rec.id in own_onsets):
            aligned_edges.setdefault(sig.subject, []).extend(t + rec.offset_s for t, v in sig.edges[1:] if v)
    align_on = case.get("align_breakers") or {}
    for spec in case["recorders"]:
        rec = by_id[spec["id"]]
        breaker = spec.get("align_breaker") or align_on.get(rec.id)
        if not breaker or rec.id in own_onsets or rec.id == reference:
            continue
        reference_open = min(aligned_edges.get(breaker) or [float("nan")])
        own = [t for sig in signals if sig.recorder == rec.id and sig.subject == breaker for t, v in sig.edges[1:] if v]
        if not own or np.isnan(reference_open):
            rec.alignment = f"tidak bisa diselaraskan ({breaker} tidak ada di rekaman lain)"
            continue
        nearest = min(own, key=lambda t: abs(t - reference_open))
        rec.offset_s = reference_open - nearest
        rec.alignment = f"pembukaan {breaker} disamakan dengan rekaman yang sudah selaras"
    for sig in signals:
        sig.offset_s = by_id[sig.recorder].offset_s
    for trace in traces:
        trace.offset_s = by_id[trace.recorder].offset_s


# --- the run ------------------------------------------------------------------------

def _evaluate(case: dict[str, Any], recorders: list[_Recorder], epoch: datetime):
    """Observations and estimates for one complete mapping."""
    for rec in recorders:
        rec.offset_s, rec.alignment = 0.0, ""
    variants = build_transformer(case["transformer"]) if "transformer" in case else build_diameter(case["diameter"])
    by_id = {r.id: r for r in recorders}
    traces = _current_traces(case, by_id, epoch)
    signals = _signal_traces(case, by_id, epoch)
    ratings = case.get("ct_ratings_a") or {}
    loads: dict[str, float] = {}
    for trace in traces:
        level = load_level(trace)
        if level is not None:
            loads[trace.ct] = max(loads.get(trace.ct, 0.0), level)
    limits = {ct: thresholds(loads.get(ct, 0.0), ratings.get(ct)) for ct in sorted({t.ct for t in traces})}
    _align(case, recorders, traces, signals, limits)

    changes = breaker_changes(signals)
    inceptions = [t + trace.offset_s for trace in traces for t in onsets(trace, limits[trace.ct][0])]
    spans = episodes(traces, {ct: v[0] for ct, v in limits.items()}, inceptions=inceptions)
    results = []
    for n, span in enumerate(spans):
        next_start = spans[n + 1][0] if n + 1 < len(spans) else None
        segs = intervals(span, changes, variants[0].label, next_start=next_start)
        observations = (current_observations(traces, segs, limits)
                        + zone_observations(signals, segs)
                        + direction_observations(traces, segs, limits, episode_start=span[0]))
        results.append(EpisodeResult(span[0], span[1], estimate(variants, segs, observations)))
    return results, loads, limits


def _score(results: list[EpisodeResult]) -> tuple[int, int, int]:
    ranks = [ep.estimate.best()[0].rank for ep in results if ep.estimate.best()]
    return tuple(sum(r[i] for r in ranks) for i in range(3)) if ranks else (0, 0, 0)


def run_case(case: dict[str, Any], root: Path) -> CaseResult:
    case = copy.deepcopy(case)
    recorders = _load_recorders(case, root)
    decisions: list[Decision] = []
    alternatives: list[list[dict[str, Any]]] = []
    first_trigger = next(r.comtrade[0] for r in recorders if r.comtrade)
    if not case.get("currents"):
        if "transformer" in case:
            raise ValueError("pemetaan otomatis belum tersedia untuk bay trafo: tulis 'currents' di case")
        inferred: Mapping = infer_mapping(case, _sources(recorders, datetime.fromisoformat(first_trigger.trigger_time_iso)))
        decisions = inferred.decisions
        for key in _MAPPING_KEYS:
            if not case.get(key):
                case[key] = getattr(inferred, key)
        case.setdefault("diameter", inferred.diameter)
        case.setdefault("ct_ratings_a", inferred.ct_ratings_a)
        case.setdefault("reference_recorder", inferred.reference_recorder)
        case.setdefault("align_breakers", inferred.align_breakers)
        alternatives = inferred.alternatives
    reference = next(r for r in recorders if r.id == case["reference_recorder"])
    epoch = datetime.fromisoformat(reference.comtrade[0].trigger_time_iso)

    options = [case["currents"]] + alternatives
    evaluated = []
    for currents in options:
        trial = {**case, "currents": currents}
        evaluated.append((trial, *_evaluate(trial, recorders, epoch)))
    best = min(range(len(evaluated)), key=lambda i: _score(evaluated[i][1]))
    trial, results, loads, limits = evaluated[best]
    if best != len(evaluated) - 1:
        _evaluate(trial, recorders, epoch)          # recorder offsets as in the chosen run
    choice = ""
    if len(evaluated) > 1:
        scores = sorted(_score(e[1]) for e in evaluated)
        decided_by = "bukti kuat" if scores[0][:2] < scores[1][:2] else "bukti lemah saja"
        picked = ", ".join(f"{s['ct']} = {s['recorder']}:{'/'.join(s['channels'])}" for s in trial["currents"]
                           if not all(s in option for option in options))
        choice = (f"{len(evaluated)} kemungkinan pemetaan diuji; dipilih skor {scores[0]} "
                  f"vs berikutnya {scores[1]} (ditentukan oleh {decided_by}): {picked}")
    notes = [f"{r.label}: {r.alignment}" for r in recorders if r.alignment.startswith("tidak")]
    return CaseResult(case.get("title") or "", epoch, recorders, loads, limits, results, notes, decisions, choice)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--root", type=Path, required=True, help="incident folder (archives extracted in place)")
    parser.add_argument("--json", type=Path, help="also write the result as JSON")
    args = parser.parse_args(argv)
    case = json.loads(args.case.read_text(encoding="utf-8"))
    result = run_case(case, args.root)
    print("\n".join(result.report_lines()))
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
