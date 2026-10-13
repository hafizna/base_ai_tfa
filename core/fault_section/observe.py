"""Observations from recordings, on one incident clock.

Turning records into the evidence ``estimate`` checks:

- *Clock alignment.* Every recorder keeps its own clock. A recorder with
  current channels is placed on the reference recorder's clock by lining up
  the fault inceptions both saw — instants where current jumps after being
  quiet — with the shift that makes the most inceptions coincide (two pin it;
  a single one is taken as the nearest match and flagged). A recorder with
  events only is placed by a breaker opening that an aligned record also shows.
- *Episodes.* A fault episode lasts while any CT carries fault current; two
  faults a second apart are two episodes, estimated one by one.
- *Intervals.* Within an episode every breaker change starts a new interval,
  after a short guard for the arc to clear at the next current zero. The span
  just after the episode is kept as an "after clearing" interval.
- *CT current.* The median one-cycle RMS of the largest phase in the interval:
  above max(1.5 × load, 10 % of CT rating) carries fault current, below
  max(10 % of load, 2 % of CT rating) carries none, anything between (load,
  line charging) is left unsaid rather than guessed.
- *Zone operation.* An operate signal high in the interval (past a short
  allowance for the previous interval's reset) means operated; low throughout
  an interval long enough to have operated means it did not.
- *Direction.* Superimposed quantities: per one-cycle window, ΔV and ΔI
  against the same window a whole number of cycles before the fault, on the
  loop whose ΔI is largest. A forward fault sees the source behind the relay,
  ΔV/ΔI ≈ −Z_source (about −100°); a reverse one sees the network in front,
  ≈ +80°. The interval's direction is the majority of windows. This works
  behind a transformer too, where the voltage barely dips and an impedance
  loop says little. Polarity is not verified here, so direction is always
  weak evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .estimate import CT_CURRENT, DIRECTION, ZONE_OPERATE, Interval, Observation

NOMINAL_HZ = 50.0
_LOOPS = (("AG", (0,), ()), ("BG", (1,), ()), ("CG", (2,), ()),
          ("AB", (0,), (1,)), ("BC", (1,), (2,)), ("CA", (2,), (0,)))


@dataclass
class CurrentTrace:
    """Three phase currents of one CT as one recorder saw them."""
    ct: str
    source: str
    recorder: str
    t: np.ndarray                           # seconds from the case epoch on the recorder's own clock
    phases: tuple[np.ndarray, ...]          # primary amperes
    voltages: tuple[np.ndarray, ...] = ()   # primary phase-to-ground volts, same phase order
    forward: str = ""                       # section the recorder's relay calls forward
    offset_s: float = 0.0                   # recorder clock -> incident clock
    _track: Optional[tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    def window(self) -> int:
        dt = float(np.median(np.diff(self.t)))
        return max(4, int(round(1.0 / (NOMINAL_HZ * dt))))

    def track(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t_end, magnitude)``: largest-phase one-cycle RMS, every quarter
        cycle, stamped at each window's last sample (recorder clock)."""
        if self._track is None:
            n = self.window()
            ends, phasors = _phasors(self.phases, n)
            self._track = (self.t[ends], np.max(np.abs(phasors), axis=0))
        return self._track


@dataclass
class SignalTrace:
    """A breaker's open state (True = open) or a zone's operate state over
    time, from one recorder."""
    subject: str
    kind: str                               # "breaker" | "zone" | "direction"
    source: str
    recorder: str
    edges: list[tuple[float, bool]]         # (recorder clock s, state); first entry is the state at coverage start
    span: tuple[float, float]               # coverage, recorder clock
    weak: bool = False
    value: str = ""                         # "direction": what a high signal says ("forward")
    forward: str = ""
    offset_s: float = 0.0

    def state_at(self, t_incident: float) -> bool:
        t = t_incident - self.offset_s
        state = self.edges[0][1] if self.edges else False
        for when, value in self.edges:
            if when > t:
                break
            state = value
        return state

    def rises_within(self, a: float, b: float) -> bool:
        return any(value and a - self.offset_s < when <= b - self.offset_s for when, value in self.edges)

    def covers(self, a: float, b: float) -> bool:
        return self.span[0] + self.offset_s <= a and b <= self.span[1] + self.offset_s


def _phasors(signals: Sequence[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """One-cycle DFT phasors (RMS scaled) every quarter cycle; returns the
    window end indices and an array (phases, windows)."""
    step = max(1, n // 4)
    k = np.arange(n)
    w = np.exp(-2j * np.pi * k / n) * (np.sqrt(2) / n)
    out = []
    ends = np.arange(n - 1, len(signals[0]), step)
    for x in signals:
        windows = np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=float), n)[::step]
        out.append(windows[: len(ends)] @ w)
    return ends[: len(out[0])], np.vstack(out)


# --- thresholds ----------------------------------------------------------------

def initial_magnitude(trace: CurrentTrace, cycles: float = 2.0) -> float:
    t_end, mag = trace.track()
    upto = t_end[0] + cycles / NOMINAL_HZ
    return float(np.median(mag[t_end <= upto])) if np.any(t_end <= upto) else float(mag[0])


def load_level(trace: CurrentTrace) -> Optional[float]:
    """The record's opening current as a load reading, or None when the
    record does not open on load: it starts inside a fault (nothing later is
    much larger) or on a dead circuit."""
    level = initial_magnitude(trace)
    _, mag = trace.track()
    return level if float(np.max(mag)) > 2.0 * max(level, 1.0) else None


def thresholds(load_a: float, rating_a: Optional[float]) -> tuple[float, float]:
    """``(carries, carries_none)`` in amperes."""
    if rating_a:
        return max(1.5 * load_a, 0.10 * rating_a), max(0.10 * load_a, 0.02 * rating_a)
    return max(1.5 * load_a, 100.0), max(0.10 * load_a, 20.0)


# --- clock alignment --------------------------------------------------------------

def onsets(trace: CurrentTrace, threshold_a: float, quiet_s: float = 0.1) -> list[float]:
    """Fault inceptions on the recorder's clock: the first sample whose
    cycle-to-cycle change exceeds ``threshold_a`` after ``quiet_s`` without
    (one cycle at the start of the record, which may hold little pre-fault),
    confirmed by fault-level RMS one cycle later."""
    n = trace.window()
    x = np.vstack([np.asarray(p, dtype=float) for p in trace.phases])
    if x.shape[1] <= n:
        return []
    delta = np.max(np.abs(x[:, n:] - x[:, :-n]), axis=0)
    above = delta > threshold_a
    t = trace.t[n:]
    t_end, mag = trace.track()
    found, last_above = [], None
    for idx in np.flatnonzero(above):
        when = float(t[idx])
        if last_above is None:
            quiet = when - t[0] >= 1.0 / NOMINAL_HZ
        else:
            quiet = when - last_above >= quiet_s
        last_above = when
        if not quiet:
            continue
        later = np.searchsorted(t_end, when + 1.0 / NOMINAL_HZ)
        if later < len(mag) and mag[later] > threshold_a:
            found.append(when)
    return found


@dataclass(frozen=True)
class ClockMatch:
    offset_s: float
    anchors: int            # inceptions that coincide
    candidates: int         # inceptions the recorder had


def match_clock(reference: Sequence[float], own: Sequence[float], tolerance_s: float = 0.03) -> Optional[ClockMatch]:
    """Shift putting most of ``own`` onto ``reference``; ties go to the
    smallest shift."""
    best: Optional[ClockMatch] = None
    for r in reference:
        for o in own:
            offset = r - o
            hits = sum(1 for x in own if any(abs(x + offset - y) <= tolerance_s for y in reference))
            candidate = ClockMatch(offset, hits, len(own))
            if best is None or (hits, -abs(offset)) > (best.anchors, -abs(best.offset_s)):
                best = candidate
    return best


def dedupe(times: Iterable[float], within_s: float = 0.03) -> list[float]:
    out: list[float] = []
    for t in sorted(times):
        if not out or t - out[-1] > within_s:
            out.append(t)
    return out


# --- episodes and intervals ----------------------------------------------------------

def episodes(traces: Sequence[CurrentTrace], carries_a: dict[str, float], merge_gap_s: float = 0.1,
             min_s: float = 0.02, inceptions: Sequence[float] = ()) -> list[tuple[float, float]]:
    """Spans (incident clock) during which some CT carried fault current. An
    RMS window lags the inception by up to a cycle, so a span starts at the
    earliest instantaneous inception (``onsets``) that falls in that lag."""
    spans = []
    for trace in traces:
        t_end, mag = trace.track()
        t_end = t_end + trace.offset_s
        above = mag > carries_a[trace.ct]
        idx = np.flatnonzero(above)
        if idx.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]]))
        for s, e in zip(starts, ends):
            spans.append((float(t_end[s]) - 1.0 / NOMINAL_HZ, float(t_end[e])))
    spans.sort()
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        near = [t for t in inceptions if a - 0.005 <= t <= a + 1.25 / NOMINAL_HZ]
        out.append((min(near) if near else a, b))
    return [(a, b) for a, b in out if b - a >= min_s]


def breaker_changes(signals: Sequence[SignalTrace]) -> dict[str, list[tuple[float, bool]]]:
    """Each breaker's open state over the incident clock, all its sources
    merged: the earliest source to report a change sets it."""
    merged: dict[str, list[tuple[float, bool]]] = {}
    by_breaker: dict[str, list[tuple[float, bool]]] = {}
    for sig in signals:
        if sig.kind != "breaker":
            continue
        by_breaker.setdefault(sig.subject, []).extend((t + sig.offset_s, s) for t, s in sig.edges)
    for breaker, edges in by_breaker.items():
        edges.sort()
        out: list[tuple[float, bool]] = []
        for when, state in edges:
            if not out or out[-1][1] != state:
                out.append((when, state))
        merged[breaker] = out
    return merged


def _open_at(changes: dict[str, list[tuple[float, bool]]], t: float) -> frozenset[str]:
    opened = set()
    for breaker, edges in changes.items():
        state = False
        for when, value in edges:
            if when > t:
                break
            state = value
        if state:
            opened.add(breaker)
    return frozenset(opened)


def intervals(episode: tuple[float, float], changes: dict[str, list[tuple[float, bool]]], label_of,
              *, guard_s: float = 0.01, min_s: float = 0.015, clearing_s: tuple[float, float] = (0.03, 0.2),
              next_start: Optional[float] = None) -> list[Interval]:
    """Constant-breaker-state intervals of one episode, plus the span after it."""
    start, end = episode
    cuts = sorted({when for edges in changes.values() for when, _ in edges if start < when < end})
    bounds = [(start, cuts[0] if cuts else end)]
    for i, cut in enumerate(cuts):
        bounds.append((cut + guard_s, cuts[i + 1] if i + 1 < len(cuts) else end))
    out: list[Interval] = []
    previous: Optional[frozenset[str]] = None
    for a, b in bounds:
        if b - a < min_s:
            continue
        opened = _open_at(changes, (a + b) / 2)
        out.append(Interval(f"S{len(out)}", a, b, opened, _describe(opened, previous, label_of)))
        previous = opened
    a, b = end + clearing_s[0], end + clearing_s[1]
    if next_start is not None:
        b = min(b, next_start - 0.01)
    # Without any breaker reading around the episode every hypothesis predicts
    # the same after clearing, so that span would tell nothing.
    known = any(start <= when <= b for edges in changes.values() for when, _ in edges)
    if known and b - a >= min_s:
        opened = _open_at(changes, (a + b) / 2)
        out.append(Interval("C", a, b, opened, "setelah gangguan padam", after_clearing=True))
    return out


def _names(breakers: Iterable[str], label_of, most: int = 4) -> str:
    names = sorted(label_of(b) for b in breakers)
    if len(names) <= most:
        return ", ".join(names)
    return ", ".join(names[:2]) + f" dan {len(names) - 2} CB lain"


def _describe(opened: frozenset[str], previous: Optional[frozenset[str]], label_of) -> str:
    if previous is None:
        return "awal: " + (_names(opened, label_of) + " open" if opened else "semua CB tertutup")
    newly, closed = opened - previous, previous - opened
    parts = ([_names(newly, label_of) + " open"] if newly else []) + ([_names(closed, label_of) + " close"] if closed else [])
    return "+ " + "; ".join(parts) if parts else "tanpa perubahan CB"


# --- observations ---------------------------------------------------------------

def _combine(observations: list[Observation]) -> list[Observation]:
    """One observation per (interval, kind, subject) when every source agrees;
    all of them kept when they disagree, so the conflict shows up."""
    grouped: dict[tuple[str, str, str], list[Observation]] = {}
    for obs in observations:
        grouped.setdefault((obs.interval, obs.kind, obs.subject), []).append(obs)
    out = []
    for group in grouped.values():
        if len({repr(o.value) for o in group}) > 1:
            out.extend(group)
            continue
        first = group[0]
        out.append(Observation(first.interval, first.kind, first.subject, first.value,
                               "; ".join(dict.fromkeys(o.source for o in group)),
                               "; ".join(dict.fromkeys(o.detail for o in group if o.detail)),
                               weak=all(o.weak for o in group), forward=first.forward))
    return out


def current_observations(traces: Sequence[CurrentTrace], spans: Sequence[Interval],
                         limits: dict[str, tuple[float, float]], *, fill_s: float = 1.0 / NOMINAL_HZ) -> list[Observation]:
    found = []
    for trace in traces:
        carries, carries_none = limits[trace.ct]
        t_end, mag = trace.track()
        t_end = t_end + trace.offset_s
        for interval in spans:
            sel = (t_end >= interval.start_s + fill_s) & (t_end <= interval.end_s)
            if not np.any(sel):
                continue
            level = float(np.median(mag[sel]))
            value = True if level > carries else False if level < carries_none else None
            if value is None:
                continue
            found.append(Observation(interval.id, CT_CURRENT, trace.ct, value, trace.source,
                                     f"{level:,.0f} A", weak=interval.after_clearing))
    return _combine(found)


def zone_observations(signals: Sequence[SignalTrace], spans: Sequence[Interval], *,
                      reset_s: float = 0.02, min_quiet_s: float = 0.03) -> list[Observation]:
    found = []
    for sig in signals:
        if sig.kind not in ("zone", "direction"):
            continue
        for interval in spans:
            if interval.after_clearing:
                continue
            a, b = interval.start_s, interval.end_s
            if not sig.covers(a, b):
                continue
            high = sig.state_at(a + reset_s) or sig.rises_within(a + reset_s, b)
            if sig.kind == "direction":
                if high:
                    found.append(Observation(interval.id, DIRECTION, sig.subject, sig.value, sig.source,
                                             "keputusan rele", weak=True, forward=sig.forward))
                continue
            low = not sig.state_at(a) and not sig.rises_within(a, b) and b - a >= min_quiet_s
            if high or low:
                found.append(Observation(interval.id, ZONE_OPERATE, sig.subject, bool(high), sig.source, weak=sig.weak))
    return _combine(found)


def direction_observations(traces: Sequence[CurrentTrace], spans: Sequence[Interval],
                           limits: dict[str, tuple[float, float]], *, source_angle_deg: float = 80.0,
                           majority: float = 0.7, fill_s: float = 1.0 / NOMINAL_HZ,
                           episode_start: Optional[float] = None) -> list[Observation]:
    found = []
    for trace in traces:
        if len(trace.voltages) != 3 or not trace.forward:
            continue
        n = trace.window()
        ends, i_ph = _phasors(trace.phases, n)
        _, v_ph = _phasors(trace.voltages, n)
        t_end = trace.t[ends] + trace.offset_s
        mag = np.max(np.abs(i_ph), axis=0)
        carries = limits[trace.ct][0]
        for interval in spans:
            if interval.after_clearing:
                continue
            start = episode_start if episode_start is not None else interval.start_s
            pre = np.flatnonzero(t_end < start - 0.002)
            if pre.size == 0:
                continue                         # no pre-fault window in this record
            last_pre = int(pre[-1])
            sel = np.flatnonzero((t_end >= interval.start_s + fill_s) & (t_end <= interval.end_s) & (mag > carries))
            votes, loops = [], []
            for k in sel:
                ref = last_pre - ((last_pre - k) % 4)    # same DFT phase: a whole number of cycles back
                if ref < 0:
                    continue
                d_i, d_v = i_ph[:, k] - i_ph[:, ref], v_ph[:, k] - v_ph[:, ref]
                best = None
                for name, (pv,), rest in _LOOPS:
                    i = d_i[pv] - (d_i[rest[0]] if rest else 0)
                    v = d_v[pv] - (d_v[rest[0]] if rest else 0)
                    if best is None or abs(i) > abs(best[1]):
                        best = (name, i, v)
                name, i, v = best
                if abs(i) < 1e-9:
                    continue
                angle = np.degrees(np.angle(v / i))
                votes.append(np.cos(np.radians(angle - (source_angle_deg - 180.0))) > 0)
                loops.append((name, angle))
            if not votes:
                continue
            share = float(np.mean(votes))
            if max(share, 1 - share) < majority:
                continue
            value = "forward" if share >= 0.5 else "reverse"
            name, angle = loops[len(loops) // 2]
            found.append(Observation(interval.id, DIRECTION, trace.ct, value, trace.source,
                                     f"ΔV/ΔI loop {name} {angle:.0f}°, {max(share, 1 - share):.0%} jendela",
                                     weak=True, forward=trace.forward))
    return _combine(found)
