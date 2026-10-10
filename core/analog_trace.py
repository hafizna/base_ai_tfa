"""What each phase's current and voltage did through one record.

The status channels say what the relay decided; the analog channels say what
the power system did. This module reads the analog side, per phase, on the
record's disturbed line (scope multi-line payloads first), and reduces it to
facts:

- spans in which a phase's fundamental current stays ``none`` / ``load`` /
  ``high`` and its voltage stays ``dead`` / ``sag`` / ``normal``;
- disturbances: stretches where a phase voltage sags while current still flows
  (a fault always pulls the voltage of its phases down at the relay; a current
  rise with every voltage normal is a load or energization change, not a
  fault);
- events timed to the sample:
  - ``disturbance_start``: from the superimposed waveform x(t) - x(t - one
    cycle), current or voltage, so a weak infeed end is timed too;
  - ``current_ceased``: the last current zero before a phase's current
    collapses. A breaker interrupts at current zero, and the decaying CT tail
    after it has no zero crossing, so it is not counted;
  - ``current_back_to_load``: a high current settling back to load without
    ceasing;
  - ``current_return``, ``current_rise``, ``voltage_lost``, ``voltage_return``.
- a summary:
  - the first fault's start, and its clearing: the last current zero of its
    phases, per the Grid Code definition (Permen ESDM 20/2020 CCA1 2.2, "from
    the fault to the arc extinguished by the opening breaker");
  - which phases' currents ceased, and whether together or one alone;
  - zero-current spans, and current or voltage returning after them;
  - an energization transient at a reclose, and a new fault after a return.

These are readings, not conclusions. A current that ceases may be this breaker
opening or the far end opening (a healthy phase loses its load flow when the
remote end opens first); which phases were faulted and which breaker opened is
decided in the reasoning chain with the status channels
(docs/fault-reasoning-rules.md, steps 3, 4 and 6).

Times are milliseconds on the record's own time axis, the same axis as
``EventWindow.inception_time_ms``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

PHASES = ("A", "B", "C")

# Current classes, against the record's reference current: the prefault load,
# or the energized current of a record that starts dead.
_NONE_OF_REFERENCE = 0.10      # below this fraction of the reference: no current
_NONE_OF_PEAK = 0.003          # ...and below this fraction of the record's highest current
# A healthy phase can carry twice its load during a single-phase fault (zero
# sequence current returning through it: 2.2x on Cibatu-Mekarsari 2), so a
# fault-level current is taken from three times the reference.
_HIGH_OF_REFERENCE = 3.0
# Voltage classes, per unit of the reference (prefault or energized) voltage.
_SAG_PU = 0.85
_DEAD_PU = 0.15
# Shorter than this a class run is jitter at a boundary, or a CT tail.
_MIN_SPAN_CYCLES = 1.0
_MIN_STATE_CYCLES = 2.0
# Phases whose current ceases within this many cycles of each other ceased together.
_TOGETHER_CYCLES = 1.5
# A disturbance starting this soon after current or voltage returns is the
# energization of a reclose unless it ends in a current ceasing (a trip).
_ENERGIZATION_WINDOW_CYCLES = 2.0


@dataclass
class Span:
    current: str                  # "none" | "load" | "high"
    voltage: Optional[str]        # "dead" | "sag" | "normal" | None without a voltage channel
    start_ms: float
    end_ms: float
    current_rms: float            # median fundamental RMS current over the span
    voltage_pu: Optional[float]   # median fundamental voltage over the span, per unit


@dataclass
class TraceEvent:
    t_ms: float
    phase: str
    kind: str
    value: Optional[float] = None  # the current (A) or voltage (pu) the event is about


@dataclass
class AnalogTrace:
    cycle_ms: float
    reference: dict[str, Any]
    phases: dict[str, list[Span]] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_ms": self.cycle_ms,
            "reference": self.reference,
            "phases": {p: [asdict(s) for s in spans] for p, spans in self.phases.items()},
            "events": [asdict(e) for e in self.events],
            "summary": self.summary,
            "warnings": self.warnings,
        }


# --- signal helpers ----------------------------------------------------------------

def _phase_channels(payload: dict, measurement: str) -> tuple[dict[str, np.ndarray], Optional[str]]:
    """First channel per phase for one measurement (a scoped payload carries one line)."""
    out: dict[str, np.ndarray] = {}
    unit: Optional[str] = None
    for ch in payload.get("analog_channels", []):
        phase = str(ch.get("phase") or "").upper()
        if ch.get("measurement") != measurement or phase not in PHASES or phase in out:
            continue
        samples = np.asarray(ch.get("samples") or [], dtype=float)
        if samples.size:
            out[phase] = samples
            unit = unit or (ch.get("unit") or None)
    return out, unit


def _fundamental_rms(x: np.ndarray, t: np.ndarray, freq: float, n: int) -> np.ndarray:
    """RMS of the fundamental over the one-cycle window ENDING at each sample
    (NaN before the first full cycle), from a running sum of the demodulated
    signal: O(N), exact for a stationary sinusoid, and blind to DC."""
    z = x * np.exp(-1j * 2.0 * np.pi * freq * t)
    c = np.concatenate([[0.0 + 0.0j], np.cumsum(z)])
    window = c[n:] - c[:-n]
    out = np.full(len(x), np.nan)
    out[n - 1:] = (2.0 / n) * np.abs(window) / np.sqrt(2.0)
    return out


def _median(values: np.ndarray) -> Optional[float]:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """(first, last) index of each run of True."""
    out: list[tuple[int, int]] = []
    k = 0
    while k < len(flags):
        if flags[k]:
            j = k
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


def _smooth(labels: list, min_steps: int) -> list:
    """Absorb runs shorter than ``min_steps`` into the run before them."""
    if not labels:
        return labels
    runs: list[list[int]] = []
    for k, label in enumerate(labels):
        if runs and labels[runs[-1][0]] == label:
            runs[-1].append(k)
        else:
            runs.append([k])
    out = list(labels)
    for pos in range(1, len(runs)):
        if len(runs[pos]) < min_steps:
            fill = out[runs[pos - 1][-1]]
            for k in runs[pos]:
                out[k] = fill
    return out


# --- sample-level timing ---------------------------------------------------------------

def _walk_back(delta: np.ndarray, start: int, floor: float, gap: int) -> int:
    """From ``start``, move back while |delta| stays above ``floor``, crossing
    dips of up to ``gap`` samples (the superimposed signal's own zeros)."""
    k = start
    quiet = 0
    while k > 0:
        if delta[k - 1] <= floor:
            quiet += 1
            if quiet > gap:
                break
        else:
            quiet = 0
        k -= 1
    return k + quiet


def _superimposed_onset(x: np.ndarray, n: int, lo: int, hi: int, threshold: float, noise: float) -> Optional[int]:
    """The instant the waveform stops repeating its previous cycle.

    Detection: the first sample in [lo, hi) from which |x(t) - x(t - one
    cycle)| stays above ``threshold`` for a quarter cycle. That sample can lag
    the real start by up to a quarter cycle when the superimposed signal grows
    slowly (a weak infeed end), so the onset is walked back while the signal
    stays above a quarter of the threshold."""
    lo = max(lo, n)
    hi = min(hi, len(x))
    if hi <= lo:
        return None
    delta = np.abs(x[lo:hi] - x[lo - n:hi - n])
    above = delta > threshold
    run = max(1, n // 4)
    start = next((k for k in range(len(above) - run + 1) if above[k] and above[k:k + run].all()), None)
    if start is None:
        return None
    # A quarter of the threshold, not the noise floor: walking down to the
    # noise would also take in small excursions before the fault proper (on
    # Bringin ZQ6D phases S and T carried ~20 A for 5 ms before the flashover).
    return lo + _walk_back(delta, start, max(noise, 0.25 * threshold), max(1, n // 16))


def _prefault_delta(x: np.ndarray, n: int) -> float:
    """95th percentile of |x(t) - x(t - one cycle)| over the record's second
    cycle: what the superimposed signal looks like with nothing happening."""
    if len(x) < 2 * n:
        return 0.0
    return float(np.percentile(np.abs(x[n:2 * n] - x[:n]), 95))


def _last_zero_before_collapse(x: np.ndarray, lo: int, hi: int, live_level: float) -> Optional[int]:
    """The current zero that ends conduction in [lo, hi): after the last
    sample still above ``live_level``, the first sample past a sign change or
    already near zero (a sixth of ``live_level``). The CT tail that follows can
    keep the polarity of the last half cycle, so a sign change alone can miss
    the zero. Without either, the last live sample."""
    lo = max(0, lo)
    hi = min(hi, len(x))
    seg = x[lo:hi]
    live = np.where(np.abs(seg) >= live_level)[0]
    if live.size == 0:
        return None
    last = int(live[-1])
    near_zero = live_level / 6.0
    for k in range(last + 1, len(seg)):
        if np.sign(seg[k]) != np.sign(seg[k - 1]) or abs(seg[k]) <= near_zero:
            return lo + k
    return lo + last


def _first_live(x: np.ndarray, lo: int, hi: int, live_level: float) -> Optional[int]:
    lo = max(0, lo)
    hi = min(hi, len(x))
    live = np.where(np.abs(x[lo:hi]) >= live_level)[0]
    return lo + int(live[0]) if live.size else None


# --- the trace -----------------------------------------------------------------------------

def trace_payload(payload: dict) -> Optional[AnalogTrace]:
    """Analog trace of a payload already scoped to one line (see
    ``core.line_selection.scope_payload``). None when the record has no phase
    currents or is shorter than three cycles."""
    t = np.asarray(payload.get("time") or [], dtype=float)
    freq = float(payload.get("frequency") or 50.0) or 50.0
    currents, current_unit = _phase_channels(payload, "current")
    voltages, voltage_unit = _phase_channels(payload, "voltage")
    if len(t) < 16 or not currents:
        return None
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    n = max(4, int(round(1.0 / (dt * freq))))
    if len(t) < 3 * n:
        return None
    step = max(1, n // 2)
    idx = np.arange(n - 1, len(t), step)          # each step is the one-cycle window ending at idx
    steps = len(idx)
    t_ms = t[idx] * 1000.0
    t_all_ms = t * 1000.0
    cycle_ms = 1000.0 / freq
    steps_per_cycle = n / step
    min_span = max(1, int(round(_MIN_SPAN_CYCLES * steps_per_cycle)))
    min_state = max(1, int(round(_MIN_STATE_CYCLES * steps_per_cycle)))
    warnings: list[str] = []

    i_rms = {p: _fundamental_rms(x, t, freq, n)[idx] for p, x in currents.items()}
    v_rms = {p: _fundamental_rms(x, t, freq, n)[idx] for p, x in voltages.items()}

    # --- references ------------------------------------------------------------------
    head = slice(0, 2)  # the first cycle of complete windows
    peak_i = max(float(np.nanmax(v)) for v in i_rms.values())
    head_i = {p: _median(v[head]) or 0.0 for p, v in i_rms.items()}
    starts_dead = all(head_i[p] < 10 * _NONE_OF_PEAK * max(peak_i, 1e-9) for p in head_i)
    if v_rms:
        peak_v = max(float(np.nanmax(v)) for v in v_rms.values())
        starts_dead = starts_dead and all((_median(v[head]) or 0.0) < _DEAD_PU * peak_v for v in v_rms.values())

    def energized_level(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return 0.0
        on = finite[finite >= 0.2 * float(np.max(finite))]
        return float(np.median(on)) if on.size else float(np.max(finite))

    load = {p: (energized_level(v) if starts_dead else head_i[p]) for p, v in i_rms.items()}
    reference_current = float(np.median(list(load.values()))) or peak_i
    none_thr = max(_NONE_OF_REFERENCE * reference_current, _NONE_OF_PEAK * peak_i)
    high_thr = max(_HIGH_OF_REFERENCE * reference_current, 4.0 * none_thr)
    if not starts_dead and reference_current < 2 * none_thr:
        warnings.append("Prefault current is near zero: the line was lightly loaded or open before the record.")

    nominal_v = {p: (energized_level(v) if starts_dead else (_median(v[head]) or 0.0)) for p, v in v_rms.items()}
    v_pu = {p: v_rms[p] / nominal_v[p] for p in v_rms if nominal_v.get(p)}
    if v_pu and not starts_dead and min(nominal_v.values()) < 0.5 * max(nominal_v.values()):
        warnings.append("Prefault voltage differs strongly between phases; voltage classes may be unreliable.")

    # --- classes per step, smoothed ----------------------------------------------------------
    def current_class(rms: float) -> str:
        if not np.isfinite(rms) or rms < none_thr:
            return "none"
        return "high" if rms >= high_thr else "load"

    def voltage_class(pu: float) -> Optional[str]:
        if not np.isfinite(pu):
            return None
        return "dead" if pu < _DEAD_PU else ("sag" if pu < _SAG_PU else "normal")

    i_cls = {p: _smooth([current_class(float(r)) for r in i_rms[p]], min_span) for p in i_rms}
    v_cls = {p: _smooth([voltage_class(float(x)) for x in v_pu[p]], min_span) for p in v_pu}

    phases: dict[str, list[Span]] = {}
    for p in i_rms:
        vc = v_cls.get(p) or [None] * steps
        keys = list(zip(i_cls[p], vc))
        spans: list[Span] = []
        for k0, k1 in _key_runs(keys):
            spans.append(Span(
                current=i_cls[p][k0],
                voltage=vc[k0],
                start_ms=round(float(t_ms[k0]), 1),
                end_ms=round(float(t_ms[k1]), 1),
                current_rms=_round(_median(i_rms[p][k0:k1 + 1])) or 0.0,
                voltage_pu=_round(_median(v_pu[p][k0:k1 + 1]), 3) if p in v_pu else None,
            ))
        phases[p] = spans

    events: list[TraceEvent] = []

    # --- zero-current spans and what returns after them --------------------------------------------
    zero_runs: dict[str, list[tuple[int, int]]] = {}
    for p, cls in i_cls.items():
        runs = [(a, b) for a, b in _runs([c == "none" for c in cls]) if (b - a + 1) >= min_state or a == 0]
        zero_runs[p] = runs
    returns_ms: dict[str, list[float]] = {p: [] for p in i_cls}
    for p, runs in zero_runs.items():
        for a, b in runs:
            if b + 1 >= steps:
                continue
            live_after = sum(1 for c in i_cls[p][b + 1:b + 1 + min_state] if c != "none")
            if live_after < min_state:
                continue
            lo = int(idx[b]) - n
            hi = int(idx[min(steps - 1, b + min_state)])
            level = max(none_thr, 0.5 * float(np.nanmedian(i_rms[p][b + 1:b + 1 + min_state]))) * np.sqrt(2.0)
            live = _first_live(currents[p], lo, hi, level)
            if live is not None:
                ret = round(float(t_all_ms[live]), 1)
                returns_ms[p].append(ret)
                events.append(TraceEvent(ret, p, "current_return", _round(float(np.nanmedian(i_rms[p][b + 1:b + 1 + min_state])))))

    voltage_returns_ms: dict[str, list[float]] = {p: [] for p in v_cls}
    for p, vc in v_cls.items():
        for a, b in _runs([c == "dead" for c in vc]):
            if (b - a + 1) < min_state:
                continue
            if a > 0:
                events.append(TraceEvent(round(float(t_ms[a] - cycle_ms / 2), 1), p, "voltage_lost", _round(float(v_pu[p][a]), 3)))
            if b + 1 < steps:
                ret = round(float(t_ms[b + 1] - cycle_ms / 2), 1)
                voltage_returns_ms[p].append(ret)
                events.append(TraceEvent(ret, p, "voltage_return", _round(float(v_pu[p][min(steps - 1, b + min_state)]), 3)))

    all_returns = sorted([r for rs in returns_ms.values() for r in rs] + [r for rs in voltage_returns_ms.values() for r in rs])

    # --- disturbances: a voltage sagging while its current flows --------------------------------
    def disturbed(k: int) -> bool:
        if v_cls:
            return any(v_cls[p][k] == "sag" and i_cls.get(p, ["load"] * steps)[k] != "none" for p in v_cls)
        return any(i_cls[p][k] == "high" for p in i_cls)  # current-only record

    dist_runs = [(a, b) for a, b in _runs([disturbed(k) for k in range(steps)]) if (b - a + 1) >= min_span]

    # Current rising to fault level with every voltage normal: a load or
    # energization change, reported but not a disturbance.
    for p, cls in i_cls.items():
        for a, b in _runs([c == "high" for c in cls]):
            overlaps = any(a <= b0 and b >= a0 for a0, b0 in dist_runs)
            if a > 0 and not overlaps and cls[a - 1] == "load":
                events.append(TraceEvent(round(float(t_ms[a] - cycle_ms), 1), p, "current_rise", _round(float(i_rms[p][a]))))

    delta_i = {p: _prefault_delta(x, n) for p, x in currents.items()} if not starts_dead else {p: 0.0 for p in currents}
    delta_v = {p: _prefault_delta(x, n) for p, x in voltages.items()} if not starts_dead else {p: 0.0 for p in voltages}

    disturbances: list[dict[str, Any]] = []
    for k0, k1 in dist_runs:
        lo = int(idx[k0]) - 2 * n
        hi = int(idx[k0]) + n
        onsets = []
        for p, x in currents.items():
            thr = max(6.0 * delta_i[p], 0.2 * np.sqrt(2.0) * reference_current, 2.0 * none_thr)
            noise = max(3.0 * delta_i[p], 0.02 * np.sqrt(2.0) * reference_current)
            found = _superimposed_onset(x, n, lo, hi, thr, noise)
            if found is not None:
                onsets.append(found)
        for p, x in voltages.items():
            if nominal_v.get(p):
                thr = max(6.0 * delta_v[p], 0.05 * np.sqrt(2.0) * nominal_v[p])
                noise = max(3.0 * delta_v[p], 0.005 * np.sqrt(2.0) * nominal_v[p])
                found = _superimposed_onset(x, n, lo, hi, thr, noise)
                if found is not None:
                    onsets.append(found)
        onset = min(onsets) if onsets else int(idx[k0]) - n + 1
        onset_ms = round(float(t_all_ms[max(0, onset)]), 1)
        high_in_run = sorted(p for p in i_cls if any(i_cls[p][k] == "high" for k in range(k0, k1 + 1)))

        # How each phase that carried current leaves the disturbance.
        ceased: dict[str, float] = {}
        back_to_load: dict[str, float] = {}
        look_end = min(steps - 1, k1 + 3 * int(np.ceil(steps_per_cycle)))
        for p, x in currents.items():
            if all(i_cls[p][k] == "none" for k in range(k0, k1 + 1)):
                continue
            # The first drop from flowing to none: after the disturbance, or
            # inside it (a healthy phase losing its load flow when the far end
            # opens first). A phase that was still at none when the
            # disturbance began (a reclose energizing it) has not ceased.
            first_none = next(
                (k for k in range(k0 + 1, look_end + 1) if i_cls[p][k] == "none" and i_cls[p][k - 1] != "none"),
                None,
            )
            if first_none is not None:
                lo_s = int(idx[k0]) - n
                hi_s = int(idx[first_none]) + 1
                seg = np.abs(x[max(0, lo_s):hi_s])
                live_level = 0.3 * float(np.max(seg)) if seg.size else 0.0
                zero = _last_zero_before_collapse(x, lo_s, hi_s, live_level)
                if zero is not None:
                    ceased[p] = round(float(t_all_ms[zero]), 1)
                    events.append(TraceEvent(ceased[p], p, "current_ceased", _round(float(np.nanmax(i_rms[p][k0:k1 + 1])))))
            elif p in high_in_run:
                end_k = next((k for k in range(k1, look_end + 1) if i_cls[p][k] != "high"), k1)
                back_to_load[p] = round(float(t_ms[end_k] - cycle_ms / 2), 1)
                events.append(TraceEvent(back_to_load[p], p, "current_back_to_load", _round(float(i_rms[p][end_k]))))

        # The fault's own phases: high or sagging BEFORE the first current
        # ceases. When the line opens, every phase voltage passes through the
        # sag band on its way to dead, which says nothing about the fault (a
        # healthy phase's load zero is no part of the fault clearing).
        fault_end_k = k1
        if ceased:
            fault_end_k = max(k0, int(np.searchsorted(t_ms, min(ceased.values()))) - 1)
        high_phases = sorted(p for p in i_cls if any(i_cls[p][k] == "high" for k in range(k0, fault_end_k + 1)))
        sag_phases = sorted(p for p in v_cls if any(v_cls[p][k] == "sag" for k in range(k0, fault_end_k + 1)))
        involved = sorted(set(high_phases) | set(sag_phases))

        after_return = [r for r in all_returns if r <= onset_ms + 0.5 * cycle_ms]
        energization = bool(after_return) and not ceased and (onset_ms - max(after_return)) <= _ENERGIZATION_WINDOW_CYCLES * cycle_ms
        kind = "energization" if energization else "fault"
        for p in involved:
            events.append(TraceEvent(onset_ms, p, "disturbance_start" if kind == "fault" else "energization_start",
                                     _round(float(np.nanmax(i_rms[p][k0:k1 + 1])))))
        clearing = None
        fault_phase_zeros = [ceased[p] for p in involved if p in ceased]
        if fault_phase_zeros:
            clearing = max(fault_phase_zeros)
        elif back_to_load:
            clearing = max(back_to_load.values())
        disturbances.append({
            "kind": kind,
            "start_ms": onset_ms,
            "end_ms": round(float(t_ms[k1]), 1),
            "high_current_phases": high_phases,
            "sagged_phases": sag_phases,
            "ceased": ceased,
            "back_to_load": back_to_load,
            "clearing_ms": clearing,
        })

    # --- summary -------------------------------------------------------------------------------------
    faults = [d for d in disturbances if d["kind"] == "fault"]
    summary: dict[str, Any] = {
        "starts_dead": starts_dead,
        "fault_start_ms": None,
        "fault_clearing_ms": None,
        "fct_ms": None,
        "high_current_phases": [],
        "sagged_phases": [],
        "ceased": {},
        "zero_current_pattern": None,
        "zero_spans": [],
        "current_returns": {},
        "voltage_returns": {},
        "dead_time_ms": {},
        "energization_ms": None,
        "refault_ms": None,
        "disturbances": disturbances,
    }
    first = faults[0] if faults else None
    if first:
        summary.update({
            "fault_start_ms": first["start_ms"],
            "fault_clearing_ms": first["clearing_ms"],
            "fct_ms": round(first["clearing_ms"] - first["start_ms"], 1) if first["clearing_ms"] is not None else None,
            "high_current_phases": first["high_current_phases"],
            "sagged_phases": first["sagged_phases"],
            "ceased": first["ceased"],
        })
        ceased = first["ceased"]
        if ceased:
            times = sorted(ceased.values())
            together = (times[-1] - times[0]) <= _TOGETHER_CYCLES * cycle_ms
            still_live = [p for p in i_cls if p not in ceased]
            if len(ceased) == 3:
                summary["zero_current_pattern"] = "three_together" if together else "staggered"
            elif len(ceased) == 1 and len(still_live) == 2:
                summary["zero_current_pattern"] = "single_phase"
            elif len(ceased) == 2 and len(still_live) == 1:
                summary["zero_current_pattern"] = "two_phase" if together else "staggered"
            else:
                summary["zero_current_pattern"] = "staggered"
        for p, t_zero in ceased.items():
            ret = next((r for r in returns_ms.get(p, []) if r > t_zero), None)
            v_ret = next((r for r in voltage_returns_ms.get(p, []) if r > t_zero), None)
            summary["zero_spans"].append({"phase": p, "start_ms": t_zero, "end_ms": ret})
            if ret is not None:
                summary["current_returns"][p] = ret
                summary["dead_time_ms"][p] = round(ret - t_zero, 1)
            elif v_ret is not None:
                summary["voltage_returns"][p] = v_ret
                summary["dead_time_ms"][p] = round(v_ret - t_zero, 1)
        return_ms = min(summary["current_returns"].values(), default=np.inf)
        # Windowed RMS onset can precede its current-return edge by a few
        # samples. A later fault that clears again is still a refault.
        later = [d for d in faults[1:] if d["start_ms"] >= return_ms - cycle_ms
                 and d.get("clearing_ms") is not None and d["clearing_ms"] > return_ms]
        if later:
            summary["refault_ms"] = later[0]["start_ms"]
    if starts_dead:
        for p in i_cls:
            if returns_ms.get(p):
                summary["current_returns"][p] = returns_ms[p][0]
        for p in v_cls:
            if voltage_returns_ms.get(p) and p not in summary["current_returns"]:
                summary["voltage_returns"][p] = voltage_returns_ms[p][0]
        energized = [d for d in disturbances if d["kind"] == "energization"]
        if energized:
            summary["energization_ms"] = energized[0]["start_ms"]
        if faults:
            summary["refault_ms"] = faults[0]["start_ms"]

    events.sort(key=lambda e: (e.t_ms, e.phase, e.kind))
    reference = {
        "current_unit": current_unit,
        "voltage_unit": voltage_unit,
        "load_current": {p: _round(v) for p, v in load.items()},
        "nominal_voltage": {p: _round(v, 3) for p, v in nominal_v.items()},
        "none_threshold": _round(none_thr),
        "high_threshold": _round(high_thr),
    }
    return AnalogTrace(
        cycle_ms=round(cycle_ms, 3),
        reference=reference,
        phases=phases,
        events=events,
        summary=summary,
        warnings=warnings,
    )


def _key_runs(keys: list) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    k = 0
    while k < len(keys):
        j = k
        while j + 1 < len(keys) and keys[j + 1] == keys[k]:
            j += 1
        out.append((k, j))
        k = j + 1
    return out
