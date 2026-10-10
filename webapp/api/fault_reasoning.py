"""The reasoning chain behind one record's fault conclusions.

One place draws each per-fault conclusion from an explicit rule, the IDs in
``docs/fault-reasoning-rules.md``: is there a fault (F1), which phases and
whether to ground (F4), when it cleared and whether that met the Grid Code
limit (F3), which protection tripped and by what path (F5), and how the
breaker tripped and reclosed (F6). Every conclusion keeps its value, the rule
IDs, the evidence it rests on, a confidence and any conflict, so the incident
page, the single-record page and the reports can show the same reasoning
instead of recomputing it.

Built from facts computed once per record (``record_facts``, the analog
trace, the event window) plus the record's own status channels — a channel
that was recorded and stayed quiet is evidence too (P4: ``DIST Sig. Send``
never asserting while Z2 tripped with a receive points to PUTT).

Text follows the UI convention: Indonesian narrative, English technical
terms, PLN phase names (R/S/T for A/B/C). Times are ms after the fault start
(the event window's inception, F3.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from core.event_analysis import EventWindow
from .record_facts import classify_status_channel
from .routers.relay_21 import _fundamental_phasor

SCHEMA = "reasoning.v1"

_PLN = {"A": "R", "B": "S", "C": "T"}
_ORDER = ("A", "B", "C")

# F5.2: PLN zone timers (user, 9 Oct 2026), tolerance ±10%.
_ZONE_TIMERS_S = {2: (0.4, 0.8), 3: (1.2, 1.6)}
_TIMER_TOLERANCE = 0.10
# F3.3: breaker interruption, trip command to current zero.
_INTERRUPTION_MS = (30.0, 80.0)
# F3.5: Grid Code (Permen ESDM 20/2020, CCA1 2.2) primary-protection FCT limits.
_FCT_LIMIT_MS = ((500.0, 90.0), (275.0, 100.0), (150.0, 120.0), (66.0, 150.0))
_NOMINAL_KV = (20.0, 66.0, 70.0, 150.0, 275.0, 500.0)
# F5.5: default reach of switch-on-to-fault / trip-on-reclose after a breaker close.
_TOR_WINDOW_MS = 1000.0
# F4.1: ground when I0/I1 exceeds this.
_GROUND_I0_I1 = 0.2

_RECEIVE = {"CR", "RCV", "RECV", "RECEIVE", "RECEIVED", "RX"}
_SEND = {"CS", "SEND", "SENDING", "SND", "TX"}
_SOTF = {"SOTF", "TOR"}


# --- formatting --------------------------------------------------------------------------

def _round(value: float, digits: int = 0) -> float:
    """Half away from zero, as the incident page rounds (Python's round() goes to even)."""
    factor = 10 ** digits
    return math.copysign(math.floor(abs(value) * factor + 0.5), value) / factor


def _int(value: float) -> int:
    return int(_round(value))


def _num(value: float, digits: int = 1) -> str:
    return f"{_round(value, digits):.{digits}f}".replace(".", ",")


def _ms(value: float) -> str:
    """"+36,7" for a time after the fault start."""
    return f"{'+' if value >= 0 else '−'}{_num(abs(value))}"


def _amps(value: float) -> str:
    return f"{_num(value / 1000.0)} kA" if value >= 1000.0 else f"{_int(value)} A"


def _phases_text(phases: set[str] | list[str], ground: bool = False) -> str:
    names = [_PLN[p] for p in _ORDER if p in phases]
    if not names:
        return "?"
    if len(names) == 1:
        return f"{names[0]}-N"
    return "-".join(names) + ("-N" if ground and len(names) == 2 else "")


def _pole_text(poles: set[str]) -> str:
    return "-".join(_PLN[p] for p in _ORDER if p in poles)


def _tokens(name: str) -> set[str]:
    import re
    return {t for t in re.split(r"[^A-Z0-9]+", (name or "").upper()) if t}


# --- data -----------------------------------------------------------------------------------

@dataclass
class Conclusion:
    key: str
    step: int
    label: str
    title: str
    evidence: list[str]
    rules: list[str]
    confidence: str                    # "high" | "medium" | "low" | "flag"
    value: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "step": self.step,
            "label": self.label,
            "title": self.title,
            "evidence": self.evidence,
            "rules": sorted(set(self.rules), key=_rule_order),
            "confidence": self.confidence,
            "value": self.value,
            "conflicts": self.conflicts,
        }


@dataclass
class _Channel:
    name: str
    role: Optional[str]
    zone: Optional[int]
    phase: Optional[str]
    tokens: set[str]
    on_ms: list[float]
    off_ms: list[float]
    initially_on: bool

    @property
    def asserted(self) -> bool:
        return self.initially_on or bool(self.on_ms)

    def first_on(self, after_ms: float) -> Optional[float]:
        return next((t for t in self.on_ms if t >= after_ms), None)

    def stable_intervals(self, min_ms: float = 5.0) -> list[tuple[float, float]]:
        """``(on, off)`` intervals with contact bounce removed: pulses and
        gaps shorter than ``min_ms`` are ignored (F6.4)."""
        edges = sorted([(t, True) for t in self.on_ms] + [(t, False) for t in self.off_ms])
        intervals: list[list[float]] = []
        start = -math.inf if self.initially_on else None
        for t, rising in edges:
            if rising and start is None:
                start = t
            elif not rising and start is not None:
                intervals.append([start, t])
                start = None
        if start is not None:
            intervals.append([start, math.inf])
        merged: list[list[float]] = []
        for on, off in intervals:
            if merged and on - merged[-1][1] < min_ms:
                merged[-1][1] = off
            else:
                merged.append([on, off])
        return [(on, off) for on, off in merged if off - on >= min_ms]

    def on_during(self, start_ms: float, end_ms: float) -> bool:
        """Asserted at some point within [start_ms, end_ms]."""
        if self.initially_on and (not self.off_ms or self.off_ms[0] >= start_ms):
            return True
        for i, on in enumerate(self.on_ms):
            off = next((t for t in self.off_ms if t > on), math.inf)
            if on <= end_ms and off >= start_ms:
                return True
        return False


def _channels(payload: dict) -> list[_Channel]:
    time = np.asarray(payload.get("time") or [], dtype=float)
    out: list[_Channel] = []
    for ch in payload.get("status_channels", []):
        name = str(ch.get("name") or "").strip()
        samples = np.asarray(ch.get("samples") or [], dtype=float)
        if not name or len(samples) != len(time) or len(time) < 2:
            continue
        role, zone, phase = classify_status_channel(name)
        state = samples != 0
        diff = np.diff(state.astype(int))
        out.append(_Channel(
            name=name, role=role, zone=zone, phase=phase, tokens=_tokens(name),
            on_ms=[round(float(time[i]) * 1000.0, 1) for i in (np.where(diff > 0)[0] + 1)],
            off_ms=[round(float(time[i]) * 1000.0, 1) for i in (np.where(diff < 0)[0] + 1)],
            initially_on=bool(state[0]),
        ))
    return out


def _is_receive(ch: _Channel) -> bool:
    return ch.role == "teleprotection" and bool(ch.tokens & _RECEIVE)


def _is_send(ch: _Channel) -> bool:
    return ch.role == "teleprotection" and bool(ch.tokens & _SEND) and not (ch.tokens & _RECEIVE)


def is_send_channel(name: str) -> bool:
    """A teleprotection send channel, by its name ("DIST Sig. Send", "LP SEND MJSNG2")."""
    role, _zone, _phase = classify_status_channel(name)
    tokens = _tokens(name)
    return role == "teleprotection" and bool(tokens & _SEND) and not (tokens & _RECEIVE)


def _is_sotf(ch: _Channel) -> bool:
    return bool(ch.tokens & _SOTF)


# --- phasors -----------------------------------------------------------------------------

def _scale(channels: list[dict], measurement: str) -> float:
    unit = next((str(c.get("unit") or "").strip().lower() for c in channels if c.get("measurement") == measurement), "")
    return {"kv": 1000.0, "mv": 1e6, "ka": 1000.0, "ma": 0.001}.get(unit, 1.0)


def _phase_samples(payload: dict, measurement: str) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for ch in payload.get("analog_channels", []):
        phase = str(ch.get("phase") or "").upper()
        if ch.get("measurement") == measurement and phase in _ORDER and phase not in found:
            found[phase] = np.asarray(ch.get("samples") or [], dtype=float)
    return found


@dataclass
class _Phasors:
    v: dict[str, complex]
    i: dict[str, complex]
    v_pre: dict[str, complex] = field(default_factory=dict)
    i_pre: dict[str, complex] = field(default_factory=dict)

    def superimposed(self) -> dict[str, complex]:
        """Fault minus prefault current: the fault's own contribution, without
        the load current (two whole-cycle-aligned windows, so their angles
        compare)."""
        return {p: self.i[p] - self.i_pre[p] for p in _ORDER if p in self.i and p in self.i_pre}

    def voltage_pu(self) -> dict[str, float]:
        """Each phase voltage during the fault over its own prefault value —
        read at the same instant as the loops, so the line opening later
        (a line-side VT reading zero) does not pass for a sag."""
        return {
            p: abs(self.v[p]) / abs(self.v_pre[p])
            for p in _ORDER
            if p in self.v and p in self.v_pre and abs(self.v_pre[p]) > 0
        }

    @property
    def residual(self) -> Optional[complex]:
        return sum(self.i.values()) if len(self.i) == 3 else None

    def i0_i1(self) -> Optional[float]:
        if len(self.i) != 3:
            return None
        a = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
        ia, ib, ic = self.i["A"], self.i["B"], self.i["C"]
        i0 = abs(ia + ib + ic) / 3.0
        i1 = abs(ia + a * ib + a * a * ic) / 3.0
        return i0 / i1 if i1 > 0 else None

    def loops(self, ground: bool) -> dict[str, float]:
        """|Z| (ohm, primary) of the measuring loops. Phase-phase loops always;
        phase-ground loops (K0 = 1, enough to compare loops) only for a ground fault."""
        out: dict[str, float] = {}
        if len(self.v) == 3 and len(self.i) == 3:
            for a, b in (("A", "B"), ("B", "C"), ("C", "A")):
                di = self.i[a] - self.i[b]
                if abs(di) > 0:
                    out[a + b] = abs((self.v[a] - self.v[b]) / di)
            residual = self.residual
            if ground and residual is not None:
                for p in _ORDER:
                    di = self.i[p] + residual
                    if abs(di) > 0:
                        out[p + "N"] = abs(self.v[p] / di)
        return out


def _fault_phasors(payload: dict, window: EventWindow) -> Optional[_Phasors]:
    """Fundamental phasors one cycle after inception (the second cycle of the
    fault, past the onset transient), or half a cycle for a very short fault."""
    time = np.asarray(payload.get("time") or [], dtype=float)
    if window.inception_idx is None or len(time) < 8:
        return None
    sr = 1.0 / float(np.median(np.diff(time)))
    freq = float(payload.get("frequency") or 50.0) or 50.0
    n = max(8, int(round(sr / freq)))
    i0 = int(window.inception_idx)
    clear = window.clearing_idx if window.clearing_idx is not None else i0 + 10 * n
    start = i0 + n if clear - i0 >= 2 * n else i0 + n // 2
    if start + n > len(time):
        return None
    channels = payload.get("analog_channels", [])
    v_scale, i_scale = _scale(channels, "voltage"), _scale(channels, "current")
    voltages = _phase_samples(payload, "voltage")
    currents = _phase_samples(payload, "current")
    volts = {p: _fundamental_phasor(x * v_scale, start, n, freq, sr, i0) for p, x in voltages.items()}
    amps = {p: _fundamental_phasor(x * i_scale, start, n, freq, sr, i0) for p, x in currents.items()}
    # Prefault window a whole number of cycles earlier, so its angles line up.
    pre_start = start - 3 * n if start >= 3 * n else start - 2 * n
    volts_pre, amps_pre = {}, {}
    if pre_start >= 0 and pre_start + n <= i0:
        volts_pre = {p: _fundamental_phasor(x * v_scale, pre_start, n, freq, sr, i0) for p, x in voltages.items()}
        amps_pre = {p: _fundamental_phasor(x * i_scale, pre_start, n, freq, sr, i0) for p, x in currents.items()}

    def finite(values: dict[str, complex]) -> dict[str, complex]:
        return {p: z for p, z in values.items() if np.isfinite(z)}

    return _Phasors(finite(volts), finite(amps), finite(volts_pre), finite(amps_pre))


# --- the chain ---------------------------------------------------------------------------

@dataclass
class _Context:
    payload: dict
    window: Optional[EventWindow]
    trace: dict[str, Any]
    measurements: dict[str, Any]
    channels: list[_Channel]
    event_class: Optional[str]
    gate_reasons: list[str]
    t0: Optional[float]                 # fault start (ms, record axis)
    nominal_kv: Optional[float]

    @property
    def summary(self) -> dict[str, Any]:
        return self.trace.get("summary") or {}

    def rel(self, t_ms: Optional[float]) -> Optional[float]:
        return None if t_ms is None or self.t0 is None else t_ms - self.t0


def _amps_scale(measurements: dict[str, Any]) -> float:
    return 1000.0 if str(measurements.get("current_unit") or "").lower() == "ka" else 1.0


def _volts_kv_scale(measurements: dict[str, Any]) -> float:
    unit = str(measurements.get("voltage_unit") or "").lower()
    return {"kv": 1.0, "v": 0.001, "mv": 1000.0}.get(unit, 1.0)


def _nominal_kv(measurements: dict[str, Any], given: Optional[float]) -> Optional[float]:
    """The system's nominal voltage class from the prefault phase voltage."""
    if given:
        return float(given)
    pre = ((measurements.get("prefault") or {}).get("voltage_rms") or {})
    values = [v for v in pre.values() if isinstance(v, (int, float)) and v > 0]
    if not values:
        return None
    line_kv = float(np.mean(values)) * _volts_kv_scale(measurements) * math.sqrt(3.0)
    nearest = min(_NOMINAL_KV, key=lambda kv: abs(kv - line_kv))
    return nearest if abs(nearest - line_kv) <= 0.2 * nearest else None


def _fct_limit_ms(kv: Optional[float]) -> Optional[float]:
    if kv is None:
        return None
    if 60.0 <= kv <= 80.0:  # 66/70 kV class
        return 150.0
    return next((limit for level, limit in _FCT_LIMIT_MS if abs(level - kv) < 1.0), None)


def _voltage_pu(ctx: _Context, phasors: Optional[_Phasors] = None) -> dict[str, float]:
    """Per-phase voltage during the fault, per unit of prefault: from the
    fault phasors when there are, else from the lowest one-cycle RMS."""
    if phasors is not None:
        pu = phasors.voltage_pu()
        if pu:
            return pu
    pre = ((ctx.measurements.get("prefault") or {}).get("voltage_rms") or {})
    low = ((ctx.measurements.get("fault") or {}).get("voltage_rms_min") or {})
    out = {}
    for p in _ORDER:
        if isinstance(pre.get(p), (int, float)) and isinstance(low.get(p), (int, float)) and pre[p] > 0:
            out[p] = float(low[p]) / float(pre[p])
    return out


def _fault_current(ctx: _Context, phasors: Optional[_Phasors] = None) -> dict[str, float]:
    """Per-phase fault current RMS (A): from the fault phasors when there
    are, else the highest one-cycle RMS between inception and clearing."""
    if phasors is not None and len(phasors.i) == 3:
        return {p: abs(z) / math.sqrt(2.0) for p, z in phasors.i.items()}
    rms = ((ctx.measurements.get("fault") or {}).get("current_rms_max") or {})
    scale = _amps_scale(ctx.measurements)
    return {p: float(v) * scale for p, v in rms.items() if isinstance(v, (int, float))}


def _prefault_current(ctx: _Context) -> dict[str, float]:
    rms = ((ctx.measurements.get("prefault") or {}).get("current_rms") or {})
    scale = _amps_scale(ctx.measurements)
    return {p: float(v) * scale for p, v in rms.items() if isinstance(v, (int, float))}


# Step 1 ----------------------------------------------------------------------------------

def _fault_presence(ctx: _Context, trip: Optional[dict[str, Any]], phasors: Optional[_Phasors]) -> Conclusion:
    if ctx.event_class == "RECLOSE_CAPTURE":
        return Conclusion(
            "fault", 1, "Gangguan", "Rekaman reclose, bukan gangguan",
            ["Rekaman dimulai saat PMT terbuka (dead time) dan hanya menangkap reclose.",
             "Penyebab tidak diklasifikasi dari rekaman ini."],
            ["F1.2"], "high", {"present": False, "class": "reclose_capture"},
        )
    if ctx.event_class == "NO_FAULT_TRIGGER":
        return Conclusion(
            "fault", 1, "Gangguan", "Tidak ada gangguan",
            [r[0].upper() + r[1:] + "." for r in ctx.gate_reasons if r],
            ["F1.1"], "medium", {"present": False, "class": "no_fault"},
        )
    evidence: list[str] = []
    peak = (ctx.measurements.get("fault") or {}).get("current_peak") or {}
    pre = _prefault_current(ctx)
    scale = _amps_scale(ctx.measurements)
    if peak:
        p = max(peak, key=lambda k: peak[k] or 0)
        line = f"Arus {_PLN[p]} naik ke {_amps(float(peak[p]) * scale)} puncak"
        if p in pre:
            line += f" (prefault {_amps(pre[p])} rms)"
        evidence.append(line + ".")
    pu = _voltage_pu(ctx, phasors)
    if pu:
        p = min(pu, key=pu.get)
        if pu[p] < 0.9:
            evidence.append(f"Tegangan {_PLN[p]} turun ke {_num(pu[p], 2)} pu.")
    operated = trip is not None
    if operated:
        evidence.append(f"Proteksi operate: {trip['channel']} {_ms(trip['at_ms'])} ms.")
    title = "Ya, dengan proteksi bekerja" if operated else "Ya, dari gelombang — tidak ada kanal trip yang aktif"
    confidence = "high" if operated and (peak or pu) else "medium"
    return Conclusion("fault", 1, "Gangguan", title, evidence, ["F1.1"], confidence,
                      {"present": True, "protection_operated": operated})


# Step 4 ----------------------------------------------------------------------------------

def _phases(ctx: _Context, trip: Optional[dict[str, Any]], phasors: Optional[_Phasors]) -> tuple[Conclusion, set[str], bool]:
    evidence: list[str] = []
    rules: list[str] = []
    sources: dict[str, set[str]] = {}

    pu = _voltage_pu(ctx, phasors)
    if pu:
        depth = {p: 1.0 - v for p, v in pu.items()}
        deepest = max(depth.values())
        if deepest >= 0.1:
            by_voltage = {p for p, d in depth.items() if d >= 0.5 * deepest and pu[p] < 0.9}
            sources["voltage"] = by_voltage
            others = [p for p in _ORDER if p in pu and p not in by_voltage]
            text = "; ".join(f"{_PLN[p]} {_num(pu[p], 2)}" for p in _ORDER if p in by_voltage)
            if others:
                text += " pu; " + " dan ".join(f"{_PLN[p]} {_num(pu[p], 2)}" for p in others) + " pu."
            else:
                text += " pu."
            evidence.append(f"Tegangan {text}")
            rules.append("F4.2")

    i0_i1 = phasors.i0_i1() if phasors else None
    ground: Optional[bool] = None if i0_i1 is None else i0_i1 > _GROUND_I0_I1

    loops = phasors.loops(bool(ground)) if phasors else {}
    if loops:
        best = min(loops, key=loops.get)
        rest = [z for k, z in loops.items() if k != best]
        loop_phases = {c for c in best if c in _ORDER}
        sources["loop"] = loop_phases
        name = f"{_PLN[best[0]]}-{'N' if best[1] == 'N' else _PLN[best[1]]}"
        line = f"Loop {name} {_num(loops[best])} Ω"
        if rest:
            line += f"; loop lain ≥ {_num(min(rest), 0) if min(rest) >= 10 else _num(min(rest))} Ω"
        if i0_i1 is not None:
            line += f". I0/I1 {_num(i0_i1, 2)}."
        evidence.append(line)
        rules += ["F4.1", "F4.3"]
    elif i0_i1 is not None:
        evidence.append(f"I0/I1 {_num(i0_i1, 2)}.")
        rules.append("F4.1")

    # F4.4 / F4.5: the relay's own phase selection, from which poles it tripped.
    if trip is not None:
        if trip["poles"] and not trip["three_pole"] and len(trip["poles"]) == 1:
            sources["relay"] = set(trip["poles"])
            rules.append("F4.4")
        elif trip["three_pole"]:
            rules.append("F4.5")

    # The answer: voltage first, then loops, then the waveform trace, then the event window.
    phases = sources.get("voltage") or sources.get("loop") or set(ctx.summary.get("high_current_phases") or [])
    if not phases and ctx.window is not None:
        phases = set(ctx.window.faulted_phases or [])
    if ground is None:
        ground = len(phases) == 1

    # F4.7: the currents must fit the fault type.
    current = _fault_current(ctx, phasors)
    conflicts: list[str] = []
    flags: list[str] = []
    if phases and current:
        if len(phases) == 1:
            (p,) = tuple(phases)
            others = {q: v for q, v in current.items() if q != p}
            if p in current and others:
                q = max(others, key=others.get)
                ratio = others[q] / current[p] if current[p] > 0 else 0.0
                if 0.1 <= ratio < 0.5:
                    evidence.append(
                        f"Arus {_PLN[q]} {_amps(others[q])} = {_int(ratio * 100)}% arus {_PLN[p]}, jadi bukan gangguan dua fasa."
                    )
                    rules.append("F4.7")
        elif len(phases) == 2:
            p, q = [x for x in _ORDER if x in phases]
            # The fault's own contribution (load removed): an LL fault drives
            # equal and opposite currents in its two phases. Total currents
            # differ at a weak-infeed end, where load dominates.
            delta = phasors.superimposed() if phasors else {}
            if p in delta and q in delta and max(abs(delta[p]), abs(delta[q])) > 0:
                dp, dq = abs(delta[p]) / math.sqrt(2.0), abs(delta[q]) / math.sqrt(2.0)
                ratio = min(dp, dq) / max(dp, dq)
                line = f"Arus gangguan (tanpa beban) {_PLN[p]} {_amps(dp)} dan {_PLN[q]} {_amps(dq)}"
                if not ground:
                    angle = abs(math.degrees(np.angle(delta[p] / delta[q])))
                    line += f", beda sudut {_int(angle)}°"
            elif p in current and q in current:
                ratio = min(current[p], current[q]) / max(current[p], current[q])
                line = f"Arus {_PLN[p]} {_amps(current[p])} dan {_PLN[q]} {_amps(current[q])}"
            else:
                ratio, line = None, ""
            if line:
                evidence.append(line + ".")
                rules.append("F4.7")
                if ratio is not None and ratio < 0.5:
                    flags.append(f"Arus {_PLN[p]} dan {_PLN[q]} jauh berbeda — tidak cocok dengan gangguan dua fasa.")

    # F7.4: an end whose fault contribution is under twice its load is a
    # weak-infeed end; its phases come from the voltage, not the currents.
    weak_infeed = False
    load = max(_prefault_current(ctx).values(), default=0.0)
    delta = phasors.superimposed() if phasors else {}
    contribution = max((abs(delta[p]) / math.sqrt(2.0) for p in phases if p in delta), default=None)
    if contribution is not None and load > 0 and contribution < 2.0 * load:
        weak_infeed = True
        evidence.append(
            f"Kontribusi arus gangguan hanya {_amps(contribution)}, kurang dari 2× arus beban {_amps(load)}: "
            "ujung weak infeed, fasa dibaca dari tegangan."
        )
        rules.append("F7.4")

    if trip is not None:
        if trip["three_pole"]:
            evidence.append("Trip 3-pole tidak dipakai sebagai bukti fasa.")
        elif len(trip["poles"]) == 1:
            (pole,) = tuple(trip["poles"])
            evidence.append(f"Relay hanya trip pole {_PLN[pole]}.")
            # F4.8: waveform and relay disagree.
            if len(phases) >= 2:
                conflicts.append(
                    f"Gelombang menunjukkan gangguan {_phases_text(phases, bool(ground))}, tetapi relay hanya trip pole "
                    f"{_PLN[pole]}: maloperasi, gangguan berkembang, atau masalah CT/VT."
                )
            elif phases and pole not in phases:
                conflicts.append(
                    f"Gelombang menunjukkan fasa {_phases_text(phases)}, tetapi relay trip pole {_PLN[pole]}."
                )
            if conflicts:
                rules.append("F4.8")

    if len(phases) == 1:
        kind = "satu fasa ke tanah"
    elif len(phases) == 2:
        kind = "dua fasa ke tanah" if ground else "antar fasa, tidak ke tanah"
    elif len(phases) == 3:
        kind = "tiga fasa"
    else:
        kind = "fasa tidak dapat ditentukan"
    agreeing = [name for name, found in sources.items() if found and found == phases]
    if conflicts or not phases:
        confidence = "low"
    elif len(agreeing) >= 2:
        confidence = "high"
    else:
        confidence = "medium"
    title = f"{_phases_text(phases, bool(ground))}, {kind}" if phases else "Fasa tidak dapat ditentukan"
    value = {
        "phases": [p for p in _ORDER if p in phases],
        "ground": bool(ground),
        "label": _phases_text(phases, bool(ground)) if phases else None,
        "voltage_pu": {p: round(v, 3) for p, v in pu.items()},
        "loops_ohm": {k: round(v, 3) for k, v in loops.items()},
        "i0_i1": round(i0_i1, 3) if i0_i1 is not None else None,
        "agreeing_sources": agreeing,
        "weak_infeed": weak_infeed,
        "flags": flags,
    }
    return (
        Conclusion("phases", 4, "Fasa", title, evidence, sorted(set(rules), key=_rule_order), confidence, value, conflicts),
        phases,
        bool(ground),
    )


def _rule_order(rule: str) -> tuple[int, ...]:
    return tuple(int(x) for x in rule[1:].split("."))


# Step 5 ----------------------------------------------------------------------------------

def _trip(ctx: _Context) -> Optional[dict[str, Any]]:
    """The first trip command after the fault started, with its poles."""
    if ctx.t0 is None:
        return None
    earliest = ctx.t0 - 20.0
    trips = [(ch, ch.first_on(earliest)) for ch in ctx.channels if ch.role == "trip"]
    trips = [(ch, t) for ch, t in trips if t is not None]
    if not trips:
        return None
    first_ch, at = min(trips, key=lambda item: item[1])
    together = [(ch, t) for ch, t in trips if t <= at + 10.0]
    poles: set[str] = set()
    three_pole = False
    for ch, _t in together:
        if ch.phase == "3P":
            three_pole = True
        elif ch.phase in _ORDER:
            poles.add(ch.phase)
    if poles == set(_ORDER):
        three_pole = True
    zone_trips = [
        {"name": ch.name, "zone": ch.zone, "at_ms": round(t - ctx.t0, 1)}
        for ch, t in together if ch.zone is not None
    ]
    return {
        "channel": first_ch.name,
        "at_ms": round(at - ctx.t0, 1),
        "abs_ms": at,
        "poles": poles,
        "three_pole": three_pole,
        "named_zones": sorted({z["zone"] for z in zone_trips}),
        "zone_trips": zone_trips,
        "channels": [ch.name for ch, _t in together],
    }


def _timer_match(zone: int, elapsed_s: float) -> Optional[float]:
    for timer in _ZONE_TIMERS_S.get(zone, ()):
        if abs(elapsed_s - timer) <= _TIMER_TOLERANCE * timer:
            return timer
    return None


def _trip_path(ctx: _Context, trip: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Which element tripped and how (F5.1–F5.5), from zone pickups, the trip
    time against the zone timers, teleprotection and SOTF/TOR channels."""
    path: dict[str, Any] = {"kind": "no_trip" if trip is None else "unknown"}
    if ctx.t0 is None:
        return path
    earliest = ctx.t0 - 20.0
    until = trip["abs_ms"] + 10.0 if trip else math.inf
    # Zone pickups (start) channels; a trip output naming its zone ("TRIP Z1")
    # says which zone tripped, not when it started.
    zone_on: dict[int, float] = {}
    zone_recorded: set[int] = set()
    for ch in ctx.channels:
        if ch.zone is None or ch.role != "zone":
            continue
        zone_recorded.add(ch.zone)
        on = ch.first_on(earliest)
        if on is not None and on <= until:
            zone_on[ch.zone] = min(on, zone_on.get(ch.zone, math.inf))
    receive = [ch for ch in ctx.channels if _is_receive(ch)]
    send = [ch for ch in ctx.channels if _is_send(ch)]
    sotf = [ch for ch in ctx.channels if _is_sotf(ch)]
    path.update({
        "zones_on_ms": {z: round(t - ctx.t0, 1) for z, t in sorted(zone_on.items())},
        "zones_recorded": sorted(zone_recorded),
        "receive_recorded": bool(receive),
        "send_recorded": bool(send),
        "sotf_recorded": bool(sotf),
    })
    if trip is None:
        dropped = [z for z in (2, 3) if z in zone_on]
        if dropped:
            path["kind"] = "pickup_without_trip"
        return path

    at = trip["abs_ms"]
    window = (min(zone_on.values(), default=at) - 5.0, at + 10.0)
    path["receive_at_trip"] = next((ch.name for ch in receive if ch.on_during(*window)), None)
    path["send_active"] = next((ch.name for ch in send if ch.on_during(window[0], at + 10.0)), None)
    path["sotf_active"] = next((ch.name for ch in sotf if ch.on_during(at - 10.0, at + 10.0)), None)

    path["trip_zones"] = list(trip["named_zones"])
    if path["sotf_active"]:
        path["kind"] = "sotf"
        return path
    if 1 in trip["named_zones"] or 1 in zone_on:
        path["kind"] = "z1"
        return path
    for zone in (2, 3):
        if zone in zone_on:
            started = zone_on[zone]
        elif zone in trip["named_zones"]:
            # Only the trip output names the zone: time it from the fault start.
            started = ctx.t0
            path["timed_from_fault_start"] = True
        else:
            continue
        elapsed_s = (at - started) / 1000.0
        path["elapsed_ms"] = round(elapsed_s * 1000.0, 1)
        path["pickup_zone"] = zone
        timer = _timer_match(zone, elapsed_s)
        if timer is not None:
            path["kind"] = f"z{zone}_delayed"
            path["timer_s"] = timer
        elif zone == 2 and elapsed_s < min(_ZONE_TIMERS_S[2]) * (1 - _TIMER_TOLERANCE):
            # F5.3: only Z2 is accelerated by teleprotection.
            path["kind"] = "accelerated"
        else:
            path["kind"] = "timer_mismatch"
        break
    else:
        path["kind"] = "trip_without_zone"

    # F5.5: tripped soon after a breaker close, with no receive -> likely TOR.
    close_ms = _last_close_before(ctx, at)
    if close_ms is not None and at - close_ms <= _TOR_WINDOW_MS and not path["receive_at_trip"]:
        path["closed_before_ms"] = round(at - close_ms, 1)
        if path["kind"] in ("accelerated", "timer_mismatch", "trip_without_zone"):
            path["kind"] = "likely_tor"
    return path


def _last_close_before(ctx: _Context, at_ms: float) -> Optional[float]:
    """The last breaker close (reclose or energization) before ``at_ms``."""
    candidates = []
    if ctx.window is not None:
        for event in ctx.window.reclose_events or []:
            t = event.get("time")
            if isinstance(t, (int, float)) and t * 1000.0 < at_ms:
                candidates.append(float(t) * 1000.0)
    energized = ctx.summary.get("energization_ms")
    if isinstance(energized, (int, float)) and energized < at_ms:
        candidates.append(float(energized))
    return max(candidates) if candidates else None


def _trip_path_conclusion(ctx: _Context, trip: Optional[dict[str, Any]], path: dict[str, Any]) -> Optional[Conclusion]:
    kind = path["kind"]
    zones_on = path.get("zones_on_ms") or {}
    evidence: list[str] = []
    pickups = "; ".join(f"Z{z} pickup {_ms(t)} ms" for z, t in zones_on.items())
    trip_text = (
        f"trip {('pole ' + _pole_text(trip['poles'])) if trip['poles'] and not trip['three_pole'] else '3-pole' if trip['three_pole'] else ''}"
        .rstrip() + f" {_ms(trip['at_ms'])} ms"
        if trip else ""
    )
    if kind == "no_trip":
        return _no_trip_conclusion(ctx, path)
    if kind == "pickup_without_trip":
        return Conclusion(
            "trip_path", 5, "Jalur trip", "Zona pickup lalu reset tanpa trip lokal",
            [pickups + ".", "Gangguan di luar line (di balik GI lawan) dan diputus proteksi lain; zona di sini hanya standby."],
            ["F5.1", "F5.4"], "medium", path,
        )
    if trip and path.get("receive_at_trip"):
        trip_text += f" bersamaan dengan {path['receive_at_trip']}"
    if trip:
        for zone_trip in trip["zone_trips"]:
            if zone_trip["name"] != trip["channel"]:
                trip_text += f"; {zone_trip['name']} {_ms(zone_trip['at_ms'])} ms"
    first_line = "; ".join(x for x in (pickups, trip_text) if x)
    if first_line:
        evidence.append(first_line[0].upper() + first_line[1:] + ".")
    zone = path.get("pickup_zone")
    since = (
        "setelah awal gangguan (kanal pickup zona tidak direkam)"
        if path.get("timed_from_fault_start") else f"setelah Z{zone} pickup"
    )

    if kind == "sotf":
        title, rules, confidence = "SOTF/TOR", ["F5.5"], "high"
        evidence.append(f"Kanal {path['sotf_active']} aktif saat trip.")
    elif kind == "z1":
        title, rules, confidence = "Z1, seketika", ["F5.1", "F5.2"], "high"
        if not path.get("receive_recorded") and not path.get("send_recorded"):
            evidence.append("Kanal teleproteksi tidak direkam; skema tidak dapat ditentukan, dan tidak diperlukan untuk trip Z1.")
    elif kind == "accelerated":
        title, rules = "Aided trip lewat teleproteksi", ["F5.1", "F5.2", "F5.3"]
        evidence.append(
            f"Trip {_num(path['elapsed_ms'], 0)} ms {since}, jauh sebelum timer Z2 0,4 s ±10%. "
            + ("Z1 tidak pickup." if 1 in path.get("zones_recorded", []) else "Kanal Z1 tidak direkam.")
        )
        confidence = "high" if path.get("receive_at_trip") else "medium"
        if not path.get("receive_at_trip"):
            title = "Trip dipercepat (sebelum timer Z2)"
    elif kind in ("z2_delayed", "z3_delayed"):
        title, rules, confidence = f"Z{zone} waktu tunda", ["F5.1", "F5.2"], "high"
        evidence.append(
            f"Trip {_num(path['elapsed_ms'], 0)} ms {since}, cocok dengan timer {_num(path['timer_s'], 1)} s ±10%: "
            "proteksi cadangan, teleproteksi tidak membantu."
        )
    elif kind == "likely_tor":
        title, rules, confidence = "Kemungkinan TOR (trip setelah PMT menutup)", ["F5.5"], "medium"
        evidence.append(
            f"Trip {_num(path['closed_before_ms'], 0)} ms setelah PMT menutup, lebih cepat dari timer zona dan tanpa receive."
            + ("" if path.get("sotf_recorded") else " Kanal SOTF/TOR tidak direkam.")
        )
    elif kind == "timer_mismatch":
        title, rules, confidence = "Waktu trip tidak cocok dengan timer zona", ["F5.2"], "low"
        evidence.append(f"Trip {_num(path['elapsed_ms'], 0)} ms {since}; timer PLN Z2 0,4/0,8 s, Z3 1,2/1,6 s.")
    else:  # trip_without_zone
        title, rules, confidence = "Trip, elemen yang men-trip tidak terekam", ["F5.1"], "low"
        evidence.append("Tidak ada kanal zona yang aktif sebelum trip.")
    return Conclusion("trip_path", 5, "Jalur trip", title, evidence, rules, confidence, path)


def _no_trip_conclusion(ctx: _Context, path: dict[str, Any]) -> Conclusion:
    """No trip channel asserted. Whether the line's own breaker opened anyway
    is read from the currents: all of them stopping means it did."""
    trip_recorded = any(ch.role == "trip" for ch in ctx.channels)
    clearing = ctx.summary.get("fault_clearing_ms")
    cleared = ctx.rel(clearing) if isinstance(clearing, (int, float)) else None
    opened = ctx.summary.get("zero_current_pattern") in ("three_together", "staggered", "single_phase", "two_phase")
    if not opened:
        if trip_recorded:
            return Conclusion(
                "trip_path", 5, "Jalur trip", "Tidak ada trip di rekaman ini",
                ["Kanal trip direkam tetapi tidak aktif dan arus tidak padam: gangguan diputus proteksi lain, atau di luar line."],
                ["F5.1"], "medium", path,
            )
        return Conclusion(
            "trip_path", 5, "Jalur trip", "Kanal trip tidak direkam",
            ["Rekaman ini tidak punya kanal trip atau zona; jalur trip dibaca dari rekaman relay."],
            ["F5.1"], "low", path,
        )

    when = f" ±{_int(cleared)} ms" if cleared is not None else ""
    evidence = [
        (f"Kanal trip direkam tetapi tidak aktif, padahal arus padam{when}: trip PMT tidak tercatat di DFR ini."
         if trip_recorded else
         f"Kanal trip tidak direkam; arus padam{when} (dari gelombang).")
    ]
    title = f"PMT terbuka{when}, trip tidak terekam"
    confidence = "low"
    # A receive answered by this end's own send right after: the signal echoed
    # back, as a POTT scheme does from a weak-infeed end (which may then trip
    # on its weak-infeed logic, an output a DFR is often not wired to).
    receive = [(ch, ch.first_on(ctx.t0 - 20.0)) for ch in ctx.channels if _is_receive(ch)]
    send = [(ch, ch.first_on(ctx.t0 - 20.0)) for ch in ctx.channels if _is_send(ch)]
    receive = [(ch, t) for ch, t in receive if t is not None]
    send = [(ch, t) for ch, t in send if t is not None]
    if receive and send:
        rx_ch, rx = min(receive, key=lambda item: item[1])
        tx_ch, tx = min(send, key=lambda item: item[1])
        if 0.0 <= tx - rx <= 30.0 and (clearing is None or rx <= clearing):
            evidence.append(
                f"{rx_ch.name} {_ms(rx - ctx.t0)} ms lalu {tx_ch.name} {_ms(tx - ctx.t0)} ms: sinyal dipantulkan (echo), "
                "pola skema POTT dari ujung weak infeed. Trip kemungkinan dari logika weak infeed yang tidak dipetakan ke DFR."
            )
            title = f"Echo teleproteksi lalu PMT terbuka{when} (trip tidak terekam)"
            confidence = "medium"
            path = {**path, "echo": {"receive_ms": round(rx - ctx.t0, 1), "send_ms": round(tx - ctx.t0, 1)}}
    return Conclusion("trip_path", 5, "Jalur trip", title, evidence, ["F5.1", "F5.4"], confidence, path)


def _scheme_conclusion(path: dict[str, Any]) -> Optional[Conclusion]:
    """F5.4: the teleprotection scheme, read from what the record shows."""
    if path["kind"] != "accelerated":
        return None
    if path.get("receive_at_trip"):
        if path.get("send_active"):
            return Conclusion("scheme", 5, "Skema", "POTT",
                              [f"Kanal {path['send_active']} aktif saat Z2 pickup: sisi ini ikut mengirim izin (overreach)."],
                              ["F5.4"], "high", {"scheme": "POTT"})
        if path.get("send_recorded"):
            return Conclusion("scheme", 5, "Skema", "PUTT", [
                "Kanal Send direkam dan tidak pernah aktif. Pada POTT, sisi ini akan mengirim begitu Z2 pickup.",
                "Bergantung pada kanal Send yang memang terhubung.",
            ], ["F5.4"], "medium", {"scheme": "PUTT"})
        return Conclusion("scheme", 5, "Skema", "Permissive (PUTT atau POTT)",
                          ["Ada receive saat trip, tetapi kanal Send tidak direkam, jadi PUTT dan POTT tidak dapat dibedakan."],
                          ["F5.4"], "medium", {"scheme": "permissive"})
    if path.get("receive_recorded"):
        return Conclusion("scheme", 5, "Skema", "Blocking (DCB)", [
            "Kanal receive direkam tetapi tidak aktif, dan trip menyusul Z2 dengan jeda pendek: tidak ada sinyal blok, jadi trip diizinkan.",
            "Bisa juga unblocking dengan loss-of-guard.",
        ], ["F5.4"], "medium", {"scheme": "blocking"})
    return Conclusion("scheme", 5, "Skema", "Tidak dapat ditentukan",
                      ["Kanal teleproteksi (send/receive) tidak direkam."], ["F5.4"], "low", {"scheme": None})


def _location_conclusion(path: dict[str, Any], scheme: Optional[Conclusion]) -> Optional[Conclusion]:
    """F5.7: what the tripping zone says about where the fault is."""
    kind = path["kind"]
    if kind == "accelerated" and 1 in path.get("zones_recorded", []) and 1 not in path.get("zones_on_ms", {}):
        overreach = scheme is not None and scheme.value.get("scheme") == "POTT"
        return Conclusion("location", 5, "Lokasi", "Di luar jangkauan Z1 GI ini, dekat GI lawan", [
            "Z1 tidak pickup dan trip lewat aided. GI lawan seharusnya melihat gangguan di "
            + ("zona overreach-nya lalu mengirim izin." if overreach else "Z1 lalu mengirim sinyal."),
        ], ["F5.7"], "medium", {"zone_reach": "beyond_z1"})
    if kind == "z1":
        return Conclusion("location", 5, "Lokasi", "Dalam jangkauan Z1 GI ini",
                          ["Z1 umumnya menjangkau ±80% panjang line dari GI ini."], ["F5.7"], "medium",
                          {"zone_reach": "within_z1"})
    if kind == "z2_delayed":
        return Conclusion("location", 5, "Lokasi", "Di luar jangkauan Z1, dalam jangkauan Z2 GI ini",
                          ["Trip Z2 waktu tunda: gangguan di ujung line atau di luar line (zona cadangan)."], ["F5.7"],
                          "medium", {"zone_reach": "within_z2"})
    return None


def _nested_zone_flag(path: dict[str, Any], phases_value: dict[str, Any]) -> Optional[Conclusion]:
    """F5.6: forward zones nest, so a Z2 start without a Z3 start needs checking."""
    zones_on = path.get("zones_on_ms") or {}
    if 2 not in zones_on or 3 in zones_on or 3 not in path.get("zones_recorded", []):
        return None
    evidence = [
        "Zona forward bersarang: bila Z2 pickup, Z3 forward seharusnya ikut.",
        "Cek apakah Z3 di-set reverse/offset/disabled, kanal Z3 di DFR adalah output trip, atau relay hanya "
        "mengeluarkan indikasi zona eksklusif.",
    ]
    loops = phases_value.get("loops_ohm") or {}
    if loops:
        best = min(loops, key=loops.get)
        if loops[best] < 10.0:
            evidence.append(f"Loop gangguan hanya {_num(loops[best])} Ω, jadi penjelasan jangkauan resistif/load blinder gugur.")
    return Conclusion("flag_nested_zones", 9, "Ditandai", "Z3 tidak pickup padahal Z2 pickup", evidence,
                      ["F5.6"], "flag", {"zones_on_ms": zones_on})


# Step 3 ----------------------------------------------------------------------------------

def _clearing(ctx: _Context, trip: Optional[dict[str, Any]], path: dict[str, Any], phases: set[str]) -> tuple[Optional[Conclusion], list[Conclusion]]:
    flags: list[Conclusion] = []
    fct = ctx.summary.get("fct_ms")
    clearing_abs = ctx.summary.get("fault_clearing_ms")
    if not isinstance(fct, (int, float)):
        if ctx.window is None or ctx.window.fault_duration_ms is None:
            return None, flags
        return Conclusion(
            "clearing", 3, "Padam", f"±{_int(ctx.window.fault_duration_ms)} ms (dari event window)",
            ["Titik padam dari gelombang tidak ditemukan; nilai ini dari event window dan bisa berupa lebar pulsa kontak trip."],
            ["F3.2"], "low", {"fct_ms": ctx.window.fault_duration_ms, "source": "event_window"},
        ), flags

    evidence: list[str] = []
    rules = ["F3.2"]
    ceased = ctx.summary.get("ceased") or {}
    zero_phases = [p for p in _ORDER if p in ceased and (not phases or p in phases)] or [p for p in _ORDER if p in ceased]
    chain: list[str] = []
    if trip:
        chain.append(f"Trip {_ms(trip['at_ms'])} ms")
    clearing_rel = ctx.rel(clearing_abs) if isinstance(clearing_abs, (int, float)) else None
    if clearing_rel is not None:
        names = "/".join(_PLN[p] for p in zero_phases) or "fasa terganggu"
        chain.append(f"arus {names} nol ±{_int(clearing_rel)} ms")
    breaker = _first_breaker_change(ctx)
    if breaker is not None:
        chain.append(f"{breaker['name']} {_ms(breaker['at_ms'])} ms")
        rules.append("F3.4")
        if clearing_rel is not None and breaker["at_ms"] < clearing_rel - 5.0:
            flags.append(Conclusion(
                "flag_breaker_contact", 9, "Ditandai", "Kontak bantu PMT berubah sebelum arus padam",
                [f"{breaker['name']} {_ms(breaker['at_ms'])} ms, arus baru padam ±{_int(clearing_rel)} ms.",
                 "Kontak bantu seharusnya berubah setelah PMT memutus arus; cek pemetaan kanal."],
                ["F3.4"], "flag",
            ))
    if chain:
        text = " → ".join(chain)
        evidence.append(text[0].upper() + text[1:] + ".")

    interruption = None
    if trip and clearing_rel is not None:
        interruption = clearing_rel - trip["at_ms"]
        low, high = _INTERRUPTION_MS
        rules.append("F3.3")
        if low <= interruption <= high:
            evidence.append(f"Waktu interupsi PMT {_int(interruption)} ms (wajar {_int(low)}–{_int(high)} ms).")
        else:
            evidence.append(f"Waktu interupsi PMT {_int(interruption)} ms, di luar rentang wajar {_int(low)}–{_int(high)} ms.")
            flags.append(Conclusion(
                "flag_interruption", 9, "Ditandai", f"Waktu interupsi PMT {_int(interruption)} ms",
                ["Selisih arus padam − trip di luar 30–80 ms: PMT lambat, pole macet, atau kanal trip bukan output relay."],
                ["F3.3"], "flag", {"interruption_ms": round(interruption, 1)},
            ))

    limit = _fct_limit_ms(ctx.nominal_kv)
    backup = path["kind"] in ("z2_delayed", "z3_delayed")
    title = f"{_int(fct)} ms setelah gangguan"
    confidence = "high" if trip and interruption is not None and _INTERRUPTION_MS[0] <= interruption <= _INTERRUPTION_MS[1] else "medium"
    over = False
    if limit is not None:
        rules.append("F3.5")
        kv = f"{_int(ctx.nominal_kv)} kV"
        if backup:
            evidence.append(f"Trip proteksi cadangan (Z2/Z3 waktu tunda): batas FCT proteksi utama {kv} tidak berlaku.")
        elif fct <= limit:
            title += f", di bawah batas {_int(limit)} ms"
        else:
            over = True
            title += f" — melebihi batas {_int(limit)} ms"
            evidence.append(f"Batas FCT proteksi utama {kv}: {_int(limit)} ms (Grid Code, Permen ESDM 20/2020, CCA1 2.2).")
    conclusion = Conclusion(
        "clearing", 3, "Padam", title, evidence, rules, "low" if over else confidence,
        {
            "fct_ms": fct,
            "clearing_ms": round(clearing_rel, 1) if clearing_rel is not None else None,
            "trip_ms": trip["at_ms"] if trip else None,
            "interruption_ms": round(interruption, 1) if interruption is not None else None,
            "nominal_kv": ctx.nominal_kv,
            "limit_ms": limit,
            "over_limit": over,
            "backup_trip": backup,
        },
    )
    if over:
        flags.append(Conclusion(
            "flag_fct_limit", 9, "Ditandai", f"FCT {_int(fct)} ms melebihi batas {_int(limit)} ms",
            [f"Grid Code CCA1 2.2: proteksi utama {_int(ctx.nominal_kv)} kV harus memutus dalam {_int(limit)} ms."],
            ["F3.5"], "flag",
        ))
    return conclusion, flags


def _stable_edges(ch: _Channel) -> list[float]:
    return [t for interval in ch.stable_intervals() for t in interval if math.isfinite(t)]


def _first_breaker_change(ctx: _Context) -> Optional[dict[str, Any]]:
    if ctx.t0 is None:
        return None
    changes = []
    for ch in ctx.channels:
        if ch.role != "breaker":
            continue
        edges = [t for t in _stable_edges(ch) if t >= ctx.t0]
        if edges:
            changes.append((min(edges), ch))
    if not changes:
        return None
    at, ch = min(changes, key=lambda item: item[0])
    return {"name": ch.name, "phase": ch.phase, "at_ms": round(at - ctx.t0, 1)}


# Step 6 ----------------------------------------------------------------------------------

def _trip_and_reclose(ctx: _Context, trip: Optional[dict[str, Any]], path: dict[str, Any], phases: set[str]) -> tuple[Optional[Conclusion], list[Conclusion]]:
    flags: list[Conclusion] = []
    if ctx.t0 is None:
        return None, flags
    evidence: list[str] = []
    rules = ["F6.1"]

    poles: set[str] = set(trip["poles"]) if trip else set()
    three_pole = bool(trip and trip["three_pole"])
    if not trip:
        # No trip channel: read the poles from which currents stopped.
        pattern = ctx.summary.get("zero_current_pattern")
        ceased = set((ctx.summary.get("ceased") or {}).keys())
        if pattern == "single_phase" and len(ceased) == 1:
            poles = ceased
            evidence.append(f"Hanya arus {_PLN[next(iter(ceased))]} yang padam (dari gelombang; kanal trip tidak aktif).")
        elif pattern == "three_together":
            three_pole = True
            evidence.append("Ketiga arus padam bersamaan (dari gelombang; kanal trip tidak aktif).")
        elif pattern == "staggered":
            three_pole = True
            evidence.append("Ketiga arus padam, tidak bersamaan (dari gelombang): ujung lain bisa membuka lebih dulu.")
    if three_pole:
        mode = "Trip 3-pole"
    elif len(poles) == 1:
        mode = f"Trip 1-pole {_PLN[next(iter(poles))]}"
    elif poles:
        mode = f"Trip pole {_pole_text(poles)}"
    else:
        mode = "Mode trip tidak terekam"

    ar_channels = [ch for ch in ctx.channels if ch.role == "reclose" and ch.asserted]
    ar_mode = None
    names = " ".join(ch.name.upper() for ch in ar_channels)
    if any(t in names for t in ("1P", "1-P", "SPAR", "SINGLE")):
        ar_mode = "SPAR"
    elif any(t in names for t in ("3P", "3-P", "TPAR", "THREE")):
        ar_mode = "TPAR"
    for ch in ar_channels:
        on = ch.first_on(ctx.t0 - 20.0)
        if on is None:
            continue
        off = next((t for t in ch.off_ms if t > on), None)
        evidence.append(f"{ch.name} {_ms(on - ctx.t0)}" + (f" → {_ms(off - ctx.t0)} ms." if off is not None else " ms."))

    reclose = None
    if ctx.window is not None:
        verified = [e for e in (ctx.window.reclose_events or []) if e.get("cb_open_verified", True) is not False]
        reclose = verified[-1] if verified else None
    dead_time_s = None
    if reclose and isinstance(reclose.get("time"), (int, float)):
        close_ms = float(reclose["time"]) * 1000.0
        opened = _breaker_open_ms(ctx)
        start = opened if opened is not None else ctx.summary.get("fault_clearing_ms")
        if isinstance(start, (int, float)) and close_ms > start:
            dead_time_s = (close_ms - float(start)) / 1000.0
        rules.append("F6.4")
    breaker_lines = _breaker_lines(ctx)
    evidence += breaker_lines
    if reclose and not ar_mode and len(poles) == 1 and not three_pole:
        # One pole tripped and the breaker closed again with the others in
        # service: a single-pole auto-reclose by definition.
        ar_mode = "SPAR"

    title = mode
    if ar_mode:
        title += f", {ar_mode}"
    if reclose:
        outcome = reclose.get("success")
        rules.append("F6.5")
        if outcome is True:
            title += ", reclose berhasil" + (f" setelah {_num(dead_time_s)} s" if dead_time_s else "")
        elif outcome is False:
            title += ", reclose gagal"
        else:
            title += ", reclose (hasil tidak dapat ditentukan)"
    else:
        title += "; reclose tidak terekam di rekaman ini"

    # F6.2 / F6.3: does the trip and reclose fit the fault?
    rules.append("F6.2")
    if phases:
        if len(phases) == 1 and len(poles) == 1 and not three_pole:
            evidence.append("Konsisten: gangguan satu fasa → trip" + (" dan reclose 1-pole." if ar_mode == "SPAR" else " 1-pole."))
        elif len(phases) >= 2 and three_pole:
            evidence.append("Konsisten: gangguan multi-fasa → trip 3-pole.")
        elif len(phases) >= 2 and poles and not three_pole:
            rules.append("F6.3")
            flags.append(Conclusion(
                "flag_trip_mode", 9, "Ditandai", f"Trip {_pole_text(poles)} untuk gangguan {_phases_text(phases)}",
                ["Gangguan multi-fasa seharusnya trip 3-pole: konflik antara gelombang dan relay."],
                ["F6.3"], "flag",
            ))
        elif len(phases) == 1 and three_pole:
            rules.append("F6.3")
            flags.append(Conclusion(
                "flag_trip_mode", 9, "Ditandai", "Trip 3-pole untuk gangguan satu fasa",
                ["Gangguan satu fasa biasanya trip 1-pole lalu SPAR. Cek setting AR bay ini: AR tidak siap atau diblok, "
                 "gangguan berkembang, atau bay memakai TPAR."],
                ["F6.2", "F6.3"], "flag",
            ))
    if reclose and path["kind"] in ("z2_delayed", "z3_delayed"):
        rules.append("F6.3")
        flags.append(Conclusion(
            "flag_reclose_after_delayed", 9, "Ditandai", "Reclose setelah trip Z2/Z3 waktu tunda",
            ["AR biasanya hanya di-inisiasi trip seketika atau aided (Z1, Z1+aided, DEF+aided)."],
            ["F6.2", "F6.3"], "flag",
        ))
    confidence = "high" if (trip or poles or three_pole) and not flags else "medium"
    value = {
        "trip_mode": "3P" if three_pole else ("1P" if len(poles) == 1 else ("poles" if poles else None)),
        "poles": [p for p in _ORDER if p in poles],
        "ar_mode": ar_mode,
        "reclose_success": reclose.get("success") if reclose else None,
        "dead_time_s": round(dead_time_s, 3) if dead_time_s else None,
        "sequence": ctx.window.sequence if ctx.window else {},
    }
    sequence = ctx.window.sequence if ctx.window else {}
    if sequence.get("refault_after_reclose"):
        evidence.append("PMT menutup kembali, tetapi gangguan muncul lagi: penutupan berhasil secara mekanis, pemulihan tidak bertahan.")
        title += "; gangguan berulang setelah reclose"
    if sequence.get("sotf_after_reclose"):
        rules.append("F5.5")
        for event in sequence["sotf_trips"]:
            evidence.append(f"{event['channel']} aktif {_ms(event['time_ms'] - (ctx.t0 or 0))} ms setelah inception gangguan, sesudah PMT reclose.")
        title += "; SOTF/TOR trip"
    return Conclusion("trip_reclose", 6, "Trip dan reclose", title, evidence, sorted(set(rules), key=_rule_order),
                      confidence, value), flags


def _breaker_open_ms(ctx: _Context) -> Optional[float]:
    after = (ctx.t0 or 0.0)
    edges = [t for ch in ctx.channels if ch.role == "breaker" for t in _stable_edges(ch) if t >= after]
    return min(edges) if edges else None


def _breaker_lines(ctx: _Context) -> list[str]:
    """The breaker contacts' changes after the fault, bounce removed: when a
    contact that was set before the fault resets, also when it sets again
    (the breaker closed again)."""
    lines = []
    for ch in ctx.channels:
        if ch.role != "breaker" or ctx.t0 is None:
            continue
        intervals = ch.stable_intervals()
        for k, (on, off) in enumerate(intervals):
            if math.isfinite(on) and on >= ctx.t0 - 20.0:
                text = f"{ch.name} {_ms(on - ctx.t0)}"
                lines.append(text + (f" → {_ms(off - ctx.t0)} ms." if math.isfinite(off) else " ms."))
                break
            if not math.isfinite(on) and math.isfinite(off) and off >= ctx.t0 - 20.0:
                back = intervals[k + 1][0] if k + 1 < len(intervals) else None
                lines.append(f"{ch.name} reset {_ms(off - ctx.t0)}"
                             + (f" → aktif lagi {_ms(back - ctx.t0)} ms." if back is not None else " ms."))
                break
    return lines[:2]


# Signals ---------------------------------------------------------------------------------

_ROLE_LABEL = {
    "trip": "Trip", "zone": "Zona", "teleprotection": "Teleproteksi", "breaker": "PMT",
    "reclose": "Reclose", "protection": "Proteksi",
}


def _signals(ctx: _Context) -> dict[str, Any]:
    """Every status change and the waveform's own events, on one time base,
    plus the channels that were recorded and never changed (P4)."""
    origin = ctx.t0 if ctx.t0 is not None else 0.0
    events: list[dict[str, Any]] = []
    for ch in ctx.channels:
        role = _ROLE_LABEL.get(ch.role or "", "Lainnya")
        kept: set[float] = set()
        for on, off in ch.stable_intervals():
            if math.isfinite(on):
                change = "Aktif"
                if ch.role == "trip":
                    change += ": trip " + ("3-pole" if ch.phase == "3P" else f"pole {_PLN[ch.phase]}" if ch.phase in _ORDER else "")
                elif _is_receive(ch):
                    change += ": sinyal diterima"
                elif _is_send(ch):
                    change += ": sinyal dikirim"
                elif ch.phase in _ORDER:
                    change += f" (pole {_PLN[ch.phase]})"
                events.append({"t_ms": round(on - origin, 1), "channel": ch.name, "role": role, "change": change.rstrip(": ")})
                kept.add(on)
            if math.isfinite(off):
                events.append({"t_ms": round(off - origin, 1), "channel": ch.name, "role": role, "change": "Reset"})
                kept.add(off)
        # Contact bounce (F6.4): one muted line per burst of short pulses.
        dropped = sorted(t for t in ch.on_ms + ch.off_ms if t not in kept)
        bursts: list[list[float]] = []
        for t in dropped:
            if bursts and t - bursts[-1][-1] < 5.0:
                bursts[-1].append(t)
            else:
                bursts.append([t])
        for burst in bursts:
            span = max(burst[-1] - burst[0], 0.1)
            events.append({
                "t_ms": round(burst[0] - origin, 1), "channel": ch.name, "role": role,
                "change": f"Bounce {_num(span)} ms, diabaikan", "muted": True,
            })
    for event in ctx.trace.get("events") or []:
        kind = event.get("kind")
        text = {
            "disturbance_start": "Gangguan mulai", "current_ceased": "Arus padam (PMT memutus)",
            "current_return": "Arus kembali", "voltage_return": "Tegangan kembali",
            "voltage_lost": "Tegangan hilang", "energization_start": "Line diberi tegangan",
            "current_back_to_load": "Arus kembali ke beban", "current_rise": "Arus naik (bukan gangguan)",
        }.get(kind)
        if text is None or not isinstance(event.get("t_ms"), (int, float)):
            continue
        phase = event.get("phase")
        events.append({
            "t_ms": round(float(event["t_ms"]) - origin, 1),
            "channel": f"Arus fasa {_PLN.get(phase, phase)}" if kind != "voltage_return" and kind != "voltage_lost"
            else f"Tegangan fasa {_PLN.get(phase, phase)}",
            "role": "Gelombang",
            "change": text,
        })
    events.sort(key=lambda e: (e["t_ms"], e["role"] != "Gelombang"))
    silent = [{"channel": ch.name, "role": _ROLE_LABEL.get(ch.role or "", None)} for ch in ctx.channels if not ch.asserted]
    return {
        "reference": "fault_start" if ctx.t0 is not None else "record_start",
        "events": events,
        "silent": silent,
        "channel_count": len(ctx.channels),
    }


# Entry point -----------------------------------------------------------------------------

def build_reasoning(
    line_payload: dict,
    event_window: Optional[EventWindow],
    analog_trace: dict[str, Any],
    measurements: dict[str, Any],
    *,
    event_class: Optional[str],
    gate_reasons: Optional[list[str]] = None,
    nominal_kv: Optional[float] = None,
) -> dict[str, Any]:
    """The record's conclusions, in the order the incident page reads them:
    fault, phases, clearing, trip path, scheme, location, trip and reclose,
    then anything flagged for review."""
    has_fault = (
        event_window is not None
        and event_window.inception_time_ms is not None
        and event_class not in ("NO_FAULT_TRIGGER", "RECLOSE_CAPTURE")
    )
    ctx = _Context(
        payload=line_payload,
        window=event_window,
        trace=analog_trace or {},
        measurements=measurements or {},
        channels=_channels(line_payload),
        event_class=event_class,
        gate_reasons=list(gate_reasons or []),
        t0=float(event_window.inception_time_ms) if has_fault else None,
        nominal_kv=None,
    )
    ctx.nominal_kv = _nominal_kv(ctx.measurements, nominal_kv)

    trip = _trip(ctx) if has_fault else None
    phasors = _fault_phasors(line_payload, event_window) if has_fault else None
    conclusions: list[Conclusion] = [_fault_presence(ctx, trip, phasors)]
    flags: list[Conclusion] = []
    if has_fault:
        phase_row, phases, _ground = _phases(ctx, trip, phasors)
        path = _trip_path(ctx, trip)
        clearing, clearing_flags = _clearing(ctx, trip, path, phases)
        trip_path = _trip_path_conclusion(ctx, trip, path)
        scheme = _scheme_conclusion(path)
        location = _location_conclusion(path, scheme)
        trip_reclose, trip_flags = _trip_and_reclose(ctx, trip, path, phases)
        conclusions += [c for c in (phase_row, clearing, trip_path, scheme, location, trip_reclose) if c is not None]
        flags += clearing_flags + trip_flags
        nested = _nested_zone_flag(path, phase_row.value)
        if nested:
            flags.append(nested)
        for text in phase_row.value.get("flags") or []:
            flags.append(Conclusion("flag_phase_currents", 9, "Ditandai", text, [], ["F4.7"], "flag"))
    conclusions += flags
    return {
        "schema": SCHEMA,
        "has_fault": has_fault,
        "fault_start_ms": ctx.t0,
        "conclusions": [c.to_dict() for c in conclusions],
        "flag_count": sum(1 for c in conclusions if c.confidence == "flag"),
        "conflict_count": sum(len(c.conflicts) for c in conclusions),
        "signals": _signals(ctx),
    }
