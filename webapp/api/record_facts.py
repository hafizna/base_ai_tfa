"""Measured facts behind a record's place in an incident story.

Two views of one stored COMTRADE payload, both facts rather than readings:

- ``electrical_measurements``: per-phase current/voltage before, during and
  after the fault (or after the reclose, for a dead-time recording).
- ``protection_operations``: when each trip / zone / teleprotection /
  breaker / auto-reclose status channel asserted and dropped.

Which zone *operated*, whether a trip was teleprotection-aided, or what caused
the fault are interpretations and stay out of here; a consumer composes them
from these facts and says so.

Times are in milliseconds on the record's own time axis, the same axis as
``EventWindow.inception_time_ms``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from core.event_analysis import EventWindow
from core.line_selection import cycle_rms_envelope

_PHASES = ("A", "B", "C")
_MAX_EDGES = 4  # per channel; contact bounce beyond this adds nothing to the story


def _samples_per_cycle(time: np.ndarray, frequency: Optional[float]) -> Optional[int]:
    if len(time) < 8:
        return None
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return max(4, int(round(1.0 / (dt * (float(frequency or 50.0) or 50.0)))))


def _phase_channels(payload: dict, measurement: str) -> tuple[dict[str, np.ndarray], Optional[str]]:
    channels: dict[str, np.ndarray] = {}
    unit: Optional[str] = None
    for ch in payload.get("analog_channels", []):
        phase = str(ch.get("phase") or "").upper()
        if ch.get("measurement") != measurement or phase not in _PHASES or phase in channels:
            continue
        channels[phase] = np.asarray(ch.get("samples") or [], dtype=float)
        unit = unit or (ch.get("unit") or None)
    return channels, unit


def _round(value: float) -> Optional[float]:
    return round(float(value), 3) if np.isfinite(value) else None


def _rms_over(envelopes: dict[str, np.ndarray], start: int, stop: int) -> dict[str, Optional[float]]:
    """Median one-cycle RMS of each phase whose envelope covers [start, stop)."""
    out: dict[str, Optional[float]] = {}
    for phase, env in envelopes.items():
        lo, hi = max(0, start), min(len(env), stop)
        out[phase] = _round(float(np.median(env[lo:hi]))) if hi > lo else None
    return out


def electrical_measurements(payload: dict, event_window: Optional[EventWindow]) -> dict[str, Any]:
    """Per-phase magnitudes around the record's event, for the incident story.

    - ``prefault``: one-cycle RMS current/voltage just before inception.
    - ``fault``: instantaneous peak and highest one-cycle RMS current, and the
      lowest one-cycle RMS voltage, between inception and clearing.
    - ``after_clearing`` / ``after_reclose``: the settled state afterwards.
    """
    if event_window is None:
        return {}
    time = np.asarray(payload.get("time") or [], dtype=float)
    n = _samples_per_cycle(time, payload.get("frequency"))
    if n is None:
        return {}
    currents, current_unit = _phase_channels(payload, "current")
    voltages, voltage_unit = _phase_channels(payload, "voltage")
    if not currents and not voltages:
        return {}
    i_env = {p: cycle_rms_envelope(x, n) for p, x in currents.items()}
    v_env = {p: cycle_rms_envelope(x, n) for p, x in voltages.items()}
    out: dict[str, Any] = {"current_unit": current_unit, "voltage_unit": voltage_unit}

    i0 = event_window.inception_idx
    if i0 is not None and event_window.method != "dead_time_recording":
        out["prefault"] = {
            "current_rms": _rms_over(i_env, i0 - 2 * n, i0 - n + 1),
            "voltage_rms": _rms_over(v_env, i0 - 2 * n, i0 - n + 1),
        }
        end = event_window.clearing_idx if event_window.clearing_idx is not None else i0 + 10 * n
        end = min(max(end, i0 + n), len(time))
        out["fault"] = {
            "current_peak": {p: _round(float(np.max(np.abs(x[i0:end])))) for p, x in currents.items() if end > i0},
            "current_rms_max": {
                p: _round(float(np.max(env[i0:max(i0 + 1, end - n + 1)]))) for p, env in i_env.items() if len(env) > i0
            },
            "voltage_rms_min": {
                p: _round(float(np.min(env[i0:max(i0 + 1, end - n + 1)]))) for p, env in v_env.items() if len(env) > i0
            },
        }
        if event_window.clearing_idx is not None:
            settle = event_window.clearing_idx + 2 * n
            out["after_clearing"] = {
                "current_rms": _rms_over(i_env, settle, settle + n),
                "voltage_rms": _rms_over(v_env, settle, settle + n),
            }

    if event_window.method == "dead_time_recording" and event_window.reclose_events:
        reclose_s = event_window.reclose_events[-1].get("time")
        if reclose_s is not None:
            settle = int(np.searchsorted(time, float(reclose_s))) + 3 * n
            out["after_reclose"] = {
                "current_rms": _rms_over(i_env, settle, settle + n),
                "voltage_rms": _rms_over(v_env, settle, settle + n),
            }
    return out


# --- status channels -------------------------------------------------------

_IGNORED = frozenset({
    "SPARE", "UNUSED", "ALARM", "ALRM", "FAIL", "FAILURE", "HEALTHY", "HEALTH", "SUPERV", "SUPERVISION",
    "TEST", "BLOCK", "BLK", "BAR", "VTS", "MCB", "SWING", "SF6", "GAS", "PRESSURE", "SPRING", "SYNCH", "SYNC",
})
_BREAKER = frozenset({"CB", "52A", "52B", "BREAKER", "PMT", "POLE"})
_RECLOSE = frozenset({"AR", "RECLOSE", "RECLOSING", "AUTORECLOSE", "AUTOCLOSE"})
_TELEPROTECTION = frozenset({
    "CR", "CS", "SEND", "SENDING", "SND", "RCV", "RECV", "RECEIVE", "RECEIVED", "RX", "TX", "CARR", "CARRIER",
    "CARIER", "CHAN", "CHANNEL", "DTT", "PUTT", "POTT", "UNB", "TELEPROT", "TELEPROTECTION",
})
_TRIP = frozenset({"TRIP", "TRP", "OPERATE", "OPERATED", "OPRT", "OPRTD", "OPR"})
_PROTECTION = frozenset({"DIST", "OCR", "OC", "GFR", "DEF", "EF", "SOTF", "TOR", "PROT", "DIFF"})
_ZONE_RE = re.compile(r"^(?:Z|ZONE)([1-5])$")
_ANSI_PROTECTION_RE = re.compile(r"^F?(?:21|50|51|67|87)[A-Z0-9]*$")
_ANSI_RECLOSE_RE = re.compile(r"^F?79[A-Z0-9]*$")
_ANSI_TELEPROTECTION_RE = re.compile(r"^F?85[A-Z0-9]*$")
_PHASE_TOKENS = {"A": "A", "B": "B", "C": "C", "R": "A", "S": "B", "T": "C", "L1": "A", "L2": "B", "L3": "C"}


def _tokens(name: str) -> list[str]:
    upper = re.sub(r"\bA\s*/\s*R\b", "AR", (name or "").upper())
    upper = re.sub(r"\bZONE\s+([1-5])\b", r"ZONE\1", upper)
    return [t for t in re.split(r"[^A-Z0-9]+", upper) if t]


def classify_status_channel(name: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """``(role, zone, phase)`` of a status channel from its name.

    role is one of ``trip``, ``zone``, ``teleprotection``, ``breaker``,
    ``reclose``, ``protection`` (other protection element), or None for
    alarms, spares and anything unrecognised. ``zone`` is set for zone
    channels (and for a trip channel naming its zone); ``phase`` is A/B/C or
    ``3P`` when the name says which pole it concerns.
    """
    tokens = _tokens(name)
    token_set = set(tokens)
    if not tokens or token_set & _IGNORED:
        return None, None, None
    zone = next((int(m.group(1)) for t in tokens if (m := _ZONE_RE.match(t))), None)
    # A pole letter can sit anywhere after the element name, e.g. "TRIP R MJSNG2".
    named_phases = {_PHASE_TOKENS[t] for t in tokens[1:] if t in _PHASE_TOKENS}
    phase = None
    if token_set & {"3P", "3PH", "3PHASE", "ABC", "RST"} or len(named_phases) == 3:
        phase = "3P"
    elif len(named_phases) == 1:
        phase = named_phases.pop()

    if token_set & _BREAKER:
        role = "breaker"
    elif (token_set & _RECLOSE or {"AUTO", "CLOSE"} <= token_set
          or any(_ANSI_RECLOSE_RE.match(t) for t in tokens)):
        # "Auto Close" is the auto-reclose close command.
        role = "reclose"
    elif token_set & _TELEPROTECTION or any(_ANSI_TELEPROTECTION_RE.match(t) for t in tokens):
        role = "teleprotection"
    elif token_set & _TRIP:
        role = "trip"
    elif zone is not None:
        role = "zone"
    elif token_set & _PROTECTION or any(_ANSI_PROTECTION_RE.match(t) for t in tokens):
        role = "protection"
    else:
        return None, None, None
    return role, zone, phase


def protection_operations(payload: dict, event_window: Optional[EventWindow] = None) -> list[dict[str, Any]]:
    """Trip / zone / teleprotection / breaker / reclose channels that asserted.

    One entry per recognised channel that changed state or was already
    asserted when the record started, with its first edges in ms on the
    record's time axis. Ordered by first assertion.
    """
    time = np.asarray(payload.get("time") or [], dtype=float)
    if len(time) < 2:
        return []
    operations: list[dict[str, Any]] = []
    for ch in payload.get("status_channels", []):
        name = str(ch.get("name") or "").strip()
        role, zone, phase = classify_status_channel(name)
        if role is None:
            continue
        samples = np.asarray(ch.get("samples") or [], dtype=float)
        if len(samples) != len(time):
            continue
        state = samples != 0
        diff = np.diff(state.astype(int))
        rises = (np.where(diff > 0)[0] + 1)[:_MAX_EDGES]
        falls = (np.where(diff < 0)[0] + 1)[:_MAX_EDGES]
        if not state[0] and rises.size == 0:
            continue
        operations.append({
            "name": name,
            "role": role,
            "zone": zone,
            "phase": phase,
            "initially_on": bool(state[0]),
            "on_ms": [_round(time[i] * 1000.0) for i in rises],
            "off_ms": [_round(time[i] * 1000.0) for i in falls],
        })
    operations.sort(key=lambda op: (0.0 if op["initially_on"] else (op["on_ms"][0] if op["on_ms"] else np.inf)))
    return operations
