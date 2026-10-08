"""Per-line channel grouping and disturbed-line selection for multi-line records.

An external DFR is frequently wired to TWO lines/bays and writes both into one
COMTRADE file, e.g. GI Bringin ``"CT R MJSNG1" .. "CT T MJSNG2"`` or GI
Mojosongo ``"IR BRINGIN 1" .. "IR BRINGIN 2"``. Every single-line analysis in
this codebase (event window, no-fault gate, AI features, locus, double-ended
fault location, report) looks channels up by canonical name (IA/VA/...) and
takes the first match, so on such a record it silently analyses whichever line
happens to be listed first — for the 21/08/2023 Bringin record that was an
out-of-service line carrying ~1 A of noise, and the AI cause reading was
computed from that noise.

This module is the single place that decides which line a record is about:

1. ``line_key`` groups channels per line by name: the line identity is what is
   left of a channel name after removing measurement/phase tokens. A record is
   only declared multi-line when there are >= 2 groups that EACH carry their
   own phase currents AND their own phase voltages. The voltage requirement is
   what separates genuine multi-bay DFR records from relay records carrying
   several CT inputs of the SAME protected object (1.5-breaker ``CB1.ia`` /
   ``CB2.ia``, Siemens ``MPI3p1`` / ``MPI3p2``, 87L local/remote currents,
   busbar/transformer differential CT inputs) — those share one voltage set
   (or have none) and must not be split into "lines".
2. ``status_line_key`` assigns status channels to a line by name (exact,
   space-insensitive, or an abbreviation such as ``"LP OPRT SRAGI1"`` for line
   ``"SUNYARAGI 1"``). Unassignable status channels stay common to all lines.
3. ``select_line`` picks the disturbed line from evidence, strongest first:
   the line's OWN protection operating (trip/operate status), then its breaker,
   then its auto-reclose, then a waveform trip signature (load current
   interrupted to ~0), then the largest superimposed (fault-minus-prefault)
   current. The other lines are classified (de-energized, quiet, impacted by
   the fault on another line, breaker activity only, also operated) and the
   situations one record cannot settle — the protection of two lines
   operating (double-circuit fault or sympathetic trip), or comparable fault
   current with no line-specific evidence (which line? or a through-fault?) —
   are flagged for review instead of being decided silently.
4. ``scope_payload`` / ``scope_record`` project a record onto the selected
   line (the other lines' analog AND status channels are dropped, untagged
   common channels are kept), so all existing single-line logic runs
   unchanged on the right line. Records that are not multi-line are returned
   untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# --- line identity from channel names -------------------------------------

_MEASUREMENT_TOKENS = frozenset({
    "V", "U", "I", "VT", "CT", "PT", "CVT", "CCVT", "VOLT", "VOLTS", "VOLTAGE",
    "CURRENT", "CURR", "AMP", "AMPS", "ARUS", "TEGANGAN", "KV", "KA", "MV", "MA",
})
_PHASE_TOKENS = frozenset({
    "A", "B", "C", "R", "S", "T", "N", "G", "E", "L1", "L2", "L3", "PH", "PHS",
    "PHASE", "PHASA", "FASA", "NEUTRAL", "NETRAL", "RES", "RESIDUAL", "GND",
    "GROUND", "EARTH", "AN", "BN", "CN", "RN", "SN", "TN",
})
_GENERIC_TOKENS = frozenset({
    "CH", "CHANNEL", "ANALOG", "ANA", "INST", "RMS", "PRI", "PRIMARY", "SEC",
    "SECONDARY", "FREQ", "FREQUENCY", "HZ", "DC", "POS", "NEG",
})
# Measurement+phase glued into one token: IR, VS, IA, UL1, VAN, VAB, I0, 3I0 ...
_MEASUREMENT_PHASE_RE = re.compile(
    r"^(?:[IVU](?:[ABCRSTNEG]|L[123]|[123]|[ABCRST]N|AB|BC|CA|RS|ST|TR|L12|L23|L31|12|23|31|0|RES)|3[IV]0)$"
)
# Measurement+phase+line digit glued into one token: IA1, VR2 -> line "1"/"2".
_MEASUREMENT_PHASE_LINE_RE = re.compile(r"^[IVU][ABCRSTN](\d{1,2})$")
_CHANNEL_NUMBER_RE = re.compile(r"^CH\d+$")
_VOLTAGE_LEVEL_RE = re.compile(r"^\d+K?V$")


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", (name or "").upper()) if t]


def line_key(channel_name: str) -> Optional[str]:
    """Line/bay identity carried by a channel name, or None if it has none.

    ``"CT R MJSNG2"`` -> ``"MJSNG2"``, ``"IR BRINGIN 2"`` -> ``"BRINGIN 2"``,
    ``"CIBADAK IR"`` -> ``"CIBADAK"``, ``"VR IBT 1 150KV"`` -> ``"IBT 1"``,
    ``"IA"`` -> None. Keys are compared space-insensitively (see
    ``_compact``) so ``"VT DELTAMAS2"`` still belongs to ``"DELTAMAS 2"``.
    """
    kept: list[str] = []
    for token in _tokens(channel_name):
        glued = _MEASUREMENT_PHASE_LINE_RE.match(token)
        if glued:
            kept.append(glued.group(1))
            continue
        if (
            token in _MEASUREMENT_TOKENS
            or token in _PHASE_TOKENS
            or token in _GENERIC_TOKENS
            or _MEASUREMENT_PHASE_RE.match(token)
            or _CHANNEL_NUMBER_RE.match(token)
            or _VOLTAGE_LEVEL_RE.match(token)
        ):
            continue
        kept.append(token)
    return " ".join(kept) or None


def _compact(key: Optional[str]) -> str:
    return (key or "").replace(" ", "")


# A name part that can be abbreviated: letters optionally followed by a line
# number ("SUNYARAGI1", "SALAKBARU", "MJSNG2").
_LETTERS_THEN_DIGITS_RE = re.compile(r"^([A-Z]+)(\d*)$")


def _is_abbreviation(candidate: str, full: str) -> bool:
    """``candidate`` is an in-order letter subsequence of ``full`` starting with
    the same letter (``SRAGI`` / ``SUNYARAGI``, ``DPK`` / ``DEPOK``)."""
    if len(candidate) < 3 or len(candidate) >= len(full) or candidate[0] != full[0]:
        return False
    it = iter(full)
    return all(ch in it for ch in candidate)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _phase_slot(channel: Any) -> Optional[str]:
    canonical = (_attr(channel, "canonical_name") or "").upper()
    measurement = _attr(channel, "measurement")
    if measurement == "current" and canonical in ("IA", "IB", "IC"):
        return canonical
    if measurement == "voltage" and canonical in ("VA", "VB", "VC"):
        return canonical
    return None


@dataclass
class _LineGroup:
    key: str
    compact: str
    currents: dict[str, Any] = field(default_factory=dict)
    voltages: dict[str, Any] = field(default_factory=dict)


# Group keys naming a terminal or winding of ONE protected object, not a
# separate line/bay: ABB RED670 "LINE CT IL1" / "REM CT IL1" (87L local and
# remote end), NR PCS-978 "HVS.Ia" / "LVS.Ia" (transformer sides). Splitting
# such a record would analyse the remote terminal or one winding as if it were
# "the line" — the record is left to single-line logic instead.
_TERMINAL_OR_WINDING_TOKENS = frozenset({
    "REM", "REMOTE", "RMT", "LOC", "LOCAL", "LCL",
    "HV", "LV", "HVS", "MVS", "LVS", "W1", "W2", "W3", "PRIMER", "SEKUNDER", "TERSIER",
})
# Groups of relay-computed channels (Siemens "iL1(Delta Prev.)*", restraint/
# differential/sum quantities) are views of the same currents, never a line.
_DERIVED_QUANTITY_TOKENS = frozenset({
    "DELTA", "PREV", "FIRST", "DIFF", "BIAS", "REST", "RESTRAINT", "SUM", "CALC", "AVG", "MAG", "ANG",
})


def _group_lines(analog_channels: list) -> dict[str, _LineGroup]:
    """Line groups keyed by compact key; empty unless the record is multi-line."""
    groups: dict[str, _LineGroup] = {}
    for channel in analog_channels:
        slot = _phase_slot(channel)
        if slot is None:
            continue
        key = line_key(_attr(channel, "name") or "")
        if not key:
            continue
        group = groups.setdefault(_compact(key), _LineGroup(key=key, compact=_compact(key)))
        if slot.startswith("I"):
            if not group.currents:
                group.key = key  # display the current channels' spelling ("DELTAMAS 2", not "DELTAMAS2")
            group.currents.setdefault(slot, channel)
        else:
            group.voltages.setdefault(slot, channel)
    if any(set(g.key.split()) & _TERMINAL_OR_WINDING_TOKENS for g in groups.values()):
        return {}
    lines = {
        c: g for c, g in groups.items()
        if len(g.currents) >= 2 and len(g.voltages) >= 2 and not set(g.key.split()) & _DERIVED_QUANTITY_TOKENS
    }
    return lines if len(lines) >= 2 else {}


def status_line_key(status_name: str, lines: list[str]) -> Optional[str]:
    """The one line (display key from ``lines``) a status channel belongs to,
    or None when it names no line or more than one."""
    tokens = _tokens(status_name)
    candidates = set(tokens)
    for width in (2, 3):
        candidates.update("".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1))

    exact = [line for line in lines if _compact(line) in candidates]
    if exact:
        return exact[0] if len(exact) == 1 else None

    # Abbreviations: "SRAGI1" for SUNYARAGI 1, "SKMD" for SUKAMANDI 2 (a status
    # name may drop the circuit number; uniqueness below keeps "BRINGIN" from
    # matching both BRINGIN 1 and BRINGIN 2). Two abbreviations of the same
    # station that are not subsequences of each other ("PGNDRN 1" vs the
    # analog channels' "PGDRAN 1") are matched by letter overlap, but only
    # when both carry the same circuit number.
    for matcher in (_abbreviation_match, _overlap_match):
        matched = [line for line in lines if matcher(candidates, _compact(line))]
        if matched:
            return matched[0] if len(matched) == 1 else None
    return None


def _abbreviation_match(candidates: set[str], line_compact: str) -> bool:
    full = _LETTERS_THEN_DIGITS_RE.match(line_compact)
    if not full:
        return False
    for candidate in candidates:
        short = _LETTERS_THEN_DIGITS_RE.match(candidate)
        if short and short.group(2) in ("", full.group(2)) and _is_abbreviation(short.group(1), full.group(1)):
            return True
    return False


def _overlap_match(candidates: set[str], line_compact: str) -> bool:
    full = _LETTERS_THEN_DIGITS_RE.match(line_compact)
    if not full or not full.group(2) or len(full.group(1)) < 4:
        return False
    for candidate in candidates:
        short = _LETTERS_THEN_DIGITS_RE.match(candidate)
        if (
            short and short.group(2) == full.group(2) and len(short.group(1)) >= 4
            and short.group(1)[0] == full.group(1)[0]
            and _lcs_length(short.group(1), full.group(1)) >= 0.75 * min(len(short.group(1)), len(full.group(1)))
        ):
            return True
    return False


def _lcs_length(a: str, b: str) -> int:
    previous = [0] * (len(b) + 1)
    for ch in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if ch == other else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


# --- status channel semantics ---------------------------------------------

_IGNORED_STATUS_TOKENS = frozenset({
    "SPARE", "ALARM", "ALRM", "FAIL", "FAILURE", "HEALTHY", "HEALTY", "HEALTH", "NOT", "PRES",
    "PRESSURE", "SF6", "GAS", "LOCKOUT", "VTS", "SUPERV", "SUPERVISION", "TEST", "BLOCK", "BLK",
    "SPRING",
})
_TELEPROTECTION_OR_PICKUP_TOKENS = frozenset({
    "SEND", "SENDING", "SND", "RCV", "RECV", "RECEIVE", "RECEIVED", "RCVE", "REC", "RX", "TX",
    "CARR", "CARRIER", "CARIER", "CAR", "SIGNAL", "DTT", "START", "STARTUP", "PICKUP", "PKP", "PU",
})
_BREAKER_TOKENS = frozenset({"CB", "CBAB", "52A", "52B", "BREAKER", "PMT", "POLE"})
_RECLOSE_TOKENS = frozenset({"AR", "RECLOSE", "RECLOSING", "AUTO"})
_PROTECTION_TOKENS = frozenset({
    "TRIP", "TRP", "OPRT", "OPRTD", "OPR", "OPERATE", "OPERATED", "OP", "Z1", "Z2", "Z3",
    "ZONE", "DIST", "OCR", "OC", "GFR", "DEF", "EF", "MPU", "LP", "PROT", "DIFF",
})
# ANSI device numbers, optionally prefixed "F": F21, 21N, 50/51, 67N, F87L / 79 / 85.
_ANSI_PROTECTION_RE = re.compile(r"^F?(?:21|50|51|67|87)[A-Z0-9]*$")
_ANSI_RECLOSE_RE = re.compile(r"^F?79[A-Z0-9]*$")
_ANSI_TELEPROTECTION_RE = re.compile(r"^F?85[A-Z0-9]*$")

EVIDENCE_PROTECTION = "protection"
EVIDENCE_BREAKER = "breaker"
EVIDENCE_RECLOSE = "reclose"
_EVIDENCE_WEAK = "teleprotection_or_pickup"


def _status_category(name: str) -> Optional[str]:
    """Evidence class of a status channel: protection operation, breaker,
    auto-reclose, teleprotection/pickup (weak), or None (alarm/spare/other)."""
    tokens = set(_tokens(re.sub(r"\bA\s*/\s*R\b", "AR", (name or "").upper())))
    if tokens & _IGNORED_STATUS_TOKENS:
        return None
    if tokens & _BREAKER_TOKENS:
        return EVIDENCE_BREAKER
    if tokens & _TELEPROTECTION_OR_PICKUP_TOKENS or any(_ANSI_TELEPROTECTION_RE.match(t) for t in tokens):
        return _EVIDENCE_WEAK
    if tokens & _RECLOSE_TOKENS or any(_ANSI_RECLOSE_RE.match(t) for t in tokens):
        return EVIDENCE_RECLOSE
    if tokens & _PROTECTION_TOKENS or any(_ANSI_PROTECTION_RE.match(t) for t in tokens):
        return EVIDENCE_PROTECTION
    return None


def _status_samples(channel: Any) -> np.ndarray:
    samples = _attr(channel, "samples")
    return np.asarray(samples if samples is not None else [], dtype=int)


# --- selection ------------------------------------------------------------

LINE_SELECTED = "SELECTED"
LINE_DE_ENERGIZED = "DE_ENERGIZED"
LINE_QUIET = "QUIET"
LINE_IMPACTED = "IMPACTED"
LINE_BREAKER_OR_RECLOSE_ONLY = "BREAKER_OR_RECLOSE_ONLY"
LINE_ALSO_OPERATED = "ALSO_OPERATED"

FLAG_OTHER_LINE_IMPACTED = "OTHER_LINE_IMPACTED"
FLAG_MULTIPLE_LINES_OPERATED = "MULTIPLE_LINES_OPERATED"
FLAG_AMBIGUOUS_SELECTION = "AMBIGUOUS_SELECTION"

# A line whose superimposed current is at least this fraction of the selected
# line's counts as impacted by the disturbance (parallel infeed/through-fault).
_IMPACTED_FRACTION = 0.10
# Without any operation evidence, a runner-up this close to the selected
# line's superimposed current makes magnitude alone an unsafe tie-breaker.
_AMBIGUOUS_FRACTION = 0.60
# Status evidence picked a line, but a line whose status channels could not be
# attributed at all carried this many times more fault current: the evidence
# may just be missing for that line.
_UNOBSERVED_DOMINANCE = 3.0
# Primary-amp floors: an out-of-service line reads a few amps of CT/ADC noise;
# a line counts as having been in service (so that its current dropping to
# ~0 means its breaker opened) only above the loaded floor.
_CURRENT_NOISE_FLOOR_A = 5.0
_LOADED_FLOOR_A = 20.0


@dataclass
class LineInfo:
    key: str
    state: str
    prefault_current_a: float
    peak_current_a: float
    final_current_a: float
    superimposed_current_a: float
    interrupted: bool
    status_channel_count: int = 0
    protection: list[str] = field(default_factory=list)
    breaker: list[str] = field(default_factory=list)
    reclose: list[str] = field(default_factory=list)
    teleprotection_or_pickup: list[str] = field(default_factory=list)
    breaker_open_throughout: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.key,
            "state": self.state,
            "prefault_current_a": round(self.prefault_current_a, 1),
            "peak_current_a": round(self.peak_current_a, 1),
            "final_current_a": round(self.final_current_a, 1),
            "superimposed_current_a": round(self.superimposed_current_a, 1),
            "current_interrupted": self.interrupted,
            "status_channel_count": self.status_channel_count,
            "protection_operations": list(self.protection),
            "breaker_operations": list(self.breaker),
            "reclose_operations": list(self.reclose),
            "teleprotection_or_pickup": list(self.teleprotection_or_pickup),
            "breaker_open_throughout": self.breaker_open_throughout,
        }


@dataclass
class LineSelection:
    selected: str
    method: str
    lines: list[LineInfo]
    flags: list[str]
    requires_review: bool
    summary: str

    @property
    def selected_compact(self) -> str:
        return _compact(self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_line": self.selected,
            "method": self.method,
            "flags": list(self.flags),
            "requires_review": self.requires_review,
            "summary": self.summary,
            "lines": [line.to_dict() for line in self.lines],
        }


def _cycle_rms_envelope(samples: Any, n: int) -> np.ndarray:
    """One-cycle sliding RMS with each window's mean removed, so a CT/ADC DC
    offset on an idle channel does not read as current."""
    x = np.asarray(samples if samples is not None else [], dtype=float)
    if n < 2 or len(x) < n:
        return np.zeros(0)
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))


def _samples_per_cycle(time: Any, frequency: Optional[float]) -> int:
    t = np.asarray(time if time is not None else [], dtype=float)
    freq = float(frequency or 50.0) or 50.0
    if len(t) < 2 or t[1] <= t[0]:
        return 0
    return max(2, int(round((1.0 / (t[1] - t[0])) / freq)))


def _line_current_stats(group: _LineGroup, n: int) -> tuple[float, float, float, float]:
    prefault = peak = final = superimposed = 0.0
    for channel in group.currents.values():
        env = _cycle_rms_envelope(_attr(channel, "samples"), n)
        if len(env) == 0:
            continue
        prefault = max(prefault, float(env[0]))
        peak = max(peak, float(np.max(env)))
        final = max(final, float(env[-1]))
        superimposed = max(superimposed, float(np.max(env) - env[0]))
    return prefault, peak, final, superimposed


def _line_voltage_level(group: _LineGroup, n: int) -> float:
    """Highest one-cycle voltage RMS anywhere in the record — compared across
    lines to recognise a de-energized line (line-side VT reads ~0 throughout),
    including in records that start in dead time where even the analysed
    line reads ~0 before its breaker closes."""
    level = 0.0
    for channel in group.voltages.values():
        env = _cycle_rms_envelope(_attr(channel, "samples"), n)
        if len(env):
            level = max(level, float(np.max(env)))
    return level


def _attach_status_evidence(infos: dict[str, LineInfo], status_channels: list) -> None:
    display_keys = [info.key for info in infos.values()]
    for channel in status_channels:
        name = _attr(channel, "name") or ""
        owner = status_line_key(name, display_keys)
        if owner is None:
            continue
        info = infos[_compact(owner)]
        info.status_channel_count += 1
        category = _status_category(name)
        samples = _status_samples(channel)
        if category is None or len(samples) < 2:
            continue
        rises = bool(np.any(np.diff(samples) > 0))
        if category == EVIDENCE_BREAKER:
            if rises or bool(np.any(np.diff(samples) < 0)):
                info.breaker.append(name)
            elif samples[0] and "OPEN" in _tokens(name):
                info.breaker_open_throughout = True
        elif rises:
            getattr(info, category).append(name)


def select_line(
    analog_channels: list,
    status_channels: list,
    time: Any,
    frequency: Optional[float] = 50.0,
) -> Optional[LineSelection]:
    """Choose the disturbed line of a multi-line record; None when the record
    is not multi-line (single-line logic then applies unchanged)."""
    groups = _group_lines(analog_channels or [])
    if not groups:
        return None
    n = _samples_per_cycle(time, frequency)
    if n == 0:
        return None

    infos: dict[str, LineInfo] = {}
    voltage_level: dict[str, float] = {}
    for compact, group in groups.items():
        prefault, peak, final, superimposed = _line_current_stats(group, n)
        infos[compact] = LineInfo(
            key=group.key, state=LINE_QUIET, prefault_current_a=prefault, peak_current_a=peak,
            final_current_a=final, superimposed_current_a=superimposed,
            # In service before the event, ~0 at the end: its breaker opened.
            interrupted=prefault > _LOADED_FLOOR_A and final < max(_CURRENT_NOISE_FLOOR_A, 0.1 * prefault),
        )
        voltage_level[compact] = _line_voltage_level(group, n)
    _attach_status_evidence(infos, status_channels or [])

    lines = list(infos.values())
    by_current = sorted(lines, key=lambda line: line.superimposed_current_a, reverse=True)
    flags: list[str] = []
    selected: Optional[LineInfo] = None
    method = ""

    for tier in (EVIDENCE_PROTECTION, EVIDENCE_BREAKER, EVIDENCE_RECLOSE):
        candidates = [line for line in lines if getattr(line, tier)]
        if candidates:
            selected = max(candidates, key=lambda line: line.superimposed_current_a)
            method = f"status_{tier}"
            if tier == EVIDENCE_PROTECTION and len(candidates) > 1:
                flags.append(FLAG_MULTIPLE_LINES_OPERATED)
            break

    if selected is not None:
        unobserved = [line for line in lines if line is not selected and line.status_channel_count == 0]
        if unobserved and max(line.superimposed_current_a for line in unobserved) > _UNOBSERVED_DOMINANCE * max(
            selected.superimposed_current_a, _CURRENT_NOISE_FLOOR_A
        ):
            flags.append(FLAG_AMBIGUOUS_SELECTION)
    else:
        interrupted = [line for line in lines if line.interrupted]
        if len(interrupted) == 1:
            selected = interrupted[0]
            method = "current_interruption"
        else:
            selected = (max(interrupted, key=lambda line: line.superimposed_current_a)
                        if interrupted else by_current[0])
            method = "superimposed_current"
            if len(interrupted) > 1:
                flags.append(FLAG_MULTIPLE_LINES_OPERATED)
            runner_up = next((line for line in by_current if line is not selected), None)
            if (
                runner_up is not None
                and selected.superimposed_current_a > _CURRENT_NOISE_FLOOR_A
                and runner_up.superimposed_current_a >= _AMBIGUOUS_FRACTION * selected.superimposed_current_a
            ):
                flags.append(FLAG_AMBIGUOUS_SELECTION)

    selected.state = LINE_SELECTED
    selected_voltage = voltage_level[_compact(selected.key)]
    for line in lines:
        if line is selected:
            continue
        compact = _compact(line.key)
        if line.protection or (line.interrupted and FLAG_MULTIPLE_LINES_OPERATED in flags):
            line.state = LINE_ALSO_OPERATED
        elif line.breaker or line.reclose:
            line.state = LINE_BREAKER_OR_RECLOSE_ONLY
        elif (line.breaker_open_throughout and line.peak_current_a < _LOADED_FLOOR_A) or (
            line.peak_current_a < _CURRENT_NOISE_FLOOR_A and voltage_level[compact] < 0.1 * selected_voltage
        ):
            line.state = LINE_DE_ENERGIZED
        elif (
            line.superimposed_current_a > _CURRENT_NOISE_FLOOR_A
            and line.superimposed_current_a >= _IMPACTED_FRACTION * selected.superimposed_current_a
        ):
            line.state = LINE_IMPACTED
        else:
            line.state = LINE_QUIET
    if any(line.state == LINE_IMPACTED for line in lines):
        flags.append(FLAG_OTHER_LINE_IMPACTED)

    return LineSelection(
        selected=selected.key,
        method=method,
        lines=lines,
        flags=flags,
        requires_review=FLAG_MULTIPLE_LINES_OPERATED in flags or FLAG_AMBIGUOUS_SELECTION in flags,
        summary=_summary(selected, lines, method, flags),
    )


_METHOD_TEXT = {
    "status_protection": "its own protection operated (trip/operate status channels)",
    "status_breaker": "its breaker operated (no protection operation was recorded for any line)",
    "status_reclose": "its auto-reclose operated (no protection or breaker operation was recorded)",
    "current_interruption": "its load current was interrupted (breaker opened) while the other line kept carrying current",
    "superimposed_current": "largest superimposed (fault minus prefault) current — no line-specific operation evidence",
}
_STATE_TEXT = {
    LINE_DE_ENERGIZED: "de-energized / out of service",
    LINE_QUIET: "no significant disturbance",
    LINE_IMPACTED: "carried fault-related current without operating (parallel infeed / through-fault)",
    LINE_BREAKER_OR_RECLOSE_ONLY: (
        "breaker/auto-reclose activity without its own protection operating "
        "(shared diameter breaker, backup or transfer trip)"
    ),
    LINE_ALSO_OPERATED: "ALSO operated — possible double-circuit fault or sympathetic trip",
}


def _summary(selected: LineInfo, lines: list[LineInfo], method: str, flags: list[str]) -> str:
    others = "; ".join(f"{line.key}: {_STATE_TEXT[line.state]}" for line in lines if line is not selected)
    text = (
        f"Multi-line record ({len(lines)} lines). Analysing {selected.key}, selected because "
        f"{_METHOD_TEXT[method]}. Other lines — {others}."
    )
    if FLAG_MULTIPLE_LINES_OPERATED in flags:
        text += (" More than one line operated: this may be a double-circuit fault (each line needs its own "
                 "analysis) or a sympathetic trip; only the selected line is analysed here.")
    if FLAG_AMBIGUOUS_SELECTION in flags:
        text += (" Line-specific evidence cannot settle which line faulted (comparable fault current, or "
                 "status channels that could not be attributed to a line) — confirm the faulted line, or "
                 "whether the fault was outside these lines (through-fault).")
    return text


def select_line_for_payload(payload: dict) -> Optional[LineSelection]:
    return select_line(
        payload.get("analog_channels") or [],
        payload.get("status_channels") or [],
        payload.get("time") or [],
        payload.get("frequency") or 50.0,
    )


def select_line_for_record(record: Any) -> Optional[LineSelection]:
    return select_line(
        list(getattr(record, "analog_channels", []) or []),
        list(getattr(record, "status_channels", []) or []),
        getattr(record, "time", None),
        getattr(record, "frequency", None) or 50.0,
    )


def _scoped_channels(analog_channels: list, status_channels: list, selection: LineSelection) -> tuple[list, list]:
    line_compacts = {_compact(line.key) for line in selection.lines}
    selected = selection.selected_compact
    own_slots = set()
    for channel in analog_channels:
        if _compact(line_key(_attr(channel, "name") or "")) == selected:
            own_slots.add((_attr(channel, "canonical_name") or "").upper())

    analog: list = []
    for channel in analog_channels:
        compact = _compact(line_key(_attr(channel, "name") or ""))
        if compact == selected:
            analog.append(channel)
        elif compact not in line_compacts:
            # Common/untagged channel — keep it unless it would shadow one of
            # the selected line's own channels in a first-match lookup.
            if (_attr(channel, "canonical_name") or "").upper() not in own_slots:
                analog.append(channel)

    display_keys = [line.key for line in selection.lines]
    status: list = []
    for channel in status_channels:
        owner = status_line_key(_attr(channel, "name") or "", display_keys)
        if owner is None or _compact(owner) == selected:
            status.append(channel)
    return analog, status


def scope_payload_with_selection(payload: dict) -> tuple[dict, Optional[LineSelection]]:
    """``payload`` restricted to its disturbed line (shallow copy; samples are
    shared), plus the selection that chose it. Returned unchanged with
    ``None`` when the record is not multi-line, so scoping an already-scoped
    payload is a no-op."""
    selection = select_line_for_payload(payload)
    if selection is None:
        return payload, None
    analog, status = _scoped_channels(
        payload.get("analog_channels") or [], payload.get("status_channels") or [], selection,
    )
    return {**payload, "analog_channels": analog, "status_channels": status}, selection


def scope_payload(payload: dict) -> dict:
    return scope_payload_with_selection(payload)[0]


class _ScopedRecord:
    """Attribute view of a parsed record with only the selected line's channels;
    every other attribute (time, frequency, ...) is delegated to the original."""

    def __init__(self, base: Any, analog_channels: list, status_channels: list, selection: LineSelection):
        self._base = base
        self.analog_channels = analog_channels
        self.status_channels = status_channels
        self.line_selection = selection

    def __getattr__(self, name: str) -> Any:
        if name == "_base":  # not yet set (copy/unpickle) — avoid infinite recursion
            raise AttributeError(name)
        return getattr(self._base, name)


def scope_record(record: Any) -> Any:
    """Object-shaped counterpart of ``scope_payload`` (ComtradeRecord or the
    event-analysis shim)."""
    selection = select_line_for_record(record)
    if selection is None:
        return record
    analog, status = _scoped_channels(
        list(record.analog_channels or []), list(record.status_channels or []), selection,
    )
    return _ScopedRecord(record, analog, status, selection)
