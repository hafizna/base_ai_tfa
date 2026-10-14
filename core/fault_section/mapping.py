"""What each recorded channel measures, inferred from the records themselves.

A case may list only its record files. This module works out what the
estimate needs from them, the way an engineer reads an unfamiliar bay without
its single-line diagram — each step a written rule, each decision logged with
its evidence:

1. *Relay roles* from folder and file names (CCP, CBF, BusPro, LCD, a line
   "MAIN … BAY …", OCR/incoming, transformer differential), else from channel
   signatures (a differential trip next to phase voltages is a line
   differential; W1/W2 sets or 87T signals a transformer differential). The
   roles, or the incident's asset type, choose the bay template: a 1½-breaker
   diameter or a transformer bay.
2. *Current sets*: three phase channels sharing one name pattern
   ("IA-2/IB-2/IC-2", "CB1.ia/ib/ic", GE "F1-IA/F2-IB/F3-IC"); computed
   channels (differential, bias, neutral) and incomplete sets are left out.
3. *Same CT*: two current sets of one station whose phase magnitudes agree
   within a few percent through fault-level current are one CT seen by two
   relays (needs the clocks lined up first, see ``observe``).
4. *CT roles*: a line relay reads the line CT; a CBF relay reads its own
   breaker's CT; a CCP reads the line CT and the two breaker CTs of one T-node
   — Kirchhoff's current law across its three terminals before the fault
   confirms it — so once two of its terminals are known the third follows; a
   set from another station is the far-end line CT. In a transformer bay the
   differential's set with the smaller load current is the HV side, an OCR's
   folder says its side, and a single REF/neutral channel with an HV/LV hint is
   that side's neutral CT. What the rules leave open comes back as alternative
   assignments for the estimate to test.
5. *Breaker states* from status names: the contact kind (52b, "OPN", "OPEN"
   read open; 52a, "CLS", "CL" read closed) and which breaker ("CB_B",
   "CB_AB", "B.B D2" = Bus B diameter 2, "CB1"/"CB2" of a dual-breaker line
   relay, a CBF relay's own "CB OPEN").
6. *Zones*: each CCP, busbar and line-differential relay bounds a zone; its
   operate signal is the differential trip/operate status; a busbar relay whose
   event log spans the incident without one never operated. A distance relay's
   zone trips say "forward" for its line CT. A transformer differential gives
   the 87T zone, REF HV/LV ("REF DIF W1/W2") and the mechanical zone
   (Buchholz / sudden pressure trips, weak evidence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .observe import CurrentTrace, _phasors, dedupe, load_level, match_clock, onsets, thresholds

_PHASE_OF = {"A": "A", "R": "A", "1": "A", "B": "B", "S": "B", "2": "B", "C": "C", "T": "C", "3": "C"}
_COMPUTED = re.compile(r"DIFF|BIAS|REST|MAX|SENS|REF\b|LOZ|CHECK|SYNC", re.I)
_GE_SLOT = re.compile(r"^([A-Z])(\d+)-([IV])([ABC])$", re.I)
_PHASE_TOKEN = re.compile(r"(?<![A-Z0-9])([IUV])(L?)([ABCRST123])(?![A-Z0-9])", re.I)
# "CT PHASE L1", "VT PHASE L2": the measurement in its own word, the phase as L1/L2/L3 or R/S/T.
_MEASURE_WORD = re.compile(r"(?<![A-Z0-9])(CT|ARUS|CURRENT|VT|PT|CVT|TEGANGAN|VOLTAGE)(?![A-Z0-9])", re.I)
_LOOSE_PHASE = re.compile(r"(?<![A-Z0-9])(L[123]|[RST])(?![A-Z0-9])", re.I)


# --- names ------------------------------------------------------------------------

def station_key(name: Optional[str]) -> str:
    """Consonant skeleton of a substation name, so "GITET PEMALANG", "PMLNG"
    and "Pemalang" agree, and so do "MANNDIRNCN" and "Mandirancan"."""
    text = re.sub(r"^(GITET|GIS|GI)\b", "", str(name or "").strip().upper()).strip()
    letters = re.sub(r"[^A-Z]", "", text)
    skeleton = re.sub(r"[AEIOU]", "", letters)
    return re.sub(r"(.)\1+", r"\1", skeleton)


def looks_like_station(name: Optional[str]) -> bool:
    return bool(re.match(r"^\s*(GITET|GIS|GI)\b", str(name or ""), re.I))


def phase_sets(names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Three-phase sets by name pattern: ``{key: {"A": name, "B": .., "C": ..}}``
    for current (key starts "I") and voltage (key starts "V") channels."""
    sets: dict[str, dict[str, list[str]]] = {}
    for name in names:
        clean = name.strip()
        slot = _GE_SLOT.match(clean)
        if slot:
            letter, number, kind, phase = slot.groups()
            key = f"{kind.upper()}:{letter.upper()}{(int(number) - 1) // 4}"
            sets.setdefault(key, {}).setdefault(_PHASE_OF[phase.upper()], []).append(clean)
            continue
        if _COMPUTED.search(clean):
            continue
        matches = list(_PHASE_TOKEN.finditer(clean))
        word = _MEASURE_WORD.search(clean)
        loose = list(_LOOSE_PHASE.finditer(clean))
        numbered = [m for m in loose if m.group(1).upper().startswith("L")]
        # "VT PHASE L1": the L-digit is the phase, "VT" only says voltage
        # (while in "VT BRINGIN 1" the T of VT is the phase).
        if word and len(numbered) == 1:
            loose = numbered
        elif len(matches) == 1:
            m = matches[0]
            kind = "I" if m.group(1).upper() == "I" else "V"
            key = f"{kind}:{(clean[:m.start(3)] + '*' + clean[m.end(3):]).lower()}"
            sets.setdefault(key, {}).setdefault(_PHASE_OF[m.group(3).upper()], []).append(clean)
            continue
        if not word or len(loose) != 1:
            continue
        m = loose[0]
        kind = "I" if word.group(1).upper() in ("CT", "ARUS", "CURRENT") else "V"
        key = f"{kind}:{(clean[:m.start()] + '*' + clean[m.end():]).lower()}"
        sets.setdefault(key, {}).setdefault(_PHASE_OF[m.group(1)[-1].upper()], []).append(clean)
    return {key: {ph: chans[0] for ph, chans in by_phase.items()}
            for key, by_phase in sets.items()
            if set(by_phase) == {"A", "B", "C"} and all(len(c) == 1 for c in by_phase.values())}


# --- relay roles --------------------------------------------------------------------

@dataclass(frozen=True)
class RelayRole:
    kind: str = "unknown"       # "ccp" | "busbar" | "line_diff" | "line" | "cbf" | "transformer_diff" | "ocr" | "unknown"
    main: str = ""              # "A", "B", "1", "2"
    bus: str = ""               # busbar relay: "a" | "b"
    breaker: str = ""           # CBF relay: "a" | "ab" | "b"
    bay: str = ""               # CCP: the T-node it guards, "a" | "b"
    bay_label: str = ""         # line relay: "Mandirancan 1"
    side: str = ""              # transformer OCR: "hv" | "lv"
    evidence: str = ""


_DIFF_OPERATE = re.compile(r"(I?DIFF?|87[LBT]?).*\b(TRIP|OP|OPERATE|ON)\b|\bBUS\s*\d*\s*(OP|TRIP)\b", re.I)
_OCR_NAME = re.compile(r"\bOCR\b|\bGFR\b|\b50\s*/\s*51\b|\bINCOMING\b|\bINC\b", re.I)
_TRAFO_NAME = re.compile(r"\b87T\b|\bDIFF?\b|\bTRF\s*#?\d*\b|\bTRAFO\b|\bIBT\b|TRANSFORM", re.I)
_TRAFO_SIGNAL = re.compile(r"87T|T\dWPDIF|XFMR|\bW[12]\b", re.I)
_OCR_SIGNAL = re.compile(r"PTOC|\bI>|\bIN>|\bIO>", re.I)


def _side(text: str) -> str:
    t = text.upper()
    if re.search(r"\bHV\b|\b150\s*KV\b|\bPRIMER\b|\bW1\b", t):
        return "hv"
    if re.search(r"\bLV\b|\b20\s*KV\b|\bINC(OMING)?\b|\bSEKUNDER\b|\bMV\b|\bW2\b", t):
        return "lv"
    return ""
_NOT_OPERATE = re.compile(r"PKP|START|ALARM|ALM|BLOCK|BLK|\bHS\d|DPO|TEST", re.I)
_PHASE_SUFFIX = re.compile(r"[\s_]+[ABCRST]$", re.I)
_ZONE_TRIP = re.compile(r"^(ZONE|Z)\s*[123]\s*TRIP$|^21[MQ][123]\.OP$", re.I)


def operate_signals(names: Iterable[str]) -> list[str]:
    """Differential operate statuses; the three-phase one when it exists."""
    found = [n for n in names if _DIFF_OPERATE.search(n) and not _NOT_OPERATE.search(n)]
    general = [n for n in found if not _PHASE_SUFFIX.search(n)]
    return general or found


def zone_trip_signals(names: Iterable[str]) -> list[str]:
    return [n for n in names if _ZONE_TRIP.match(n.strip())]


def relay_role(folder: str, files: Iterable[str] = (), signals: Iterable[str] = (),
               has_voltages: bool = False, set_keys: Iterable[str] = ()) -> RelayRole:
    """Role of a relay from its folder and file names, else its channels."""
    text = " | ".join([folder, *files])
    t = re.sub(r"[_]+", " ", text.upper())
    main = re.search(r"\bMAIN\s*([AB12])\b", t)
    main_id = main.group(1) if main else ""
    bays = (re.search(r"\bBAY\s+([A-Z][A-Z .]*?\d*)\s*$", re.sub(r"[_\s]+", " ", part.upper()).strip())
            for part in re.split(r"[\\/]", folder))
    bay = next((b for b in bays if b), None)
    bay_label = bay.group(1).strip().title() if bay else ""
    if re.search(r"\bCBF\b|BREAKER FAIL", t):
        letter = re.search(r"\bCBF\s+(AB|A|B)\b", t)
        return RelayRole("cbf", main_id, breaker=letter.group(1).lower() if letter else "",
                         evidence=f"nama '{text}' memuat CBF" + (f" {letter.group(1)}" if letter else ""))
    if re.search(r"\bCCP\b", t):
        pair = re.search(r"\b(AB|A|B)\s*-\s*(AB|A|B)\b", t)
        side = next((x for x in (pair.groups() if pair else ()) if x != "AB"), "")
        return RelayRole("ccp", main_id, bay=side.lower(), evidence=f"nama '{text}' memuat CCP" + (f" {pair.group(0)}" if pair else ""))
    if re.search(r"BUS\s*PRO|BUSBAR|\bBBP\b", t):
        bus = re.search(r"BUS\s*PRO\s+([AB])\b|\bBUS\s*([AB])\b", t)
        letter = (bus.group(1) or bus.group(2)).lower() if bus else ""
        return RelayRole("busbar", main_id, bus=letter, evidence=f"nama '{text}' memuat BUSPRO")
    if re.search(r"\bLCD\b|\b87L\b|LINE DIFF", t):
        return RelayRole("line_diff", main_id, bay_label=bay_label, evidence=f"nama '{text}' memuat LCD")
    if _OCR_NAME.search(t):
        side = _side(t)
        return RelayRole("ocr", main_id, side=side,
                         evidence=f"nama '{text}' memuat OCR/incoming" + (f", sisi {side.upper()}" if side else ""))
    if _TRAFO_NAME.search(t):
        return RelayRole("transformer_diff", main_id, evidence=f"nama '{text}' memuat diff/trafo")
    names = list(signals)
    keys = list(set_keys)
    if any(_TRAFO_SIGNAL.search(n) for n in names + keys):
        return RelayRole("transformer_diff", main_id, evidence="kanal diff trafo (87T/W1/W2)")
    if names and sum(bool(_OCR_SIGNAL.search(n)) for n in names) >= 2:
        return RelayRole("ocr", main_id, side=_side(t), evidence="kanal overcurrent (PTOC/I>)")
    if operate_signals(names) and has_voltages:
        return RelayRole("line_diff", main_id, bay_label=bay_label,
                         evidence=f"kanal {operate_signals(names)[0]!r} dengan tegangan fasa")
    if bay_label and main_id:
        return RelayRole("line", main_id, bay_label=bay_label, evidence=f"nama '{text}': MAIN {main_id} BAY {bay_label}")
    if zone_trip_signals(names) and has_voltages:
        return RelayRole("line", main_id, bay_label=bay_label, evidence="kanal zona distance dengan tegangan fasa")
    return RelayRole(evidence="tidak dikenali dari nama maupun kanal")


# --- breaker status names --------------------------------------------------------------

@dataclass(frozen=True)
class BreakerStatus:
    open_when: bool             # True: the signal is high when the breaker is open
    letter: str = ""            # "a" | "ab" | "b" (CB_B, CB_AB ...)
    number: int = 0             # CB1 / CB2 of a dual-breaker relay
    bus: str = ""               # "b" of "B.B D2"
    diameter: int = 0           # 2 of "B.B D2" / "53 P2"
    convention: str = ""        # vendor convention the reading relies on


_CB_SKIP = re.compile(r"TRIP|TRP|ALM|ALARM|INIT|FAIL|MANCLS|\bMAN\b|\b25\b|SYN|CTS|HEALTH|SERV|\bDS\b|\bES\b|BRF|&", re.I)
_OPEN_WORDS = {"OPN", "OPEN", "OP", "OPENED", "52B"}
_CLOSED_WORDS = {"CLS", "CLOSE", "CL", "CLOSED", "52A"}


def breaker_status(name: str) -> Optional[BreakerStatus]:
    raw = name.strip().upper()
    if _CB_SKIP.search(raw):
        return None
    tokens = [t for t in re.split(r"[\s_.\-]+", raw) if t]
    if tokens and re.fullmatch(r"[LBR]\d+", tokens[0]) and len(tokens) > 1:
        tokens = tokens[1:]                          # opto / relay label ("L1", "B08")
    ge_point = re.search(r"\b53\s*P(\d+)\b", raw)
    has_cb = any(t == "CB" or re.fullmatch(r"CB\d", t) for t in tokens) or ge_point or "52A" in tokens or "52B" in tokens
    if not has_cb:
        return None
    is_open = any(t in _OPEN_WORDS for t in tokens)
    is_closed = any(t in _CLOSED_WORDS for t in tokens)
    convention = ""
    if ge_point and not is_open and not is_closed:
        is_closed, convention = True, "GE '53 Pn' = kontak CB tertutup"
    if is_open == is_closed:
        return None
    bus_dia = re.search(r"\b([AB])\.B\s*D(\d+)\b", raw)
    if bus_dia:
        return BreakerStatus(is_open, bus=bus_dia.group(1).lower(), diameter=int(bus_dia.group(2)))
    if ge_point:
        return BreakerStatus(is_open, diameter=int(ge_point.group(1)), convention=convention)
    for i, t in enumerate(tokens):
        if t == "CB" and i + 1 < len(tokens) and tokens[i + 1] in ("A", "AB", "B"):
            return BreakerStatus(is_open, letter=tokens[i + 1].lower())
        numbered = re.fullmatch(r"CB(\d)", t)
        if numbered:
            return BreakerStatus(is_open, number=int(numbered.group(1)),
                                 convention="CB1 = CB sisi bus, CB2 = CB tengah (rele line 1½ CB)")
    return BreakerStatus(is_open)


# --- the inference ---------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    subject: str
    rule: str
    evidence: str
    certain: bool = True


@dataclass
class SourceRecorder:
    """What the inference needs of one recorder."""
    id: str
    folder: str
    files: tuple[str, ...]
    station_names: tuple[str, ...]              # COMTRADE station field(s)
    current_sets: dict[str, list[CurrentTrace]]  # set key -> one trace per record file
    voltage_sets: dict[str, dict[str, str]]
    current_channels: dict[str, dict[str, str]]
    signals: tuple[str, ...]                    # status channel and event signal names
    has_event_log: bool = False
    ratings: dict[str, float] = field(default_factory=dict)          # set key -> CT primary rating
    single_currents: tuple[str, ...] = ()                            # current channels outside any set
    single_ratings: dict[str, float] = field(default_factory=dict)
    voltage_kv: float = 0.0                                           # VT primary, kV


@dataclass
class Mapping:
    template: str = "diameter"                  # "diameter" | "transformer"
    spec: dict[str, Any] = field(default_factory=dict)   # for build_diameter / build_transformer
    currents: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[list[dict[str, Any]]] = field(default_factory=list)   # other current mappings to test
    breakers: list[dict[str, Any]] = field(default_factory=list)
    zones: list[dict[str, Any]] = field(default_factory=list)
    direction_signals: list[dict[str, Any]] = field(default_factory=list)
    ct_ratings_a: dict[str, float] = field(default_factory=dict)
    reference_recorder: str = ""
    align_breakers: dict[str, str] = field(default_factory=dict)
    roles: dict[str, RelayRole] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)

    def note(self, subject: str, rule: str, evidence: str, certain: bool = True) -> None:
        self.decisions.append(Decision(subject, rule, evidence, certain))


def _labels(title: str, given: dict[str, str]) -> tuple[dict[str, str], int]:
    """CB names and the diameter number from "7B2" / "DIA #2" in the title."""
    labels = dict(given)
    code = re.search(r"\b(\d)(AB|A|B)(\d+)\b", title.upper())
    number = int(code.group(3)) if code else 0
    if not number:
        dia = re.search(r"\bDIA(?:METER)?\s*#?\s*(\d+)", title.upper())
        number = int(dia.group(1)) if dia else 0
    if code:
        prefix = code.group(1)
        for key, letter in (("cb_a", "A"), ("cb_ab", "AB"), ("cb_b", "B")):
            labels.setdefault(key, f"CB {prefix}{letter}{code.group(3)}")
    return labels, number


def _magnitudes(trace: CurrentTrace) -> tuple[np.ndarray, np.ndarray]:
    """``(t_end incident clock, magnitudes (3, windows))``."""
    n = trace.window()
    ends, ph = _phasors(trace.phases, n)
    return trace.t[ends] + trace.offset_s, np.abs(ph)


def same_ct_score(a: Sequence[CurrentTrace], b: Sequence[CurrentTrace], load_a: float, load_b: float,
                  min_windows: int = 8) -> Optional[float]:
    """Median relative phase-magnitude difference over the windows where either
    set carries fault-level current (above twice the larger load), or None when
    they share too few such windows to compare."""
    floor = max(2.0 * max(load_a, load_b), 50.0)
    errors = []
    for ta in a:
        t1, m1 = _magnitudes(ta)
        for tb in b:
            t2, m2 = _magnitudes(tb)
            lo, hi = max(t1[0], t2[0]), min(t1[-1], t2[-1])
            sel = (t1 >= lo) & (t1 <= hi)
            if not np.any(sel):
                continue
            for p in range(3):
                x = m1[p, sel]
                y = np.interp(t1[sel], t2, m2[p])
                big = np.maximum(x, y)
                keep = big > floor
                errors.extend((np.abs(x - y)[keep] / big[keep]).tolist())
    if len(errors) < 3 * min_windows:
        return None
    return float(np.median(errors))


def kcl_residual(traces: Sequence[CurrentTrace], cycles: float = 2.0) -> Optional[float]:
    """Smallest ``|I1 ± I2 ± I3| / max |I|`` (worst phase, median over the
    first ``cycles``) for three sets of one record. They share its sampling,
    so phasors of the same DFT window compare in angle."""
    if len(traces) != 3 or len({len(t.t) for t in traces}) != 1:
        return None
    n = traces[0].window()
    phasors = [_phasors(trace.phases, n)[1] for trace in traces]      # (phase, window) each
    windows = slice(0, max(1, int(cycles * 4)))
    scale = np.max([np.abs(p[:, windows]) for p in phasors], axis=0)
    best = None
    for s2 in (1, -1):
        for s3 in (1, -1):
            total = phasors[0][:, windows] + s2 * phasors[1][:, windows] + s3 * phasors[2][:, windows]
            residual = float(np.median(np.max(np.abs(total) / np.maximum(scale, 1e-9), axis=0)))
            best = residual if best is None else min(best, residual)
    return best


@dataclass
class _Sets:
    """Every recorder's current sets, lined up on one clock and grouped by
    the physical CT they measure."""
    by_id: dict[str, SourceRecorder]
    remote: set[str]
    loads: dict[tuple[str, str], float]
    cluster_of: dict[tuple[str, str], tuple[str, str]]
    clusters: dict[tuple[str, str], list[tuple[str, str]]]
    offsets: dict[str, float]

    def text(self, cluster: tuple[str, str]) -> str:
        return _cluster_text(self.by_id, self.clusters[cluster])


class _Assigner:
    """CT roles of clusters, first rule wins, conflicts noted."""

    def __init__(self, mapping: Mapping, sets: _Sets):
        self.mapping, self.sets = mapping, sets
        self.role_of: dict[tuple[str, str], str] = {}

    def __call__(self, cluster, role: str, rule: str, evidence: str, certain: bool = True) -> None:
        if cluster in self.role_of and self.role_of[cluster] != role:
            self.mapping.note(role, "konflik", f"{self.sets.text(cluster)} sudah {self.role_of[cluster]}; {evidence}", False)
            return
        if cluster not in self.role_of:
            self.role_of[cluster] = role
            self.mapping.note(f"{role} ← {self.sets.text(cluster)}", rule, evidence, certain)


def infer_mapping(case: dict[str, Any], sources: list[SourceRecorder]) -> Mapping:
    mapping = Mapping()
    home = case.get("station") or ""

    remote: set[str] = set()
    for src in sources:
        role = relay_role(src.folder, src.files, src.signals, bool(src.voltage_sets), src.current_sets.keys())
        mapping.roles[src.id] = role
        mapping.note(src.id, "peran rele", f"{role.kind}" + (f" main {role.main}" if role.main else "") + f" — {role.evidence}")
        foreign = [s for s in src.station_names if looks_like_station(s) and station_key(s) != station_key(home)]
        if home and foreign:
            remote.add(src.id)
            mapping.note(src.id, "gardu", f"ujung lawan: header COMTRADE '{foreign[0]}' bukan {home}")

    mapping.template = _template(case, mapping.roles)
    mapping.note("bay", "template", "trafo" if mapping.template == "transformer" else "diameter 1½ CB")
    sets = _line_up_and_group(case, sources, remote, mapping)
    if mapping.template == "transformer":
        _transformer_mapping(case, sources, sets, mapping)
    else:
        _diameter_mapping(case, sources, sets, mapping)
    return mapping


def _template(case: dict[str, Any], roles: dict[str, RelayRole]) -> str:
    asset = str(case.get("asset_type") or "").upper()
    if asset == "TRANSFORMER" or "transformer" in case:
        return "transformer"
    if asset in ("TRANSMISSION_LINE", "BUSBAR") or "diameter" in case:
        return "diameter"
    kinds = {r.kind for r in roles.values()}
    if kinds & {"transformer_diff", "ocr"} and not kinds & {"ccp", "cbf", "busbar", "line", "line_diff"}:
        return "transformer"
    return "diameter"


def _line_up_and_group(case: dict[str, Any], sources: list[SourceRecorder], remote: set[str],
                       mapping: Mapping) -> _Sets:
    loads: dict[tuple[str, str], float] = {}
    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for src in sources:
        for key, traces in src.current_sets.items():
            loads[(src.id, key)] = max((load_level(t) or 0.0) for t in traces)
            limits[(src.id, key)] = thresholds(loads[(src.id, key)], src.ratings.get(key))
    own_onsets = {}
    for src in sources:
        found = [t for key, traces in src.current_sets.items() for tr in traces
                 for t in onsets(tr, limits[(src.id, key)][0])]
        if found:
            own_onsets[src.id] = dedupe(found)
    candidates = [s for s in sources if s.id in own_onsets and s.id not in remote]
    reference = case.get("reference_recorder") or (max(
        candidates,
        key=lambda s: (len(own_onsets[s.id]), max(tr.t[-1] - tr.t[0] for trs in s.current_sets.values() for tr in trs)),
    ).id if candidates else next(s.id for s in sources if s.current_sets))
    mapping.reference_recorder = reference
    offsets = {reference: 0.0}
    for src in sources:
        if src.id != reference and src.id in own_onsets and reference in own_onsets:
            match = match_clock(own_onsets[reference], own_onsets[src.id])
            if match and match.anchors:
                offsets[src.id] = match.offset_s
    for src in sources:
        for traces in src.current_sets.values():
            for tr in traces:
                tr.offset_s = offsets.get(src.id, 0.0)

    by_id = {s.id: s for s in sources}
    live = [(src.id, key) for src in sources for key in src.current_sets
            if max(np.max(_magnitudes(t)[1]) for t in src.current_sets[key]) > 10.0]
    for src in sources:
        for key in src.current_sets:
            if (src.id, key) not in live:
                mapping.note(f"{src.id} {key}", "set arus kosong", "tidak pernah > 10 A — tidak terpasang", certain=True)
    parent = {s: s for s in live}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if (a[0] in remote) != (b[0] in remote):
                continue
            score = same_ct_score(by_id[a[0]].current_sets[a[1]], by_id[b[0]].current_sets[b[1]], loads[a], loads[b])
            if score is not None and score < 0.08:
                parent[find(a)] = find(b)
                mapping.note(f"{_set_text(by_id, a)} = {_set_text(by_id, b)}", "CT yang sama",
                             f"magnitudo per fasa beda median {score:.1%} saat arus gangguan")
    cluster_of = {s: find(s) for s in live}
    clusters: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for s in live:
        clusters.setdefault(cluster_of[s], []).append(s)
    return _Sets(by_id, remote, loads, cluster_of, clusters, offsets)


def _current_specs(mapping: Mapping, sets: _Sets, roles: dict, forward_of) -> list[dict[str, Any]]:
    """One spec per (CT, relay); ``forward_of(source, relay role, CT role)``
    names the section a relay with voltages calls forward, or None."""
    specs = []
    for cluster, members in sets.clusters.items():
        if cluster not in roles:
            continue
        seen = set()
        for rid, key in members:
            if rid in seen:          # one relay reading the same CT twice (sum and CB1 inputs)
                continue
            seen.add(rid)
            src = sets.by_id[rid]
            chans = src.current_channels[key]
            spec = {"ct": roles[cluster], "recorder": rid, "channels": [chans["A"], chans["B"], chans["C"]]}
            forward = forward_of(src, mapping.roles[rid], roles[cluster])
            if forward and len(src.voltage_sets) == 1:
                (volts,) = src.voltage_sets.values()
                spec.update(voltages=[volts["A"], volts["B"], volts["C"]], forward=forward)
            specs.append(spec)
    return specs


def _ratings(sets: _Sets, roles: dict) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for cluster, members in sets.clusters.items():
        role = roles.get(cluster)
        for rid, key in members:
            rating = sets.by_id[rid].ratings.get(key)
            if role and rating:
                ratings[role] = max(ratings.get(role, 0.0), rating)
    return ratings


def _group_breakers(mapping: Mapping, grouped: dict, conventions: dict) -> None:
    for (rid, breaker, convention), names in conventions.items():
        mapping.note(f"{breaker} ← {rid}: {', '.join(names)}", "konvensi", convention, False)
    for (rid, breaker, open_when), names in grouped.items():
        mapping.breakers.append({"breaker": breaker, "recorder": rid, "channels": names, "open_when": open_when})
        mapping.note(f"{breaker} ← {rid}: {', '.join(names[:3])}{' …' if len(names) > 3 else ''}", "status CB",
                     f"kontak {'52b/OPEN (tinggi = open)' if open_when else '52a/CLOSE (tinggi = tertutup)'}")


def _align_event_recorders(sources: list[SourceRecorder], sets: _Sets, mapping: Mapping, prefer: list[str]) -> None:
    """Event-only recorders line up on a breaker an aligned recorder also reports."""
    reported = {b["breaker"] for b in mapping.breakers if b["recorder"] in sets.offsets}
    for src in sources:
        if src.id in sets.offsets:
            continue
        own = [b["breaker"] for b in mapping.breakers if b["recorder"] == src.id and b["breaker"] in reported]
        if own:
            mapping.align_breakers[src.id] = sorted(own, key=lambda b: (b not in prefer, prefer.index(b) if b in prefer else 0, b))[0]


# --- 1½-breaker diameter -------------------------------------------------------------

def _diameter_mapping(case: dict[str, Any], sources: list[SourceRecorder], sets: _Sets, mapping: Mapping) -> None:
    labels, own_diameter = _labels(case.get("title") or "", (case.get("diameter") or {}).get("labels") or {})
    remote, by_id = sets.remote, sets.by_id
    # Which T-node the recorded line hangs on: a CCP says it, else B.
    bay = next((r.bay for r in mapping.roles.values() if r.kind == "ccp" and r.bay), "b")
    bus_side = {"a": "cb_a", "b": "cb_b"}[bay]
    assign = _Assigner(mapping, sets)

    for cluster, members in sets.clusters.items():
        for rid, key in members:
            role = mapping.roles[rid]
            if rid in remote and role.kind in ("line", "line_diff"):
                assign(cluster, f"ct_line_{bay}_remote", "rele line ujung lawan", f"{rid} ada di gardu lain")
            elif role.kind in ("line", "line_diff") and by_id[rid].voltage_sets:
                assign(cluster, f"ct_line_{bay}", "rele line", f"{rid} adalah rele {role.kind} dengan tegangan line")
    for cluster, members in sets.clusters.items():
        for rid, key in members:
            role = mapping.roles[rid]
            if role.kind == "cbf" and role.breaker and rid not in remote:
                assign(cluster, f"ct_cb_{role.breaker}", "rele CBF", f"{rid} adalah CBF {role.breaker.upper()}")
    open_roles: list[dict] = [{}]
    for src in sources:
        role = mapping.roles[src.id]
        keys = [k for k in src.current_sets if (src.id, k) in sets.cluster_of]
        if role.kind != "ccp" or len(keys) != 3 or src.id in remote:
            continue
        residual = kcl_residual([src.current_sets[k][0] for k in keys])
        if residual is not None:
            mapping.note(src.id, "KCL", f"tiga terminal CCP: |ΣI|/max|I| = {residual:.1%} sebelum gangguan"
                         + (" — satu T-node" if residual < 0.1 else " — bukan satu T-node?"), residual < 0.1)
        terminal_clusters = [sets.cluster_of[(src.id, k)] for k in keys]
        wanted = [f"ct_line_{bay}", f"ct_{bus_side}", "ct_cb_ab"]
        free_roles = [r for r in wanted if r not in assign.role_of.values()]
        free = [c for c in terminal_clusters if c not in assign.role_of]
        if len(free) == 1 and len(free_roles) == 1:
            assign(free[0], free_roles[0], "terminal ketiga CCP", f"dua terminal lain {src.id} sudah diketahui")
        elif free and len(free) == len(free_roles):
            options = list(permutations(free_roles))
            mapping.note(src.id, "ambigu", f"{len(free)} terminal CCP tanpa petunjuk; {len(options)} kemungkinan diuji estimator", False)
            open_roles = [dict(zip(free, perm)) for perm in options]

    def forward_of(src, role, ct):
        return f"line_{bay}" if role.kind in ("line", "line_diff") and ct.startswith("ct_line") else None

    mapping.currents = _current_specs(mapping, sets, {**assign.role_of, **open_roles[0]}, forward_of)
    mapping.alternatives = [_current_specs(mapping, sets, {**assign.role_of, **alt}, forward_of) for alt in open_roles[1:]]
    for cluster, members in sets.clusters.items():
        if cluster not in assign.role_of and all(cluster not in alt for alt in open_roles):
            mapping.note(sets.text(cluster), "tanpa peran", "tidak ada aturan yang menentukan CT ini", False)
    mapping.ct_ratings_a = _ratings(sets, {**assign.role_of, **open_roles[0]})

    bus_bays: dict[str, set[int]] = {}
    grouped: dict[tuple[str, str, bool], list[str]] = {}
    conventions: dict[tuple[str, str, str], list[str]] = {}
    for src in sources:
        role = mapping.roles[src.id]
        for name in src.signals:
            status = breaker_status(name)
            if status is None:
                continue
            breaker = _breaker_role(status, role, src.id in remote, bay, bus_side, own_diameter, bus_bays)
            if breaker is None:
                continue
            grouped.setdefault((src.id, breaker, status.open_when), []).append(name)
            if status.convention:
                conventions.setdefault((src.id, breaker, status.convention), []).append(name)
    _group_breakers(mapping, grouped, conventions)
    _align_event_recorders(sources, sets, mapping, [bus_side, "cb_ab"])

    zone_ids: dict[tuple[str, str, str], str] = {}
    for src in sources:
        role = mapping.roles[src.id]
        if role.kind in ("ccp", "line_diff", "busbar"):
            bus = role.bus or ("b" if bus_side == "cb_b" else "a")
            zid = {"ccp": f"ccp_main_{role.main.lower() or 'x'}",
                   "line_diff": f"lcd_main_{role.main.lower() or 'x'}",
                   "busbar": f"busbar_{bus}_main_{role.main.lower() or 'x'}"}[role.kind]
            zone_ids[(role.kind, role.main, bus)] = zid
            operate = operate_signals(src.signals)
            if operate:
                mapping.zones.append({"zone": zid, "recorder": src.id, "channels": operate})
                mapping.note(f"{zid} ← {src.id}: {', '.join(operate)}", "sinyal operate", "status trip/operate diferensial")
            elif role.kind == "busbar" and src.has_event_log:
                mapping.zones.append({"zone": zid, "recorder": src.id, "never_operated": True, "weak": True})
                mapping.note(f"{zid} ← {src.id}", "tidak operate", "log event mencakup insiden tanpa trip diferensial", False)
        if role.kind in ("line", "line_diff") and src.id not in remote:
            for name in zone_trip_signals(src.signals):
                if any(s["recorder"] == src.id and s["ct"] == f"ct_line_{bay}" for s in mapping.currents):
                    mapping.direction_signals.append({"ct": f"ct_line_{bay}", "recorder": src.id, "channels": [name],
                                                      "value": "forward", "forward": f"line_{bay}"})

    diameter = dict(case.get("diameter") or {})
    diameter.setdefault("labels", labels)
    diameter.setdefault("ct_side", {"cb_a": "unknown", "cb_ab": "unknown", "cb_b": "unknown"})
    line_role = next((mapping.roles[s.id] for s in sources if s.id not in remote
                      and mapping.roles[s.id].kind in ("line", "line_diff") and mapping.roles[s.id].bay_label), None)
    if any(mapping.roles[s.id].kind in ("line", "line_diff") for s in sources):
        bay_label = line_role.bay_label if line_role else "line"
        far = re.sub(r"\s*\d+$", "", bay_label).strip()
        diameter.setdefault(f"bay_{bay}", {"kind": "line", "label": f"Line {bay_label}",
                                           "remote": f"GITET {far}" if far else "ujung lawan"})
    for bus, numbers in bus_bays.items():
        diameter.setdefault(f"bus_{bus}_bays", [{"id": f"d{n}", "label": f"Diameter {n}"} for n in sorted(numbers)])
    if "zones" not in diameter:
        zones = []
        for src in sources:
            role = mapping.roles[src.id]
            if role.kind not in ("ccp", "line_diff", "busbar"):
                continue
            bus = role.bus or ("b" if bus_side == "cb_b" else "a")
            zid = zone_ids[(role.kind, role.main, bus)]
            if any(z["id"] == zid for z in zones):
                continue
            spec = {"id": zid, "kind": role.kind, "label": re.sub(r"[_/]+", " ", src.folder).strip()}
            if role.kind == "busbar":
                spec.update(bus=bus, end_zone="unknown")
            else:
                spec["bay"] = bay
            zones.append(spec)
        diameter["zones"] = zones
    mapping.spec = diameter


def _breaker_role(status: BreakerStatus, role: RelayRole, is_remote: bool, bay: str, bus_side: str,
                  own_diameter: int, bus_bays: dict[str, set[int]]) -> Optional[str]:
    if is_remote:
        return f"cb_remote_{bay}"
    if status.diameter:
        bus = status.bus or role.bus or bus_side[-1]
        if own_diameter and status.diameter == own_diameter:
            return f"cb_{bus}"
        bus_bays.setdefault(bus, set()).add(status.diameter)
        return f"cb_d{status.diameter}"
    if status.letter:
        return f"cb_{status.letter}"
    if status.number:
        return bus_side if status.number == 1 else "cb_ab" if status.number == 2 else None
    if role.kind == "cbf" and role.breaker:
        return f"cb_{role.breaker}"
    return None


# --- transformer bay -----------------------------------------------------------------

_NEUTRAL = re.compile(r"\bREF\b|NEUTRAL|NETRAL|\bNGR\b|\bNCT\b|STAR\s*POINT|\bIY\b", re.I)
_NOT_NEUTRAL = re.compile(r"DIFF|BIAS|REST|SENS", re.I)
_REF_SIGNAL = re.compile(r"\bREF\b|\b64\s*REF\b|\b87N\b", re.I)
_MECH_SIGNAL = re.compile(r"BUCHH?OLT?Z|SUDDEN\s*PRESS|PRESSURE|\bPRD\b|JANSEN", re.I)


def _hv_lv(rid: str, keys: list[str], loads: dict) -> tuple[str, str, str, bool]:
    """Which of a transformer differential's two sets is the HV side: the one
    carrying the smaller load current (the current ratio is the inverse of the
    voltage ratio, whatever the vendor calls the windings); W1/HV/-1 names
    when there is no load to compare."""
    a, b = keys
    la, lb = loads[(rid, a)], loads[(rid, b)]
    if min(la, lb) > 5.0 and max(la, lb) / min(la, lb) > 1.5:
        hv, lv = (a, b) if la < lb else (b, a)
        return hv, lv, f"arus beban {min(la, lb):,.0f} A vs {max(la, lb):,.0f} A: arus lebih kecil = sisi tegangan tinggi", True
    named = [k for k in keys if re.search(r"w1|hv|-1\b", k, re.I)]
    if len(named) == 1:
        hv = named[0]
        return hv, (b if hv == a else a), "nama belitan W1/HV", True
    return a, b, "tanpa beban dan tanpa nama belitan: urutan kanal", False


def _transformer_mapping(case: dict[str, Any], sources: list[SourceRecorder], sets: _Sets, mapping: Mapping) -> None:
    assign = _Assigner(mapping, sets)
    local = [s for s in sources if s.id not in sets.remote]
    for src in local:
        role = mapping.roles[src.id]
        if role.kind != "transformer_diff":
            continue
        keys = [k for k in src.current_sets if (src.id, k) in sets.cluster_of]
        if len(keys) < 2:
            mapping.note(src.id, "diff trafo", f"hanya {len(keys)} set arus fasa: sisi HV/LV tidak bisa dipetakan", False)
            continue
        if len(keys) > 2:
            keys = sorted(keys, key=lambda k: sets.loads[(src.id, k)], reverse=True)[:2]
            mapping.note(src.id, "diff trafo", "lebih dari dua set arus (trafo tiga belitan?): dua dengan beban terbesar dipakai", False)
        hv, lv, evidence, certain = _hv_lv(src.id, keys, sets.loads)
        assign(sets.cluster_of[(src.id, hv)], "ct_hv", "rele diff trafo", evidence, certain)
        assign(sets.cluster_of[(src.id, lv)], "ct_lv", "rele diff trafo", evidence, certain)
    for src in local:
        role = mapping.roles[src.id]
        if role.kind != "ocr" or not role.side:
            continue
        for key in src.current_sets:
            if (src.id, key) in sets.cluster_of:
                assign(sets.cluster_of[(src.id, key)], f"ct_{role.side}", f"rele OCR {role.side.upper()}",
                       f"{src.id} adalah OCR sisi {role.side.upper()}")

    def forward_of(src, role, ct):
        return {"ct_hv": "hv_winding", "ct_lv": "lv_winding"}.get(ct) if role.kind in ("ocr", "transformer_diff") else None

    mapping.currents = _current_specs(mapping, sets, assign.role_of, forward_of)
    for cluster in sets.clusters:
        if cluster not in assign.role_of:
            mapping.note(sets.text(cluster), "tanpa peran", "tidak ada aturan yang menentukan CT ini", False)
    mapping.ct_ratings_a = _ratings(sets, assign.role_of)

    neutral_sides = set()
    for src in local:
        for name in src.single_currents:
            side = _side(name)
            if not side or not _NEUTRAL.search(name) or _NOT_NEUTRAL.search(name):
                continue
            mapping.currents.append({"ct": f"ct_{side}_n", "recorder": src.id, "channels": [name]})
            neutral_sides.add(side)
            if src.single_ratings.get(name):
                mapping.ct_ratings_a[f"ct_{side}_n"] = src.single_ratings[name]
            mapping.note(f"ct_{side}_n ← {src.id}:{name}", "CT netral", f"kanal tunggal '{name}' sisi {side.upper()}")

    zones: list[dict[str, Any]] = []

    def zone(zid: str, kind: str, label: str, src: SourceRecorder, names: list[str], weak: bool = False, **extra):
        if not any(z["id"] == zid for z in zones):
            zones.append({"id": zid, "kind": kind, "label": label, **extra})
        mapping.zones.append({"zone": zid, "recorder": src.id, "channels": names, **({"weak": True} if weak else {})})
        mapping.note(f"{zid} ← {src.id}: {', '.join(names)}", "sinyal operate", label + (" (bukti lemah)" if weak else ""))

    for src in local:
        if mapping.roles[src.id].kind != "transformer_diff":
            continue
        names = list(src.signals)
        diff = [n for n in operate_signals(names) if not _REF_SIGNAL.search(n)]
        if diff:
            zone("diff_87t", "transformer_diff", "Diff trafo (87T)", src, diff)
        for side in ("hv", "lv"):
            ref = [n for n in names if _REF_SIGNAL.search(n) and _side(n) == side and not _NOT_OPERATE.search(n)]
            if ref:
                zone(f"ref_{side}", "ref", f"REF {side.upper()}", src, ref, side=side)
                neutral_sides.add(side)
        mech = [n for n in names if _MECH_SIGNAL.search(n) and re.search(r"TRIP", n, re.I)]
        if mech:
            zone("mech", "mechanical", "Buchholz / sudden pressure", src, mech, weak=True)

    grouped: dict[tuple[str, str, bool], list[str]] = {}
    for src in local:
        role = mapping.roles[src.id]
        for name in src.signals:
            status = breaker_status(name)
            side = _side(name) or (role.side if role.kind == "ocr" else "")
            if status is not None and side:
                grouped.setdefault((src.id, f"cb_{side}", status.open_when), []).append(name)
    _group_breakers(mapping, grouped, {})
    _align_event_recorders(sources, sets, mapping, ["cb_hv", "cb_lv"])

    title = case.get("title") or ""
    unit = re.search(r"\b(TRAFO|TRF|IBT)\s*#?\s*(\d+)", title.upper())
    trafo = (("IBT" if unit.group(1) == "IBT" else "Trafo") + f" #{unit.group(2)}") if unit else "trafo"
    hv_kv = max((s.voltage_kv for s in local), default=0.0)
    spec = dict(case.get("transformer") or {})
    spec.setdefault("labels", {"trafo": trafo, "bus_hv": f"Bus {hv_kv:.0f} kV" if hv_kv else "Bus HV", "bus_lv": "Bus LV",
                               "cb_hv": f"CB HV {trafo}", "cb_lv": f"CB incoming {trafo}"})
    spec.setdefault("ct_side", {"cb_hv": "unknown", "cb_lv": "unknown"})
    spec.setdefault("hv_neutral", "hv" in neutral_sides)
    spec.setdefault("lv_neutral", "lv" in neutral_sides)
    spec.setdefault("lv_source", "unknown")
    spec.setdefault("zones", zones)
    mapping.spec = spec


def _set_text(by_id: dict[str, SourceRecorder], member: tuple[str, str]) -> str:
    rid, key = member
    chans = by_id[rid].current_channels[key]
    return f"{rid}:{chans['A']}/{chans['B']}/{chans['C']}"


def _cluster_text(by_id: dict[str, SourceRecorder], members: list[tuple[str, str]]) -> str:
    return " = ".join(_set_text(by_id, m) for m in members)
