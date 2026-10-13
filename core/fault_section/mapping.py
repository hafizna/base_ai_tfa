"""What each recorded channel measures, inferred from the records themselves.

A case may list only its record files. This module works out what the
estimate needs from them, the way an engineer reads an unfamiliar bay without
its single-line diagram — each step a written rule, each decision logged with
its evidence:

1. *Relay roles* from folder and file names (CCP, CBF, BusPro, LCD, a line
   "MAIN … BAY …"), else from channel signatures (a differential trip next to
   phase voltages is a line differential).
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
   set from another station is the far-end line CT. What the rules leave open
   comes back as alternative assignments for the estimate to test.
5. *Breaker states* from status names: the contact kind (52b, "OPN", "OPEN"
   read open; 52a, "CLS", "CL" read closed) and which breaker ("CB_B",
   "CB_AB", "B.B D2" = Bus B diameter 2, "CB1"/"CB2" of a dual-breaker line
   relay, a CBF relay's own "CB OPEN").
6. *Zones*: each CCP, busbar and line-differential relay bounds a zone; its
   operate signal is the differential trip/operate status; a busbar relay whose
   event log spans the incident without one never operated. A distance relay's
   zone trips say "forward" for its line CT.
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
        if len(matches) != 1:
            continue
        m = matches[0]
        kind = "I" if m.group(1).upper() == "I" else "V"
        key = f"{kind}:{(clean[:m.start(3)] + '*' + clean[m.end(3):]).lower()}"
        sets.setdefault(key, {}).setdefault(_PHASE_OF[m.group(3).upper()], []).append(clean)
    return {key: {ph: chans[0] for ph, chans in by_phase.items()}
            for key, by_phase in sets.items()
            if set(by_phase) == {"A", "B", "C"} and all(len(c) == 1 for c in by_phase.values())}


# --- relay roles --------------------------------------------------------------------

@dataclass(frozen=True)
class RelayRole:
    kind: str = "unknown"       # "ccp" | "busbar" | "line_diff" | "line" | "cbf" | "unknown"
    main: str = ""              # "A", "B", "1", "2"
    bus: str = ""               # busbar relay: "a" | "b"
    breaker: str = ""           # CBF relay: "a" | "ab" | "b"
    bay: str = ""               # CCP: the T-node it guards, "a" | "b"
    bay_label: str = ""         # line relay: "Mandirancan 1"
    evidence: str = ""


_DIFF_OPERATE = re.compile(r"(I?DIFF|87[LBT]?).*\b(TRIP|OP|OPERATE|ON)\b|\bBUS\s*\d*\s*(OP|TRIP)\b", re.I)
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
               has_voltages: bool = False) -> RelayRole:
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
    names = list(signals)
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
    ratings: dict[str, float] = field(default_factory=dict)   # set key -> CT primary rating


@dataclass
class Mapping:
    currents: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[list[dict[str, Any]]] = field(default_factory=list)   # other current mappings to test
    breakers: list[dict[str, Any]] = field(default_factory=list)
    zones: list[dict[str, Any]] = field(default_factory=list)
    direction_signals: list[dict[str, Any]] = field(default_factory=list)
    diameter: dict[str, Any] = field(default_factory=dict)
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


def infer_mapping(case: dict[str, Any], sources: list[SourceRecorder]) -> Mapping:
    mapping = Mapping()
    home = case.get("station") or ""
    labels, own_diameter = _labels(case.get("title") or "", (case.get("diameter") or {}).get("labels") or {})

    # Relay roles and station side.
    remote: set[str] = set()
    for src in sources:
        role = relay_role(src.folder, src.files, src.signals, bool(src.voltage_sets))
        mapping.roles[src.id] = role
        mapping.note(src.id, "peran rele", f"{role.kind}" + (f" main {role.main}" if role.main else "") + f" — {role.evidence}")
        foreign = [s for s in src.station_names if looks_like_station(s) and station_key(s) != station_key(home)]
        if home and foreign:
            remote.add(src.id)
            mapping.note(src.id, "gardu", f"ujung lawan: header COMTRADE '{foreign[0]}' bukan {home}")

    # Which T-node the recorded line hangs on: a CCP says it, else B.
    bay = next((r.bay for r in mapping.roles.values() if r.kind == "ccp" and r.bay), "b")
    bus_side = {"a": "cb_a", "b": "cb_b"}[bay]

    # Reference clock and alignment of every recorder with current sets.
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
    reference = case.get("reference_recorder") or max(
        (s for s in sources if s.id in own_onsets and s.id not in remote),
        key=lambda s: (len(own_onsets[s.id]), max(tr.t[-1] - tr.t[0] for trs in s.current_sets.values() for tr in trs)),
    ).id
    mapping.reference_recorder = reference
    offsets = {reference: 0.0}
    for src in sources:
        if src.id != reference and src.id in own_onsets:
            match = match_clock(own_onsets[reference], own_onsets[src.id])
            if match and match.anchors:
                offsets[src.id] = match.offset_s
    for src in sources:
        for traces in src.current_sets.values():
            for tr in traces:
                tr.offset_s = offsets.get(src.id, 0.0)

    # Same CT across relays (union-find over sets of one station side).
    sets = [(src.id, key) for src in sources for key in src.current_sets
            if max(np.max(_magnitudes(t)[1]) for t in src.current_sets[key]) > 10.0]
    by_id = {s.id: s for s in sources}
    for src in sources:
        for key in src.current_sets:
            if (src.id, key) not in sets:
                mapping.note(f"{src.id} {key}", "set arus kosong", "tidak pernah > 10 A — tidak terpasang", certain=True)
    parent = {s: s for s in sets}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            if (a[0] in remote) != (b[0] in remote):
                continue
            score = same_ct_score(by_id[a[0]].current_sets[a[1]], by_id[b[0]].current_sets[b[1]], loads[a], loads[b])
            if score is not None and score < 0.08:
                parent[find(a)] = find(b)
                mapping.note(f"{_set_text(by_id, a)} = {_set_text(by_id, b)}", "CT yang sama",
                             f"magnitudo per fasa beda median {score:.1%} saat arus gangguan")
    clusters: dict[Any, list[tuple[str, str]]] = {}
    for s in sets:
        clusters.setdefault(find(s), []).append(s)

    # CT roles.
    role_of: dict[Any, str] = {}

    def assign(cluster, role, rule, evidence, certain=True):
        if cluster in role_of and role_of[cluster] != role:
            mapping.note(role, "konflik", f"{_cluster_text(by_id, clusters[cluster])} sudah {role_of[cluster]}; {evidence}", False)
            return
        if cluster not in role_of:
            role_of[cluster] = role
            mapping.note(f"{role} ← {_cluster_text(by_id, clusters[cluster])}", rule, evidence, certain)

    for cluster, members in clusters.items():
        for rid, key in members:
            role = mapping.roles[rid]
            if rid in remote and role.kind in ("line", "line_diff"):
                assign(cluster, f"ct_line_{bay}_remote", "rele line ujung lawan", f"{rid} ada di gardu lain")
            elif role.kind in ("line", "line_diff") and by_id[rid].voltage_sets:
                assign(cluster, f"ct_line_{bay}", "rele line", f"{rid} adalah rele {role.kind} dengan tegangan line")
    for cluster, members in clusters.items():
        for rid, key in members:
            role = mapping.roles[rid]
            if role.kind == "cbf" and role.breaker and rid not in remote:
                assign(cluster, f"ct_cb_{role.breaker}", "rele CBF", f"{rid} adalah CBF {role.breaker.upper()}")
    for src in sources:
        role = mapping.roles[src.id]
        keys = [k for k in src.current_sets if (src.id, k) in parent]
        if role.kind != "ccp" or len(keys) != 3 or src.id in remote:
            continue
        first = [src.current_sets[k][0] for k in keys]
        residual = kcl_residual(first)
        if residual is not None:
            mapping.note(src.id, "KCL", f"tiga terminal CCP: |ΣI|/max|I| = {residual:.1%} sebelum gangguan"
                         + (" — satu T-node" if residual < 0.1 else " — bukan satu T-node?"), residual < 0.1)
        terminal_clusters = [find((src.id, k)) for k in keys]
        wanted = [f"ct_line_{bay}", f"ct_{bus_side}", "ct_cb_ab"]
        free_roles = [r for r in wanted if r not in role_of.values()]
        free = [c for c in terminal_clusters if c not in role_of]
        if len(free) == 1 and len(free_roles) == 1:
            assign(free[0], free_roles[0], "terminal ketiga CCP", f"dua terminal lain {src.id} sudah diketahui")
        elif free and len(free) == len(free_roles):
            options = list(permutations(free_roles))
            mapping.note(src.id, "ambigu", f"{len(free)} terminal CCP tanpa petunjuk; {len(options)} kemungkinan diuji estimator", False)
            mapping.alternatives = [dict(zip(free, perm)) for perm in options]

    # Current specs (the first alternative is the main mapping).
    def current_specs(extra: dict) -> list[dict[str, Any]]:
        specs = []
        roles = {**role_of, **extra}
        for cluster, members in clusters.items():
            if cluster not in roles:
                continue
            seen = set()
            for rid, key in members:
                if rid in seen:          # one relay reading the same CT twice (sum and CB1 inputs)
                    continue
                seen.add(rid)
                src = by_id[rid]
                chans = src.current_channels[key]
                spec = {"ct": roles[cluster], "recorder": rid, "channels": [chans["A"], chans["B"], chans["C"]]}
                role = mapping.roles[rid]
                if role.kind in ("line", "line_diff") and len(src.voltage_sets) == 1 and roles[cluster].startswith("ct_line"):
                    (volts,) = src.voltage_sets.values()
                    spec.update(voltages=[volts["A"], volts["B"], volts["C"]], forward=f"line_{bay}")
                specs.append(spec)
        return specs

    open_roles = [dict(alt) for alt in mapping.alternatives] or [{}]
    mapping.currents = current_specs(open_roles[0])
    mapping.alternatives = [current_specs(alt) for alt in open_roles[1:]]
    for cluster, members in clusters.items():
        if cluster not in role_of and all(cluster not in alt for alt in open_roles):
            mapping.note(_cluster_text(by_id, members), "tanpa peran", "tidak ada aturan yang menentukan CT ini", False)
    ratings: dict[str, float] = {}
    for cluster, members in clusters.items():
        role = role_of.get(cluster) or open_roles[0].get(cluster)
        for rid, key in members:
            rating = by_id[rid].ratings.get(key)
            if role and rating:
                ratings[role] = max(ratings.get(role, 0.0), rating)
    mapping.ct_ratings_a = ratings

    # Breakers.
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
    for (rid, breaker, convention), names in conventions.items():
        mapping.note(f"{breaker} ← {rid}: {', '.join(names)}", "konvensi", convention, False)
    for (rid, breaker, open_when), names in grouped.items():
        mapping.breakers.append({"breaker": breaker, "recorder": rid, "channels": names, "open_when": open_when})
        mapping.note(f"{breaker} ← {rid}: {', '.join(names[:3])}{' …' if len(names) > 3 else ''}", "status CB",
                     f"kontak {'52b/OPEN (tinggi = open)' if open_when else '52a/CLOSE (tinggi = tertutup)'}")

    # Event-only recorders line up on a breaker an aligned recorder also reports.
    aligned = {rid for rid in offsets}
    reported = {b["breaker"] for b in mapping.breakers if b["recorder"] in aligned}
    for src in sources:
        if src.id in aligned:
            continue
        own = [b["breaker"] for b in mapping.breakers if b["recorder"] == src.id and b["breaker"] in reported]
        if own:
            choice = sorted(own, key=lambda b: (b != bus_side, b != "cb_ab", b))[0]
            mapping.align_breakers[src.id] = choice

    # Zones and distance decisions.
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
                line_sets = [s for s in mapping.currents if s["recorder"] == src.id and s["ct"] == f"ct_line_{bay}"]
                if line_sets:
                    mapping.direction_signals.append({"ct": f"ct_line_{bay}", "recorder": src.id, "channels": [name],
                                                      "value": "forward", "forward": f"line_{bay}"})

    # The diameter around it.
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
            label = re.sub(r"[_/]+", " ", src.folder).strip()
            spec = {"id": zid, "kind": role.kind, "label": label}
            if role.kind == "busbar":
                spec.update(bus=bus, end_zone="unknown")
            else:
                spec["bay"] = bay
            zones.append(spec)
        diameter["zones"] = zones
    mapping.diameter = diameter
    return mapping


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


def _set_text(by_id: dict[str, SourceRecorder], member: tuple[str, str]) -> str:
    rid, key = member
    chans = by_id[rid].current_channels[key]
    return f"{rid}:{chans['A']}/{chans['B']}/{chans['C']}"


def _cluster_text(by_id: dict[str, SourceRecorder], members: list[tuple[str, str]]) -> str:
    return " = ".join(_set_text(by_id, m) for m in members)
