"""Relay 87L (Differential Line) - diff/restraint plot + AI analysis."""

import asyncio
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    AIFaultResult,
    DiffRestraintAnalysisRequest,
    DiffRestraintResponse,
    DiffRestraintSample,
    TripMarker,
    PhaseClassification,
)
from ..storage import load_analysis
from ..ml_predict import run_ml_prediction
from ..fault_detection import detect_fault_presence

router = APIRouter(prefix="/api/analyze/87l", tags=["relay-87l"])

PHASE_CURRENT_MAP = {
    "L1": ["IA", "IL1", "I1"],
    "L2": ["IB", "IL2", "I2"],
    "L3": ["IC", "IL3", "I3"],
}

# Relay-computed differential channels (already a true two-terminal quantity).
# If present, we have real diff data; otherwise the record is local-terminal only.
RELAY_DIFF_CHANNELS = {
    "L1": ["87L.IDA", "IDIFFA", "IDIFF_A", "IDIF_A", "IDA", "IDL1", "IL1D", "IDIFFL1"],
    "L2": ["87L.IDB", "IDIFFB", "IDIFF_B", "IDIF_B", "IDB", "IDL2", "IL2D", "IDIFFL2"],
    "L3": ["87L.IDC", "IDIFFC", "IDIFF_C", "IDIF_C", "IDC", "IDL3", "IL3D", "IDIFFL3"],
}

RELAY_RESTRAINT_CHANNELS = {
    "L1": ["87L.IRSTA", "IRESTA", "IREST_A", "IRESTR_A", "IRSTR_A", "IBIASA", "IBIAS_A", "IBIASL1", "BIASA"],
    "L2": ["87L.IRSTB", "IRESTB", "IREST_B", "IRESTR_B", "IRSTR_B", "IBIASB", "IBIAS_B", "IBIASL2", "BIASB"],
    "L3": ["87L.IRSTC", "IRESTC", "IREST_C", "IRESTR_C", "IRSTR_C", "IBIASC", "IBIAS_C", "IBIASL3", "BIASC"],
}


def _find_ch(channels, candidates):
    cand = _candidate_set(candidates)
    for ch in channels:
        canon = str(ch.get("canonical_name", "") or "").upper()
        name = str(ch.get("name", "") or "").upper()
        if canon in cand or name in cand:
            return np.asarray(ch.get("samples") or [], dtype=float)
    return None


def _find_ch_obj(channels, candidates):
    cand = _candidate_set(candidates)
    for ch in channels:
        canon = str(ch.get("canonical_name", "") or "").upper()
        name = str(ch.get("name", "") or "").upper()
        if canon in cand or name in cand:
            return ch
    return None


# Remote-terminal current candidates per phase. Differential ALWAYS needs two
# sides on the same relay: line diff = local + remote, transformer diff = HV + LV.
# A single-side record yields an absurd differential, so we look for the second
# terminal explicitly. Common conventions: suffixed (Ia1, IA2, IA_REM, IAR) or
# Siemens side tags (-S2). The local set is PHASE_CURRENT_MAP above.
REMOTE_CURRENT_MAP = {
    "L1": ["IA1", "IA2", "IA_REM", "IAR", "IL1-S2", "IA-S2", "I1R", "REM L IL1 D", "REMOTE IL1"],
    "L2": ["IB1", "IB2", "IB_REM", "IBR", "IL2-S2", "IB-S2", "I2R", "REM L IL2 D", "REMOTE IL2"],
    "L3": ["IC1", "IC2", "IC_REM", "ICR", "IL3-S2", "IC-S2", "I3R", "REM L IL3 D", "REMOTE IL3"],
}


def _candidate_set(candidates) -> set[str]:
    return {str(c).upper() for c in candidates}


def _rms_np(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


_REMOTE_PHASE_TOKENS = {
    "L1": ("IL1", "IA", "I1", "L1"),
    "L2": ("IL2", "IB", "I2", "L2"),
    "L3": ("IL3", "IC", "I3", "L3"),
}


def _is_remote_current_channel(ch: dict, phase: str) -> bool:
    """ABB/REL670 remote current names often look like 'REM L IL1 D'.

    Some uploaded records normalize those channels to IDIFF_A/B/C, so name
    context must win over canonical_name: REM ILx is remote terminal current,
    not relay-computed differential.
    """
    name = str(ch.get("name", "") or "")
    compact = _compact(name)
    if "REM" not in compact and "REMOTE" not in compact:
        return False
    return any(tok in compact for tok in _REMOTE_PHASE_TOKENS[phase])


def _find_remote_ch(channels, phase: str):
    exact = _find_ch(channels, REMOTE_CURRENT_MAP[phase])
    if exact is not None:
        return exact
    for ch in channels:
        if _is_remote_current_channel(ch, phase):
            return np.asarray(ch.get("samples") or [], dtype=float)
    return None


def _find_relay_diff_ch(channels, phase: str):
    candidates = _candidate_set(RELAY_DIFF_CHANNELS[phase])
    compact_candidates = {_compact(c) for c in candidates}
    for ch in channels:
        if _is_remote_current_channel(ch, phase):
            continue
        canon = str(ch.get("canonical_name", "") or "").upper()
        name = str(ch.get("name", "") or "").upper()
        compact_name = _compact(f"{name} {canon}")
        if canon in candidates or name in candidates or any(c in compact_name for c in compact_candidates):
            return np.asarray(ch.get("samples") or [], dtype=float)
    return None


def _find_relay_restraint_ch(channels, phase: str):
    candidates = _candidate_set(RELAY_RESTRAINT_CHANNELS[phase])
    compact_candidates = {_compact(c) for c in candidates}
    for ch in channels:
        canon = str(ch.get("canonical_name", "") or "").upper()
        name = str(ch.get("name", "") or "").upper()
        compact_name = _compact(f"{name} {canon}")
        if canon in candidates or name in candidates or any(c in compact_name for c in compact_candidates):
            return np.asarray(ch.get("samples") or [], dtype=float)
    return None


def _detect_terminal_pairs(channels, time, frequency):
    """Find a local/remote current pair per phase for a TRUE two-terminal diff.

    Physics-first: in a prefault window, healthy phases of the same line carry
    pure through-current, so I_local + s*I_remote ≈ 0 for the correct polarity s.
    We confirm a pair by finding a sign s in {+1,-1} that makes the prefault
    differential small relative to the through-current. Name patterns are only a
    fallback for locating the candidate remote channel.

    Returns: (pairs, sign) where pairs = {phase: (local_arr, remote_arr)} and
    sign is the shared polarity convention, or (None, +1) if no valid pair.
    """
    t = np.asarray(time, dtype=float)
    n = t.size
    if n < 8:
        return None, 1
    pf = max(4, n // 5)  # prefault window

    pairs = {}
    for ph in ("L1", "L2", "L3"):
        local = _find_ch(channels, PHASE_CURRENT_MAP[ph])
        remote = _find_remote_ch(channels, ph)
        if local is None or remote is None:
            continue
        pairs[ph] = (np.asarray(local, dtype=float), np.asarray(remote, dtype=float))

    if len(pairs) < 2:
        return None, 1  # need at least two phases to validate physically

    # Determine polarity from the healthiest phase (smallest through-differential
    # for its best sign) so a faulted phase doesn't pick the wrong convention.
    best = None  # (residual_ratio, sign)
    for ph, (loc, rem) in pairs.items():
        lo, re = loc[:pf], rem[:pf]
        through = _rms_np(lo) + _rms_np(re)
        if through <= 0:
            continue
        for s in (1.0, -1.0):
            resid = _rms_np(lo + s * re) / through
            if best is None or resid < best[0]:
                best = (resid, s)

    if best is None:
        return None, 1
    resid_ratio, sign = best
    # A genuine two-terminal pair: at least one phase balances to a small residual
    # (healthy through-current cancels). Loose threshold tolerates CT mismatch.
    if resid_ratio > 0.35:
        return None, 1
    return pairs, sign


def _detect_diff_mode(channels, time=None, frequency=50.0) -> str:
    """Decide how the differential will be computed:
    - TWO_TERMINAL: relay recorded its own computed differential channel.
    - TWO_TERMINAL_RAW: both terminal currents present -> we compute true diff.
    - LOCAL_ONLY: only one side present (rare/chopped) -> diff cannot be trusted.
    """
    phases = ("L1", "L2", "L3")
    has_relay_diff = any(_find_relay_diff_ch(channels, ph) is not None for ph in phases)
    has_full_relay_restraint = all(_find_relay_restraint_ch(channels, ph) is not None for ph in phases)
    pairs = None
    if time is not None:
        pairs, _ = _detect_terminal_pairs(channels, time, frequency)
        # Raw terminal pairs are preferred when relay-only data is incomplete.
        # A relay-computed Idiff without matching Irest/IBias can float above the
        # characteristic just because it is in amperes while the plot expects pu.
        if pairs and not has_full_relay_restraint:
            return "TWO_TERMINAL_RAW"
    if has_relay_diff:
        return "TWO_TERMINAL"
    if pairs:
        return "TWO_TERMINAL_RAW"
    return "LOCAL_ONLY"


# Per-phase differential-trip digital channels reported by the relay itself.
# These are the authoritative verdict when waveform reconstruction is impossible.
_RELAY_DIFF_TRIP = {
    "L1": ["DIF-A", "DIFF-A", "DIFA", "87-A", "DIFF_A", "IDIFF-A"],
    "L2": ["DIF-B", "DIFF-B", "DIFB", "87-B", "DIFF_B", "IDIFF-B"],
    "L3": ["DIF-C", "DIFF-C", "DIFC", "87-C", "DIFF_C", "IDIFF-C"],
}


def _relay_diff_phases(comtrade_data: dict) -> list:
    """Which phases the relay's own 87L diff element reports as operated (from status channels)."""
    out = []
    for ch in comtrade_data.get("status_channels", []):
        name_up = str(ch.get("name", "")).upper()
        samp = np.asarray(ch.get("samples", []), dtype=float)
        if samp.size == 0 or not np.any(samp > 0.5):
            continue
        for ph, pats in _RELAY_DIFF_TRIP.items():
            if any(p in name_up for p in pats) and ph not in out:
                out.append(ph)
    return sorted(out)


def _compute_diff_restraint(comtrade_data: dict, params: dict) -> dict:
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])
    freq = comtrade_data.get("frequency", 50.0)
    samples = []
    operated_phases = []

    idiff_pickup = params["idiff_pickup"]
    idiff_fast = params["idiff_fast"]
    slope1 = params["slope1"]
    intersection1 = params["intersection1"]
    slope2 = params["slope2"]
    intersection2 = params["intersection2"]

    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
    win = max(1, int(sr / freq))
    n = len(time)
    pf = max(4, n // 5)

    mode = _detect_diff_mode(channels, time, freq)
    pairs, sign = _detect_terminal_pairs(channels, time, freq) if mode == "TWO_TERMINAL_RAW" else (None, 1.0)

    if mode == "TWO_TERMINAL":
        in_user = float(params.get("in_base_a", 0.0) or 0.0)
        in_base = in_user if in_user > 0 else 1.0
        for phase in ("L1", "L2", "L3"):
            diff = _find_relay_diff_ch(channels, phase)
            if diff is None:
                continue
            rest = _find_relay_restraint_ch(channels, phase)
            for k in range(0, n, max(1, win // 4)):
                s = max(0, k - win + 1)
                i_diff = _rms_np(np.asarray(diff[s:k + 1], dtype=float)) / in_base
                i_rest = (
                    _rms_np(np.asarray(rest[s:k + 1], dtype=float)) / in_base
                    if rest is not None else 0.0
                )
                samples.append({"t": float(time[k]), "i_diff": i_diff, "i_rest": i_rest, "phase": phase})
    elif pairs:
        # TRUE two-terminal differential. Idiff = |Ilocal + s*Iremote|,
        # Irest = (|Ilocal| + |Iremote|) / 2, normalised to base current In so the
        # p.u. dual-slope characteristic applies. In comes from the user (CT/relay
        # setting) when provided, else auto-estimated from the healthiest phase's
        # prefault through-current — a SHARED base for all phases so a distorted
        # phase doesn't get its own (too-small) In and float above the slope.
        in_user = float(params.get("in_base_a", 0.0) or 0.0)
        if in_user > 0:
            in_base = in_user
        else:
            healthiest = None  # (prefault residual, through-current)
            for loc, rem in pairs.values():
                through = _rms_np(loc[:pf]) + _rms_np(rem[:pf])
                resid = _rms_np(loc[:pf] + sign * rem[:pf])
                if through > 0 and (healthiest is None or resid / through < healthiest[0]):
                    healthiest = (resid / through, through / 2.0)
            in_base = max(healthiest[1] if healthiest else 1.0, 1.0)

        for phase, (loc, rem) in pairs.items():
            for k in range(0, n, max(1, win // 4)):
                s = max(0, k - win + 1)
                lo = loc[s:k + 1]
                re = rem[s:k + 1]
                i_diff = _rms_np(lo + sign * re) / in_base
                i_rest = (_rms_np(lo) + _rms_np(re)) / 2.0 / in_base
                samples.append({"t": float(time[k]), "i_diff": i_diff, "i_rest": i_rest, "phase": phase})
    else:
        # Single side only — NOT a real differential. Plot local current so the
        # waveform is at least visible; the frontend flags this as untrustworthy
        # and defers the verdict to the relay's own diff trip signals.
        for phase, candidates in PHASE_CURRENT_MAP.items():
            i = _find_ch(channels, candidates)
            if i is None:
                continue
            for k in range(0, n, max(1, win // 4)):
                s = max(0, k - win + 1)
                i_rms = _rms_np(i[s:k + 1])
                samples.append({"t": float(time[k]), "i_diff": abs(i_rms), "i_rest": abs(i_rms), "phase": phase})

    trip_markers = _detect_trip_markers(comtrade_data, samples)
    relay_diff_phases = _relay_diff_phases(comtrade_data)
    relay_diff_asserted = bool(
        relay_diff_phases or any(m["kind"] in ("DIFF", "DIFF_FAST") for m in trip_markers)
    )
    relay_fast_asserted = any(m["kind"] == "DIFF_FAST" for m in trip_markers)

    eval_window = _differential_evaluation_window(trip_markers, freq)
    eval_samples = _samples_in_window(samples, eval_window) if eval_window else samples

    computed_operated_phases = []
    for phase in ("L1", "L2", "L3"):
        phase_samples = [s for s in eval_samples if s["phase"] == phase]
        if not phase_samples:
            continue
        if any(
            s["i_diff"] >= _characteristic_threshold(
                s["i_rest"], idiff_pickup, slope1, intersection1, slope2, intersection2
            )
            for s in phase_samples
        ):
            computed_operated_phases.append(phase)

    if relay_diff_phases:
        operated_phases = relay_diff_phases
    else:
        operated_phases = computed_operated_phases

    if relay_fast_asserted:
        status = "IDIFF_FAST_OPERATED"
    elif relay_diff_asserted:
        status = "IDIFF_OPERATED"
    elif any(s["i_diff"] >= idiff_fast for s in eval_samples):
        status = "IDIFF_FAST_OPERATED"
    elif operated_phases:
        status = "IDIFF_OPERATED"
    else:
        status = "NOT_OPERATED"

    return {
        "samples": samples,
        "operated_status": status,
        "operated_phases": operated_phases,
        "trip_markers": trip_markers,
        "phase_classification": _classify_phases(
            samples,
            params,
            authoritative_operated_phases=relay_diff_phases if relay_diff_phases else None,
            evaluation_window=eval_window,
        ),
        "diff_data_mode": mode,
        "relay_diff_phases": relay_diff_phases,
    }


def _characteristic_threshold(i_rest, pickup, slope1, int1, slope2, int2):
    """Return the I-DIFF operate threshold for a given I_rest value."""
    if i_rest <= int1:
        return pickup
    if i_rest <= int2:
        return pickup + slope1 * (i_rest - int1)
    return pickup + slope1 * (int2 - int1) + slope2 * (i_rest - int2)


# --- Trip detection + per-phase classification (shared by 87L and 87T) -----

# Digital channel name patterns that indicate a trip assertion, mapped to a kind.
# Checked in order; first match wins. Phase suffix (A/B/C/L1..) parsed separately.
_TRIP_PATTERNS = [
    ("DIFF_FAST", ["DIFF>>", "IDIFF>>", "87-2", "I-DIFF>>", "DIFF FAST", "UNRESTRAINED"]),
    # "DIF" (single-F) and "DIFG" cover many DFR vendors (e.g. DIF-A_TRIP, DIFG_TRIP).
    ("DIFF",      ["DIFF>", "IDIFF>", "87-1", "I-DIFF>", "87T TRIP", "87L TRIP",
                   "DIFF PICKUP", "DIFF TRIP", "87 TRIP", "DIF-", "DIF_", "DIFG", "DIF "]),
    ("RELAY_TRIP", ["TRIP", "GEN TRIP", "RELAY TRIP", "PMT", "GENERAL TRIP"]),
]


def _phase_of_name(name_upper: str) -> "str | None":
    """Extract a phase label (L1/L2/L3) from a digital channel name, if present.

    Only matches explicit phase markers (PH A, L1, _R, "PHASE B"), not an
    incidental trailing letter — e.g. "PMT BUKA" must NOT be read as phase A.
    """
    import re
    # Trailing boundary: token must not be followed by another alphanumeric,
    # but "_" / "-" / end-of-string are valid separators (so "DIF-A_TRIP" -> A).
    end = r"(?![A-Z0-9])"
    for pnum, letters in (("L1", ("A", "1", "R")), ("L2", ("B", "2", "S")), ("L3", ("C", "3", "T"))):
        for tok in letters:
            if re.search(rf"(?:PHASE|PH|L|_|-)\s*{tok}{end}", name_upper):
                return pnum
            if re.search(rf"\b{tok}{end}", name_upper) and tok not in ("R", "S", "T"):
                # bare A/B/C/1/2/3 token (e.g. "DIFF> A"); skip R/S/T to avoid
                # false hits on common words. Phase letters R/S/T require a prefix.
                return pnum
    return None


def _detect_trip_markers(comtrade_data: dict, samples: list) -> list:
    """Find first assertion of each trip-type digital channel and map to a sample point."""
    status_channels = comtrade_data.get("status_channels", [])
    time = comtrade_data.get("time", [])
    if not status_channels or not time or not samples:
        return []
    import numpy as _np
    time_arr = _np.asarray(time, dtype=float)
    markers = []
    seen_kinds = set()

    for ch in status_channels:
        name_up = str(ch.get("name", "")).upper()
        kind = None
        for k, pats in _TRIP_PATTERNS:
            if any(p in name_up for p in pats):
                kind = k
                break
        if kind is None:
            continue
        samp = _np.asarray(ch.get("samples", []), dtype=float)
        if samp.size == 0:
            continue
        asserted = _np.nonzero(samp > 0.5)[0]
        if asserted.size == 0:
            continue
        idx = int(asserted[0])
        if idx >= time_arr.size:
            continue
        t_trip = float(time_arr[idx])
        phase = _phase_of_name(name_up)

        # Map onto the nearest operating-point sample (optionally phase-matched).
        candidates = [s for s in samples if phase is None or s["phase"] == phase] or samples
        nearest = min(candidates, key=lambda s: abs(s["t"] - t_trip))

        dedup_key = (kind, phase)
        if dedup_key in seen_kinds:
            continue
        seen_kinds.add(dedup_key)
        markers.append({
            "kind": kind,
            "channel_name": str(ch.get("name", "")),
            "t": t_trip,
            "phase": phase,
            "i_diff": float(nearest["i_diff"]),
            "i_rest": float(nearest["i_rest"]),
        })
    return markers


def _differential_evaluation_window(trip_markers: list, frequency: float) -> "tuple[float, float] | None":
    """Limit operate verdicts to the protection event, not later breaker/reclose transients."""
    event_times = [
        float(m["t"])
        for m in trip_markers
        if m.get("kind") in ("DIFF", "DIFF_FAST")
    ]
    if not event_times:
        event_times = [
            float(m["t"])
            for m in trip_markers
            if m.get("kind") == "RELAY_TRIP"
        ]
    if not event_times:
        return None

    cycle = 1.0 / max(float(frequency or 50.0), 1.0)
    t0 = min(event_times)
    return t0 - cycle, t0 + (2.0 * cycle)


def _samples_in_window(samples: list, window: "tuple[float, float] | None") -> list:
    if window is None:
        return samples
    start, end = window
    scoped = [s for s in samples if start <= float(s.get("t", 0.0)) <= end]
    return scoped or samples


def _classify_phases(
    samples: list,
    params: dict,
    authoritative_operated_phases: "list[str] | None" = None,
    evaluation_window: "tuple[float, float] | None" = None,
) -> list:
    """Per-phase verdict + max operating stats, evidence-based on the trajectory."""
    pickup = params["idiff_pickup"]
    fast = params["idiff_fast"]
    s1, i1, s2, i2 = params["slope1"], params["intersection1"], params["slope2"], params["intersection2"]
    scoped_samples = _samples_in_window(samples, evaluation_window)
    authoritative_set = set(authoritative_operated_phases) if authoritative_operated_phases is not None else None
    out = []
    for phase in ["L1", "L2", "L3"]:
        pts = [s for s in scoped_samples if s["phase"] == phase]
        if not pts:
            continue
        max_idiff = max(s["i_diff"] for s in pts)
        max_irest = max(s["i_rest"] for s in pts)
        # Ratio of operating point to its threshold along the trajectory.
        max_ratio = 0.0
        for s in pts:
            thr = _characteristic_threshold(s["i_rest"], pickup, s1, i1, s2, i2)
            if thr > 0:
                max_ratio = max(max_ratio, s["i_diff"] / thr)

        if authoritative_set is not None:
            verdict = "Internal Fault" if phase in authoritative_set else "Not Operated"
            conf = "high"
        elif max_idiff >= fast:
            verdict, conf = "Internal Fault", "high"
        elif max_ratio >= 1.0:
            # Inside operate region. High restraint with diff just above slope often = through-fault leakage.
            verdict = "Internal Fault" if max_ratio >= 1.3 else "Through Fault"
            conf = "high" if max_ratio >= 1.5 else "medium"
        else:
            verdict, conf = "Not Operated", "high" if max_idiff < pickup else "medium"

        out.append({
            "phase": phase,
            "verdict": verdict,
            "confidence": conf,
            "max_idiff": round(max_idiff, 3),
            "max_irest": round(max_irest, 3),
            "max_ratio": round(max_ratio, 3),
        })
    return out


def _load_analysis_or_404(analysis_id: str) -> dict:
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    return payload


@router.post("/diff-restraint", response_model=DiffRestraintResponse)
async def diff_restraint(body: DiffRestraintAnalysisRequest):
    payload = _load_analysis_or_404(body.analysis_id)
    det = detect_fault_presence(payload)
    if det.no_fault:
        return DiffRestraintResponse(
            samples=[],
            params=body.params,
            operated_status="NOT_OPERATED",
            operated_phases=[],
            trip_markers=[],
            phase_classification=[],
            diff_data_mode="NO_FAULT",
            relay_diff_phases=[],
            no_fault=True,
            no_fault_reasons=det.reasons,
        )
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(_compute_diff_restraint, payload, body.params.model_dump()),
    )
    return DiffRestraintResponse(
        samples=[DiffRestraintSample(**sample) for sample in result["samples"]],
        params=body.params,
        operated_status=result["operated_status"],
        operated_phases=result["operated_phases"],
        trip_markers=[TripMarker(**m) for m in result.get("trip_markers", [])],
        phase_classification=[PhaseClassification(**c) for c in result.get("phase_classification", [])],
        diff_data_mode=result.get("diff_data_mode", "TWO_TERMINAL"),
        relay_diff_phases=result.get("relay_diff_phases", []),
        no_fault=False,
        no_fault_reasons=[],
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis_87l(body: DiffRestraintAnalysisRequest):
    """Run LightGBM fault cause analysis for relay 87L (differential line)."""
    payload = _load_analysis_or_404(body.analysis_id)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ml_prediction, payload, "87L")
    return AIFaultResult(**result)
