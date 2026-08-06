"""Double Ended Fault Locator (Relay 21, two-terminal Kirchhoff method).

Implements the two-terminal fault-location method documented in PLN's
"Double Ended Fault Locator Aplikasi SiGRA 4.6" training material: given two
independently-uploaded COMTRADE records — one per line terminal (A and B) —
manually time-synchronized by the user, Kirchhoff's voltage law across both
terminals eliminates the fault resistance (Rf) and the zero-sequence
compensation factor (K0) from the single-ended distance calculation entirely.

Single-ended distance (webapp/api/routers/relay_21.py, ``_compute_locus``)
estimates impedance from ONE terminal's V/I and is sensitive to Rf and to
the accuracy of an assumed K0 — the source material's stated motivation for
this feature (high-resistance faults inflate single-ended error). The
double-ended method instead solves:

    V_A - m*Zline*I_A = V_F                  (fault voltage, seen from A)
    V_B - (1-m)*Zline*I_B = V_F               (fault voltage, seen from B)

Both equations describe the SAME unknown fault-point voltage V_F, so
eliminating it (subtracting) never involves Rf (which only ever multiplied
the unknown fault current at V_F, canceling along with V_F) or K0 (never
appears — the phasors used here are plain per-phase/loop V and I, not a
K0-compensated residual current):

    m = (V_A - V_B + Zline*I_B) / (Zline*(I_A + I_B))
    distance_km = Re(m) * line_len_km

This module deliberately duplicates the small per-window phasor/scaling body
of relay_21.py's ``_compute_locus`` (evaluated ONCE at the fault-inception
index instead of swept across the whole record) rather than modifying that
function — the two features have different accuracy requirements (a single
high-quality phasor pair vs. a smoothed trajectory) and single-ended
behavior must not change.

Time alignment: unlike webapp/api/incidents/joined_waveform.py (which joins
records SEQUENTIALLY, e.g. trip followed by reclose), a double-ended
calculation needs both terminals' fault-inception instant on the SAME time
origin, since line-length-scale accuracy requires sub-millisecond precision
that neither GPS-free relay clocks nor the coarse ISO-timestamp gap
(``align-estimate`` below) can reliably provide alone. Per the source
material's own SIGRA workflow (cursor-drag sync, confirmed/adjusted by the
user, never auto-trusted), ``manual_shift_ms`` is the authoritative offset
for ``/compute`` — ``/align-estimate`` only offers a starting point.
"""

import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from ..schemas import (
    DoubleEndedAlignRequest, DoubleEndedAlignResponse,
    DoubleEndedComputeRequest, DoubleEndedComputeResponse,
)
from ..storage import load_analysis
from .relay_21 import (
    LOOP_CHANNELS,
    _canonical_inception_idx,
    _detect_active_line_tag,
    _find_channel,
    _find_phase_current,
    _find_voltage_for_loop,
    _fundamental_phasor,
    _voltage_to_volts_scale,
)

router = APIRouter(prefix="/api/analyze/21de", tags=["relay-21-double-ended"])


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _record_trigger_abs_time(payload: dict) -> Optional[datetime]:
    """Absolute wall-clock time of this record's ``time == 0`` instant.
    core/comtrade_parser.py's ``time`` axis is relative to the COMTRADE
    relay's own trigger point (NOT the first sample — ``time[0]`` is
    typically negative, a pre-trigger buffer), and ``trigger_time_iso`` is
    exactly that instant's wall-clock timestamp. Anchoring here (rather than
    ``start_time_iso``) is what makes it valid to add a record-relative
    inception time (itself measured against ``time == 0``) straight onto
    this anchor below without double-counting the pre-trigger offset."""
    return _parse_iso(payload.get("trigger_time_iso"))


def _load_or_404(analysis_id: str) -> dict:
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Analysis session {analysis_id} not found or expired.")
    return payload


def _swap_bc(loop: str) -> tuple[str, bool]:
    """Phase-sequence inversion (PPTX's "pilih konfigurasi urutan phasa"):
    swap which physical channel is read as B vs C for one terminal, for a
    record whose CT/PT wiring (or CFG phase labeling) has B and C reversed.
    Returns (effective_loop, negate) — ZB/ZC and ZAB/ZCA swap to a genuinely
    different channel pair, so no sign flip is needed; ZBC's channel pair
    (B, C) is unordered under the swap (still "the B-C loop"), so the swap
    instead flips the sign of both Vbc and the B/C current difference (Vcb =
    -Vbc). ZA and ZCA-independent phase-A paths are unaffected."""
    swap = {"ZB": "ZC", "ZC": "ZB", "ZAB": "ZCA", "ZCA": "ZAB"}
    if loop in swap:
        return swap[loop], False
    if loop == "ZBC":
        return "ZBC", True
    return loop, False


def _terminal_phasor(
    payload: dict,
    loop: str,
    invert_i: bool,
    invert_phase_sequence: bool,
    shift_s: float = 0.0,
) -> dict:
    """Fundamental-frequency V/I phasor pair for ONE terminal at its fault
    inception, in PRIMARY volts/amps (stored COMTRADE samples are already
    primary-scaled — core/comtrade_parser.py — so no CT/VT ratio is applied
    here, unlike ``_compute_locus``'s additional relay-secondary-ohm scaling
    for zone-overlay display). This is ``_compute_locus``'s per-window body
    (relay_21.py), evaluated once at the (possibly shifted) inception index
    rather than swept across the whole record — no locus trajectory is
    needed here, only a single high-quality phasor pair.

    ``shift_s`` moves the evaluation window relative to this record's own
    detected inception (used for terminal B, per the caller-supplied
    manual_shift_ms) — it does NOT change which sample is reported as
    "inception" for diagnostics, only which window feeds the phasor calc.
    """
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    if len(time) < 4:
        raise HTTPException(status_code=422, detail="Record too short for double-ended analysis.")

    freq = float(payload.get("frequency", 50.0))
    inception_idx, timing_source, _confidence = _canonical_inception_idx(payload, time)

    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else freq * 20.0
    win = max(1, int(round(sr / freq)))  # one cycle window, same convention as _compute_locus

    shift_samples = int(round(shift_s * sr))
    eval_idx = inception_idx + shift_samples
    # Center the analysis window a few samples into the post-fault region so
    # the fundamental-phasor estimate reflects steady fault current rather
    # than the inception transient itself, while staying inside the record.
    k = min(max(eval_idx + win, win - 1), len(time) - 1)
    s = k - win + 1
    if s < 0:
        raise HTTPException(
            status_code=422,
            detail="Synchronization shift moves the evaluation window before the start of the record.",
        )

    active_tag = _detect_active_line_tag(channels)
    if invert_phase_sequence:
        effective_loop, negate_bc = _swap_bc(loop)
    else:
        effective_loop, negate_bc = loop, False
    mapping = LOOP_CHANNELS.get(effective_loop, LOOP_CHANNELS["ZA"])

    voltage_scale = _voltage_to_volts_scale(channels)

    v = _find_voltage_for_loop(channels, mapping, active_tag)
    if v is None:
        raise HTTPException(status_code=422, detail=f"Could not find voltage channel for loop {loop}")

    i_channels = []
    for candidate in mapping["i"]:
        current = _find_channel(channels, [candidate], active_tag)
        if current is None and candidate.startswith("I") and len(candidate) >= 2:
            current = _find_phase_current(channels, candidate[-1], active_tag)
        i_channels.append(current)
    i_channels = [c for c in i_channels if c is not None]
    if not i_channels:
        raise HTTPException(status_code=422, detail=f"Could not find current channel(s) for loop {loop}")

    if mapping.get("diff") and len(i_channels) == 2:
        i = i_channels[0] - i_channels[1]
    else:
        i = i_channels[0]

    if invert_i:
        i = -i
    if negate_bc:
        v = -v
        i = -i

    v_ph = _fundamental_phasor(v * voltage_scale, s, win, freq, sr, inception_idx)
    i_ph = _fundamental_phasor(i, s, win, freq, sr, inception_idx)

    if not np.isfinite(abs(v_ph)) or not np.isfinite(abs(i_ph)):
        raise HTTPException(
            status_code=422,
            detail="Could not compute a valid V/I phasor at the synchronized fault window "
                   "(record too short, or shift moved the window off the fault).",
        )

    return {
        "v_primary": v_ph,
        "i_primary": i_ph,
        "inception_idx": inception_idx,
        "inception_time_s": float(time[inception_idx]) if inception_idx < len(time) else 0.0,
        "timing_source": timing_source,
        "active_tag": active_tag,
        "eval_sample": k,
    }


def _compute_double_ended(
    payload_a: dict,
    payload_b: dict,
    loop: str,
    line_len_km: float,
    r1_ohm_per_km: float,
    x1_ohm_per_km: float,
    manual_shift_ms: float,
    invert_i_a: bool,
    invert_i_b: bool,
    invert_phase_sequence_a: bool,
    invert_phase_sequence_b: bool,
) -> dict:
    if line_len_km <= 0:
        raise HTTPException(status_code=422, detail="line_len_km must be positive.")

    term_a = _terminal_phasor(payload_a, loop, invert_i_a, invert_phase_sequence_a, shift_s=0.0)
    # Positive manual_shift_ms means "shift record B later by this much" —
    # matching the PPTX's "Shift Fault record B by ..." framing (slide 27) —
    # so B's evaluation window moves back (earlier, into B's own samples) by
    # that amount to land on the same physical instant as A's window.
    term_b = _terminal_phasor(
        payload_b, loop, invert_i_b, invert_phase_sequence_b, shift_s=-manual_shift_ms / 1000.0,
    )

    v_a, i_a = term_a["v_primary"], term_a["i_primary"]
    v_b, i_b = term_b["v_primary"], term_b["i_primary"]

    warnings: list[str] = []
    min_i = 1e-6
    if abs(i_a) < min_i or abs(i_b) < min_i:
        raise HTTPException(
            status_code=422,
            detail="Fault current at one or both terminals is effectively zero — "
                   "check that both records actually see this fault and that the sync offset is correct.",
        )

    z_line = complex(r1_ohm_per_km, x1_ohm_per_km) * line_len_km
    denom = z_line * (i_a + i_b)
    if abs(denom) < 1e-9:
        raise HTTPException(
            status_code=422,
            detail="Degenerate solution (I_A + I_B ~= 0) — the two terminals' currents "
                   "nearly cancel, which usually means a polarity/invert-current mismatch "
                   "between the two records.",
        )

    m_complex = (v_a - v_b + z_line * i_b) / denom
    m = float(np.real(m_complex))
    m_residual_imag = float(np.imag(m_complex))

    if m < 0.0 or m > 1.0:
        warnings.append(
            f"Solved distance falls outside the line (m={m:.3f}) — check synchronization, "
            f"CT/PT ratios, phase sequence, and loop selection."
        )
    if abs(m_residual_imag) > 0.15:
        warnings.append(
            f"Large residual imaginary component (Im(m)={m_residual_imag:.3f}) suggests "
            f"a synchronization or line-parameter error rather than a clean solution."
        )

    fault_current_a = float(abs(i_a + i_b))
    distance_km = m * line_len_km

    return {
        "loop": loop,
        "distance_km": distance_km,
        "distance_pct": (distance_km / line_len_km) * 100.0,
        "fault_current_a": fault_current_a,
        "m_residual_imag": m_residual_imag,
        "inception_time_a_s": term_a["inception_time_s"],
        "inception_time_b_s": term_b["inception_time_s"],
        "active_tag_a": term_a["active_tag"],
        "active_tag_b": term_b["active_tag"],
        "warnings": warnings,
    }


@router.post("/align-estimate", response_model=DoubleEndedAlignResponse)
async def align_estimate(body: DoubleEndedAlignRequest):
    """Coarse starting point for the manual sync step: each record's own
    (record-relative) inception time, plus — only if BOTH records carry an
    absolute wall-clock trigger timestamp — a millisecond-level estimate of
    how far apart those instants are. This is a STARTING POINT for the
    user's cursor-drag confirmation, never treated as ground truth: relay
    clocks are frequently un-synced or GPS-free, which is exactly why the
    source workflow has the user drag cursors and read the delta off the
    waveform rather than trust wall-clock time alone."""
    payload_a = _load_or_404(body.analysis_id_a)
    payload_b = _load_or_404(body.analysis_id_b)

    loop = asyncio.get_event_loop()

    def _inceptions():
        time_a = np.array(payload_a.get("time", []))
        time_b = np.array(payload_b.get("time", []))
        idx_a, src_a, _ = _canonical_inception_idx(payload_a, time_a) if len(time_a) >= 4 else (0, "insufficient_data", 0.0)
        idx_b, src_b, _ = _canonical_inception_idx(payload_b, time_b) if len(time_b) >= 4 else (0, "insufficient_data", 0.0)
        t_a = float(time_a[idx_a]) if idx_a < len(time_a) else None
        t_b = float(time_b[idx_b]) if idx_b < len(time_b) else None
        return t_a, src_a, t_b, src_b

    t_a, src_a, t_b, src_b = await loop.run_in_executor(None, _inceptions)

    abs_a = _record_trigger_abs_time(payload_a)
    abs_b = _record_trigger_abs_time(payload_b)

    if abs_a is not None and abs_b is not None and t_a is not None and t_b is not None:
        # Both records' inception instants expressed on one absolute axis:
        # each side's trigger-instant wall clock + its own trigger-relative
        # inception time, then differenced — targets "B's inception minus
        # A's inception" directly.
        abs_inception_a = abs_a.timestamp() + t_a
        abs_inception_b = abs_b.timestamp() + t_b
        estimated_shift_ms = (abs_inception_b - abs_inception_a) * 1000.0
        estimate_available = True
        reason = "Derived from both records' absolute trigger/start timestamps — confirm against the waveform cursors."
    else:
        estimated_shift_ms = None
        estimate_available = False
        reason = "One or both records lack an absolute wall-clock timestamp — set the shift from the cursors only."

    return DoubleEndedAlignResponse(
        inception_time_a_s=t_a,
        inception_time_b_s=t_b,
        timing_source_a=src_a,
        timing_source_b=src_b,
        estimated_shift_ms=estimated_shift_ms,
        estimate_available=estimate_available,
        estimate_reason=reason,
    )


@router.post("/compute", response_model=DoubleEndedComputeResponse)
async def compute(body: DoubleEndedComputeRequest):
    payload_a = _load_or_404(body.analysis_id_a)
    payload_b = _load_or_404(body.analysis_id_b)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _compute_double_ended(
            payload_a, payload_b, body.loop, body.line_len_km,
            body.r1_ohm_per_km, body.x1_ohm_per_km, body.manual_shift_ms,
            body.invert_i_a, body.invert_i_b,
            body.invert_phase_sequence_a, body.invert_phase_sequence_b,
        ),
    )
    return DoubleEndedComputeResponse(**result)
