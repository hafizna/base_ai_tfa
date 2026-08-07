"""Double Ended Fault Locator (Relay 21, two-terminal Kirchhoff method).

Implements the two-terminal fault-location method documented in PLN's
"Double Ended Fault Locator Aplikasi SiGRA 4.6" training material: given two
independently-uploaded COMTRADE records — one per line terminal (A and B) —
time-synchronized on a common reference, Kirchhoff's voltage law across both
terminals eliminates fault resistance (Rf). Ground loops are solved with
negative-sequence quantities (Z2 approximated by the configured Z1), rather
than incorrectly applying scalar Z1 to an uncompensated phase current.

Single-ended distance (webapp/api/routers/relay_21.py, ``_compute_locus``)
estimates impedance from ONE terminal's V/I and is sensitive to Rf and to
the accuracy of an assumed K0 — the source material's stated motivation for
this feature (high-resistance faults inflate single-ended error). The
double-ended method instead solves:

    V_A - m*Zline*I_A = V_F                  (fault voltage, seen from A)
    V_B - (1-m)*Zline*I_B = V_F               (fault voltage, seen from B)

Both equations describe the SAME unknown fault-point voltage V_F, so
eliminating it (subtracting) never involves Rf, which only multiplies the
unknown fault current at V_F and cancels with the common fault-point voltage.
For phase-to-phase loops, V/I loop quantities are used directly; for ground
faults, V2/I2 are used so the scalar line impedance is physically meaningful:

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
    DoubleEndedSingleEndedResult,
    DoubleEndedSuggestShiftRequest, DoubleEndedSuggestShiftResponse,
)
from ..storage import load_analysis
from core.event_analysis import build_event_window
from .relay_21 import (
    LOOP_CHANNELS,
    _canonical_inception_idx,
    _detect_active_line_tag,
    _find_channel,
    _find_phase_current,
    _find_phase_voltage,
    _find_voltage_for_loop,
    _fundamental_phasor,
    _voltage_to_volts_scale,
)

router = APIRouter(prefix="/api/analyze/21de", tags=["relay-21-double-ended"])

GROUND_LOOPS = {"ZA", "ZB", "ZC"}


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


def _build_terminal_context(
    payload: dict,
    loop: str,
    invert_i: bool,
    invert_phase_sequence: bool,
    *,
    sequence_for_ground: bool = False,
) -> dict:
    """The parts of ``_terminal_phasor`` that do NOT depend on ``shift_s``:
    channel resolution (which voltage/current channel this loop maps to)
    and fault-inception detection. Factored out so a shift-search sweep
    (``_find_optimal_shift``) can compute this ONCE per terminal and reuse
    it across hundreds of candidate shifts, rather than re-running
    ``_canonical_inception_idx``'s full fault-detection pass (measured at
    ~65ms/call — the dominant cost) on every single candidate. A sweep of
    ~1000 shift values previously took minutes; with this context reused,
    the same sweep completes in a couple of seconds."""
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    if len(time) < 4:
        raise HTTPException(status_code=422, detail="Record too short for double-ended analysis.")

    freq = float(payload.get("frequency", 50.0))
    event_window = build_event_window(payload)
    inception_idx = event_window.inception_idx if event_window.inception_idx is not None else 0
    timing_source = event_window.method
    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else freq * 20.0
    win = max(1, int(round(sr / freq)))  # one cycle window, same convention as _compute_locus

    active_tag = _detect_active_line_tag(channels)

    # A phase-to-ground voltage drop cannot in general be represented by
    # Z1 * Iphase: the zero-sequence path has a different impedance.  For the
    # two-ended solve, use negative-sequence quantities instead.  The
    # negative-sequence network is independent of fault resistance and uses
    # Z2 ~= Z1, avoiding an unsafe implicit assumption that Z0 == Z1.
    if sequence_for_ground and loop in GROUND_LOOPS:
        voltages = [_find_phase_voltage(channels, phase, active_tag) for phase in "ABC"]
        currents = [_find_phase_current(channels, phase, active_tag) for phase in "ABC"]
        if any(value is None for value in voltages) or any(value is None for value in currents):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Ground-fault double-ended location requires all three phase voltages and currents "
                    "so negative-sequence quantities can be calculated safely."
                ),
            )
        if invert_phase_sequence:
            voltages[1], voltages[2] = voltages[2], voltages[1]
            currents[1], currents[2] = currents[2], currents[1]
        if invert_i:
            currents = [-value for value in currents]
        voltage_scale = _voltage_to_volts_scale(channels)
        return {
            "time": time,
            "freq": freq,
            "sr": sr,
            "win": win,
            "inception_idx": inception_idx,
            "clearing_idx": event_window.clearing_idx,
            "timing_source": timing_source,
            "active_tag": active_tag,
            "basis": "negative_sequence",
            "v_phases_scaled": [value * voltage_scale for value in voltages],
            "i_phases": currents,
        }

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

    return {
        "time": time,
        "freq": freq,
        "sr": sr,
        "win": win,
        "inception_idx": inception_idx,
        "clearing_idx": event_window.clearing_idx,
        "timing_source": timing_source,
        "active_tag": active_tag,
        "basis": "phase_loop",
        "v_scaled": v * voltage_scale,
        "i": i,
    }


def _phasor_at_window_start(ctx: dict, start: int, aligned_start_s: Optional[float] = None) -> dict:
    """Extract one full-cycle V/I pair beginning at ``start``.

    ``_fundamental_phasor`` reports angle relative to the window start.  When
    ``aligned_start_s`` is supplied, rotate both phasors to a shared t=0
    reference so phasors from independently-triggered records can be used in
    one Kirchhoff equation without a hidden window-angle error.
    """
    win = ctx["win"]
    inception_idx = ctx["inception_idx"]
    if start < 0 or start + win > len(ctx["time"]):
        raise HTTPException(status_code=422, detail="Synchronization shift moves the phasor window off the record.")
    if start + win - 1 < inception_idx:
        raise HTTPException(
            status_code=422,
            detail="Synchronization shift moves the phasor window entirely before fault inception.",
        )

    if ctx.get("basis") == "negative_sequence":
        a = np.exp(1j * 2.0 * np.pi / 3.0)
        vabc = [
            _fundamental_phasor(value, start, win, ctx["freq"], ctx["sr"], inception_idx)
            for value in ctx["v_phases_scaled"]
        ]
        iabc = [
            _fundamental_phasor(value, start, win, ctx["freq"], ctx["sr"], inception_idx)
            for value in ctx["i_phases"]
        ]
        v_ph = (vabc[0] + (a ** 2) * vabc[1] + a * vabc[2]) / 3.0
        i_ph = (iabc[0] + (a ** 2) * iabc[1] + a * iabc[2]) / 3.0
    else:
        v_ph = _fundamental_phasor(ctx["v_scaled"], start, win, ctx["freq"], ctx["sr"], inception_idx)
        i_ph = _fundamental_phasor(ctx["i"], start, win, ctx["freq"], ctx["sr"], inception_idx)

    if aligned_start_s is not None:
        rotation = np.exp(-1j * 2.0 * np.pi * ctx["freq"] * aligned_start_s)
        v_ph *= rotation
        i_ph *= rotation

    if not np.isfinite(abs(v_ph)) or not np.isfinite(abs(i_ph)):
        raise HTTPException(status_code=422, detail="Could not compute a valid synchronized V/I phasor.")

    return {
        "v_primary": v_ph,
        "i_primary": i_ph,
        "inception_idx": inception_idx,
        "clearing_idx": ctx.get("clearing_idx"),
        "inception_time_s": float(ctx["time"][inception_idx]),
        "timing_source": ctx["timing_source"],
        "active_tag": ctx["active_tag"],
        "eval_sample": start + win - 1,
        "window_start_sample": start,
        "basis": ctx.get("basis", "phase_loop"),
    }


def _terminal_phasor_from_context(ctx: dict, shift_s: float) -> dict:
    """The shift_s-dependent part of ``_terminal_phasor``: pick the
    evaluation window and extract the fundamental phasor pair from an
    already-built ``_build_terminal_context`` result. Cheap enough to call
    hundreds of times per second — no channel resolution or inception
    detection happens here."""
    time = ctx["time"]
    sr = ctx["sr"]
    win = ctx["win"]
    inception_idx = ctx["inception_idx"]

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
    # A large negative shift_s can push the whole [s, k] window before this
    # record's OWN detected inception without ever going negative (it's
    # still a valid slice into the record — just the wrong part of it):
    # verified against a real record where shift values the user was
    # actively trying (100-230ms) silently landed the window entirely in
    # the pre-fault/load-current region, producing a plausible-looking but
    # physically meaningless phasor (steady load current, not fault
    # current) with no error at all — the caller had no way to tell this
    # apart from a genuine fault-window reading, and no amount of further
    # shift adjustment could ever converge because the "signal" being
    # chased was pre-fault noise, not the fault. Guard explicitly instead
    # of letting this pass silently.
    if k < inception_idx:
        raise HTTPException(
            status_code=422,
            detail="Synchronization shift moves the evaluation window entirely before this record's own "
                   "detected fault inception — the phasor would describe pre-fault load current, not the "
                   "fault. Reduce the magnitude of the sync shift.",
        )

    return _phasor_at_window_start(ctx, s)


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

    Single-call convenience wrapper around ``_build_terminal_context`` +
    ``_terminal_phasor_from_context`` — callers that need many shifts for
    the SAME terminal (the shift-search sweep) should call those two
    directly instead, building the context once."""
    ctx = _build_terminal_context(payload, loop, invert_i, invert_phase_sequence)
    return _terminal_phasor_from_context(ctx, shift_s)


def _solve_m(
    v_a: complex, i_a: complex, v_b: complex, i_b: complex, z_line: complex, min_i: float = 1e-6,
) -> Optional[tuple[float, float]]:
    """The Kirchhoff closed-form solve itself, factored out of
    ``_compute_double_ended`` so both the single authoritative result AND
    the distance-histogram sweep below call the exact same algebra rather
    than maintaining two copies of it. Given one fixed V/I phasor pair per
    terminal, this has exactly ONE solution — it is NOT a sweep-able
    function on its own; the sweep happens by calling this repeatedly with
    phasors from different evaluation windows (see
    ``_compute_distance_histogram``).

    ``min_i`` is an ABSOLUTE amps floor by default (fine for
    ``_compute_double_ended``, which only ever evaluates the one window at
    real fault inception — current there is never marginal). The histogram
    sweep passes a floor scaled to that record's own peak fault current
    instead: a real record's current genuinely approaches zero near the
    fault's clearing edge (the fault is actually extinguishing), and 1e-6 A
    is not a real floor at line-current scale (hundreds to thousands of
    amps) — verified against a real dual-terminal record where windows near
    clearing produced distance readings off by hundreds of km from the
    fault-duration cluster, caused by dividing by a near-zero-but-not-quite
    -zero current rather than a genuine second solution.

    Returns (m, residual_imag) or None for a degenerate window (near-zero
    combined current — I_A + I_B ~= 0, usually a polarity mismatch or a
    window with no real fault current flowing)."""
    if abs(i_a) < min_i or abs(i_b) < min_i:
        return None
    denom = z_line * (i_a + i_b)
    if abs(denom) < 1e-9:
        return None
    m_complex = (v_a - v_b + z_line * i_b) / denom
    return float(np.real(m_complex)), float(np.imag(m_complex))


def _aligned_terminal_pair(ctx_a: dict, ctx_b: dict, shift_ms: float, offset_s: float = 0.0) -> tuple[dict, dict]:
    """Return A/B phasors from the same aligned physical time.

    The UI convention is ``t_B_aligned = t_B - shift``.  Therefore a window
    beginning at ``t_A`` must be read from B at ``t_A + shift``.  Both DFT
    results are then rotated from their window-local angle reference to the
    common aligned time axis.
    """
    start_a = ctx_a["inception_idx"] + 1 + int(round(offset_s * ctx_a["sr"]))
    if start_a + ctx_a["win"] > len(ctx_a["time"]):
        raise HTTPException(status_code=422, detail="Terminal A has no complete phasor window at this fault offset.")

    common_start_s = float(ctx_a["time"][start_a])
    shift_s = shift_ms / 1000.0
    target_b_s = common_start_s + shift_s
    start_b = int(np.searchsorted(ctx_b["time"], target_b_s, side="left"))
    if start_b >= len(ctx_b["time"]):
        raise HTTPException(status_code=422, detail="Synchronization shift moves terminal B beyond its record.")
    if start_b > 0 and abs(float(ctx_b["time"][start_b - 1]) - target_b_s) < abs(float(ctx_b["time"][start_b]) - target_b_s):
        start_b -= 1
    if start_b < ctx_b["inception_idx"]:
        raise HTTPException(
            status_code=422,
            detail="Synchronization shift maps terminal A's fault window before terminal B's detected inception.",
        )

    term_a = _phasor_at_window_start(ctx_a, start_a, aligned_start_s=common_start_s)
    aligned_b_start_s = float(ctx_b["time"][start_b]) - shift_s
    term_b = _phasor_at_window_start(ctx_b, start_b, aligned_start_s=aligned_b_start_s)
    return term_a, term_b


def _paired_fault_windows(
    ctx_a: dict,
    ctx_b: dict,
    shift_ms: float,
    *,
    n_windows: int = 17,
) -> list[tuple[dict, dict]]:
    """Select simultaneous, high-current windows from the fault interval."""
    freq = ctx_a["freq"]
    max_a_idx = (ctx_a.get("clearing_idx") or len(ctx_a["time"]) - 1) - ctx_a["win"]
    max_b_idx = (ctx_b.get("clearing_idx") or len(ctx_b["time"]) - 1) - ctx_b["win"]
    duration_a_s = max(0.0, (max_a_idx - ctx_a["inception_idx"]) / ctx_a["sr"])
    duration_b_s = max(0.0, (max_b_idx - ctx_b["inception_idx"]) / ctx_b["sr"])
    # Four cycles are enough to move beyond inception transients while not
    # drifting into breaker clearing or a later event in a long record.
    max_offset_s = min(duration_a_s, duration_b_s, 4.0 / freq)
    offsets = np.linspace(0.0, max_offset_s, max(3, n_windows))

    pairs: list[tuple[dict, dict]] = []
    for offset_s in offsets:
        try:
            pairs.append(_aligned_terminal_pair(ctx_a, ctx_b, shift_ms, float(offset_s)))
        except HTTPException:
            continue
    if not pairs:
        return []

    peak_a = max(abs(a["i_primary"]) for a, _b in pairs)
    peak_b = max(abs(b["i_primary"]) for _a, b in pairs)
    selected = [
        pair for pair in pairs
        if abs(pair[0]["i_primary"]) >= 0.60 * peak_a and abs(pair[1]["i_primary"]) >= 0.60 * peak_b
    ]
    return selected if len(selected) >= 3 else pairs


def _solve_multiwindow(pairs: list[tuple[dict, dict]], z_line: complex) -> Optional[dict]:
    """Least-squares real distance using several simultaneous KVL equations."""
    rows: list[tuple[complex, complex, float]] = []
    for term_a, term_b in pairs:
        i_a, i_b = term_a["i_primary"], term_b["i_primary"]
        if abs(i_a) < 1e-6 or abs(i_b) < 1e-6:
            continue
        lhs = term_a["v_primary"] - term_b["v_primary"] + z_line * i_b
        rhs = z_line * (i_a + i_b)
        if abs(rhs) < 1e-9:
            continue
        weight = min(abs(i_a), abs(i_b)) ** 2
        rows.append((lhs, rhs, weight))
    if not rows:
        return None

    denom = sum(weight * abs(rhs) ** 2 for _lhs, rhs, weight in rows)
    if denom <= 1e-18:
        return None
    m = float(np.real(sum(weight * np.conj(rhs) * lhs for lhs, rhs, weight in rows)) / denom)
    individual = np.array([lhs / rhs for lhs, rhs, _weight in rows], dtype=complex)
    weights = np.array([weight for _lhs, _rhs, weight in rows], dtype=float)
    weights /= float(np.sum(weights))
    residual_imag = float(np.sqrt(np.sum(weights * np.imag(individual) ** 2)))
    distance_spread_pu = float(np.sqrt(np.sum(weights * (np.real(individual) - m) ** 2)))
    kvl_residual = float(
        np.sqrt(
            sum(weight * abs(lhs - m * rhs) ** 2 for lhs, rhs, weight in rows)
            / denom
        )
    )
    return {
        "m": m,
        "residual_imag": residual_imag,
        "distance_spread_pu": distance_spread_pu,
        "kvl_residual": kvl_residual,
        "individual_m": individual,
        "window_count": len(rows),
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
    z_line = complex(r1_ohm_per_km, x1_ohm_per_km) * line_len_km
    if abs(z_line) < 1e-9:
        raise HTTPException(status_code=422, detail="R1/X1 must define a non-zero line impedance.")

    use_sequence = loop in GROUND_LOOPS
    ctx_a = _build_terminal_context(
        payload_a, loop, invert_i_a, invert_phase_sequence_a,
        sequence_for_ground=use_sequence,
    )
    ctx_b = _build_terminal_context(
        payload_b, loop, invert_i_b, invert_phase_sequence_b,
        sequence_for_ground=use_sequence,
    )
    if abs(ctx_a["freq"] - ctx_b["freq"]) > 0.1:
        raise HTTPException(status_code=422, detail="Terminal A and B nominal frequencies do not match.")

    pairs = _paired_fault_windows(ctx_a, ctx_b, manual_shift_ms, n_windows=21)
    solved = _solve_multiwindow(pairs, z_line)
    if solved is None:
        raise HTTPException(
            status_code=422,
            detail="No valid simultaneous high-current windows were found — check synchronization, polarity, "
                   "fault inception detection, and that both records contain the same event.",
        )
    m = solved["m"]
    m_residual_imag = solved["residual_imag"]
    term_a, term_b = pairs[0]

    warnings: list[str] = []
    if m < 0.0 or m > 1.0:
        warnings.append(
            f"Solved distance falls outside the line (m={m:.3f}) — check synchronization, "
            f"CT/PT ratios, phase sequence, and loop selection."
        )
    if abs(m_residual_imag) > 0.15 or solved["kvl_residual"] > 0.15:
        # This residual is the tool's own honest failure signal: a genuine
        # two-terminal solution to V_A - m*Zline*I_A = V_B - (1-m)*Zline*I_B
        # collapses Im(m) toward zero once A and B's phasors actually
        # describe the same physical instant. A residual this large means
        # the "intersection" the two terminals' equations are supposed to
        # agree on was never found — changing line_len_km alone cannot fix
        # this (Zline scales with it, but the underlying disagreement
        # between A's and B's phasors does not), so the message must tell
        # the user WHERE to go fix it, not just that something is wrong.
        warnings.append(
            f"Large multi-window inconsistency (RMS Im(m)={m_residual_imag:.3f}, normalized KVL "
            f"residual={solved['kvl_residual']:.3f}) — terminal A and B do not agree across the selected "
            f"simultaneous fault windows at this synchronization. "
            f"Re-entering a different line length will NOT fix this on its own. If the manual sync shift "
            f"(step 3) is still at its default/unconfirmed value, go there first and drag it until terminal "
            f"B's current step visually lines up with terminal A's on the overlay plot — a residual this size "
            f"almost always means the two records are not yet time-aligned. If the shift is already confirmed "
            f"and the residual is still large, check next: the loop selection (step 4 — does it match the "
            f"actually-faulted phase?), then each terminal's CT/PT ratio (step 2), then phase sequence."
        )

    if use_sequence:
        phase_ctx_a = _build_terminal_context(
            payload_a, loop, invert_i_a, invert_phase_sequence_a,
            sequence_for_ground=False,
        )
        phase_ctx_b = _build_terminal_context(
            payload_b, loop, invert_i_b, invert_phase_sequence_b,
            sequence_for_ground=False,
        )
        phase_pairs = _paired_fault_windows(phase_ctx_a, phase_ctx_b, manual_shift_ms, n_windows=21)
        fault_current_a = float(np.median([abs(a["i_primary"] + b["i_primary"]) for a, b in phase_pairs]))
    else:
        fault_current_a = float(np.median([abs(a["i_primary"] + b["i_primary"]) for a, b in pairs]))
    distance_km = m * line_len_km

    return {
        "loop": loop,
        "distance_km": distance_km,
        "distance_pct": (distance_km / line_len_km) * 100.0,
        "fault_current_a": fault_current_a,
        "m_residual_imag": m_residual_imag,
        "kvl_residual": solved["kvl_residual"],
        "distance_spread_km": solved["distance_spread_pu"] * line_len_km,
        "selected_window_count": solved["window_count"],
        "calculation_basis": "negative_sequence" if use_sequence else "phase_loop",
        "inception_time_a_s": term_a["inception_time_s"],
        "inception_time_b_s": term_b["inception_time_s"],
        "active_tag_a": term_a["active_tag"],
        "active_tag_b": term_b["active_tag"],
        "warnings": warnings,
    }


def _find_optimal_shift(
    payload_a: dict,
    payload_b: dict,
    loop: str,
    line_len_km: float,
    r1_ohm_per_km: float,
    x1_ohm_per_km: float,
    invert_i_a: bool,
    invert_i_b: bool,
    invert_phase_sequence_a: bool,
    invert_phase_sequence_b: bool,
) -> dict:
    """Estimate time-axis alignment from inception, refined by multi-window KVL.

    Searching an entire record for one minimum of Im(m) is ambiguous every
    power-frequency cycle and can fit a spurious window.  Here the detected
    inception-time difference fixes the correct cycle, while several paired
    high-current windows provide an overdetermined complex KVL objective.
    """
    if line_len_km <= 0:
        raise HTTPException(status_code=422, detail="line_len_km must be positive.")
    use_sequence = loop in GROUND_LOOPS
    ctx_a = _build_terminal_context(
        payload_a, loop, invert_i_a, invert_phase_sequence_a,
        sequence_for_ground=use_sequence,
    )
    ctx_b = _build_terminal_context(
        payload_b, loop, invert_i_b, invert_phase_sequence_b,
        sequence_for_ground=use_sequence,
    )
    z_line = complex(r1_ohm_per_km, x1_ohm_per_km) * line_len_km
    if abs(z_line) < 1e-9:
        raise HTTPException(status_code=422, detail="R1/X1 must define a non-zero line impedance.")

    inception_shift_ms = (
        float(ctx_b["time"][ctx_b["inception_idx"]])
        - float(ctx_a["time"][ctx_a["inception_idx"]])
    ) * 1000.0
    half_cycle_ms = 500.0 / ctx_a["freq"]

    def _score_at(shift_ms: float) -> Optional[tuple[float, dict]]:
        pairs = _paired_fault_windows(ctx_a, ctx_b, shift_ms, n_windows=13)
        solved = _solve_multiwindow(pairs, z_line)
        if solved is None or solved["window_count"] < 3:
            return None
        # KVL mismatch already contains both the imaginary inconsistency and
        # the real-distance spread.  A soft out-of-line penalty prevents an
        # algebraically neat but physically impossible candidate from winning.
        m = solved["m"]
        outside_penalty = max(0.0, -m, m - 1.0)
        return solved["kvl_residual"] + outside_penalty, solved

    coarse_step_ms = 0.5
    best_shift_ms: Optional[float] = None
    best_score = float("inf")
    best_solution: Optional[dict] = None
    shift_ms = inception_shift_ms - half_cycle_ms
    end_ms = inception_shift_ms + half_cycle_ms
    while shift_ms <= end_ms + 1e-9:
        candidate = _score_at(shift_ms)
        if candidate is not None and candidate[0] < best_score:
            best_score, best_solution = candidate
            best_shift_ms = shift_ms
        shift_ms += coarse_step_ms

    if best_shift_ms is None:
        return {
            "shift_ms": None,
            "residual": None,
            "searched_range_ms": half_cycle_ms,
            "reason": "No common high-current fault windows were found near the detected inception alignment.",
        }

    fine_step_ms = 0.02
    fine_shift_ms = best_shift_ms - coarse_step_ms
    fine_end_ms = best_shift_ms + coarse_step_ms
    while fine_shift_ms <= fine_end_ms + 1e-9:
        candidate = _score_at(fine_shift_ms)
        if candidate is not None and candidate[0] < best_score:
            best_score, best_solution = candidate
            best_shift_ms = fine_shift_ms
        fine_shift_ms += fine_step_ms

    assert best_solution is not None
    return {
        "shift_ms": round(best_shift_ms, 2),
        "residual": round(best_solution["kvl_residual"], 5),
        "searched_range_ms": half_cycle_ms,
        "reason": (
            f"Detected-inception alignment is {inception_shift_ms:.2f} ms; multi-window refinement selected "
            f"{best_shift_ms:.2f} ms using {best_solution['window_count']} simultaneous high-current windows "
            f"(normalized KVL residual={best_solution['kvl_residual']:.4f}). Confirm the inception markers on "
            "the zoomed waveform before running the final calculation."
        ),
    }


def _single_ended_distance(
    term: dict, r1_ohm_per_km: float, x1_ohm_per_km: float, line_len_km: float, terminal_label: str,
) -> dict:
    """One terminal's OWN single-ended distance reading — the classic
    single-ended calculation (webapp/api/routers/relay_21.py's locus, in
    spirit), computed here directly from primary V/I rather than reusing
    ``_compute_locus`` (which additionally scales to relay-secondary ohms
    for zone-overlay display; this needs primary ohms to match
    ``r1_ohm_per_km``/``x1_ohm_per_km``, the same convention
    ``_compute_double_ended`` already uses).

    m_single = Re(Z_measured / Z_per_km), same Re() convention as the
    two-ended m = Re(m_complex) above — NOT |Z|/|Z_per_km|. This matters
    physically: a fault resistance Rf adds a real (resistive) term to
    Z_measured (Z_measured = m*Z_per_km*L + Rf, to first order, when this
    terminal supplies essentially all of the fault current), and
    Re(Z_measured/Z_per_km) inflates by Rf*Re(1/Z_per_km) — a positive
    km-equivalent addition — reproducing exactly the "single-ended error
    grows with fault resistance" effect the source PPTX motivates this
    whole feature with. |Z|/|Z_per_km| would not isolate that resistive
    term the same way.

    Deliberately NOT clamped to [0, line_len_km] — an out-of-range
    single-ended reading (e.g. a K2 reading landing beyond the line's own
    length) is itself the illustrative point of showing it alongside the
    two-ended answer, not an error to hide."""
    v, i = term["v_primary"], term["i_primary"]
    warnings: list[str] = []
    if abs(i) < 1e-6:
        return {
            "terminal": terminal_label, "distance_km": 0.0, "distance_pct": 0.0,
            "fault_current_a": 0.0, "r_measured_ohm": 0.0, "x_measured_ohm": 0.0,
            "warnings": [f"Terminal {terminal_label}: fault current too small for a single-ended reading."],
        }
    z_measured = v / i
    z_per_km = complex(r1_ohm_per_km, x1_ohm_per_km)
    m_single = float(np.real(z_measured / z_per_km))
    distance_km = m_single * line_len_km
    if distance_km < 0.0 or distance_km > line_len_km:
        warnings.append(
            f"Terminal {terminal_label} single-ended reading ({distance_km:.2f} km) falls outside the line "
            f"length — expected under high fault resistance, since this reading (unlike the two-ended answer) "
            f"cannot separate Rf from line impedance."
        )
    return {
        "terminal": terminal_label,
        "distance_km": distance_km,
        "distance_pct": (distance_km / line_len_km) * 100.0,
        "fault_current_a": float(abs(i)),
        "r_measured_ohm": float(np.real(z_measured)),
        "x_measured_ohm": float(np.imag(z_measured)),
        "warnings": warnings,
    }


def _compute_distance_histogram(
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
    n_windows: int = 41,
) -> list[float]:
    """Per-window solutions from simultaneous, high-current fault windows."""
    z_line = complex(r1_ohm_per_km, x1_ohm_per_km) * line_len_km
    use_sequence = loop in GROUND_LOOPS
    ctx_a = _build_terminal_context(
        payload_a, loop, invert_i_a, invert_phase_sequence_a,
        sequence_for_ground=use_sequence,
    )
    ctx_b = _build_terminal_context(
        payload_b, loop, invert_i_b, invert_phase_sequence_b,
        sequence_for_ground=use_sequence,
    )
    pairs = _paired_fault_windows(ctx_a, ctx_b, manual_shift_ms, n_windows=n_windows)
    samples: list[float] = []
    for term_a, term_b in pairs:
        solved = _solve_m(
            term_a["v_primary"], term_a["i_primary"],
            term_b["v_primary"], term_b["i_primary"], z_line,
        )
        if solved is None:
            continue
        m, _residual = solved
        samples.append(m * line_len_km)

    return samples


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


def _run_compute(payload_a: dict, payload_b: dict, body: DoubleEndedComputeRequest) -> dict:
    result = _compute_double_ended(
        payload_a, payload_b, body.loop, body.line_len_km,
        body.r1_ohm_per_km, body.x1_ohm_per_km, body.manual_shift_ms,
        body.invert_i_a, body.invert_i_b,
        body.invert_phase_sequence_a, body.invert_phase_sequence_b,
    )

    # Single-ended K1/K2 readings and the distance histogram both need their
    # own terminal phasor pair — recomputed here rather than threading
    # _compute_double_ended's internal term_a/term_b out through its return
    # value, deliberately keeping that function's existing, already-tested
    # contract untouched. Each call is one cheap one-cycle-window phasor
    # extraction, not a meaningful cost next to the histogram sweep below.
    single_ctx_a = _build_terminal_context(
        payload_a, body.loop, body.invert_i_a, body.invert_phase_sequence_a,
        sequence_for_ground=False,
    )
    single_ctx_b = _build_terminal_context(
        payload_b, body.loop, body.invert_i_b, body.invert_phase_sequence_b,
        sequence_for_ground=False,
    )
    term_a, term_b = _aligned_terminal_pair(single_ctx_a, single_ctx_b, body.manual_shift_ms)
    result["single_ended_a"] = DoubleEndedSingleEndedResult(
        **_single_ended_distance(term_a, body.r1_ohm_per_km, body.x1_ohm_per_km, body.line_len_km, "A")
    )
    result["single_ended_b"] = DoubleEndedSingleEndedResult(
        **_single_ended_distance(term_b, body.r1_ohm_per_km, body.x1_ohm_per_km, body.line_len_km, "B")
    )
    result["distance_histogram_km"] = _compute_distance_histogram(
        payload_a, payload_b, body.loop, body.line_len_km,
        body.r1_ohm_per_km, body.x1_ohm_per_km, body.manual_shift_ms,
        body.invert_i_a, body.invert_i_b,
        body.invert_phase_sequence_a, body.invert_phase_sequence_b,
    )
    return result


@router.post("/suggest-shift", response_model=DoubleEndedSuggestShiftResponse)
async def suggest_shift(body: DoubleEndedSuggestShiftRequest):
    """Search for the manual_shift_ms that minimizes the two terminals'
    Kirchhoff residual — a grounded suggestion (unlike /align-estimate,
    which extrapolates from possibly-wrong wall-clock timestamps), but
    still only a starting point: the frontend must show it as something
    to visually confirm on the sync overlay, never auto-apply it. See
    _find_optimal_shift's docstring for why multiple local minima can
    exist and why user confirmation still matters."""
    payload_a = _load_or_404(body.analysis_id_a)
    payload_b = _load_or_404(body.analysis_id_b)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _find_optimal_shift(
            payload_a, payload_b, body.loop, body.line_len_km,
            body.r1_ohm_per_km, body.x1_ohm_per_km,
            body.invert_i_a, body.invert_i_b,
            body.invert_phase_sequence_a, body.invert_phase_sequence_b,
        ),
    )
    return DoubleEndedSuggestShiftResponse(**result)


@router.post("/compute", response_model=DoubleEndedComputeResponse)
async def compute(body: DoubleEndedComputeRequest):
    payload_a = _load_or_404(body.analysis_id_a)
    payload_b = _load_or_404(body.analysis_id_b)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: _run_compute(payload_a, payload_b, body))
    return DoubleEndedComputeResponse(**result)
