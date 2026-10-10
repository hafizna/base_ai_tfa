"""
Fault Event Detector
====================
Detects fault inception point and reclose events from waveform data.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import logging
import re

from .line_selection import cycle_rms_envelope, scope_record

logger = logging.getLogger(__name__)


def _normalize_fault_phase_list(phases: Optional[List[str]]) -> List[str]:
    """Keep only active phase labels in stable A/B/C order."""
    order = {"A": 0, "B": 1, "C": 2}
    cleaned = []
    for ph in phases or []:
        key = str(ph or "").upper().strip()
        if key in order and key not in cleaned:
            cleaned.append(key)
    return sorted(cleaned, key=lambda ph: order[ph])


def _prefer_waveform_fault_phases(status_phases: Optional[List[str]], waveform_phases: Optional[List[str]]) -> List[str]:
    """
    Reconcile phase picks from status and waveform detection.

    Status trip outputs can reflect a three-pole trip command rather than the true
    faulted phases. When waveform evidence is available and is more specific, prefer it.
    """
    status_clean = _normalize_fault_phase_list(status_phases)
    wave_clean = _normalize_fault_phase_list(waveform_phases)
    if not wave_clean:
        return status_clean
    if not status_clean:
        return wave_clean

    status_set = set(status_clean)
    wave_set = set(wave_clean)
    if status_set == wave_set:
        return wave_clean
    if status_set.issuperset(wave_set):
        return wave_clean
    return status_clean


def _extract_line_tag(channel_name: str) -> Optional[str]:
    """Extract a line/circuit/bay tag from a channel name (e.g. "IR BRINGIN 2"
    -> "2"). Shared with webapp/api/routers/relay_21.py, which imports this
    directly — an external DFR CFG recording two lines in one file (common:
    "IR BRINGIN 1"/"IR BRINGIN 2" side by side) needs the SAME tag extraction
    used to pick the active line here, or the two call sites can disagree
    about which line a channel belongs to."""
    s = (channel_name or "").upper()
    m = re.search(r"(?:LINE|BAY|JEPARA|SIRKIT|CCT|CIRCUIT)\s*#?\s*([0-9A-Z]+)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([0-9])\b", s)
    if m:
        return m.group(1)
    return None


def _normalize_status_name(name: str) -> str:
    """Normalize status channel name for robust matching."""
    if not name:
        return ""
    s = name.upper()
    # Replace separators (., _, /, -) with spaces.
    s = re.sub(r"[._/\\-]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _pick_current_channel(record, canonical_name: str, preferred_tag: Optional[str] = None):
    candidates = [
        ch for ch in record.analog_channels
        if ch.canonical_name == canonical_name and ch.measurement == "current"
    ]
    if not candidates:
        return None
    if preferred_tag:
        tagged = [c for c in candidates if _extract_line_tag(getattr(c, "name", "")) == preferred_tag]
        if tagged:
            return tagged[0]
    return candidates[0]


def _pick_voltage_channel(record, canonical_name: str, preferred_tag: Optional[str] = None):
    """Same lookup as _pick_current_channel, for voltage. Used to cross-check
    a waveform-inferred reclose against an actual CB-open dead-time (V AND I
    both near zero), not current alone — see _detect_reclose_from_waveforms."""
    candidates = [
        ch for ch in record.analog_channels
        if ch.canonical_name == canonical_name and ch.measurement == "voltage"
    ]
    if not candidates:
        return None
    if preferred_tag:
        tagged = [c for c in candidates if _extract_line_tag(getattr(c, "name", "")) == preferred_tag]
        if tagged:
            return tagged[0]
    return candidates[0]


def _detect_active_line_tag_from_currents(record) -> Optional[str]:
    """Pick line/circuit tag with largest overall current activity in the record."""
    scores = {}
    for ch in record.analog_channels:
        if ch.measurement != "current" or ch.canonical_name not in {"IA", "IB", "IC"}:
            continue
        if len(ch.samples) == 0:
            continue
        tag = _extract_line_tag(getattr(ch, "name", ""))
        if not tag:
            continue
        scores[tag] = scores.get(tag, 0.0) + float(np.max(np.abs(ch.samples)))
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


@dataclass
class FaultEvent:
    """Represents a detected fault event."""
    inception_idx: int           # Sample index of fault start
    inception_time: float        # Time in seconds
    clearing_idx: Optional[int]  # Sample index of fault clearing
    clearing_time: Optional[float]
    duration_ms: float           # Fault duration in milliseconds
    detection_method: str        # "current_derivative" or "rms_change" or "status_channel"
    confidence: float            # 0-1
    faulted_phases: List[str]    # ["A"], ["A", "B"], etc.

    # Reclose detection
    reclose_events: List[dict]   # List of {time, success: bool} for each reclose attempt


def detect_fault(record) -> Optional[FaultEvent]:
    """
    Detect fault inception from waveform data.

    Strategy (try in order):
    1. Status channels: If trip/pickup channels exist, use their transition times
    2. Current derivative: Where |dI/dt| exceeds 3x pre-fault max on any phase
    3. RMS change: Where RMS current changes by >50% in one cycle

    Also detects reclose events:
    - After fault clearing, look for current returning (breaker reclose)
    - If current returns and stays stable → successful reclose
    - If current returns and another fault occurs → failed reclose
    - Track all reclose attempts with timestamps

    Args:
        record: ComtradeRecord with waveform data

    Returns:
        FaultEvent or None if no fault detected
    """

    # A DFR recording two lines in one file is analysed on its disturbed line
    # only — the other line's analog AND status channels are dropped, so an
    # out-of-service neighbour's "CB OPEN" or noise can't leak into timing,
    # phases or reclose detection below. No-op for single-line records.
    record = scope_record(record)

    # Detect dead-time recordings: CB was already open when recording started.
    # In this case the fault occurred before this recording — no fault current present.
    # Skip straight to waveform reclose detection; fault duration is not measurable.
    if _recording_starts_in_dead_time(record):
        logger.debug("Recording started in CB dead time — fault preceded this file")
        return _build_dead_time_event(record)
    # Same situation on a DFR whose breaker status isn't wired (or never
    # asserts): the line itself shows it — line-side voltage AND current ~0 at
    # the start, then the line re-energizes. Without this, the energization
    # inrush reads as a brand-new three-phase "fault" (seen on the GI
    # Mojosongo Qualitrol record of the 21/08/2023 reclose).
    energized_idx = _energization_after_dead_start(record)
    if energized_idx is not None:
        logger.debug("Recording starts with the line de-energized — treating as a reclose capture")
        return _build_dead_time_event(record, energized_idx=energized_idx)

    # Status-channel candidate (trip/pickup based)
    fault = _detect_from_status_channels(record)

    # Waveform candidate (current onset based)
    wf_fault = _detect_from_waveforms(record)

    # Reconcile onset time: status signals may occur after fault has already started.
    if fault and wf_fault:
        status_t = float(fault.inception_time or 0.0)
        wave_t = float(wf_fault.inception_time or 0.0)
        if len(record.time) > 1:
            dt = float(record.time[1] - record.time[0])
        else:
            dt = 0.0001
        onset_slip = status_t - wave_t
        # If status onset is >1 cycle later, prefer waveform onset for electrical features.
        if onset_slip > max(0.010, 20.0 * dt):
            logger.debug(
                "Status onset appears late vs waveform onset "
                f"(status={status_t:.4f}s, wave={wave_t:.4f}s). Using waveform onset."
            )
            fault.inception_idx = wf_fault.inception_idx
            fault.inception_time = wf_fault.inception_time
            fault.detection_method = "status_waveform_aligned"

        reconciled_phases = _prefer_waveform_fault_phases(
            fault.faulted_phases,
            wf_fault.faulted_phases,
        )
        if reconciled_phases != _normalize_fault_phase_list(fault.faulted_phases):
            logger.debug(
                "Using waveform-derived fault phases over status phases "
                f"(status={fault.faulted_phases}, wave={wf_fault.faulted_phases}, "
                f"resolved={reconciled_phases})"
            )
            fault.faulted_phases = reconciled_phases

    if fault and fault.confidence > 0.7:
        # If status channels found a plausible duration (>= 5ms), return it.
        # Sub-5ms durations are toggle noise (e.g. A/R signal bouncing), not real clearing.
        if fault.duration_ms >= 5.0:
            logger.debug(f"Fault detected from status channels: {fault.inception_time:.4f}s  dur={fault.duration_ms:.1f}ms")
            return fault
        if fault.duration_ms > 0:
            logger.debug(f"Status duration {fault.duration_ms:.2f}ms too short (toggle noise) — discarding, using waveform clearing")
            fault.duration_ms = 0.0
            fault.clearing_idx = None
            fault.clearing_time = None
        logger.debug(f"Status channel found inception at {fault.inception_time:.4f}s but no reliable clearing — trying waveform clearing")

    # Fall back to (or supplement with) waveform-based detection
    if wf_fault:
        # If status channel gave us a valid inception, use it but take waveform clearing
        if fault and fault.duration_ms == 0 and wf_fault.clearing_idx:
            fault.clearing_idx  = wf_fault.clearing_idx
            fault.clearing_time = wf_fault.clearing_time
            fault.duration_ms   = wf_fault.duration_ms
            fault.faulted_phases = fault.faulted_phases or wf_fault.faulted_phases
            fault.reclose_events = fault.reclose_events or wf_fault.reclose_events
            logger.debug(f"Waveform clearing applied: dur={fault.duration_ms:.1f}ms")
            return fault
        logger.debug(f"Fault detected from waveforms: {wf_fault.inception_time:.4f}s ({wf_fault.detection_method})")
        return wf_fault

    # Return status-only result even with 0ms if nothing better found
    if fault:
        return fault

    logger.warning("No fault detected in recording")
    return None


def _extract_phase_from_name(name_upper: str) -> Optional[str]:
    """Extract faulted phase from channel name (handles ABC and RST notations)."""
    import re
    if '(R)' in name_upper: return 'A'
    if '(S)' in name_upper: return 'B'
    if '(T)' in name_upper: return 'C'
    if any(k in name_upper for k in ['PHA FAULT', 'A PHASE FAULT', 'PHASE A FAULT', 'TRIP PHA', 'PHS A', 'TRIP PH A']): return 'A'
    if any(k in name_upper for k in ['PHB FAULT', 'B PHASE FAULT', 'PHASE B FAULT', 'TRIP PHB', 'PHS B', 'TRIP PH B']): return 'B'
    if any(k in name_upper for k in ['PHC FAULT', 'C PHASE FAULT', 'PHASE C FAULT', 'TRIP PHC', 'PHS C', 'TRIP PH C']): return 'C'
    if re.search(r'\bOPRT R\b|\bTRIP R\b|OPRT R$| R$', name_upper): return 'A'
    if re.search(r'\bOPRT S\b|\bTRIP S\b|OPRT S$| S$', name_upper): return 'B'
    if re.search(r'\bOPRT T\b|\bTRIP T\b|OPRT T$| T$', name_upper): return 'C'
    # L1/L2/L3 notation (ABB REL, some Siemens): L1=A, L2=B, L3=C
    if re.search(r'\bL1\b|L1$| L1[^0-9]', name_upper): return 'A'
    if re.search(r'\bL2\b|L2$| L2[^0-9]', name_upper): return 'B'
    if re.search(r'\bL3\b|L3$| L3[^0-9]', name_upper): return 'C'

    # PCS900: PhSA/PhSB/PhSC, TrpA/TrpB/TrpC, DZ1R/DZ1S/DZ1T
    if name_upper in ('PHSA',) or 'TRPA' in name_upper: return 'A'
    if name_upper in ('PHSB',) or 'TRPB' in name_upper: return 'B'
    if name_upper in ('PHSC',) or 'TRPC' in name_upper: return 'C'
    if re.search(r'DZ\d+R$', name_upper): return 'A'
    if re.search(r'DZ\d+S$', name_upper): return 'B'
    if re.search(r'DZ\d+T$', name_upper): return 'C'

    if name_upper.endswith(' A'): return 'A'
    if name_upper.endswith(' B'): return 'B'
    if name_upper.endswith(' C'): return 'C'
    return None


_CB_OPEN_KW = ['CB OPEN', 'POLE DEAD', 'ANY POLE', 'ALL POLE', '52B', 'CB1.52B']
_CB_EXCL_KW = ['ALARM', 'TEST', 'BLOCK']


def _cb_open_status_channels(record):
    for ch in record.status_channels:
        nu = _normalize_status_name(ch.name)
        if any(e in nu for e in _CB_EXCL_KW) or not any(k in nu for k in _CB_OPEN_KW):
            continue
        yield ch


def _recording_starts_in_dead_time(record) -> bool:
    """
    Returns True if the CB was already open when the recording started.
    This happens when an external DFR is triggered by the open-CB signal
    rather than by the fault itself — the fault is not captured in this file.

    Indicators: a CB-open / pole-dead / 52b channel that is HIGH (=1) from
    the very first sample and has no rising edge (only a falling edge later
    when the breaker recloses).
    """
    for ch in _cb_open_status_channels(record):
        if len(ch.samples) < 10:
            continue
        # High from start AND has at least one falling edge (CB eventually reclosed)
        if ch.samples[0] == 1 and ch.samples[:5].sum() == 5:
            diff = np.diff(ch.samples)
            if (diff < 0).any():  # has falling edge = CB reclosed
                return True
    return False


def _samples_per_cycle(record) -> int:
    if len(record.time) < 2 or record.time[1] <= record.time[0]:
        return 0
    freq = float(getattr(record, "frequency", None) or 50.0)
    return max(4, int(round(1.0 / (float(record.time[1] - record.time[0]) * freq))))


def _line_phase_channels(record):
    tag = _detect_active_line_tag_from_currents(record)
    volts = [_pick_voltage_channel(record, name, tag) for name in ('VA', 'VB', 'VC')]
    amps = [_pick_current_channel(record, name, tag) for name in ('IA', 'IB', 'IC')]
    return volts, amps


def _energization_after_dead_start(record) -> Optional[int]:
    """Sample index at which a line that is de-energized at the start of the
    record (line-side voltage AND current ~0) becomes energized, or None.

    The breaker-status counterpart is ``_recording_starts_in_dead_time``;
    this one needs no status at all. Deliberately strict so an ordinary fault
    record can never match: every phase voltage must start below 10% of the
    level the line later settles at (a record that starts in load, or in a
    fault that still has voltage on a healthy phase, fails this), and the
    current at the start must not exceed what flows once energized (a record
    that starts inside a close-in three-phase fault — voltage ~0 but fault
    current — fails this). Requires all three phase voltages and currents.
    Returns the index of the first pole energizing (mid-window estimate,
    within half a cycle), which is the reclose instant used for dead time.
    """
    n = _samples_per_cycle(record)
    volts, amps = _line_phase_channels(record)
    if n == 0 or any(ch is None for ch in volts + amps):
        return None
    v_env = [cycle_rms_envelope(ch.samples, n) for ch in volts]
    i_env = [cycle_rms_envelope(ch.samples, n) for ch in amps]
    if any(len(env) < 4 * n for env in v_env + i_env):
        return None

    v_all = np.vstack(v_env)
    v_on = float(np.percentile(v_all.min(axis=0), 90))  # level once every phase is live
    if v_on <= 0.0 or float(v_all[:, 0].max()) > 0.10 * v_on:
        return None
    first_live = np.where(v_all.max(axis=0) >= 0.5 * v_on)[0]
    if len(first_live) == 0 or first_live[0] < n:
        return None  # needs at least one dead cycle before the line comes alive
    window_idx = int(first_live[0])

    i_all = np.vstack(i_env)
    settled = min(window_idx + 3 * n, i_all.shape[1] - 1)
    i_after = float(np.median(i_all.max(axis=0)[settled:]))
    i_start = float(i_all[:, 0].max())
    if i_start > max(3.0 * i_after, 0.01 * float(i_all.max())):
        return None

    # The line must come alive HEALTHY: balanced positive-sequence voltage
    # once settled. A voltage step into a fault (unbalanced; strong negative
    # sequence) is a fault inception, not an energization — even when the
    # pre-step voltage happens to be small next to it. Closing onto a fault
    # therefore falls through to normal fault detection on records without
    # breaker status (with status, _recording_starts_in_dead_time catches it
    # and _reclose_outcome_after reports the failure).
    if settled + n > len(record.time):
        return None
    kernel = np.exp(-2j * np.pi * np.arange(n) / n)
    va, vb, vc = (complex(np.dot(np.asarray(ch.samples[settled:settled + n], dtype=float), kernel)) for ch in volts)
    a = np.exp(2j * np.pi / 3)
    v1 = abs(va + a * vb + a * a * vc)
    v2 = abs(va + a * a * vb + a * vc)
    if v1 <= 0.0 or v2 > 0.2 * v1:
        return None
    return min(window_idx + n // 2, len(record.time) - 1)


def _reclose_outcome_after(record, reclose_idx: int, default: Optional[bool]) -> Optional[bool]:
    """Did the reclose at ``reclose_idx`` hold for the rest of the record?

    False when a breaker-open status asserts again, or — after a 3-cycle
    settling window that lets energization inrush decay — the line voltage
    collapses below 60% of its settled level, or the current surges above 3x
    its settled level while the voltage also dips below 90% (a fault drags
    the voltage down; the local breaker closing onto a line already charged
    from the far end takes current from ~0 to load with no voltage dip).
    ``default`` (True for a status-confirmed close, None for a waveform-only
    one) when the record ends before that can be judged or no analog
    evidence contradicts it.
    """
    n = _samples_per_cycle(record)
    hold = max(n, 1)
    for ch in _cb_open_status_channels(record):
        samples = np.asarray(ch.samples, dtype=int)
        rises = np.where(np.diff(samples) > 0)[0] + 1
        # A breaker that really re-opened stays open for at least a cycle; a
        # blip of a few ms right after the close is auxiliary-contact bounce
        # (seen on a real successful reclose: CB OPEN 1->0->1->0 within 5 ms).
        if any(np.all(samples[r:r + hold] == 1) for r in rises[rises > reclose_idx]):
            return False

    settle = reclose_idx + 3 * n
    if n == 0 or settle >= len(record.time) - 2 * n:
        return default
    volts, amps = _line_phase_channels(record)
    if any(ch is None for ch in volts):
        return default
    v_min = np.vstack([cycle_rms_envelope(ch.samples, n) for ch in volts]).min(axis=0)
    v_settled = float(np.median(v_min[settle:settle + n]))
    if v_settled <= 0.0:
        return default
    if np.any(v_min[settle:] < 0.6 * v_settled):
        return False
    if all(ch is not None for ch in amps):
        i_max = np.vstack([cycle_rms_envelope(ch.samples, n) for ch in amps]).max(axis=0)
        i_settled = float(np.median(i_max[settle:settle + n]))
        surge = i_max[settle:] > 3.0 * i_settled
        if i_settled > 0.0 and np.any(surge & (v_min[settle:len(i_max)] < 0.9 * v_settled)):
            return False
    return True if default is None else default


def _build_dead_time_event(record, energized_idx: Optional[int] = None) -> Optional[FaultEvent]:
    """
    Build a minimal FaultEvent for a dead-time recording (a "reclose
    capture"): the fault happened before this file, so its duration is
    unknown. The reclose instant is where the CB-open signal drops (or an AR
    success signal rises) — or, for a DFR without breaker status, where the
    line voltage returns (``energized_idx``). Its outcome is judged from the
    rest of the record rather than assumed successful.
    """
    reclose_idx = energized_idx
    source = "waveform" if energized_idx is not None else "status"
    if reclose_idx is None:
        for ch in _cb_open_status_channels(record):
            falls = np.where(np.diff(ch.samples) < 0)[0]
            if len(falls) and (reclose_idx is None or falls[0] + 1 < reclose_idx):
                reclose_idx = int(falls[0] + 1)

        # Also check AR success channel
        AR_SUCCESS_KW = ['AR SUCC', 'SUCC_RCLS', 'RECLOSE SUCC', '.79.SUCC']
        for ch in record.status_channels:
            nu = _normalize_status_name(ch.name)
            if any(k in nu for k in AR_SUCCESS_KW) and ch.samples.sum() > 0:
                rises = np.where(np.diff(ch.samples) > 0)[0]
                if len(rises) and (reclose_idx is None or rises[0] + 1 < reclose_idx):
                    reclose_idx = int(rises[0] + 1)

    # Use t=0 as nominal inception (fault was before recording)
    inception_time = record.time[0]

    reclose_events = []
    if reclose_idx is not None:
        reclose_idx = min(reclose_idx, len(record.time) - 1)
        reclose_events = [{
            'time': record.time[reclose_idx],
            'success': _reclose_outcome_after(record, reclose_idx, default=True if source == "status" else None),
            # The whole pre-reclose part of this record IS the open-breaker
            # dead time (that's how it was recognised), so the reclose is
            # verified; the dead time itself started before the file.
            'cb_open_verified': True,
            'dead_time_ms': None,
            'source': source,
        }]

    return FaultEvent(
        inception_idx=0,
        inception_time=inception_time,
        clearing_idx=None,
        clearing_time=None,
        duration_ms=0.0,        # genuinely unknown — fault not in this recording
        detection_method="dead_time_recording",
        confidence=0.6 if source == "status" else 0.55,
        faulted_phases=[],
        reclose_events=reclose_events,
    )


def _detect_from_status_channels(record) -> Optional[FaultEvent]:
    """
    Detect fault inception from status channel transitions.

    Look for channels like:
    - "Trip", "Operate", "Pickup", "Start"
    - Find the first transition from 0 to 1

    Returns:
        FaultEvent or None
    """

    # Keywords that indicate a protection operate/trip — expanded for all naming conventions.
    # NOTE: 'START' and 'STARTUP' are intentionally excluded — they are pickup/pre-trip
    # indicators (e.g. "Relay Startup", "B Phase Startup") that may not have a clearing edge.
    TRIP_KEYWORDS = [
        'TRIP', 'OPERATE', 'OPRT', 'PICKUP',
        'LP OPRT',      # External DFR Indonesian: "LP OPRT R WTS2"
        'MPU MAIN',     # External DFR: "MPU MAIN 1 TRIP (S) UNGARAN 1"
        'CB1.TRP',      # PCS900 Siemens: "CB1.TrpA/B/C"
        '.OP',          # PCS900: "21Q1.Op"
        'DZ1', 'DZ2',   # PCS900: "DZ1R/S/T"
        'RELAY TRIP',   # ABB REL: "Relay TRIP L2"
    ]
    # Keywords that should NOT trigger fault detection even if they contain TRIP/START
    EXCLUDE_KEYWORDS = [
        'RECLOSE', 'CLOSURE', 'A/R', 'AR INPROG', 'INPROGRESS',
        'CB CLOSE', 'CLOSE CMD',
        'SEND', 'RCV', 'RECV',
        'RELAY TEST', 'RELAY BLOCK',
        'OVERLOAD', 'ALARM',
        'SUCC', 'FAIL', 'LOCKOUT',
        'SWITCH SETGRP', 'BLK REM',
    ]

    best_inception_idx = None
    best_clearing_idx = None
    best_channel_name = None
    faulted_phases = []

    _extract_phase = _extract_phase_from_name

    for ch in record.status_channels:
        name_upper = _normalize_status_name(ch.name)

        # Skip non-trip channels
        if any(ex in name_upper for ex in EXCLUDE_KEYWORDS):
            continue

        is_trip = any(kw in name_upper for kw in TRIP_KEYWORDS)
        if not is_trip:
            continue

        if len(ch.samples) < 2:
            continue

        transitions = np.diff(ch.samples)
        rising_edges = np.where(transitions > 0)[0]

        if len(rising_edges) == 0:
            continue

        # Use earliest rising edge across all trip channels
        first_on = rising_edges[0] + 1
        if best_inception_idx is None or first_on < best_inception_idx:
            best_inception_idx = first_on
            best_channel_name = ch.name

        # For clearing: find the LAST falling edge within 500ms of the first ON for this channel.
        # This handles external DFR contact bounce (multiple brief pulses = one sustained event).
        # IMPORTANT: skip channels that represent CB open/dead-time (pole-open position signals)
        # — those stay high during the entire AR dead time and would inflate fault duration.
        name_upper_ch = _normalize_status_name(ch.name)
        is_pole_position = (
            'POSITION' in name_upper_ch and 'OPEN' in name_upper_ch
        ) or any(k in name_upper_ch for k in ['POLE DEAD', '1-POLE OPEN', '1POLE OPEN', 'ANY POLE', 'ALL POLE', '52B'])
        if is_pole_position:
            continue   # don't use CB-open position channels for fault duration

        falling_edges = np.where(transitions < 0)[0]
        later_falls = falling_edges[falling_edges >= rising_edges[0]]
        if len(later_falls) > 0:
            # Cap search window at 500ms after inception
            t_inception = record.time[first_on]
            within_window = [
                fi for fi in later_falls
                if fi + 1 < len(record.time) and record.time[fi + 1] - t_inception <= 0.5
            ]
            last_fall = within_window[-1] if within_window else later_falls[0]
            ch_clearing = last_fall + 1
        else:
            ch_clearing = None

        # Keep the clearing from the channel with the latest clearing time
        # (gives us the full fault duration across all trip channels)
        if ch_clearing is not None:
            if best_clearing_idx is None or ch_clearing > best_clearing_idx:
                best_clearing_idx = ch_clearing

        # Collect faulted phases from this channel
        if _extract_phase:
            ph = _extract_phase(name_upper)
            if ph and ph not in faulted_phases:
                faulted_phases.append(ph)

    if best_inception_idx is None:
        return None

    inception_time = record.time[best_inception_idx] if best_inception_idx < len(record.time) else 0.0
    clearing_time = (record.time[best_clearing_idx]
                     if best_clearing_idx and best_clearing_idx < len(record.time) else None)
    duration_ms = (clearing_time - inception_time) * 1000 if clearing_time else 0.0

    reclose_events = _detect_reclose_from_status(record, best_inception_idx)

    logger.debug(f"Fault detected from status channel '{best_channel_name}': "
                 f"inception={inception_time:.4f}s dur={duration_ms:.1f}ms phases={faulted_phases}")

    return FaultEvent(
        inception_idx=best_inception_idx,
        inception_time=inception_time,
        clearing_idx=best_clearing_idx,
        clearing_time=clearing_time,
        duration_ms=duration_ms,
        detection_method="status_channel",
        confidence=0.9,
        faulted_phases=faulted_phases,
        reclose_events=reclose_events
    )


def _detect_from_waveforms(record) -> Optional[FaultEvent]:
    """
    Detect fault inception from current waveforms.

    Uses current derivative (dI/dt) method:
    1. Calculate dI/dt for each phase
    2. Find where |dI/dt| exceeds threshold (3x pre-fault max)
    3. Use earliest detection across all phases
    """

    active_line_tag = _detect_active_line_tag_from_currents(record)

    # Get current channels (prefer dominant line tag for multi-line COMTRADE)
    ia = _pick_current_channel(record, 'IA', active_line_tag)
    ib = _pick_current_channel(record, 'IB', active_line_tag)
    ic = _pick_current_channel(record, 'IC', active_line_tag)

    if not (ia and ib and ic):
        # Fallback: transformer / DFR recordings may not have IA/IB/IC canonical names.
        # Pick the 3 highest-energy current channels (by peak amplitude) as surrogates.
        all_current_chs = [
            ch for ch in record.analog_channels
            if ch.measurement == "current" and len(ch.samples) > 0
        ]
        if len(all_current_chs) >= 3:
            all_current_chs.sort(key=lambda c: float(np.max(np.abs(c.samples))), reverse=True)
            ia, ib, ic = all_current_chs[0], all_current_chs[1], all_current_chs[2]
            logger.debug(
                "No IA/IB/IC canonical channels — using highest-energy surrogates: "
                f"{ia.name}, {ib.name}, {ic.name}"
            )
        elif len(all_current_chs) > 0:
            # Fewer than 3 channels — duplicate the best one so maths still work
            while len(all_current_chs) < 3:
                all_current_chs.append(all_current_chs[-1])
            ia, ib, ic = all_current_chs[0], all_current_chs[1], all_current_chs[2]
            logger.debug("Using fewer than 3 current channels (duplicated for fault detection)")
        else:
            logger.warning("Cannot detect fault: no current channels found")
            return None

    if len(ia.samples) == 0 or len(record.time) == 0:
        logger.warning("Cannot detect fault: no samples")
        return None

    # Calculate sampling interval
    if len(record.time) > 1:
        dt = record.time[1] - record.time[0]
    else:
        logger.warning("Cannot detect fault: insufficient time samples")
        return None

    # Calculate dI/dt for each phase
    di_dt_a = np.gradient(ia.samples, dt)
    di_dt_b = np.gradient(ib.samples, dt)
    di_dt_c = np.gradient(ic.samples, dt)

    # Pre-fault baseline: cap at 50ms to avoid swallowing the fault in long recordings.
    # Long external DFR recordings (e.g. 2.4s) with fault at 128ms would otherwise
    # include the fault in the 10% baseline window and make the threshold too high.
    max_prefault_ms = 50.0  # ms
    max_prefault_samples = max(10, int(max_prefault_ms / 1000.0 / dt))
    prefault_length = min(int(len(ia.samples) * 0.1), max_prefault_samples)
    if prefault_length < 10:
        prefault_length = min(10, len(ia.samples) // 2)

    # Calculate pre-fault threshold
    prefault_di_dt_max_a = np.max(np.abs(di_dt_a[:prefault_length]))
    prefault_di_dt_max_b = np.max(np.abs(di_dt_b[:prefault_length]))
    prefault_di_dt_max_c = np.max(np.abs(di_dt_c[:prefault_length]))

    threshold_a = 3.0 * prefault_di_dt_max_a if prefault_di_dt_max_a > 0 else 1000.0
    threshold_b = 3.0 * prefault_di_dt_max_b if prefault_di_dt_max_b > 0 else 1000.0
    threshold_c = 3.0 * prefault_di_dt_max_c if prefault_di_dt_max_c > 0 else 1000.0

    # Find where dI/dt exceeds threshold
    fault_candidates_a = np.where(np.abs(di_dt_a) > threshold_a)[0]
    fault_candidates_b = np.where(np.abs(di_dt_b) > threshold_b)[0]
    fault_candidates_c = np.where(np.abs(di_dt_c) > threshold_c)[0]

    # Combine all candidates and find earliest
    all_candidates = np.concatenate([fault_candidates_a, fault_candidates_b, fault_candidates_c])

    if len(all_candidates) == 0:
        logger.warning("No fault detected: dI/dt never exceeded threshold")
        return None

    inception_idx = int(np.min(all_candidates))
    inception_time = record.time[inception_idx]

    # Determine which phases faulted
    faulted_phases = []
    if len(fault_candidates_a) > 0 and fault_candidates_a[0] <= inception_idx + 5:
        faulted_phases.append('A')
    if len(fault_candidates_b) > 0 and fault_candidates_b[0] <= inception_idx + 5:
        faulted_phases.append('B')
    if len(fault_candidates_c) > 0 and fault_candidates_c[0] <= inception_idx + 5:
        faulted_phases.append('C')

    # Detect clearing (when current drops back to pre-fault levels)
    clearing_idx = _detect_fault_clearing(ia, ib, ic, inception_idx, prefault_length, dt=dt)
    clearing_time = record.time[clearing_idx] if clearing_idx and clearing_idx < len(record.time) else None

    duration_ms = (clearing_time - inception_time) * 1000 if clearing_time else 0.0

    # Detect reclose events. Pass voltage channels (same active line tag as
    # the currents) when available so a "current came back" reading can be
    # cross-checked against an actual CB-open dead-time window (V AND I both
    # near zero) rather than trusting current alone — see
    # _detect_reclose_from_waveforms docstring.
    va = _pick_voltage_channel(record, 'VA', active_line_tag)
    vb = _pick_voltage_channel(record, 'VB', active_line_tag)
    vc = _pick_voltage_channel(record, 'VC', active_line_tag)
    reclose_events = (
        _detect_reclose_from_waveforms(ia, ib, ic, record.time, clearing_idx, va, vb, vc)
        if clearing_idx else []
    )

    return FaultEvent(
        inception_idx=inception_idx,
        inception_time=inception_time,
        clearing_idx=clearing_idx,
        clearing_time=clearing_time,
        duration_ms=duration_ms,
        detection_method="current_derivative",
        confidence=0.8,
        faulted_phases=faulted_phases,
        reclose_events=reclose_events
    )


def _detect_fault_clearing(ia, ib, ic, inception_idx, prefault_length, dt=None):
    """
    Detect when fault current returns to pre-fault levels.

    Uses a sliding half-cycle RMS window (not instantaneous samples) to avoid
    false-early clearing on:
      - Transformer inrush (current naturally drops in missing half-cycles)
      - Single-phase faults where unfaulted phases stay near zero
      - Pre-fault load ≈ 0 (energisation) where threshold_rms would be ~0

    Threshold = max(2.0 × prefault_rms, 0.15 × peak_fault_rms)
    Confirmation = half-cycle RMS stays below threshold for one full cycle.
    """
    n = len(ia.samples)

    # Estimate samples-per-cycle (spc); default 96 for 4800 S/s / 50 Hz
    if dt and dt > 0:
        spc = max(8, int(round(0.02 / dt)))   # 1 cycle at 50 Hz
    else:
        spc = 96
    half_spc = max(4, spc // 2)

    # Pre-fault RMS (cycle-window, not the whole prefault region)
    pf_win = slice(max(0, prefault_length - spc), prefault_length)
    prefault_rms_a = float(np.sqrt(np.mean(ia.samples[pf_win] ** 2)))
    prefault_rms_b = float(np.sqrt(np.mean(ib.samples[pf_win] ** 2)))
    prefault_rms_c = float(np.sqrt(np.mean(ic.samples[pf_win] ** 2)))

    # Peak fault RMS (first 3 cycles after inception)
    fault_win = slice(inception_idx, min(inception_idx + 3 * spc, n))
    peak_rms_a = float(np.sqrt(np.mean(ia.samples[fault_win] ** 2)))
    peak_rms_b = float(np.sqrt(np.mean(ib.samples[fault_win] ** 2)))
    peak_rms_c = float(np.sqrt(np.mean(ic.samples[fault_win] ** 2)))

    # Threshold: at least 2× prefault, floor at 15% of peak fault RMS
    threshold_a = max(prefault_rms_a * 2.0, peak_rms_a * 0.15)
    threshold_b = max(prefault_rms_b * 2.0, peak_rms_b * 0.15)
    threshold_c = max(prefault_rms_c * 2.0, peak_rms_c * 0.15)

    # Minimum search start: at least 10ms after inception
    min_start = inception_idx + half_spc

    for i in range(min_start, n - spc):
        win = slice(i, i + half_spc)
        rms_a = float(np.sqrt(np.mean(ia.samples[win] ** 2)))
        rms_b = float(np.sqrt(np.mean(ib.samples[win] ** 2)))
        rms_c = float(np.sqrt(np.mean(ic.samples[win] ** 2)))

        if rms_a < threshold_a and rms_b < threshold_b and rms_c < threshold_c:
            # Confirm: sustained low for one full cycle
            confirm_win = slice(i, i + spc)
            rms_a2 = float(np.sqrt(np.mean(ia.samples[confirm_win] ** 2)))
            rms_b2 = float(np.sqrt(np.mean(ib.samples[confirm_win] ** 2)))
            rms_c2 = float(np.sqrt(np.mean(ic.samples[confirm_win] ** 2)))
            if rms_a2 < threshold_a and rms_b2 < threshold_b and rms_c2 < threshold_c:
                return i

    return None


def _detect_reclose_from_status(record, inception_idx):
    """
    Detect reclose events and their outcome from status channels.

    Success determined by:
    - AR Succ / AR Success channel active → True
    - CB Close command issued (BO CB CLOSE) → True
    - AR Fail / AR Lockout / AR Final Trip / 79 Final Trip → False
    - AR in progress but recording ends before completion → None (truncated)
    """
    reclose_events = []

    AR_ATTEMPT_KW  = ['RECLOSE', 'CLOSURE', 'A/R', 'AR ', 'AR INPROG', 'INPROGRESS',
                      '1P TRIP INIT', '3P TRIP INIT', 'AR 1POLE', '1POLE IN PROG', 'A/R OPRT']
    AR_SUCCESS_KW  = ['AR SUCC', 'RECLOSE SUCC', 'CB CLOSE', 'BO14', 'BO13', 'SYN MEET', 'VOL MEET']
    AR_FAILURE_KW  = ['AR FAIL', 'AR LOCKOUT', 'AR FINAL', '79 FINAL', 'FINAL TRIP', 'TOR', 'TRIP ON RECLOSE']
    POLE_DEAD_KW   = ['POLE DEAD', 'ANY POLE', 'ALL POLE']

    # Pre-scan: collect success/failure evidence
    success_times = []
    failure_times = []
    for ch in record.status_channels:
        name_upper = _normalize_status_name(ch.name)
        transitions = np.diff(ch.samples)
        rising = np.where(transitions > 0)[0]
        if len(rising) == 0:
            continue
        for edge in rising:
            t = record.time[edge + 1] if edge + 1 < len(record.time) else record.time[-1]
            if edge > inception_idx:
                if any(k in name_upper for k in AR_SUCCESS_KW):
                    success_times.append(t)
                if any(k in name_upper for k in AR_FAILURE_KW):
                    failure_times.append(t)

    # Detect AR attempts and assign success/failure
    seen_reclose_times = set()
    for ch in record.status_channels:
        name_upper = _normalize_status_name(ch.name)
        if not any(kw in name_upper for kw in AR_ATTEMPT_KW):
            continue

        transitions = np.diff(ch.samples)
        rising_edges = np.where(transitions > 0)[0]

        for edge_idx in rising_edges:
            if edge_idx <= inception_idx:
                continue
            reclose_time = record.time[edge_idx + 1] if edge_idx + 1 < len(record.time) else record.time[-1]

            t_key = round(float(reclose_time) * 100)
            if t_key in seen_reclose_times:
                continue
            seen_reclose_times.add(t_key)

            success = None
            if any(st > reclose_time - 0.05 for st in failure_times):
                success = False
            elif any(st > reclose_time - 0.05 for st in success_times):
                success = True

            reclose_events.append({'time': reclose_time, 'success': success})

    # "Any/All Pole Dead" falling edge (1→0) after inception = CB reclosed
    for ch in record.status_channels:
        name_upper = _normalize_status_name(ch.name)
        if not any(k in name_upper for k in POLE_DEAD_KW):
            continue
        transitions = np.diff(ch.samples)
        rising_edges = np.where(transitions > 0)[0]
        falling_edges = np.where(transitions < 0)[0]
        # Must have both a rising (CB opened) and falling (CB reclosed) after inception
        post_rise = rising_edges[rising_edges > inception_idx]
        if len(post_rise) == 0:
            continue
        post_fall = falling_edges[falling_edges > post_rise[0]]
        for edge_idx in post_fall:
            reclose_time = record.time[edge_idx + 1] if edge_idx + 1 < len(record.time) else record.time[-1]
            t_key = round(float(reclose_time) * 100)
            if t_key in seen_reclose_times:
                continue
            seen_reclose_times.add(t_key)
            # Falling edge of pole-dead = CB closed = successful reclose (no fault recurrence)
            success = True
            if any(ft > reclose_time and ft < reclose_time + 0.3 for ft in failure_times):
                success = False
            reclose_events.append({'time': reclose_time, 'success': success})

    if not reclose_events:
        reclose_events = _breaker_position_reclose(record, inception_idx, failure_times)

    return reclose_events


# A breaker position contact: closed-state ("CB Closed C ph", "52A") or
# open-state ("CB Open", "52B"). Health, spring, gas and command channels are
# not positions.
_BREAKER_POSITION_TOKENS = frozenset({"CB", "52A", "52B", "PMT", "BREAKER"})
_NOT_A_POSITION = ("HEALTH", "ALARM", "FAIL", "SPRING", "GAS", "SF6", "LOCK", "BLOCK", "TRIP", "SUPERV",
                   "CMD", "COMMAND", "READY")
# An auto-reclose dead time lasts hundreds of ms; a shorter opening is not one.
_MIN_DEAD_TIME_S = 0.1
_REOPEN_WINDOW_S = 0.5


def _stable_changes(samples: np.ndarray, time: np.ndarray, start_idx: int, min_hold_s: float = 0.005) -> list:
    """Indices after ``start_idx`` where the contact changes state and holds
    the new state for at least ``min_hold_s`` (contact bounce removed)."""
    state = bool(samples[start_idx])
    edges = np.flatnonzero(np.diff(samples[start_idx:].astype(int)) != 0) + start_idx + 1
    out = []
    for k, idx in enumerate(edges):
        new_state = bool(samples[idx])
        until = edges[k + 1] if k + 1 < len(edges) else len(samples) - 1
        if new_state != state and float(time[until] - time[idx]) >= min_hold_s:
            out.append(int(idx))
            state = new_state
    return out


def _breaker_position_reclose(record, inception_idx, failure_times) -> list:
    """A breaker contact that left its prefault position after the fault and
    came back to it: the breaker reclosed. Reading the return to the prefault
    state works for closed-state (52A) and open-state (52B) contacts alike,
    and for a single pole as well as three — the waveform reading cannot see a
    single-pole dead time, since the healthy phases stay energised.

    The reclose failed when the breaker opens again, or a failure channel
    asserts, shortly after; it is undetermined when the record ends first."""
    time = np.asarray(record.time, dtype=float)
    returns = []
    for ch in record.status_channels:
        name = _normalize_status_name(ch.name)
        tokens = set(re.split(r"[^A-Z0-9]+", name))
        # "CB CLOSE" is a close command pulse; "CB CLOSED" is the position.
        if (not tokens & _BREAKER_POSITION_TOKENS or "CLOSE" in tokens
                or any(word in name for word in _NOT_A_POSITION)):
            continue
        samples = np.asarray(ch.samples) != 0
        if inception_idx >= len(samples) - 1 or len(samples) != len(time):
            continue
        changes = _stable_changes(samples, time, inception_idx)
        if len(changes) < 2:
            continue
        opened, closed = changes[0], changes[1]
        if float(time[closed] - time[opened]) < _MIN_DEAD_TIME_S:
            continue
        reclose_time = float(time[closed])
        reopened = len(changes) > 2 and float(time[changes[2]]) - reclose_time < _REOPEN_WINDOW_S
        failed = any(reclose_time - 0.05 < ft < reclose_time + _REOPEN_WINDOW_S for ft in failure_times)
        if reopened or failed:
            success = False
        elif float(time[-1]) - reclose_time < _MIN_DEAD_TIME_S:
            success = None  # the record ends before the outcome shows
        else:
            success = True
        returns.append({'time': reclose_time, 'success': success, 'source': 'breaker_position'})
    # One reclose: several contacts of one breaker return within milliseconds.
    return sorted(returns, key=lambda e: e['time'])[:1]


def _detect_cb_open_window(va, vb, vc, ia, ib, ic, time, search_start_idx, dt):
    """Look for a window (>= 3 cycles, i.e. not a single noisy sample) where
    BOTH voltage AND current on all three phases sit near zero relative to
    system nominal — the direct physical signature of the breaker actually
    being open, as opposed to inferring a reclose from current shape alone.

    Thresholds are relative to nominal (not the pre-fault level): 10% of
    nominal phase voltage (from ct_primary/sqrt(3)) and 10% of the pre-fault
    load current, each measured over a 1-cycle RMS window. Requires BOTH V
    and I low together — current alone can look "low" briefly during a
    zero-crossing-adjacent clearing edge, and voltage alone can look low on
    a channel that is simply unpowered/unwired; requiring both cuts that
    ambiguity down.

    Returns (start_time_s, end_time_s, duration_ms) for the first qualifying
    window found after search_start_idx, or None if none is found. va/vb/vc
    may be None (voltage not available) — this then returns None rather than
    silently degrading, so callers must treat "no verified window" and
    "voltage unavailable" as the same "not confirmed" case.
    """
    if va is None or vb is None or vc is None:
        return None
    if ia is None or ib is None or ic is None:
        return None

    cycle_n = max(4, int(round(0.02 / dt)))  # ~1 cycle at 50 Hz
    min_run_cycles = 3
    min_run_samples = cycle_n * min_run_cycles

    def nominal_v(ch):
        primary = float(getattr(ch, "ct_primary", 0.0) or 0.0)
        return (primary / np.sqrt(3)) if primary > 0 else None

    nominal = [nominal_v(va), nominal_v(vb), nominal_v(vc)]
    if any(n is None for n in nominal):
        return None
    v_thresholds = [n * 0.10 for n in nominal]

    # Pre-fault load current as the current-side reference (before
    # search_start_idx, i.e. before clearing) — falls back to a small
    # absolute floor if the pre-fault window itself was near-zero.
    pre_len = min(search_start_idx, cycle_n * 2) if search_start_idx > 0 else 0
    i_pre_rms = [
        float(np.sqrt(np.mean(arr.samples[:pre_len] ** 2))) if pre_len > cycle_n else 0.0
        for arr in (ia, ib, ic)
    ]
    i_thresholds = [max(0.05, r * 0.10) for r in i_pre_rms]

    n = len(va.samples)
    run_start = None
    for i in range(search_start_idx, n - cycle_n, cycle_n // 2 or 1):
        v_rms = [float(np.sqrt(np.mean(ch.samples[i:i + cycle_n] ** 2))) for ch in (va, vb, vc)]
        i_rms = [float(np.sqrt(np.mean(ch.samples[i:i + cycle_n] ** 2))) for ch in (ia, ib, ic)]
        both_low = all(v_rms[k] < v_thresholds[k] for k in range(3)) and all(i_rms[k] < i_thresholds[k] for k in range(3))
        if both_low:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_run_samples:
                return (time[run_start], time[i], (time[i] - time[run_start]) * 1000.0)
            run_start = None
    if run_start is not None and (n - run_start) >= min_run_samples:
        return (time[run_start], time[n - 1], (time[n - 1] - time[run_start]) * 1000.0)
    return None


def _detect_reclose_from_waveforms(ia, ib, ic, time, clearing_idx, va=None, vb=None, vc=None):
    """
    Detect reclose events from current waveforms.

    After clearing, look for:
    1. Current returning (breaker reclose)
    2. If current returns and stays stable → successful reclose
    3. If current returns and another fault spike → failed reclose

    Every event additionally carries ``cb_open_verified`` and
    ``dead_time_ms``: True/a real duration only when a genuine V-AND-I-both-
    near-zero window (see _detect_cb_open_window) was found between clearing
    and the current returning — i.e. actual physical evidence the breaker
    was open, not just an inference from "current went away then came
    back" (which a self-clearing arcing fault, a CT/relay blind spot, or
    plain noise can also produce). ``confidence`` is lower when unverified:
    this reading should not be trusted as strongly as a status-channel-based
    reclose read (see _detect_reclose_from_status).
    """
    reclose_events = []

    if clearing_idx is None or clearing_idx >= len(ia.samples) - 100:
        return reclose_events

    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0 / 1200.0

    # Calculate post-clearing baseline
    post_clear_window = slice(clearing_idx, min(clearing_idx + 50, len(ia.samples)))
    baseline_rms_a = np.sqrt(np.mean(ia.samples[post_clear_window]**2))
    baseline_rms_b = np.sqrt(np.mean(ib.samples[post_clear_window]**2))
    baseline_rms_c = np.sqrt(np.mean(ic.samples[post_clear_window]**2))

    # Look for current returning (load current)
    for i in range(clearing_idx + 50, len(ia.samples)):
        rms_a = np.sqrt(np.mean(ia.samples[max(0, i-10):i]**2))
        rms_b = np.sqrt(np.mean(ib.samples[max(0, i-10):i]**2))
        rms_c = np.sqrt(np.mean(ic.samples[max(0, i-10):i]**2))

        # If current increased significantly from dead time
        if (rms_a > baseline_rms_a * 2 or rms_b > baseline_rms_b * 2 or rms_c > baseline_rms_c * 2):
            reclose_time = time[i]

            cb_open_window = _detect_cb_open_window(va, vb, vc, ia, ib, ic, time, clearing_idx, dt)
            cb_open_verified = cb_open_window is not None
            dead_time_ms = cb_open_window[2] if cb_open_window else None

            # Check if fault re-occurs (current spikes again)
            if i + 50 < len(ia.samples):
                future_max_a = np.max(np.abs(ia.samples[i:i+50]))
                future_max_b = np.max(np.abs(ib.samples[i:i+50]))
                future_max_c = np.max(np.abs(ic.samples[i:i+50]))

                # If future current is very high → failed reclose
                success = not (future_max_a > rms_a * 5 or future_max_b > rms_b * 5 or future_max_c > rms_c * 5)
            else:
                # Recording ends before we can see whether the fault
                # recurred. Without a verified CB-open dead-time either,
                # there is no positive evidence of ANY reclose (successful
                # or otherwise) — only that current came back, which a
                # self-clearing fault also produces. Report success as
                # unknown (None) rather than assuming True.
                success = True if cb_open_verified else None

            reclose_events.append({
                'time': reclose_time,
                'success': success,
                'cb_open_verified': cb_open_verified,
                'dead_time_ms': dead_time_ms,
                'confidence': 0.75 if cb_open_verified else 0.35,
            })
            break  # Only detect first reclose

    return reclose_events


def extract_soe(record, fault_inception_s: float = None) -> list:
    """
    Extract Sequence of Events (SOE) from COMTRADE status channels.

    Returns list of dicts sorted by timestamp:
      { time_s, rel_ms, channel, state }
    rel_ms is relative to fault_inception_s (negative = pre-fault).
    If fault_inception_s is None, relative to first event.
    """
    events = []
    for ch in record.status_channels:
        if len(ch.samples) < 2:
            continue
        transitions = np.diff(ch.samples)
        for idx in np.where(transitions > 0)[0]:
            t = float(record.time[idx + 1]) if idx + 1 < len(record.time) else float(record.time[-1])
            events.append({'time_s': t, 'channel': ch.name, 'state': 1})
        for idx in np.where(transitions < 0)[0]:
            t = float(record.time[idx + 1]) if idx + 1 < len(record.time) else float(record.time[-1])
            events.append({'time_s': t, 'channel': ch.name, 'state': 0})

    if not events:
        return []

    events.sort(key=lambda x: x['time_s'])
    ref = fault_inception_s if fault_inception_s is not None else events[0]['time_s']
    for ev in events:
        ev['rel_ms'] = round((ev['time_s'] - ref) * 1000, 2)
    return events
