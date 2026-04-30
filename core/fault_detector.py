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

    # Detect dead-time recordings: CB was already open when recording started.
    # In this case the fault occurred before this recording — no fault current present.
    # Skip straight to waveform reclose detection; fault duration is not measurable.
    if _recording_starts_in_dead_time(record):
        logger.debug("Recording started in CB dead time — fault preceded this file")
        return _build_dead_time_event(record)

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


def _recording_starts_in_dead_time(record) -> bool:
    """
    Returns True if the CB was already open when the recording started.
    This happens when an external DFR is triggered by the open-CB signal
    rather than by the fault itself — the fault is not captured in this file.

    Indicators: a CB-open / pole-dead / 52b channel that is HIGH (=1) from
    the very first sample and has no rising edge (only a falling edge later
    when the breaker recloses).
    """
    CB_OPEN_KW = ['CB OPEN', 'POLE DEAD', 'ANY POLE', 'ALL POLE', '52B', 'CB1.52B']
    EXCL_KW    = ['ALARM', 'TEST', 'BLOCK']

    for ch in record.status_channels:
        nu = _normalize_status_name(ch.name)
        if any(e in nu for e in EXCL_KW):
            continue
        if not any(k in nu for k in CB_OPEN_KW):
            continue
        if len(ch.samples) < 10:
            continue
        # High from start AND has at least one falling edge (CB eventually reclosed)
        if ch.samples[0] == 1 and ch.samples[:5].sum() == 5:
            diff = np.diff(ch.samples)
            if (diff < 0).any():  # has falling edge = CB reclosed
                return True
    return False


def _build_dead_time_event(record) -> Optional[FaultEvent]:
    """
    Build a minimal FaultEvent for a dead-time recording.
    Duration is unknown (fault happened before this file).
    Reclose time is taken from when CB-open signal drops.
    """
    CB_OPEN_KW = ['CB OPEN', 'POLE DEAD', 'ANY POLE', 'ALL POLE', '52B', 'CB1.52B']
    EXCL_KW    = ['ALARM', 'TEST', 'BLOCK']

    reclose_time = None
    for ch in record.status_channels:
        nu = _normalize_status_name(ch.name)
        if any(e in nu for e in EXCL_KW):
            continue
        if not any(k in nu for k in CB_OPEN_KW):
            continue
        diff = np.diff(ch.samples)
        falls = np.where(diff < 0)[0]
        if len(falls):
            t = record.time[falls[0] + 1] if falls[0] + 1 < len(record.time) else record.time[-1]
            if reclose_time is None or t < reclose_time:
                reclose_time = t

    # Also check AR success channel
    AR_SUCCESS_KW = ['AR SUCC', 'SUCC_RCLS', 'RECLOSE SUCC', '.79.SUCC']
    for ch in record.status_channels:
        nu = _normalize_status_name(ch.name)
        if any(k in nu for k in AR_SUCCESS_KW) and ch.samples.sum() > 0:
            diff = np.diff(ch.samples)
            rises = np.where(diff > 0)[0]
            if len(rises):
                t = record.time[rises[0] + 1]
                if reclose_time is None or t < reclose_time:
                    reclose_time = t

    # Use t=0 as nominal inception (fault was before recording)
    inception_time = record.time[0]

    # Reclose successful if CB-open dropped (CB closed again)
    reclose_events = []
    if reclose_time is not None:
        reclose_events = [{'time': reclose_time, 'success': True}]

    return FaultEvent(
        inception_idx=0,
        inception_time=inception_time,
        clearing_idx=None,
        clearing_time=None,
        duration_ms=0.0,        # genuinely unknown — fault not in this recording
        detection_method="dead_time_recording",
        confidence=0.6,
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

    # Detect reclose events
    reclose_events = _detect_reclose_from_waveforms(ia, ib, ic, record.time, clearing_idx) if clearing_idx else []

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

    return reclose_events


def _detect_reclose_from_waveforms(ia, ib, ic, time, clearing_idx):
    """
    Detect reclose events from current waveforms.

    After clearing, look for:
    1. Current returning (breaker reclose)
    2. If current returns and stays stable → successful reclose
    3. If current returns and another fault spike → failed reclose
    """
    reclose_events = []

    if clearing_idx is None or clearing_idx >= len(ia.samples) - 100:
        return reclose_events

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

            # Check if fault re-occurs (current spikes again)
            if i + 50 < len(ia.samples):
                future_max_a = np.max(np.abs(ia.samples[i:i+50]))
                future_max_b = np.max(np.abs(ib.samples[i:i+50]))
                future_max_c = np.max(np.abs(ic.samples[i:i+50]))

                # If future current is very high → failed reclose
                success = not (future_max_a > rms_a * 5 or future_max_b > rms_b * 5 or future_max_c > rms_c * 5)
            else:
                success = True  # Assume successful if recording ends soon after

            reclose_events.append({'time': reclose_time, 'success': success})
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
