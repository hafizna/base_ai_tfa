"""Shared no-fault gate — single source of truth for "is there a fault?".

A DFR record can be triggered by a fault-detector / superimposed-component
pickup (FD.Pkp / FD.DPFC.Pkp) that resets itself without any protection
operating. On such records there is no current step, no voltage sag, the system
stays balanced, and no .Op/.Trip/CB-open bit asserts. Classifying that as a
fault — or computing an impedance locus from load V/I — is meaningless.

This module operates directly on the stored COMTRADE *payload* (the dict shape
produced by the upload router), so every consumer (relay-21, relay-87L, the AI
classifier, electrical params, impedance locus, and the PDF report) shares one
definition of "no fault" regardless of which feature extractor it otherwise uses.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Thresholds are unit-free (ratios), so they hold for primary OR secondary
# scaling. A genuine fault shows at least one of these.
CURRENT_STEP_RATIO = 2.0    # peak / prefault-RMS on the most-active phase
VOLTAGE_SAG_PU = 0.10       # fractional drop in faulted-phase RMS
I0_I1_RATIO = 0.20          # zero-sequence unbalance
I2_I1_RATIO = 0.15          # negative-sequence unbalance


@dataclass
class FaultDetection:
    """Result of the shared no-fault gate."""
    is_fault: bool
    reasons: list[str] = field(default_factory=list)   # why we decided fault / no-fault
    peak_to_prefault_ratio: float = 0.0
    voltage_sag_pu: float = 0.0
    i0_i1_ratio: float = 0.0
    i2_i1_ratio: float = 0.0
    protection_operated: bool = False

    @property
    def no_fault(self) -> bool:
        return not self.is_fault


def _find_ch(channels: list, candidates: set[str]) -> Optional[np.ndarray]:
    for ch in channels:
        canon = (ch.get("canonical_name") or "").upper()
        name = (ch.get("name") or "").upper()
        if canon in candidates or name in candidates:
            return np.asarray(ch.get("samples") or [], dtype=float)
    return None


def _fundamental_phasor(seg: np.ndarray) -> complex:
    n = len(seg)
    if n < 2:
        return 0j
    k = np.arange(n)
    return complex(2.0 / n * np.dot(seg.astype(float), np.exp(-2j * np.pi * k / n)))


def _is_operate_status(name: str) -> bool:
    """True for a protection operate/trip bit, excluding standing-status bits.

    Standing bits (.On/.Valid/.Ready/_OK/Pkp/Alarm/etc.) are HIGH in normal
    service or are mere pickups that reset themselves; they must not count as
    "protection operated".
    """
    import re
    nm = name.upper()
    is_operate = (
        re.search(r"(?:^|[._\s])OP(?:[._\s]|$)", nm) is not None
        or "TRIP" in nm
        or "OP_PROT" in nm.replace(" ", "")
    )
    is_standing = any(
        tok in nm for tok in (".ON", ".OFF", ".VALID", ".READY", ".BLOCKED",
                              "_OK", "PKP", "PICKUP", "PICK UP", "ALM", "ALARM",
                              "MODE", "SWITCH", "HEALTHY", "INPROG", "ACTIVE")
    )
    return is_operate and not is_standing


def detect_fault_presence(payload: dict) -> FaultDetection:
    """Decide whether a COMTRADE payload actually contains a fault.

    Named distinctly from ``core.fault_detector.detect_fault`` (the CLI/training
    fault-event extractor) to avoid confusion: this one is the webapp no-fault
    gate and takes the stored COMTRADE *payload* dict.

    Conservative: ANY hard evidence (protection operated, current step, voltage
    sag, or sequence unbalance) ⇒ is_fault=True. Only when none is present do we
    return no-fault.
    """
    channels = payload.get("analog_channels", [])
    status = payload.get("status_channels", [])
    time = np.asarray(payload.get("time", []), dtype=float)
    freq = float(payload.get("frequency", 50.0) or 50.0)

    if len(time) < 4:
        # Too little data to judge — treat as fault to avoid silently hiding it.
        return FaultDetection(is_fault=True, reasons=["data terlalu pendek untuk evaluasi"])

    sr = 1.0 / (time[1] - time[0]) if time[1] > time[0] else 1000.0
    cycle_n = max(4, int(sr / freq))

    ia = _find_ch(channels, {"IA", "IL1", "I1"})
    ib = _find_ch(channels, {"IB", "IL2", "I2"})
    ic = _find_ch(channels, {"IC", "IL3", "I3"})
    va = _find_ch(channels, {"VA", "VAN", "UA"})
    vb = _find_ch(channels, {"VB", "VBN", "UB"})
    vc = _find_ch(channels, {"VC", "VCN", "UC"})

    reasons: list[str] = []

    # 1) Protection operated?
    protection_operated = any(
        _is_operate_status(str(sch.get("name", "") or ""))
        and any(int(v) == 1 for v in (sch.get("samples") or []))
        for sch in status
    )
    if protection_operated:
        reasons.append("kanal proteksi beroperasi (TRIP/Op)")

    # 2) Current step on the most-active phase.
    peak_ratio = 0.0
    currents = [arr for arr in (ia, ib, ic) if arr is not None and len(arr) >= 4]
    if currents:
        def step(arr: np.ndarray) -> float:
            pre_n = min(2 * cycle_n, len(arr) // 4)
            pre = float(np.sqrt(np.mean(arr[:pre_n] ** 2))) if pre_n > 1 else 0.0
            return float(np.max(np.abs(arr)) / pre) if pre > 0 else 0.0
        peak_ratio = max(step(arr) for arr in currents)
    has_current_step = peak_ratio >= CURRENT_STEP_RATIO
    if has_current_step:
        reasons.append(f"lonjakan arus {peak_ratio:.2f}× prefault")

    # 3) Voltage sag (deepest phase).
    sag_pu = 0.0
    v_channels = [v for v in (va, vb, vc) if v is not None]
    if v_channels:
        pre_rms, fault_rms = [], []
        for v in v_channels:
            pre_n = min(2 * cycle_n, len(v) // 4)
            if pre_n > 1:
                pre_rms.append(float(np.sqrt(np.mean(v[:pre_n] ** 2))))
            # scan minimum 1-cycle RMS across the record
            mins = [float(np.sqrt(np.mean(v[i:i + cycle_n] ** 2)))
                    for i in range(0, max(1, len(v) - cycle_n), cycle_n)]
            if mins:
                fault_rms.append(min(mins))
        if pre_rms and fault_rms:
            pre_mean = float(np.mean(pre_rms))
            if pre_mean > 0:
                sag_pu = max(0.0, (pre_mean - min(fault_rms)) / pre_mean)
    has_sag = sag_pu >= VOLTAGE_SAG_PU
    if has_sag:
        reasons.append(f"voltage sag {sag_pu * 100:.1f}%")

    # 4) Sequence unbalance (fundamental phasors over the most-active window).
    i0_i1 = i2_i1 = 0.0
    if ia is not None and ib is not None and ic is not None:
        n = min(cycle_n, len(ia), len(ib), len(ic))
        if n >= 4:
            # window around the largest |i| sample
            ref = max((ia, ib, ic), key=lambda a: float(np.max(np.abs(a))))
            c = int(np.argmax(np.abs(ref)))
            s = max(0, min(c - n // 2, len(ref) - n))
            a = np.exp(1j * 2 * np.pi / 3)
            pa = _fundamental_phasor(ia[s:s + n])
            pb = _fundamental_phasor(ib[s:s + n])
            pc = _fundamental_phasor(ic[s:s + n])
            i0 = abs((pa + pb + pc) / 3.0)
            i1 = abs((pa + a * pb + a ** 2 * pc) / 3.0)
            i2 = abs((pa + a ** 2 * pb + a * pc) / 3.0)
            if i1 > 0:
                i0_i1 = i0 / i1
                i2_i1 = i2 / i1
    has_unbalance = i0_i1 > I0_I1_RATIO or i2_i1 > I2_I1_RATIO
    if has_unbalance:
        reasons.append(f"ketidakseimbangan urutan (I0/I1 {i0_i1 * 100:.1f}%, I2/I1 {i2_i1 * 100:.1f}%)")

    is_fault = protection_operated or has_current_step or has_sag or has_unbalance
    if not is_fault:
        reasons = [
            "tidak ada proteksi beroperasi",
            f"arus tetap pada level beban (rasio puncak/prefault {peak_ratio:.2f}×)",
            f"tidak ada voltage sag berarti ({sag_pu * 100:.1f}%)",
            f"sistem seimbang (I0/I1 {i0_i1 * 100:.1f}%, I2/I1 {i2_i1 * 100:.1f}%)",
        ]

    return FaultDetection(
        is_fault=is_fault,
        reasons=reasons,
        peak_to_prefault_ratio=round(peak_ratio, 2),
        voltage_sag_pu=round(sag_pu, 3),
        i0_i1_ratio=round(i0_i1, 3),
        i2_i1_ratio=round(i2_i1, 3),
        protection_operated=protection_operated,
    )
