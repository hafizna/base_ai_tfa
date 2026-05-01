"""Waveform checks for current-transformer and measurement anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def _cycle_rms(values: np.ndarray, start: int, end: int, cycle_n: int) -> np.ndarray:
    if end - start < cycle_n:
        return np.array([], dtype=float)

    step = max(1, cycle_n // 2)
    result = []
    for idx in range(start, end - cycle_n + 1, step):
        result.append(_rms(values[idx : idx + cycle_n]))
    return np.array(result, dtype=float)


def _score_phase(
    phase: str,
    current: np.ndarray | None,
    voltage: np.ndarray | None,
    inception_idx: int,
    clearing_idx: int,
    cycle_n: int,
) -> dict[str, Any] | None:
    if current is None or voltage is None:
        return None

    sample_count = min(len(current), len(voltage))
    if sample_count < cycle_n * 4:
        return None

    start = max(0, min(inception_idx, sample_count - 1))
    end = max(start + 1, min(clearing_idx, sample_count))
    if end - start < cycle_n * 4:
        return None

    pre_start = max(0, start - 3 * cycle_n)
    pre_current = current[pre_start:start]
    pre_voltage = voltage[pre_start:start]
    if pre_current.size < cycle_n or pre_voltage.size < cycle_n:
        return None

    fault_current_rms = _cycle_rms(current, start, end, cycle_n)
    fault_voltage_rms = _cycle_rms(voltage, start, end, cycle_n)
    if fault_current_rms.size < 4 or fault_voltage_rms.size < 4:
        return None

    current_mean = float(np.mean(fault_current_rms))
    current_cv = float(np.std(fault_current_rms) / max(current_mean, 1e-9))
    current_gain = current_mean / max(_rms(pre_current), 1.0)
    voltage_ratio = float(np.mean(fault_voltage_rms) / max(_rms(pre_voltage), 1e-9))

    suspicious = (
        current_mean >= 100.0
        and current_gain >= 2.5
        and current_cv <= 0.05
        and voltage_ratio >= 0.80
    )

    return {
        "phase": phase,
        "current_rms_a": current_mean,
        "current_gain": current_gain,
        "current_cv": current_cv,
        "voltage_ratio": voltage_ratio,
        "suspicious": suspicious,
    }


def detect_ct_measurement_anomaly(
    phase_currents: dict[str, np.ndarray | None],
    phase_voltages: dict[str, np.ndarray | None],
    sampling_rate_hz: float,
    system_freq_hz: float,
    inception_idx: int,
    clearing_idx: int,
    duration_ms: float,
) -> dict[str, Any]:
    """Detect sustained flat current on a phase whose voltage remains healthy."""
    if duration_ms < 180 or sampling_rate_hz <= 0 or system_freq_hz <= 0:
        return {"detected": False, "phase": "", "evidence": "", "phases": []}

    cycle_n = max(4, int(round(sampling_rate_hz / system_freq_hz)))
    phase_scores = []
    for phase in ("A", "B", "C"):
        score = _score_phase(
            phase,
            phase_currents.get(phase),
            phase_voltages.get(phase),
            inception_idx,
            clearing_idx,
            cycle_n,
        )
        if score is not None:
            phase_scores.append(score)

    suspicious = [score for score in phase_scores if score["suspicious"]]
    if not suspicious:
        return {"detected": False, "phase": "", "evidence": "", "phases": phase_scores}

    best = max(suspicious, key=lambda item: item["current_gain"])
    evidence = (
        f"arus fase {best['phase']} naik stabil "
        f"({best['current_rms_a']:.0f} A RMS, gain {best['current_gain']:.1f}x, "
        f"CV {best['current_cv']:.3f}) sementara tegangan fase tersebut tetap "
        f"{best['voltage_ratio']:.2f} pu"
    )
    return {
        "detected": True,
        "phase": best["phase"],
        "evidence": evidence,
        "phases": phase_scores,
    }


def detect_ct_measurement_anomaly_record(record: Any, fault: Any) -> dict[str, Any]:
    if record is None or fault is None or len(getattr(record, "time", [])) < 4:
        return {"detected": False, "phase": "", "evidence": "", "phases": []}

    time = np.asarray(record.time, dtype=float)
    sampling_rate = 1.0 / float(np.median(np.diff(time)))
    currents: dict[str, np.ndarray | None] = {"A": None, "B": None, "C": None}
    voltages: dict[str, np.ndarray | None] = {"A": None, "B": None, "C": None}
    for channel in record.analog_channels:
        phase = getattr(channel, "phase", None)
        if phase not in currents:
            continue
        samples = np.asarray(channel.samples, dtype=float)
        if getattr(channel, "measurement", "") == "current":
            currents[phase] = samples
        elif getattr(channel, "measurement", "") == "voltage":
            voltages[phase] = samples

    return detect_ct_measurement_anomaly(
        currents,
        voltages,
        sampling_rate,
        float(getattr(record, "frequency", 50.0) or 50.0),
        int(getattr(fault, "inception_idx", 0) or 0),
        int(getattr(fault, "clearing_idx", len(time) - 1) or len(time) - 1),
        float(getattr(fault, "duration_ms", 0.0) or 0.0),
    )
