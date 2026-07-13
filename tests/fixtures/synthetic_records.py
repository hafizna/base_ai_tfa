"""Synthetic COMTRADE-payload builders for Stage 0 golden regression cases.

These build the same JSON payload dict shape produced by
``webapp/api/routers/upload.py::_record_to_out`` (analog_channels /
status_channels / time / frequency / ...), so they can be fed directly into
``core.event_analysis.build_event_window``, ``webapp.api.record_analysis``,
and the relay_21 router functions without parsing real COMTRADE files.

Follows the synthetic-payload pattern already used in
``tests/test_no_fault_gate.py`` (``_balanced_load_payload``).
"""

from __future__ import annotations

import numpy as np


def _analog_channel(canon: str, name: str, samples: np.ndarray, unit: str, measurement: str) -> dict:
    return {
        "id": canon,
        "name": name,
        "canonical_name": canon,
        "unit": unit,
        "phase": canon[-1] if canon[-1] in "ABC" else None,
        "measurement": measurement,
        "ct_primary": 1.0,
        "ct_secondary": 1.0,
        "pors": "P",
        "samples": samples.tolist(),
    }


def _status_channel(channel_id: str, name: str, samples: np.ndarray) -> dict:
    return {"id": channel_id, "name": name, "samples": samples.astype(int).tolist()}


def _sine(amp: float, freq: float, phase_deg: float, t: np.ndarray) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * t + np.deg2rad(phase_deg))


def _base_payload(
    t: np.ndarray,
    freq: float,
    analog: list[dict],
    status: list[dict],
    trigger_offset_s: float = 0.0,
) -> dict:
    return {
        "station_name": "GOLDEN TEST",
        "rec_dev_id": "SYNTH",
        "rev_year": "2013",
        "sampling_rates": [[float(1.0 / (t[1] - t[0])) if len(t) > 1 else 1000.0, len(t)]],
        "trigger_time": trigger_offset_s,
        "start_time_iso": None,
        "trigger_time_iso": None,
        "trigger_offset_s": trigger_offset_s,
        "time_code": None,
        "local_code": None,
        "clock_quality": None,
        "total_samples": len(t),
        "frequency": freq,
        "time": t.tolist(),
        "analog_channels": analog,
        "status_channels": status,
        "warnings": [],
    }


def no_fault_trigger(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.0) -> dict:
    """Case 1: balanced load, self-resetting FD pickup, no protection operate."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    ia = _sine(3.7, freq, 0, t)
    ib = _sine(3.7, freq, -120, t)
    ic = _sine(3.7, freq, 120, t)
    va = _sine(80.0, freq, 0, t)
    vb = _sine(80.0, freq, -120, t)
    vc = _sine(80.0, freq, 120, t)
    analog = [
        _analog_channel("IA", "Ia", ia, "A", "current"),
        _analog_channel("IB", "Ib", ib, "A", "current"),
        _analog_channel("IC", "Ic", ic, "A", "current"),
        _analog_channel("VA", "Ua", va, "kV", "voltage"),
        _analog_channel("VB", "Ub", vb, "kV", "voltage"),
        _analog_channel("VC", "Uc", vc, "kV", "voltage"),
    ]
    fd = np.zeros(n, dtype=int)
    fd[int(0.5 * sr): int(0.5 * sr) + 7] = 1
    status = [_status_channel("fd", "FD.Pkp", fd)]
    return _base_payload(t, freq, analog, status)


def transient_slg_successful_reclose(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.5) -> dict:
    """Case 2: transient single-phase-to-ground fault, breaker recloses OK."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    fault_idx = int(0.3 * sr)
    clear_idx = fault_idx + int(0.10 * sr)  # 100 ms fault
    reclose_idx = clear_idx + int(0.4 * sr)  # 400 ms dead time

    def phase_a_current():
        base = _sine(3.7, freq, 0, t)
        base[fault_idx:clear_idx] = _sine(45.0, freq, 0, t)[fault_idx:clear_idx]
        base[clear_idx:reclose_idx] = 0.0
        base[reclose_idx:] = _sine(3.7, freq, 0, t)[reclose_idx:]
        return base

    ia = phase_a_current()
    ib = _sine(3.7, freq, -120, t)
    ic = _sine(3.7, freq, 120, t)
    va = _sine(80.0, freq, 0, t)
    va[fault_idx:clear_idx] *= 0.15  # deep sag on faulted phase
    vb = _sine(80.0, freq, -120, t)
    vc = _sine(80.0, freq, 120, t)

    analog = [
        _analog_channel("IA", "Ia", ia, "A", "current"),
        _analog_channel("IB", "Ib", ib, "A", "current"),
        _analog_channel("IC", "Ic", ic, "A", "current"),
        _analog_channel("VA", "Ua", va, "kV", "voltage"),
        _analog_channel("VB", "Ub", vb, "kV", "voltage"),
        _analog_channel("VC", "Uc", vc, "kV", "voltage"),
    ]

    trip = np.zeros(n, dtype=int)
    trip[fault_idx + 15: clear_idx + 15] = 1
    ar_attempt = np.zeros(n, dtype=int)
    ar_attempt[fault_idx + 15: reclose_idx] = 1
    ar_succ = np.zeros(n, dtype=int)
    ar_succ[reclose_idx: reclose_idx + 20] = 1

    status = [
        _status_channel("trip", "21Z1.Trip A", trip),
        _status_channel("arattempt", "AR 1POLE", ar_attempt),
        _status_channel("arsucc", "AR SUCC", ar_succ),
    ]
    return _base_payload(t, freq, analog, status, trigger_offset_s=float(t[fault_idx]))


def permanent_fault_failed_reclose(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.5) -> dict:
    """Case 3: permanent fault — reclose attempted but fault recurs (failed)."""
    payload = transient_slg_successful_reclose(sr, freq, dur_s)
    t = np.asarray(payload["time"])
    n = len(t)
    reclose_idx = int(0.3 * sr) + int(0.10 * sr) + int(0.4 * sr)

    # Re-fault after reclose: current spikes again and stays high to end of record.
    for ch in payload["analog_channels"]:
        if ch["canonical_name"] == "IA":
            arr = np.asarray(ch["samples"], dtype=float)
            arr[reclose_idx:] = _sine(50.0, freq, 0, t)[reclose_idx:]
            ch["samples"] = arr.tolist()
        if ch["canonical_name"] == "VA":
            arr = np.asarray(ch["samples"], dtype=float)
            arr[reclose_idx:] *= 0.15
            ch["samples"] = arr.tolist()

    for sch in payload["status_channels"]:
        if sch["name"] == "21Z1.Trip A":
            samples = np.asarray(sch["samples"], dtype=int)
            samples[reclose_idx + 15:] = 1
            sch["samples"] = samples.tolist()
        if sch["name"] == "AR SUCC":
            # Fault recurs right after reclose — no genuine success evidence.
            sch["samples"] = np.zeros(n, dtype=int).tolist()

    payload["status_channels"].append(
        _status_channel("arfail", "AR LOCKOUT", (np.arange(n) >= reclose_idx + 40).astype(int))
    )
    return payload


def status_trip_lags_waveform(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.0) -> dict:
    """Case 4: status trip bit asserts noticeably later than the waveform onset."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    fault_idx = int(0.2 * sr)
    status_delay_samples = int(0.03 * sr)  # 30 ms — over the 1-cycle "prefer waveform" threshold

    ia = _sine(3.7, freq, 0, t)
    ia[fault_idx:] = _sine(40.0, freq, 0, t)[fault_idx:]
    ib = _sine(3.7, freq, -120, t)
    ic = _sine(3.7, freq, 120, t)
    va = _sine(80.0, freq, 0, t)
    va[fault_idx:] *= 0.2
    vb = _sine(80.0, freq, -120, t)
    vc = _sine(80.0, freq, 120, t)

    analog = [
        _analog_channel("IA", "Ia", ia, "A", "current"),
        _analog_channel("IB", "Ib", ib, "A", "current"),
        _analog_channel("IC", "Ic", ic, "A", "current"),
        _analog_channel("VA", "Ua", va, "kV", "voltage"),
        _analog_channel("VB", "Ub", vb, "kV", "voltage"),
        _analog_channel("VC", "Uc", vc, "kV", "voltage"),
    ]
    trip = np.zeros(n, dtype=int)
    trip[fault_idx + status_delay_samples:] = 1
    status = [_status_channel("trip", "21Z1.Trip A", trip)]
    return _base_payload(t, freq, analog, status)


def recording_starts_in_dead_time(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.0) -> dict:
    """Case 5: recording starts with CB already open (fault preceded this file)."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    reclose_idx = int(0.6 * sr)

    ia = np.zeros(n)
    ia[reclose_idx:] = _sine(3.7, freq, 0, t)[reclose_idx:]
    ib = np.zeros(n)
    ib[reclose_idx:] = _sine(3.7, freq, -120, t)[reclose_idx:]
    ic = np.zeros(n)
    ic[reclose_idx:] = _sine(3.7, freq, 120, t)[reclose_idx:]
    va = _sine(80.0, freq, 0, t)
    vb = _sine(80.0, freq, -120, t)
    vc = _sine(80.0, freq, 120, t)

    analog = [
        _analog_channel("IA", "Ia", ia, "A", "current"),
        _analog_channel("IB", "Ib", ib, "A", "current"),
        _analog_channel("IC", "Ic", ic, "A", "current"),
        _analog_channel("VA", "Ua", va, "kV", "voltage"),
        _analog_channel("VB", "Ub", vb, "kV", "voltage"),
        _analog_channel("VC", "Uc", vc, "kV", "voltage"),
    ]
    cb_open = np.ones(n, dtype=int)
    cb_open[reclose_idx:] = 0
    status = [_status_channel("cbopen", "CB1.52B", cb_open)]
    return _base_payload(t, freq, analog, status)


def current_only_recording(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.0) -> dict:
    """Case 6: current-only recording, no voltage channels."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    fault_idx = int(0.3 * sr)
    ia = _sine(3.7, freq, 0, t)
    ia[fault_idx:] = _sine(35.0, freq, 0, t)[fault_idx:]
    ib = _sine(3.7, freq, -120, t)
    ic = _sine(3.7, freq, 120, t)
    analog = [
        _analog_channel("IA", "Ia", ia, "A", "current"),
        _analog_channel("IB", "Ib", ib, "A", "current"),
        _analog_channel("IC", "Ic", ic, "A", "current"),
    ]
    trip = np.zeros(n, dtype=int)
    trip[fault_idx + 10:] = 1
    status = [_status_channel("trip", "21Z1.Trip A", trip)]
    return _base_payload(t, freq, analog, status)


def missing_ambiguous_digital(sr: float = 1200.0, freq: float = 50.0, dur_s: float = 1.0) -> dict:
    """Case 7: fault present in waveform, but no digital/status channels at all."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    fault_idx = int(0.25 * sr)
    ia = _sine(3.7, freq, 0, t)
    ia[fault_idx:] = _sine(38.0, freq, 0, t)[fault_idx:]
    ib = _sine(3.7, freq, -120, t)
    ic = _sine(3.7, freq, 120, t)
    va = _sine(80.0, freq, 0, t)
    va[fault_idx:] *= 0.2
    vb = _sine(80.0, freq, -120, t)
    vc = _sine(80.0, freq, 120, t)
    analog = [
        _analog_channel("IA", "Ia", ia, "A", "current"),
        _analog_channel("IB", "Ib", ib, "A", "current"),
        _analog_channel("IC", "Ic", ic, "A", "current"),
        _analog_channel("VA", "Ua", va, "kV", "voltage"),
        _analog_channel("VB", "Ub", vb, "kV", "voltage"),
        _analog_channel("VC", "Uc", vc, "kV", "voltage"),
    ]
    return _base_payload(t, freq, analog, [])
