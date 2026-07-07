"""No-fault gate + symmetrical-component regression tests.

A DFR record triggered by a fault-detector pickup (FD.Pkp / FD.DPFC.Pkp) that
resets itself — no protection operate, no current step, no sag, balanced system —
must NOT be classified as a fault. Previously the pipeline always emitted a cause
label and fabricated an impedance locus from load V/I.
"""

import numpy as np

from webapp.api.ml_predict import (
    extract_ml_features,
    run_ml_prediction,
    _no_fault_gate,
    _symmetrical_components,
)
from webapp.api.fault_detection import detect_fault_presence


def _balanced_load_payload(freq=50.0, sr=1200.0, dur_s=1.0, i_amp=3.7, v_amp=80.0):
    """Steady balanced three-phase load: no fault anywhere in the record."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    w = 2 * np.pi * freq

    def ch(canon, name, amp, ph_deg, unit, meas):
        sig = amp * np.sin(w * t + np.deg2rad(ph_deg))
        return {
            "id": canon, "name": name, "canonical_name": canon,
            "unit": unit, "phase": canon[-1], "measurement": meas,
            "samples": sig.tolist(),
        }

    analog = [
        ch("IA", "Prot.CB1.Ia", i_amp, 0, "A", "current"),
        ch("IB", "Prot.CB1.Ib", i_amp, -120, "A", "current"),
        ch("IC", "Prot.CB1.Ic", i_amp, 120, "A", "current"),
        ch("VA", "Prot.Ua", v_amp, 0, "kV", "voltage"),
        ch("VB", "Prot.Ub", v_amp, -120, "kV", "voltage"),
        ch("VC", "Prot.Uc", v_amp, 120, "kV", "voltage"),
    ]
    # A self-resetting fault-detector pickup — a standing/pickup bit, NOT an operate.
    fd = np.zeros(n, dtype=int)
    fd[int(0.5 * sr): int(0.5 * sr) + 7] = 1
    status = [
        {"id": "3", "name": "FD.Pkp", "samples": fd.tolist()},
        {"id": "4", "name": "FD.DPFC.Pkp", "samples": fd.tolist()},
        {"id": "14", "name": "87L.On", "samples": np.ones(n, dtype=int).tolist()},
    ]
    return {"frequency": freq, "time": t.tolist(),
            "analog_channels": analog, "status_channels": status}


def test_balanced_load_triggers_no_fault_gate():
    payload = _balanced_load_payload()
    assert _no_fault_gate(payload) is not None, "balanced load must hit the no-fault gate"

    res = run_ml_prediction(payload, "87L")
    assert res["no_fault"] is True
    assert res["fault_type"] == "none"
    assert res["overall_confidence"] == 0.0
    assert res["cause_ranking"] == []


def test_sync_fail_with_minor_analog_blip_is_no_fault():
    """SYNC FAIL only, no protection operate, and no sustained fault current."""
    payload = _balanced_load_payload(sr=1000.0, dur_s=1.4, i_amp=250.0, v_amp=80.0)
    t = np.asarray(payload["time"])
    event = (t >= 0.85) & (t <= 0.912)

    # Mild analog disturbance: enough to look tempting to the old OR gate
    # (sag > 10%), but not a real relay fault because current does not step
    # up in a sustained way and SOE has only sync failure.
    for ch in payload["analog_channels"]:
        if ch["measurement"] == "voltage":
            arr = np.asarray(ch["samples"], dtype=float)
            arr[event] *= 0.88
            ch["samples"] = arr.tolist()
        if ch["canonical_name"] == "IA":
            arr = np.asarray(ch["samples"], dtype=float)
            arr[event] *= 1.15
            ch["samples"] = arr.tolist()

    sync = np.zeros(len(t), dtype=int)
    sync[(t >= 1.0) & (t <= 1.209)] = 1
    payload["status_channels"] = [
        {"id": "sync", "name": "SYNC FAIL", "samples": sync.tolist()},
    ]

    det = detect_fault_presence(payload)
    assert det.no_fault, det.reasons
    assert _no_fault_gate(payload) is not None

    res = run_ml_prediction(payload, "21")
    assert res["no_fault"] is True
    assert res["fault_type"] == "none"


def test_symmetrical_components_balanced_set_is_positive_sequence_only():
    """Balanced ABC sinusoids → I1 dominant, I2 and I0 ≈ 0 (the old bug gave I1≈I2)."""
    sr, freq, amp = 1200.0, 50.0, 5.0
    n = int(sr / freq)  # exactly one cycle
    t = np.arange(n) / sr
    w = 2 * np.pi * freq
    ia = amp * np.sin(w * t)
    ib = amp * np.sin(w * t - 2 * np.pi / 3)
    ic = amp * np.sin(w * t + 2 * np.pi / 3)
    i0, i1, i2 = _symmetrical_components(ia, ib, ic)
    assert i1 > 0.9 * amp
    assert i2 < 0.05 * i1
    assert i0 < 0.05 * i1


def test_real_operate_bit_bypasses_gate():
    """If a protection .Op channel asserts, the gate must let the record through."""
    payload = _balanced_load_payload()
    n = len(payload["time"])
    op = np.zeros(n, dtype=int)
    op[int(0.5 * 1200):] = 1
    payload["status_channels"].append({"id": "17", "name": "87L.Op", "samples": op.tolist()})
    row = extract_ml_features(payload, "87L")
    assert row["protection_operated"] is True
    assert _no_fault_gate(payload) is None
