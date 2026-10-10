"""Measured record facts used by the incident story (webapp/api/record_facts.py)."""

import math

import numpy as np
import pytest

from core.event_analysis import EventWindow
from webapp.api.record_facts import classify_status_channel, electrical_measurements, protection_operations

FS = 1000.0  # samples per second -> 20 samples per 50 Hz cycle
N = 20


@pytest.mark.parametrize(
    "name, expected",
    [
        # Cibatu–Mekarsari 2 (distance relay with teleprotection)
        ("DIST Trip A", ("trip", None, "A")),
        ("Z2", ("zone", 2, None)),
        ("Z4", ("zone", 4, None)),
        ("DIST UNB CR", ("teleprotection", None, None)),
        ("DIST. Chan Recv", ("teleprotection", None, None)),
        ("DIST Sig. Send", ("teleprotection", None, None)),
        ("A/R 1P In Prog", ("reclose", None, None)),
        ("CB Aux A", ("breaker", None, "A")),
        ("Any Pole Dead", ("breaker", None, None)),
        ("SOTF/TOR Trip", ("trip", None, None)),
        ("Power Swing", (None, None, None)),
        ("MCB VT FAIL", (None, None, None)),
        ("Check Synch. OK", (None, None, None)),
        ("Unused", (None, None, None)),
        # Bringin external DFR, two lines in one file
        ("TRIP R MJSNG2", ("trip", None, "A")),
        ("TRIP T MJSNG2", ("trip", None, "C")),
        ("TRIP Z1 MJSNG2", ("trip", 1, None)),
        ("CB OPEN MJSNG2", ("breaker", None, None)),
        ("AR OPRT MJSNG2", ("reclose", None, None)),
        ("LP RCV MJSNG2", ("teleprotection", None, None)),
        ("PRES ALRM MJSNG2", (None, None, None)),
        # Mojosongo Qualitrol
        ("DIST RECEIVE BRINGIN 2", ("teleprotection", None, None)),
        # Generic forms
        ("TRIP 3P", ("trip", None, "3P")),
        ("ZONE 3 START", ("zone", 3, None)),
        ("21 OPERATE", ("trip", None, None)),
        ("67N", ("protection", None, None)),
        ("79 IN PROGRESS", ("reclose", None, None)),
        # RWALO-PLTU #1: position contacts and the AR close command
        ("CB Closed C ph", ("breaker", None, "C")),
        ("L3 Status 52A T", ("breaker", None, "C")),
        ("Auto Close", ("reclose", None, None)),
        ("L5 CB Healthy", (None, None, None)),
    ],
)
def test_classify_status_channel(name, expected):
    assert classify_status_channel(name) == expected


def _status(name, on_spans, length):
    samples = [0] * length
    for start, stop in on_spans:
        for i in range(start, min(stop, length)):
            samples[i] = 1
    return {"name": name, "samples": samples}


def test_protection_operations_reports_edges_in_record_ms_and_orders_by_first_assert():
    length = 500
    payload = {
        "time": [i / FS for i in range(length)],
        "status_channels": [
            _status("CB Aux A", [(180, 400)], length),
            _status("DIST Trip A", [(130, 190)], length),
            _status("Z2", [(120, 190)], length),
            _status("CB OPEN", [(0, 50)], length),  # open at start, closes at 50
            _status("PRES ALRM", [(10, 20)], length),  # alarm: ignored
            _status("DIST Sig. Send", [], length),  # never asserted: omitted
        ],
    }
    ops = protection_operations(payload)
    assert [op["name"] for op in ops] == ["CB OPEN", "Z2", "DIST Trip A", "CB Aux A"]
    by_name = {op["name"]: op for op in ops}
    assert by_name["CB OPEN"]["initially_on"] is True
    assert by_name["CB OPEN"]["on_ms"] == [] and by_name["CB OPEN"]["off_ms"] == [50.0]
    assert by_name["Z2"]["on_ms"] == [120.0] and by_name["Z2"]["off_ms"] == [190.0]
    assert by_name["DIST Trip A"]["role"] == "trip" and by_name["DIST Trip A"]["phase"] == "A"


def test_protection_operations_caps_edges_per_channel():
    length = 200
    spans = [(10 * k, 10 * k + 3) for k in range(1, 10)]  # nine bounces
    payload = {"time": [i / FS for i in range(length)], "status_channels": [_status("CB OPEN", spans, length)]}
    (op,) = protection_operations(payload)
    assert len(op["on_ms"]) == 4 and len(op["off_ms"]) == 4


def _sine(rms, length, phase_deg=0.0):
    t = np.arange(length) / FS
    return rms * math.sqrt(2) * np.sin(2 * np.pi * 50 * t + np.radians(phase_deg))


def _window(inception_idx, clearing_idx, method="status_waveform_aligned", reclose_events=None):
    return EventWindow(
        record_start_ms=0.0,
        trigger_time_ms=None,
        inception_idx=inception_idx,
        inception_time_ms=inception_idx,
        clearing_idx=clearing_idx,
        clearing_time_ms=clearing_idx,
        fault_duration_ms=None if clearing_idx is None else clearing_idx - inception_idx,
        method=method,
        confidence=0.9,
        faulted_phases=["A"],
        reclose_events=reclose_events or [],
    )


def _payload(currents, voltages, length):
    channels = []
    for phase, samples in currents.items():
        channels.append({"measurement": "current", "phase": phase, "unit": "A", "samples": list(samples)})
    for phase, samples in voltages.items():
        channels.append({"measurement": "voltage", "phase": phase, "unit": "kV", "samples": list(samples)})
    return {"time": [i / FS for i in range(length)], "frequency": 50.0, "analog_channels": channels}


def test_electrical_measurements_for_a_cleared_single_phase_fault():
    length, inception, clearing = 600, 200, 260  # 3-cycle fault on phase A, then pole A open
    ia = np.concatenate([_sine(400, inception), _sine(8000, clearing - inception), np.zeros(length - clearing)])
    ib = _sine(380, length, -120)
    va = np.concatenate([_sine(87, inception), _sine(30, clearing - inception), _sine(5, length - clearing)])
    vb = _sine(87, length, -120)
    payload = _payload({"A": ia, "B": ib}, {"A": va, "B": vb}, length)

    facts = electrical_measurements(payload, _window(inception, clearing))

    assert facts["current_unit"] == "A" and facts["voltage_unit"] == "kV"
    assert facts["prefault"]["current_rms"]["A"] == pytest.approx(400, rel=0.02)
    assert facts["prefault"]["voltage_rms"]["A"] == pytest.approx(87, rel=0.02)
    assert facts["fault"]["current_rms_max"]["A"] == pytest.approx(8000, rel=0.02)
    assert facts["fault"]["current_peak"]["A"] == pytest.approx(8000 * math.sqrt(2), rel=0.02)
    assert facts["fault"]["current_rms_max"]["B"] == pytest.approx(380, rel=0.02)
    assert facts["fault"]["voltage_rms_min"]["A"] == pytest.approx(30, rel=0.05)
    assert facts["after_clearing"]["current_rms"]["A"] == pytest.approx(0, abs=1)
    assert facts["after_clearing"]["current_rms"]["B"] == pytest.approx(380, rel=0.02)
    assert "after_reclose" not in facts


def test_electrical_measurements_for_a_reclose_capture():
    length, energized = 600, 150  # line dead, then re-energized at 150 ms
    ia = np.concatenate([np.zeros(energized), _sine(270, length - energized)])
    va = np.concatenate([np.zeros(energized), _sine(86, length - energized)])
    payload = _payload({"A": ia}, {"A": va}, length)
    window = _window(None, None, method="dead_time_recording", reclose_events=[{"time": energized / FS, "success": True}])

    facts = electrical_measurements(payload, window)

    assert "prefault" not in facts and "fault" not in facts
    assert facts["after_reclose"]["current_rms"]["A"] == pytest.approx(270, rel=0.02)
    assert facts["after_reclose"]["voltage_rms"]["A"] == pytest.approx(86, rel=0.02)


def test_electrical_measurements_without_window_or_channels_is_empty():
    assert electrical_measurements({"time": [0, 0.001]}, None) == {}
    payload = {"time": [i / FS for i in range(100)], "analog_channels": []}
    assert electrical_measurements(payload, _window(50, 60)) == {}
