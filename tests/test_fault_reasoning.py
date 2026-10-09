"""The per-record reasoning chain (``webapp.api.fault_reasoning``).

Synthetic records with known answers: each test builds the waveform and the
status channels a relay would record, runs the record analysis, and checks the
conclusion each rule draws (rule IDs as in docs/fault-reasoning-rules.md).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.fixtures.synthetic_records import _analog_channel, _base_payload, _sine, _status_channel
from webapp.api.fault_reasoning import _Channel, _round
from webapp.api.record_analysis import build_record_analysis

SR, FREQ, FAULT_S = 1200.0, 50.0, 0.3


def _record(*, duration_s=2.0, faulted=("A",), ground=True, clear_ms=80.0, reclose_ms=None, status=(), load=200.0,
            fault_current=8000.0, sag=0.4, current_only_rise=False):
    """A 150 kV line record: load, then a fault on ``faulted`` phases from
    0.3 s, its currents stopping ``clear_ms`` later (the line-side VT reads
    zero while the breaker is open), and an optional reclose."""
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    i0 = int(FAULT_S * SR)
    i_clear = i0 + int(clear_ms / 1000.0 * SR)
    i_close = i0 + int(reclose_ms / 1000.0 * SR) if reclose_ms else n
    angles = {"A": 0.0, "B": -120.0, "C": 120.0}
    v_peak = 150_000 / np.sqrt(3) * np.sqrt(2) / 1000.0  # kV
    currents, voltages = {}, {}
    for p, ang in angles.items():
        i = _sine(load * np.sqrt(2), FREQ, ang - 20, t)
        v = _sine(v_peak, FREQ, ang, t)
        if current_only_rise:
            i[i0:] = _sine(load * 30 * np.sqrt(2), FREQ, ang - 20, t)[i0:]
        elif p in faulted:
            if len(faulted) == 2 and not ground:
                # LL fault: equal and opposite currents in the two phases.
                sign = 1.0 if p == faulted[0] else -1.0
                i[i0:i_clear] = sign * _sine(fault_current * np.sqrt(2), FREQ, -80, t)[i0:i_clear]
            else:
                i[i0:i_clear] = _sine(fault_current * np.sqrt(2), FREQ, ang - 80, t)[i0:i_clear]
            v[i0:i_clear] *= sag
        if not current_only_rise:
            i[i_clear:i_close] = 0.0
            v[i_clear:i_close] = 0.0
        currents[p], voltages[p] = i, v
    analog = [_analog_channel(f"I{p}", f"I{p}", currents[p], "A", "current") for p in angles]
    analog += [_analog_channel(f"V{p}", f"V{p}", voltages[p], "kV", "voltage") for p in angles]
    channels = []
    for name, spans in status:
        samples = np.zeros(n, dtype=int)
        for on_ms, off_ms in spans:
            start = i0 + int(on_ms / 1000.0 * SR)
            stop = i0 + int(off_ms / 1000.0 * SR) if off_ms is not None else n
            samples[start:stop] = 1
        channels.append(_status_channel(name, name, samples))
    return _base_payload(t, FREQ, analog, channels, trigger_offset_s=FAULT_S)


def _conclusions(payload):
    reasoning = build_record_analysis("synthetic", payload).reasoning
    return reasoning, {c["key"]: c for c in reasoning["conclusions"]}


def test_single_phase_fault_aided_trip_reads_as_putt():
    payload = _record(faulted=("A",), reclose_ms=1080.0, status=[
        ("Z1", []), ("Z2", [(25, 85)]), ("Z3", []),
        ("DIST Trip A", [(35, 85)]), ("DIST Trip B", []), ("DIST Trip C", []),
        ("DIST Chan Recv", [(35, 120)]), ("DIST Sig. Send", []),
        ("A/R 1P In Prog", [(35, 1030)]), ("CB Aux A", [(85, 1085)]),
    ])
    reasoning, rows = _conclusions(payload)

    assert rows["fault"]["title"] == "Ya, dengan proteksi bekerja"
    assert rows["phases"]["title"] == "R-N, satu fasa ke tanah"
    assert rows["phases"]["value"]["phases"] == ["A"]
    assert {"F4.2", "F4.3", "F4.4"} <= set(rows["phases"]["rules"])
    assert rows["trip_path"]["title"] == "Aided trip lewat teleproteksi"
    assert rows["trip_path"]["value"]["receive_at_trip"] == "DIST Chan Recv"
    assert rows["scheme"]["title"] == "PUTT"
    assert rows["location"]["title"].startswith("Di luar jangkauan Z1")
    assert rows["trip_reclose"]["title"].startswith("Trip 1-pole R, SPAR")
    assert "Konsisten: gangguan satu fasa → trip dan reclose 1-pole." in rows["trip_reclose"]["evidence"]
    # Z2 picked up while the recorded Z3 never did (F5.6).
    assert rows["flag_nested_zones"]["rules"] == ["F5.6"]
    assert reasoning["flag_count"] == 1
    assert "DIST Sig. Send" in [s["channel"] for s in reasoning["signals"]["silent"]]


def test_clearing_is_the_last_current_zero_within_the_grid_code_limit():
    payload = _record(faulted=("A",), clear_ms=80.0, status=[("Z1", [(20, 80)]), ("DIST Trip A", [(30, 80)])])
    _reasoning, rows = _conclusions(payload)
    clearing = rows["clearing"]
    assert clearing["value"]["fct_ms"] == pytest.approx(80.0, abs=3.0)
    assert clearing["value"]["nominal_kv"] == 150.0
    assert clearing["value"]["limit_ms"] == 120.0
    assert clearing["title"].endswith("di bawah batas 120 ms")
    assert clearing["value"]["interruption_ms"] == pytest.approx(50.0, abs=3.0)
    assert rows["trip_path"]["title"] == "Z1, seketika"


def test_phase_to_phase_fault_three_pole_trip_is_not_phase_evidence():
    payload = _record(faulted=("B", "C"), ground=False, sag=0.6, status=[
        ("TRIP R", [(40, 80)]), ("TRIP S", [(40, 80)]), ("TRIP T", [(40, 80)]), ("TRIP Z1", [(42, 80)]),
    ])
    _reasoning, rows = _conclusions(payload)
    phases = rows["phases"]
    assert phases["title"] == "S-T, antar fasa, tidak ke tanah"
    assert "F4.5" in phases["rules"]
    assert "Trip 3-pole tidak dipakai sebagai bukti fasa." in phases["evidence"]
    # The fault's own currents (load removed) in S and T are equal and opposite.
    line = next(line for line in phases["evidence"] if line.startswith("Arus gangguan (tanpa beban) S"))
    assert 175 <= int(re.search(r"beda sudut (\d+)°", line).group(1)) <= 185
    assert rows["trip_path"]["title"] == "Z1, seketika"
    assert rows["trip_reclose"]["title"].startswith("Trip 3-pole")
    assert "Konsisten: gangguan multi-fasa → trip 3-pole." in rows["trip_reclose"]["evidence"]


def test_delayed_zone_2_trip_is_backup_and_outside_the_primary_limit():
    payload = _record(faulted=("A",), clear_ms=450.0, status=[
        ("Z1", []), ("Z2", [(20, 450)]), ("Z3", [(22, 450)]), ("DIST Trip A", [(420, 450)]),
    ])
    _reasoning, rows = _conclusions(payload)
    assert rows["trip_path"]["title"] == "Z2 waktu tunda"
    assert rows["trip_path"]["value"]["timer_s"] == 0.4
    clearing = rows["clearing"]
    assert clearing["value"]["backup_trip"] is True
    assert "melebihi" not in clearing["title"]
    assert "flag_nested_zones" not in rows  # Z3 started with Z2


def test_a_current_rise_on_normal_voltage_is_not_a_fault():
    # F1.1: a feeder current stepping up 30x while every voltage stays normal.
    payload = _record(current_only_rise=True)
    analysis = build_record_analysis("rise", payload)
    rows = {c["key"]: c for c in analysis.reasoning["conclusions"]}
    assert analysis.protection_interpretation["event_class"] == "NO_FAULT_TRIGGER"
    assert rows["fault"]["title"] == "Tidak ada gangguan"
    assert any("tanpa voltage sag" in line for line in rows["fault"]["evidence"])
    assert analysis.reasoning["has_fault"] is False


def test_contact_bounce_is_ignored():
    channel = _Channel("CB OPEN", "breaker", None, None, set(), on_ms=[100.0, 100.4, 300.0], off_ms=[100.2, 200.0], initially_on=False)
    assert channel.stable_intervals() == [(100.0, 200.0), (300.0, float("inf"))]


def test_numbers_round_half_up_like_the_incident_page():
    assert _round(76.5) == 77
    assert _round(-0.5) == -1
    assert _round(0.465, 2) == pytest.approx(0.47)
