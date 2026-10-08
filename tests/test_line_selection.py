"""Multi-line DFR records: per-line grouping, disturbed-line selection, and
projection of every analysis onto the selected line (core.line_selection).

Channel/status names are taken from real external-DFR records in the PLN
corpus (GI Bringin "CT R MJSNG2", GI Mojosongo "IR BRINGIN 2", Qualitrol
abbreviations such as "LP OPRT SRAGI1"), and the relay-record layouts that
must NOT be split into lines (1.5-breaker CB1/CB2 inputs, ABB RED670
LINE/REM, NR PCS-978 HVS/MVS/LVS, Siemens computed Delta channels).
"""

from __future__ import annotations

import numpy as np
import pytest

from core import line_selection as ls
from core.event_analysis import build_event_window
from webapp.api.ml_predict import extract_ml_features, run_ml_prediction
from webapp.api.record_analysis import build_record_analysis
from webapp.api.routers.relay_21 import _find_phase_current, _find_phase_voltage
from webapp.api.routers.relay_21_de import _build_terminal_context

SR = 4800.0
FREQ = 50.0
DURATION_S = 0.6
T = np.arange(int(SR * DURATION_S)) / SR
W = 2 * np.pi * FREQ


def _sine(amp, phase_deg, t=T):
    return amp * np.sin(W * t + np.deg2rad(phase_deg))


def _between(start_s, end_s):
    return (T >= start_s) & (T < end_s)


def _analog(name, canonical, samples, measurement):
    return {
        "id": name, "name": name, "canonical_name": canonical,
        "unit": "kV" if measurement == "voltage" else "A",
        "phase": canonical[-1] if canonical[-1] in "ABC" else None,
        "measurement": measurement, "ct_primary": 1.0, "ct_secondary": 1.0, "pors": "P",
        "samples": np.asarray(samples, dtype=float).tolist(),
    }


def _status(name, rises_at=None, falls_at=None, initial=0):
    samples = np.full(len(T), initial, dtype=int)
    if rises_at is not None:
        samples[T >= rises_at] = 1
    if falls_at is not None:
        samples[T >= falls_at] = 0
    return {"id": name, "name": name, "samples": samples.tolist()}


def _line(names, currents, voltages):
    """``names``: (fmt for current, fmt for voltage) with a {ph} placeholder (R/S/T)."""
    channels = []
    for ph, canon, samples in zip("RST", ("IA", "IB", "IC"), currents):
        channels.append(_analog(names[0].format(ph=ph), canon, samples, "current"))
    for ph, canon, samples in zip("RST", ("VA", "VB", "VC"), voltages):
        channels.append(_analog(names[1].format(ph=ph), canon, samples, "voltage"))
    return channels


def _load_currents(amp):
    return [_sine(amp, 0), _sine(amp, -120), _sine(amp, 120)]


def _bus_voltages(kv, sag_window=None, sag_kv=None):
    va, vb, vc = _sine(kv, 0), _sine(kv, -120), _sine(kv, 120)
    if sag_window is not None:
        vb = np.where(sag_window, _sine(sag_kv, -150), vb)
        vc = np.where(sag_window, _sine(sag_kv, 150), vc)
    return [va, vb, vc]


def _bc_fault_currents(load_a, fault_a, fault, after):
    """Phase B-C fault during ``fault``; ``after`` gives the post-fault current
    level per phase (0 for a tripped line, load for one still in service)."""
    ia, ib, ic = _load_currents(load_a)
    ib = np.where(fault, _sine(fault_a, -200), ib)
    ic = np.where(fault, -_sine(fault_a, -200), ic)
    post = T >= fault.nonzero()[0][-1] / SR
    return [np.where(post, _sine(after, p), x) for x, p in zip((ia, ib, ic), (0, -120, 120))]


def _payload(analog, status):
    return {
        "station_name": "GI TEST", "rec_dev_id": "DFR", "frequency": FREQ, "time": T.tolist(),
        "trigger_offset_s": 0.1, "trigger_time": 0.1, "total_samples": len(T),
        "analog_channels": analog, "status_channels": status, "warnings": [],
    }


FAULT = _between(0.12, 0.21)


def bringin_record():
    """GI Bringin ZQ6D shape: MJSNG1 out of service (CB open throughout, noise),
    MJSNG2 B-C fault, Z1 trip 3-pole, breaker opens."""
    rng = np.random.default_rng(0)
    dead = [rng.normal(0.0, 1.0, len(T)) for _ in range(3)]
    dead_v = [rng.normal(0.0, 0.3, len(T)) for _ in range(3)]
    line2_i = _bc_fault_currents(420.0, 5700.0, FAULT, after=0.0)
    line2_v = _bus_voltages(87.0, FAULT, 54.0)
    line2_v = [np.where(T >= 0.21, 0.0, v) for v in line2_v]
    analog = (
        _line(("CT {ph} MJSNG1", "VT {ph} MJSNG1"), dead, dead_v)
        + [_analog("V DC POS", "V DC POS", np.zeros(len(T)), "voltage")]
        + _line(("CT {ph} MJSNG2", "VT {ph} MJSNG2"), line2_i, line2_v)
    )
    status = [
        _status("CB OPEN MJSNG1", initial=1),
        _status("TRIP R MJSNG1"),
        _status("TRIP R MJSNG2", rises_at=0.165, falls_at=0.228),
        _status("TRIP S MJSNG2", rises_at=0.166, falls_at=0.228),
        _status("TRIP T MJSNG2", rises_at=0.165, falls_at=0.228),
        _status("TRIP Z1 MJSNG2", rises_at=0.167, falls_at=0.225),
        _status("CB OPEN MJSNG2", rises_at=0.231),
        _status("SPARE"),
    ]
    return _payload(analog, status)


def parallel_lines(line1_fault_a=1500.0, line1_after=600.0, line2_fault_a=5000.0, line2_after=0.0, status=()):
    """Mojosongo-style naming, both lines in service: line 2 faults, line 1
    carries parallel infeed (or its own fault current when a test says so)."""
    line1_i = _bc_fault_currents(300.0, line1_fault_a, FAULT, after=line1_after)
    line2_i = _bc_fault_currents(300.0, line2_fault_a, FAULT, after=line2_after)
    v = _bus_voltages(86.0, FAULT, 50.0)
    analog = (
        _line(("I{ph} BRINGIN 1", "V{ph} BRINGIN 1"), line1_i, v)
        + _line(("I{ph} BRINGIN 2", "V{ph} BRINGIN 2"), line2_i, v)
    )
    return _payload(analog, list(status))


def _names(channels):
    return [ch["name"] for ch in channels]


# --- line identity ---------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("CT R MJSNG2", "MJSNG2"),
    ("VT T MJSNG1", "MJSNG1"),
    ("CT N MJSNG1", "MJSNG1"),
    ("Freq:VT R MJSNG1", "MJSNG1"),
    ("IR BRINGIN 2", "BRINGIN 2"),
    ("IN BRINGIN 2", "BRINGIN 2"),
    ("CIBADAK IR", "CIBADAK"),
    ("SALAK BARU VT", "SALAK BARU"),
    ("IR PMPK-1", "PMPK 1"),
    ("VR IBT 1 150KV", "IBT 1"),
    ("IR SPARE 2", "SPARE 2"),
    ("MRANGGEN1_Ia", "MRANGGEN1"),
    ("IA1", "1"),
    ("V DC POS", None),
    ("A - CH04", None),
    ("IA", None),
    ("Ia", None),
    ("UL1", None),
    ("I L1", None),
])
def test_line_key_from_real_channel_names(name, expected):
    assert ls.line_key(name) == expected


@pytest.mark.parametrize("status, lines, expected", [
    ("TRIP Z1 MJSNG2", ["MJSNG1", "MJSNG2"], "MJSNG2"),
    ("TRIP Z2/3 MJSNG1", ["MJSNG1", "MJSNG2"], "MJSNG1"),
    ("DIST Z1 TRIP BRINGIN 2", ["BRINGIN 1", "BRINGIN 2"], "BRINGIN 2"),
    ("PRESSURE STEP 2 BRINGIN 1", ["BRINGIN 1", "BRINGIN 2"], "BRINGIN 1"),
    ("21 PROT TRIP BAYAH1", ["BAYAH 1", "BAYAH 2"], "BAYAH 1"),
    ("LP OPRT SRAGI1", ["SUNYARAGI 1", "SUNYARAGI 2"], "SUNYARAGI 1"),
    ("F21-1 DPK1 OPRT", ["DEPOK1", "DEPOK2"], "DEPOK1"),
    ("DIST SLKBR FASA A TRIP", ["CIBADAK", "SALAK BARU"], "SALAK BARU"),
    ("BACKUP PROT SKMD", ["JATIBARANG", "SUKAMANDI 2"], "SUKAMANDI 2"),
    ("MAIN PROT PGNDRN 1", ["PGDRAN 1", "PGDRAN 2"], "PGDRAN 1"),
    ("LP OPRT", ["TSMYA1", "TSMYA2"], None),
    ("CB 5B4 OPEN", ["KSBRU 1", "KSBRU 2"], None),
    ("SRAGI OPRT", ["SUNYARAGI 1", "SUNYARAGI 2"], None),  # names no circuit -> ambiguous
])
def test_status_channel_assignment(status, lines, expected):
    assert ls.status_line_key(status, lines) == expected


def _relay_channels(current_names, voltage_names):
    channels = []
    for name, canon in current_names:
        channels.append(_analog(name, canon, _sine(100.0, 0), "current"))
    for name, canon in voltage_names:
        channels.append(_analog(name, canon, _sine(60.0, 0), "voltage"))
    return channels


@pytest.mark.parametrize("channels", [
    # 1.5-breaker relay: two CT inputs of one line, one shared voltage set.
    _relay_channels(
        [("CB1.ia", "IA"), ("CB1.ib", "IB"), ("CB1.ic", "IC"), ("CB2.ia", "IA"), ("CB2.ib", "IB"), ("CB2.ic", "IC")],
        [("ua", "VA"), ("ub", "VB"), ("uc", "VC")],
    ),
    # ABB RED670 87L: local and remote terminal of the SAME line.
    _relay_channels(
        [("LINE CT IL1", "IA"), ("LINE CT IL2", "IB"), ("LINE CT IL3", "IC"),
         ("REM CT IL1", "IA"), ("REM CT IL2", "IB"), ("REM CT IL3", "IC")],
        [("LINE VT UL1", "VA"), ("LINE VT UL2", "VB"), ("LINE VT UL3", "VC"),
         ("REM VT UL1", "VA"), ("REM VT UL2", "VB"), ("REM VT UL3", "VC")],
    ),
    # NR PCS-978 transformer: windings, not lines.
    _relay_channels(
        [("HVS.Ia", "IA"), ("HVS.Ib", "IB"), ("HVS.Ic", "IC"), ("LVS.Ia", "IA"), ("LVS.Ib", "IB"), ("LVS.Ic", "IC")],
        [("HVS.Ua", "VA"), ("HVS.Ub", "VB"), ("HVS.Uc", "VC"), ("LVS.Ua", "VA"), ("LVS.Ub", "VB"), ("LVS.Uc", "VC")],
    ),
    # Siemens computed delta channels next to the measured ones.
    _relay_channels(
        [("iL1", "IA"), ("iL2", "IB"), ("iL3", "IC"),
         ("iL1(Delta Prev.)*", "IA"), ("iL2(Delta Prev.)*", "IB"), ("iL3(Delta Prev.)*", "IC"),
         ("iL1(Delta First)*", "IA"), ("iL2(Delta First)*", "IB"), ("iL3(Delta First)*", "IC")],
        [("uL1", "VA"), ("uL2", "VB"), ("uL3", "VC"),
         ("uL1(Delta Prev.)*", "VA"), ("uL2(Delta Prev.)*", "VB"), ("uL3(Delta Prev.)*", "VC"),
         ("uL1(Delta First)*", "VA"), ("uL2(Delta First)*", "VB"), ("uL3(Delta First)*", "VC")],
    ),
    # Ordinary single-line record.
    _relay_channels([("IA", "IA"), ("IB", "IB"), ("IC", "IC")], [("VA", "VA"), ("VB", "VB"), ("VC", "VC")]),
], ids=["cb1-cb2-ct-inputs", "87l-local-remote", "transformer-windings", "computed-delta", "single-line"])
def test_records_that_are_not_multi_line_are_left_alone(channels):
    payload = _payload(channels, [])
    assert ls.select_line_for_payload(payload) is None
    assert ls.scope_payload(payload) is payload


# --- disturbed-line selection ----------------------------------------------

def test_out_of_service_line_is_ignored_and_faulted_line_selected_by_its_trip():
    selection = ls.select_line_for_payload(bringin_record())
    states = {line.key: line.state for line in selection.lines}
    assert selection.selected == "MJSNG2"
    assert selection.method == "status_protection"
    assert states["MJSNG1"] == ls.LINE_DE_ENERGIZED
    assert selection.flags == []
    assert selection.requires_review is False


def test_parallel_line_carrying_infeed_is_impacted_not_selected():
    payload = parallel_lines(status=[
        _status("DIST Z1 TRIP BRINGIN 1"),
        _status("DIST Z1 TRIP BRINGIN 2", rises_at=0.15, falls_at=0.22),
        _status("CB TRIP BRINGIN 2", rises_at=0.16),
    ])
    selection = ls.select_line_for_payload(payload)
    states = {line.key: line.state for line in selection.lines}
    assert selection.selected == "BRINGIN 2"
    assert states["BRINGIN 1"] == ls.LINE_IMPACTED
    assert selection.flags == [ls.FLAG_OTHER_LINE_IMPACTED]
    assert selection.requires_review is False


def test_without_status_the_line_whose_current_was_interrupted_is_selected():
    selection = ls.select_line_for_payload(parallel_lines())
    assert selection.selected == "BRINGIN 2"
    assert selection.method == "current_interruption"
    assert selection.requires_review is False


def test_comparable_fault_current_without_line_evidence_is_flagged_ambiguous():
    payload = parallel_lines(line1_fault_a=4000.0, line1_after=300.0, line2_fault_a=4500.0, line2_after=300.0)
    selection = ls.select_line_for_payload(payload)
    assert selection.selected == "BRINGIN 2"
    assert selection.method == "superimposed_current"
    assert ls.FLAG_AMBIGUOUS_SELECTION in selection.flags
    assert selection.requires_review is True


def test_both_lines_protection_operating_is_flagged_as_double_circuit_or_sympathetic():
    payload = parallel_lines(line1_fault_a=3000.0, line1_after=0.0, status=[
        _status("DIST Z1 TRIP BRINGIN 1", rises_at=0.15, falls_at=0.22),
        _status("DIST Z1 TRIP BRINGIN 2", rises_at=0.15, falls_at=0.22),
    ])
    selection = ls.select_line_for_payload(payload)
    states = {line.key: line.state for line in selection.lines}
    assert selection.selected == "BRINGIN 2"  # larger fault current of the two
    assert states["BRINGIN 1"] == ls.LINE_ALSO_OPERATED
    assert ls.FLAG_MULTIPLE_LINES_OPERATED in selection.flags
    assert selection.requires_review is True


def test_shared_diameter_breaker_on_the_other_bay_is_not_a_second_operated_line():
    payload = parallel_lines(status=[
        _status("CB AB OPEN BRINGIN 1", rises_at=0.23),
        _status("DIST Z1 TRIP BRINGIN 2", rises_at=0.15, falls_at=0.22),
    ])
    selection = ls.select_line_for_payload(payload)
    states = {line.key: line.state for line in selection.lines}
    assert selection.selected == "BRINGIN 2"
    assert states["BRINGIN 1"] == ls.LINE_BREAKER_OR_RECLOSE_ONLY
    assert ls.FLAG_MULTIPLE_LINES_OPERATED not in selection.flags


def test_dead_time_recording_selects_the_line_whose_breaker_recloses():
    rng = np.random.default_rng(1)
    energized = T >= 0.135
    line2_i = [np.where(energized, x, rng.normal(0.0, 0.5, len(T))) for x in _load_currents(280.0)]
    line2_v = [np.where(energized, v, 0.0) for v in _bus_voltages(87.0)]
    analog = (
        _line(("CT {ph} MJSNG1", "VT {ph} MJSNG1"),
              [rng.normal(0.0, 1.0, len(T)) for _ in range(3)], [rng.normal(0.0, 0.3, len(T)) for _ in range(3)])
        + _line(("CT {ph} MJSNG2", "VT {ph} MJSNG2"), line2_i, line2_v)
    )
    payload = _payload(analog, [
        _status("CB OPEN MJSNG1", initial=1),
        _status("CB OPEN MJSNG2", initial=1, falls_at=0.135),
    ])
    selection = ls.select_line_for_payload(payload)
    states = {line.key: line.state for line in selection.lines}
    assert selection.selected == "MJSNG2"
    assert selection.method == "status_breaker"
    assert states["MJSNG1"] == ls.LINE_DE_ENERGIZED


# --- projection onto the selected line --------------------------------------

def test_scope_payload_keeps_selected_line_and_common_channels_only():
    payload = bringin_record()
    scoped = ls.scope_payload(payload)
    assert all("MJSNG1" not in name for name in _names(scoped["analog_channels"]))
    assert "V DC POS" in _names(scoped["analog_channels"])
    assert _names(scoped["status_channels"]) == [
        "TRIP R MJSNG2", "TRIP S MJSNG2", "TRIP T MJSNG2", "TRIP Z1 MJSNG2", "CB OPEN MJSNG2", "SPARE",
    ]
    assert ls.scope_payload(scoped) is scoped  # already single-line: no-op
    assert len(payload["analog_channels"]) == 13  # the stored payload itself is untouched


def test_common_channel_never_shadows_the_selected_lines_own_channel():
    payload = parallel_lines(status=[_status("DIST Z1 TRIP BRINGIN 2", rises_at=0.15)])
    payload["analog_channels"].insert(0, _analog("VR BUS", "VA", _sine(86.0, 0), "voltage"))
    names = _names(ls.scope_payload(payload)["analog_channels"])
    assert "VR BUS" not in names
    assert "VR BRINGIN 2" in names


# --- every analysis runs on the selected line --------------------------------

def test_event_window_uses_the_faulted_line_not_the_out_of_service_one():
    window = build_event_window(bringin_record())
    # Before: MJSNG1 noise -> no waveform onset -> inception = TRIP time (165 ms),
    # phases A/B/C read off the 3-pole trip.
    assert window.inception_time_ms == pytest.approx(120.0, abs=5.0)
    assert window.faulted_phases == ["B", "C"]


def test_ai_features_are_extracted_from_the_faulted_line():
    row = extract_ml_features(bringin_record())
    assert row["peak_fault_current_a"] > 4000.0


def test_ai_result_reports_which_line_was_analysed():
    result = run_ml_prediction(bringin_record())
    assert result["meta"]["analyzed_line"] == "MJSNG2"
    assert result["meta"]["line_selection"]["method"] == "status_protection"


def test_record_analysis_exposes_selection_and_flags_review_cases():
    plain = build_record_analysis("bringin", bringin_record())
    assert plain.provenance["line_selection"]["selected_line"] == "MJSNG2"
    assert plain.data_quality["multi_line_record"] is True
    assert not any(item["type"] == "MULTI_LINE_DISTURBANCE" for item in plain.missing_evidence)

    double = parallel_lines(line1_fault_a=3000.0, line1_after=0.0, status=[
        _status("DIST Z1 TRIP BRINGIN 1", rises_at=0.15, falls_at=0.22),
        _status("DIST Z1 TRIP BRINGIN 2", rises_at=0.15, falls_at=0.22),
    ])
    review = build_record_analysis("double", double)
    flagged = [item for item in review.missing_evidence if item["type"] == "MULTI_LINE_DISTURBANCE"]
    assert flagged and flagged[0]["requires_review"] is True


def test_double_ended_terminal_reports_the_analysed_line():
    ctx = _build_terminal_context(bringin_record(), "ZBC", invert_i=False, invert_phase_sequence=False)
    assert ctx["active_tag"] == "MJSNG2"


def test_phase_finder_does_not_read_a_line_number_as_a_phase():
    channels = [
        _analog("VT R MJSNG2", "VA", _sine(1.0, 0), "voltage"),
        _analog("VT S MJSNG2", "VB", _sine(2.0, 0), "voltage"),
        _analog("VT T MJSNG2", "VC", _sine(3.0, 0), "voltage"),
        _analog("CT R MJSNG2", "IA", _sine(10.0, 0), "current"),
        _analog("CT S MJSNG2", "IB", _sine(20.0, 0), "current"),
        _analog("CT T MJSNG2", "IC", _sine(30.0, 0), "current"),
    ]
    # "...MJSNG2" ends in "2", which the loose suffix alias reads as phase B.
    assert np.max(_find_phase_voltage(channels, "B")) == pytest.approx(2.0, rel=1e-3)
    assert np.max(_find_phase_current(channels, "B")) == pytest.approx(20.0, rel=1e-3)
    assert np.max(_find_phase_current(channels, "A")) == pytest.approx(10.0, rel=1e-3)
