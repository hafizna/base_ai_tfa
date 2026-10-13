import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.fault_section.mapping import (
    breaker_status,
    kcl_residual,
    operate_signals,
    phase_sets,
    relay_role,
    same_ct_score,
    station_key,
    zone_trip_signals,
)
from core.fault_section.observe import CurrentTrace

FS = 2400.0


def test_station_names_agree_across_spellings_and_abbreviations():
    assert station_key("GITET PEMALANG") == station_key("PMLNG") == station_key("Pemalang")
    assert station_key("GITET MANNDIRNCN") == station_key("Mandirancan")
    assert station_key("GITET PEMALANG") != station_key("GITET MANNDIRNCN")


@pytest.mark.parametrize("names, expected", [
    # NARI PCS-902: summed and per-breaker currents; "Ul2"/"3i0" are not phase sets
    (["ua", "ub", "uc", "3u0", "Ub1", "Ul2", "ia", "ib", "ic", "3i0", "CB1.ia", "CB1.ib", "CB1.ic"],
     {"V:u*", "I:i*", "I:cb1.i*"}),
    # MiCOM P643: three terminals, differential/bias channels left out
    (["IA-1", "IB-1", "IC-1", "IN-1", "IA-2", "IB-2", "IC-2", "IA-3", "IB-3", "IC-3", "IA-DIFF", "IB-DIFF",
      "IC-DIFF", "IA-BIAS", "LoZREF-DIFF-HV"], {"I:i*-1", "I:i*-2", "I:i*-3"}),
    # GE UR: one CT bank per four slot channels
    (["F1-IA", "F2-IB", "F3-IC", "F4-IG", "F5-IA", "F6-IB", "F7-IC", "F8-IG", "M1-IA", "M2-IB", "M3-IC"],
     {"I:F0", "I:F1", "I:M0"}),
    # MiCOM P546: incomplete IC2 set and computed channels left out
    (["VA", "VB", "VC", "IA", "IB", "IC", "IN", "IN Sensitive", "IA Differential", "IC2", "IN2",
      "V Checksync2", "Max I Bias"], {"V:v*", "I:i*"}),
    # PLN DFR naming with R/S/T phases
    (["IR BRINGIN 1", "IS BRINGIN 1", "IT BRINGIN 1", "VR BRINGIN 1", "VS BRINGIN 1", "VT BRINGIN 1"],
     {"I:i* bringin 1", "V:v* bringin 1"}),
])
def test_three_phase_sets_are_read_from_vendor_channel_names(names, expected):
    assert set(phase_sets(names)) == expected


@pytest.mark.parametrize("folder, files, kind, extra", [
    ("CCP AB-B MICOM MAIN A/DR", [], "ccp", {"main": "A", "bay": "b"}),
    ("CBF B DIAMETER 2_/DR", [], "cbf", {"breaker": "b"}),
    ("BUSPRO MAIN 2 FASA S", ["BUSPRO B MAIN 2 EVENT.jpg"], "busbar", {"main": "2", "bus": "b"}),
    ("LCD MAIN A BAY MANDIRANCAN 1/DR", [], "line_diff", {"main": "A", "bay_label": "Mandirancan 1"}),
    ("MAIN B BAY MANDIRANCAN 1", [], "line", {"main": "B", "bay_label": "Mandirancan 1"}),
])
def test_relay_roles_come_from_folder_names(folder, files, kind, extra):
    role = relay_role(folder, files)
    assert role.kind == kind
    for field_name, value in extra.items():
        assert getattr(role, field_name) == value


def test_an_unnamed_relay_with_a_differential_trip_and_voltages_is_a_line_differential():
    role = relay_role("TRIP MDCAN-PMLG 1 (26 MEI 2025)/DFR/MAIN A", signals=["Diff Trip", "Zone 1 Trip"],
                      has_voltages=True)
    assert (role.kind, role.main) == ("line_diff", "A")


@pytest.mark.parametrize("name, expected", [
    ("CB1.52b_PhA", (True, "", 1, "", 0)),
    ("L1_CB_B_R_OPN", (True, "b", 0, "", 0)),
    ("L9_CB_AB_R_OPN", (True, "ab", 0, "", 0)),
    ("L1_CB_B_R_CLS", (False, "b", 0, "", 0)),
    ("L7 CB CL B.B D2", (False, "", 0, "b", 2)),
    ("L36 CB OP B.B D8", (True, "", 0, "b", 8)),
    ("53 P2", (False, "", 0, "", 2)),
    ("L3 CB OPEN", (True, "", 0, "", 0)),
])
def test_breaker_status_names_say_contact_kind_and_breaker(name, expected):
    status = breaker_status(name)
    assert (status.open_when, status.letter, status.number, status.bus, status.diameter) == expected


@pytest.mark.parametrize("name", ["CB1 Trip OutputA", "R9_CB_AB_TRP2_R", "CB1.Alm_52b", "CB1.25.Ok_SynChk",
                                  "L3 DS OP B.B D1", "L10 BRF INIT", "25P2&53P2", "CB Fail Alarm", "R4 CBF STAGE 1"])
def test_trip_commands_alarms_and_disconnectors_are_not_breaker_states(name):
    assert breaker_status(name) is None


def test_operate_signals_prefer_the_three_phase_status():
    assert operate_signals(["Idiff Trip A", "Idiff Trip B", "Idiff Trip", "Idiff HS1 Trip A", "Any Start"]) == ["Idiff Trip"]
    ge = operate_signals(["XFMR PCNT DIFF OP", "XFMR PCNT DIFF OP A", "XFMR PCNT DIFF PKP A", "Trip Diff Op On"])
    assert set(ge) == {"XFMR PCNT DIFF OP", "Trip Diff Op On"}
    assert operate_signals(["BUS 1 OP", "BUS 1 BIASED PKP"]) == ["BUS 1 OP"]
    assert zone_trip_signals(["Zone 1 Trip", "21M1.Op", "Zone.1", "Diff Trip"]) == ["Zone 1 Trip", "21M1.Op"]


def _set(magnitudes, phase_shift=0.0, t0=0.0):
    """Three-phase current from per-segment magnitudes [(until_s, amps)]."""
    t = np.arange(0.0, magnitudes[-1][0], 1.0 / FS)
    amp = np.zeros_like(t)
    start = 0.0
    for until, value in magnitudes:
        amp[(t >= start) & (t < until)] = value
        start = until
    phases = tuple(np.sqrt(2) * amp * np.sin(2 * np.pi * 50 * t + phase_shift - k * 2 * np.pi / 3) for k in range(3))
    return CurrentTrace(ct="x", source="sintetis", recorder="r", t=t + t0, phases=phases)


def test_kirchhoff_across_three_terminals_of_one_node():
    line = _set([(0.1, 450.0)], phase_shift=0.0)
    b = _set([(0.1, 300.0)], phase_shift=np.pi)          # 450 = 300 + 150, flowing the other way
    ab = _set([(0.1, 150.0)], phase_shift=np.pi)
    assert kcl_residual([line, b, ab]) < 0.02
    unrelated = _set([(0.1, 150.0)], phase_shift=np.pi / 2)
    assert kcl_residual([line, b, unrelated]) > 0.2


def test_same_ct_is_recognised_by_fault_current_not_by_load():
    a = _set([(0.1, 450.0), (0.2, 2500.0), (0.3, 0.0)])
    a_other_relay = _set([(0.1, 445.0), (0.2, 2480.0), (0.3, 0.0)])
    similar_load_other_ct = _set([(0.1, 430.0), (0.2, 8400.0), (0.3, 0.0)])
    assert same_ct_score([a], [a_other_relay], 450.0, 445.0) < 0.02
    assert same_ct_score([a], [similar_load_other_ct], 450.0, 430.0) > 0.5
    load_only = _set([(0.3, 450.0)])
    assert same_ct_score([load_only], [_set([(0.3, 440.0)])], 450.0, 440.0) is None


# --- records only, end to end -----------------------------------------------------

_ROOT = os.environ.get("FAULT_SECTION_PEMALANG_ROOT")
_CASES = Path(__file__).resolve().parents[1] / "config" / "fault_section_cases"


@pytest.mark.skipif(not _ROOT or not Path(_ROOT).is_dir(), reason="Pemalang records not available")
def test_records_only_case_reaches_the_hand_mapped_answer():
    from core.fault_section.case import run_case

    auto = run_case(json.loads((_CASES / "pemalang_7b2_20250526.auto.json").read_text(encoding="utf-8")), Path(_ROOT))

    subjects = [d.subject for d in auto.decisions]
    assert any(s.startswith("ct_cb_b ← rec3:IA/IB/IC = rec4:IA-2") for s in subjects)
    assert any(s.startswith("ct_cb_ab ← rec4:IA-3") for s in subjects)
    assert any(s.startswith("ct_line_b_remote ← rec8") for s in subjects)
    first, second = auto.episodes
    assert {v.hypothesis.section for v in first.estimate.best()} == {"stub_cb_b"}
    assert "CT CB 7B2 di sisi Bus B" in first.estimate.inferred_assumptions()
    assert {v.hypothesis.section for v in second.estimate.best()} == {"t_b"}


@pytest.mark.skipif(not _ROOT or not Path(_ROOT).is_dir(), reason="Pemalang records not available")
def test_without_the_cbf_record_the_ambiguous_terminals_are_settled_and_flagged_as_weak():
    from core.fault_section.case import run_case

    case = json.loads((_CASES / "pemalang_7b2_20250526.auto.json").read_text(encoding="utf-8"))
    case["recorders"] = [r for r in case["recorders"] if r["id"] != "rec3"]
    result = run_case(case, Path(_ROOT))

    assert "2 kemungkinan pemetaan diuji" in result.mapping_choice
    assert "bukti lemah saja" in result.mapping_choice
    assert "ct_cb_b = rec4:IA-2/IB-2/IC-2" in result.mapping_choice
    assert {v.hypothesis.section for v in result.episodes[0].estimate.best()} == {"stub_cb_b"}
