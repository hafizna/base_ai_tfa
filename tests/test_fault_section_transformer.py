import json
import os
from pathlib import Path

import pytest

from core.fault_section import (
    CT_CURRENT,
    DIRECTION,
    ZONE_OPERATE,
    Interval,
    Observation,
    build_transformer,
    estimate,
    expected_observations,
)

ZONES = [
    {"id": "diff_87t", "kind": "transformer_diff", "label": "87T"},
    {"id": "ref_hv", "kind": "ref", "side": "hv", "label": "REF HV"},
    {"id": "ref_lv", "kind": "ref", "side": "lv", "label": "REF LV"},
    {"id": "mech", "kind": "mechanical", "label": "Buchholz"},
]
CTS = ["ct_hv", "ct_hv_n", "ct_lv", "ct_lv_n"]
S0 = [Interval("S0", 0.0, 0.08)]


def _variants(**spec):
    return build_transformer({"labels": {"trafo": "Trafo #2"}, "zones": ZONES, **spec})


def _one(**spec):
    (topology,) = _variants(ct_side={"cb_hv": "trafo", "cb_lv": "trafo"}, **spec)
    return topology


def _best(result):
    return {(v.hypothesis.section, v.hypothesis.earth) for v in result.best()}


def test_lv_earth_fault_inside_the_zone_is_located_and_needs_an_lv_source_to_light_the_lv_ct():
    truth = _one(lv_source=True)
    observations = expected_observations(truth, "lv_terminal", S0, earth=True, cts=CTS)
    result = estimate(_variants(lv_source="unknown"), S0, observations)

    assert ("lv_terminal", True) in _best(result)
    assert all(earth for _, earth in _best(result))
    assert "ada sumber lain di Bus LV" in result.inferred_assumptions()


def test_without_an_lv_source_the_lv_ct_stays_dark_and_the_source_is_ruled_out():
    truth = _one(lv_source=False)
    observations = expected_observations(truth, "lv_terminal", S0, earth=True, cts=CTS)
    assert next(o for o in observations if o.subject == "ct_lv").value is False
    result = estimate(_variants(lv_source="unknown"), S0, observations)

    assert "tidak ada sumber lain di Bus LV" in result.inferred_assumptions()


def test_a_phase_to_phase_fault_leaves_the_neutrals_dark_and_the_ref_quiet():
    truth = _one(lv_source=True)
    observations = expected_observations(truth, "hv_terminal", S0, earth=False, cts=CTS)
    by_subject = {o.subject: o.value for o in observations}
    assert by_subject["ct_hv_n"] is False and by_subject["ref_hv"] is False and by_subject["diff_87t"] is True

    assert _best(estimate([truth], S0, observations)) >= {("hv_terminal", False)}


def test_an_external_hv_earth_fault_drives_neutral_current_without_tripping_the_87t():
    truth = _one(lv_source=False)
    observations = expected_observations(truth, "bus_hv", S0, earth=True, cts=CTS)
    by_subject = {o.subject: o.value for o in observations}

    assert by_subject["ct_hv_n"] is True           # the grounded HV winding feeds the earth fault
    assert by_subject["diff_87t"] is False          # ... but that is zero sequence, not differential
    assert by_subject["ref_hv"] is False
    result = estimate([truth], S0, observations)
    assert {s for s, _ in _best(result)} <= {"bus_hv", "grid_hv", "stub_cb_hv"}


def test_forward_is_the_side_a_named_section_lies_on_even_two_devices_away():
    for side in ("bus", "trafo"):
        (topology,) = _variants(ct_side={"cb_hv": side, "cb_lv": "trafo"}, lv_source=False)
        observation = Observation("S0", DIRECTION, "ct_hv", "forward", "uji", weak=True, forward="hv_winding")
        result = estimate([topology], S0, [observation])
        verdict = next(v for v in result.per_section() if v.hypothesis.section == "lv_terminal")
        assert verdict.checks[0].predicted == "forward"


# --- the recorded incident ----------------------------------------------------------
# Point FAULT_SECTION_TAMBAKLOROK_ROOT at the incident folder (archives extracted).

_ROOT = os.environ.get("FAULT_SECTION_TAMBAKLOROK_ROOT")


@pytest.mark.skipif(not _ROOT or not Path(_ROOT).is_dir(), reason="Tambaklorok records not available")
def test_tambaklorok_trafo_2_lv_earth_fault_end_to_end():
    from core.fault_section.case import run_case

    case_path = Path(__file__).resolve().parents[1] / "config" / "fault_section_cases" / "tambaklorok_trf2_20240605.json"
    result = run_case(json.loads(case_path.read_text(encoding="utf-8")), Path(_ROOT))

    (episode,) = result.episodes
    best = {(v.hypothesis.section, v.hypothesis.earth) for v in episode.estimate.best()}
    assert best == {("lv_terminal", True), ("stub_cb_lv", True)}
    assert "ada sumber lain di Bus 20 kV" in episode.estimate.inferred_assumptions()
    by_section = {v.hypothesis.section: v for v in episode.estimate.per_section()}
    assert by_section["hv_terminal"].rank[0] > 0               # REF HV stayed quiet
    assert by_section["lv_winding"].rank == (0, 0, 2)          # only Buchholz/pressure (weak) and the HV polarity note
