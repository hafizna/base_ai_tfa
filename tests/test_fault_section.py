import json
from pathlib import Path

from core.fault_section import (
    BREAKER,
    CT,
    CT_CURRENT,
    DIRECTION,
    ZONE_OPERATE,
    Interval,
    Observation,
    Topology,
    Zone,
    build_diameter,
    estimate,
    expected_observations,
)

CASE = Path(__file__).resolve().parents[1] / "config" / "fault_section_cases" / "pemalang_7b2_20250526.json"
BUS_B_BAYS = ["cb_d1", "cb_d3", "cb_d4", "cb_d5", "cb_d6", "cb_d7", "cb_d8"]


def _pemalang_variants():
    return build_diameter(json.loads(CASE.read_text(encoding="utf-8"))["diameter"])


def _obs(interval, kind, subject, value, source, *, weak=False, forward=""):
    return Observation(interval, kind, subject, value, source, weak=weak, forward=forward)


def _best_sections(result):
    return {v.hypothesis.section for v in result.best()}


# --- topology primitives ----------------------------------------------------

def _chain():
    """grid — CB1 — bus — CT1 — stub — CB2 — feeder (source on both ends)."""
    topo = Topology()
    for sid, label, source in (("grid", "grid", True), ("bus", "bus", False), ("stub", "stub", False),
                               ("feeder", "feeder", True)):
        topo.add_section(sid, label, source=source)
    topo.add_device("cb1", BREAKER, "grid", "bus", "CB1")
    topo.add_device("ct1", CT, "bus", "stub", "CT1")
    topo.add_device("cb2", BREAKER, "stub", "feeder", "CB2")
    topo.add_zone(Zone("busbar", "busbar", (("ct1", "bus"),), end_zone=(("ct1", "cb2"),)))
    return topo


def test_ct_carries_current_towards_the_fault_and_not_through_an_open_breaker():
    topo = _chain()
    closed = topo.conducting(["cb1", "cb2"])
    assert topo.ct_flows("stub", closed) == {"ct1": frozenset({"stub"})}
    assert topo.ct_flows("bus", closed) == {"ct1": frozenset({"bus"})}
    only_grid = topo.conducting(["cb1"])
    assert topo.ct_flows("stub", only_grid) == {"ct1": frozenset({"stub"})}
    assert topo.ct_flows("bus", only_grid) == {}


def test_end_zone_logic_moves_the_busbar_boundary_to_the_open_breaker():
    topo = _chain()
    assert topo.zone_region("busbar", frozenset()) == {"bus", "grid"}
    assert "stub" in topo.zone_region("busbar", frozenset({"cb2"}))


# --- Pemalang 26-05-2025, observations read from the records ----------------
# Reference clock: NARI PCS-902 Main B at Pemalang. Values are fundamental RMS
# primary amperes measured in each interval (see the analysis notes).

def _episode_1():
    intervals = [
        Interval("S0", 42.850, 42.897, frozenset(), "semua CB tertutup"),
        Interval("S1", 42.914, 43.034, frozenset({"cb_b", "cb_ab"}), "CB 7B2, CB 7AB2 open"),
        Interval("S2", 43.045, 43.082, frozenset({"cb_b", "cb_ab", "cb_d1", "cb_d7", "cb_d8"}),
                 "D1, D7, D8 juga open"),
        Interval("C", 43.130, 43.300, frozenset({"cb_b", "cb_ab", *BUS_B_BAYS}), "setelah bus B lepas",
                 after_clearing=True),
    ]
    observations = [
        _obs("S0", CT_CURRENT, "ct_line_b", True, "PCS-902 Main B ia/ib/ic", ),
        _obs("S0", CT_CURRENT, "ct_cb_b", True, "P841 CBF 7B2 IA/IB/IC"),
        _obs("S0", CT_CURRENT, "ct_cb_ab", True, "P643 CCP Main A IA-3..IC-3"),
        _obs("S0", ZONE_OPERATE, "ccp_main_a", True, "P643 Idiff Trip"),
        _obs("S0", ZONE_OPERATE, "ccp_main_b", True, "GE event screenshot", weak=True),
        _obs("S0", ZONE_OPERATE, "busbar_b_main_2", False, "GE event screenshot", weak=True),
        _obs("S0", ZONE_OPERATE, "busbar_b_main_1", False, "P746 event log (tanpa trip)", weak=True),
        _obs("S0", ZONE_OPERATE, "lcd_main_a", False, "P546 IB Differential ~60 A"),
        _obs("S0", DIRECTION, "ct_line_b", "reverse", "P546 loop BG -102 deg", weak=True, forward="line_b"),
        _obs("S1", CT_CURRENT, "ct_line_b", False, "PCS-902 Main B ia/ib/ic"),
        _obs("S1", CT_CURRENT, "ct_cb_b", True, "P841 CBF 7B2 IA/IB/IC"),
        _obs("S1", CT_CURRENT, "ct_cb_ab", False, "P643 CCP Main A IA-3..IC-3"),
        _obs("S1", ZONE_OPERATE, "ccp_main_a", True, "P643 Idiff Trip"),
        _obs("S1", ZONE_OPERATE, "busbar_b_main_2", True, "GE event screenshot", weak=True),
        _obs("S1", ZONE_OPERATE, "busbar_b_main_1", False, "P746 event log (tanpa trip)", weak=True),
        _obs("S2", CT_CURRENT, "ct_cb_b", True, "P841 CBF 7B2 IA/IB/IC"),
        _obs("S2", CT_CURRENT, "ct_line_b", False, "PCS-902 Main B ia/ib/ic"),
        _obs("C", CT_CURRENT, "ct_cb_b", False, "P841 CBF 7B2 IA/IB/IC", weak=True),
        _obs("C", CT_CURRENT, "ct_line_b", False, "PCS-902 Main B ia/ib/ic", weak=True),
    ]
    return intervals, observations


def test_pemalang_first_fault_is_between_ct_7b2_and_cb_7b2_with_the_ct_on_the_bus_side():
    result = estimate(_pemalang_variants(), *_episode_1())

    assert _best_sections(result) == {"stub_cb_b"}
    assert all(v.rank == (0, 0, 0) for v in result.best())
    inferred = result.inferred_assumptions()
    assert "CT CB 7B2 di sisi Bus B" in inferred
    assert "BusPro B Main 2 (GE): end-zone aktif" in inferred
    assert "BusPro B Main 1 (P746): end-zone tidak aktif" in inferred


def test_pemalang_first_fault_rejects_the_t_zone_and_the_busbar_with_a_reason():
    result = estimate(_pemalang_variants(), *_episode_1())
    by_section = {v.hypothesis.section: v for v in result.per_section()}

    t_zone = by_section["t_b"]
    assert t_zone.rank[0] > 0
    reasons = [result.explain(c, t_zone) for c in t_zone.contradictions()]
    assert any("CT line Line Mandirancan 1" in r and "S1" in r for r in reasons)
    assert by_section["bus_b"].rank[0] > 0
    assert by_section["line_b"].rank[0] > 0


def _episode_2():
    everything_local = frozenset({"cb_b", "cb_ab", *BUS_B_BAYS})
    intervals = [
        Interval("S0", 44.040, 44.190, everything_local, "semua CB Pemalang open, Mandirancan tertutup"),
        Interval("C", 44.240, 44.400, everything_local | {"cb_remote_b"}, "setelah Mandirancan trip",
                 after_clearing=True),
    ]
    observations = [
        _obs("S0", CT_CURRENT, "ct_line_b", True, "PCS-902 Main B ia/ib/ic"),
        _obs("S0", CT_CURRENT, "ct_cb_b", False, "P643 CCP Main A IA-2..IC-2"),
        _obs("S0", CT_CURRENT, "ct_cb_ab", False, "P643 CCP Main A IA-3..IC-3"),
        _obs("S0", ZONE_OPERATE, "ccp_main_a", True, "P643 Idiff Trip"),
        _obs("S0", ZONE_OPERATE, "ccp_main_b", True, "GE T-series XFMR PCNT DIFF OP"),
        _obs("S0", DIRECTION, "ct_line_b", "reverse", "P546 loop AB -170 deg", weak=True, forward="line_b"),
        _obs("C", CT_CURRENT, "ct_line_b", False, "PCS-902 Main B ia/ib/ic", weak=True),
    ]
    return intervals, observations


def test_pemalang_second_fault_is_in_the_t_zone():
    result = estimate(_pemalang_variants(), *_episode_2())

    assert _best_sections(result) == {"t_b"}
    by_section = {v.hypothesis.section: v for v in result.per_section()}
    assert by_section["line_b"].rank[0] > 0          # no local source: the line CT would read nothing
    # The CB 7B2 stub only fits if CB 7B2 is assumed to have failed to open.
    assert by_section["stub_cb_b"].rank == (0, 1, 0)
    assert by_section["stub_cb_b"].hypothesis.stuck == {"cb_b"}


# --- synthetic scenarios: the forward model run backwards -------------------

def _diameter(**ct_side):
    spec = json.loads(CASE.read_text(encoding="utf-8"))["diameter"]
    spec["ct_side"] = {"cb_a": "bus", "cb_ab": "t_a", "cb_b": "bus", **ct_side}
    for zone in spec["zones"]:
        if zone["kind"] == "busbar":
            zone["end_zone"] = True
    (topology,) = build_diameter(spec)
    return topology


def test_a_line_fault_is_told_apart_from_the_t_zone():
    topo = _diameter()
    intervals = [Interval("S0", 0.0, 0.05), Interval("S1", 0.07, 0.15, frozenset({"cb_b", "cb_ab"}))]
    observations = expected_observations(topo, "line_b", intervals, cts=["ct_line_b", "ct_cb_b", "ct_cb_ab"],
                                         zones=["ccp_main_a", "lcd_main_a"])
    result = estimate([topo], intervals, observations)

    assert _best_sections(result) == {"line_b"}


def test_a_stuck_breaker_is_only_assumed_when_nothing_else_fits():
    topo = _diameter()
    intervals = [Interval("S0", 0.0, 0.05), Interval("S1", 0.07, 0.15, frozenset({"cb_b", "cb_ab"}))]
    observations = expected_observations(topo, "t_b", intervals, stuck=["cb_b"],
                                         cts=["ct_line_b", "ct_cb_b", "ct_cb_ab"], zones=["ccp_main_a"])
    result = estimate([topo], intervals, observations)

    best = result.best()
    assert {(v.hypothesis.section, v.hypothesis.stuck) for v in best} >= {("t_b", frozenset({"cb_b"}))}
    assert all(v.rank == (0, 1, 0) for v in best)


def test_weak_observations_rank_but_never_rule_out():
    topo = _diameter()
    intervals = [Interval("S0", 0.0, 0.05)]
    observations = expected_observations(topo, "stub_cb_b", intervals, cts=["ct_line_b", "ct_cb_b"],
                                         zones=["ccp_main_a"])
    observations.append(_obs("S0", ZONE_OPERATE, "busbar_b_main_2", True, "screenshot", weak=True))
    result = estimate([topo], intervals, observations)

    verdict = next(v for v in result.per_section() if v.hypothesis.section == "stub_cb_b")
    assert verdict.rank[0] == 0 and verdict.rank[2] == 1
