import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.fault_section.observe import (
    CurrentTrace,
    current_observations,
    episodes,
    intervals,
    match_clock,
    onsets,
    thresholds,
)

FS = 2400.0


def _trace(fault_at=0.10, clear_at=0.20, load=400.0, fault=5000.0, duration=0.4, offset=0.0, ct="ct_x"):
    t = np.arange(0.0, duration, 1.0 / FS)
    amp = np.where((t >= fault_at) & (t < clear_at), fault, np.where(t < fault_at, load, 0.0)) * np.sqrt(2)
    phases = tuple(amp * np.sin(2 * np.pi * 50 * t - k * 2 * np.pi / 3) for k in range(3))
    return CurrentTrace(ct=ct, source="sintetis", recorder="r", t=t + offset, phases=phases)


def test_inception_is_found_within_a_few_milliseconds_and_not_at_interruption():
    found = onsets(_trace(), thresholds(400.0, 4000.0)[0])

    assert len(found) == 1
    assert abs(found[0] - 0.10) < 0.004


def test_clock_match_needs_the_inception_pattern_not_just_one_instant():
    reference = [42.848, 44.020]
    own = [44.392, 45.564]                      # same 1.172 s gap, clock 1.544 s ahead
    match = match_clock(reference, own)

    assert match.anchors == 2
    assert match.offset_s == pytest.approx(-1.544, abs=1e-6)
    single = match_clock(reference, [42.851])
    assert single.anchors == 1 and single.offset_s == pytest.approx(-0.003, abs=1e-6)


def test_current_is_called_present_absent_or_left_unsaid():
    trace = _trace()
    limits = {"ct_x": thresholds(400.0, 4000.0)}
    spans = intervals((0.10, 0.20), {"cb": [(0.0, False), (0.15, True)]}, lambda b: b.upper())
    obs = {o.interval: o.value for o in current_observations([trace], spans, limits)}

    assert [s.id for s in spans] == ["S0", "S1", "C"]
    assert obs["S0"] is True
    assert obs["C"] is False
    assert spans[1].label == "+ CB open"

    load_only = _trace(fault=400.0, clear_at=0.4)
    quiet = current_observations([load_only], spans[:1], limits)
    assert quiet == []                          # 400 A of load is neither fault current nor zero


def test_episodes_split_two_faults_a_second_apart_and_start_at_the_inception():
    first, second = _trace(fault_at=0.10, clear_at=0.20, duration=1.6), _trace(fault_at=1.30, clear_at=1.40, duration=1.6)
    carries = {"ct_x": thresholds(400.0, 4000.0)[0]}
    spans = episodes([first, second], carries, inceptions=[0.1005, 1.3002])

    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(0.1005)
    assert spans[1][0] == pytest.approx(1.3002)


# --- end to end on the real records, when they are available -------------------
# Point FAULT_SECTION_PEMALANG_ROOT at the incident folder with its archives
# extracted in place (see the case file's notes).

_ROOT = os.environ.get("FAULT_SECTION_PEMALANG_ROOT")


@pytest.mark.skipif(not _ROOT or not Path(_ROOT).is_dir(), reason="Pemalang records not available")
def test_pemalang_records_end_to_end():
    from core.fault_section.case import run_case

    case_path = Path(__file__).resolve().parents[1] / "config" / "fault_section_cases" / "pemalang_7b2_20250526.json"
    result = run_case(json.loads(case_path.read_text(encoding="utf-8")), Path(_ROOT))

    offsets = {r.id: r.offset_s for r in result.recorders}
    assert offsets["p643_ccp_a"] == pytest.approx(0.58, abs=0.02)
    assert offsets["p546_mandirancan"] == pytest.approx(-1.544, abs=0.02)
    assert offsets["ge_buspro_b_main_2"] == pytest.approx(39095.69, abs=0.05)
    first, second = result.episodes
    assert {v.hypothesis.section for v in first.estimate.best()} == {"stub_cb_b"}
    assert "CT CB 7B2 di sisi Bus B" in first.estimate.inferred_assumptions()
    assert {v.hypothesis.section for v in second.estimate.best()} == {"t_b"}
    # The one thing the T-zone does not explain: Main A Pemalang's Zone 1 trip
    # (a forward decision) on what the currents show was a reverse fault.
    weak = second.estimate.best()[0].contradictions(weak=True)
    assert len(weak) == 1 and "Zone 1 Trip" in weak[0].observation.source
