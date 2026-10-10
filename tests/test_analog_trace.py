"""core.analog_trace on synthetic records with known physics.

Each case builds per-phase currents and voltages with a known fault start,
breaker current zeros, CT tails, reclose and the like, and checks the trace
reads them back: the fault clearing time from the last current zero (not the
CT tail), a healthy phase's elevated current not counted as fault current, a
current rise with every voltage normal not counted as a fault, a weak infeed
end timed from its voltage, the far end opening first, an energization at a
reclose, and a re-fault after it.
"""

import math

import numpy as np
import pytest

from core.analog_trace import trace_payload

F = 50.0
SR = 4800.0
VN = 86.6  # kV phase-to-neutral, 150 kV system


def _t(duration_s: float) -> np.ndarray:
    return np.arange(int(round(duration_s * SR))) / SR


def _sine(t: np.ndarray, rms: float, angle_deg: float) -> np.ndarray:
    return rms * math.sqrt(2.0) * np.cos(2 * math.pi * F * t + math.radians(angle_deg))


def _piecewise(t: np.ndarray, pieces: list[tuple[float, float, float]]) -> np.ndarray:
    """(start_s, rms, angle_deg) pieces, each holding until the next one."""
    x = np.zeros_like(t)
    for start, rms, angle in pieces:
        mask = t >= start
        x[mask] = _sine(t[mask], rms, angle)
    return x


def _set_from(x: np.ndarray, t: np.ndarray, start: float, rms: float, angle: float) -> None:
    mask = t >= start
    x[mask] = _sine(t[mask], rms, angle)


def _open_at_zero(x: np.ndarray, t: np.ndarray, t_open: float, tail: float = 0.0, tau: float = 0.02):
    """A breaker pole: the current stops at its first zero at or after
    ``t_open``; an optional decaying CT tail follows. Returns (current, time
    of the zero in s)."""
    k = int(np.searchsorted(t, t_open))
    while k < len(x) - 1 and x[k] * x[k + 1] > 0:
        k += 1
    out = x.copy()
    out[k + 1:] = tail * np.exp(-(t[k + 1:] - t[k + 1]) / tau) if tail else 0.0
    return out, float(t[k])


def _payload(t: np.ndarray, currents: dict[str, np.ndarray], voltages: dict[str, np.ndarray]) -> dict:
    channels = [
        {"name": f"V{p}", "phase": p, "measurement": "voltage", "unit": "kV", "samples": x.tolist()}
        for p, x in voltages.items()
    ] + [
        {"name": f"I{p}", "phase": p, "measurement": "current", "unit": "A", "samples": x.tolist()}
        for p, x in currents.items()
    ]
    return {"time": t.tolist(), "frequency": F, "analog_channels": channels, "status_channels": []}


def _trace(payload: dict) -> dict:
    trace = trace_payload(payload)
    assert trace is not None
    return trace.to_dict()


def _kinds(trace: dict, kind: str) -> list[dict]:
    return [e for e in trace["events"] if e["kind"] == kind]


def test_single_phase_fault_single_pole_open_and_reclose():
    """Cibatu-Mekarsari 2 pattern: R-N fault, pole R opens, the healthy phases
    carry 2.2x load meanwhile, R recloses a second later."""
    t = _t(1.4)
    ia = _piecewise(t, [(0, 500, -20), (0.1, 8000, -80)])
    ia, a_open = _open_at_zero(ia, t, 0.18, tail=400, tau=0.02)
    _set_from(ia, t, 1.18, 500, -20)
    ib = _piecewise(t, [(0, 500, -140), (0.1, 1100, -150)])
    ic = _piecewise(t, [(0, 500, 100), (0.1, 1000, 95)])
    _set_from(ib, t, a_open, 500, -140)
    _set_from(ic, t, a_open, 500, 100)
    va = _piecewise(t, [(0, VN, 0), (0.1, 0.45 * VN, 0)])
    _set_from(va, t, a_open, 0.2 * VN, 0)  # open pole: coupled voltage from the healthy phases
    _set_from(va, t, 1.18, VN, 0)
    vb = _piecewise(t, [(0, VN, -120), (0.1, 0.93 * VN, -120)])
    vc = _piecewise(t, [(0, VN, 120), (0.1, 0.93 * VN, 120)])
    _set_from(vb, t, a_open, VN, -120)
    _set_from(vc, t, a_open, VN, 120)

    s = _trace(_payload(t, {"A": ia, "B": ib, "C": ic}, {"A": va, "B": vb, "C": vc}))["summary"]
    assert s["fault_start_ms"] == pytest.approx(100.0, abs=1.5)
    assert s["high_current_phases"] == ["A"]  # 2.2x load on B and C is not fault current
    assert set(s["ceased"]) == {"A"}
    assert s["ceased"]["A"] == pytest.approx(a_open * 1000, abs=1.0)
    assert s["fct_ms"] == pytest.approx(a_open * 1000 - 100, abs=2.0)
    assert s["zero_current_pattern"] == "single_phase"
    assert s["dead_time_ms"]["A"] == pytest.approx(1180 - a_open * 1000, abs=5.0)
    assert s["refault_ms"] is None


def test_clearing_time_ends_at_the_current_zero_not_the_ct_tail():
    """Bringin ZQ6D pattern: S-T fault, three-pole opening, a CT tail on the
    faulted phases. The trip-pulse width and the RMS-envelope reading both
    mis-stated this clearing time; the current zero gives it."""
    t = _t(0.5)
    ia = _piecewise(t, [(0, 400, -20)])
    ib = _piecewise(t, [(0, 400, -140), (0.1, 6000, -100)])
    ic = _piecewise(t, [(0, 400, 100), (0.1, 6000, 80)])
    ia, a0 = _open_at_zero(ia, t, 0.17)
    ib, b0 = _open_at_zero(ib, t, 0.17, tail=900, tau=0.015)
    ic, c0 = _open_at_zero(ic, t, 0.17, tail=-900, tau=0.015)
    va = _piecewise(t, [(0, VN, 0), (0.1, 0.97 * VN, 0)])
    vb = _piecewise(t, [(0, VN, -120), (0.1, 0.6 * VN, -110)])
    vc = _piecewise(t, [(0, VN, 120), (0.1, 0.6 * VN, 130)])
    line_open = max(a0, b0, c0)
    for v, angle in ((va, 0), (vb, -120), (vc, 120)):
        _set_from(v, t, line_open, 0.01 * VN, angle)

    trace = _trace(_payload(t, {"A": ia, "B": ib, "C": ic}, {"A": va, "B": vb, "C": vc}))
    s = trace["summary"]
    assert s["fault_start_ms"] == pytest.approx(100.0, abs=1.5)
    assert s["high_current_phases"] == ["B", "C"]
    assert s["ceased"]["B"] == pytest.approx(b0 * 1000, abs=1.0)
    assert s["ceased"]["C"] == pytest.approx(c0 * 1000, abs=1.0)
    assert s["fct_ms"] == pytest.approx(max(b0, c0) * 1000 - 100, abs=2.0)
    assert s["zero_current_pattern"] == "three_together"
    assert {e["phase"] for e in _kinds(trace, "voltage_lost")} == {"A", "B", "C"}


def test_a_current_rise_with_every_voltage_normal_is_not_a_fault():
    """Cirata #2 (20 Nov 2024, second file): 3 A rising to 70 A with every
    voltage normal triggered the DFR; no fault."""
    t = _t(0.4)
    currents = {p: _piecewise(t, [(0, 3, a), (0.1, 70, a - 10)]) for p, a in (("A", -20), ("B", -140), ("C", 100))}
    voltages = {p: _piecewise(t, [(0, VN, a)]) for p, a in (("A", 0), ("B", -120), ("C", 120))}
    trace = _trace(_payload(t, currents, voltages))
    assert trace["summary"]["fault_start_ms"] is None
    assert trace["summary"]["disturbances"] == []
    assert {e["phase"] for e in _kinds(trace, "current_rise")} == {"A", "B", "C"}


def test_a_record_starting_dead_reads_the_reclose_as_an_energization():
    """Bringin ZQ6E pattern: the record starts in dead time; the line is
    re-energized with a two-cycle inrush and a sagging voltage. Not a fault."""
    t = _t(0.6)
    rng = np.random.default_rng(1)
    currents, voltages = {}, {}
    for p, a in (("A", -20), ("B", -140), ("C", 100)):
        i = rng.normal(0.0, 0.3, len(t))
        _set_from(i, t, 0.15, 1000, a)
        _set_from(i, t, 0.19, 300, a)
        currents[p] = i
        v = _sine(t, 0.005 * VN, a + 20)
        _set_from(v, t, 0.15, 0.6 * VN, a + 20)
        _set_from(v, t, 0.19, VN, a + 20)
        voltages[p] = v

    s = _trace(_payload(t, currents, voltages))["summary"]
    assert s["starts_dead"] is True
    assert s["fault_start_ms"] is None
    assert s["energization_ms"] == pytest.approx(150.0, abs=3.0)
    assert set(s["current_returns"]) == {"A", "B", "C"}
    for p in "ABC":
        assert s["current_returns"][p] == pytest.approx(150.0, abs=2.0)


def test_a_weak_infeed_end_is_timed_from_its_voltage():
    """Mojosongo pattern: the faulted phases carry under 2x their load, but
    their voltage halves."""
    t = _t(0.4)
    ia = _piecewise(t, [(0, 400, -20)])
    ib = _piecewise(t, [(0, 400, -140), (0.1, 700, -120)])
    ic = _piecewise(t, [(0, 400, 100), (0.1, 750, 110)])
    opened = {}
    ia, opened["A"] = _open_at_zero(ia, t, 0.16)
    ib, opened["B"] = _open_at_zero(ib, t, 0.16)
    ic, opened["C"] = _open_at_zero(ic, t, 0.16)
    va = _piecewise(t, [(0, VN, 0)])
    vb = _piecewise(t, [(0, VN, -120), (0.1, 0.5 * VN, -110)])
    vc = _piecewise(t, [(0, VN, 120), (0.1, 0.5 * VN, 130)])
    line_open = max(opened.values())
    for v, angle in ((va, 0), (vb, -120), (vc, 120)):
        _set_from(v, t, line_open, 0.01 * VN, angle)

    s = _trace(_payload(t, {"A": ia, "B": ib, "C": ic}, {"A": va, "B": vb, "C": vc}))["summary"]
    assert s["fault_start_ms"] == pytest.approx(100.0, abs=2.0)
    assert s["high_current_phases"] == []
    assert s["sagged_phases"] == ["B", "C"]
    assert s["fct_ms"] == pytest.approx(max(opened["B"], opened["C"]) * 1000 - 100, abs=2.5)
    assert s["zero_current_pattern"] == "three_together"


def test_the_far_end_opening_first_stops_the_healthy_phase_first():
    """Cirata #2 (20 Nov 2024, first file) pattern: the healthy phase loses its
    load flow when the far end opens, while this end feeds the S-T fault until
    its own breaker opens 100 ms later. The clearing is the faulted phases'
    last zero, and the zeros are not one breaker's."""
    t = _t(0.5)
    ia = _piecewise(t, [(0, 400, -20)])
    ia, a_stop = _open_at_zero(ia, t, 0.15)
    ib = _piecewise(t, [(0, 400, -140), (0.1, 5000, -100)])
    ic = _piecewise(t, [(0, 400, 100), (0.1, 5000, 80)])
    ib, b0 = _open_at_zero(ib, t, 0.25)
    ic, c0 = _open_at_zero(ic, t, 0.25)
    va = _piecewise(t, [(0, VN, 0)])
    vb = _piecewise(t, [(0, VN, -120), (0.1, 0.5 * VN, -110)])
    vc = _piecewise(t, [(0, VN, 120), (0.1, 0.5 * VN, 130)])
    line_open = max(b0, c0)
    for v, angle in ((va, 0), (vb, -120), (vc, 120)):
        _set_from(v, t, line_open, 0.01 * VN, angle)

    s = _trace(_payload(t, {"A": ia, "B": ib, "C": ic}, {"A": va, "B": vb, "C": vc}))["summary"]
    assert s["ceased"]["A"] == pytest.approx(a_stop * 1000, abs=1.0)
    assert s["fault_clearing_ms"] == pytest.approx(max(b0, c0) * 1000, abs=1.0)
    assert s["zero_current_pattern"] == "staggered"
    assert s["fct_ms"] == pytest.approx(max(b0, c0) * 1000 - 100, abs=2.0)


def test_a_fault_after_a_reclose_is_a_refault():
    t = _t(1.0)
    currents, voltages, opened = {}, {}, {}
    for p, a in (("A", -20), ("B", -140), ("C", 100)):
        fault = p in ("B", "C")
        i = _piecewise(t, [(0, 400, a)] + ([(0.1, 6000, a + 40)] if fault else []))
        i, opened[p] = _open_at_zero(i, t, 0.17)
        _set_from(i, t, 0.6, 400, a)
        if fault:
            _set_from(i, t, 0.7, 6000, a + 40)
        i, _second = _open_at_zero(i, t, 0.77)
        currents[p] = i
        v = _piecewise(t, [(0, VN, a + 20)] + ([(0.1, 0.6 * VN, a + 30)] if fault else []))
        voltages[p] = v
    for p, a in (("A", 0), ("B", -120), ("C", 120)):
        v = voltages[p]
        _set_from(v, t, max(opened.values()), 0.01 * VN, a)
        _set_from(v, t, 0.6, VN, a)
        if p in ("B", "C"):
            _set_from(v, t, 0.7, 0.6 * VN, a + 10)
        _set_from(v, t, 0.78, 0.01 * VN, a)

    s = _trace(_payload(t, currents, voltages))["summary"]
    assert s["fault_start_ms"] == pytest.approx(100.0, abs=1.5)
    assert s["refault_ms"] == pytest.approx(700.0, abs=2.0)
    assert s["dead_time_ms"]["B"] == pytest.approx(600 - opened["B"] * 1000, abs=5.0)


def test_a_current_only_record_reads_the_fault_from_current():
    t = _t(0.4)
    ia = _piecewise(t, [(0, 500, -20), (0.1, 8000, -80)])
    ia, a_open = _open_at_zero(ia, t, 0.18)
    ib = _piecewise(t, [(0, 500, -140)])
    ic = _piecewise(t, [(0, 500, 100)])
    s = _trace(_payload(t, {"A": ia, "B": ib, "C": ic}, {}))["summary"]
    assert s["fault_start_ms"] == pytest.approx(100.0, abs=1.5)
    assert s["fct_ms"] == pytest.approx(a_open * 1000 - 100, abs=2.0)


def test_no_trace_without_phase_currents_or_samples():
    t = _t(0.2)
    assert trace_payload(_payload(t, {}, {"A": _sine(t, VN, 0)})) is None
    assert trace_payload({"time": [0.0, 0.001], "frequency": F, "analog_channels": []}) is None
