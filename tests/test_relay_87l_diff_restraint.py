import math

from webapp.api.routers.relay_87l import (
    _compute_diff_restraint,
    _detect_diff_mode,
    _detect_terminal_pairs,
)


PARAMS = {
    "device_type": "agr-20-80",
    "idiff_pickup": 0.20,
    "slope1": 0.20,
    "intersection1": 0.0,
    "slope2": 0.80,
    "intersection2": 2.0,
    "idiff_fast": 7.50,
    "in_base_a": 0.0,
}


def _channel(name, canonical_name, samples):
    return {
        "name": name,
        "canonical_name": canonical_name,
        "samples": samples,
    }


def _status(name, n, first_on=None):
    samples = [0] * n
    if first_on is not None:
        for idx in range(first_on, n):
            samples[idx] = 1
    return {"name": name, "samples": samples}


def _payload(*, include_diff_trip=True, include_fast_trip=False):
    n = 220
    time = [idx / 1000.0 for idx in range(n)]
    base = [math.sin(2 * math.pi * 50 * t) for t in time]

    ia = base[:]
    ib = [0.95 * v for v in base]
    ic = [0.90 * v for v in base]
    ia_remote = [-v for v in ia]
    ib_remote = [-v for v in ib]
    ic_remote = [-v for v in ic]

    # A-phase internal differential near the relay trip time. This exceeds the
    # numeric I-DIFF>> setting, but the relay status below is only generic DIF-A.
    for idx in range(55, 95):
        ia[idx] = 12.0 * base[idx]
        ia_remote[idx] = 0.0

    # Later B/C imbalance must not back-classify the protection event phases.
    for idx in range(160, 200):
        ib_remote[idx] = 0.0
        ic_remote[idx] = 0.0

    statuses = []
    if include_diff_trip:
        statuses.append(_status("DIF-A_TRIP", n, 60))
    if include_fast_trip:
        statuses.append(_status("I-DIFF>> A", n, 60))

    return {
        "time": time,
        "frequency": 50.0,
        "analog_channels": [
            _channel("Ia", "IA", ia),
            _channel("Ib", "IB", ib),
            _channel("Ic", "IC", ic),
            _channel("Ia1", "Ia1", ia_remote),
            _channel("Ib1", "Ib1", ib_remote),
            _channel("Ic1", "Ic1", ic_remote),
        ],
        "status_channels": statuses,
    }


def test_87l_relay_diff_trip_does_not_infer_fast_without_fast_status():
    result = _compute_diff_restraint(_payload(), PARAMS)

    assert result["operated_status"] == "IDIFF_OPERATED"
    assert result["operated_phases"] == ["L1"]
    assert result["relay_diff_phases"] == ["L1"]

    by_phase = {row["phase"]: row for row in result["phase_classification"]}
    assert by_phase["L1"]["verdict"] == "Internal Fault"
    assert by_phase["L2"]["verdict"] == "Not Operated"
    assert by_phase["L3"]["verdict"] == "Not Operated"


def test_87l_fast_status_is_used_when_relay_reports_fast_element():
    result = _compute_diff_restraint(_payload(include_fast_trip=True), PARAMS)

    assert result["operated_status"] == "IDIFF_FAST_OPERATED"
    assert result["operated_phases"] == ["L1"]


def test_87l_can_still_infer_fast_when_no_relay_diff_status_exists():
    result = _compute_diff_restraint(_payload(include_diff_trip=False), PARAMS)

    assert result["operated_status"] == "IDIFF_FAST_OPERATED"
    assert "L1" in result["operated_phases"]


def test_87l_uses_relay_computed_idiff_and_ibias_when_available():
    n = 80
    time = [idx / 1000.0 for idx in range(n)]
    idiff_a = [0.05] * n
    ibias_a = [0.20] * n
    idiff_b = [0.05] * n
    ibias_b = [0.20] * n
    idiff_c = [0.05] * n
    ibias_c = [0.20] * n
    for idx in range(35, 55):
        idiff_b[idx] = 1.5
        ibias_b[idx] = 0.8

    payload = {
        "time": time,
        "frequency": 50.0,
        "analog_channels": [
            _channel("IDIFF_A", "IDIFF_A", idiff_a),
            _channel("IBIAS_A", "IBIAS_A", ibias_a),
            _channel("IDIFF_B", "IDIFF_B", idiff_b),
            _channel("IBIAS_B", "IBIAS_B", ibias_b),
            _channel("IDIFF_C", "IDIFF_C", idiff_c),
            _channel("IBIAS_C", "IBIAS_C", ibias_c),
        ],
        "status_channels": [_status("DIF-B_TRIP", n, 40)],
    }

    result = _compute_diff_restraint(payload, PARAMS)

    assert result["diff_data_mode"] == "TWO_TERMINAL"
    assert result["operated_status"] == "IDIFF_OPERATED"
    assert result["operated_phases"] == ["L2"]

    by_phase = {row["phase"]: row for row in result["phase_classification"]}
    assert by_phase["L2"]["max_idiff"] > 1.4
    assert by_phase["L2"]["max_irest"] > 0.75


def test_87l_abb_remote_current_names_are_not_misread_as_relay_idiff():
    n = 220
    time = [idx / 1000.0 for idx in range(n)]
    base_a = [math.sin(2 * math.pi * 50 * t) for t in time]
    base_b = [0.95 * math.sin(2 * math.pi * 50 * t - 2 * math.pi / 3) for t in time]
    base_c = [0.90 * math.sin(2 * math.pi * 50 * t + 2 * math.pi / 3) for t in time]

    payload = {
        "time": time,
        "frequency": 50.0,
        "analog_channels": [
            _channel("LINE IL1", "IA", base_a),
            _channel("LINE IL2", "IB", base_b),
            _channel("LINE IL3", "IC", base_c),
            # ABB/REL670 remote currents can be normalized to IDIFF_* by the
            # generic channel mapper. Raw name must win: these are remote line
            # currents, not relay-computed differential channels.
            _channel("REM L IL1 D", "IDIFF_A", [-v for v in base_a]),
            _channel("REM L IL2 D", "IDIFF_B", [-v for v in base_b]),
            _channel("REM L IL3 D", "IDIFF_C", [-v for v in base_c]),
        ],
        "status_channels": [],
    }

    mode = _detect_diff_mode(payload["analog_channels"], time, 50.0)
    pairs, sign = _detect_terminal_pairs(payload["analog_channels"], time, 50.0)
    result = _compute_diff_restraint(payload, PARAMS)

    assert mode == "TWO_TERMINAL_RAW"
    assert pairs is not None
    assert sorted(pairs) == ["L1", "L2", "L3"]
    assert sign == 1.0
    assert result["diff_data_mode"] == "TWO_TERMINAL_RAW"
    assert result["operated_status"] == "NOT_OPERATED"
    assert result["operated_phases"] == []
