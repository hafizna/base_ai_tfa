import math

from webapp.api.routers.relay_87t import _compute_87t


PARAMS = {
    "device_type": "std-30-70",
    "idiff_pickup": 0.20,
    "slope1": 0.30,
    "intersection1": 0.30,
    "slope2": 0.70,
    "intersection2": 2.50,
    "idiff_fast": 7.50,
    "in_base_a": 0.0,
}


def _channel(name, canonical_name, samples):
    return {
        "name": name,
        "canonical_name": canonical_name,
        "measurement": "current",
        "samples": samples,
    }


def _payload(hv_names, lv_names):
    n = 220
    time = [idx / 1000.0 for idx in range(n)]
    waves = {
        "L1": [math.sin(2 * math.pi * 50 * t) for t in time],
        "L2": [math.sin(2 * math.pi * 50 * t - 2 * math.pi / 3) for t in time],
        "L3": [math.sin(2 * math.pi * 50 * t + 2 * math.pi / 3) for t in time],
    }
    phases = ["L1", "L2", "L3"]
    channels = []
    for phase, name in zip(phases, hv_names):
        channels.append(_channel(name, name, waves[phase]))
    for phase, name in zip(phases, lv_names):
        channels.append(_channel(name, name, [0.98 * v for v in waves[phase]]))
    return {
        "time": time,
        "frequency": 50.0,
        "analog_channels": channels,
        "status_channels": [],
    }


def test_87t_detects_abb_winding_current_names_as_hv_lv():
    result = _compute_87t(
        _payload(
            ["IW1A", "IW1B", "IW1C"],
            ["IW2A", "IW2B", "IW2C"],
        ),
        PARAMS,
    )

    assert result["diff_data_mode"] == "TWO_TERMINAL"
    assert result["samples"]


def test_87t_detects_sel_winding_current_names_as_hv_lv():
    result = _compute_87t(
        _payload(
            ["IAW1", "IBW1", "ICW1"],
            ["IAW2", "IBW2", "ICW2"],
        ),
        PARAMS,
    )

    assert result["diff_data_mode"] == "TWO_TERMINAL"
    assert result["samples"]

