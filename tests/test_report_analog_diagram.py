"""Tests for the report-native Analog Time Diagram (ABB-style composite).

Covers channel selection (canonical ordering, dedup, fallback) and the
matplotlib renderer (returns a valid PNG), plus the tick formatters.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.api.routers.report import (
    _fmt_peak,
    _nice_peak,
    _render_analog_time_diagram,
    _select_analog_diagram_channels,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _sine(freq_hz: float, amp: float, time_s: np.ndarray) -> list:
    return (amp * np.sin(2 * np.pi * freq_hz * time_s)).tolist()


def _payload_with_canonicals() -> dict:
    time_s = np.arange(0, 0.3, 1 / 1200.0)  # 300 ms @ 1200 Hz
    def ch(cid, name, canon, unit, meas, amp):
        return {
            "id": cid, "name": name, "canonical_name": canon, "unit": unit,
            "measurement": meas, "samples": _sine(50.0, amp, time_s),
        }
    return {
        "time": time_s.tolist(),
        "trigger_time": 0.1,  # 100 ms
        "analog_channels": [
            ch("1", "LINE UL1", "VA", "kV", "voltage", 80),
            ch("2", "LINE UL2", "VB", "kV", "voltage", 80),
            ch("3", "LINE UL3", "VC", "kV", "voltage", 80),
            ch("4", "LINE IL1", "IA", "A", "current", 500),
            ch("5", "LINE IL2", "IB", "A", "current", 500),
            ch("6", "LINE IL3", "IC", "A", "current", 27000),
            ch("7", "LINE IN", "IN", "A", "current", 27000),
            ch("8", "AUX SIGNAL", "AUX", "V", "voltage", 1),  # not in order
        ],
        "status_channels": [],
    }


def test_select_orders_currents_then_voltages_and_drops_unranked():
    payload = _payload_with_canonicals()
    selected = _select_analog_diagram_channels(payload, "21")
    canon = [c["canonical_name"] for c in selected]
    # ABB-style order: phase currents, neutral current, then phase voltages.
    assert canon == ["IA", "IB", "IC", "IN", "VA", "VB", "VC"]
    assert "AUX" not in canon  # unranked channel excluded


def test_select_appends_diff_restraint_for_differential_relays():
    payload = _payload_with_canonicals()
    time_s = np.asarray(payload["time"])
    payload["analog_channels"].append({
        "id": "9", "name": "IDIFF L1", "canonical_name": "IDIFF_A",
        "unit": "A", "measurement": "current",
        "samples": _sine(50.0, 100, time_s),
    })
    selected = _select_analog_diagram_channels(payload, "87L")
    canon = [c["canonical_name"] for c in selected]
    assert "IDIFF_A" in canon


def test_select_falls_back_to_raw_channels_when_no_canonicals():
    time_s = np.arange(0, 0.2, 1 / 1200.0)
    payload = {
        "time": time_s.tolist(),
        "trigger_time": 0.05,
        "analog_channels": [
            {"id": "1", "name": "W1 IL1", "canonical_name": "W1 IL1", "unit": "A",
             "measurement": "current", "samples": _sine(50.0, 300, time_s)},
            {"id": "2", "name": "W1 UL1", "canonical_name": "W1 UL1", "unit": "kV",
             "measurement": "voltage", "samples": _sine(50.0, 70, time_s)},
            {"id": "3", "name": "W2 IL1", "canonical_name": "W2 IL1", "unit": "A",
             "measurement": "current", "samples": _sine(50.0, 300, time_s)},
        ],
        "status_channels": [],
    }
    selected = _select_analog_diagram_channels(payload, "87T")
    assert len(selected) == 3  # fallback grabbed all by type


def test_select_caps_channel_count():
    time_s = np.arange(0, 0.1, 1 / 1200.0)
    channels = [
        {"id": str(i), "name": f"CH{i}", "canonical_name": f"X{i}", "unit": "A",
         "measurement": "current", "samples": _sine(50.0, 100, time_s)}
        for i in range(40)
    ]
    payload = {"time": time_s.tolist(), "analog_channels": channels, "status_channels": []}
    selected = _select_analog_diagram_channels(payload, "OCR")
    assert len(selected) <= 14


def test_render_returns_valid_png():
    payload = _payload_with_canonicals()
    selected = _select_analog_diagram_channels(payload, "21")
    center_ms = float(payload["trigger_time"]) * 1000.0
    raw = _render_analog_time_diagram(payload, selected, center_ms)
    assert isinstance(raw, bytes) and len(raw) > 1000
    assert raw[:8] == PNG_MAGIC


def test_render_empty_when_no_channels_or_time():
    assert _render_analog_time_diagram({"time": []}, [], 0.0) == b""
    payload = _payload_with_canonicals()
    assert _render_analog_time_diagram(payload, [], 0.0) == b""


def test_nice_peak_rounds_up_to_clean_values():
    assert _nice_peak(0) == 1.0
    assert _nice_peak(1.0) == 1.0
    assert _nice_peak(1.2) == 1.5
    assert _nice_peak(18777.0) == 20000.0
    assert _nice_peak(-450.0) == 500.0


def test_fmt_peak_is_compact():
    assert _fmt_peak(40) == "40"
    assert _fmt_peak(200) == "200"
    assert _fmt_peak(27000) == "27k"
    assert _fmt_peak(1.5) == "1.5"
