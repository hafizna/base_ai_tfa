import numpy as np

from core.event_analysis import build_event_window
from webapp.api.ml_predict import extract_ml_features, run_ml_prediction
from webapp.api.record_analysis import build_record_analysis


def full_reclose_refault_payload():
    sr, duration = 1200, 2.8
    time = np.arange(int(sr * duration)) / sr
    analog = []
    for phase, angle in zip("ABC", (0, -2 * np.pi / 3, 2 * np.pi / 3)):
        current = 200 * np.sin(2 * np.pi * 50 * time + angle)
        voltage = 120000 * np.sin(2 * np.pi * 50 * time + angle)
        if phase == "B":
            first = (time >= 0.5) & (time < 0.57)
            second = (time >= 1.6) & (time < 1.67)
            current[first | second] *= 40
            voltage[first | second] *= 0.4
            current[(time >= 0.57) & (time < 1.6)] = 0
            voltage[(time >= 0.57) & (time < 1.6)] = 0
        current[time >= 1.67] = 0
        voltage[time >= 1.67] = 0
        for prefix, samples, unit, measurement in (("I", current, "A", "current"), ("V", voltage, "V", "voltage")):
            analog.append({"name": prefix + phase, "canonical_name": prefix + phase, "phase": phase,
                           "unit": unit, "measurement": measurement, "samples": samples.tolist()})
    def pulse(start, end):
        return ((time >= start) & (time < end)).astype(int).tolist()
    b = np.ones(len(time), dtype=int)
    b[(time >= 0.55) & (time < 1.6)] = 0
    b[time >= 1.65] = 0
    statuses = [
        {"name": "DIST Trip B", "samples": pulse(0.51, 0.585)},
        {"name": "Any Trip", "samples": (np.array(pulse(0.51, 0.59)) | np.array(pulse(1.61, 1.69))).tolist()},
        {"name": "CB Aux B (52-A)", "samples": b.tolist()},
        {"name": "A/R Close", "samples": pulse(1.5, 1.61)},
        {"name": "Any Pole Dead", "samples": (np.array(pulse(0.552, 1.6)) | np.array(pulse(1.65, duration))).tolist()},
        {"name": "SOTF/TOR Trip", "samples": pulse(1.61, 1.69)},
    ]
    return {"station_name": "TEST", "frequency": 50, "time": time.tolist(),
            "analog_channels": analog, "status_channels": statuses}


def test_one_record_preserves_two_episodes_and_sotf_after_reclose():
    payload = full_reclose_refault_payload()
    window = build_event_window(payload)
    features = extract_ml_features(payload)
    analysis = build_record_analysis("whole-record", payload)
    assert 40 < window.fault_duration_ms < 100  # excludes ~1 s dead time and the later fault
    assert len(window.fault_episodes) == 2
    assert features["fault_count"] == 2
    assert features["reclose_successful"] is False
    assert features["digital_ar_dead_time_ms"] > 1000
    assert window.sequence["mechanical_close_confirmed"] is True
    assert window.sequence["restoration_outcome"] == "failed"
    assert window.sequence["sotf_after_reclose"] is True
    assert window.sequence["refault_after_reclose"] is True
    assert len(analysis.fault_episodes) == 2
    assert analysis.protection_interpretation["event_class"] == "PERMANENT_LINE_FAULT"
    row = next(c for c in analysis.reasoning["conclusions"] if c["key"] == "trip_reclose")
    assert "SOTF/TOR" in row["title"]
    assert row["value"]["sequence"]["restoration_outcome"] == "failed"
    result = run_ml_prediction(payload)
    assert result["fault_type"] == "permanent"
    assert any("SOTF/TOR" in e["text"] for e in result["evidence"])


def test_reclose_command_without_a_contact_return_does_not_confirm_closure():
    payload = full_reclose_refault_payload()
    payload["status_channels"] = [ch for ch in payload["status_channels"] if ch["name"] in {"DIST Trip B", "A/R Close"}]
    window = build_event_window(payload)
    assert not window.sequence.get("mechanical_close_confirmed")
    assert not window.sequence.get("sotf_after_reclose")
