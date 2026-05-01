import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rules import apply_rules
from models.predict import _augment_row_with_soe_context


def test_ct_measurement_anomaly_routes_to_peralatan():
    row = {
        "fault_count": 1,
        "faulted_phases": "A+B",
        "fault_type": "DLG",
        "fault_duration_ms": 448.5,
        "reclose_successful": None,
        "trip_type": "unknown",
        "zone_operated": "UNKNOWN",
        "peak_fault_current_a": 560.0,
        "ct_anomaly_detected": True,
        "ct_anomaly_evidence": (
            "arus fase A naik stabil (277 A RMS, gain 4.6x, CV 0.014) "
            "sementara tegangan fase tersebut tetap 0.85 pu"
        ),
    }

    result = apply_rules(row)

    assert result is not None
    assert result.label == "PERALATAN / PROTEKSI"
    assert result.rule_name == "ct_measurement_anomaly"
    assert "daripada sambaran petir" in result.evidence


def test_soe_waveform_phase_mismatch_routes_to_peralatan():
    row = {
        "fault_count": 1,
        "faulted_phases": "A+B+C",
        "fault_type": "3PH",
        "fault_duration_ms": 105.0,
        "reclose_successful": True,
        "trip_type": "three_pole",
        "zone_operated": "UNKNOWN",
        "peak_fault_current_a": 24040.0,
        "soe_faulted_phases": "B+C",
        "soe_phase_hint_source": "Distance Loop L23 selected forward",
        "soe_phase_mismatch": True,
        "voltage_phase_ratio_spread_pu": 0.31,
        "healthy_phase_voltage_ratio": 0.99,
        "v2_v1_ratio": 0.22,
    }

    result = apply_rules(row)

    assert result is not None
    assert result.label == "PERALATAN / PROTEKSI"
    assert result.rule_name == "soe_waveform_phase_mismatch"
    assert "jangan terima label PETIR" in result.evidence


def test_augment_row_with_soe_context_reconstructs_two_phase_pickup_pair():
    row = {"fault_type": "3PH"}
    soe = [
        {"channel": "Dis.Pickup L2", "state": 1, "rel_ms": 0.0},
        {"channel": "Dis.Pickup L3", "state": 1, "rel_ms": 0.0},
        {"channel": "Relay TRIP L1", "state": 1, "rel_ms": 1.0},
        {"channel": "Relay TRIP L2", "state": 1, "rel_ms": 1.0},
        {"channel": "Relay TRIP L3", "state": 1, "rel_ms": 1.0},
    ]

    updated = _augment_row_with_soe_context(row, soe)

    assert updated["soe_faulted_phases"] == "B+C"
    assert updated["soe_phase_mismatch"] is True
