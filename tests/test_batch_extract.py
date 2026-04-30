import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_extract import flatten_differential_features, infer_label
from core.feature_extractor import DifferentialFeatures


def test_flatten_differential_features_keeps_universal_and_87l_fields():
    feat = DifferentialFeatures(
        di_dt_max=1250.0,
        di_dt_phase="A",
        peak_fault_current_a=980.0,
        i0_i1_ratio=0.62,
        thd_percent=3.4,
        inception_angle_degrees=87.0,
        idiff_max_percent=140.0,
        irestraint_max_percent=210.0,
        idiff_rise_rate=25.0,
        reclose_attempted=True,
        reclose_successful=True,
        faulted_phases=["A"],
        fault_type="SLG",
        is_ground_fault=True,
        station_name="GI TEST",
        relay_model="7SA522",
        voltage_kv=150.0,
        sampling_rate_hz=1600.0,
    )
    prot = SimpleNamespace(
        primary_protection=SimpleNamespace(name="DIFFERENTIAL"),
        trip_type="single_pole",
        permission_received=True,
        comms_failure=False,
    )
    fault = SimpleNamespace(duration_ms=80.0, inception_time=0.12, reclose_events=[object()])
    record = SimpleNamespace(time=[0.0, 0.5])

    row = flatten_differential_features(feat, "PETIR", Path("dummy.cfg"), prot, fault, record)

    assert row["protection_type"] == "DIFFERENTIAL"
    assert row["trip_type"] == "single_pole"
    assert row["faulted_phases"] == "A"
    assert row["fault_count"] == 2
    assert row["voltage_sag_depth_pu"] is None
    assert row["voltage_phase_ratio_spread_pu"] is None
    assert row["v2_v1_ratio"] is None
    assert row["z_magnitude_ohms"] is None
    assert row["teleprotection_rx"] is True
    assert row["comms_failure"] is False
    assert row["idiff_max_percent"] == 140.0
    assert row["classification_status"].startswith("UNCLASSIFIED")


def test_infer_label_maps_isolator_to_peralatan():
    path = (
        r"C:\data\raw_data\UPT PURWOKERTO\2025\8. AGUSTUS"
        r"\01. 01082025 TRIP SUTT KSGHN-LMNIS ALAT ISOLATOR\DISTANCE\FR000037.cfg"
    )

    assert infer_label(path) == "PERALATAN"
