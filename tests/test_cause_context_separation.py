import numpy as np

from core.cause_context import episode_signatures
from core.event_analysis import build_event_window
from models.rules import apply_rules
from tests.test_single_record_context import full_reclose_refault_payload
from webapp.api import ml_predict
from webapp.api.record_analysis import build_record_analysis
from webapp.api.routers.relay_21 import _compute_electrical_params


class CauseModel:
    classes_ = np.array(["PETIR", "POHON", "HEWAN", "LAYANG", "BENDA_ASING", "KONDUKTOR", "PERALATAN"])
    calls = 0

    def predict(self, values):
        assert list(values.columns) == ["peak_fault_current_a"]
        return np.array(["POHON"])

    def predict_proba(self, values):
        self.calls += 1
        return np.array([[.04, .75, .05, .04, .04, .04, .04]])


def test_failed_reclose_keeps_cause_inference_and_separate_confidences(monkeypatch):
    model = CauseModel()
    monkeypatch.setattr(ml_predict, "_load_model", lambda: {"clf": model,
        "feature_cols": ["peak_fault_current_a"], "class_counts": {c: 10 for c in model.classes_}})
    payload = full_reclose_refault_payload()
    result = ml_predict.run_ml_prediction(payload)
    canonical = build_record_analysis("test", payload)
    assert model.calls == 1
    assert len(result["cause_ranking"]) == 7
    assert result["cause_ranking"][0]["cause"] == "POHON"
    assert result["raw_probabilities"] is not None
    assert result["event_context"]["rule"]["scope"] == "context"
    assert result["event_context"]["restoration_outcome"] == "failed"
    assert result["event_context"]["rule"]["confidence"] == .9
    assert result["overall_confidence"] != .9
    assert result["cause_assessment"]["confirmed"] is False
    assert result["cause_assessment"]["additional_episodes_evaluated_by_model"] is False
    # API and canonical incident snapshot share exactly the same signatures.
    assert result["episode_signatures"] == canonical.episode_signatures
    assert len(result["episode_signatures"]["episodes"]) == 2


def test_missing_model_does_not_invent_uniform_cause_probabilities(monkeypatch):
    monkeypatch.setattr(ml_predict, "_load_model", lambda: None)
    result = ml_predict.run_ml_prediction(full_reclose_refault_payload())
    assert result["cause_ranking"] == []
    assert result["overall_confidence"] == 0
    assert result["cause_assessment"]["status"] == "unavailable"
    assert result["event_context"]["restoration_outcome"] == "failed"


def test_two_phases_and_repetition_are_not_a_mechanical_cause_override():
    rule = apply_rules({"fault_count": 2, "faulted_phases": "A+B", "fault_duration_ms": 100,
                        "peak_fault_current_a": 5000, "reclose_successful": False})
    assert rule.scope == "context"


def test_second_episode_uses_initial_healthy_reference_not_dead_time():
    payload = full_reclose_refault_payload()
    signatures = episode_signatures(payload, build_event_window(payload))
    first, second = signatures["episodes"]
    assert second["reference"] == "first_episode_prefault"
    first_b = next(ch for ch in first["channels"] if ch["name"] == "IB")
    second_b = next(ch for ch in second["channels"] if ch["name"] == "IB")
    assert first_b["prefault_rms"] == second_b["prefault_rms"] > 0
    assert second_b["cycles"]
    assert signatures["used_by_deployed_model"] is False


def test_fault_b_measurements_do_not_report_healthy_phase_a_impedance():
    result = _compute_electrical_params(full_reclose_refault_payload())
    assert result["reference_phase"] == "B"
    assert result["v_sag_pct"] > 40
    assert result["impedance_kind"] == "phase_apparent_uncompensated"
    assert result["z_min_ohm"] > 0  # no zero-crossing instantaneous V/I minimum


def test_receive_after_trip_is_not_trip_permission():
    payload = full_reclose_refault_payload()
    time = np.array(payload["time"])
    payload["status_channels"].append({"name": "Distance Receive",
        "samples": ((time >= .515) & (time < .55)).astype(int).tolist()})
    analysis = build_record_analysis("late-receive", payload)
    trip = next(row for row in analysis.reasoning["conclusions"] if row["key"] == "trip_path")
    assert trip["value"]["receive_recorded"] is True
    assert trip["value"]["receive_at_trip"] is None
    assert not any("bersamaan dengan Distance Receive" in evidence for evidence in trip["evidence"])


def test_three_pole_trip_does_not_reweight_petir_cause(monkeypatch):
    class LightningModel(CauseModel):
        def predict(self, values):
            return np.array(["PETIR"])

        def predict_proba(self, values):
            return np.array([[.75, .04, .05, .04, .04, .04, .04]])

    monkeypatch.setattr(ml_predict, "_load_model", lambda: {"clf": LightningModel(),
        "feature_cols": ["peak_fault_current_a"], "class_counts": {c: 10 for c in CauseModel.classes_}})
    payload = full_reclose_refault_payload()
    original = ml_predict.extract_ml_features
    monkeypatch.setattr(ml_predict, "extract_ml_features", lambda value, relay_type="21": {
        **original(value, relay_type), "digital_cb_open_phases": ["A", "B", "C"], "digital_ar_lockout": True})
    result = ml_predict.run_ml_prediction(payload)
    assert "petir_digital_caution" not in [cap["name"] for cap in result["applied_caps"]]
    animal = next(item for item in result["cause_ranking"] if item["cause"] == "HEWAN")
    assert animal["confidence"] == round(result["calibrated_probabilities"]["HEWAN"], 3)
    assert not any("Mekanisme petir terindikasi" in item["text"] for item in result["evidence"])
