import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from core.comtrade_parser import parse_comtrade
from models.build_dataset import (
    apply_feedback, build_dataset, corrected_window, identity, read_feedback, select_feedback,
    event_group, pipeline_fingerprint,
)
from models.feature_schema import build_feature_frame
from models.predict import _build_feature_vector
from models.retrain import promotion_decision, promote_candidate, evaluate_and_train, file_sha, grouped_folds
from models.train import ALL_CLASSES, load_and_prepare
from tests.fixtures.comtrade_writer import synthetic_cfg_dat_bytes
from tests.fixtures.synthetic_records import transient_slg_successful_reclose
from webapp.api.ml_predict import extract_ml_features
from webapp.api.record_payload import record_to_payload


def pair(tmp_path):
    cfg, dat = synthetic_cfg_dat_bytes()
    path = tmp_path / "test.cfg"
    path.write_bytes(cfg)
    path.with_suffix(".dat").write_bytes(dat)
    return path


def feedback(**kwargs):
    return {"ground_truth_confidence": "CONFIRMED", "include_for_training": True,
            "submitted_at_utc": "2026-10-11T00:00:00Z", **kwargs}


def test_training_inference_use_identical_encoding(tmp_path):
    payload = record_to_payload(parse_comtrade(str(pair(tmp_path))))
    row = extract_ml_features(payload)
    row.update(label="PETIR", scaling_ok=True, duration_ok=True, reclose_successful=None,
               thd_percent=float("nan"))
    path = tmp_path / "features.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, row.keys())
        writer.writeheader()
        writer.writerow(row)
    X, _, _ = load_and_prepare(path)
    assert_frame_equal(X.reset_index(drop=True), _build_feature_vector(row))
    assert_frame_equal(_build_feature_vector(row), build_feature_frame([row]))


def test_trusted_feedback_overrides_cause_and_phase_independently():
    row = apply_feedback({"faulted_phases": "A"}, feedback(actual_label="LAYANG",
        faulted_phases_correct=False, actual_faulted_phases=["T"]), "PETIR")
    assert row["label"] == "LAYANG" and row["label_source"] == "feedback"
    assert row["faulted_phases"] == "C"
    phases_only = apply_feedback({}, feedback(faulted_phases_correct=False, actual_faulted_phases=["C"]), "PETIR")
    assert phases_only["label"] == "PETIR" and phases_only["label_source"] == "folder"


@pytest.mark.parametrize("level", ["POSSIBLE", "UNKNOWN"])
def test_unconfirmed_feedback_does_not_replace_folder_labels(level):
    assert select_feedback([feedback(ground_truth_confidence=level, actual_label="LAYANG")]) is None


def test_latest_trusted_feedback_wins_but_opt_out_always_excludes():
    first = feedback(actual_label="PETIR")
    second = feedback(ground_truth_confidence="PROBABLE", actual_label="LAYANG", submitted_at_utc="2026-10-12T00:00:00Z")
    assert select_feedback([second, first])["actual_label"] == "LAYANG"
    third = feedback(include_for_training=False, ground_truth_confidence="UNKNOWN", submitted_at_utc="2026-10-13T00:00:00Z")
    with pytest.raises(ValueError, match="excluded"):
        apply_feedback({}, select_feedback([first, third]), "PETIR")


def test_corrected_timing_recomputes_waveform_features():
    payload = transient_slg_successful_reclose()
    time = np.array(payload["time"]) * 1000
    correction = feedback(inception_correct=False, corrected_inception_time_ms=float(time[400]),
                          clearing_correct=False, corrected_clearing_time_ms=float(time[500]))
    window = corrected_window(payload, correction)
    row = extract_ml_features(payload, event_window=window)
    assert window.inception_idx == 400 and window.clearing_idx == 500
    assert row["fault_duration_ms"] == pytest.approx(time[500] - time[400], abs=0.1)
    with pytest.raises(ValueError, match="Clearing"):
        corrected_window(payload, feedback(inception_correct=False, corrected_inception_time_ms=float(time[500]),
            clearing_correct=False, corrected_clearing_time_ms=float(time[400])))


@pytest.mark.parametrize("kwargs", [
    {"parsing_correct": False}, {"channel_mapping_correct": False},
    {"actual_label": "NO_FAULT"}, {"cause_correct": False},
    {"faulted_phases_correct": False, "actual_faulted_phases": ["D"]},
])
def test_invalid_or_unresolved_corrections_are_quarantined(kwargs):
    with pytest.raises(ValueError):
        apply_feedback({}, feedback(**kwargs), "PETIR")


def test_retained_bytes_join_feedback_across_renamed_copies(tmp_path):
    source = pair(tmp_path)
    retained = tmp_path / "archive" / "raw" / "retained"
    retained.mkdir(parents=True)
    for suffix in (".cfg", ".dat"):
        (retained / ("renamed" + suffix)).write_bytes(source.with_suffix(suffix).read_bytes())
    import hashlib
    manifest = {"analysis_id": "uuid", "files": [
        {"stored_name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in retained.iterdir()
    ]}
    (retained / "metadata.json").write_text(json.dumps(manifest))
    labels = tmp_path / "archive" / "labels"
    labels.mkdir()
    (labels / "feedback.jsonl").write_text(json.dumps(feedback(analysis_id="uuid", actual_label="LAYANG")) + "\n")
    corrections, _ = read_feedback(tmp_path / "archive")
    assert identity(source) in corrections
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(f"cfg_path,label\n{source},PETIR\n", encoding="utf-8")
    rows, report = build_dataset(inventory, tmp_path / "out.csv", tmp_path / "archive")
    assert len(rows) == 1 and rows[0]["label"] == "LAYANG"
    assert report["feedback_labeled_rows"] == 1
    (retained / "renamed.dat").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_feedback(tmp_path / "archive")


def test_promotion_requires_macro_improvement_without_large_class_regression():
    per_class = {c: 0.5 for c in ALL_CLASSES}
    assert promotion_decision(0.6, 0.5, per_class, per_class)[0]
    assert not promotion_decision(0.5, 0.5, per_class, per_class)[0]
    assert not promotion_decision(0.6, 0.5, {**per_class, "POHON": 0.1}, per_class)[0]


def test_failed_gate_and_changed_artifacts_never_replace_active_model(tmp_path):
    active, candidate, report = (tmp_path / p for p in ("active.pkl", "candidate.pkl", "report.json"))
    active.write_bytes(pickle.dumps({"model": "old"}))
    candidate.write_bytes(pickle.dumps({"model": "new"}))
    original = active.read_bytes()
    report.write_text(json.dumps({"promotion_allowed": False, "reasons": ["F1 declined"]}))
    with pytest.raises(ValueError, match="blocked"):
        promote_candidate(candidate, active, report)
    assert active.read_bytes() == original
    report.write_text(json.dumps({"promotion_allowed": True, "candidate_sha256": "wrong", "baseline_model_sha256": "wrong"}))
    with pytest.raises(ValueError, match="changed"):
        promote_candidate(candidate, active, report)
    assert active.read_bytes() == original


def test_event_group_keeps_nested_terminals_with_ddmmyyyy_date_together():
    a = Path("raw/2024/11/10. 20112024 STAR-RWALO PETIR/terminal-a/DR/a.cfg")
    b = Path("raw/2024/11/10. 20112024 STAR-RWALO PETIR/terminal-b/DR/b.cfg")
    assert event_group(a) == event_group(b)


def test_grouped_folds_keep_all_classes_and_related_records_together():
    y = np.array([c for c in ALL_CLASSES for _ in range(6)])
    groups = np.array([f"{c}-{i // 2}" for c in ALL_CLASSES for i in range(6)])
    folds, config = grouped_folds(y, groups, 3)
    assert folds is not None and config["n_splits"] == 3
    for train, test in folds:
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(y[train]) == set(y[test]) == set(ALL_CLASSES)


def test_retrain_rejects_stale_reader_and_active_output(tmp_path):
    dataset = tmp_path / "features.csv"
    dataset.write_text("label,feature_version,record_fingerprint,pipeline_fingerprint\nPETIR,v2.shared-2026-10,id,old\n")
    with pytest.raises(ValueError, match="Reader changed"):
        evaluate_and_train(dataset, tmp_path / "candidate.pkl", tmp_path / "active.pkl")
    with pytest.raises(ValueError, match="candidate path"):
        evaluate_and_train(dataset, tmp_path / "active.pkl", tmp_path / "active.pkl")


def test_evaluation_runs_candidate_and_baseline_on_identical_oof_records(tmp_path):
    from lightgbm import LGBMClassifier
    from models.feature_schema import FEATURE_VERSION, FEATURE_COLS
    rows = []
    for cls_idx, cause in enumerate(ALL_CLASSES):
        for idx in range(12):
            rows.append({
                "label": cause, "scaling_ok": True, "duration_ok": True,
                "record_fingerprint": f"{cause}-{idx}", "event_group": f"{cause}-{idx // 3}",
                "feature_version": FEATURE_VERSION, "pipeline_fingerprint": pipeline_fingerprint(),
                "fault_duration_ms": 10 + cls_idx * 100, "reclose_successful": True,
                "peak_fault_current_a": 1000 + cls_idx * 1000,
                "faulted_phases": "A", "legacy_features_json": json.dumps({"fault_duration_ms": 1}),
            })
    dataset = tmp_path / "features.csv"
    with dataset.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    active = tmp_path / "active.pkl"
    active.write_bytes(pickle.dumps({"clf": LGBMClassifier(n_estimators=10, n_jobs=1, verbose=-1), "feature_cols": FEATURE_COLS}))
    before = active.read_bytes()
    candidate = tmp_path / "candidate.pkl"
    report = evaluate_and_train(dataset, candidate, active)
    assert report["paired_rows"] == len(rows)
    assert len({key for fold in report["folds"] for key in fold["test_record_ids"]}) == len(rows)
    assert sum(len(fold["test_record_ids"]) for fold in report["folds"]) == len(rows)
    assert report["candidate_sha256"] == file_sha(candidate)
    assert active.read_bytes() == before


def test_passing_promotion_keeps_backup_and_installs_exact_validated_bytes(tmp_path):
    active, candidate, report = (tmp_path / p for p in ("active.pkl", "candidate.pkl", "report.json"))
    active.write_bytes(b"old")
    candidate.write_bytes(b"new")
    report.write_text(json.dumps({"promotion_allowed": True, "candidate_sha256": file_sha(candidate),
                                  "baseline_model_sha256": file_sha(active), "pipeline_fingerprint": pipeline_fingerprint()}))
    backup = promote_candidate(candidate, active, report)
    assert backup.read_bytes() == b"old" and active.read_bytes() == b"new"
