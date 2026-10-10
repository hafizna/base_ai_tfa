import json

from models.context_training import CONTEXT_FEATURES, reviewed_targets, train_context, build_context_dataset
from models.build_dataset import pipeline_fingerprint, identity
from tests.test_feedback_training_pipeline import pair


def test_context_targets_require_explicit_review_and_never_use_folder_causes():
    assert reviewed_targets({"actual_label": "PETIR", "ground_truth_confidence": "CONFIRMED"}) == {}
    assert reviewed_targets({"ground_truth_confidence": "UNKNOWN", "reclose_correct": False, "actual_reclose_outcome": "failed"}) == {}
    targets = reviewed_targets({"ground_truth_confidence": "CONFIRMED",
        "protection_interpretation_correct": False, "actual_event_class": "RECLOSE_REFAULT_SOTF",
        "event_segmentation_correct": False, "actual_episode_count": 2,
        "sotf_correct": False, "actual_sotf_after_reclose": True,
        "scheme_correct": False, "actual_scheme": "putt"})
    assert targets == {"sequence_class": "RECLOSE_REFAULT_SOTF", "episode_count": "2",
                       "sotf_after_reclose": "True", "scheme": "PUTT"}


def test_context_builder_deduplicates_and_one_case_does_not_train_a_model(tmp_path):
    source = pair(tmp_path)
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(f"cfg_path,label\n{source},PETIR\n{source},PETIR\n")
    annotations = tmp_path / "review.jsonl"
    annotations.write_text(json.dumps({"record_fingerprint": identity(source),
        "ground_truth_confidence": "CONFIRMED", "submitted_at_utc": "2026-10-11T00:00:00Z",
        "trip_type_correct": False, "actual_trip_type": "single_pole"}) + "\n")
    dataset = tmp_path / "context.jsonl"
    rows = build_context_dataset(inventory, dataset, annotations=annotations)
    assert len(rows) == 1
    assert "label" not in rows[0]
    candidate = tmp_path / "context.pkl"
    report = train_context(dataset, candidate)
    assert report["tasks"]["trip_type"]["status"] == "insufficient_reviewed_examples"
    assert not candidate.exists()


def test_learned_context_candidate_needs_independent_groups_and_beats_rules_baseline(tmp_path):
    rows = []
    for outcome, value in (("successful", 0), ("failed", 1)):
        for group in range(4):
            for repeat in range(6):
                vector = [0] * len(CONTEXT_FEATURES)
                vector[CONTEXT_FEATURES.index("refault")] = value
                rows.append({"event_group": f"{outcome}-{group}", "record_fingerprint": f"{outcome}-{group}-{repeat}",
                    "pipeline_fingerprint": pipeline_fingerprint(), "feature_columns": CONTEXT_FEATURES,
                    "features": vector, "targets": {"restoration_outcome": outcome},
                    "baseline": {"restoration_outcome": "unknown"}})
    dataset = tmp_path / "context.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    candidate = tmp_path / "context.pkl"
    report = train_context(dataset, candidate)
    assert report["tasks"]["restoration_outcome"]["promotion_allowed"]
    assert candidate.exists()
