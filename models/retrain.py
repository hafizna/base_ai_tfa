"""Paired grouped CV, candidate training, and explicit metric-gated promotion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from models.feature_schema import FEATURE_COLS, FEATURE_VERSION, build_feature_frame
from models.rules import apply_rules
from models.train import ALL_CLASSES, build_class_weight


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_row(row):
    row = dict(row)
    for key in ("reclose_successful", "ct_anomaly_detected", "soe_phase_mismatch"):
        value = row.get(key)
        if value in ("True", "true"):
            row[key] = True
        elif value in ("False", "false"):
            row[key] = False
        elif value in ("", "nan"):
            row[key] = None
    return row


def eligible_rows(rows):
    return [normalized_row(r) for r in rows
            if r["label"] in ALL_CLASSES
            and str(r.get("scaling_ok")).lower() == "true"
            and str(r.get("duration_ok")).lower() == "true"
            and apply_rules(normalized_row(r)) is None]


def promotion_decision(candidate_score, baseline_score, candidate_per_class, baseline_per_class, *, max_regression=0.05):
    reasons = []
    if not np.isfinite(candidate_score) or not np.isfinite(baseline_score) or candidate_score <= baseline_score + 1e-6:
        reasons.append("Candidate F1-macro did not improve")
    for label in ALL_CLASSES:
        if candidate_per_class.get(label, 0) + max_regression < baseline_per_class.get(label, 0):
            reasons.append(f"Per-class F1 regressed beyond {max_regression}: {label}")
    return not reasons, reasons


def grouped_folds(y, groups, n_splits):
    """Choose a supported split using labels only, never model scores."""
    for shuffle, seed in [(True, 42), (False, None), *[(True, i) for i in range(20)]]:
        folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
                     .split(np.zeros(len(y)), y, groups))
        if all(set(y[train]) == set(ALL_CLASSES) and set(y[test]) == set(ALL_CLASSES) for train, test in folds):
            return folds, {"shuffle": shuffle, "random_state": seed, "n_splits": n_splits}
    return None, None


def evaluate_and_train(csv_path, candidate_path, incumbent_path=Path("models/fault_classifier.pkl"), report_path=None,
                       *, focus_transient_lines=True, petir_multiplier=0.80, non_petir_transient_multiplier=1.35):
    csv_path, candidate_path, incumbent_path = map(Path, (csv_path, candidate_path, incumbent_path))
    if candidate_path.resolve() == incumbent_path.resolve():
        raise ValueError("Training output must be a candidate path; use gated promotion for the active model")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    from models.build_dataset import pipeline_fingerprint
    reader_sha = pipeline_fingerprint()
    if any(r.get("pipeline_fingerprint") != reader_sha for r in rows):
        raise ValueError("Reader changed since extraction; rebuild the dataset before training")
    if not rows or any(r.get("feature_version") != FEATURE_VERSION or not r.get("record_fingerprint") for r in rows):
        raise ValueError("Rebuild dataset with python -m models.build_dataset before training")
    incumbent_bytes = incumbent_path.read_bytes()
    incumbent = pickle.loads(incumbent_bytes)
    rows = eligible_rows(rows)
    paired = [r for r in rows if json.loads(r.get("legacy_features_json") or "{}")]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_version": FEATURE_VERSION,
        "classes": ALL_CLASSES, "dataset_sha256": file_sha(csv_path),
        "baseline_model_sha256": hashlib.sha256(incumbent_bytes).hexdigest(),
        "pipeline_fingerprint": reader_sha,
        "comparison": "Paired out-of-fold CV: refitted incumbent estimator on legacy features vs candidate on shared features",
        "frozen_model_test": "Not claimed: historical training overlap is unknown; in-sample incumbent predictions are not a baseline",
        "eligible_rows": len(rows), "paired_rows": len(paired), "promotion_allowed": False, "reasons": [],
        "class_counts": {c: sum(r["label"] == c for r in rows) for c in ALL_CLASSES},
    }
    report_path = Path(report_path or candidate_path.with_suffix(".evaluation.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def finish():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = {k: v for k, v in report.items() if k not in {"folds", "candidate_class_report", "baseline_class_report"}}
        print(json.dumps(summary, indent=2), flush=True)
        return report

    groups_per_class = {c: len({r["event_group"] for r in paired if r["label"] == c}) for c in ALL_CLASSES}
    report["independent_groups_per_class"] = groups_per_class
    if min(groups_per_class.values()) < 2:
        report["reasons"] = ["Seven-class CV requires at least two independent paired groups per class"]
        return finish()
    y = np.array([r["label"] for r in paired])
    groups = np.array([r["event_group"] for r in paired])
    folds, split_config = grouped_folds(y, groups, min(5, min(groups_per_class.values())))
    if folds is None:
        report["reasons"] = ["At least one grouped training fold lacks a class; collect more independent labeled events"]
        return finish()
    report["cv_split_config"] = split_config
    X = build_feature_frame(paired)
    legacy = [normalized_row(json.loads(r["legacy_features_json"])) for r in paired]
    baseline_cols = incumbent.get("feature_cols", FEATURE_COLS)
    X_old = build_feature_frame(legacy, baseline_cols)
    candidate = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
                               random_state=42, n_jobs=2, verbose=-1)
    oof_new, oof_old = np.empty(len(y), dtype=object), np.empty(len(y), dtype=object)
    results = []
    for i, (train, test) in enumerate(folds):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Event leakage in CV split")
        weights = build_class_weight(pd.Series(y[train]), focus_transient_lines, petir_multiplier, non_petir_transient_multiplier)
        new = clone(candidate).set_params(class_weight=weights)
        old = clone(incumbent["clf"]).set_params(class_weight=weights, n_jobs=2)
        new.fit(X.iloc[train], y[train])
        old.fit(X_old.iloc[train], y[train])
        oof_new[test], oof_old[test] = new.predict(X.iloc[test]), old.predict(X_old.iloc[test])
        results.append({
            "fold": i, "test_record_ids": [paired[j]["record_fingerprint"] for j in test],
            "candidate_f1_macro": f1_score(y[test], oof_new[test], labels=ALL_CLASSES, average="macro", zero_division=0),
            "baseline_f1_macro": f1_score(y[test], oof_old[test], labels=ALL_CLASSES, average="macro", zero_division=0),
        })
        print(f"Completed paired fold {i + 1}/{len(folds)}", flush=True)
    new_classes = classification_report(y, oof_new, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    old_classes = classification_report(y, oof_old, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    score_new = float(np.mean([r["candidate_f1_macro"] for r in results]))
    score_old = float(np.mean([r["baseline_f1_macro"] for r in results]))
    allowed, reasons = promotion_decision(score_new, score_old,
        {c: new_classes[c]["f1-score"] for c in ALL_CLASSES}, {c: old_classes[c]["f1-score"] for c in ALL_CLASSES})
    if len(paired) != len(rows):
        allowed = False
        reasons.append("Unpaired new records need a comparable baseline before promotion")
    candidate.set_params(class_weight=build_class_weight(pd.Series([r["label"] for r in rows]),
        focus_transient_lines, petir_multiplier, non_petir_transient_multiplier))
    candidate.fit(build_feature_frame(rows), [r["label"] for r in rows])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "clf": candidate, "feature_cols": FEATURE_COLS, "feature_version": FEATURE_VERSION,
        "classes": list(candidate.classes_), "all_classes": ALL_CLASSES, "class_counts": report["class_counts"],
        "model_type": "multiclass_lightgbm", "training_profile": {
            "trained_at_utc": report["created_at_utc"], "dataset_sha256": report["dataset_sha256"],
            "record_ids": [r["record_fingerprint"] for r in rows],
            "cv": "StratifiedGroupKFold", "cv_split_config": split_config,
            "focus_transient_lines": focus_transient_lines, "petir_multiplier": petir_multiplier,
            "non_petir_transient_multiplier": non_petir_transient_multiplier,
        },
    }
    with candidate_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    report.update(folds=results, candidate_f1_macro=score_new, baseline_f1_macro=score_old,
                  candidate_class_report=new_classes, baseline_class_report=old_classes,
                  candidate_sha256=file_sha(candidate_path), promotion_allowed=allowed, reasons=reasons)
    return finish()


def promote_candidate(candidate_path, active_path, report_path):
    candidate_path, active_path = Path(candidate_path), Path(active_path)
    if candidate_path.resolve() == active_path.resolve():
        raise ValueError("Candidate and active paths must differ")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("promotion_allowed") is not True:
        raise ValueError("Promotion blocked: " + "; ".join(report.get("reasons", [])))
    candidate_bytes, active_bytes = candidate_path.read_bytes(), active_path.read_bytes()
    if hashlib.sha256(candidate_bytes).hexdigest() != report.get("candidate_sha256") or hashlib.sha256(active_bytes).hexdigest() != report.get("baseline_model_sha256"):
        raise ValueError("Candidate or incumbent changed since evaluation; rerun validation")
    from models.build_dataset import pipeline_fingerprint
    if report.get("pipeline_fingerprint") != pipeline_fingerprint():
        raise ValueError("Reader changed since evaluation; rebuild and reevaluate")
    backup = candidate_path.parent / ("incumbent-" + report["baseline_model_sha256"][:12] + ".pkl")
    backup.write_bytes(active_bytes)
    fd, staging = tempfile.mkstemp(prefix=active_path.name + ".", suffix=".staging", dir=active_path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(candidate_bytes)
        os.replace(staging, active_path)
    finally:
        if Path(staging).exists():
            Path(staging).unlink()
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/features/labeled_features_v2.csv"))
    parser.add_argument("--candidate", type=Path, default=Path("models/candidates/fault_classifier.pkl"))
    parser.add_argument("--active", type=Path, default=Path("models/fault_classifier.pkl"))
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    report = evaluate_and_train(args.csv, args.candidate, args.active)
    if args.promote:
        promote_candidate(args.candidate, args.active, args.candidate.with_suffix(".evaluation.json"))
        print("Promoted validated candidate. Restart/redeploy to load it.")
    elif not report["promotion_allowed"]:
        print("Active model retained; promotion gate did not pass.")


if __name__ == "__main__":
    main()
