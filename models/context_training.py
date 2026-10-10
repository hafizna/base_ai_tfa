"""Reviewed context targets are separate from physical-cause labels.

Run with --build to regenerate the reviewed context dataset. Training requires
independent examples of multiple outcomes; one case is a regression, not proof
that a learned context classifier generalizes.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from models.build_dataset import identity, raw_files, read_feedback, select_feedback, pipeline_fingerprint, event_group
from webapp.api.record_payload import record_to_payload
from webapp.api.record_analysis import build_record_analysis
from core.comtrade_parser import parse_comtrade
from core.cff_parser import parse_cff_bytes

CONTEXT_FEATURES = ["episode_count", "closure_confirmed", "refault", "sotf", "first_fct_ms", "last_fct_ms", "initial_phase_count",
                    "trip_delay_ms", "zone1_at_trip", "zone2_at_trip", "zone3_at_trip",
                    "receive_at_trip", "send_at_trip", "ar_command_before_close",
                    *[f"current_gain_{p}" for p in "ABC"], *[f"voltage_ratio_{p}" for p in "ABC"]]


def context_vector(analysis):
    seq = analysis.event_window.sequence if analysis.event_window else {}
    episodes = analysis.fault_episodes
    operations = analysis.protection_operations
    origin = analysis.event_window.inception_time_ms if analysis.event_window else None
    trip_times = [t for op in operations if op.get("role") == "trip" for t in op.get("on_ms", []) if origin is not None and t >= origin]
    trip = min(trip_times) if trip_times else None
    def active_at_trip(op):
        if trip is None:
            return False
        return any(t <= trip + 10 and not any(t < off < trip - 10 for off in op.get("off_ms", [])) for t in op.get("on_ms", []))
    measured = analysis.electrical_measurements
    pre = measured.get("prefault") or {}
    fault = measured.get("fault") or {}
    gains = []
    for field, before, after in (("current", "current_rms", "current_peak"), ("voltage", "voltage_rms", "voltage_rms_min")):
        for phase in "ABC":
            denominator = float((pre.get(before) or {}).get(phase) or 0)
            numerator = float((fault.get(after) or {}).get(phase) or 0)
            gains.append(numerator / denominator if denominator > 0 else 0)
    return [len(episodes), int(seq.get("mechanical_close_confirmed", False)),
            int(seq.get("refault_after_reclose", False)), int(seq.get("sotf_after_reclose", False)),
            float(episodes[0].get("fault_duration_ms") or 0) if episodes else 0,
            float(episodes[-1].get("fault_duration_ms") or 0) if episodes else 0,
            len(episodes[0].get("faulted_phases") or []) if episodes else 0,
            trip - origin if trip is not None and origin is not None else 0,
            *[int(any(op.get("role") == "zone" and op.get("zone") == z and active_at_trip(op) for op in operations)) for z in (1, 2, 3)],
            int(any(op.get("role") == "teleprotection" and any(s in op.get("name", "").upper() for s in ("RECV", "RCV", "RECEIVE")) and active_at_trip(op) for op in operations)),
            int(any(op.get("role") == "teleprotection" and "SEND" in op.get("name", "").upper() and active_at_trip(op) for op in operations)),
            int(any(op.get("role") == "reclose" and any(t <= min(seq.get("reclose_times_ms") or [float("inf")]) for t in op.get("on_ms", [])) for op in operations)), *gains]


def reviewed_targets(feedback):
    """Never label from an unreviewed prediction, folder cause, or notes."""
    if feedback.get("ground_truth_confidence") not in {"CONFIRMED", "PROBABLE"}:
        return {}
    targets = {}
    for flag, field, target in (
        ("protection_interpretation_correct", "actual_event_class", "event_class"),
        ("reclose_correct", "actual_reclose_outcome", "restoration_outcome"),
        ("event_segmentation_correct", "actual_episode_count", "episode_count"),
        ("trip_type_correct", "actual_trip_type", "trip_type"),
        ("faulted_phases_correct", "actual_faulted_phases", "phases"),
        ("scheme_correct", "actual_scheme", "scheme"),
        ("trip_path_correct", "actual_trip_path", "trip_path"),
        ("sotf_correct", "actual_sotf_after_reclose", "sotf_after_reclose"),
    ):
        if feedback.get(flag) is False and feedback.get(field) not in (None, "", []):
            value = feedback[field]
            targets[target] = "+".join(sorted(value)) if isinstance(value, list) else str(value)
        elif feedback.get(flag) is True:
            snapshot = feedback.get("canonical_analysis_snapshot") or {}
            window = snapshot.get("event_window") or {}
            seq = window.get("sequence") or {}
            conclusions = {r["key"]: r for r in (snapshot.get("reasoning") or {}).get("conclusions", [])}
            values = {
                "event_class": (snapshot.get("protection_interpretation") or {}).get("event_class"),
                "restoration_outcome": seq.get("restoration_outcome"),
                "episode_count": len(snapshot["fault_episodes"]) if "fault_episodes" in snapshot else None,
                "phases": "+".join(window.get("faulted_phases") or []),
                "scheme": (conclusions.get("scheme", {}).get("value") or {}).get("scheme"),
                "trip_path": (conclusions.get("trip_path", {}).get("value") or {}).get("kind"),
                "sotf_after_reclose": seq.get("sotf_after_reclose"),
                "trip_type": {"1P": "single_pole", "3P": "three_pole"}.get((conclusions.get("trip_reclose", {}).get("value") or {}).get("trip_mode")),
            }
            value = values.get(target)
            if value is not None and value != "":
                targets[target] = str(value)
    sequence_classes = {"RECLOSE_REFAULT_SOTF", "RECLOSE_REFAULT", "RECLOSE_FAILED", "RECLOSE_SUCCESSFUL"}
    if targets.get("event_class") in sequence_classes:
        targets["sequence_class"] = targets.pop("event_class")
    if "scheme" in targets:
        targets["scheme"] = targets["scheme"].strip().upper()
    for task in ("trip_path", "trip_type", "restoration_outcome"):
        if task in targets:
            targets[task] = targets[task].strip().lower()
    if "phases" in targets:
        aliases = {"R": "A", "S": "B", "T": "C", "A": "A", "B": "B", "C": "C"}
        targets["phases"] = "+".join(sorted({aliases[p] for p in targets["phases"].upper().split("+")}))
    return targets


def build_context_dataset(inventory, output, training_dir=None, annotations=Path("config/context_annotations.jsonl")):
    sources = [Path(r["cfg_path"]) for r in csv.DictReader(Path(inventory).open(encoding="utf-8-sig"))]
    feedback, retained = read_feedback(training_dir)
    sources += [r["path"] for r in retained]
    if annotations and Path(annotations).exists():
        for line in Path(annotations).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                feedback.setdefault(row["record_fingerprint"], []).append(row)
    selected = {key: select_feedback(rows) for key, rows in feedback.items()}
    rows, seen = [], set()
    reader_sha = pipeline_fingerprint()
    for path in sources:
        try:
            key = identity(path)
        except (OSError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        correction = selected.get(key)
        if not correction or correction.get("include_for_training") is False:
            continue
        targets = reviewed_targets(correction)
        if not targets:
            continue
        record = parse_cff_bytes(path.read_bytes(), path.name) if path.suffix.lower() == ".cff" else parse_comtrade(str(path))
        analysis = build_record_analysis(key, record_to_payload(record))
        conclusions = {r["key"]: r for r in analysis.reasoning.get("conclusions", [])}
        rows.append({"record_fingerprint": key, "event_group": event_group(path),
                     "pipeline_fingerprint": reader_sha, "features": context_vector(analysis),
                     "feature_columns": CONTEXT_FEATURES,
                     "targets": targets, "confidence": correction["ground_truth_confidence"],
                     "source": correction.get("ground_truth_source", []),
                     "baseline": {"event_class": analysis.protection_interpretation.get("event_class"),
                                  "sequence_class": analysis.event_window.sequence.get("interpretation") if analysis.event_window else None,
                                  "episode_count": str(len(analysis.fault_episodes)),
                                  "restoration_outcome": analysis.event_window.sequence.get("restoration_outcome") if analysis.event_window else None}})
        rows[-1]["baseline"].update(
            scheme=(conclusions.get("scheme", {}).get("value") or {}).get("scheme"),
            trip_path=(conclusions.get("trip_path", {}).get("value") or {}).get("kind"),
            sotf_after_reclose=str(analysis.event_window.sequence.get("sotf_after_reclose")) if analysis.event_window else None,
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def train_context(dataset, output):
    rows = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    reader_sha = pipeline_fingerprint()
    if any(row["pipeline_fingerprint"] != reader_sha for row in rows):
        raise ValueError("Rebuild reviewed context dataset after reader changes")
    if any(row.get("feature_columns") != CONTEXT_FEATURES for row in rows):
        raise ValueError("Rebuild reviewed context dataset after feature schema changes")
    reports, models = {}, {}
    for task in sorted({key for row in rows for key in row["targets"]}):
        samples = [r for r in rows if task in r["targets"]]
        y = np.array([r["targets"][task] for r in samples])
        groups = np.array([r["event_group"] for r in samples])
        labels = sorted(set(y))
        group_support = {label: len(set(groups[y == label])) for label in labels}
        if len(labels) < 2 or min(group_support.values()) < 2:
            reports[task] = {"status": "insufficient_reviewed_examples", "records": len(samples), "group_support": group_support}
            continue
        X = pd.DataFrame([r["features"] for r in samples], columns=CONTEXT_FEATURES, dtype=float)
        folds = list(StratifiedGroupKFold(min(5, min(group_support.values())), shuffle=False).split(X, y, groups))
        if any(set(y[train]) != set(labels) for train, _ in folds):
            reports[task] = {"status": "insufficient_independent_folds"}
            continue
        predictions = np.empty(len(y), dtype=object)
        clf = LGBMClassifier(n_estimators=100, max_depth=3, min_child_samples=3, class_weight="balanced", n_jobs=2, verbosity=-1)
        for train, test in folds:
            clf.fit(X.iloc[train], y[train])
            predictions[test] = clf.predict(X.iloc[test])
        baseline = np.array([str(r["baseline"].get(task) or "unknown") for r in samples])
        score = f1_score(y, predictions, labels=labels, average="macro", zero_division=0)
        base_score = f1_score(y, baseline, labels=labels, average="macro", zero_division=0)
        allowed = score > base_score + 1e-6
        reports[task] = {"status": "evaluated", "f1_macro": score, "rules_baseline_f1_macro": base_score,
                         "promotion_allowed": bool(allowed), "records": len(samples)}
        if allowed:
            clf.fit(X, y)
            models[task] = clf
    report = {"reviewed_records": len(rows), "tasks": reports,
              "note": "Learned context candidates never replace measured sequence/rules automatically."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).with_suffix(".evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if models:
        Path(output).write_bytes(pickle.dumps({"models": models, "features": CONTEXT_FEATURES, "pipeline_fingerprint": reader_sha}))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("data/features/labeled_features.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training-runs/context.jsonl"))
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("models/candidates/context.pkl"))
    args = parser.parse_args()
    if args.build:
        build_context_dataset(args.inventory, args.dataset, args.training_dir)
    train_context(args.dataset, args.out)


if __name__ == "__main__":
    main()
