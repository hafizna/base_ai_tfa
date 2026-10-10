"""Re-extract a corpus using the live analysis reader and trusted feedback.

Run: python -m models.build_dataset --inventory data/features/labeled_features.csv
Optional: --training-dir <extracted training archive> --corpus-root <raw directory>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.cff_parser import parse_cff_bytes
from core.comtrade_parser import parse_comtrade
from core.event_analysis import build_event_window
from core.record_identity import fingerprint_files
from models.feature_schema import FEATURE_VERSION
from models.train import ALL_CLASSES
from webapp.api.fault_detection import detect_fault_presence
from webapp.api.ml_predict import extract_ml_features
from webapp.api.record_payload import record_to_payload

TRUSTED = {"CONFIRMED", "PROBABLE"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_files(path):
    if path.suffix.lower() == ".cff":
        return [path]
    dat = next((p for p in (path.with_suffix(".dat"), path.with_suffix(".DAT")) if p.exists()), None)
    if dat is None:
        raise ValueError("Missing matching DAT")
    return [path, dat]


def identity(path):
    return fingerprint_files([{"suffix": p.suffix.lower(), "sha256": sha(p)} for p in raw_files(path)])


def event_group(path):
    # Both terminals in one incident directory must stay in one CV fold.
    for i, part in enumerate(path.parts[:-1]):
        if re.search(r"(?:19|20)\d{6}|(?<!\d)\d{4}(?:19|20)\d{2}(?!\d)", part):
            return str(Path(*path.parts[:i + 1])).lower()
    return str(path.parent).lower()


def pipeline_fingerprint():
    """Refuse stale datasets after reader/rule/encoding code changes."""
    root = Path(__file__).resolve().parents[1]
    files = list((root / "core").glob("*.py")) + [
        Path(__file__), root / "models/feature_schema.py",
        root / "webapp/api/ml_predict.py", root / "webapp/api/record_payload.py",
        root / "webapp/api/fault_detection.py", root / "webapp/api/fault_reasoning.py",
        root / "webapp/api/record_analysis.py", root / "webapp/api/record_facts.py",
        root / "webapp/api/routers/relay_21.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def assign_event_groups(rows):
    """Unite incident aliases and identical DAT bytes across copied CFGs."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    keys_by_row = []
    for row in rows:
        keys = ["event:" + event_group(Path(p)) for p in json.loads(row["aliases"])]
        paths = raw_files(Path(row["cfg_path"]))
        data = next((p for p in paths if p.suffix.lower() in {".dat", ".cff"}), paths[0])
        keys.append("data:" + sha(data))
        for key in keys[1:]:
            first, other = find(keys[0]), find(key)
            if first != other:
                parents[max(first, other)] = min(first, other)
        keys_by_row.append(keys[0])
    for row, key in zip(rows, keys_by_row):
        row["event_group"] = hashlib.sha256(find(key).encode()).hexdigest()


def read_feedback(training_dir):
    """Join old archives by manifest analysis_id; newer ones also carry hashes."""
    by_id, sources = {}, []
    if training_dir is None:
        return {}, sources
    for manifest_path in sorted((training_dir / "raw").glob("*/metadata.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files") or []
        files = []
        for entry in entries:
            name = entry["stored_name"]
            if Path(name).name != name or "\\" in name or "/" in name:
                raise ValueError("Unsafe retained filename")
            path = manifest_path.parent / name
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Retained file checksum mismatch: {path}")
            files.append(path)
        cfg = next((p for p in files if p.suffix.lower() in {".cfg", ".cff"}), None)
        if cfg is None:
            continue
        key = identity(cfg)
        by_id[manifest["analysis_id"]] = key
        sources.append({"path": cfg, "label": "", "group": key})
    corrections = {}
    feedback_path = training_dir / "labels" / "feedback.jsonl"
    if feedback_path.exists():
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = by_id.get(row.get("analysis_id"))
            if not key:
                continue
            if row.get("record_fingerprint") and row["record_fingerprint"] != key:
                raise ValueError("Feedback identity does not match retained bytes")
            corrections.setdefault(key, []).append(row)
    return corrections, sources


def select_feedback(rows):
    rows = sorted(rows, key=lambda r: r.get("submitted_at_utc", ""))
    if rows and rows[-1].get("include_for_training") is False:
        return {"include_for_training": False}
    trusted = [r for r in rows if r.get("ground_truth_confidence") in TRUSTED]
    return trusted[-1] if trusted else None


def corrected_window(payload, feedback):
    window = build_event_window(payload)
    if not feedback:
        return window
    time = np.asarray(payload["time"], dtype=float) * 1000
    changes = {}
    for prefix, field in (("inception", "corrected_inception_time_ms"), ("clearing", "corrected_clearing_time_ms")):
        if feedback.get(prefix + "_correct") is False:
            value = feedback.get(field)
            if value is None or not np.isfinite(value) or not time[0] <= value <= time[-1]:
                raise ValueError(f"Invalid {prefix} correction")
            idx = int(np.argmin(abs(time - value)))
            changes[prefix + "_idx"] = idx
            changes[prefix + "_time_ms"] = float(time[idx])
    window = replace(window, **changes)
    if window.inception_time_ms is not None and window.clearing_time_ms is not None:
        if window.clearing_time_ms <= window.inception_time_ms:
            raise ValueError("Clearing must follow inception")
        window = replace(window, fault_duration_ms=window.clearing_time_ms - window.inception_time_ms)
    return window


def apply_feedback(row, feedback, folder_label):
    row = dict(row)
    row.update(label=folder_label, label_source="folder" if folder_label else "unlabeled")
    if not feedback:
        return row
    if feedback.get("include_for_training") is False:
        raise ValueError("Operator excluded this record")
    if feedback.get("parsing_correct") is False or feedback.get("channel_mapping_correct") is False:
        raise ValueError("Unresolved parsing/channel mapping correction")
    if feedback.get("protection_interpretation_correct") is False and feedback.get("actual_event_class") not in {
        "TRANSIENT_LINE_FAULT", "PERMANENT_LINE_FAULT", "LINE_FAULT",
        "RECLOSE_REFAULT_SOTF", "RECLOSE_REFAULT", "RECLOSE_FAILED", "RECLOSE_SUCCESSFUL",
    }:
        raise ValueError("Corrected event is outside the line-cause model or unresolved")
    cause = feedback.get("actual_cause") if feedback.get("cause_correct") is False else feedback.get("actual_label")
    if feedback.get("cause_correct") is False and not cause:
        raise ValueError("Missing actual cause")
    if cause:
        if cause not in ALL_CLASSES:
            raise ValueError(f"Cause outside seven-class taxonomy: {cause}")
        row.update(label=cause, label_source="feedback")
    for flag, source, target in (
        ("faulted_phases_correct", "actual_faulted_phases", "faulted_phases"),
        ("zone_correct", "actual_zone", "zone_operated"),
        ("trip_type_correct", "actual_trip_type", "trip_type"),
        ("fault_type_correct", "actual_fault_type", "fault_type"),
        ("event_segmentation_correct", "actual_episode_count", "fault_count"),
    ):
        if feedback.get(flag) is False:
            value = feedback.get(source)
            if value is None or value == "" or value == []:
                raise ValueError(f"Missing correction: {source}")
            if target == "faulted_phases":
                aliases = {"R": "A", "S": "B", "T": "C", "A": "A", "B": "B", "C": "C"}
                try:
                    value = "+".join(sorted({aliases[p.upper()] for p in value}))
                except (KeyError, TypeError):
                    raise ValueError("Invalid corrected phases")
            if target == "fault_count" and (not isinstance(value, int) or value < 1):
                raise ValueError("Invalid episode count")
            row[target] = value
    if feedback.get("fault_type_correct") is False:
        topology = str(row["fault_type"]).upper()
        if topology in {"SLG", "DLG"}:
            row["is_ground_fault"] = True
        elif topology in {"LL", "3PH", "SL"}:
            row["is_ground_fault"] = False
        elif topology.lower() not in {"transient", "permanent", "unknown"}:
            raise ValueError("Corrected fault type is outside the line-cause model")
    if feedback.get("reclose_correct") is False:
        value = str(feedback.get("actual_reclose_outcome", "")).lower()
        outcomes = {"successful": True, "failed": False, "not_attempted": None, "unknown": None}
        if value not in outcomes:
            raise ValueError("Invalid reclose correction")
        row["reclose_successful"] = outcomes[value]
    row.update(ground_truth_confidence=feedback.get("ground_truth_confidence"),
               ground_truth_source=json.dumps(feedback.get("ground_truth_source") or []),
               feedback_submitted_at=feedback.get("submitted_at_utc", ""))
    return row


def extract_source(path, feedback=None):
    record = parse_cff_bytes(path.read_bytes(), path.name) if path.suffix.lower() == ".cff" else parse_comtrade(str(path))
    if record is None:
        raise ValueError("COMTRADE parse failed")
    payload = record_to_payload(record)
    if detect_fault_presence(payload).no_fault:
        raise ValueError("No-fault gate; cannot use a cause label")
    row = extract_ml_features(payload, "21", event_window=corrected_window(payload, feedback))
    row.update(station_name=record.station_name, feature_version=FEATURE_VERSION,
               scaling_ok=row["peak_fault_current_a"] >= 200,
               duration_ok=row["fault_duration_ms"] >= 5)
    return row


def build_dataset(inventory, output, training_dir=None, corpus_root=None):
    reader_sha = pipeline_fingerprint()
    legacy = list(csv.DictReader(inventory.open(encoding="utf-8-sig", newline="")))
    sources = [{"path": Path(r["cfg_path"]), "label": r["label"], "group": event_group(Path(r["cfg_path"]))} for r in legacy]
    corrections, retained = read_feedback(training_dir)
    sources += retained
    if corpus_root:
        from batch_extract import find_labeled_cfgs
        sources += [{"path": p, "label": label, "group": event_group(p)} for p, label in find_labeled_cfgs(corpus_root)]
    audit, buckets, rows = [], {}, []
    for source in sources:
        try:
            key = identity(source["path"])
            buckets.setdefault(key, []).append(source)
        except Exception as exc:
            audit.append({"path": str(source["path"]), "reason": str(exc)})
    for idx, (key, copies) in enumerate(buckets.items(), 1):
        source = copies[0]
        feedback = select_feedback(corrections.get(key, []))
        try:
            labels = {c["label"] for c in copies if c["label"]}
            cause_override = feedback and (feedback.get("actual_label") or (feedback.get("cause_correct") is False and feedback.get("actual_cause")))
            if len(labels) > 1 and not cause_override:
                raise ValueError("Conflicting folder labels for identical record")
            row = apply_feedback(extract_source(source["path"], feedback), feedback, next(iter(labels), ""))
            if row["label"] not in ALL_CLASSES:
                raise ValueError("No verified cause label and no known folder cause")
            row.update(record_fingerprint=key, cfg_path=str(source["path"]),
                       event_group=next((c["group"] for c in copies if c["label"]), key),
                       aliases=json.dumps(sorted({str(c["path"]) for c in copies})))
            # Keep legacy values only for the paired historical baseline, never as new features.
            row["legacy_features_json"] = json.dumps(next((r for r in legacy if Path(r["cfg_path"]) == source["path"]), {}))
            rows.append(row)
        except Exception as exc:
            audit.append({"path": str(source["path"]), "record_fingerprint": key, "reason": str(exc)})
        if idx % 25 == 0:
            print(f"Extracted {idx}/{len(buckets)} unique records; kept {len(rows)}", flush=True)
    if not rows:
        raise ValueError("No usable records; dataset was not overwritten")
    assign_event_groups(rows)
    if pipeline_fingerprint() != reader_sha:
        raise ValueError("Reader changed during extraction; rerun before training")
    for row in rows:
        row["pipeline_fingerprint"] = reader_sha
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in rows for k in r})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_version": FEATURE_VERSION,
              "source_count": len(sources), "unique_records": len(buckets), "rows": len(rows),
              "feedback_labeled_rows": sum(r["label_source"] == "feedback" for r in rows),
              "feedback_corrected_rows": sum(bool(r.get("feedback_submitted_at")) for r in rows), "excluded": audit}
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"}), flush=True)
    return rows, report


def main():
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("data/features/labeled_features.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/features/labeled_features_v2.csv"))
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    args = parser.parse_args()
    build_dataset(args.inventory, args.out, args.training_dir, args.corpus_root)


if __name__ == "__main__":
    main()
