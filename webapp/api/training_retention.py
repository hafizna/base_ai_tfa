"""Persistent training-data retention for uploaded disturbance records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "training-data")).resolve()
RETENTION_ENABLED = os.getenv("TRAINING_RETENTION_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADMIN_TOKEN = os.getenv("TRAINING_ADMIN_TOKEN", "").strip()

RAW_DIR = TRAINING_DATA_DIR / "raw"
LABELS_DIR = TRAINING_DATA_DIR / "labels"
EXPORTS_DIR = TRAINING_DATA_DIR / "exports"

# NOTE: the CSV header is written once, from whichever process creates
# feedback.csv first (see `write_header` below) - if this list grows, older
# CSV files on disk keep their original header and simply leave new columns
# blank via DictWriter's restval. feedback.jsonl is schema-flexible and
# always carries the full granular payload; treat it as the source of truth
# for anything not present in an older CSV header.
FEEDBACK_COLUMNS = [
    "submitted_at_utc",
    "analysis_id",
    "relay_type",
    # Legacy (Stage -1) fields — kept for backward compatibility.
    "ai_correct",
    "actual_label",
    "fault_type",
    "include_for_training",
    "operator",
    "notes",
    # Stage 0: per-layer ground-truth correction fields.
    "parsing_correct",
    "channel_mapping_correct",
    "inception_correct",
    "corrected_inception_time_ms",
    "clearing_correct",
    "corrected_clearing_time_ms",
    "faulted_phases_correct",
    "actual_faulted_phases",
    "fault_type_correct",
    "actual_fault_type",
    "zone_correct",
    "actual_zone",
    "trip_type_correct",
    "actual_trip_type",
    "reclose_correct",
    "actual_reclose_outcome",
    "event_segmentation_correct",
    "actual_episode_count",
    "protection_interpretation_correct",
    "actual_event_class",
    "cause_correct",
    "actual_cause",
    "ground_truth_source",
    "ground_truth_confidence",
]


@dataclass(frozen=True)
class RetainedUploadFile:
    field_name: str
    filename: str
    content_type: str | None
    data: bytes


def is_retention_enabled() -> bool:
    return RETENTION_ENABLED


def admin_token_configured() -> bool:
    return bool(ADMIN_TOKEN)


def verify_admin_token(token: str | None) -> bool:
    return bool(ADMIN_TOKEN and token and token == ADMIN_TOKEN)


def ensure_dirs() -> None:
    for path in (RAW_DIR, LABELS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    cleaned = Path(name or "upload.bin").name.replace("\\", "_").replace("/", "_")
    return cleaned or "upload.bin"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _raw_record_dir(analysis_id: str, created_at: datetime) -> Path:
    date_part = created_at.strftime("%Y%m%d_%H%M%S")
    return RAW_DIR / f"{date_part}_{analysis_id[:12]}"


def retain_upload(
    *,
    analysis_id: str,
    source_type: str,
    files: list[RetainedUploadFile],
    metadata: dict[str, Any],
) -> Path | None:
    """Persist raw upload files and a metadata manifest.

    Retention is best-effort. Upload analysis must continue even if archive
    writing fails, so callers should treat ``None`` as a non-fatal miss.
    """
    if not RETENTION_ENABLED:
        return None

    ensure_dirs()
    created_at = datetime.now(timezone.utc)
    record_dir = _raw_record_dir(analysis_id, created_at)
    record_dir.mkdir(parents=True, exist_ok=False)

    manifest_files: list[dict[str, Any]] = []
    for idx, file in enumerate(files, start=1):
        filename = _safe_filename(file.filename)
        target = record_dir / filename
        if target.exists():
            stem = target.stem or f"file_{idx}"
            suffix = target.suffix
            target = record_dir / f"{stem}_{idx}{suffix}"
        target.write_bytes(file.data)
        manifest_files.append(
            {
                "field_name": file.field_name,
                "filename": filename,
                "stored_name": target.name,
                "content_type": file.content_type or "",
                "size_bytes": len(file.data),
                "sha256": _sha256_bytes(file.data),
            }
        )

    manifest = {
        "analysis_id": analysis_id,
        "source_type": source_type,
        "created_at_utc": created_at.isoformat(),
        "files": manifest_files,
        "metadata": metadata,
    }
    (record_dir / "metadata.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return record_dir


def _bool_cell(value: Any) -> str:
    return "" if value is None else str(bool(value))


def append_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs()
    submitted_at = datetime.now(timezone.utc).isoformat()
    row = {
        "submitted_at_utc": submitted_at,
        "analysis_id": str(feedback.get("analysis_id") or ""),
        "relay_type": str(feedback.get("relay_type") or ""),
        # Legacy fields.
        "ai_correct": _bool_cell(feedback.get("ai_correct")),
        "actual_label": str(feedback.get("actual_label") or ""),
        "fault_type": str(feedback.get("fault_type") or ""),
        "include_for_training": str(bool(feedback.get("include_for_training", True))),
        "operator": str(feedback.get("operator") or ""),
        "notes": str(feedback.get("notes") or ""),
        # Stage 0 per-layer ground-truth correction fields.
        "parsing_correct": _bool_cell(feedback.get("parsing_correct")),
        "channel_mapping_correct": _bool_cell(feedback.get("channel_mapping_correct")),
        "inception_correct": _bool_cell(feedback.get("inception_correct")),
        "corrected_inception_time_ms": "" if feedback.get("corrected_inception_time_ms") is None else str(feedback.get("corrected_inception_time_ms")),
        "clearing_correct": _bool_cell(feedback.get("clearing_correct")),
        "corrected_clearing_time_ms": "" if feedback.get("corrected_clearing_time_ms") is None else str(feedback.get("corrected_clearing_time_ms")),
        "faulted_phases_correct": _bool_cell(feedback.get("faulted_phases_correct")),
        "actual_faulted_phases": "+".join(feedback.get("actual_faulted_phases") or []),
        "fault_type_correct": _bool_cell(feedback.get("fault_type_correct")),
        "actual_fault_type": str(feedback.get("actual_fault_type") or ""),
        "zone_correct": _bool_cell(feedback.get("zone_correct")),
        "actual_zone": str(feedback.get("actual_zone") or ""),
        "trip_type_correct": _bool_cell(feedback.get("trip_type_correct")),
        "actual_trip_type": str(feedback.get("actual_trip_type") or ""),
        "reclose_correct": _bool_cell(feedback.get("reclose_correct")),
        "actual_reclose_outcome": str(feedback.get("actual_reclose_outcome") or ""),
        "event_segmentation_correct": _bool_cell(feedback.get("event_segmentation_correct")),
        "actual_episode_count": "" if feedback.get("actual_episode_count") is None else str(feedback.get("actual_episode_count")),
        "protection_interpretation_correct": _bool_cell(feedback.get("protection_interpretation_correct")),
        "actual_event_class": str(feedback.get("actual_event_class") or ""),
        "cause_correct": _bool_cell(feedback.get("cause_correct")),
        "actual_cause": str(feedback.get("actual_cause") or ""),
        "ground_truth_source": "+".join(feedback.get("ground_truth_source") or []),
        "ground_truth_confidence": str(feedback.get("ground_truth_confidence") or "UNKNOWN"),
    }

    csv_path = LABELS_DIR / "feedback.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    jsonl_path = LABELS_DIR / "feedback.jsonl"
    jsonl_payload = {
        **feedback,
        "submitted_at_utc": submitted_at,
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonl_payload, ensure_ascii=False, default=_json_default) + "\n")

    return row


def get_training_status() -> dict[str, Any]:
    ensure_dirs()
    raw_records = [p for p in RAW_DIR.iterdir() if p.is_dir()] if RAW_DIR.exists() else []
    feedback_csv = LABELS_DIR / "feedback.csv"
    feedback_count = 0
    if feedback_csv.exists():
        with feedback_csv.open("r", newline="", encoding="utf-8") as handle:
            feedback_count = max(0, sum(1 for _ in handle) - 1)

    total_bytes = 0
    if TRAINING_DATA_DIR.exists():
        for path in TRAINING_DATA_DIR.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size

    return {
        "enabled": RETENTION_ENABLED,
        "admin_token_configured": bool(ADMIN_TOKEN),
        "data_dir": str(TRAINING_DATA_DIR),
        "raw_record_count": len(raw_records),
        "feedback_count": feedback_count,
        "total_bytes": total_bytes,
    }


def build_training_archive() -> Path:
    ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fd, tmp_name = tempfile.mkstemp(prefix=f"training-data-{timestamp}-", suffix=".zip")
    os.close(fd)
    zip_path = Path(tmp_name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if TRAINING_DATA_DIR.exists():
            for path in TRAINING_DATA_DIR.rglob("*"):
                if path.is_file() and path != zip_path:
                    archive.write(path, path.relative_to(TRAINING_DATA_DIR))

    return zip_path


def clear_training_archive() -> dict[str, Any]:
    ensure_dirs()
    removed_raw = 0
    if RAW_DIR.exists():
        for path in RAW_DIR.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
                removed_raw += 1
            elif path.is_file():
                path.unlink()

    removed_label_files = 0
    if LABELS_DIR.exists():
        for path in LABELS_DIR.iterdir():
            if path.is_file():
                path.unlink()
                removed_label_files += 1

    if EXPORTS_DIR.exists():
        for path in EXPORTS_DIR.iterdir():
            if path.is_file():
                path.unlink()

    return {
        "removed_raw_records": removed_raw,
        "removed_label_files": removed_label_files,
    }
