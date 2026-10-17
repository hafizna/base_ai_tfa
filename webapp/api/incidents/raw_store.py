"""Original files of incident records, kept for as long as the incident.

A record's parsed payload lives in the analysis cache (``webapp.api.storage``),
which expires after ``ANALYSIS_SESSION_TTL_HOURS`` and, on a server whose
cache sits in the container, with every redeploy. An incident must not lose
its records that way, so the original COMTRADE files (a few MB at most, where
the parsed payload of the same record can be tens of MB of JSON) are kept
here under ``INCIDENTS_DATA_DIR`` and re-parsed when the cache no longer
holds them:

- ``raw/records/<analysis_id>/``: one record's .cfg + .dat (or .cff), with a
  ``manifest.json`` naming the files and the folder path they were uploaded
  from (the folder names say which relay a record came from).
- ``raw/events/<incident_id>/<folder path>``: relay event lists uploaded with
  the records (MiCOM ``.evt``), kept under their folder path for the same
  reason.

A record uploaded on its own and attached later is adopted from the training
retention archive when that archive kept its files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from core.cff_parser import parse_cff_bytes
from core.comtrade_parser import ComtradeRecord, parse_comtrade

from . import storage as incident_storage

_MANIFEST = "manifest.json"


def _records_dir() -> Path:
    # Read at call time: tests point the incident store elsewhere by reloading it.
    return incident_storage.INCIDENTS_DIR / "raw" / "records"


def _events_dir() -> Path:
    return incident_storage.INCIDENTS_DIR / "raw" / "events"


def safe_relative_path(name: str) -> str:
    """An uploaded file's folder path made safe to store: forward slashes, no
    drive, no leading slash, no ``..``."""
    parts = [p for p in PurePosixPath(str(name or "").replace("\\", "/")).parts
             if p not in ("", ".", "..", "/") and not p.endswith(":")]
    return "/".join(parts) or "upload.bin"


def _record_dir(analysis_id: str) -> Path:
    return _records_dir() / safe_relative_path(analysis_id).replace("/", "_")


def save_record_files(analysis_id: str, files: Iterable[tuple[str, bytes]]) -> Path:
    """Keep a record's original files; ``files`` are (uploaded path, bytes).

    Stored under short names ("record.cfg", "record.dat"): a DFR names its
    files after the station, bay and maker ("240226,010033697,0,GI JEPARA,
    TANJUNG JATI 1-2,Qualitrol LLC.cfg"), which can run past a path-length
    limit. The manifest keeps the uploaded path."""
    target = _record_dir(analysis_id)
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    used: set[str] = set()
    for uploaded, data in files:
        rel = safe_relative_path(uploaded)
        suffix = PurePosixPath(rel).suffix.lower()
        stored, n = f"record{suffix}", 1
        while stored in used:
            n += 1
            stored = f"record_{n}{suffix}"
        used.add(stored)
        (target / stored).write_bytes(data)
        entries.append({"stored_name": stored, "uploaded_path": rel})
    (target / _MANIFEST).write_text(json.dumps({"analysis_id": analysis_id, "files": entries},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def has_record_files(analysis_id: str) -> bool:
    return (_record_dir(analysis_id) / _MANIFEST).exists()


def uploaded_paths(analysis_id: str) -> list[str]:
    manifest = _record_dir(analysis_id) / _MANIFEST
    if not manifest.exists():
        return []
    return [f["uploaded_path"] for f in json.loads(manifest.read_text(encoding="utf-8")).get("files", [])]


def delete_record_files(analysis_id: str) -> None:
    shutil.rmtree(_record_dir(analysis_id), ignore_errors=True)


def adopt_retained_upload(analysis_id: str) -> bool:
    """Copy a single upload's files from the training retention archive."""
    if has_record_files(analysis_id):
        return True
    try:
        from ..training_retention import RAW_DIR as RETAINED_DIR
    except Exception:  # pragma: no cover
        return False
    if not RETAINED_DIR.exists():
        return False
    for manifest_path in RETAINED_DIR.glob(f"*_{analysis_id[:12]}/metadata.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("analysis_id") != analysis_id:
            continue
        files = []
        for entry in manifest.get("files", []):
            source = manifest_path.parent / entry["stored_name"]
            if source.exists():
                files.append((entry.get("filename") or entry["stored_name"], source.read_bytes()))
        if files:
            save_record_files(analysis_id, files)
            return True
    return False


def parse_record(analysis_id: str) -> Optional[ComtradeRecord]:
    """The record parsed again from its original files, or None."""
    folder = _record_dir(analysis_id)
    if not (folder / _MANIFEST).exists():
        return None
    names = [f["stored_name"] for f in json.loads((folder / _MANIFEST).read_text(encoding="utf-8")).get("files", [])]
    cff = next((n for n in names if n.lower().endswith(".cff")), None)
    if cff:
        return parse_cff_bytes((folder / cff).read_bytes(), cff)
    cfg = next((n for n in names if n.lower().endswith(".cfg")), None)
    dat = next((n for n in names if n.lower().endswith(".dat")), None)
    if not cfg or not dat:
        return None
    return parse_comtrade(str(folder / cfg), str(folder / dat))


def restore_payload(analysis_id: str) -> Optional[dict]:
    """Rebuild an expired analysis payload from the record's files and put it
    back in the cache under the same id, so every page that opens the record
    by its analysis id keeps working."""
    record = parse_record(analysis_id)
    if record is None or record.total_samples <= 0:
        return None
    from ..record_payload import record_to_payload
    from ..storage import update_analysis

    payload = record_to_payload(record)
    update_analysis(analysis_id, payload)
    return payload


def save_event_file(incident_id: str, uploaded: str, data: bytes) -> str:
    rel = safe_relative_path(uploaded)
    target = _events_dir() / safe_relative_path(incident_id) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return rel


def event_files(incident_id: str) -> list[tuple[str, Path]]:
    """``(uploaded folder path, stored file)`` of an incident's event lists."""
    root = _events_dir() / safe_relative_path(incident_id)
    if not root.exists():
        return []
    return sorted((p.relative_to(root).as_posix(), p) for p in root.rglob("*") if p.is_file())


def delete_event_files(incident_id: str) -> None:
    shutil.rmtree(_events_dir() / safe_relative_path(incident_id), ignore_errors=True)
