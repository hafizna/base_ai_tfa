"""Batch multi-COMTRADE upload — Stage 2.

Pairs uploaded files into COMTRADE records (.cfg+.dat by case-insensitive
stem, or one .cff each), parses them sequentially (bounded — one record's
waveform in memory at a time, matching the single-record upload path), saves
each as a Stage 0 analysis session, and attaches it to an incident.

Default mode is atomic: if any file/pairing error occurs, nothing is saved
and nothing is attached. ``partial_success=True`` processes every valid pair
and reports per-file errors for the rest, without rolling back the valid
ones.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.cff_parser import CffParseError, parse_cff_bytes
from core.comtrade_parser import ComtradeRecord, parse_comtrade
from ..record_analysis import build_record_analysis
from ..routers.upload import _record_to_out  # reuse the exact Stage 0 payload shape
from ..storage import delete_analysis, save_analysis
from . import service as incident_service
from .service import IncidentServiceError


@dataclass
class UploadedFile:
    filename: str
    content_type: Optional[str]
    data: bytes


@dataclass
class PairingError:
    files: list[str]
    reason: str


@dataclass
class RecordPair:
    kind: str  # "cff" | "cfg_dat"
    files: list[UploadedFile]
    stem: str


@dataclass
class BatchRecordResult:
    analysis_id: Optional[str]
    incident_record_id: Optional[str]
    source_files: list[str]
    status: str  # "created" | "error"
    error: Optional[str] = None


@dataclass
class BatchUploadResult:
    incident_id: str
    records_created: list[BatchRecordResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    reconstruction_status: str = "not_run"


def _stem(filename: str) -> str:
    return Path(filename).stem.casefold()


def _suffix(filename: str) -> str:
    return Path(filename).suffix.casefold()


def pair_files(files: list[UploadedFile]) -> tuple[list[RecordPair], list[PairingError]]:
    """Pair uploaded files into COMTRADE records.

    Rules (spec section 1):
      1. .cfg/.dat paired by case-insensitive filename stem.
      2. Each .cff is its own record.
      3. Orphan .cfg -> error.
      4. Orphan .dat -> error.
      5. Duplicate same-stem pair in one request -> error.
      6. Every problem file is reported individually, not one generic failure.
    """
    cff_files = [f for f in files if _suffix(f.filename) == ".cff"]
    cfg_files = [f for f in files if _suffix(f.filename) == ".cfg"]
    dat_files = [f for f in files if _suffix(f.filename) == ".dat"]
    unsupported = [f for f in files if _suffix(f.filename) not in (".cff", ".cfg", ".dat")]

    pairs: list[RecordPair] = []
    errors: list[PairingError] = []

    for f in unsupported:
        errors.append(PairingError(files=[f.filename], reason=f"Unsupported file type '{_suffix(f.filename) or '(none)'}'."))

    for f in cff_files:
        pairs.append(RecordPair(kind="cff", files=[f], stem=_stem(f.filename)))

    cfg_by_stem: dict[str, list[UploadedFile]] = {}
    for f in cfg_files:
        cfg_by_stem.setdefault(_stem(f.filename), []).append(f)
    dat_by_stem: dict[str, list[UploadedFile]] = {}
    for f in dat_files:
        dat_by_stem.setdefault(_stem(f.filename), []).append(f)

    all_stems = sorted(set(cfg_by_stem) | set(dat_by_stem))
    for stem in all_stems:
        cfgs = cfg_by_stem.get(stem, [])
        dats = dat_by_stem.get(stem, [])

        if len(cfgs) > 1 or len(dats) > 1:
            errors.append(PairingError(
                files=[f.filename for f in cfgs + dats],
                reason=f"Duplicate files for stem '{stem}' — expected exactly one .cfg and one .dat.",
            ))
            continue
        if cfgs and not dats:
            errors.append(PairingError(files=[cfgs[0].filename], reason="Orphan .cfg file with no matching .dat file."))
            continue
        if dats and not cfgs:
            errors.append(PairingError(files=[dats[0].filename], reason="Orphan .dat file with no matching .cfg file."))
            continue

        pairs.append(RecordPair(kind="cfg_dat", files=[cfgs[0], dats[0]], stem=stem))

    return pairs, errors


def _parse_pair(pair: RecordPair) -> ComtradeRecord:
    if pair.kind == "cff":
        cff_file = pair.files[0]
        try:
            return parse_cff_bytes(cff_file.data, cff_file.filename)
        except CffParseError as exc:
            raise ValueError(f"Could not parse ABB CFF file '{cff_file.filename}'. {exc}") from exc

    cfg_file = next(f for f in pair.files if _suffix(f.filename) == ".cfg")
    dat_file = next(f for f in pair.files if _suffix(f.filename) == ".dat")
    with tempfile.TemporaryDirectory(prefix="dfr_batch_upload_") as tmp_dir:
        tmp = Path(tmp_dir)
        cfg_path = tmp / cfg_file.filename
        dat_path = tmp / dat_file.filename
        cfg_path.write_bytes(cfg_file.data)
        dat_path.write_bytes(dat_file.data)
        record = parse_comtrade(str(cfg_path), str(dat_path))
        if record is None:
            raise ValueError(f"Could not parse COMTRADE pair '{cfg_file.filename}' / '{dat_file.filename}'.")
        return record


def run_batch_upload(
    incident_id: str,
    files: list[UploadedFile],
    *,
    partial_success: bool = False,
    attachment_role: str = "UNKNOWN",
    bay_name: Optional[str] = None,
    protection_type: Optional[str] = None,
) -> BatchUploadResult:
    """Pair, parse, save, and attach every valid record in ``files`` to
    ``incident_id``. Sequential processing keeps at most one record's
    waveform in memory at a time (bounded memory, per spec section 1/16).

    Atomic by default: if pairing produces any error, or parsing/attachment
    of any pair fails, no analysis is saved and nothing is attached — the
    incident is left exactly as it was. With ``partial_success=True``,
    already-successful records are kept and only the failing files are
    reported as errors.
    """
    incident_service.get_incident(incident_id)  # 404 if missing

    pairs, pairing_errors = pair_files(files)
    result = BatchUploadResult(incident_id=incident_id)

    if pairing_errors and not partial_success:
        result.errors = [{"files": e.files, "reason": e.reason} for e in pairing_errors]
        result.reconstruction_status = "aborted_atomic"
        return result

    result.errors = [{"files": e.files, "reason": e.reason} for e in pairing_errors]

    saved_analysis_ids: list[str] = []
    attached: list[BatchRecordResult] = []
    failures: list[BatchRecordResult] = []

    for pair in pairs:
        filenames = [f.filename for f in pair.files]
        try:
            record = _parse_pair(pair)
            if record is None or record.total_samples <= 0:
                raise ValueError("Parsed record has no samples.")

            payload = _record_to_out(record)
            analysis_id = save_analysis(payload)
            saved_analysis_ids.append(analysis_id)

            incident_record = incident_service.attach_record(
                incident_id,
                analysis_id=analysis_id,
                attachment_role=attachment_role,
                bay_name=bay_name,
                protection_type=protection_type,
                source_filename=filenames[0] if pair.kind == "cff" else f"{filenames[0]}+{filenames[1]}",
            )
            attached.append(BatchRecordResult(
                analysis_id=analysis_id,
                incident_record_id=incident_record.incident_record_id,
                source_files=filenames,
                status="created",
            ))
        except (ValueError, IncidentServiceError) as exc:
            failures.append(BatchRecordResult(
                analysis_id=None,
                incident_record_id=None,
                source_files=filenames,
                status="error",
                error=str(exc),
            ))
        except Exception as exc:  # pragma: no cover - defensive; parser/attach must not crash the batch
            failures.append(BatchRecordResult(
                analysis_id=None,
                incident_record_id=None,
                source_files=filenames,
                status="error",
                error=f"Unexpected error: {exc}",
            ))

    if failures and not partial_success:
        # Roll back: detach anything we attached and delete saved analyses
        # so the incident and analysis-session store are left unchanged.
        for rec in attached:
            try:
                incident_service.detach_record(incident_id, rec.incident_record_id)
            except IncidentServiceError:
                pass
        for analysis_id in saved_analysis_ids:
            delete_analysis(analysis_id)

        result.records_created = []
        result.errors = result.errors + [{"files": f.source_files, "reason": f.error} for f in failures]
        result.reconstruction_status = "aborted_atomic"
        return result

    result.records_created = attached
    result.errors = result.errors + [{"files": f.source_files, "reason": f.error} for f in failures]
    result.reconstruction_status = "completed" if attached and not failures else ("partial" if attached else "failed")
    return result
