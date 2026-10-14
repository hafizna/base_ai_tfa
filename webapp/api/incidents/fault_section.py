"""Fault section estimate for an incident: where in the bay the fault was.

Runs ``core.fault_section`` on the incident's attached records:

- One recorder per relay. Records uploaded from one folder came from one relay
  (the folder path is kept with each record's original files); records
  without a folder share a recorder when their COMTRADE headers name the same
  station and device (``time_axis.clock_group``), since those share a clock.
- Each record is parsed again from its original files (``raw_store``); a
  record whose files were not kept falls back to its cached payload.
- Relay event lists uploaded with the records join the recorder of their
  folder ("CBF …/Events/x.evt" with "CBF …/DR/y.cfg"), or stand as an
  event-only recorder (a busbar relay that left no disturbance record).
- The bay template follows the incident's asset type (TRANSFORMER: a
  transformer bay), else what the relays are.

The latest result is stored with the incident.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Optional

import numpy as np

from core.comtrade_parser import AnalogChannel, ComtradeRecord, StatusChannel
from core.fault_section.case import Recorder, folder_label, run_recorders
from core.micom_evt_parser import parse_micom_evt

from ..storage import load_analysis
from . import raw_store
from . import storage as incident_storage
from .models import Incident, IncidentRecord
from .time_axis import clock_group


class FaultSectionError(Exception):
    pass


def _record_folder(record: IncidentRecord) -> str:
    paths = raw_store.uploaded_paths(record.analysis_id)
    first = paths[0] if paths else (record.source_filename or "").split("+")[0]
    parent = PurePosixPath(raw_store.safe_relative_path(first)).parent.as_posix() if first else ""
    return "" if parent in ("", ".") else parent


def _from_payload(payload: dict[str, Any]) -> ComtradeRecord:
    """A cached analysis payload back as a ComtradeRecord (for a record whose
    original files were not kept)."""
    analog = [AnalogChannel(
        id=str(ch.get("id", "")), name=ch.get("name", ""), canonical_name=ch.get("canonical_name") or "",
        unit=ch.get("unit") or "", phase=ch.get("phase"), measurement=ch.get("measurement") or "",
        ct_primary=float(ch.get("ct_primary") or 1.0), ct_secondary=float(ch.get("ct_secondary") or 1.0),
        scale_a=1.0, scale_b=0.0, samples=np.asarray(ch.get("samples") or [], dtype=float), pors=ch.get("pors") or "P",
    ) for ch in payload.get("analog_channels", [])]
    status = [StatusChannel(id=str(ch.get("id", "")), name=ch.get("name", ""),
                            samples=np.asarray(ch.get("samples") or [], dtype=int))
              for ch in payload.get("status_channels", [])]
    return ComtradeRecord(
        station_name=payload.get("station_name") or "", rec_dev_id=payload.get("rec_dev_id") or "",
        rev_year=str(payload.get("rev_year") or ""), sampling_rates=payload.get("sampling_rates") or [],
        trigger_time=float(payload.get("trigger_time") or 0.0), total_samples=int(payload.get("total_samples") or 0),
        frequency=float(payload.get("frequency") or 50.0), analog_channels=analog, status_channels=status,
        time=np.asarray(payload.get("time") or [], dtype=float), cfg_path="", dat_path=None,
        start_time_iso=payload.get("start_time_iso"), trigger_time_iso=payload.get("trigger_time_iso"),
        trigger_offset_s=float(payload.get("trigger_offset_s") or 0.0),
    )


def _parse(record: IncidentRecord) -> Optional[ComtradeRecord]:
    parsed = raw_store.parse_record(record.analysis_id)
    if parsed is not None:
        return parsed
    payload = load_analysis(record.analysis_id)
    return _from_payload(payload) if payload else None


def build_recorders(incident_id: str, records: list[IncidentRecord]) -> tuple[list[Recorder], list[str]]:
    warnings: list[str] = []
    groups: dict[tuple[str, str], list[IncidentRecord]] = {}
    for record in records:
        folder = _record_folder(record)
        key = ("folder", folder) if folder else ("clock", clock_group(record) or record.incident_record_id)
        groups.setdefault(key, []).append(record)

    recorders: list[Recorder] = []
    by_label: dict[str, Recorder] = {}
    for n, ((kind, name), members) in enumerate(groups.items(), 1):
        label = folder_label(name) if kind == "folder" else (name.replace("|", " ").strip(" ?") or f"recorder {n}")
        files = tuple(PurePosixPath((m.source_filename or "").split("+")[0]).name for m in members)
        recorder = Recorder(f"r{n}", label, name if kind == "folder" else label, files, weak=False)
        for member in members:
            parsed = _parse(member)
            if parsed is None or not len(parsed.time) or not parsed.start_time_iso:
                warnings.append(f"{member.source_filename or member.analysis_id}: rekaman tidak bisa dibaca "
                                "atau tanpa waktu absolut — tidak dipakai")
                continue
            recorder.comtrade.append(parsed)
        if recorder.comtrade:
            recorders.append(recorder)
            by_label[label.casefold()] = recorder

    for uploaded, path in raw_store.event_files(incident_id):
        parent = PurePosixPath(uploaded).parent.as_posix()
        label = folder_label(parent if parent != "." else uploaded)
        try:
            events = parse_micom_evt(path)
        except (ValueError, OSError) as exc:
            warnings.append(f"{uploaded}: bukan daftar event MiCOM yang terbaca ({exc})")
            continue
        recorder = by_label.get(label.casefold())
        if recorder is None:
            recorder = Recorder(f"r{len(recorders) + 1}", label, parent, (PurePosixPath(uploaded).name,), weak=False)
            recorders.append(recorder)
            by_label[label.casefold()] = recorder
        else:
            recorder.files = recorder.files + (PurePosixPath(uploaded).name,)
        recorder.add_events(events)
    return recorders, warnings


def _case(incident: Incident) -> dict[str, Any]:
    title = " ".join(t for t in (incident.title, incident.bay_name, incident.asset_name) if t)
    return {"title": title, "station": incident.station_name or "", "asset_type": incident.asset_type}


def run_fault_section(incident: Incident, records: list[IncidentRecord]) -> dict[str, Any]:
    included = [r for r in records if r.inclusion_status == "INCLUDED"]
    if not included:
        raise FaultSectionError("Insiden ini belum punya rekaman yang disertakan.")
    recorders, warnings = build_recorders(incident.incident_id, included)
    try:
        result = run_recorders(_case(incident), recorders)
    except ValueError as exc:
        raise FaultSectionError(str(exc)) from exc
    out = result.to_dict()
    out.update({
        "incident_id": incident.incident_id,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(included),
        # The record set it was computed from: the page asks for a new run when it changes.
        "analysis_ids": sorted(r.analysis_id for r in included),
        "warnings": warnings,
    })
    incident_storage.save_fault_section(incident.incident_id, out)
    return out


def latest_fault_section(incident_id: str) -> Optional[dict[str, Any]]:
    return incident_storage.get_fault_section(incident_id)
