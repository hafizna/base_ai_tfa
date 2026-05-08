"""Parser for Qualitrol/Cashel TWS FL .cdb export archives.

The observed .cdb export is a ZIP container with XML table exports and nested
.cdf ZIP archives. Each .cdf contains an XML descriptor plus a float32 .dat
waveform file. This parser intentionally keeps inferred fields explicit in
warnings so new samples can refine the format without breaking callers.
"""

from __future__ import annotations

import io
import math
import struct
import zipfile
from pathlib import PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET


SPEED_OF_LIGHT_KM_S = 299_792.458
PHASES = ("A", "B", "C")


class TwsCdbParseError(ValueError):
    """Raised when a TWS FL export cannot be parsed."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(row: ET.Element, name: str, default: str = "") -> str:
    for child in row:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_xml_rows(zip_file: zipfile.ZipFile, member_name: str) -> list[dict[str, str]]:
    try:
        raw = zip_file.read(member_name)
    except KeyError:
        return []

    root = ET.fromstring(raw)
    rows: list[dict[str, str]] = []
    for child in root:
        name = _local_name(child.tag)
        if name == "schema":
            continue
        if not list(child):
            continue
        row: dict[str, str] = {"_table": name}
        for field in child:
            row[_local_name(field.tag)] = (field.text or "").strip()
        rows.append(row)
    return rows


def _read_cdf(cdf_bytes: bytes, archive_name: str, warnings: list[str]) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(cdf_bytes)) as cdf_zip:
            xml_name = next((n for n in cdf_zip.namelist() if n.lower().endswith(".xml")), "")
            dat_name = next((n for n in cdf_zip.namelist() if n.lower().endswith(".dat")), "")
            if not xml_name or not dat_name:
                raise TwsCdbParseError(f"{archive_name} is missing XML descriptor or DAT payload.")

            descriptor_root = ET.fromstring(cdf_zip.read(xml_name))
            dat_bytes = cdf_zip.read(dat_name)
    except zipfile.BadZipFile as exc:
        raise TwsCdbParseError(f"{archive_name} is not a readable CDF ZIP archive.") from exc

    device_descriptor = descriptor_root.find("./DeviceDescriptors/DeviceDescriptor")
    header = descriptor_root.find("./FLRecordDataHeader")
    data_descriptor = descriptor_root.find("./FLRecordDataHeader/DataDescriptor")
    channels_info = descriptor_root.find("./FLRecordDataHeader/FLChannelsInformation")
    if header is None:
        raise TwsCdbParseError(f"{archive_name} is missing FLRecordDataHeader.")

    total_samples = _as_int(_text(header, "TotalNumberOfSamples"), 0)
    sample_rate_hz = _as_float(_text(header, "SampleRateInHz"), 0.0)
    channel_count = _as_int(_text(channels_info, "NoOfChannels") if channels_info is not None else "", 3)
    channel_count = max(1, min(channel_count, len(PHASES)))

    float_count = len(dat_bytes) // 4
    expected = total_samples * channel_count
    if total_samples <= 0 or sample_rate_hz <= 0:
        warnings.append(f"{archive_name}: missing usable sample count or sample rate.")
    if float_count < expected:
        warnings.append(
            f"{archive_name}: DAT contains {float_count} float32 values, expected at least {expected}."
        )

    usable_float_count = min(float_count, expected)
    values = struct.unpack("<" + "f" * usable_float_count, dat_bytes[: usable_float_count * 4])

    channels: list[dict[str, Any]] = []
    for idx in range(channel_count):
        start = idx * total_samples
        end = min(start + total_samples, len(values))
        samples = [float(v) for v in values[start:end]]
        phase = PHASES[idx]
        channels.append(
            {
                "phase": phase,
                "name": f"TW Phase {phase}",
                "samples": samples,
                "min": min(samples) if samples else 0.0,
                "max": max(samples) if samples else 0.0,
            }
        )

    return {
        "archive_name": archive_name,
        "record_file_name": PurePosixPath(archive_name.replace("\\", "/")).name,
        "station_name": _text(device_descriptor, "StationName") if device_descriptor is not None else "",
        "device_name": _text(device_descriptor, "DeviceName") if device_descriptor is not None else "",
        "feeder_name": _text(device_descriptor, "FeederName") if device_descriptor is not None else "",
        "device_id": _as_int(_text(device_descriptor, "DeviceID") if device_descriptor is not None else ""),
        "device_type": _text(device_descriptor, "DeviceType") if device_descriptor is not None else "",
        "time_locked": _text(device_descriptor, "TimeLocked") if device_descriptor is not None else "",
        "record_type": _text(descriptor_root, "RecordType", "FL"),
        "record_number": _as_int(_text(header, "RecordNumber")),
        "line_module": _text(header, "LineModule"),
        "trigger_time": _text(header, "TriggerTime"),
        "trigger_time_us": _as_float(_text(header, "TriggerTimeUS")),
        "gps_time_tag": _text(header, "GPSTag"),
        "corrected_gps": _text(header, "CorrectedGPS"),
        "sample_rate_hz": sample_rate_hz,
        "total_samples": total_samples,
        "total_frames": _as_int(_text(header, "TotalNumberOfFrames")),
        "decimation": _as_int(_text(header, "decimation")),
        "post_pre_trigger_factor": _as_int(_text(header, "PostPreTrgFactor")),
        "trigger_phase": _text(data_descriptor, "TriggerPhase") if data_descriptor is not None else "",
        "software_trigger_phase": _text(data_descriptor, "SoftwareTriggerPhase") if data_descriptor is not None else "",
        "software_trigger_point": _as_int(_text(data_descriptor, "SoftwareTriggerPoint") if data_descriptor is not None else ""),
        "trigger_delay": _as_int(_text(data_descriptor, "TriggerDelay") if data_descriptor is not None else ""),
        "signalling_value": _text(data_descriptor, "SignallingValue") if data_descriptor is not None else "",
        "gain": _as_float(_text(header, "Gain")),
        "channels": channels,
    }


def _event_table(outer_zip: zipfile.ZipFile) -> dict[int, dict[str, str]]:
    events: dict[int, dict[str, str]] = {}
    for name in outer_zip.namelist():
        normalized = name.replace("\\", "/")
        if normalized.startswith("FL/") and normalized.endswith(".XML"):
            for row in _read_xml_rows(outer_zip, name):
                if "IndexId" in row:
                    events[_as_int(row.get("IndexId"))] = row
    return events


def parse_tws_cdb_bytes(data: bytes, source_filename: str = "record.cdb") -> dict[str, Any]:
    """Parse a TWS FL .cdb export into JSON-serializable data."""
    warnings: list[str] = []
    try:
        outer_zip = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise TwsCdbParseError("TWS FL .cdb export is not a ZIP-based Cashel export.") from exc

    with outer_zip:
        devices = {
            _as_int(row.get("DeviceId")): row
            for row in _read_xml_rows(outer_zip, "DEVICE.XML")
            if row.get("_table") == "device"
        }
        feeders = {
            _as_int(row.get("FeederId")): row
            for row in _read_xml_rows(outer_zip, "FEEDER.XML")
            if row.get("_table") == "Feeders"
        }
        circuit_rows = _read_xml_rows(outer_zip, "CIRCUIT2.XML")
        circuits = {
            _as_int(row.get("CircuitId")): row
            for row in circuit_rows
            if row.get("_table") == "FLCircuits"
        }
        segments = [
            row
            for row in circuit_rows
            if "SegmentID" in row and "CircuitID" in row and row.get("_table") != "FLCircuitEndpoints"
        ]
        results = _read_xml_rows(outer_zip, "FLResults\\1\\0.XML") or _read_xml_rows(
            outer_zip,
            "FLResults/1/0.XML",
        )
        events = _event_table(outer_zip)

        cdf_by_name: dict[str, dict[str, Any]] = {}
        for member in outer_zip.namelist():
            if member.lower().endswith(".cdf"):
                cdf = _read_cdf(outer_zip.read(member), member, warnings)
                cdf_by_name[cdf["record_file_name"]] = cdf

    parsed_results: list[dict[str, Any]] = []
    for row in results:
        circuit_id = _as_int(row.get("CircuitID"))
        faulted_segment_id = _as_int(row.get("FaultedSegment"))
        circuit = circuits.get(circuit_id, {})
        segment = next((s for s in segments if _as_int(s.get("SegmentID")) == faulted_segment_id), None)
        if segment is None:
            segment = next((s for s in segments if _as_int(s.get("CircuitID")) == circuit_id), {})

        endpoints: list[dict[str, Any]] = []
        for role, index_key, distance_key in (
            ("X", "IndexIdX", "DTFX"),
            ("Y", "IndexIdY", "DTFY"),
            ("Z", "IndexIdZ", "DTFZ"),
        ):
            index_id = _as_int(row.get(index_key))
            if index_id <= 0:
                continue
            event = events.get(index_id, {})
            cdf = cdf_by_name.get(event.get("RecordFileName", ""))
            device = devices.get(_as_int(event.get("DeviceId")), {})
            feeder = feeders.get(_as_int(event.get("FeederId")), {})
            if cdf is None:
                warnings.append(f"Missing waveform CDF for event {index_id}.")
                continue

            endpoint = {
                **cdf,
                "role": role,
                "index_id": index_id,
                "fault_distance_km": _as_float(row.get(distance_key)),
                "event_time_us": _as_float(event.get("EventTimeUS")),
                "event_time_local": _as_float(event.get("EventTimeLocal")),
                "gps_locked": _as_int(event.get("GPSLocked")) == 1,
                "trigger_type": _as_int(event.get("TriggerType")),
                "feeder_id": _as_int(event.get("FeederId")),
                "feeder_display_name": feeder.get("FeederName", cdf.get("feeder_name", "")),
                "device_display_name": device.get("DeviceName", cdf.get("device_name", "")),
                "station_display_name": device.get("StationName", cdf.get("station_name", "")),
            }
            endpoints.append(endpoint)

        velocity_factor = _as_float(circuit.get("VelocityFactor"), 100.0)
        sample_rate = max((e.get("sample_rate_hz") or 0.0 for e in endpoints), default=0.0)
        sample_distance_km = SPEED_OF_LIGHT_KM_S * (velocity_factor / 100.0) / sample_rate if sample_rate > 0 else 0.0
        line_length_km = _as_float(segment.get("Length"))
        parsed_results.append(
            {
                "result_id": _as_int(row.get("ResultID")),
                "result_time_us": _as_float(row.get("ResultTimeStampUS")),
                "result_time_local": _as_float(row.get("ResultTimeStampLocal")),
                "result_type": _as_int(row.get("ResultType")),
                "circuit_id": circuit_id,
                "circuit_name": circuit.get("CircuitName", ""),
                "segment_id": faulted_segment_id,
                "segment_name": segment.get("Name", ""),
                "line_length_km": line_length_km,
                "velocity_factor": velocity_factor,
                "sample_distance_km": sample_distance_km,
                "distance_from_segment_end_a": _as_float(row.get("DistanceFromSegmentEndA")),
                "is_component_fault": str(row.get("IsComponentFault", "")).lower() == "true",
                "endpoints": endpoints,
            }
        )

    if not parsed_results:
        raise TwsCdbParseError("No FLResults rows were found in the TWS FL export.")

    first = parsed_results[0]
    first_endpoint = first["endpoints"][0] if first["endpoints"] else {}
    return {
        "source_type": "tws_cdb",
        "source_file": source_filename,
        "station_name": first.get("circuit_name") or first_endpoint.get("station_name", ""),
        "rec_dev_id": "TWS FL",
        "total_samples": max((e.get("total_samples") or 0 for r in parsed_results for e in r["endpoints"]), default=0),
        "results": parsed_results,
        "warnings": warnings,
    }
