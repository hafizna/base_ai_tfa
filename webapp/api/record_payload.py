"""Shared COMTRADE-to-session adapter for uploads and dataset extraction."""
from core.comtrade_parser import ComtradeRecord
from .json_safety import replace_non_finite_numbers


def record_to_payload(record: ComtradeRecord) -> dict:
    return replace_non_finite_numbers({
        **{name: getattr(record, name) for name in (
            "station_name", "rec_dev_id", "rev_year", "sampling_rates", "trigger_time",
            "start_time_iso", "trigger_time_iso", "trigger_offset_s", "time_code",
            "local_code", "clock_quality", "total_samples", "frequency", "warnings",
        )},
        "time": record.time.tolist(),
        "analog_channels": [
            {**{name: getattr(ch, name) for name in (
                "id", "name", "canonical_name", "unit", "phase", "measurement",
                "ct_primary", "ct_secondary", "pors",
            )}, "samples": ch.samples.tolist()}
            for ch in record.analog_channels
        ],
        "status_channels": [
            {"id": ch.id, "name": ch.name, "samples": ch.samples.tolist()}
            for ch in record.status_channels
        ],
    })
