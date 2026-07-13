"""Canonical single-record analysis result — Stage 0.

Builds one :class:`RecordAnalysis` per COMTRADE payload, combining:
  - the canonical :class:`~core.event_analysis.EventWindow` (timing),
  - the shared no-fault gate (:mod:`webapp.api.fault_detection`),
  - fault classification facts (phases/type/zone/trip/reclose), reusing the
    same feature extraction already used by ``relay_21``,
  - AI cause hypotheses (LightGBM via :mod:`webapp.api.ml_predict`),
  - data-quality / missing-evidence flags,
  - provenance (timing source + confidence, parser warnings, model version).

This module is the single place that assembles "the story" of one record.
Individual endpoints (SOE, locus events, electrical params, feature
extraction, report) should read timing from ``RecordAnalysis.event_window``
rather than recomputing it — see ``webapp/api/routers/relay_21.py``.

Explicitly NOT in scope here (Stage 1-2): multi-COMTRADE stitching, incident
grouping, correlation across bays/relays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.event_analysis import EventWindow, build_event_window
from .fault_detection import detect_fault_presence


@dataclass
class RecordAnalysis:
    """Canonical single-COMTRADE analysis result."""

    record_id: str
    source_metadata: dict[str, Any]
    data_quality: dict[str, Any]
    event_window: Optional[EventWindow]
    fault_episodes: list[dict[str, Any]] = field(default_factory=list)
    protection_operations: list[dict[str, Any]] = field(default_factory=list)
    electrical_measurements: dict[str, Any] = field(default_factory=dict)
    cause_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    # Convenience view split per the Stage 0 spec: observed facts vs.
    # interpretation vs. hypothesis vs. missing evidence must never be mixed.
    observed_facts: dict[str, Any] = field(default_factory=dict)
    protection_interpretation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_metadata": self.source_metadata,
            "data_quality": self.data_quality,
            "event_window": self.event_window.to_dict() if self.event_window else None,
            "fault_episodes": self.fault_episodes,
            "protection_operations": self.protection_operations,
            "electrical_measurements": self.electrical_measurements,
            "observed_facts": self.observed_facts,
            "protection_interpretation": self.protection_interpretation,
            "cause_hypotheses": self.cause_hypotheses,
            "missing_evidence": self.missing_evidence,
            "provenance": self.provenance,
        }


def _source_metadata(payload: dict) -> dict[str, Any]:
    return {
        "station_name": payload.get("station_name", ""),
        "rec_dev_id": payload.get("rec_dev_id", ""),
        "rev_year": payload.get("rev_year", ""),
        "frequency": payload.get("frequency"),
        "total_samples": payload.get("total_samples"),
        "trigger_time_s": payload.get("trigger_time"),
        "start_time_iso": payload.get("start_time_iso"),
        "trigger_time_iso": payload.get("trigger_time_iso"),
        "trigger_offset_s": payload.get("trigger_offset_s", payload.get("trigger_time")),
        "time_code": payload.get("time_code"),
        "local_code": payload.get("local_code"),
        "clock_quality": payload.get("clock_quality"),
    }


def _data_quality(payload: dict, event_window: Optional[EventWindow]) -> dict[str, Any]:
    channels = payload.get("analog_channels", [])
    has_voltage = any(c.get("measurement") == "voltage" for c in channels)
    has_current = any(c.get("measurement") == "current" for c in channels)
    has_digital = len(payload.get("status_channels", [])) > 0
    return {
        "has_voltage_channels": has_voltage,
        "has_current_channels": has_current,
        "has_digital_channels": has_digital,
        "current_only": has_current and not has_voltage,
        "analog_channel_count": len(channels),
        "status_channel_count": len(payload.get("status_channels", [])),
        "parser_warnings": list(payload.get("warnings") or []),
        "timing_warnings": list(event_window.warnings) if event_window else [],
    }


def _missing_evidence(payload: dict, data_quality: dict, event_window: Optional[EventWindow]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    if not data_quality["has_voltage_channels"]:
        missing.append({
            "type": "VOLTAGE_CHANNELS",
            "description": "No voltage channels present - impedance/locus and voltage-sag evidence unavailable.",
        })
    if not data_quality["has_digital_channels"]:
        missing.append({
            "type": "DIGITAL_CHANNELS",
            "description": "No digital/status channels present - protection interpretation relies on waveform evidence only.",
        })
    if event_window is not None and event_window.method in ("no_fault_evidence", "insufficient_data"):
        missing.append({
            "type": "FAULT_EVIDENCE",
            "description": "No usable fault evidence found in this recording.",
        })
    if event_window is not None and event_window.method == "trigger_fallback":
        missing.append({
            "type": "DETECTED_INCEPTION",
            "description": "Canonical inception could not be detected; only the CFG trigger time is available.",
        })
    if event_window is not None and event_window.clearing_time_ms is None and event_window.inception_time_ms is not None:
        missing.append({
            "type": "CLEARING_EVIDENCE",
            "description": "Fault clearing time could not be determined from available evidence.",
        })
    missing.append({
        "type": "REMOTE_END_RECORD",
        "description": "Remote-end COMTRADE has not been provided (single-record analysis only).",
    })
    return missing


def build_record_analysis(analysis_id: str, payload: dict) -> RecordAnalysis:
    """Build the canonical :class:`RecordAnalysis` for a stored COMTRADE payload.

    Deliberately Stage-0 scoped: at most one fault episode, no cross-record
    correlation. ``fault_episodes`` schema is a list so Stage 1-2 can extend
    it to multiple episodes without a breaking change here.
    """
    event_window = build_event_window(payload)
    source_metadata = _source_metadata(payload)
    data_quality = _data_quality(payload, event_window)

    det = detect_fault_presence(payload)
    no_fault = det.no_fault

    provenance = {
        "timing_source": event_window.method if event_window else "insufficient_data",
        "timing_confidence": round(event_window.confidence, 3) if event_window else 0.0,
        "no_fault_gate_reasons": det.reasons,
        "schema_version": "stage0",
    }

    observed_facts: dict[str, Any] = {
        "faulted_phases": event_window.faulted_phases if event_window else [],
        "reclose_attempted": bool(event_window.reclose_events) if event_window else False,
        "reclose_events": event_window.reclose_events if event_window else [],
    }

    protection_interpretation: dict[str, Any] = {}
    fault_episodes: list[dict[str, Any]] = []

    if no_fault:
        protection_interpretation = {
            "event_class": "NO_FAULT_TRIGGER",
            "summary": "Recording triggered without protection operating and without a fault signature.",
        }
    elif event_window and event_window.inception_time_ms is not None:
        episode = {
            "episode_index": 0,
            "inception_time_ms": event_window.inception_time_ms,
            "clearing_time_ms": event_window.clearing_time_ms,
            "fault_duration_ms": event_window.fault_duration_ms,
            "faulted_phases": event_window.faulted_phases,
            "detection_method": event_window.method,
            "confidence": event_window.confidence,
            "reclose_events": event_window.reclose_events,
        }
        fault_episodes.append(episode)
        protection_interpretation = {
            "event_class": "TRANSIENT_LINE_FAULT" if event_window.reclose_events else "FAULT_EVENT",
            "summary": "Fault detected from available waveform/status evidence.",
        }
    else:
        protection_interpretation = {
            "event_class": "UNDETERMINED",
            "summary": "Fault presence could not be confirmed or denied from available evidence.",
        }

    missing_evidence = _missing_evidence(payload, data_quality, event_window)

    return RecordAnalysis(
        record_id=analysis_id,
        source_metadata=source_metadata,
        data_quality=data_quality,
        event_window=event_window,
        fault_episodes=fault_episodes,
        protection_operations=[],
        electrical_measurements={},
        cause_hypotheses=[],
        missing_evidence=missing_evidence,
        provenance=provenance,
        observed_facts=observed_facts,
        protection_interpretation=protection_interpretation,
    )
