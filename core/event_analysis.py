"""Canonical single-record event window — Stage 0.

Single source of truth for "when did the fault start/clear on this COMTRADE
record?". Every consumer (SOE, locus events, electrical parameters, feature
extraction / AI inference, waveform display, report generation) must derive
its inception/clearing timing from the :class:`EventWindow` produced here
instead of recomputing its own threshold-based detector.

This module operates on the stored COMTRADE *payload* dict (the JSON shape
produced by ``webapp/api/routers/upload.py``), the same convention used by
``webapp/api/fault_detection.py`` — so it can be called directly by any
webapp router without re-parsing files. Internally it reuses
``core.fault_detector.detect_fault``, which already implements status-channel
and waveform-based inception/clearing/reclose detection; this module does not
re-implement that logic, it adapts the payload into the shape
``fault_detector`` expects and translates the result into the canonical
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .fault_detector import FaultEvent, detect_fault
from .fault_detector import _normalize_fault_phase_list  # reuse phase ordering


@dataclass
class EventWindow:
    """Canonical fault timing for one COMTRADE record."""

    record_start_ms: float
    trigger_time_ms: Optional[float]
    inception_idx: Optional[int]
    inception_time_ms: Optional[float]
    clearing_idx: Optional[int]
    clearing_time_ms: Optional[float]
    fault_duration_ms: Optional[float]
    method: str                      # detection_method from FaultEvent, or "trigger_fallback" / "no_fault" / "insufficient_data"
    confidence: float                # 0-1
    faulted_phases: list[str] = field(default_factory=list)
    reclose_events: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def timing_source(self) -> str:
        """Provenance label for UI/report consumption."""
        return self.method

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_start_ms": self.record_start_ms,
            "trigger_time_ms": self.trigger_time_ms,
            "inception_idx": self.inception_idx,
            "inception_time_ms": self.inception_time_ms,
            "clearing_idx": self.clearing_idx,
            "clearing_time_ms": self.clearing_time_ms,
            "fault_duration_ms": self.fault_duration_ms,
            "method": self.method,
            "timing_source": self.method,
            "confidence": self.confidence,
            "faulted_phases": self.faulted_phases,
            "reclose_events": self.reclose_events,
            "warnings": self.warnings,
        }


class _ShimChannel:
    """Adapts a payload channel dict to the attribute shape fault_detector expects."""

    __slots__ = ("id", "name", "canonical_name", "unit", "phase", "measurement", "samples")

    def __init__(self, ch: dict):
        self.id = ch.get("id", "")
        self.name = ch.get("name", "") or ""
        self.canonical_name = ch.get("canonical_name", "") or ""
        self.unit = ch.get("unit", "") or ""
        self.phase = ch.get("phase")
        self.measurement = ch.get("measurement", "") or ""
        self.samples = np.asarray(ch.get("samples") or [], dtype=float)


class _ShimStatusChannel:
    __slots__ = ("id", "name", "samples")

    def __init__(self, ch: dict):
        self.id = ch.get("id", "")
        self.name = ch.get("name", "") or ""
        self.samples = np.asarray(ch.get("samples") or [], dtype=int)


class _ShimRecord:
    """Minimal adapter so core.fault_detector.detect_fault can run on a payload dict."""

    __slots__ = ("analog_channels", "status_channels", "time")

    def __init__(self, payload: dict):
        self.analog_channels = [_ShimChannel(c) for c in payload.get("analog_channels", [])]
        self.status_channels = [_ShimStatusChannel(c) for c in payload.get("status_channels", [])]
        self.time = np.asarray(payload.get("time") or [], dtype=float)


def _insufficient_data_window(payload: dict, warnings: list[str]) -> EventWindow:
    trigger_s = payload.get("trigger_offset_s", payload.get("trigger_time", 0.0)) or 0.0
    return EventWindow(
        record_start_ms=0.0,
        trigger_time_ms=float(trigger_s) * 1000.0 if trigger_s else None,
        inception_idx=None,
        inception_time_ms=None,
        clearing_idx=None,
        clearing_time_ms=None,
        fault_duration_ms=None,
        method="insufficient_data",
        confidence=0.0,
        faulted_phases=[],
        reclose_events=[],
        warnings=warnings,
    )


def build_event_window(payload: dict) -> EventWindow:
    """Compute the canonical :class:`EventWindow` for a stored COMTRADE payload.

    This is the ONLY place in the webapp that should call
    ``core.fault_detector.detect_fault``. All endpoints needing inception,
    clearing, duration, faulted phases, or reclose events must go through
    this function (directly or via the cached :class:`RecordAnalysis`).
    """
    time = np.asarray(payload.get("time") or [], dtype=float)
    warnings: list[str] = list(payload.get("warnings") or [])

    if len(time) < 4:
        warnings.append("Recording too short to determine a fault window")
        return _insufficient_data_window(payload, warnings)

    record_start_ms = float(time[0]) * 1000.0
    trigger_s = payload.get("trigger_offset_s", payload.get("trigger_time", 0.0)) or 0.0
    trigger_time_ms = float(trigger_s) * 1000.0

    has_voltage = any(
        (c.get("measurement") == "voltage") for c in payload.get("analog_channels", [])
    )
    has_current = any(
        (c.get("measurement") == "current") for c in payload.get("analog_channels", [])
    )
    has_digital = len(payload.get("status_channels", [])) > 0

    if not has_voltage:
        warnings.append("No voltage channels present - timing/impedance evidence is current-only")
    if not has_current:
        warnings.append("No current channels present - cannot detect fault from waveform evidence")
    if not has_digital:
        warnings.append("No digital/status channels present - timing relies on waveform evidence only")

    shim = _ShimRecord(payload)

    try:
        fault: Optional[FaultEvent] = detect_fault(shim)
    except Exception as exc:  # pragma: no cover - defensive; detector must not crash the endpoint
        warnings.append(f"Fault window detection failed: {exc}")
        fault = None

    if fault is None:
        # No evidence of a fault at all — this is a legitimate outcome (no-fault
        # trigger), not an error. Do not fabricate an inception time; fall back
        # to CFG trigger time only, clearly labeled as a fallback.
        warnings.append("No fault evidence found in this recording (no-fault or undetectable)")
        return EventWindow(
            record_start_ms=record_start_ms,
            trigger_time_ms=trigger_time_ms if trigger_time_ms else None,
            inception_idx=None,
            inception_time_ms=None,
            clearing_idx=None,
            clearing_time_ms=None,
            fault_duration_ms=None,
            method="trigger_fallback" if trigger_time_ms else "no_fault_evidence",
            confidence=0.0,
            faulted_phases=[],
            reclose_events=[],
            warnings=warnings,
        )

    inception_time_ms = float(fault.inception_time) * 1000.0 if fault.inception_time is not None else None
    clearing_time_ms = (
        float(fault.clearing_time) * 1000.0 if fault.clearing_time is not None else None
    )
    duration_ms = fault.duration_ms if fault.duration_ms else (
        None if clearing_time_ms is None else max(0.0, clearing_time_ms - (inception_time_ms or 0.0))
    )
    if fault.detection_method == "dead_time_recording":
        warnings.append(
            "Recording started during CB dead time - fault preceded this file; "
            "clearing time and duration are not measurable from this record"
        )
    if fault.duration_ms == 0.0 and fault.clearing_idx is None:
        warnings.append("Fault clearing evidence not available - duration left unset rather than guessed")

    return EventWindow(
        record_start_ms=record_start_ms,
        trigger_time_ms=trigger_time_ms if trigger_time_ms else None,
        inception_idx=int(fault.inception_idx) if fault.inception_idx is not None else None,
        inception_time_ms=inception_time_ms,
        clearing_idx=int(fault.clearing_idx) if fault.clearing_idx is not None else None,
        clearing_time_ms=clearing_time_ms,
        fault_duration_ms=duration_ms,
        method=fault.detection_method,
        confidence=float(fault.confidence),
        faulted_phases=_normalize_fault_phase_list(fault.faulted_phases),
        reclose_events=list(fault.reclose_events or []),
        warnings=warnings,
    )
