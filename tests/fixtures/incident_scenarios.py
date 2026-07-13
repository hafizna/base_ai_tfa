"""Golden multi-record incident scenarios — Stage 2.

Each function returns a list of ``(payload, overrides)`` tuples suitable for
saving through ``webapp.api.storage.save_analysis`` and attaching to an
incident via ``webapp.api.incidents.service.attach_record``. Built from the
Stage 0 synthetic single-record payloads in ``tests/fixtures/synthetic_records.py``
so timing/fault-detection stays exactly Stage-0-canonical; this module only
adds incident-level metadata (absolute timestamps, relay id, station name)
needed to exercise Stage 2 alignment/relationship/episode logic.
"""

from __future__ import annotations

import copy
from typing import Any

from . import synthetic_records as sr

STATION = "GOLDEN TEST"


def _with_time(payload: dict, trigger_iso: str, start_iso: str | None = None) -> dict:
    payload = copy.deepcopy(payload)
    payload["trigger_time_iso"] = trigger_iso
    payload["start_time_iso"] = start_iso or trigger_iso
    return payload


def duplicate_captures_different_relays() -> list[tuple[dict, dict]]:
    """Case 1: two duplicate captures of the same electrical event from
    different relays/DFRs — identical waveform, same absolute time."""
    base = sr.transient_slg_successful_reclose()
    p1 = _with_time(base, "2026-02-01T10:00:00.300+00:00", "2026-02-01T10:00:00+00:00")
    p1["rec_dev_id"] = "RELAY_21"
    p2 = _with_time(base, "2026-02-01T10:00:00.300+00:00", "2026-02-01T10:00:00+00:00")
    p2["rec_dev_id"] = "DFR_EXTERNAL"
    return [
        (p1, {"relay_id": "RELAY_21", "protection_type": "21"}),
        (p2, {"relay_id": "DFR_EXTERNAL", "protection_type": "21"}),
    ]


def overlapping_not_full_duplicate() -> list[tuple[dict, dict]]:
    """Case 2: overlapping time ranges but not close enough to be a
    duplicate (different fault magnitude / phase content)."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00.300+00:00", "2026-02-01T10:00:00+00:00")
    p1["rec_dev_id"] = "RELAY_A"

    p2 = sr.permanent_fault_failed_reclose()
    # Shift so its window overlaps p1's window but signature differs (permanent, not transient).
    p2 = _with_time(p2, "2026-02-01T10:00:00.350+00:00", "2026-02-01T10:00:00.05+00:00")
    p2["rec_dev_id"] = "RELAY_B"
    return [
        (p1, {"relay_id": "RELAY_A", "protection_type": "21"}),
        (p2, {"relay_id": "RELAY_B", "protection_type": "21"}),
    ]


def continuation_sequence() -> list[tuple[dict, dict]]:
    """Case 3: second record starts while the first's fault/reclose sequence
    is still in progress (continuation, not independent new fault)."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00.300+00:00", "2026-02-01T10:00:00+00:00")

    p2 = sr.permanent_fault_failed_reclose()
    # Starts 0.5s after p1's trigger — well within p1's ~0.9s reclose window.
    p2 = _with_time(p2, "2026-02-01T10:00:00.800+00:00", "2026-02-01T10:00:00.5+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def successful_reclose_separate_record() -> list[tuple[dict, dict]]:
    """Case 4: fault captured in one record, the breaker's successful
    reclose captured in a second, closely-following record."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00.300+00:00", "2026-02-01T10:00:00+00:00")

    p2 = sr.transient_slg_successful_reclose()
    p2 = _with_time(p2, "2026-02-01T10:00:01.000+00:00", "2026-02-01T10:00:00.9+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def failed_reclose_trip_on_reclose() -> list[tuple[dict, dict]]:
    """Case 5: failed reclose and trip-on-reclose captured across records."""
    p1 = sr.permanent_fault_failed_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00.300+00:00", "2026-02-01T10:00:00+00:00")

    p2 = sr.permanent_fault_failed_reclose()
    p2 = _with_time(p2, "2026-02-01T10:00:01.200+00:00", "2026-02-01T10:00:01.0+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def repeated_ag_faults_with_gap() -> list[tuple[dict, dict]]:
    """Case 6: two independent A-G faults on the same bay, minutes apart."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00+00:00")

    p2 = sr.transient_slg_successful_reclose()
    p2 = _with_time(p2, "2026-02-01T10:07:00+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def possible_evolving_fault() -> list[tuple[dict, dict]]:
    """Case 7: A-G transient (successful reclose) followed shortly by an
    A-B-G permanent fault with failed reclose — phase set progresses."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00+00:00")

    p2 = sr.permanent_fault_failed_reclose()
    # Force a 2-phase signature to represent phase progression A-G -> A-B-G.
    p2 = copy.deepcopy(p2)
    for ch in p2["analog_channels"]:
        if ch["canonical_name"] == "IB":
            import numpy as np
            arr = np.asarray(ch["samples"], dtype=float)
            t = np.asarray(p2["time"], dtype=float)
            fault_region = slice(int(0.3 * 1200), None)
            arr[fault_region] = 45.0 * np.sin(2 * np.pi * 50.0 * t[fault_region] + np.deg2rad(-120))
            ch["samples"] = arr.tolist()
    p2 = _with_time(p2, "2026-02-01T10:01:30+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def unrelated_records_wrongly_grouped() -> list[tuple[dict, dict]]:
    """Case 8: two records that should NOT have been grouped — different
    stations entirely, far apart in time, no fault-signature relationship."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-01-01T00:00:00+00:00")
    p1["station_name"] = "GI ALPHA"

    p2 = sr.current_only_recording()
    p2 = _with_time(p2, "2026-06-15T12:00:00+00:00")
    p2["station_name"] = "GI OMEGA"
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "OTHER", "protection_type": "21", "override_warnings": True}),
    ]


def missing_absolute_timestamp_manual_order() -> list[tuple[dict, dict]]:
    """Case 9: neither record has absolute time; rely on manual order."""
    p1 = sr.transient_slg_successful_reclose()
    for key in ("start_time_iso", "trigger_time_iso"):
        p1.pop(key, None)
    p2 = sr.permanent_fault_failed_reclose()
    for key in ("start_time_iso", "trigger_time_iso"):
        p2.pop(key, None)
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def conflicting_timestamp_order() -> list[tuple[dict, dict]]:
    """Case 10: manual order says A-then-B, but timestamps say B-then-A."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:05:00+00:00")  # later in time

    p2 = sr.permanent_fault_failed_reclose()
    p2 = _with_time(p2, "2026-02-01T10:00:00+00:00")  # earlier in time
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),  # manual_order=0 despite later timestamp
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),  # manual_order=1
    ]


def long_gap_compressed_timeline() -> list[tuple[dict, dict]]:
    """Case 11: two records separated by a very long gap (hours)."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T08:00:00+00:00")

    p2 = sr.transient_slg_successful_reclose()
    p2 = _with_time(p2, "2026-02-01T14:00:00+00:00")  # 6 hours later
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def current_only_record_in_incident() -> list[tuple[dict, dict]]:
    """Case 12: one normal record plus one current-only record (no voltage)."""
    p1 = sr.transient_slg_successful_reclose()
    p1 = _with_time(p1, "2026-02-01T10:00:00+00:00")

    p2 = sr.current_only_recording()
    p2 = _with_time(p2, "2026-02-01T10:05:00+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]


def no_fault_plus_actual_fault() -> list[tuple[dict, dict]]:
    """Case 13: one no-fault trigger record plus one real fault record."""
    p1 = sr.no_fault_trigger()
    p1 = _with_time(p1, "2026-02-01T09:00:00+00:00")

    p2 = sr.transient_slg_successful_reclose()
    p2 = _with_time(p2, "2026-02-01T10:00:00+00:00")
    return [
        (p1, {"relay_id": "SYNTH", "protection_type": "21"}),
        (p2, {"relay_id": "SYNTH", "protection_type": "21"}),
    ]
