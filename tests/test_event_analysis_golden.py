"""Stage 0 golden regression cases for the canonical event window.

Each case verifies: expected inception, expected clearing, expected
duration, expected faulted phases, expected trip/reclose outcome, expected
timing source, and expected warnings — using synthetic payloads from
tests/fixtures/synthetic_records.py (same style as
tests/test_no_fault_gate.py's _balanced_load_payload).
"""

import numpy as np
import pytest

from core.event_analysis import build_event_window
from webapp.api.record_analysis import build_record_analysis
from webapp.api.routers.relay_21 import (
    _compute_electrical_params,
    _compute_locus_events,
    _compute_full_soe_events,
    _extract_features_from_payload,
)
from webapp.api.ml_predict import extract_ml_features

from tests.fixtures import synthetic_records as sr


# 1. No-fault trigger.
def test_golden_no_fault_trigger():
    payload = sr.no_fault_trigger()
    window = build_event_window(payload)
    assert window.method in ("no_fault_evidence", "trigger_fallback")
    assert window.inception_time_ms is None
    assert window.clearing_time_ms is None
    assert window.fault_duration_ms is None
    assert window.faulted_phases == []
    assert window.confidence == 0.0
    assert any("no-fault" in w.lower() or "no fault" in w.lower() for w in window.warnings)

    analysis = build_record_analysis("golden-1", payload)
    assert analysis.protection_interpretation["event_class"] == "NO_FAULT_TRIGGER"
    assert analysis.fault_episodes == []


# 2. Transient single-phase-to-ground fault with successful reclose.
def test_golden_transient_slg_successful_reclose():
    payload = sr.transient_slg_successful_reclose()
    window = build_event_window(payload)
    assert window.method == "status_channel"
    assert window.timing_source == "status_channel"
    assert window.inception_time_ms == pytest.approx(312.5, abs=2.0)
    assert window.clearing_time_ms == pytest.approx(412.5, abs=2.0)
    assert window.fault_duration_ms == pytest.approx(100.0, abs=3.0)
    assert window.faulted_phases == ["A"]
    assert len(window.reclose_events) == 1
    assert window.reclose_events[0]["success"] is True

    analysis = build_record_analysis("golden-2", payload)
    assert len(analysis.fault_episodes) == 1
    assert analysis.fault_episodes[0]["faulted_phases"] == ["A"]
    assert analysis.observed_facts["reclose_attempted"] is True


# 3. Permanent fault with failed reclose.
def test_golden_permanent_fault_failed_reclose():
    payload = sr.permanent_fault_failed_reclose()
    window = build_event_window(payload)
    assert window.inception_time_ms == pytest.approx(312.5, abs=2.0)
    assert window.faulted_phases == ["A"]
    assert len(window.reclose_events) == 1
    assert window.reclose_events[0]["success"] is False


# 4. Status trip appears later than waveform inception.
def test_golden_status_trip_lags_waveform():
    payload = sr.status_trip_lags_waveform()
    window = build_event_window(payload)
    # Status onset lags the waveform onset by > 1 cycle. core.fault_detector
    # reconciles this by preferring the waveform-derived inception time (not
    # the late status-channel edge) — whether the returned method label ends
    # up "status_waveform_aligned" or "current_derivative" depends on whether
    # clearing evidence was also found, but the TIME must be the waveform
    # onset either way, never the late status edge (230ms).
    assert window.method in ("status_waveform_aligned", "current_derivative")
    assert window.inception_time_ms == pytest.approx(200.0, abs=2.0)
    assert window.inception_time_ms < 230.0
    assert window.faulted_phases == ["A"]


# 5. Recording starts during CB dead time.
def test_golden_recording_starts_in_dead_time():
    payload = sr.recording_starts_in_dead_time()
    window = build_event_window(payload)
    assert window.method == "dead_time_recording"
    assert window.inception_time_ms == 0.0
    # Duration is genuinely unknown — must not be fabricated.
    assert window.fault_duration_ms is None
    assert window.clearing_time_ms is None
    assert len(window.reclose_events) == 1
    assert window.reclose_events[0]["success"] is True
    assert any("dead time" in w.lower() for w in window.warnings)


# 6. Current-only recording (no voltage channels).
def test_golden_current_only_recording():
    payload = sr.current_only_recording()
    window = build_event_window(payload)
    assert window.inception_time_ms is not None
    assert window.faulted_phases == ["A"]
    assert any("no voltage" in w.lower() for w in window.warnings)

    analysis = build_record_analysis("golden-6", payload)
    assert analysis.data_quality["current_only"] is True
    assert any(m["type"] == "VOLTAGE_CHANNELS" for m in analysis.missing_evidence)


# 7. Missing / ambiguous digital channels.
def test_golden_missing_digital_channels():
    payload = sr.missing_ambiguous_digital()
    window = build_event_window(payload)
    assert window.method == "current_derivative"
    assert window.inception_time_ms is not None
    assert any("no digital" in w.lower() for w in window.warnings)

    analysis = build_record_analysis("golden-7", payload)
    assert analysis.data_quality["has_digital_channels"] is False
    assert any(m["type"] == "DIGITAL_CHANNELS" for m in analysis.missing_evidence)


# 8. CT/VT ratio correction does not change canonical timing.
def test_golden_ct_vt_ratio_correction_preserves_timing():
    payload = sr.transient_slg_successful_reclose()
    window_before = build_event_window(payload)

    scaled_payload = {
        **payload,
        "analog_channels": [
            {
                **ch,
                "samples": (np.asarray(ch["samples"], dtype=float) * 5.0).tolist(),
                "ct_primary": ch["ct_primary"] * 5.0,
            }
            if ch["measurement"] == "current"
            else ch
            for ch in payload["analog_channels"]
        ],
    }
    window_after = build_event_window(scaled_payload)

    # Amplitude scaling must not change which sample index is inception —
    # the threshold logic is ratio-based (relative to prefault), not absolute.
    assert window_after.inception_idx == window_before.inception_idx
    assert window_after.method == window_before.method


# 9. SOE relative time uses the same inception as the waveform marker.
def test_golden_soe_rel_ms_matches_canonical_inception():
    payload = sr.transient_slg_successful_reclose()
    window = build_event_window(payload)
    soe = _compute_full_soe_events(payload)
    locus_events = _compute_locus_events(payload)

    assert soe["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.1)
    assert locus_events["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.1)

    # Every event's rel_ms must be computed against that same inception.
    for ev in soe["events"]:
        expected_rel = ev["time_ms"] - soe["inception_time_ms"]
        assert ev["rel_ms"] == pytest.approx(expected_rel, abs=0.05)


# 10. (Backend-side proxy for "frontend does not compute independent inception
# when backend timing is available") — every consumer of canonical timing
# agrees with the same EventWindow, so there is one number for the frontend
# to adopt rather than several to choose between.
def test_golden_all_backend_consumers_agree_on_inception():
    payload = sr.transient_slg_successful_reclose()
    window = build_event_window(payload)

    elec = _compute_electrical_params(payload)
    features = _extract_features_from_payload(payload)
    ml_row = extract_ml_features(payload, "21")
    soe = _compute_full_soe_events(payload)

    assert elec["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.1)
    assert soe["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.1)
    # extract-features and ML feature extraction report duration derived from
    # the same inception index, so their fault_duration_ms values must match
    # (both consume the identical canonical inception_idx).
    assert features["fault_duration_ms"] == pytest.approx(ml_row["fault_duration_ms"], abs=0.1)
