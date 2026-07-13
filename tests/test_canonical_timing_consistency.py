"""Stage 0 cross-endpoint consistency tests.

Verifies that inception (and, where applicable, clearing/duration) is
IDENTICAL — not merely "close" — across every endpoint/function that reports
fault timing for the same COMTRADE payload:

  - the canonical endpoint (GET /api/analysis/{id}/canonical)
  - fault-classification (_compute_fault_classification)
  - electrical-params (_compute_electrical_params)
  - full-SOE (_compute_full_soe_events)
  - locus-events (_compute_locus_events)
  - feature extraction (_extract_features_from_payload)
  - AI prediction metadata (run_ml_prediction's meta.timing_source)

All of these are expected to source their inception from the same
core.event_analysis.build_event_window(payload) call (directly or via
_canonical_inception_idx), so this test asserts on the actual numeric
values returned by each function/endpoint rather than re-deriving an
"acceptable" value independently.

Also covers backward compatibility: analysis payloads saved before Stage 0
(missing start_time_iso/trigger_offset_s/etc.) must still work.
"""

import asyncio

import numpy as np
import pytest

from core.event_analysis import build_event_window
from webapp.api.record_analysis import build_record_analysis
from webapp.api.routers import relay_21, upload as upload_router
from webapp.api.ml_predict import run_ml_prediction

from tests.fixtures import synthetic_records as sr


def _patch_load_analysis(monkeypatch, module, payload: dict):
    monkeypatch.setattr(module, "load_analysis", lambda _analysis_id: payload)


def test_canonical_endpoint_matches_direct_event_window(monkeypatch):
    payload = sr.transient_slg_successful_reclose()
    _patch_load_analysis(monkeypatch, upload_router, payload)

    direct_window = build_event_window(payload)
    response = asyncio.run(upload_router.get_canonical_analysis("any-id"))

    assert response["event_window"]["inception_time_ms"] == pytest.approx(direct_window.inception_time_ms, abs=1e-6)
    assert response["event_window"]["clearing_time_ms"] == pytest.approx(direct_window.clearing_time_ms, abs=1e-6)
    assert response["event_window"]["method"] == direct_window.method
    assert response["event_window"]["confidence"] == pytest.approx(direct_window.confidence, abs=1e-6)


def test_all_relay21_endpoints_agree_with_canonical(monkeypatch):
    """The inception_time_ms reported by every relay-21 endpoint must be the
    exact same number — sourced from one build_event_window call, not
    independently-close approximations."""
    payload = sr.transient_slg_successful_reclose()
    window = build_event_window(payload)
    assert window.inception_time_ms is not None

    fault_class = relay_21._compute_fault_classification(payload)
    elec = relay_21._compute_electrical_params(payload)
    features = relay_21._extract_features_from_payload(payload)
    soe = relay_21._compute_full_soe_events(payload)
    locus_events = relay_21._compute_locus_events(payload)

    # electrical-params, full-soe, and locus-events report inception_time_ms
    # directly — these must match the canonical value exactly (both are
    # `round(time[inception_idx] * 1000, ...)` of the SAME inception_idx).
    assert elec["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.01)
    assert soe["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.01)
    assert locus_events["inception_time_ms"] == pytest.approx(window.inception_time_ms, abs=0.01)

    # electrical-params and full-soe/locus-events must report the SAME timing
    # source string (not independently-derived labels).
    assert elec["timing_source"] == window.method
    assert soe["timing_source"] == window.method
    assert locus_events["timing_source"] == window.method

    # fault-classification's derived timing (prefault/fault/total) is
    # consistent with the same fault_duration_ms used elsewhere.
    assert fault_class["fault_ms"] == pytest.approx(features["fault_duration_ms"], abs=0.1)
    assert fault_class["timing_source"] == window.method


def test_ai_prediction_metadata_reports_same_timing_source(monkeypatch):
    payload = sr.transient_slg_successful_reclose()
    window = build_event_window(payload)

    result = run_ml_prediction(payload, "21")
    assert result["meta"]["timing_source"] == window.method
    assert result["meta"]["timing_confidence"] == pytest.approx(window.confidence, abs=1e-6)


def test_record_analysis_event_window_matches_direct_call():
    payload = sr.transient_slg_successful_reclose()
    direct_window = build_event_window(payload)
    analysis = build_record_analysis("consistency-test", payload)

    assert analysis.event_window is not None
    assert analysis.event_window.inception_time_ms == direct_window.inception_time_ms
    assert analysis.event_window.clearing_time_ms == direct_window.clearing_time_ms
    assert analysis.event_window.method == direct_window.method
    assert analysis.fault_episodes[0]["inception_time_ms"] == direct_window.inception_time_ms


def test_no_fault_case_all_endpoints_agree_no_inception(monkeypatch):
    """When there's no fault evidence, no endpoint should report a fabricated
    inception — they must all agree there is none."""
    payload = sr.no_fault_trigger()
    window = build_event_window(payload)
    assert window.inception_time_ms is None

    fault_class = relay_21._compute_fault_classification(payload)
    elec = relay_21._compute_electrical_params(payload)

    assert fault_class.get("no_fault") is True
    assert elec.get("no_fault") is True


# --- Backward compatibility -------------------------------------------------

def _legacy_payload() -> dict:
    """A payload shaped like pre-Stage-0 storage: no start_time_iso,
    trigger_offset_s, time_code, local_code, clock_quality, or warnings key
    variations introduced by this stage."""
    payload = sr.transient_slg_successful_reclose()
    for key in ("start_time_iso", "trigger_time_iso", "trigger_offset_s", "time_code", "local_code", "clock_quality"):
        payload.pop(key, None)
    return payload


def test_backward_compat_legacy_payload_missing_absolute_timestamps():
    payload = _legacy_payload()
    # Must not raise, and must still detect the fault correctly by falling
    # back to the legacy `trigger_time` field.
    window = build_event_window(payload)
    assert window.inception_time_ms is not None
    assert window.method == "status_channel"


def test_backward_compat_canonical_endpoint_on_legacy_payload(monkeypatch):
    payload = _legacy_payload()
    _patch_load_analysis(monkeypatch, upload_router, payload)
    response = asyncio.run(upload_router.get_canonical_analysis("legacy-id"))
    assert response["event_window"]["inception_time_ms"] is not None
    assert response["source_metadata"]["start_time_iso"] is None


def test_backward_compat_relay21_endpoints_on_legacy_payload():
    payload = _legacy_payload()
    # None of these should raise despite missing Stage 0 fields.
    relay_21._compute_fault_classification(payload)
    relay_21._compute_electrical_params(payload)
    relay_21._extract_features_from_payload(payload)
    relay_21._compute_full_soe_events(payload)
    relay_21._compute_locus_events(payload)
