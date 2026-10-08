"""A fault, its reclose and a re-fault captured as separate COMTRADE files.

External DFRs re-trigger when the breaker recloses, so one incident often
arrives as three files: the fault and trip; a file that starts in breaker
dead time and holds only the reclose; and, if the reclose does not hold, the
re-fault. Shape taken from the 21/08/2023 Bringin-Mojosongo #2 tree fault
(dead time ~5 s, re-fault 5.7 s after the reclose, same B-C phases), at both
line ends: GI Bringin has breaker status, GI Mojosongo's Qualitrol DFR has
none, so its reclose record has to be recognised from the waveform alone.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import numpy as np
import pytest

from core.event_analysis import _ShimRecord, build_event_window
from core.fault_detector import _energization_after_dead_start
from webapp.api.incidents.models import AlignmentAssessment, IncidentRecord
from webapp.api.incidents.relationships import classify_pair
from webapp.api.ml_predict import run_ml_prediction
from webapp.api.record_analysis import build_record_analysis
from webapp.api.storage import save_analysis

SR = 2400.0
FREQ = 50.0
T = np.arange(int(SR * 1.0)) / SR
W = 2 * np.pi * FREQ


def _sine(amp, phase_deg):
    return amp * np.sin(W * T + np.deg2rad(phase_deg))


def _three(amp):
    return [_sine(amp, 0), _sine(amp, -120), _sine(amp, 120)]


def _status(name, rises_at=None, falls_at=None, initial=0):
    samples = np.full(len(T), initial, dtype=int)
    if rises_at is not None:
        samples[T >= rises_at] = 1
    if falls_at is not None:
        samples[T >= falls_at] = 0
    return {"id": name, "name": name, "samples": samples.tolist()}


def _payload(currents, voltages, status, start_iso, trigger_offset_s=0.1):
    analog = []
    for canon, samples in zip(("IA", "IB", "IC"), currents):
        analog.append({"id": canon, "name": canon, "canonical_name": canon, "unit": "A", "phase": canon[-1],
                       "measurement": "current", "ct_primary": 1.0, "ct_secondary": 1.0, "pors": "P",
                       "samples": np.asarray(samples, dtype=float).tolist()})
    for canon, samples in zip(("VA", "VB", "VC"), voltages):
        analog.append({"id": canon, "name": canon, "canonical_name": canon, "unit": "kV", "phase": canon[-1],
                       "measurement": "voltage", "ct_primary": 1.0, "ct_secondary": 1.0, "pors": "P",
                       "samples": np.asarray(samples, dtype=float).tolist()})
    start = datetime.fromisoformat(start_iso)
    return {
        "station_name": "GI TEST", "rec_dev_id": "DFR", "frequency": FREQ, "time": T.tolist(),
        "trigger_offset_s": trigger_offset_s, "trigger_time": trigger_offset_s, "total_samples": len(T),
        "start_time_iso": start_iso, "trigger_time_iso": (start + timedelta(seconds=trigger_offset_s)).isoformat(),
        "analog_channels": analog, "status_channels": status, "warnings": [],
    }


def fault_and_trip(start_iso, fault=(0.10, 0.20), with_status=True):
    """B-C fault, Z1 trip, breaker opens; line-side VT and CT read ~0 after."""
    in_fault = (T >= fault[0]) & (T < fault[1])
    dead = T >= fault[1]
    ia, ib, ic = _three(300.0)
    ib = np.where(in_fault, _sine(5000.0, -200), ib)
    ic = np.where(in_fault, -_sine(5000.0, -200), ic)
    va, vb, vc = _three(87.0)
    vb = np.where(in_fault, _sine(52.0, -150), vb)
    vc = np.where(in_fault, _sine(52.0, 150), vc)
    currents = [np.where(dead, 0.0, x) for x in (ia, ib, ic)]
    voltages = [np.where(dead, 0.0, v) for v in (va, vb, vc)]
    status = [_status("TRIP Z1", fault[0] + 0.04, fault[1] + 0.01), _status("CB OPEN", fault[1] + 0.01)] if with_status else []
    return _payload(currents, voltages, status, start_iso)


def reclose_capture(start_iso, close_at=0.135, with_status=True, onto_fault=False):
    """Starts in dead time (line ~0); the breaker closes at ``close_at``."""
    live = T >= close_at
    currents = [np.where(live, x, 0.0) for x in _three(280.0)]
    voltages = [np.where(live, v, 0.0) for v in _three(87.0)]
    status = [_status("CB OPEN", initial=1, falls_at=close_at)] if with_status else []
    if onto_fault:
        # Closes onto the still-present B-C fault and trips again 100 ms later.
        in_fault = live & (T < close_at + 0.1)
        currents[1] = np.where(in_fault, _sine(5000.0, -200), currents[1])
        currents[2] = np.where(in_fault, -_sine(5000.0, -200), currents[2])
        voltages[1] = np.where(in_fault, _sine(26.0, -150), voltages[1])
        voltages[2] = np.where(in_fault, _sine(26.0, 150), voltages[2])
        tripped = T >= close_at + 0.1
        currents = [np.where(tripped, 0.0, x) for x in currents]
        voltages = [np.where(tripped, 0.0, v) for v in voltages]
        if with_status:
            status[0]["samples"] = np.where(tripped, 1, np.asarray(status[0]["samples"])).tolist()
    return _payload(currents, voltages, status, start_iso)


def bringin_like_sequence(with_status=True, refault_start="2026-02-01T10:00:10.700"):
    """fault+trip @10:00:00, reclose @10:00:05.135 (dead time 4.925 s), re-fault
    inception @10:00:10.820 (5.685 s after the reclose)."""
    return [
        fault_and_trip("2026-02-01T10:00:00.000", with_status=with_status),
        reclose_capture("2026-02-01T10:00:05.000", with_status=with_status),
        fault_and_trip(refault_start, fault=(0.12, 0.22), with_status=with_status),
    ]


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from webapp.api.incidents import storage as storage_module
    importlib.reload(storage_module)
    from webapp.api.incidents import service as service_module
    importlib.reload(service_module)
    return service_module


def _reconstruct(service, payloads):
    incident = service.create_incident(title="reclose linking", station_name="GI TEST")
    ids = []
    for payload in payloads:
        record = service.attach_record(incident.incident_id, analysis_id=save_analysis(payload), protection_type="21")
        ids.append(record.incident_record_id)
    recon = service.reconstruct(incident.incident_id)
    return incident, recon, ids


# --- the reclose record on its own ------------------------------------------

def test_reclose_record_with_breaker_status_is_a_reclose_capture():
    window = build_event_window(reclose_capture("2026-02-01T10:00:05.000"))
    assert window.method == "dead_time_recording"
    assert window.reclose_events[0]["success"] is True
    assert window.reclose_events[0]["time"] == pytest.approx(0.135, abs=1.0 / SR)


def test_reclose_record_without_breaker_status_is_recognised_from_the_waveform():
    window = build_event_window(reclose_capture("2026-02-01T10:00:05.000", with_status=False))
    assert window.method == "dead_time_recording"
    event = window.reclose_events[0]
    assert event["source"] == "waveform"
    assert event["success"] is True
    assert event["time"] == pytest.approx(0.135, abs=0.5 / FREQ)  # within half a cycle
    # Before: the energization inrush read as a three-phase fault inception.
    assert window.faulted_phases == []


def test_breaker_auxiliary_contact_bounce_is_not_a_reopen():
    # Real case (Kudus-Jekulo #2): CB OPEN 1 -> 0 -> 1 -> 0 within 5 ms at the
    # close, line healthy afterwards — a successful reclose.
    payload = reclose_capture("2026-02-01T10:00:05.000")
    samples = np.asarray(payload["status_channels"][0]["samples"])
    samples[(T >= 0.136) & (T < 0.140)] = 1
    payload["status_channels"][0]["samples"] = samples.tolist()
    assert build_event_window(payload).reclose_events[0]["success"] is True


def test_closing_onto_the_fault_is_a_failed_reclose():
    window = build_event_window(reclose_capture("2026-02-01T10:00:05.000", onto_fault=True))
    assert window.method == "dead_time_recording"
    assert window.reclose_events[0]["success"] is False


@pytest.mark.parametrize("payload", [
    fault_and_trip("2026-02-01T10:00:00.000", with_status=False),
    # Starts inside a close-in three-phase fault: voltage ~0 but fault current flows.
    _payload([np.where(T < 0.1, _sine(6000.0, a), 0.0) for a in (0, -120, 120)],
             [np.zeros(len(T))] * 3, [], "2026-02-01T10:00:00.000"),
], ids=["ordinary-fault-record", "starts-inside-close-in-fault"])
def test_waveform_dead_start_never_matches_a_fault_record(payload):
    window = build_event_window(payload)
    assert window.method != "dead_time_recording"


def test_reclose_record_gets_its_own_event_class_and_no_ai_cause():
    payload = reclose_capture("2026-02-01T10:00:05.000", with_status=False)
    analysis = build_record_analysis("reclose", payload)
    assert analysis.protection_interpretation["event_class"] == "RECLOSE_CAPTURE"
    missing = {item["type"] for item in analysis.missing_evidence}
    assert "PRECEDING_FAULT_RECORD" in missing
    assert "NO_PROTECTION_OPERATION" not in missing
    assert "CLEARING_EVIDENCE" not in missing

    result = run_ml_prediction(payload)
    assert result["record_kind"] == "reclose_capture"
    assert result["reclose_outcome"] == "successful"
    assert result["cause_ranking"] == []


# --- the three files as one incident ------------------------------------------

@pytest.mark.parametrize("with_status", [True, False], ids=["breaker-status", "waveform-only"])
def test_fault_reclose_refault_reconstructs_as_one_sequence(service, with_status):
    incident, recon, (fault_id, reclose_id, refault_id) = _reconstruct(service, bringin_like_sequence(with_status))
    relationships = service.get_relationships(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)

    assert [r.relationship_type for r in relationships] == ["RECLOSE_SEQUENCE", "REFAULT_AFTER_RECLOSE"]
    assert relationships[0].metrics["dead_time_s"] == pytest.approx(4.925, abs=0.03)
    assert relationships[1].metrics["seconds_after_reclose"] == pytest.approx(5.685, abs=0.03)
    assert any(e["type"] == "SAME_FAULTED_PHASES_AS_RECLOSED_FAULT" for e in relationships[1].evidence_for)

    assert [ep.member_record_ids for ep in episodes] == [[fault_id, reclose_id], [refault_id]]
    first, second = episodes
    assert first.reclose_outcome == "successful"
    assert first.observed_facts["reclose_dead_time_s"] == pytest.approx(4.925, abs=0.03)
    assert first.observed_facts["refault_after_reclose_s"] == pytest.approx(5.685, abs=0.03)
    assert second.relationship_to_previous == "REFAULT_AFTER_RECLOSE"
    assert second.reclose_outcome is None

    roles = {e["incident_record_id"]: e for e in recon.physical_cause_evidence["records"]}
    assert roles[reclose_id]["evidence_role"] == "aftermath"
    assert roles[reclose_id]["skip_reason"] == "reclose_capture"
    assert roles[fault_id]["evidence_role"] == roles[refault_id]["evidence_role"] == "inception"

    assert recon.protection_sequence_interpretation["event_class"] == "RECLOSE_THEN_REFAULT"
    assert "REFAULT_AFTER_SUCCESSFUL_RECLOSE" in [h["hypothesis"] for h in recon.incident_hypotheses]
    assert "faulted again" in recon.narrative


def test_refault_contradicts_a_transient_cause_reading(service):
    _incident, recon, (fault_id, _reclose_id, _refault_id) = _reconstruct(service, bringin_like_sequence())
    entry = next(e for e in recon.physical_cause_evidence["records"] if e["incident_record_id"] == fault_id)
    if entry.get("fault_type") != "transient":
        pytest.skip("classifier did not read the synthetic fault as transient")
    cap = next(c for c in entry["applied_caps"] if c["name"].startswith("reclose_outcome"))
    assert cap["name"] == "reclose_outcome_conflict"
    assert cap["after"] < cap["before"]


def test_failed_reclose_capture_makes_the_episode_outcome_failed(service):
    payloads = [
        fault_and_trip("2026-02-01T10:00:00.000"),
        reclose_capture("2026-02-01T10:00:05.000", onto_fault=True),
    ]
    incident, recon, _ids = _reconstruct(service, payloads)
    episodes = service.get_episodes(incident.incident_id)
    assert [r.relationship_type for r in service.get_relationships(incident.incident_id)] == ["RECLOSE_SEQUENCE"]
    assert len(episodes) == 1 and episodes[0].reclose_outcome == "failed"
    assert recon.protection_sequence_interpretation["event_class"] == "SINGLE_FAULT_FAILED_RECLOSE"


def test_fault_long_after_the_reclose_is_not_a_refault(service):
    payloads = bringin_like_sequence(refault_start="2026-02-01T10:02:00.000")  # ~115 s after the reclose
    incident, _recon, _ids = _reconstruct(service, payloads)
    types = [r.relationship_type for r in service.get_relationships(incident.incident_id)]
    assert types[0] == "RECLOSE_SEQUENCE"
    assert types[1] != "REFAULT_AFTER_RECLOSE"


# --- relationship rules on hand-built snapshots -------------------------------

def _record(rid, start_iso, *, inception_ms=None, clearing_ms=None, method="status_channel",
            phases=(), reclose_events=(), event_class="FAULT_EVENT"):
    window = {"record_start_ms": 0.0, "trigger_time_ms": 100.0, "inception_time_ms": inception_ms,
              "clearing_time_ms": clearing_ms, "method": method, "faulted_phases": list(phases)}
    return IncidentRecord(
        incident_record_id=rid, incident_id="inc", analysis_id=f"missing-{rid}",
        record_start_iso=start_iso, trigger_time_iso=start_iso, trigger_offset_s=0.1,
        canonical_snapshot={
            "event_window": window,
            "observed_facts": {"faulted_phases": list(phases), "reclose_events": list(reclose_events)},
            "protection_interpretation": {"event_class": event_class},
        },
    )


def _classify(left, right, prior_fault=None):
    return classify_pair(left, right, AlignmentAssessment(), lambda: "rel", "inc", prior_fault=prior_fault)


def test_unverified_waveform_reclose_is_not_used_to_link_a_refault():
    left = _record("a", "2026-02-01T10:00:00", inception_ms=100.0, clearing_ms=115.0, phases=["B", "C"],
                   reclose_events=[{"time": 0.17, "success": True, "cb_open_verified": False, "confidence": 0.35}])
    right = _record("b", "2026-02-01T10:00:05", inception_ms=120.0, phases=["B", "C"])
    assert _classify(left, right).relationship_type != "REFAULT_AFTER_RECLOSE"


def test_fault_at_the_reclose_instant_is_one_fault_with_a_failed_reclose():
    left = _record("a", "2026-02-01T10:00:00", method="dead_time_recording", inception_ms=0.0,
                   reclose_events=[{"time": 0.9, "success": True}], event_class="RECLOSE_CAPTURE")
    right = _record("b", "2026-02-01T10:00:00.95", inception_ms=50.0, phases=["B", "C"])  # 0.1 s after the close
    rel = _classify(left, right)
    assert rel.relationship_type == "RECLOSE_SEQUENCE"
    assert rel.metrics["reclose_outcome_correction"] == "failed"


def test_reclose_record_without_absolute_time_is_linked_by_order():
    left = _record("a", None, inception_ms=100.0, clearing_ms=200.0, phases=["B", "C"])
    right = _record("b", None, method="dead_time_recording", inception_ms=0.0,
                    reclose_events=[{"time": 0.135, "success": True}], event_class="RECLOSE_CAPTURE")
    rel = _classify(left, right)
    assert rel.relationship_type == "RECLOSE_SEQUENCE"
    assert rel.confidence == 0.5


def test_refault_phase_comparison_uses_the_reclosed_fault_not_the_reclose_record():
    fault = _record("a", "2026-02-01T10:00:00", inception_ms=100.0, clearing_ms=200.0, phases=["B", "C"])
    reclose = _record("b", "2026-02-01T10:00:05", method="dead_time_recording", inception_ms=0.0,
                      reclose_events=[{"time": 0.135, "success": True}], event_class="RECLOSE_CAPTURE")
    refault = _record("c", "2026-02-01T10:00:10.7", inception_ms=120.0, phases=["A", "B", "C"])
    rel = _classify(reclose, refault, prior_fault=fault)
    assert rel.relationship_type == "REFAULT_AFTER_RECLOSE"
    assert {"type": "FAULT_PHASE_PROGRESSED", "from": ["B", "C"], "to": ["A", "B", "C"]} in rel.evidence_for


def test_energization_index_is_where_the_line_comes_alive():
    payload = reclose_capture("2026-02-01T10:00:05.000", close_at=0.3, with_status=False)
    idx = _energization_after_dead_start(_ShimRecord(payload))
    assert idx is not None
    assert T[idx] == pytest.approx(0.3, abs=0.5 / FREQ)
