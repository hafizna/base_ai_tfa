"""Stage 1 incident domain/service/storage tests.

Uses the Stage 0 synthetic COMTRADE payload builders (tests/fixtures) saved
through the real analysis-session storage, so incident attachment exercises
the actual ``load_analysis`` -> ``build_record_analysis`` path rather than a
mock.
"""

from __future__ import annotations

import importlib

import pytest

from tests.fixtures import synthetic_records as sr
from webapp.api.storage import save_analysis


@pytest.fixture()
def incident_storage(tmp_path, monkeypatch):
    """Reload the incidents.storage module pointed at a scratch directory so
    tests don't share state with each other or a real dev environment."""
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from webapp.api.incidents import storage as incident_storage_module

    importlib.reload(incident_storage_module)

    from webapp.api.incidents import service as incident_service_module

    importlib.reload(incident_service_module)

    yield incident_service_module


@pytest.fixture()
def service(incident_storage):
    return incident_storage


def _upload(payload: dict) -> str:
    return save_analysis(payload)


# --- 1. Create incident kosong -------------------------------------------------

def test_create_empty_incident(service):
    incident = service.create_incident(title="Test incident")
    assert incident.incident_id
    assert incident.status == "DRAFT"
    assert incident.record_ids == []
    assert incident.observed_summary["record_count"] == 0

    fetched = service.get_incident(incident.incident_id)
    assert fetched.title == "Test incident"


# --- 2. Attach satu canonical record -------------------------------------------

def test_attach_single_record(service):
    incident = service.create_incident(title="Single record incident")
    analysis_id = _upload(sr.transient_slg_successful_reclose())

    record = service.attach_record(incident.incident_id, analysis_id=analysis_id, attachment_role="PRIMARY")
    assert record.analysis_id == analysis_id
    assert record.canonical_snapshot["record_id"] == analysis_id

    updated = service.get_incident(incident.incident_id)
    assert updated.record_ids == [analysis_id]
    assert updated.observed_summary["record_count"] == 1
    assert "Single record attached" in updated.incident_interpretation["summary"]


# --- 3. Attach beberapa records -------------------------------------------------

def test_attach_multiple_records(service):
    incident = service.create_incident(title="Multi record incident")
    a1 = _upload(sr.transient_slg_successful_reclose())
    a2 = _upload(sr.permanent_fault_failed_reclose())

    service.attach_record(incident.incident_id, analysis_id=a1, attachment_role="PRIMARY")
    service.attach_record(incident.incident_id, analysis_id=a2, attachment_role="SUPPORTING")

    updated = service.get_incident(incident.incident_id)
    assert set(updated.record_ids) == {a1, a2}
    assert updated.observed_summary["record_count"] == 2
    assert "Multiple records are attached" in updated.incident_interpretation["summary"]
    assert "Automated inter-record reconstruction has not yet been performed" in updated.incident_interpretation["summary"]


# --- 4. Mencegah duplicate attachment -------------------------------------------

def test_prevent_duplicate_attachment(service):
    incident = service.create_incident(title="Dup test")
    analysis_id = _upload(sr.transient_slg_successful_reclose())
    service.attach_record(incident.incident_id, analysis_id=analysis_id)

    with pytest.raises(service.IncidentServiceError) as exc_info:
        service.attach_record(incident.incident_id, analysis_id=analysis_id)
    assert exc_info.value.status_code == 409


# --- 5. Record dengan timestamp lama/backward-compatible payload ---------------

def test_attach_legacy_payload_without_absolute_time_fields(service):
    payload = sr.transient_slg_successful_reclose()
    for key in ("start_time_iso", "trigger_time_iso", "trigger_offset_s", "time_code", "local_code", "clock_quality"):
        payload.pop(key, None)
    analysis_id = _upload(payload)

    incident = service.create_incident(title="Legacy payload test")
    record = service.attach_record(incident.incident_id, analysis_id=analysis_id)
    assert record.canonical_snapshot["event_window"]["inception_time_ms"] is not None


# --- 6. Record tanpa absolute time ----------------------------------------------

def test_attach_record_without_absolute_time_flags_missing_evidence(service):
    incident = service.create_incident(title="No absolute time test")
    analysis_id = _upload(sr.transient_slg_successful_reclose())  # synthetic: no wall-clock time
    record = service.attach_record(incident.incident_id, analysis_id=analysis_id)

    assert any(w["type"] == "NO_ABSOLUTE_TIME" for w in record.attachment_warnings)
    updated = service.get_incident(incident.incident_id)
    assert updated.observed_summary["records_without_absolute_time"] == 1


# --- 7. Station mismatch warning ------------------------------------------------

def test_station_mismatch_requires_override(service):
    incident = service.create_incident(title="Station mismatch test", station_name="GI ALPHA")
    payload = sr.transient_slg_successful_reclose()
    payload["station_name"] = "GI BETA"
    analysis_id = _upload(payload)

    with pytest.raises(service.IncidentServiceError) as exc_info:
        service.attach_record(incident.incident_id, analysis_id=analysis_id)
    assert exc_info.value.status_code == 409

    record = service.attach_record(incident.incident_id, analysis_id=analysis_id, override_warnings=True)
    assert any(w["type"] == "STATION_MISMATCH" for w in record.attachment_warnings)


# --- 8. Manual ordering ----------------------------------------------------------

def test_manual_ordering(service):
    incident = service.create_incident(title="Ordering test")
    a1 = _upload(sr.transient_slg_successful_reclose())
    a2 = _upload(sr.permanent_fault_failed_reclose())
    r1 = service.attach_record(incident.incident_id, analysis_id=a1)
    r2 = service.attach_record(incident.incident_id, analysis_id=a2)

    reordered = service.reorder_records(incident.incident_id, [r2.incident_record_id, r1.incident_record_id])
    assert reordered[0].incident_record_id == r2.incident_record_id
    assert reordered[0].manual_order == 0
    assert reordered[0].order_source == "MANUAL"

    listed = service.list_records(incident.incident_id)
    assert listed[0].incident_record_id == r2.incident_record_id


def test_reorder_rejects_unknown_or_incomplete_ids(service):
    incident = service.create_incident(title="Ordering validation test")
    a1 = _upload(sr.transient_slg_successful_reclose())
    r1 = service.attach_record(incident.incident_id, analysis_id=a1)

    with pytest.raises(service.IncidentServiceError):
        service.reorder_records(incident.incident_id, ["not-a-real-id"])

    with pytest.raises(service.IncidentServiceError):
        service.reorder_records(incident.incident_id, [])
    # sanity: correct single-id reorder still works
    service.reorder_records(incident.incident_id, [r1.incident_record_id])


# --- 9. Incident start/end dari record timestamps -------------------------------

def test_incident_start_end_derived_from_record_timestamps(service):
    payload = sr.transient_slg_successful_reclose()
    payload["start_time_iso"] = "2026-01-01T00:00:00+00:00"
    payload["trigger_time_iso"] = "2026-01-01T00:00:30+00:00"
    analysis_id = _upload(payload)

    incident = service.create_incident(title="Timestamp derivation test")
    service.attach_record(incident.incident_id, analysis_id=analysis_id)

    updated = service.get_incident(incident.incident_id)
    assert updated.incident_start_iso == "2026-01-01T00:00:30+00:00"
    assert updated.incident_end_iso == "2026-01-01T00:00:30+00:00"


# --- 10. Detach record -----------------------------------------------------------

def test_detach_record(service):
    incident = service.create_incident(title="Detach test")
    analysis_id = _upload(sr.transient_slg_successful_reclose())
    record = service.attach_record(incident.incident_id, analysis_id=analysis_id)

    service.detach_record(incident.incident_id, record.incident_record_id)

    updated = service.get_incident(incident.incident_id)
    assert updated.record_ids == []
    assert service.list_records(incident.incident_id) == []


def test_detach_unknown_record_raises_404(service):
    incident = service.create_incident(title="Detach 404 test")
    with pytest.raises(service.IncidentServiceError) as exc_info:
        service.detach_record(incident.incident_id, "does-not-exist")
    assert exc_info.value.status_code == 404


# --- 11. Evidence CRUD -----------------------------------------------------------

def test_evidence_crud(service):
    incident = service.create_incident(title="Evidence test")
    evidence = service.add_evidence(
        incident.incident_id,
        evidence_type="LIGHTNING_DETECTION",
        source="BMKG",
        description="Lightning strike detected near tower 45",
        confidence="PROBABLE",
    )
    assert evidence.evidence_id

    items = service.list_evidence(incident.incident_id)
    assert len(items) == 1

    service.remove_evidence(incident.incident_id, evidence.evidence_id)
    assert service.list_evidence(incident.incident_id) == []


def test_evidence_rejects_unknown_type(service):
    incident = service.create_incident(title="Evidence validation test")
    with pytest.raises(service.IncidentServiceError):
        service.add_evidence(incident.incident_id, evidence_type="NOT_A_REAL_TYPE")


# --- 12. Incident feedback snapshot -----------------------------------------------

def test_incident_feedback_snapshot(service):
    incident = service.create_incident(title="Feedback test")
    analysis_id = _upload(sr.transient_slg_successful_reclose())
    service.attach_record(incident.incident_id, analysis_id=analysis_id)

    feedback = service.save_feedback(incident.incident_id, {
        "operator": "test-operator",
        "record_grouping_correct": True,
        "notes": "Looks right",
    })
    assert feedback.incident_snapshot["incident_id"] == incident.incident_id
    assert len(feedback.incident_snapshot["records"]) == 1

    stored = service.list_feedback(incident.incident_id)
    assert len(stored) == 1
    assert stored[0].operator == "test-operator"


# --- 13. Existing single-record endpoints tetap berjalan (see test_canonical_timing_consistency.py) ---
# Covered by tests/test_canonical_timing_consistency.py and tests/test_event_analysis_golden.py;
# this module does not duplicate those, only verifies incidents wrap them without modification.

def test_incident_snapshot_matches_direct_canonical_call():
    from webapp.api.record_analysis import build_record_analysis
    from webapp.api.storage import load_analysis

    payload = sr.transient_slg_successful_reclose()
    analysis_id = _upload(payload)
    direct = build_record_analysis(analysis_id, load_analysis(analysis_id))
    assert direct.event_window is not None


# --- 14. Incident storage survive service reinitialization -----------------------

def test_incident_storage_survives_reinitialization(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "incidents_persist")
    monkeypatch.setenv("INCIDENTS_DATA_DIR", data_dir)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from webapp.api.incidents import storage as storage_module
    importlib.reload(storage_module)
    from webapp.api.incidents import service as service_module
    importlib.reload(service_module)

    incident = service_module.create_incident(title="Persistence test")
    incident_id = incident.incident_id

    # Simulate process restart: reload both modules again against the same dir.
    importlib.reload(storage_module)
    importlib.reload(service_module)

    fetched = service_module.get_incident(incident_id)
    assert fetched.title == "Persistence test"


# --- 15. Legacy analysis tetap dapat ditambahkan ke incident ----------------------

def test_legacy_analysis_can_be_attached(service):
    """A payload missing every Stage-0 absolute-time field (simulating an
    analysis_id saved before those fields existed) must still attach cleanly."""
    payload = sr.no_fault_trigger()
    for key in ("start_time_iso", "trigger_time_iso", "trigger_offset_s", "time_code", "local_code", "clock_quality"):
        payload.pop(key, None)
    analysis_id = _upload(payload)

    incident = service.create_incident(title="Legacy attach test")
    record = service.attach_record(incident.incident_id, analysis_id=analysis_id)
    assert record.canonical_snapshot["protection_interpretation"]["event_class"] == "NO_FAULT_TRIGGER"


# --- Attach against unknown analysis_id / incident_id -----------------------------

def test_attach_unknown_analysis_id_raises_404(service):
    incident = service.create_incident(title="Unknown analysis test")
    with pytest.raises(service.IncidentServiceError) as exc_info:
        service.attach_record(incident.incident_id, analysis_id="does-not-exist")
    assert exc_info.value.status_code == 404


def test_get_unknown_incident_raises_404(service):
    with pytest.raises(service.IncidentServiceError) as exc_info:
        service.get_incident("does-not-exist")
    assert exc_info.value.status_code == 404
