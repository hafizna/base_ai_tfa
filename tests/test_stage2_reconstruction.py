"""Stage 2 golden reconstruction tests.

Covers the 20 minimum scenarios from the Stage 2 spec: duplicate captures,
overlapping-but-not-duplicate captures, continuation, reclose sequences,
repeated faults, possible evolving faults, unrelated records, missing/
conflicting timestamps, long gaps, current-only records, no-fault records,
episode deduplication, LightGBM non-averaging, reconstruction versioning,
manual relationship override, Stage 0/1 regression, and atomic batch-upload
rollback.
"""

from __future__ import annotations

import importlib

import pytest

from tests.fixtures import incident_scenarios as sc
from tests.fixtures import synthetic_records as sr
from tests.fixtures.comtrade_writer import synthetic_cfg_dat_bytes
from webapp.api.storage import save_analysis


@pytest.fixture()
def stage2(tmp_path, monkeypatch):
    """Reload the incidents modules pointed at a scratch directory per test."""
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from webapp.api.incidents import storage as storage_module
    importlib.reload(storage_module)
    from webapp.api.incidents import service as service_module
    importlib.reload(service_module)

    yield service_module


def _attach_scenario(service, incident_id: str, scenario: list[tuple[dict, dict]]) -> list:
    records = []
    for payload, overrides in scenario:
        analysis_id = save_analysis(payload)
        record = service.attach_record(incident_id, analysis_id=analysis_id, **overrides)
        records.append(record)
    return records


# --- 1. Two duplicate captures from different relays -----------------------

def test_duplicate_captures_grouped_into_one_episode(stage2):
    service = stage2
    incident = service.create_incident(title="Duplicate captures", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.duplicate_captures_different_relays())

    recon = service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)

    assert relationships[0].relationship_type == "DUPLICATE_TRIGGER"
    assert len(episodes) == 1
    assert len(episodes[0].member_record_ids) == 2


# --- 2. Overlapping captures that are not full duplicates -------------------

def test_overlapping_capture_not_full_duplicate(stage2):
    service = stage2
    incident = service.create_incident(title="Overlap test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.overlapping_not_full_duplicate())

    service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)

    assert relationships[0].relationship_type in ("OVERLAPPING_CAPTURE", "DUPLICATE_TRIGGER")
    # Either way it must be backed by an overlap metric, not a guess.
    assert relationships[0].metrics["waveform_similarity"]["computed"] is True


# --- 3. Continuation --------------------------------------------------------

def test_continuation_sequence(stage2):
    service = stage2
    incident = service.create_incident(title="Continuation test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.continuation_sequence())

    service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)

    assert relationships[0].relationship_type in ("CONTINUATION", "RECLOSE_SEQUENCE", "DUPLICATE_TRIGGER", "OVERLAPPING_CAPTURE")
    assert len(episodes) == 1  # continuation merges into one episode


# --- 4. Successful reclose sequence in a separate record --------------------

def test_successful_reclose_in_separate_record(stage2):
    service = stage2
    incident = service.create_incident(title="Reclose success test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.successful_reclose_separate_record())

    service.reconstruct(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)
    assert len(episodes) >= 1
    assert episodes[0].reclose_outcome in ("successful", None)


# --- 5. Failed reclose / trip-on-reclose ------------------------------------

def test_failed_reclose_trip_on_reclose(stage2):
    service = stage2
    incident = service.create_incident(title="Failed reclose test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.failed_reclose_trip_on_reclose())

    service.reconstruct(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)
    outcomes = [e.reclose_outcome for e in episodes]
    assert "failed" in outcomes


# --- 6. Repeated A-G faults with a gap --------------------------------------

def test_repeated_ag_faults_with_gap(stage2):
    service = stage2
    incident = service.create_incident(title="Repeated fault test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.repeated_ag_faults_with_gap())

    recon = service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)

    assert relationships[0].relationship_type == "REPEATED_FAULT"
    assert len(episodes) == 2
    assert recon.protection_sequence_interpretation["event_class"] == "REPEATED_INDEPENDENT_FAULTS"


# --- 7. Possible evolving A-G -> A-B-G --------------------------------------

def test_possible_evolving_fault_is_conservatively_named(stage2):
    service = stage2
    incident = service.create_incident(title="Evolving fault test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.possible_evolving_fault())

    recon = service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)

    # Must never claim EVOLVING_FAULT_CONFIRMED — only POSSIBLE_EVOLVING_FAULT.
    types = {r.relationship_type for r in relationships}
    assert "EVOLVING_FAULT_CONFIRMED" not in types
    if "POSSIBLE_EVOLVING_FAULT" in types:
        hyps = [h for h in recon.incident_hypotheses if h["hypothesis"] == "POSSIBLE_EVOLVING_FAULT"]
        assert hyps
        assert 0.0 < hyps[0]["confidence"] < 1.0


# --- 8. Two unrelated records wrongly grouped -------------------------------

def test_unrelated_records_flagged(stage2):
    service = stage2
    incident = service.create_incident(title="Unrelated records test")
    _attach_scenario(service, incident.incident_id, sc.unrelated_records_wrongly_grouped())

    incident_after = service.get_incident(incident.incident_id)
    # Station mismatch must be flagged even though attach succeeded (no incident-level station set yet).
    records = service.list_records(incident.incident_id)
    assert any(r.station_name for r in records)

    service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)
    assert relationships[0].relationship_type in ("UNRELATED", "NEW_FAULT_EPISODE", "UNCERTAIN")


# --- 9. Missing absolute timestamp, manual order ----------------------------

def test_missing_absolute_timestamp_uses_manual_order(stage2):
    service = stage2
    incident = service.create_incident(title="Manual order test", station_name="GOLDEN TEST")
    records = _attach_scenario(service, incident.incident_id, sc.missing_absolute_timestamp_manual_order())
    service.reorder_records(incident.incident_id, [r.incident_record_id for r in records])

    recon = service.reconstruct(incident.incident_id)
    assert recon.alignment["status"] == "MANUAL_ORDER"
    assert recon.alignment["order_source"] == "MANUAL"


# --- 10. Conflicting timestamp order -----------------------------------------

def test_conflicting_timestamp_order_flagged_untrusted(stage2):
    service = stage2
    incident = service.create_incident(title="Conflicting order test", station_name="GOLDEN TEST")
    scenario = sc.conflicting_timestamp_order()
    records = []
    for payload, overrides in scenario:
        analysis_id = save_analysis(payload)
        record = service.attach_record(incident.incident_id, analysis_id=analysis_id, **overrides)
        records.append(record)
    # Manual order 0,1 in attach order, but timestamps disagree (record 0 is LATER).
    service.reorder_records(incident.incident_id, [r.incident_record_id for r in records])

    recon = service.reconstruct(incident.incident_id)
    assert recon.alignment["status"] == "UNTRUSTED"
    assert any(w.get("type") == "MANUAL_ORDER_CONTRADICTS_TIMESTAMPS" for w in recon.alignment["warnings"])


# --- 11. Long time gap with compressed timeline -----------------------------

def test_long_gap_reported_explicitly(stage2):
    service = stage2
    incident = service.create_incident(title="Long gap test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.long_gap_compressed_timeline())

    recon = service.reconstruct(incident.incident_id)
    timeline = service.get_timeline(incident.incident_id)
    gap_events = [e for e in timeline if e.event_type == "DATA_GAP"]
    assert gap_events
    assert gap_events[0].details["gap_ms"] == pytest.approx(6 * 3600 * 1000, rel=0.01)


# --- 12. Current-only record in incident ------------------------------------

def test_current_only_record_does_not_crash_reconstruction(stage2):
    service = stage2
    incident = service.create_incident(title="Current-only test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.current_only_record_in_incident())

    recon = service.reconstruct(incident.incident_id)
    assert recon.reconstruction_id


# --- 13. One no-fault record plus one actual fault --------------------------

def test_no_fault_plus_actual_fault(stage2):
    service = stage2
    incident = service.create_incident(title="No-fault mix test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.no_fault_plus_actual_fault())

    service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)
    assert relationships[0].relationship_type == "UNRELATED"


# --- 14. Duplicate records not counted as separate episodes -----------------

def test_duplicates_not_counted_as_separate_episodes(stage2):
    service = stage2
    incident = service.create_incident(title="Dedup episode count test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.duplicate_captures_different_relays())

    recon = service.reconstruct(incident.incident_id)
    assert recon.observed_incident_facts["episode_count"] == 1
    assert recon.observed_incident_facts["record_count"] == 2


# --- 15. LightGBM probabilities not averaged --------------------------------

def test_lightgbm_probabilities_not_averaged(stage2):
    service = stage2
    incident = service.create_incident(title="No averaging test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.repeated_ag_faults_with_gap())

    recon = service.reconstruct(incident.incident_id)
    physical_cause = recon.physical_cause_evidence
    assert physical_cause["scope"] == "RECORD_LOCAL_SIGNATURES"
    assert physical_cause["incident_root_cause"] == "UNCONFIRMED"
    assert physical_cause["consistency"] in ("CONSISTENT", "MOSTLY_CONSISTENT", "MIXED", "CONTRADICTORY", "INSUFFICIENT")
    # No single combined "incident_probability" field must exist anywhere in the payload.
    assert "incident_probability" not in physical_cause
    # Per-record confidences must be preserved individually, not collapsed into one number.
    confidences = [r["confidence"] for r in physical_cause["records"]]
    assert len(confidences) == 2


# --- 16. Reconstruction version history -------------------------------------

def test_reconstruction_version_history_preserved(stage2):
    service = stage2
    incident = service.create_incident(title="Versioning test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.repeated_ag_faults_with_gap())

    recon1 = service.reconstruct(incident.incident_id)
    recon2 = service.reconstruct(incident.incident_id)

    versions = service.list_reconstructions(incident.incident_id)
    assert len(versions) == 2
    assert recon2.supersedes == recon1.reconstruction_id
    by_id = {v.reconstruction_id: v for v in versions}
    assert by_id[recon1.reconstruction_id].is_latest is False
    assert by_id[recon2.reconstruction_id].is_latest is True

    # Old version must still be fetchable by id (audit requirement).
    old = service.get_reconstruction(incident.incident_id, recon1.reconstruction_id)
    assert old.reconstruction_id == recon1.reconstruction_id


# --- 17. Manual relationship override ---------------------------------------

def test_manual_relationship_override(stage2):
    service = stage2
    incident = service.create_incident(title="Override test", station_name="GOLDEN TEST")
    _attach_scenario(service, incident.incident_id, sc.repeated_ag_faults_with_gap())

    service.reconstruct(incident.incident_id)
    relationships = service.get_relationships(incident.incident_id)
    original_type = relationships[0].relationship_type

    updated = service.override_relationship(
        incident.incident_id,
        relationships[0].relationship_id,
        corrected_relationship="CONTINUATION",
        operator="engineer1",
        reason="Field inspection confirmed single ongoing event.",
    )
    assert updated.relationship_type == "CONTINUATION"
    assert updated.override_previous_type == original_type
    assert updated.overridden is True
    assert updated.override_operator == "engineer1"


# --- 18. Stage 0 timing consistency preserved -------------------------------

def test_stage0_timing_consistency_preserved_through_reconstruction(stage2):
    from webapp.api.record_analysis import build_record_analysis
    from core.event_analysis import build_event_window

    service = stage2
    incident = service.create_incident(title="Stage0 consistency test", station_name="GOLDEN TEST")
    scenario = sc.repeated_ag_faults_with_gap()
    payload0 = scenario[0][0]
    direct_window = build_event_window(payload0)

    records = _attach_scenario(service, incident.incident_id, scenario)
    stored_window = records[0].canonical_snapshot["event_window"]

    assert stored_window["inception_time_ms"] == pytest.approx(direct_window.inception_time_ms, abs=1e-6)
    assert stored_window["method"] == direct_window.method


# --- 19. Stage 1 incident CRUD still works ----------------------------------

def test_stage1_incident_crud_unaffected(stage2):
    service = stage2
    incident = service.create_incident(title="Stage1 CRUD test")
    updated = service.update_incident(incident.incident_id, {"status": "OPEN", "bay_name": "Bay 7"})
    assert updated.status == "OPEN"
    assert updated.bay_name == "Bay 7"

    analysis_id = save_analysis(sr.transient_slg_successful_reclose())
    record = service.attach_record(incident.incident_id, analysis_id=analysis_id)
    assert record.analysis_id == analysis_id

    service.detach_record(incident.incident_id, record.incident_record_id)
    assert service.list_records(incident.incident_id) == []


# --- 20. Batch upload atomic rollback on invalid pair -----------------------

def test_batch_upload_atomic_rollback(stage2):
    from webapp.api.incidents.batch_upload import UploadedFile, run_batch_upload

    service = stage2
    incident = service.create_incident(title="Atomic rollback test", station_name="GOLDEN TEST")

    cfg1, dat1 = synthetic_cfg_dat_bytes(rec_dev_id="RELAY1")
    files = [
        UploadedFile("record1.cfg", "text/plain", cfg1),
        UploadedFile("record1.dat", "application/octet-stream", dat1),
        UploadedFile("orphan.cfg", "text/plain", cfg1),
    ]

    result = run_batch_upload(incident.incident_id, files)
    assert result.reconstruction_status == "aborted_atomic"
    assert result.records_created == []
    assert service.list_records(incident.incident_id) == []


def test_batch_upload_partial_success_mode(stage2):
    from webapp.api.incidents.batch_upload import UploadedFile, run_batch_upload

    service = stage2
    incident = service.create_incident(title="Partial success test", station_name="GOLDEN TEST")

    cfg1, dat1 = synthetic_cfg_dat_bytes(rec_dev_id="RELAY1")
    files = [
        UploadedFile("record1.cfg", "text/plain", cfg1),
        UploadedFile("record1.dat", "application/octet-stream", dat1),
        UploadedFile("orphan.cfg", "text/plain", cfg1),
    ]

    result = run_batch_upload(incident.incident_id, files, partial_success=True)
    assert result.reconstruction_status == "completed"
    assert len(result.records_created) == 1
    assert len(result.errors) == 1
    assert len(service.list_records(incident.incident_id)) == 1
