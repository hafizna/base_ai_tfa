"""Incident records kept with their original files, folder uploads, and the
fault section estimate run on an incident (service and HTTP)."""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.comtrade_writer import synthetic_cfg_dat_bytes


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    """Incident store and analysis cache in scratch directories."""
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from webapp.api import storage as analysis_storage
    monkeypatch.setattr(analysis_storage, "ANALYSIS_DIR", tmp_path / "analysis")
    (tmp_path / "analysis").mkdir()
    from webapp.api.incidents import storage as incident_storage
    importlib.reload(incident_storage)
    from webapp.api.incidents import service
    importlib.reload(service)
    yield service


def _file(name: str, data: bytes):
    from webapp.api.incidents.batch_upload import UploadedFile
    return UploadedFile(filename=name, content_type=None, data=data)


# --- original files outlive the analysis cache ---------------------------------------

def test_an_expired_record_is_rebuilt_from_its_original_files(stores):
    from webapp.api import storage as analysis_storage
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="persist")
    cfg, dat = synthetic_cfg_dat_bytes()
    result = run_batch_upload(incident.incident_id, [_file("REL A/DR/rec.cfg", cfg), _file("REL A/DR/rec.dat", dat)])
    analysis_id = result.records_created[0].analysis_id

    analysis_storage._analysis_path(analysis_id).unlink()          # cache gone (expiry, redeploy)
    payload = analysis_storage.load_analysis(analysis_id)

    assert payload is not None and payload["total_samples"] > 0
    assert analysis_storage._analysis_path(analysis_id).exists()   # put back under the same id


def test_the_file_cache_expires_after_its_ttl(stores, monkeypatch):
    from webapp.api import storage as analysis_storage

    analysis_id = analysis_storage.save_analysis({"station_name": "X", "total_samples": 1})
    path = analysis_storage._analysis_path(analysis_id)
    old = time.time() - (analysis_storage.SESSION_TTL_HOURS * 3600 + 60)
    os.utime(path, (old, old))

    assert analysis_storage.load_analysis(analysis_id) is None
    assert not path.exists()


def test_folder_upload_pairs_same_named_records_per_folder_and_keeps_event_lists(stores):
    from webapp.api.incidents import raw_store
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="folders")
    cfg, dat = synthetic_cfg_dat_bytes()
    cfg2, dat2 = synthetic_cfg_dat_bytes(rec_dev_id="OTHER")
    files = [_file("CBF B/DR/Monday 11.34.43.000.CFG", cfg), _file("CBF B/DR/Monday 11.34.43.000.DAT", dat),
             _file("CCP A/DR/Monday 11.34.43.000.CFG", cfg2), _file("CCP A/DR/Monday 11.34.43.000.DAT", dat2),
             _file("BUSPRO B/2025-05-26 14.40.58.evt", b"APP: Courier\r\nTYPE: EventList\r\n")]
    result = run_batch_upload(incident.incident_id, files)

    assert [r.status for r in result.records_created] == ["created", "created"]
    assert {r.source_files[0] for r in result.records_created} == {"CBF B/DR/Monday 11.34.43.000.CFG",
                                                                 "CCP A/DR/Monday 11.34.43.000.CFG"}
    assert result.event_files == ["BUSPRO B/2025-05-26 14.40.58.evt"]
    assert [rel for rel, _ in raw_store.event_files(incident.incident_id)] == ["BUSPRO B/2025-05-26 14.40.58.evt"]
    assert all(raw_store.has_record_files(r.analysis_id) for r in result.records_created)


def test_original_files_are_kept_under_short_names(stores):
    """A DFR names its files after station, bay and maker; with the folder
    path that can run past a path-length limit, so the stored copy is short."""
    from webapp.api.incidents import raw_store
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="long names")
    cfg, dat = synthetic_cfg_dat_bytes()
    long = "Eksternal sisi jepara/240226,010033697,0,GI JEPARA,TANJUNG JATI 1-2,Qualitrol LLC"
    created = run_batch_upload(incident.incident_id, [_file(f"{long}.cfg", cfg), _file(f"{long}.dat", dat)])
    analysis_id = created.records_created[0].analysis_id

    assert sorted(p.name for p in raw_store._record_dir(analysis_id).iterdir()) == ["manifest.json", "record.cfg", "record.dat"]
    assert raw_store.uploaded_paths(analysis_id) == [f"{long}.cfg", f"{long}.dat"]
    assert raw_store.parse_record(analysis_id).total_samples > 0


def test_detaching_the_last_holder_drops_the_original_files(stores):
    from webapp.api.incidents import raw_store
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="detach")
    cfg, dat = synthetic_cfg_dat_bytes()
    created = run_batch_upload(incident.incident_id, [_file("a.cfg", cfg), _file("a.dat", dat)]).records_created[0]

    stores.detach_record(incident.incident_id, created.incident_record_id)

    assert not raw_store.has_record_files(created.analysis_id)


# --- station names --------------------------------------------------------------------

def test_a_relay_name_in_the_station_field_does_not_block_attaching(stores):
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="names", station_name="GITET Pemalang")
    cfg, dat = synthetic_cfg_dat_bytes(station_name="NR")
    result = run_batch_upload(incident.incident_id, [_file("x.cfg", cfg), _file("x.dat", dat)])

    assert result.records_created and result.records_created[0].status == "created"
    record = stores.list_records(incident.incident_id)[0]
    assert any(w["type"] == "STATION_NOT_NAMED" for w in record.attachment_warnings)


def test_one_substation_written_two_ways_is_not_a_mismatch(stores):
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="names", station_name="Mojosongo")
    cfg, dat = synthetic_cfg_dat_bytes(station_name="GI MOJOSONGO")
    result = run_batch_upload(incident.incident_id, [_file("x.cfg", cfg), _file("x.dat", dat)])

    assert result.records_created and result.records_created[0].status == "created"


def test_fault_section_without_records_is_a_clear_422(stores):
    incident = stores.create_incident(title="empty")
    with pytest.raises(stores.IncidentServiceError) as exc_info:
        stores.run_fault_section(incident.incident_id)
    assert exc_info.value.status_code == 422


def test_line_records_from_two_substations_say_why_there_is_no_bay_location(stores):
    from webapp.api.incidents.batch_upload import run_batch_upload

    incident = stores.create_incident(title="line")
    for station in ("GI JEPARA", "GI TANJUNGJATI"):
        cfg, dat = synthetic_cfg_dat_bytes(station_name=station, rec_dev_id=f"DFR {station}")
        run_batch_upload(incident.incident_id, [_file(f"{station}/x.cfg", cfg), _file(f"{station}/x.dat", dat)],
                         override_warnings=True)
    with pytest.raises(stores.IncidentServiceError) as exc_info:
        stores.run_fault_section(incident.incident_id)
    assert exc_info.value.status_code == 422
    message = str(exc_info.value)
    assert "rekaman line dari GI JEPARA dan GI TANJUNGJATI" in message and "Double Ended FL" in message


def test_the_bay_type_can_go_back_to_automatic(stores):
    incident = stores.create_incident(title="trafo", asset_type="TRANSFORMER")

    assert stores.update_incident(incident.incident_id, {"asset_type": None}).asset_type is None
    with pytest.raises(stores.IncidentServiceError):
        stores.update_incident(incident.incident_id, {"asset_type": "KAPAL"})


# --- the real incidents through the HTTP API, when their records are available -------

@pytest.fixture()
def client(stores):
    from webapp.api.routers import incidents as router
    importlib.reload(router)
    from webapp.api import main
    importlib.reload(main)
    with TestClient(main.app) as test_client:
        yield test_client


def _upload_folder(client, incident_id: str, root: Path, rels: list[str]) -> dict:
    files = [("files", (rel, (root / rel).read_bytes(), "application/octet-stream")) for rel in rels]
    resp = client.post(f"/api/incidents/{incident_id}/upload-records", files=files,
                       params={"override_warnings": "true"})
    assert resp.status_code == 200, resp.text
    return resp.json()


_TBROK = os.environ.get("FAULT_SECTION_TAMBAKLOROK_ROOT")
_PMLNG = os.environ.get("FAULT_SECTION_PEMALANG_ROOT")


@pytest.mark.skipif(not _TBROK or not Path(_TBROK).is_dir(), reason="Tambaklorok records not available")
def test_tambaklorok_incident_through_the_api(client):
    root = Path(_TBROK)
    incident_id = client.post("/api/incidents", json={
        "title": "Trafo #2 trip", "station_name": "GIS Tambak Lorok 3", "asset_type": "TRANSFORMER"}).json()["incident_id"]
    # The three relay folders (the incident folder's top level holds loose copies of two of them).
    rels = [p.relative_to(root).as_posix() for p in sorted(root.rglob("*"))
            if p.suffix.lower() in (".cfg", ".dat") and len(p.relative_to(root).parts) > 1]
    uploaded = _upload_folder(client, incident_id, root, rels)

    resp = client.post(f"/api/incidents/{incident_id}/fault-section")
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["template"] == "transformer"
    assert result["analysis_ids"] == sorted(r["analysis_id"] for r in uploaded["records_created"])
    (episode,) = result["episodes"]
    assert {b["section"] for b in episode["best"]} == {"lv_terminal", "stub_cb_lv"}
    assert client.get(f"/api/incidents/{incident_id}/fault-section").json()["computed_at"] == result["computed_at"]


@pytest.mark.skipif(not _PMLNG or not Path(_PMLNG).is_dir(), reason="Pemalang records not available")
def test_pemalang_incident_through_the_api(client):
    root = Path(_PMLNG)
    incident_id = client.post("/api/incidents", json={
        "title": "TRIP 7B2 DIA #2", "station_name": "GITET Pemalang"}).json()["incident_id"]
    rels = [
        "MAIN B BAY MANDIRANCAN 1/NR MAIN B MDRCN 1.CFG", "MAIN B BAY MANDIRANCAN 1/NR MAIN B MDRCN 1.DAT",
        "LCD MAIN A BAY MANDIRANCAN 1/DR/Monday 26 May 2025 11.34.44.000.CFG",
        "LCD MAIN A BAY MANDIRANCAN 1/DR/Monday 26 May 2025 11.34.44.000.DAT",
        "CBF B DIAMETER 2_/DR/Monday 26 May 2025 11.34.43.000.CFG",
        "CBF B DIAMETER 2_/DR/Monday 26 May 2025 11.34.43.000.DAT",
        "CBF B DIAMETER 2_/Events/2025-05-26 15.03.47.evt",
        "CCP AB-B MICOM MAIN A/DR/Monday 26 May 2025 11.34.42.000.CFG",
        "CCP AB-B MICOM MAIN A/DR/Monday 26 May 2025 11.34.42.000.DAT",
        "CCP AB-B MICOM MAIN A/DR/Monday 26 May 2025 11.34.43.000.CFG",
        "CCP AB-B MICOM MAIN A/DR/Monday 26 May 2025 11.34.43.000.DAT",
        "BUSPRO B MAIN 1 FASA S/2025-05-26 14.40.58.evt",
        "TRIP MDCAN-PMLG 1 (26 MEI 2025)/DFR/MAIN A/20250526_043445_716_r137.cfg",
        "TRIP MDCAN-PMLG 1 (26 MEI 2025)/DFR/MAIN A/20250526_043445_716_r137.dat",
    ]
    uploaded = _upload_folder(client, incident_id, root, rels)
    assert len(uploaded["records_created"]) == 6 and len(uploaded["event_files"]) == 2

    result = client.post(f"/api/incidents/{incident_id}/fault-section").json()
    assert result["template"] == "diameter"
    first, second = result["episodes"]
    assert {b["section"] for b in first["best"]} == {"stub_cb_b"}
    assert "CT CB 7B2 di sisi Bus B" in first["inferred_assumptions"]
    assert {b["section"] for b in second["best"]} == {"t_b"}
