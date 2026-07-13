"""Stage 1 incident API (FastAPI router) tests.

Exercises the HTTP layer end-to-end via TestClient, using the real app so
that wiring in webapp/api/main.py (router registration) is also verified.
Also checks that adding the incidents router did not disturb existing
Stage 0 endpoints (upload/canonical).
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from tests.fixtures import synthetic_records as sr


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents_api"))
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from webapp.api.incidents import storage as incident_storage_module
    importlib.reload(incident_storage_module)
    from webapp.api.incidents import service as incident_service_module
    importlib.reload(incident_service_module)
    from webapp.api.routers import incidents as incidents_router_module
    importlib.reload(incidents_router_module)

    from webapp.api import main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as test_client:
        yield test_client


def _upload_payload(test_client: TestClient, payload: dict) -> str:
    from webapp.api.storage import save_analysis
    return save_analysis(payload)


def test_health_endpoint_unaffected(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_list_get_incident(client):
    resp = client.post("/api/incidents", json={"title": "API incident"})
    assert resp.status_code == 200
    incident_id = resp.json()["incident_id"]

    resp = client.get("/api/incidents")
    assert resp.status_code == 200
    assert any(i["incident_id"] == incident_id for i in resp.json())

    resp = client.get(f"/api/incidents/{incident_id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "API incident"
    assert resp.json()["records"] == []


def test_get_missing_incident_404(client):
    resp = client.get("/api/incidents/does-not-exist")
    assert resp.status_code == 404


def test_patch_incident(client):
    incident_id = client.post("/api/incidents", json={"title": "Patch me"}).json()["incident_id"]
    resp = client.patch(f"/api/incidents/{incident_id}", json={"status": "OPEN", "bay_name": "Bay 1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OPEN"
    assert body["bay_name"] == "Bay 1"


def test_patch_invalid_status_422_or_400(client):
    incident_id = client.post("/api/incidents", json={"title": "Bad status"}).json()["incident_id"]
    resp = client.patch(f"/api/incidents/{incident_id}", json={"status": "NOT_A_STATUS"})
    assert resp.status_code == 400


def test_attach_and_list_records(client):
    incident_id = client.post("/api/incidents", json={"title": "Attach test"}).json()["incident_id"]
    analysis_id = _upload_payload(client, sr.transient_slg_successful_reclose())

    resp = client.post(f"/api/incidents/{incident_id}/records", json={"analysis_id": analysis_id, "attachment_role": "PRIMARY"})
    assert resp.status_code == 200
    assert resp.json()["analysis_id"] == analysis_id

    resp = client.get(f"/api/incidents/{incident_id}/records")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_attach_duplicate_returns_409(client):
    incident_id = client.post("/api/incidents", json={"title": "Dup API test"}).json()["incident_id"]
    analysis_id = _upload_payload(client, sr.transient_slg_successful_reclose())
    client.post(f"/api/incidents/{incident_id}/records", json={"analysis_id": analysis_id})

    resp = client.post(f"/api/incidents/{incident_id}/records", json={"analysis_id": analysis_id})
    assert resp.status_code == 409


def test_detach_record(client):
    incident_id = client.post("/api/incidents", json={"title": "Detach API test"}).json()["incident_id"]
    analysis_id = _upload_payload(client, sr.transient_slg_successful_reclose())
    incident_record_id = client.post(
        f"/api/incidents/{incident_id}/records", json={"analysis_id": analysis_id}
    ).json()["incident_record_id"]

    resp = client.delete(f"/api/incidents/{incident_id}/records/{incident_record_id}")
    assert resp.status_code == 200

    resp = client.get(f"/api/incidents/{incident_id}/records")
    assert resp.json() == []


def test_reorder_records_endpoint(client):
    incident_id = client.post("/api/incidents", json={"title": "Reorder API test"}).json()["incident_id"]
    a1 = _upload_payload(client, sr.transient_slg_successful_reclose())
    a2 = _upload_payload(client, sr.permanent_fault_failed_reclose())
    r1 = client.post(f"/api/incidents/{incident_id}/records", json={"analysis_id": a1}).json()["incident_record_id"]
    r2 = client.post(f"/api/incidents/{incident_id}/records", json={"analysis_id": a2}).json()["incident_record_id"]

    resp = client.patch(f"/api/incidents/{incident_id}/records/order", json={"incident_record_ids": [r2, r1]})
    assert resp.status_code == 200
    ordered = resp.json()
    assert ordered[0]["incident_record_id"] == r2


def test_evidence_endpoints(client):
    incident_id = client.post("/api/incidents", json={"title": "Evidence API test"}).json()["incident_id"]
    resp = client.post(
        f"/api/incidents/{incident_id}/evidence",
        json={"evidence_type": "FIELD_INSPECTION", "description": "Broken insulator found", "confidence": "CONFIRMED"},
    )
    assert resp.status_code == 200
    evidence_id = resp.json()["evidence_id"]

    resp = client.get(f"/api/incidents/{incident_id}/evidence")
    assert len(resp.json()) == 1

    resp = client.delete(f"/api/incidents/{incident_id}/evidence/{evidence_id}")
    assert resp.status_code == 200
    assert client.get(f"/api/incidents/{incident_id}/evidence").json() == []


def test_feedback_endpoints(client):
    incident_id = client.post("/api/incidents", json={"title": "Feedback API test"}).json()["incident_id"]
    resp = client.post(f"/api/incidents/{incident_id}/feedback", json={"operator": "op1", "notes": "ok"})
    assert resp.status_code == 200

    resp = client.get(f"/api/incidents/{incident_id}/feedback")
    assert len(resp.json()) == 1
    assert resp.json()[0]["operator"] == "op1"


def test_refresh_snapshots_endpoint(client):
    incident_id = client.post("/api/incidents", json={"title": "Refresh test"}).json()["incident_id"]
    analysis_id = _upload_payload(client, sr.transient_slg_successful_reclose())
    client.post(f"/api/incidents/{incident_id}/records", json={"analysis_id": analysis_id})

    resp = client.post(f"/api/incidents/{incident_id}/refresh-snapshots")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_delete_incident_archives(client):
    incident_id = client.post("/api/incidents", json={"title": "Archive test"}).json()["incident_id"]
    resp = client.delete(f"/api/incidents/{incident_id}")
    assert resp.status_code == 200

    resp = client.get("/api/incidents")
    assert not any(i["incident_id"] == incident_id for i in resp.json())


def test_existing_upload_and_canonical_endpoints_still_work(client):
    """Backward compatibility: Stage 0 upload/canonical flow works unchanged
    with the incidents router mounted."""
    payload = sr.transient_slg_successful_reclose()
    analysis_id = _upload_payload(client, payload)

    resp = client.get(f"/api/analysis/{analysis_id}/canonical")
    assert resp.status_code == 200
    body = resp.json()
    assert body["record_id"] == analysis_id
    assert body["event_window"]["inception_time_ms"] is not None
