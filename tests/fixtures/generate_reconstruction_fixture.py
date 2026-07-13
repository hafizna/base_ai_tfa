"""One-off generator for tests/fixtures/reconstruction_response.json.

Run manually (not part of the test suite) whenever the Stage 2 API response
shape changes intentionally:

    python tests/fixtures/generate_reconstruction_fixture.py

Produces a deterministic, ID-normalized snapshot of a real
POST /api/incidents/{id}/reconstruct response (plus its companion
timeline/relationships/episodes/records endpoints) for frontend contract
tests and mock data — real backend shapes, not hand-written guesses.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "reconstruction_response.json"


def _normalize_ids(obj, id_map: dict[str, str], counters: dict[str, int]):
    """Replace hex/uuid-like IDs with stable placeholders so the fixture
    diffs cleanly across regenerations."""
    if isinstance(obj, dict):
        return {k: _normalize_ids(v, id_map, counters) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_ids(v, id_map, counters) for v in obj]
    if isinstance(obj, str) and re.fullmatch(r"[0-9a-f]{32}", obj):
        if obj not in id_map:
            counters["analysis"] += 1
            id_map[obj] = f"analysis-id-{counters['analysis']:02d}"
        return id_map[obj]
    return obj


def main() -> None:
    import os

    os.environ["INCIDENTS_DATA_DIR"] = tempfile.mkdtemp()
    os.environ.pop("DATABASE_URL", None)

    from fastapi.testclient import TestClient

    from tests.fixtures.comtrade_writer import synthetic_cfg_dat_bytes
    from webapp.api.main import app

    client = TestClient(app)

    incident_id = client.post(
        "/api/incidents",
        json={"title": "GI COMAL Bay 1 recloser sequence", "station_name": "GOLDEN TEST", "bay_name": "BAY 1"},
    ).json()["incident_id"]

    cfg1, dat1 = synthetic_cfg_dat_bytes(fault_idx_fraction=0.3, rec_dev_id="RELAY_21")
    cfg2, dat2 = synthetic_cfg_dat_bytes(fault_idx_fraction=0.5, rec_dev_id="RELAY_87L")

    files = [
        ("files", ("record1.cfg", cfg1, "text/plain")),
        ("files", ("record1.dat", dat1, "application/octet-stream")),
        ("files", ("record2.cfg", cfg2, "text/plain")),
        ("files", ("record2.dat", dat2, "application/octet-stream")),
    ]
    upload = client.post(f"/api/incidents/{incident_id}/upload-records", files=files).json()
    reconstruction = client.post(f"/api/incidents/{incident_id}/reconstruct", json={}).json()
    incident = client.get(f"/api/incidents/{incident_id}").json()
    timeline = client.get(f"/api/incidents/{incident_id}/timeline").json()
    relationships = client.get(f"/api/incidents/{incident_id}/relationships").json()
    episodes = client.get(f"/api/incidents/{incident_id}/episodes").json()
    records = client.get(f"/api/incidents/{incident_id}/records").json()
    reconstructions = client.get(f"/api/incidents/{incident_id}/reconstructions").json()

    id_map: dict[str, str] = {}
    counters = {"analysis": 0}

    fixture = {
        "incident_id": incident_id,
        "upload": upload,
        "incident": incident,
        "reconstruction": reconstruction,
        "reconstructions": reconstructions,
        "timeline": timeline,
        "relationships": relationships,
        "episodes": episodes,
        "records": records,
    }
    fixture = _normalize_ids(fixture, id_map, counters)

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
