"""Each fault's reasoning ledger in a reconstructed incident
(``webapp.api.incidents.episode_reasoning``): the fault record's chain,
completed with the reclose file, the other line end and the cause."""

from __future__ import annotations

import copy
import importlib

import pytest

from tests.test_incident_reclose_linking import fault_and_trip, reclose_capture
from webapp.api.storage import save_analysis


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from webapp.api.incidents import storage as storage_module
    importlib.reload(storage_module)
    from webapp.api.incidents import service as service_module
    importlib.reload(service_module)
    return service_module


def _at(payload: dict, station: str, device: str) -> dict:
    payload = copy.deepcopy(payload)
    payload["station_name"], payload["rec_dev_id"] = station, device
    return payload


def _ledger(service, payloads):
    incident = service.create_incident(title="Ledger", station_name="GI TEST")
    for payload in payloads:
        service.attach_record(incident.incident_id, analysis_id=save_analysis(payload), protection_type="21",
                              override_warnings=True)
    service.reconstruct(incident.incident_id)
    episodes = service.get_episodes(incident.incident_id)
    return [ep.interpretation.get("reasoning") for ep in episodes]


def test_the_reclose_captured_in_its_own_file_completes_the_trip_row(service):
    (ledger,) = _ledger(service, [fault_and_trip("2026-02-01T10:00:00.000"), reclose_capture("2026-02-01T10:00:05.000")])
    rows = {row["key"]: row for row in ledger["rows"]}

    assert [row["key"] for row in ledger["rows"]][:3] == ["fault", "phases", "clearing"]
    assert rows["phases"]["title"].startswith("S-T")
    assert rows["trip_path"]["title"] == "Z1, seketika"
    trip_reclose = rows["trip_reclose"]
    assert "reclose berhasil setelah" in trip_reclose["title"]
    assert trip_reclose["title"].endswith(")")  # names the reclose record
    assert {"F6.4", "F6.5"} <= set(trip_reclose["rules"])
    assert rows["cause"]["label"] == "Penyebab"
    assert rows["cause"]["confidence"] == "ai"
    assert "other_end" not in rows


def test_the_far_end_adds_its_own_reading_of_the_fault(service):
    ledgers = _ledger(service, [
        fault_and_trip("2026-02-01T10:00:00.000"),
        reclose_capture("2026-02-01T10:00:05.000"),
        _at(fault_and_trip("2026-02-01T03:00:00.020"), "GI FAR", "QUALITROL"),
        _at(reclose_capture("2026-02-01T03:00:05.100"), "GI FAR", "QUALITROL"),
    ])
    (ledger,) = ledgers
    far = next(row for row in ledger["rows"] if row["key"] == "other_end")
    assert far["title"].startswith("GI FAR: fasa S-T")
    assert "Fasa sama di kedua ujung." in far["evidence"]
    assert any(line.startswith("Reclose berhasil setelah dead time") for line in far["evidence"])
    assert "Jam perekam diselaraskan pada awal gangguan (koreksi −7 jam, +20 ms)." in far["evidence"]
    assert far["rules"][0] == "F7.3"
    # The ledger's fault record is this end's, not the far end's.
    rows = {row["key"]: row for row in ledger["rows"]}
    assert rows["trip_path"]["title"] == "Z1, seketika"
