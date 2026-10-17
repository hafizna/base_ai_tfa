"""A single-pole reclose read from records without breaker contacts, and a
reclose this end did not record read from the far end.

TanjungjatiÃ¢â‚¬â€œJepara #1 (26 Feb 2024): R-N fault, pole R tripped at both ends.
The Jepara DFR stopped 0.8 s in with pole R still open; the Tanjungjati DFR
(no breaker contacts, an "A/R" command channel) showed pole R closing ~1.0 s
after the trip and the load coming back. The incident read "no further
reclose Ã¢â‚¬â€ possibly lockout" with a 0.0 s dead time: the "AR in progress" edge
at the trip was taken for a reclose, the "A/R" channel was not read at all,
and the single-pole dead time was invisible to a three-phase dead-window
reading.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from core.event_analysis import build_event_window
from core.fault_detector import _ar_channel_role, _normalize_status_name
from webapp.api.storage import save_analysis

SR = 2400.0
FREQ = 50.0
W = 2 * np.pi * FREQ
FAULT = (0.10, 0.17)


def _spar_payload(start_iso: str, station: str, device: str, *, length_s: float = 2.0, close_at: float | None = 1.17,
                  load_back: bool = True, ar_command_at: float | None = None, ar_in_progress: bool = False) -> dict:
    """R-N fault, pole R trips; phases S and T carry load throughout. The
    line-side VT of the open pole reads the voltage induced by the healthy
    phases (~20%)."""
    t = np.arange(int(SR * length_s)) / SR
    sine = lambda amp, deg: amp * np.sin(W * t + np.deg2rad(deg))  # noqa: E731
    in_fault = (t >= FAULT[0]) & (t < FAULT[1])
    dead = (t >= FAULT[1]) & ((t < close_at) if close_at is not None else True)
    after = (t >= close_at) if close_at is not None else np.zeros(len(t), dtype=bool)

    ia = np.where(in_fault, sine(5000.0, -80), sine(300.0, 0))
    ia = np.where(dead, 0.0, ia)
    if not load_back:
        ia = np.where(after, 0.0, ia)  # line alive again, this breaker still open
    va = np.where(in_fault, sine(30.0, 0), sine(87.0, 0))
    va = np.where(dead, sine(17.0, 60), va)
    currents = [ia, sine(300.0, -120), sine(300.0, 120)]
    voltages = [va, sine(87.0, -120), sine(87.0, 120)]

    def status(name, on, off=None):
        samples = ((t >= on) & ((t < off) if off is not None else True)).astype(int)
        return {"id": name, "name": name, "samples": samples.tolist()}

    channels = [status("LP OPRT R", FAULT[0] + 0.02, FAULT[1] + 0.01)]
    if ar_command_at is not None:
        channels.append(status("A/R JPARA 1", ar_command_at, ar_command_at + 0.2))
    if ar_in_progress:
        channels.append(status("AR IN PROGRESS", FAULT[0] + 0.03))
        channels.append(status("AR NOT READY", FAULT[1] - 0.01))

    analog = []
    for canon, samples, unit, measurement in [*(("I" + p, s, "A", "current") for p, s in zip("ABC", currents)),
                                              *(("V" + p, s, "kV", "voltage") for p, s in zip("ABC", voltages))]:
        analog.append({"id": canon, "name": canon, "canonical_name": canon, "unit": unit, "phase": canon[-1],
                       "measurement": measurement, "ct_primary": 1.0, "ct_secondary": 1.0, "pors": "P",
                       "samples": np.asarray(samples, dtype=float).tolist()})
    start = datetime.fromisoformat(start_iso)
    return {
        "station_name": station, "rec_dev_id": device, "frequency": FREQ, "time": t.tolist(),
        "trigger_offset_s": 0.1, "trigger_time": 0.1, "total_samples": len(t), "start_time_iso": start_iso,
        "trigger_time_iso": (start + timedelta(seconds=0.1)).isoformat(),
        "analog_channels": analog, "status_channels": channels, "warnings": [],
    }


# --- one record -----------------------------------------------------------------------

def test_slashed_ar_names_read_as_auto_reclose_channels():
    assert _normalize_status_name("A/R JPARA 1") == "AR JPARA 1"
    assert _normalize_status_name("A\\R OPRT JEPARA2") == "AR OPRT JEPARA2"
    assert _ar_channel_role("AR JPARA 1") == "close"
    assert _ar_channel_role("AR IN PROGRESS TJ JATI 1") == "state"
    assert _ar_channel_role("AR NOT READY TJ JATI 1") == "state"
    assert _ar_channel_role("BUSBAR TRIP") is None  # "AR " inside a word is not AR


def test_single_pole_reclose_inside_a_record_without_breaker_contacts():
    window = build_event_window(_spar_payload("2024-02-26T01:00:33.000", "GI T", "DFR", ar_command_at=1.12))
    close = window.reclose_events[-1]
    assert close["source"] == "waveform_phase"
    assert close["phases"] == ["A"] and close["single_pole"] is True
    assert close["success"] is True and close["load_restored"] is True
    assert 950 < close["dead_time_ms"] < 1050
    assert abs(close["command_time"] - 1.12) < 0.01
    assert window.sequence["mechanical_close_confirmed"] is True
    assert window.sequence["restoration_outcome"] == "successful"
    assert window.open_at_end_phases == []


def test_a_record_that_stops_inside_the_dead_time_shows_no_reclose():
    window = build_event_window(_spar_payload("2024-02-26T01:00:33.000", "GI J", "DFR", length_s=0.8,
                                              close_at=None, ar_in_progress=True))
    # "AR in progress" / "AR not ready" rise as the breaker trips: not a reclose.
    assert window.reclose_events == []
    assert window.open_at_end_phases == ["A"]
    assert not window.sequence["mechanical_close_confirmed"]


def test_line_voltage_back_without_the_load_may_be_the_far_end_closing():
    window = build_event_window(_spar_payload("2024-02-26T01:00:33.000", "GI T", "DFR", load_back=False))
    close = window.reclose_events[-1]
    assert close["source"] == "waveform_phase" and close["success"] is True
    assert close["load_restored"] is False


# --- the incident -------------------------------------------------------------------------

@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from webapp.api.incidents import storage as storage_module
    importlib.reload(storage_module)
    from webapp.api.incidents import service as service_module
    importlib.reload(service_module)
    return service_module


def test_a_reclose_this_end_did_not_record_is_read_from_the_far_end(service):
    incident = service.create_incident(title="two ends", station_name="GI JEPARA")
    here = _spar_payload("2024-02-26T01:00:33.000", "GI JEPARA", "DFR J", length_s=0.8, close_at=None,
                         ar_in_progress=True)
    there = _spar_payload("2024-02-26T01:00:33.000", "GI TANJUNGJATI", "DFR T", ar_command_at=1.12)
    for payload in (here, there):
        service.attach_record(incident.incident_id, analysis_id=save_analysis(payload), protection_type="21",
                              override_warnings=True)  # the far end is another substation
    recon = service.reconstruct(incident.incident_id)

    (episode,) = service.get_episodes(incident.incident_id)
    facts = episode.observed_facts
    assert episode.reclose_outcome == "successful"
    assert facts["open_at_end_phases"] == ["A"]
    source = facts["reclose_outcome_source"]
    assert source["station"] == "GI TANJUNGJATI" and source["same_station"] is False
    assert source["reason"] == "this_end_open_at_end"
    assert 0.95 < source["dead_time_s"] < 1.05
    assert recon.protection_sequence_interpretation["event_class"] == "SINGLE_TRANSIENT_FAULT"


def test_this_ends_own_reclose_is_not_replaced_by_the_far_ends(service):
    incident = service.create_incident(title="two ends", station_name="GI JEPARA")
    here = _spar_payload("2024-02-26T01:00:33.000", "GI JEPARA", "DFR J", ar_command_at=1.12)
    there = _spar_payload("2024-02-26T01:00:33.000", "GI TANJUNGJATI", "DFR T", close_at=None)
    for payload in (here, there):
        service.attach_record(incident.incident_id, analysis_id=save_analysis(payload), protection_type="21",
                              override_warnings=True)  # the far end is another substation
    service.reconstruct(incident.incident_id)
    (episode,) = service.get_episodes(incident.incident_id)
    assert episode.reclose_outcome == "successful"
    assert "reclose_outcome_source" not in episode.observed_facts


# --- the real records, when available ---------------------------------------------------------

_ROOT = os.environ.get("RECLOSE_TJATI_JEPARA_ROOT")


@pytest.mark.skipif(not _ROOT or not Path(_ROOT).is_dir(), reason="TanjungjatiÃ¢â‚¬â€œJepara records not available")
def test_tanjungjati_jepara_records():
    from core.comtrade_parser import parse_comtrade
    from core.fault_detector import detect_fault

    root = Path(_ROOT)
    jepara = detect_fault(parse_comtrade(str(next(root.rglob("*GI JEPARA*.cfg")))))
    tjati = detect_fault(parse_comtrade(str(next(root.rglob("ZQF.cfg")))))
    assert jepara.reclose_events == [] and jepara.open_at_end_phases == ["A"]
    (close,) = tjati.reclose_events
    assert close["success"] is True and close["single_pole"] is True and close["phases"] == ["A"]
    assert 900 < close["dead_time_ms"] < 1100
