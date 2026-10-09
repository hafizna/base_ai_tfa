"""One time axis for an incident's records (``webapp.api.incidents.time_axis``).

Records from one recorder are placed by their own clock; another recorder is
shifted onto the reference clock only by lining up a fault both recorded. The
joined waveform, the duplicate check and the timeline all read that axis, with
each record anchored at its first sample.
"""

from __future__ import annotations

import copy
import importlib
from datetime import datetime, timedelta

import numpy as np
import pytest

from core.clock_offsets import split_clock_offset
from tests.fixtures import synthetic_records as sr
from tests.test_incident_reclose_linking import fault_and_trip, reclose_capture
from webapp.api.incidents import joined_waveform
from webapp.api.incidents.alignment import assess_alignment
from webapp.api.incidents.episodes import group_episodes
from webapp.api.incidents.models import FaultEpisode, IncidentRecord
from webapp.api.incidents.relationships import build_relationships
from webapp.api.incidents.time_axis import build_time_axis
from webapp.api.storage import save_analysis


def _record(rid, start_iso, *, station="GI BRINGIN", device="ZQ6", seq=0, fault_ms=None, phases=("B", "C"),
            span_s=1.0, trigger_s=0.1):
    start = datetime.fromisoformat(start_iso) if start_iso else None
    snapshot = {
        "source_metadata": {"station_name": station, "rec_dev_id": device, "duration_s": span_s},
        "event_window": {
            "record_start_ms": 0.0,
            "trigger_time_ms": trigger_s * 1000.0,
            "inception_time_ms": fault_ms,
            "method": "waveform" if fault_ms is not None else "no_fault_evidence",
            "faulted_phases": list(phases),
        },
        "analog_trace": (
            {"summary": {"fault_start_ms": fault_ms, "high_current_phases": list(phases), "sagged_phases": list(phases)}}
            if fault_ms is not None else {}
        ),
    }
    return IncidentRecord(
        incident_record_id=rid, incident_id="inc", analysis_id=f"analysis-{rid}",
        station_name=station or None, relay_id=device or None,
        record_start_iso=start_iso,
        trigger_time_iso=(start + timedelta(seconds=trigger_s)).isoformat() if start else None,
        trigger_offset_s=trigger_s, sequence_index=seq, canonical_snapshot=snapshot,
    )


# --- clock offsets ---------------------------------------------------------------------------------

def test_a_whole_time_zone_is_split_from_the_clock_remainder():
    remainder, zone = split_clock_offset(7 * 3600.0 + 0.3)
    assert zone == 7 * 3600.0
    assert remainder == pytest.approx(0.3)


def test_days_apart_is_not_a_time_zone():
    # Two records 165.5 days apart, both on the hour: a whole number of
    # quarter hours, yet no pair of time zones is that far apart.
    raw = 165.5 * 86400.0
    assert split_clock_offset(raw) == (raw, None)


# --- placing recorders -------------------------------------------------------------------------------

def test_one_recorder_is_placed_by_its_own_clock():
    fault = _record("fault", "2023-08-21T15:15:03.600", seq=0, fault_ms=170.0)
    reclose = _record("reclose", "2023-08-21T15:15:08.700", seq=1)
    axis = build_time_axis([fault, reclose])

    assert {p.method for p in axis.placements.values()} == {"reference_clock"}
    assert axis.zero == datetime(2023, 8, 21, 15, 15, 3, 600000)
    assert axis.offset_s(reclose) == pytest.approx(5.1)
    assert axis.warnings == []


def test_other_end_in_another_time_zone_is_lined_up_on_the_shared_fault():
    # Bringin's relay runs on WIB; the Mojosongo DFR stamps UTC and reads
    # 0.02 ms late. The fault starts at one instant at both ends.
    bringin = _record("bringin", "2023-08-21T15:15:03.600", seq=0, fault_ms=170.0)
    mojosongo = _record("mojosongo", "2023-08-21T08:15:03.670020", station="GI MOJOSONGO", device="Qualitrol",
                        seq=1, fault_ms=100.0)
    axis = build_time_axis([bringin, mojosongo])

    placement = axis.placement(mojosongo)
    assert placement.method == "fault_aligned"
    assert placement.zone_offset_h == -7.0
    assert placement.clock_offset_ms == pytest.approx(0.02, abs=1e-3)
    assert axis.absolute(mojosongo, 0.100) == axis.absolute(bringin, 0.170)
    assert axis.start(mojosongo) == datetime(2023, 8, 21, 15, 15, 3, 670000)
    assert axis.zero == datetime(2023, 8, 21, 15, 15, 3, 600000)
    assert [w["type"] for w in axis.warnings] == ["TIME_ZONE_OFFSET_REMOVED"]


def test_a_recorder_sharing_no_fault_keeps_its_clock_and_is_flagged():
    bringin = _record("bringin", "2023-08-21T15:15:03.600", seq=0, fault_ms=170.0)
    reclose_only = _record("mojosongo", "2023-08-21T08:15:08.780", station="GI MOJOSONGO", device="Qualitrol", seq=1)
    axis = build_time_axis([bringin, reclose_only])

    placement = axis.placement(reclose_only)
    assert placement.method == "own_clock_unverified"
    assert axis.start(reclose_only) == datetime(2023, 8, 21, 8, 15, 8, 780000)
    warning = next(w for w in axis.warnings if w["type"] == "CLOCK_NOT_VERIFIED")
    assert "7.0 h before" in warning["description"]


def test_faults_on_different_phases_are_not_lined_up():
    relay = _record("relay", "2026-02-01T10:00:00", device="RELAY", seq=0, fault_ms=100.0, phases=("A",))
    dfr = _record("dfr", "2026-02-01T10:00:00.5", device="DFR", seq=1, fault_ms=100.0, phases=("B", "C"))
    axis = build_time_axis([relay, dfr])

    assert axis.placement(dfr).method == "own_clock_unverified"
    assert axis.offset_s(dfr) == pytest.approx(0.5)


def test_the_offset_most_fault_pairs_agree_on_wins():
    # The reference recorder caught two faults 0.6 s apart; the second recorder
    # runs 0.45 s fast. Pairing its first fault with the reference's second
    # gives a smaller (wrong) offset of 0.15 s; both true pairings agree on 0.45 s.
    first = _record("f1", "2026-02-01T10:00:00", device="RELAY", seq=0, fault_ms=100.0)
    second = _record("f2", "2026-02-01T10:00:00.6", device="RELAY", seq=1, fault_ms=100.0)
    remote_first = _record("r1", "2026-02-01T10:00:00.45", device="DFR", seq=2, fault_ms=100.0)
    remote_second = _record("r2", "2026-02-01T10:00:01.05", device="DFR", seq=3, fault_ms=100.0)
    axis = build_time_axis([first, second, remote_first, remote_second])

    placement = axis.placement(remote_first)
    assert placement.method == "fault_aligned"
    assert placement.correction_s == pytest.approx(-0.45)
    assert placement.aligned_on["agreeing_pairs"] == 2
    assert axis.absolute(remote_second, 0.1) == axis.absolute(second, 0.1)


def test_a_record_naming_no_recorder_is_never_shifted():
    relay = _record("relay", "2026-02-01T10:00:00", seq=0, fault_ms=100.0)
    anonymous = _record("anon", "2026-02-01T10:00:00.4", station="", device="", seq=1, fault_ms=100.0)
    axis = build_time_axis([relay, anonymous])

    assert axis.placement(anonymous).method == "own_clock_unverified"
    assert axis.offset_s(anonymous) == pytest.approx(0.4)
    assert any(w["type"] == "CLOCK_UNIDENTIFIED" for w in axis.warnings)


def test_trigger_reads_the_records_own_trigger_timestamp():
    record = _record("a", "2026-01-01T00:00:00", fault_ms=100.0)
    record.trigger_time_iso = "2026-01-01T00:00:30"
    axis = build_time_axis([record])

    assert axis.trigger(record) == datetime(2026, 1, 1, 0, 0, 30)
    assert axis.start(record) == datetime(2026, 1, 1, 0, 0, 0)


# --- joined waveform -----------------------------------------------------------------------------------

def _payload(duration_s: float, sr_hz: float = 1000.0) -> dict:
    t = np.arange(int(duration_s * sr_hz)) / sr_hz
    return {"time": t.tolist(), "analog_channels": [
        {"canonical_name": "IA", "measurement": "current", "samples": np.sin(2 * np.pi * 50 * t).tolist()},
    ]}


def _join(monkeypatch, records, payloads):
    monkeypatch.setattr(joined_waveform, "_load_line_payload", lambda analysis_id: payloads[analysis_id])
    episode = FaultEpisode(episode_id="ep", incident_id="inc", member_record_ids=[r.incident_record_id for r in records])
    return joined_waveform.build_joined_waveform(episode, {r.incident_record_id: r for r in records})


def test_joined_waveform_places_each_record_at_its_first_sample(monkeypatch):
    fault = _record("fault", "2026-02-01T10:00:00", seq=0, fault_ms=100.0, span_s=1.499)
    reclose = _record("reclose", "2026-02-01T10:00:05", seq=1, span_s=0.999)
    joined = _join(monkeypatch, [fault, reclose], {"analysis-fault": _payload(1.5), "analysis-reclose": _payload(1.0)})

    # 5 s apart on the recorder clock — not 1.5 s of the first record plus a
    # 5 s trigger-to-trigger gap.
    assert joined["segments"][1]["t_offset_s"] == pytest.approx(5.0)
    assert joined["segments"][1]["gap_precision"] == "measured"
    assert joined["segments"][1]["gap_seconds"] == pytest.approx(3.501, abs=1e-3)
    assert joined["gap_ranges"] == [{"start_s": pytest.approx(1.499), "end_s": pytest.approx(5.0),
                                     "precision": "measured", "long_gap": True}]


def test_joined_waveform_overlapping_records_leave_no_gap(monkeypatch):
    first = _record("first", "2026-02-01T10:00:00", seq=0, fault_ms=100.0)
    second = _record("second", "2026-02-01T10:00:00.5", seq=1, fault_ms=None)
    joined = _join(monkeypatch, [first, second], {"analysis-first": _payload(1.0), "analysis-second": _payload(1.0)})

    assert joined["segments"][1]["t_offset_s"] == pytest.approx(0.5)
    assert joined["segments"][1]["gap_seconds"] == 0.0
    assert joined["gap_ranges"] == []


def test_joined_waveform_without_absolute_time_goes_back_to_back(monkeypatch):
    first = _record("first", "2026-02-01T10:00:00", seq=0, fault_ms=100.0)
    untimed = _record("untimed", None, seq=1)
    joined = _join(monkeypatch, [first, untimed], {"analysis-first": _payload(1.0), "analysis-untimed": _payload(1.0)})

    assert joined["segments"][1]["t_offset_s"] == pytest.approx(0.999)
    assert joined["segments"][1]["gap_precision"] == "assumed_back_to_back"
    assert [w["type"] for w in joined["warnings"]] == ["GAP_NOT_MEASURED"]


# --- duplicate check across different pre-trigger lengths ------------------------------------------------

@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTS_DATA_DIR", str(tmp_path / "incidents"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from webapp.api.incidents import storage as storage_module
    importlib.reload(storage_module)
    from webapp.api.incidents import service as service_module
    importlib.reload(service_module)
    return service_module


def _with_longer_pretrigger(payload: dict, extra_s: float) -> dict:
    """The same capture from a device that started recording ``extra_s``
    earlier: prefault samples prepended, everything after unchanged."""
    payload = copy.deepcopy(payload)
    t = np.asarray(payload["time"], dtype=float)
    dt = t[1] - t[0]
    extra = int(round(extra_s / dt))
    t_new = np.arange(len(t) + extra) * dt
    for ch in payload["analog_channels"]:
        samples = np.asarray(ch["samples"], dtype=float)
        # The prefault cycle repeats, so a whole number of cycles of it extends the record backwards.
        cycle = int(round(1 / (payload["frequency"] * dt)))
        reps = int(np.ceil(extra / cycle))
        prefix = np.tile(samples[:cycle], reps)[-extra:] if extra else np.empty(0)
        ch["samples"] = np.concatenate([prefix, samples]).tolist()
    for ch in payload["status_channels"]:
        ch["samples"] = ([ch["samples"][0]] * extra) + list(ch["samples"])
    payload["time"] = t_new.tolist()
    payload["total_samples"] = len(t_new)
    payload["trigger_time"] = payload["trigger_offset_s"] = payload["trigger_offset_s"] + extra * dt
    return payload


def test_duplicate_captures_compared_from_their_first_samples(service):
    base = sr.transient_slg_successful_reclose()
    relay = copy.deepcopy(base)
    relay.update({"rec_dev_id": "RELAY_21", "start_time_iso": "2026-02-01T10:00:00",
                  "trigger_time_iso": "2026-02-01T10:00:00.300"})
    # The DFR started 0.21 s (10.5 cycles) earlier and triggered at the same instant.
    dfr = _with_longer_pretrigger(base, 0.21)
    dfr.update({"rec_dev_id": "DFR_EXTERNAL", "start_time_iso": "2026-02-01T09:59:59.790",
                "trigger_time_iso": "2026-02-01T10:00:00.300"})

    incident = service.create_incident(title="Duplicate with longer pre-trigger", station_name="GOLDEN TEST")
    for payload, device in ((relay, "RELAY_21"), (dfr, "DFR_EXTERNAL")):
        service.attach_record(incident.incident_id, analysis_id=save_analysis(payload), relay_id=device, protection_type="21")
    service.reconstruct(incident.incident_id)
    relationship = service.get_relationships(incident.incident_id)[0]

    assert relationship.relationship_type == "DUPLICATE_TRIGGER"
    assert relationship.metrics["waveform_similarity"]["mean_correlation"] > 0.99


# --- the other line end --------------------------------------------------------------------------------

def _at(payload: dict, station: str, device: str) -> dict:
    payload = copy.deepcopy(payload)
    payload["station_name"], payload["rec_dev_id"] = station, device
    return payload


def test_the_far_end_joins_the_episode_without_overwriting_this_ends_facts(service):
    # This end: fault and trip, then the reclose in its own file. The far end's
    # DFR stamps UTC, starts 20 ms later on its own clock, and its breaker
    # closes 80 ms after this end's.
    records = {
        "far_fault": _at(fault_and_trip("2026-02-01T03:00:00.020"), "GI FAR", "QUALITROL"),
        "far_reclose": _at(reclose_capture("2026-02-01T03:00:05.100"), "GI FAR", "QUALITROL"),
        "fault": fault_and_trip("2026-02-01T10:00:00.000"),
        "reclose": reclose_capture("2026-02-01T10:00:05.000"),
    }
    incident = service.create_incident(title="Two ends", station_name="GI TEST")
    ids = {}
    # The far end is attached first: the incident's own substation still sets
    # the clock and the end the episode is read from.
    for name, payload in records.items():
        record = service.attach_record(incident.incident_id, analysis_id=save_analysis(payload),
                                       protection_type="21", override_warnings=True)
        ids[name] = record.incident_record_id
    recon = service.reconstruct(incident.incident_id)
    rels = {(r.left_record_id, r.right_record_id): r for r in service.get_relationships(incident.incident_id)}
    episodes = service.get_episodes(incident.incident_id)

    assert recon.alignment["time_axis"]["reference_group"] == "GI TEST | DFR"
    # The far end is the other end of the line, not a record from the wrong bay.
    assert recon.same_bay_status != "MISMATCH_REQUIRES_REVIEW"
    assert any(e["type"] == "OTHER_LINE_END" for e in recon.same_bay_evidence)
    assert "REMOTE_END_UNAVAILABLE" not in {m["type"] for m in service.get_incident(incident.incident_id).missing_evidence}
    far_tie = rels[(ids["fault"], ids["far_fault"])]
    assert far_tie.relationship_type == "REMOTE_END_CAPTURE"
    assert far_tie.metrics["fault_start_difference_ms"] == pytest.approx(0.0, abs=0.5)
    assert rels[(ids["reclose"], ids["far_reclose"])].relationship_type == "REMOTE_END_CAPTURE"
    local_sequence = rels[(ids["fault"], ids["reclose"])]
    far_sequence = rels[(ids["far_fault"], ids["far_reclose"])]
    assert local_sequence.relationship_type == far_sequence.relationship_type == "RECLOSE_SEQUENCE"

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.member_record_ids == [ids["fault"], ids["reclose"], ids["far_fault"], ids["far_reclose"]]
    assert episode.faulted_phases == ["B", "C"]
    # Each end's dead time is its own breaker's.
    assert episode.observed_facts["reclose_dead_time_s"] == local_sequence.metrics["dead_time_s"]
    far = episode.observed_facts["other_recorders"][0]
    assert far["station"] == "GI FAR" and far["same_station"] is False
    assert far["clock"]["zone_offset_h"] == -7.0
    assert far["reclose_dead_time_s"] - episode.observed_facts["reclose_dead_time_s"] == pytest.approx(0.08, abs=0.005)

    roles = {e["incident_record_id"]: e["evidence_role"] for e in recon.physical_cause_evidence["records"]}
    assert roles[ids["fault"]] == "inception"
    assert roles[ids["far_fault"]] == roles[ids["far_reclose"]] == "remote_end"

    joined = service.get_joined_waveform(incident.incident_id, episode.episode_id)
    assert [s["incident_record_id"] for s in joined["segments"]] == [ids["fault"], ids["reclose"]]
    lane = joined["other_lanes"][0]
    assert lane["station"] == "GI FAR"
    assert [s["t_offset_s"] for s in lane["segments"]] == [pytest.approx(0.0, abs=1e-3), pytest.approx(5.08, abs=1e-3)]


def test_ends_reading_different_phases_are_flagged_for_review():
    local = _record("local", "2026-02-01T10:00:00", seq=0, fault_ms=100.0, phases=("A", "B", "C"))
    far = _record("far", "2026-02-01T10:00:00", station="GI FAR", device="DFR", seq=1, fault_ms=100.0, phases=("B", "C"))
    records = [local, far]
    counter = iter(range(100))

    def new_id() -> str:
        return f"id-{next(counter)}"

    axis = build_time_axis(records)
    alignment = assess_alignment(records, axis)
    relationships = build_relationships("inc", records, alignment, new_id, axis)
    episodes = group_episodes("inc", records, relationships, alignment.record_order, new_id, axis=axis)

    assert [r.relationship_type for r in relationships] == ["REMOTE_END_CAPTURE"]
    assert len(episodes) == 1
    flag = next(m for m in episodes[0].missing_evidence if m["type"] == "ENDS_DISAGREE_ON_FAULTED_PHASES")
    assert flag["requires_review"] is True
    assert "GI FAR on B-C" in flag["description"]
