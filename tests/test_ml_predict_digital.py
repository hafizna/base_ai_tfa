import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.api.ml_predict import _digital_sequence_features


def _time_ms(stop_ms: int = 1400) -> np.ndarray:
    return np.arange(0, stop_ms + 1, dtype=float) / 1000.0


def test_cb_open_flicker_does_not_become_reclose_success():
    time = _time_ms()
    samples = np.zeros(len(time), dtype=int)
    samples[194:] = 1
    samples[196:198] = 0  # contact bounce, not a real close
    samples[1288:] = 0

    features = _digital_sequence_features(
        [{"name": "CB OPEN R GBG2", "samples": samples.tolist()}],
        time,
        inception_idx=100,
    )

    assert features["digital_ar_status"] is True
    assert features["digital_first_cb_open_ms"] == 194.0
    assert features["digital_first_cb_close_ms"] == 1288.0
    assert features["digital_ar_dead_time_ms"] == 1094.0
    assert features["digital_reclose_mode"] == "single_pole"


def test_cb_open_short_off_flicker_without_stable_close_is_unknown():
    time = _time_ms()
    samples = np.zeros(len(time), dtype=int)
    samples[194:] = 1
    samples[196:198] = 0  # contact bounce only; breaker remains open

    features = _digital_sequence_features(
        [{"name": "CB OPEN R GBG2", "samples": samples.tolist()}],
        time,
        inception_idx=100,
    )

    assert features["digital_ar_attempted"] is True
    assert features["digital_ar_status"] is None
    assert features["digital_first_cb_open_ms"] == 194.0
    assert features["digital_first_cb_close_ms"] is None
    assert features["digital_ar_dead_time_ms"] is None


def test_ar_slash_and_cb_aux_phase_contact_detect_reclose_success():
    time = _time_ms(5000)

    ar_in_prog = np.zeros(len(time), dtype=int)
    ar_in_prog[3413:4415] = 1

    ar_close = np.zeros(len(time), dtype=int)
    ar_close[4415:4615] = 1

    cb_aux_b = np.zeros(len(time), dtype=int)
    cb_aux_b[3478:4473] = 1

    features = _digital_sequence_features(
        [
            {"name": "A/R 1P In Prog", "samples": ar_in_prog.tolist()},
            {"name": "A/R Close", "samples": ar_close.tolist()},
            {"name": "CB Aux B", "samples": cb_aux_b.tolist()},
        ],
        time,
        inception_idx=3300,
    )

    assert features["digital_ar_attempted"] is True
    assert features["digital_ar_status"] is True
    assert features["digital_cb_open_phases"] == ["B"]
    assert features["digital_cb_close_phases"] == ["B"]
    assert features["digital_first_cb_open_ms"] == 3478.0
    assert features["digital_first_cb_close_ms"] == 4473.0
    assert features["digital_ar_dead_time_ms"] == 995.0
    assert features["digital_reclose_mode"] == "single_pole"


def test_closed_state_and_bare_52a_contacts_with_auto_close_read_a_single_pole_reclose():
    """RWALO-PLTU #1 (12 Jul 2025): the breaker position was recorded as
    "CB Closed C ph" and "L3 Status 52A T" (1 = closed) and the AR command as
    "Auto Close". None was read, so the relay-21 page showed A/R N/A for a
    single-pole reclose inside the record."""
    time = _time_ms(4100)
    trip_c = np.zeros(len(time), dtype=int)
    trip_c[506:606] = 1
    closed_c = np.ones(len(time), dtype=int)
    closed_c[553:1588] = 0
    status_52a_t = np.ones(len(time), dtype=int)
    status_52a_t[551:1583] = 0
    auto_close = np.zeros(len(time), dtype=int)
    auto_close[1525:1625] = 1

    features = _digital_sequence_features(
        [
            {"name": "Trip Output C", "samples": trip_c.tolist()},
            {"name": "CB Closed C ph", "samples": closed_c.tolist()},
            {"name": "CB Closed A ph", "samples": np.ones(len(time), dtype=int).tolist()},
            {"name": "L3 Status 52A T", "samples": status_52a_t.tolist()},
            {"name": "Auto Close", "samples": auto_close.tolist()},
        ],
        time,
        inception_idx=493,
    )

    assert features["digital_ar_attempted"] is True
    assert features["digital_ar_status"] is True
    assert features["digital_cb_open_phases"] == ["C"]
    assert features["digital_cb_close_phases"] == ["C"]
    assert features["digital_ar_dead_time_ms"] == 1032.0
    assert features["digital_reclose_mode"] == "single_pole"


def test_a_closed_state_channel_already_open_at_the_fault_is_not_a_trip():
    time = _time_ms(2000)
    features = _digital_sequence_features(
        [{"name": "CB Closed B ph", "samples": np.zeros(len(time), dtype=int).tolist()}],
        time,
        inception_idx=500,
    )
    assert features["digital_cb_open_phases"] == []
    assert features["digital_ar_status"] is None


def test_phase_select_channel_recognized_as_startup_evidence():
    """Regression test for a real Kebumen-Gombong SLG-A record: a
    "Phase Select A" digital channel is a distance relay's own authoritative
    single-pole fault-phase determination, but it matched none of the old
    is_startup keywords (STARTUP/PICKUP) — so faulted-phase determination
    fell through to the waveform per-phase RMS threshold in
    _extract_electrical_features, which misread the healthy phase C's small
    mutual-coupling current as a second faulted phase and reported a
    non-existent DLG (A+C) fault instead of the real SLG (A)."""
    time = _time_ms()
    samples = np.zeros(len(time), dtype=int)
    samples[240:] = 1
    samples[400:] = 0

    features = _digital_sequence_features(
        [{"name": "Phase Select A", "samples": samples.tolist()}],
        time,
        inception_idx=200,
    )

    assert features["digital_startup_phases"] == ["A"]


def test_pcs900_phs_channel_recognized_as_startup_evidence():
    """PCS900 vendor convention names this channel PhSA/PhSB/PhSC (see
    core/protection_router.py's PhS notation comment) — same "phase
    selector output" concept as "Phase Select A", different naming."""
    time = _time_ms()
    samples = np.zeros(len(time), dtype=int)
    samples[240:] = 1
    samples[400:] = 0

    features = _digital_sequence_features(
        [{"name": "PhSA", "samples": samples.tolist()}],
        time,
        inception_idx=200,
    )

    assert features["digital_startup_phases"] == ["A"]
