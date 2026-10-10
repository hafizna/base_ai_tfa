"""Reclose read from the breaker position contacts (core.fault_detector).

A record that names its breaker contacts "CB Closed C ph" / "L3 Status 52A T"
and its AR command "Auto Close" (RWALO-PLTU #1, 12 Jul 2025) matched none of
the AR keywords, so a single-pole reclose inside the record went unseen."""

from types import SimpleNamespace

import numpy as np

from core.fault_detector import _detect_reclose_from_status

SR = 2000.0  # samples per second
N = int(4.0 * SR)
FAULT = int(0.5 * SR)


def _record(*channels: tuple[str, np.ndarray]):
    time = np.arange(N) / SR
    return SimpleNamespace(
        time=time,
        status_channels=[SimpleNamespace(name=name, samples=samples) for name, samples in channels],
    )


def _closed_contact(open_s: float, close_s: float | None, reopen_s: float | None = None) -> np.ndarray:
    """1 while the breaker is closed."""
    samples = np.ones(N, dtype=int)
    samples[int(open_s * SR):int(close_s * SR) if close_s else N] = 0
    if reopen_s is not None:
        samples[int(reopen_s * SR):] = 0
    return samples


def test_a_closed_contact_coming_back_is_a_successful_reclose():
    record = _record(
        ("CB Closed C ph", _closed_contact(0.553, 1.588)),
        ("L3 Status 52A T", _closed_contact(0.551, 1.583)),
        ("CB Closed A ph", np.ones(N, dtype=int)),
        ("L5 CB Healthy", _closed_contact(1.633, None)),
    )
    events = _detect_reclose_from_status(record, FAULT)
    assert len(events) == 1
    assert abs(events[0]["time"] - 1.583) < 1e-3  # the first contact back
    assert events[0]["success"] is True
    assert events[0]["source"] == "breaker_position"


def test_an_open_state_contact_reads_the_same_way():
    samples = np.zeros(N, dtype=int)
    samples[int(0.55 * SR):int(1.55 * SR)] = 1  # 52B: 1 while open
    events = _detect_reclose_from_status(_record(("PMT 52B R", samples)), FAULT)
    assert abs(events[0]["time"] - 1.55) < 1e-3 and events[0]["success"] is True


def test_the_breaker_opening_again_is_a_failed_reclose():
    events = _detect_reclose_from_status(_record(("CB Closed C ph", _closed_contact(0.553, 1.588, reopen_s=1.66))), FAULT)
    assert events[0]["success"] is False


def test_a_record_ending_right_after_the_close_leaves_the_outcome_open():
    events = _detect_reclose_from_status(_record(("CB Closed C ph", _closed_contact(0.553, 3.95))), FAULT)
    assert events[0]["success"] is None


def test_contact_bounce_is_not_an_opening():
    samples = _closed_contact(0.553, 1.588)
    samples[int(0.52 * SR):int(0.522 * SR)] = 0  # 2 ms bounce before the real opening
    events = _detect_reclose_from_status(_record(("CB Closed C ph", samples)), FAULT)
    assert abs(events[0]["time"] - 1.588) < 1e-3


def test_a_close_command_pulse_is_not_a_position():
    pulse = np.zeros(N, dtype=int)
    pulse[int(1.5 * SR):int(1.7 * SR)] = 1
    assert _detect_reclose_from_status(_record(("BO CB CLOSE", pulse)), FAULT) == []


def test_an_opening_without_a_return_is_no_reclose():
    assert _detect_reclose_from_status(_record(("CB Closed C ph", _closed_contact(0.553, None))), FAULT) == []
