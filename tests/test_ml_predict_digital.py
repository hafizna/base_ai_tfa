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
