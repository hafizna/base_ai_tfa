import numpy as np

from webapp.api.routers.relay_ocr import _build_curve_points, _find_max_current


def test_curve_extends_past_high_measured_ratio():
    points = _build_curve_points("NI", 0.1, measured_ratio=8.0)

    assert points
    assert max(point.current_ratio for point in points) >= 10.0
    assert any(7.5 <= point.current_ratio <= 8.5 for point in points)


def test_max_current_uses_short_window_rms_not_whole_record_rms():
    time = np.arange(0, 2.0, 0.001)
    samples = np.zeros_like(time)
    samples[500:520] = 800.0
    channels = [
        {
            "canonical_name": "IA",
            "name": "IA",
            "samples": samples.tolist(),
        }
    ]

    measured = _find_max_current(channels, time.tolist(), frequency=50.0)
    whole_record_rms = float(np.sqrt(np.mean(samples**2)))

    assert measured > whole_record_rms * 5
    assert measured == 800.0
