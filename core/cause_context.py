"""Shared recorded context and episode signatures, independent of cause labels.

These measurements are evidence, not new inputs to the deployed 17-feature
classifier. Neither repeated faults nor distortion identify a physical cause.
"""
from __future__ import annotations

import numpy as np


def recorded_context(window) -> dict:
    sequence = window.sequence or {}
    outcome = sequence.get("restoration_outcome") or "unknown"
    return {
        "scope": "recorded_interval_only",
        "restoration_outcome": outcome,
        "mechanical_close_confirmed": bool(sequence.get("mechanical_close_confirmed")),
        "refault_after_reclose": bool(sequence.get("refault_after_reclose")),
        "sotf_after_reclose": bool(sequence.get("sotf_after_reclose")),
        "episode_count": len(window.fault_episodes or []) or (1 if window.inception_idx is not None else 0),
        "timing_confidence": window.confidence,
        "outcome_is_physical_cause": False,
        "summary": {
            "failed": "Pemulihan gagal pada percobaan reclose yang terekam; penyebab fisik dinilai terpisah.",
            "successful": "Pemulihan bertahan sampai akhir rekaman; penyebab fisik dinilai terpisah.",
        }.get(outcome, "Hasil pemulihan belum dapat ditentukan dari rekaman ini."),
    }


def episode_signatures(payload: dict, window) -> dict:
    time = np.asarray(payload.get("time") or [], dtype=float)
    if len(time) < 4 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        return {"episodes": [], "comparison": [], "unavailable_reason": "invalid_time_axis"}
    frequency = float(payload.get("frequency") or 50)
    if not np.isfinite(frequency) or frequency <= 0:
        return {"episodes": [], "comparison": [], "unavailable_reason": "invalid_frequency"}
    dt = float(np.median(np.diff(time)))
    fs = 1 / dt
    period = 1 / frequency
    channels = [(ch, np.asarray(ch.get("samples") or [], dtype=float))
                for ch in payload.get("analog_channels", []) if ch.get("measurement") in {"current", "voltage"}]
    episodes = window.fault_episodes or ([{
        "episode_index": 0, "inception_time_ms": window.inception_time_ms,
        "clearing_time_ms": window.clearing_time_ms, "faulted_phases": window.faulted_phases,
    }] if window.inception_time_ms is not None and window.method != "dead_time_recording" else [])
    result = []
    first_start = episodes[0].get("inception_time_ms") if episodes else None
    baseline_mask = ((time >= first_start / 1000 - 2 * period) & (time < first_start / 1000)) if first_start is not None else np.zeros(len(time), dtype=bool)
    # A dead-time capture has no energized baseline. Later episodes never use
    # the immediately preceding dead time as a healthy reference.
    baseline_valid = (window.method != "dead_time_recording" and first_start is not None
                      and first_start / 1000 - time[0] >= 2 * period)
    for episode in episodes:
        start, stop = episode.get("inception_time_ms"), episode.get("clearing_time_ms")
        entry = {"episode_index": episode.get("episode_index", len(result)),
                 "inception_time_ms": start, "clearing_time_ms": stop,
                 "faulted_phases": episode.get("faulted_phases") or [],
                 "after_reclose": bool(episode.get("after_reclose")),
                 "reference": "first_episode_prefault" if baseline_valid else "unavailable",
                 "sample_rate_hz": round(fs, 3), "nyquist_hz": round(fs / 2, 3),
                 "channels": [], "limitations": ["Signature is not a confirmed physical cause.",
                    "Distortion can also arise from DC offset, CT response, switching or other arcs."]}
        if start is None or stop is None or stop <= start:
            entry["limitations"].append("Onset/clearing unavailable; episode magnitudes are not estimated.")
            result.append(entry)
            continue
        mask = (time >= start / 1000) & (time <= stop / 1000)
        entry["duration_ms"] = stop - start
        # Reuse each cycle's fit across channels; do not scan the whole raw
        # waveform again for every channel/cycle combination.
        cycle_fits = []
        for number in range(int((stop - start) / (period * 1000))):
            lo, hi = np.searchsorted(time, [start / 1000 + number * period,
                                           start / 1000 + (number + 1) * period])
            if hi - lo < 4:
                continue
            cycle_time = time[lo:hi]
            design = np.column_stack((np.sin(2 * np.pi * frequency * cycle_time),
                                      np.cos(2 * np.pi * frequency * cycle_time), np.ones(hi - lo)))
            cycle_fits.append((slice(lo, hi), design, np.linalg.pinv(design)))
        for ch, values in channels:
            if len(values) != len(time) or not np.all(np.isfinite(values[mask])) or mask.sum() < 4:
                continue
            segment = values[mask]
            cycles = []
            for cm, design, inverse in cycle_fits:
                coeffs = inverse @ values[cm]
                rms = float(np.sqrt(np.mean(values[cm] ** 2)))
                cycles.append({"rms": round(rms, 6), "dc_mean": round(float(coeffs[2]), 6),
                    "fundamental_rms": round(float(np.hypot(*coeffs[:2]) / np.sqrt(2)), 6),
                    "fit_residual_ratio": round(float(np.sqrt(np.mean((values[cm] - design @ coeffs) ** 2)) / rms), 6) if rms > 0 else None})
            entry["channels"].append({"id": ch.get("id"), "phase": ch.get("phase"),
                "name": ch.get("canonical_name") or ch.get("name"), "measurement": ch.get("measurement"),
                "unit": ch.get("unit"), "peak_abs": round(float(np.max(np.abs(segment))), 6),
                "rms": round(float(np.sqrt(np.mean(segment ** 2))), 6),
                "prefault_rms": round(float(np.sqrt(np.mean(values[baseline_mask] ** 2))), 6) if baseline_valid and baseline_mask.any() else None,
                "cycles": cycles})
        result.append(entry)
    comparison = []
    for previous, current in zip(result, result[1:]):
        previous_channels = {(ch["name"], ch["unit"]): ch for ch in previous["channels"]}
        ratios = [{"channel": ch["name"], "unit": ch["unit"],
                   "rms_ratio": round(ch["rms"] / old["rms"], 6)}
                  for ch in current["channels"] if (old := previous_channels.get((ch["name"], ch["unit"]))) and old["rms"] > 0]
        comparison.append({"from_episode": previous["episode_index"], "to_episode": current["episode_index"],
            "same_faulted_phases": sorted(previous["faulted_phases"]) == sorted(current["faulted_phases"]),
            "channel_ratios": ratios, "physical_cause": "unconfirmed"})
    return {"schema_version": "episode_signatures_v1", "episodes": result, "comparison": comparison,
            "used_by_deployed_model": False}
