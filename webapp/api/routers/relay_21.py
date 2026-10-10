"""Relay 21 (Distance) — impedance locus + AI fault analysis."""

import sys
import asyncio
from pathlib import Path
from functools import partial
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ..schemas import (
    LocusAnalysisRequest, LocusResponse, LocusPoint, LocusDiagnostics,
    LocusBatchRequest, LocusBatchResponse,
    LocusEventsResponse,
    AIFaultFeatures, AIFaultResult,
)
from ..storage import load_analysis
from ..ml_predict import run_ml_prediction, extract_ml_features, _digital_sequence_features
from ..fault_detection import detect_fault_presence, _is_operate_status
from core.event_analysis import build_event_window
from core.fault_detector import _extract_line_tag
from core.line_selection import scope_payload

router = APIRouter(prefix="/api/analyze/21", tags=["relay-21"])


def _canonical_inception_idx(payload: dict, time: np.ndarray) -> tuple[int, str, float]:
    """Single source of truth for the fault-inception sample index.

    Returns (inception_idx, timing_source, timing_confidence). Falls back to
    index 0 with source "trigger_fallback"/"no_fault_evidence" when the
    canonical detector found no usable evidence — callers must not treat that
    as a detected inception.
    """
    window = build_event_window(payload)
    if window.inception_idx is not None and window.inception_idx < len(time):
        return window.inception_idx, window.method, window.confidence
    return 0, window.method, window.confidence

# Phase-to-channel mappings for each loop
LOOP_CHANNELS = {
    "ZA":  {"v": ["VA", "VAN", "UA"], "i": ["IA"], "phase": "A"},
    "ZB":  {"v": ["VB", "VBN", "UB"], "i": ["IB"], "phase": "B"},
    "ZC":  {"v": ["VC", "VCN", "UC"], "i": ["IC"], "phase": "C"},
    "ZAB": {"v": ["VAB", "UAB"], "i": ["IA", "IB"], "diff": True, "phases": ("A", "B")},
    "ZBC": {"v": ["VBC", "UBC"], "i": ["IB", "IC"], "diff": True, "phases": ("B", "C")},
    "ZCA": {"v": ["VCA", "UCA"], "i": ["IC", "IA"], "diff": True, "phases": ("C", "A")},
}


def _detect_active_line_tag(channels: list) -> Optional[str]:
    """Pick the line/circuit tag with the largest peak current in this
    record — same tag-extraction and scoring rule as
    core.fault_detector._detect_active_line_tag_from_currents, reimplemented
    here against the dict-shaped payload channels (that one takes parsed
    AnalogChannel objects).

    Why this matters: an external DFR CFG can record TWO lines side by side
    in one file (e.g. "IR BRINGIN 1" AND "IR BRINGIN 2" both present, one
    healthy, one faulted). Every channel finder below used to return the
    FIRST name/phase match regardless of which line it belonged to — for a
    real Mojosongo Bay Bringin #2 incident this meant the whole electrical
    summary (I peak, V sag, locus, report) silently described the healthy
    line while the actual fault (current at >5x CT nominal) was on the
    other one. Channels with no extractable tag (single-line CFGs, the
    overwhelming majority of records) are unaffected — this returns None
    and every finder below falls back to its original first-match
    behaviour."""
    scores: dict[str, float] = {}
    for ch in channels:
        if ch.get("measurement") != "current":
            continue
        canonical = (ch.get("canonical_name") or "").upper()
        if canonical not in ("IA", "IB", "IC"):
            continue
        samples = ch.get("samples") or []
        if not samples:
            continue
        tag = _extract_line_tag(ch.get("name") or "")
        if not tag:
            continue
        peak = float(np.max(np.abs(np.asarray(samples, dtype=float))))
        scores[tag] = scores.get(tag, 0.0) + peak
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def _find_channel(channels, candidates: list[str], preferred_tag: Optional[str] = None) -> Optional[np.ndarray]:
    """Return samples for the first matching canonical name, preferring a
    channel tagged with ``preferred_tag`` (see _detect_active_line_tag) when
    more than one channel matches the same name/canonical_name."""
    wanted = {c.upper() for c in candidates}
    matches = []
    for ch in channels:
        canonical = (ch.get("canonical_name") or "").upper()
        name = (ch.get("name") or "").upper()
        if canonical in wanted or name in wanted:
            matches.append(ch)
    if not matches:
        return None
    if preferred_tag:
        tagged = [ch for ch in matches if _extract_line_tag(ch.get("name") or "") == preferred_tag]
        if tagged:
            return np.array(tagged[0]["samples"], dtype=float)
    return np.array(matches[0]["samples"], dtype=float)


def _secondary_impedance_scale(channels: list) -> float:
    """Convert primary ohms to relay secondary ohms: Zsec = Zpri * CT_ratio / VT_ratio."""
    vt_ratio = 1.0
    ct_ratio = 1.0
    for ch in channels:
        measurement = ch.get("measurement")
        primary = float(ch.get("ct_primary") or 1.0)
        secondary = float(ch.get("ct_secondary") or 1.0)
        if primary <= 0 or secondary <= 0:
            continue
        ratio = primary / secondary
        if measurement == "voltage" and ratio > 1.0 and vt_ratio == 1.0:
            vt_ratio = ratio
        elif measurement == "current" and ratio > 1.0 and ct_ratio == 1.0:
            ct_ratio = ratio

    return (ct_ratio / vt_ratio) if vt_ratio > 0 else 1.0


def _voltage_to_volts_scale(channels: list) -> float:
    """COMTRADE parser stores power-system voltages in kV; impedance math needs volts."""
    for ch in channels:
        if ch.get("measurement") != "voltage":
            continue
        unit = (ch.get("unit") or "").strip().lower()
        if unit == "kv":
            return 1000.0
        if unit == "mv":
            return 1_000_000.0
        return 1.0
    return 1.0


def _find_phase_channel(
    channels, measurement: str, phase: str, preferred_tag: Optional[str] = None,
) -> Optional[np.ndarray]:
    channel = _pick_phase_channel(channels, measurement, phase, preferred_tag)
    return None if channel is None else np.array(channel["samples"], dtype=float)


def _pick_phase_channel(
    channels, measurement: str, phase: str, preferred_tag: Optional[str] = None,
) -> Optional[dict]:
    """Phase voltage/current channel. An exact canonical-name (VA/IB/...) or
    phase-field match always wins; the loose name-suffix aliases ("...L2",
    "...2") are only a fallback for channels the normalizer couldn't name —
    a suffix digit is usually a LINE number ("VT R MJSNG2" is phase A of line
    2, not phase B), so letting it compete with exact matches returned the
    wrong phase on multi-line DFR records."""
    phase = phase.upper()
    aliases = {phase}
    if phase == "A":
        aliases.update({"L1", "1"})
    elif phase == "B":
        aliases.update({"L2", "2"})
    elif phase == "C":
        aliases.update({"L3", "3"})
    canonical_wanted = ("V" if measurement == "voltage" else "I") + phase

    exact = []
    loose = []
    for ch in channels:
        if ch.get("measurement") != measurement:
            continue
        ch_phase = (ch.get("phase") or "").upper()
        canonical = (ch.get("canonical_name") or "").upper()
        name = (ch.get("name") or "").upper()
        if canonical == canonical_wanted or ch_phase == phase:
            exact.append(ch)
        elif ch_phase in aliases or any(canonical.endswith(alias) or name.endswith(alias) for alias in aliases):
            loose.append(ch)
    matches = exact or loose
    if not matches:
        return None
    if preferred_tag:
        tagged = [ch for ch in matches if _extract_line_tag(ch.get("name") or "") == preferred_tag]
        if tagged:
            return tagged[0]
    return matches[0]


def _find_phase_voltage(channels, phase: str, preferred_tag: Optional[str] = None) -> Optional[np.ndarray]:
    return _find_phase_channel(channels, "voltage", phase, preferred_tag)


def _find_phase_current(channels, phase: str, preferred_tag: Optional[str] = None) -> Optional[np.ndarray]:
    return _find_phase_channel(channels, "current", phase, preferred_tag)


def _find_voltage_for_loop(channels, mapping: dict, preferred_tag: Optional[str] = None) -> Optional[np.ndarray]:
    direct = _find_channel(channels, mapping["v"], preferred_tag)
    if direct is not None:
        return direct.astype(float)

    phase = mapping.get("phase")
    if phase:
        return _find_phase_voltage(channels, phase, preferred_tag)

    phases = mapping.get("phases")
    if phases:
        left = _find_phase_voltage(channels, phases[0], preferred_tag)
        right = _find_phase_voltage(channels, phases[1], preferred_tag)
        if left is not None and right is not None:
            return left - right

    return None


def _fundamental_phasor(
    samples: np.ndarray, start: int, win: int, freq: float, sr: float,
    inception_idx: Optional[int] = None,
) -> complex:
    """Return the complex fundamental phasor for one analysis window.

    Uses half-cycle differencing (y[n] = x[n] - x[n-win/2]) before the DFT
    when a half-cycle of history is available. This is the classical
    decaying-DC rejection technique for digital relay phasor estimation
    (Sachdev & Baribeau 1979): a pure exponential DC term e^(-t/tau) is not
    periodic, so a rectangular one-cycle DFT alone leaks part of it into the
    fundamental estimate — differencing against a half-cycle-earlier sample
    cancels most of that leakage (verified numerically: ~60-70% error
    reduction for windows entirely after inception, across tau = 0.5-5
    cycles) while a genuine 50/60 Hz sinusoid is preserved exactly
    (x[n] - x[n-half] = 2*x[n], since cos(theta-pi) = -cos(theta)), just
    needing a 180 deg phase correction and a matching halved amplitude scale.

    Falls back to plain mean-removal (rectangular window) when:
    - the required half-cycle of history before `start` isn't available
      (e.g. the very first windows of a record), or
    - `inception_idx` is given and the reference window [start-half,
      start-half+win) straddles it — verified numerically that in this
      specific case (reference window itself contains the fault-inception
      step discontinuity) half-cycle differencing is LESS accurate than
      plain rectangular+mean-removal, because the "half-cycle-earlier"
      sample is no longer a clean pre-fault or post-fault reference. Once
      the reference window is entirely on one side of inception, the
      differencing technique is strictly better and is used.
    """
    segment = np.asarray(samples[start:start + win], dtype=float)
    if len(segment) != win:
        return complex(np.nan, np.nan)

    half = win // 2
    n = np.arange(win, dtype=float)
    kernel = np.exp(-1j * 2.0 * np.pi * freq * n / sr)

    ref_start = start - half
    ref_end = ref_start + win  # exclusive
    ref_straddles_inception = (
        inception_idx is not None and ref_start < inception_idx < ref_end
    )

    if ref_start >= 0 and not ref_straddles_inception:
        prev = np.asarray(samples[ref_start:ref_end], dtype=float)
        if len(prev) == win:
            diff = segment - prev
            # x[n]-x[n-half] = 2*x[n] for a pure fundamental (half-cycle = pi
            # phase shift) -> divide raw estimate by 2, then rotate +180 deg
            # to undo the sign flip from that same shift.
            phasor = 2.0 * np.mean(diff * kernel) / 2.0
            return phasor * np.exp(1j * np.pi)

    segment = segment - float(np.mean(segment))
    return complex(2.0 * np.mean(segment * kernel))


def _window_thd_ratio(segment: np.ndarray, freq: float, sr: float, win: int) -> float:
    """Harmonic-to-fundamental RMS ratio for one analysis window.

    A clean 50/60 Hz current has ~0 harmonic content. CT saturation distorts
    the secondary current with flat-topping during the saturated portion of
    each cycle, which shows up as strong harmonic content (predominantly
    2nd/3rd/5th) riding on the fundamental. This ratio (analogous to THD, but
    computed directly from RMS energy rather than per-harmonic FFT bins,
    since the analysis window is exactly one cycle) is a standard,
    inexpensive proxy for saturation severity — verified numerically: clean
    sinusoid -> ~0, 90% hard-clip -> ~4%, 70% hard-clip -> ~14%.
    """
    if len(segment) != win or win < 4:
        return 0.0
    centered = segment - float(np.mean(segment))
    n = np.arange(win, dtype=float)
    kernel = np.exp(-1j * 2.0 * np.pi * freq * n / sr)
    fund = 2.0 * np.mean(centered * kernel)
    fund_rms = abs(fund) / np.sqrt(2.0)
    total_rms = float(np.sqrt(np.mean(centered ** 2)))
    if fund_rms <= 1e-9:
        return 0.0
    harmonic_rms_sq = max(total_rms ** 2 - fund_rms ** 2, 0.0)
    return float(np.sqrt(harmonic_rms_sq) / fund_rms)


def _smooth_locus_values(values: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small display smoother that preserves endpoints and broad shape."""
    if len(values) < 5:
        return values

    smoothed = values.astype(float).copy()
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel = kernel / np.sum(kernel)
    for _ in range(passes):
        padded = np.pad(smoothed, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed


def _despike_locus(r_arr: np.ndarray, x_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated one-window jumps with local medians."""
    if len(r_arr) < 7:
        return r_arr, x_arr

    r = r_arr.astype(float).copy()
    x = x_arr.astype(float).copy()
    step_mag = np.sqrt(np.diff(r) ** 2 + np.diff(x) ** 2)
    finite = step_mag[np.isfinite(step_mag)]
    if finite.size == 0:
        return r, x

    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    threshold = max(med + 8.0 * mad, med * 6.0, 1.0)

    for idx in range(1, len(r) - 1):
        jump_in = step_mag[idx - 1]
        jump_out = step_mag[idx] if idx < len(step_mag) else 0.0
        neighbor_gap = float(np.hypot(r[idx + 1] - r[idx - 1], x[idx + 1] - x[idx - 1]))
        if jump_in > threshold and jump_out > threshold and neighbor_gap < threshold:
            r[idx] = float(np.median(r[idx - 1:idx + 2]))
            x[idx] = float(np.median(x[idx - 1:idx + 2]))

    return r, x


def _compute_locus(
    comtrade_data: dict,
    loop: str,
    k0: float = 0.0,
    k0_angle_deg: float = 0.0,
    invert_i: bool = False,
    ct_ratio_override: Optional[float] = None,
    vt_ratio_override: Optional[float] = None,
) -> tuple[list[dict], dict]:
    """Returns (points, diagnostics).

    diagnostics carries data-quality signals a credible locus needs to
    disclose rather than silently smooth over: how many analysis windows
    were evaluated vs kept after outlier/validity filtering
    (`windows_evaluated`, `windows_kept`, `retention_pct` — a low retention
    means most of the record's Z estimates were unreliable and the plotted
    trajectory rests on a small fraction of the data), whether CT saturation
    was detected in the fault-current waveform (`ct_saturation_suspected`,
    `ct_saturation_thd_ratio`), and whether the half-cycle-differencing
    DC-offset rejection was active (`dc_offset_corrected` — False only for
    the pre-computed-complex-samples code path, which bypasses windowed DFT
    entirely).
    """
    comtrade_data = scope_payload(comtrade_data)  # multi-line DFR record -> its disturbed line only
    channels = comtrade_data["analog_channels"]
    time = np.array(comtrade_data["time"])
    mapping = LOOP_CHANNELS.get(loop, LOOP_CHANNELS["ZA"])
    voltage_scale = _voltage_to_volts_scale(channels)
    inception_idx, _timing_source, _timing_confidence = (
        _canonical_inception_idx(comtrade_data, time) if len(time) >= 4 else (None, "", 0.0)
    )

    if ct_ratio_override is not None and vt_ratio_override is not None and vt_ratio_override > 0:
        secondary_scale = ct_ratio_override / vt_ratio_override
    else:
        secondary_scale = _secondary_impedance_scale(channels)

    # See _detect_active_line_tag docstring: an external DFR CFG can record
    # two lines side by side (e.g. "IR BRINGIN 1" / "IR BRINGIN 2"). Compute
    # once per record and thread through every channel lookup below so the
    # locus is always built from the line that actually faulted.
    active_tag = _detect_active_line_tag(channels)

    v = _find_voltage_for_loop(channels, mapping, active_tag)
    if v is None:
        raise HTTPException(status_code=422, detail=f"Could not find voltage channel for loop {loop}")

    i_channels = []
    for candidate in mapping["i"]:
        current = _find_channel(channels, [candidate], active_tag)
        if current is None and candidate.startswith("I") and len(candidate) >= 2:
            current = _find_phase_current(channels, candidate[-1], active_tag)
        i_channels.append(current)
    i_channels = [c for c in i_channels if c is not None]
    if not i_channels:
        raise HTTPException(status_code=422, detail=f"Could not find current channel(s) for loop {loop}")

    if mapping.get("diff") and len(i_channels) == 2:
        i = i_channels[0] - i_channels[1]
    else:
        i = i_channels[0]

    if invert_i:
        i = -i

    if np.iscomplexobj(v) or np.iscomplexobj(i):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(np.abs(i) > 0.01, (v * voltage_scale / i) * secondary_scale, np.nan + 1j * np.nan)
        r = np.real(z)
        x = np.imag(z)
    else:
        freq = float(comtrade_data.get("frequency", 50.0))
        sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
        win = max(1, int(sr / freq))  # one cycle window
        step = max(1, win // 16)
        i_max_global = float(np.max(np.abs(i))) if len(i) > 0 else 1.0
        min_i = max(0.01, i_max_global * 0.01)
        k0_complex = k0 * np.exp(1j * np.radians(k0_angle_deg))
        i_n = None
        if loop in ("ZA", "ZB", "ZC") and k0 != 0.0:
            i_n = _find_channel(channels, ["IN", "I0", "3I0", "IRESIDUAL", "IEN", "IE", "IR"], active_tag)
            if i_n is None:
                ia = _find_phase_current(channels, "A", active_tag)
                ib = _find_phase_current(channels, "B", active_tag)
                ic = _find_phase_current(channels, "C", active_tag)
                if ia is not None and ib is not None and ic is not None:
                    i_n = ia + ib + ic
        r_list, x_list, t_list, thd_list = [], [], [], []
        windows_evaluated = 0
        for k in range(win - 1, len(time), step):
            s = k - win + 1
            v_w = v[s:k+1]
            i_w = i[s:k+1]
            if len(v_w) < 2 or np.max(np.abs(i_w)) < min_i:
                continue
            windows_evaluated += 1
            v_ph = _fundamental_phasor(v * voltage_scale, s, win, freq, sr, inception_idx)
            i_ph = _fundamental_phasor(i, s, win, freq, sr, inception_idx)
            if i_n is not None and len(i_n) == len(i):
                i_ph = i_ph + k0_complex * _fundamental_phasor(i_n, s, win, freq, sr, inception_idx)
            if not np.isfinite(abs(v_ph)) or not np.isfinite(abs(i_ph)) or abs(i_ph) < min_i:
                continue
            z = (v_ph / i_ph) * secondary_scale
            r_list.append(float(np.real(z)))
            x_list.append(float(np.imag(z)))
            t_list.append(float(time[k]))
            # Only meaningful once the fault current is actually flowing —
            # pre-fault load current has no saturation to speak of, and
            # scoring it would dilute the fault-window saturation signal.
            if inception_idx is None or k >= inception_idx:
                thd_list.append(_window_thd_ratio(i[s:k + 1], freq, sr, win))

        windows_kept_after_min_i = len(r_list)

        # IQR outlier removal on |Z| — drops wild inception transients
        # without killing valid fault points the way R² filtering does.
        if r_list:
            r_arr = np.array(r_list)
            x_arr = np.array(x_list)
            z_mag = np.sqrt(r_arr ** 2 + x_arr ** 2)
            q25, q75 = np.percentile(z_mag, 25), np.percentile(z_mag, 75)
            iqr = q75 - q25
            fence = q75 + 4.0 * iqr  # 4× IQR keeps most valid points
            mask = z_mag <= fence
            r_arr, x_arr = r_arr[mask], x_arr[mask]
            time = np.array(t_list)[mask]
            r_arr, x_arr = _despike_locus(r_arr, x_arr)
            r_arr = _smooth_locus_values(r_arr)
            x_arr = _smooth_locus_values(x_arr)
        else:
            r_arr = x_arr = np.array([])
            time = np.array([])
        r, x = r_arr, x_arr

    points = []
    for k in range(len(time)):
        rv, xv = float(r[k]), float(x[k])
        if np.isnan(rv) or np.isnan(xv):
            continue
        if abs(rv) > 1_000_000 or abs(xv) > 1_000_000:
            continue
        points.append({"t": float(time[k]), "r": rv, "x": xv})

    if np.iscomplexobj(v) or np.iscomplexobj(i):
        # Pre-computed complex COMTRADE (e.g. synthetic/replay sources) skip
        # the DFT windowing path entirely, so none of the per-window
        # diagnostics below apply.
        diagnostics = {
            "windows_evaluated": len(points),
            "windows_kept": len(points),
            "retention_pct": 100.0 if points else 0.0,
            "ct_saturation_suspected": False,
            "ct_saturation_thd_ratio": 0.0,
            "dc_offset_corrected": False,
        }
    else:
        retention_pct = (
            round(100.0 * len(points) / windows_kept_after_min_i, 1)
            if windows_kept_after_min_i > 0 else 0.0
        )
        thd_arr = np.array(thd_list) if thd_list else np.array([])
        # Threshold chosen from numerical calibration: a clean sinusoid scores
        # ~0, a mild (90%) hard-clip ~4%, a severe (70%) hard-clip ~14%. 6%
        # sits above normal fault-current harmonic content (CT/VT transients,
        # non-linear loads) but below the mild-saturation case, erring toward
        # not crying wolf on ordinary fault waveforms.
        sat_fraction = float(np.mean(thd_arr > 0.06)) if len(thd_arr) > 0 else 0.0
        diagnostics = {
            "windows_evaluated": windows_evaluated,
            "windows_kept": len(points),
            "retention_pct": retention_pct,
            # Flag only when a meaningful share of fault-window samples show
            # high harmonic content — a single noisy window shouldn't trigger
            # a saturation warning on an otherwise clean record.
            "ct_saturation_suspected": bool(sat_fraction > 0.2),
            "ct_saturation_thd_ratio": round(float(np.max(thd_arr)), 4) if len(thd_arr) > 0 else 0.0,
            "dc_offset_corrected": True,
        }

    return points, diagnostics


def _extract_features_from_payload(payload: dict) -> dict:
    """Auto-extract fault analysis features from a stored COMTRADE payload."""
    payload = scope_payload(payload)
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    freq = float(payload.get("frequency", 50.0))

    empty = {
        "fault_inception_angle_deg": 0.0,
        "fault_duration_ms": 0.0,
        "prefault_load_a": 0.0,
        "impedance_at_trip_ohm": 0.0,
        "waveform_asymmetry": 0.0,
        "dc_offset": 0.0,
        "ar_result": None,
    }

    if len(time) < 4:
        return empty

    sr = 1.0 / (time[1] - time[0])
    cycle_n = max(4, int(sr / freq))

    # See _detect_active_line_tag: an external DFR CFG can record two lines
    # side by side. Restrict the candidate search to the line with the
    # actual fault current before picking the highest-peak phase within it —
    # otherwise a healthy line's IA could still edge out a faulted line's IB
    # by name-match order alone.
    active_tag = _detect_active_line_tag(channels)

    # Pick the phase with the highest peak current — fault may be on B or C only
    candidates = [_find_channel(channels, [n], active_tag) for n in ["IA", "IL1", "I1", "IB", "IL2", "IC", "IL3"]]
    candidates = [c for c in candidates if c is not None]
    i = max(candidates, key=lambda arr: float(np.max(np.abs(arr)))) if candidates else None
    v = _find_channel(channels, ["VA", "VAN", "UA", "VB", "VBN", "VC", "VCN"], active_tag)
    if i is None:
        return empty

    pre_end = min(2 * cycle_n, len(i) // 4)
    pre_rms = float(np.sqrt(np.mean(i[:pre_end] ** 2))) if pre_end > 1 else 0.0
    inception_idx, _timing_source, _timing_confidence = _canonical_inception_idx(payload, time)
    shared = extract_ml_features(payload, "21")
    fault_duration_ms = shared["fault_duration_ms"]
    fia_deg = shared["inception_angle_degrees"]

    # DC offset and asymmetry from first fault cycle
    fw = i[inception_idx : inception_idx + cycle_n]
    dc_offset, asymmetry = 0.0, 0.0
    if len(fw) > 4:
        dc_component = float(np.mean(fw))
        ac_amp = float(np.sqrt(2) * np.sqrt(np.mean(fw ** 2)))
        dc_offset = float(np.clip(dc_component / ac_amp, -1.0, 1.0)) if ac_amp > 0 else 0.0
        pos, neg = float(np.max(fw)), float(np.min(fw))
        denom = pos + abs(neg)
        asymmetry = abs(pos - abs(neg)) / denom if denom > 0 else 0.0

    # |Z| at inception
    impedance_ohm = 0.0
    if v is not None and inception_idx < len(v) and abs(i[inception_idx]) > 0.01:
        impedance_ohm = float(abs(v[inception_idx] / i[inception_idx]))

    # AR result from binary channels. Use the same tightened sequence logic as
    # the ML conclusion so A/R command bits are not mistaken for breaker success.
    digital = _digital_sequence_features(payload.get("status_channels", []), time, inception_idx)
    ar_ok = digital.get("digital_ar_status")
    ar_result = "successful" if ar_ok is True else ("failed" if ar_ok is False else None)

    return {
        "fault_inception_angle_deg": round(fia_deg, 1),
        "fault_duration_ms": round(max(fault_duration_ms, 0.0), 1),
        "prefault_load_a": round(pre_rms, 2),
        "impedance_at_trip_ohm": round(impedance_ohm, 3),
        "waveform_asymmetry": round(asymmetry, 3),
        "dc_offset": round(dc_offset, 3),
        "ar_result": ar_result,
    }


def _compute_electrical_params(payload: dict) -> dict:
    """Compute extended electrical parameters for the workspace panel."""
    payload = scope_payload(payload)
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    freq = float(payload.get("frequency", 50.0))

    # See _detect_active_line_tag docstring. Without this, an external DFR
    # recording two lines in one CFG (e.g. "IR BRINGIN 1" healthy, "IR
    # BRINGIN 2" faulted) silently reports I peak / V sag / the whole
    # electrical summary and PDF report for whichever line's channels
    # happen to appear first in the CFG — regardless of which line actually
    # faulted.
    active_tag = _detect_active_line_tag(channels)

    ia = _find_phase_current(channels, "A", active_tag)
    ib = _find_phase_current(channels, "B", active_tag)
    ic = _find_phase_current(channels, "C", active_tag)
    va = _find_phase_voltage(channels, "A", active_tag)
    vb = _find_phase_voltage(channels, "B", active_tag)
    vc = _find_phase_voltage(channels, "C", active_tag)

    result: dict = {}
    if len(time) < 4:
        return result

    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else 1000.0
    cycle_n = max(4, int(sr / freq))

    # Use the phase with the highest peak fault current as reference — avoids
    # misdetection when the fault is on B or C and IA barely rises.
    available = [arr for arr in (ia, ib, ic) if arr is not None]
    i_ref = max(available, key=lambda arr: float(np.max(np.abs(arr)))) if available else None

    pre_end = min(2 * cycle_n, len(i_ref) // 4) if i_ref is not None else 0
    extinction_idx = len(time) - 1

    # Canonical inception — single source of truth shared with extract-features,
    # locus-events, full-soe, and AI feature extraction.
    inception_idx, timing_source, timing_confidence = _canonical_inception_idx(payload, time)

    if i_ref is not None and pre_end > 1:
        threshold = max(
            float(np.sqrt(np.mean(i_ref[:pre_end] ** 2))),
            np.max(np.abs(i_ref)) * 0.05,
            0.05,
        )
        for idx in range(inception_idx + cycle_n, len(i_ref)):
            start = max(0, idx - cycle_n // 2)
            if float(np.sqrt(np.mean(i_ref[start : idx + 1] ** 2))) < threshold * 0.6:
                extinction_idx = idx
                break

    canonical_window = build_event_window(payload)
    if canonical_window.clearing_idx is not None:
        extinction_idx = canonical_window.clearing_idx

    fault_slice = slice(inception_idx, min(extinction_idx + 1, len(time)))

    for label, arr in [("IA", ia), ("IB", ib), ("IC", ic)]:
        if arr is not None and len(arr) > inception_idx:
            result[f"i_peak_{label.lower()}_a"] = round(float(np.max(np.abs(arr[fault_slice]))), 2)

    if va is not None and pre_end > 1:
        v_pre_rms = float(np.sqrt(np.mean(va[:pre_end] ** 2)))
        v_fault_rms = float(np.sqrt(np.mean(va[fault_slice] ** 2))) if len(va) > inception_idx else v_pre_rms
        # v_sag_pct = (1 - v_fault/v_pre) * 100 divides by v_pre_rms, so a
        # near-zero prefault window (not just exactly zero) still blows the
        # ratio up to an absurd magnitude (e.g. -143255%). This is a REAL
        # recorded condition here, not sensor noise: a line captured
        # mid-energization (breaker closing onto the line moments before the
        # record's prefault window) legitimately has ~0V before it ramps to
        # nominal — verified against a real Qualitrol external-DFR record
        # where VA read ~7% of nominal for the first ~40ms then settled at
        # the nominal ~86.4kV a few ms after inception. 7% is itself already
        # far below any plausible pre-fault system voltage (even a deep
        # upstream sag rarely exceeds ~50-60% dip), so use 20% of nominal
        # phase voltage as the "this isn't a real pre-fault baseline"
        # threshold rather than trying to guess a tighter one.
        va_channel = next((ch for ch in channels if ch is not None and ch.get("measurement") == "voltage"
                            and _extract_line_tag(ch.get("name") or "") == active_tag), None)
        vt_primary = float((va_channel or {}).get("ct_primary") or 0.0)
        nominal_phase_v = (vt_primary / np.sqrt(3)) / 1000.0 if vt_primary > 0 else None  # kV, matches va's unit
        pre_energization = nominal_phase_v is not None and v_pre_rms < nominal_phase_v * 0.2
        if pre_energization:
            result["v_sag_pct"] = None
            result["v_sag_note"] = (
                "Prefault voltage window reads near-zero relative to nominal — this record likely captures "
                "the line being energized (breaker closing) rather than a steady pre-fault condition. "
                "Voltage-sag percentage is not meaningful here."
            )
        elif v_pre_rms > 0:
            result["v_sag_pct"] = round((1.0 - v_fault_rms / v_pre_rms) * 100, 1)

    a_op = np.exp(1j * 2 * np.pi / 3)
    a2_op = np.exp(-1j * 2 * np.pi / 3)

    def rms_phasor(arr, idx, n):
        if arr is None or idx + n > len(arr):
            return None
        seg = arr[idx : idx + n]
        t_seg = np.arange(n) / sr
        cos_ref = np.cos(2 * np.pi * freq * t_seg)
        sin_ref = np.sin(2 * np.pi * freq * t_seg)
        re = 2 * np.mean(seg * cos_ref)
        im = -2 * np.mean(seg * sin_ref)
        return complex(re, im) / np.sqrt(2)

    p_i_a = rms_phasor(ia, inception_idx, cycle_n)
    p_i_b = rms_phasor(ib, inception_idx, cycle_n)
    p_i_c = rms_phasor(ic, inception_idx, cycle_n)

    if p_i_a is not None and p_i_b is not None and p_i_c is not None:
        i_zero = (p_i_a + p_i_b + p_i_c) / 3
        i_pos = (p_i_a + a_op * p_i_b + a2_op * p_i_c) / 3
        i_neg = (p_i_a + a2_op * p_i_b + a_op * p_i_c) / 3
        result["i_pos_seq_a"] = round(abs(i_pos), 2)
        result["i_neg_seq_a"] = round(abs(i_neg), 2)
        result["i_zero_seq_a"] = round(abs(i_zero), 2)

    if va is not None and ia is not None:
        win = min(cycle_n, len(va) - inception_idx)
        if win >= 4:
            v_w = va[inception_idx : inception_idx + win] * 1000.0
            i_w = ia[inception_idx : inception_idx + win]
            i_90 = np.gradient(i_w) / (2 * np.pi * freq / sr)
            matrix = np.column_stack([i_w, i_90])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(matrix, v_w, rcond=None)
                r_val = float(coeffs[0])
                x_val = float(coeffs[1])
                result["r_at_fault_ohm"] = round(r_val, 2)
                result["x_at_fault_ohm"] = round(x_val, 2)
                if x_val != 0:
                    result["rx_ratio"] = round(r_val / x_val, 3)
                z_mag = float(np.sqrt(r_val ** 2 + x_val ** 2))
                result["z_at_inception_ohm"] = round(z_mag, 2)
                if z_mag > 0:
                    result["z_angle_deg"] = round(float(np.degrees(np.arctan2(x_val, r_val))), 1)
            except Exception:
                pass

    z_min_values = []
    for v_phase, i_phase in ((va, ia), (vb, ib), (vc, ic)):
        if v_phase is None or i_phase is None:
            continue
        limit = min(len(v_phase), len(i_phase), len(time))
        if limit <= pre_end + 1:
            continue
        i_abs = np.abs(i_phase[:limit])
        pre_i = i_abs[:pre_end] if pre_end > 1 else i_abs[: max(1, limit // 10)]
        current_gate = max(float(np.sqrt(np.mean(pre_i ** 2))) * 2.0, float(np.max(i_abs)) * 0.05, 0.1)
        valid = i_abs[pre_end:limit] > current_gate
        if not np.any(valid):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            z_vals = np.abs((v_phase[pre_end:limit] * 1000.0) / i_phase[pre_end:limit])
        z_vals = z_vals[np.isfinite(z_vals) & valid]
        if z_vals.size:
            z_min_values.append(float(np.min(z_vals)))
    if z_min_values:
        result["z_min_ohm"] = round(float(np.min(z_min_values)), 2)
    elif "z_at_inception_ohm" in result:
        result["z_min_ohm"] = result["z_at_inception_ohm"]

    digital = _digital_sequence_features(payload.get("status_channels", []), time, inception_idx)
    ar_dead_ms = digital.get("digital_ar_dead_time_ms")
    if ar_dead_ms is not None:
        result["ar_dead_time_ms"] = ar_dead_ms

    trip_time_ms = digital.get("digital_first_trip_ms")
    trip_time_source = "soe" if trip_time_ms is not None else None
    if trip_time_ms is None:
        start_idx = max(0, inception_idx - 2)
        for sch in payload.get("status_channels", []):
            name = str(sch.get("name", "") or "")
            if not _is_operate_status(name):
                continue
            samples = sch.get("samples", [])
            n = min(len(samples), len(time))
            if n == 0:
                continue
            prev = int(samples[start_idx - 1]) if start_idx > 0 and start_idx < n else 0
            for idx in range(start_idx, n):
                val = int(samples[idx])
                if prev == 0 and val == 1:
                    candidate_ms = float(time[idx] * 1000)
                    trip_time_ms = candidate_ms if trip_time_ms is None else min(trip_time_ms, candidate_ms)
                    trip_time_source = "status_edge"
                    break
                prev = val
    if trip_time_ms is not None:
        result["trip_time_ms"] = round(trip_time_ms, 1)
        result["trip_time_source"] = trip_time_source or "status_edge"

    result["fault_duration_ms"] = round((time[extinction_idx] - time[inception_idx]) * 1000, 1)
    result["inception_time_ms"] = round(float(time[inception_idx]) * 1000, 1)
    result["timing_source"] = timing_source
    result["timing_confidence"] = round(timing_confidence, 3)

    # Shared no-fault gate: if no real fault is present, the fault-window
    # parameters above are computed from load V/I and are misleading. Flag it
    # and blank them so the UI and PDF render a clean "no fault" state instead.
    det = detect_fault_presence(payload)
    if det.no_fault:
        return {
            "no_fault": True,
            "no_fault_reasons": det.reasons,
            "peak_to_prefault_ratio": det.peak_to_prefault_ratio,
            "voltage_sag_pu": det.voltage_sag_pu,
            "i0_i1_ratio": det.i0_i1_ratio,
            "i2_i1_ratio": det.i2_i1_ratio,
        }
    result["no_fault"] = False
    return result


def _evidence_based_fault_phases(payload: dict, row: dict) -> tuple[list[str], bool]:
    """Combined-evidence fault-phase determination — replaces a single
    waveform-amplitude threshold (which can't tell a genuine multi-phase
    fault apart from one healthy phase's mutual-coupling current, e.g. a
    real Kebumen-Gombong SLG-R record whose small IC perturbation used to
    get reported as a second faulted phase, C).

    Ground involvement (``to_ground``) is decided from 3I0/I1 alone — that
    ratio genuinely does distinguish "ground is/isn't part of this fault"
    (confirmed: an SLG and a DLG both show high 3I0; an LL fault does not).
    It does NOT and must not decide WHICH phase(s) are involved — that
    needs per-phase evidence, combined:

    - Digital trip/pole-open phase(s) (``digital_trip_phases`` /
      ``digital_cb_open_phases``): the relay's own operate decision — the
      strongest available evidence when present.
    - Per-phase ground-loop impedance magnitude (|V_phase / (I_phase +
      K0*3I0)|, K0=1 as a rough discriminator, not a calibrated relay
      setting): the loop actually seeing the fault has a small |Z|; a
      healthy phase's loop, even with some mutual-coupling current, sits
      far outside — this is the loop-comparison the source analysis
      argued for (Z_RG vs Z_TG), reusing the exact V/I phasor machinery
      _compute_locus already uses rather than re-deriving fault impedance
      from scratch.
    - Reclose evidence (single-pole AR success on a specific phase) is
      used ONLY as an optional corroborating boost, never as a
      requirement — a record can be truncated before reclose completes
      (or have no AR at all) and still be a perfectly clear single-phase
      fault; treating "no reclose evidence" as evidence AGAINST a phase
      would penalize truncated-but-otherwise-clear records for a gap in
      the recording, not a gap in the fault. Mirrors the same absence-is-
      not-evidence stance core/fault_detector.py already takes for
      truncated AR sequences (returns None, not False).

    Falls back to the plain per-phase amplitude threshold (previous
    behavior) when there isn't enough of the above evidence to do better
    (e.g. no voltage channels, single-line CFG without a residual current
    channel or computable 3I0) — never raises, never silently guesses past
    what the record actually supports.
    """
    payload = scope_payload(payload)
    channels = payload.get("analog_channels", [])
    time = np.array(payload.get("time", []))
    freq = float(payload.get("frequency", 50.0))
    if len(time) < 4:
        phases_str = row.get("faulted_phases", "") or "A"
        return [p for p in phases_str.split("+") if p], bool(row.get("is_ground_fault", False))

    to_ground = bool(row.get("is_ground_fault", False))

    active_tag = _detect_active_line_tag(channels)
    inception_idx, _src, _conf = _canonical_inception_idx(payload, time)
    sr = 1.0 / (time[1] - time[0]) if len(time) > 1 else freq * 20.0
    win = max(1, int(round(sr / freq)))
    voltage_scale = _voltage_to_volts_scale(channels)

    va = _find_phase_voltage(channels, "A", active_tag)
    vb = _find_phase_voltage(channels, "B", active_tag)
    vc = _find_phase_voltage(channels, "C", active_tag)
    ia = _find_phase_current(channels, "A", active_tag)
    ib = _find_phase_current(channels, "B", active_tag)
    ic = _find_phase_current(channels, "C", active_tag)
    i_n = _find_channel(channels, ["IN", "I0", "3I0", "IRESIDUAL", "IEN", "IE", "IR"], active_tag)
    if i_n is None and ia is not None and ib is not None and ic is not None:
        i_n = ia + ib + ic

    have_voltage = va is not None and vb is not None and vc is not None
    have_current = ia is not None and ib is not None and ic is not None

    if not have_current:
        phases_str = row.get("faulted_phases", "") or "A"
        return [p for p in phases_str.split("+") if p], to_ground

    # Evaluate one cycle a short way into the fault — same convention as
    # _compute_electrical_params (inception + win), avoids the inception
    # transient itself.
    s = min(inception_idx + win, max(0, len(time) - win))
    k0 = complex(1.0, 0.0)  # rough discriminator only — not a calibrated relay K0 setting.

    def _loop_z(v_arr, i_arr) -> Optional[complex]:
        if v_arr is None or i_arr is None:
            return None
        v_ph = _fundamental_phasor(v_arr * voltage_scale, s, win, freq, sr, inception_idx)
        i_ph = _fundamental_phasor(i_arr, s, win, freq, sr, inception_idx)
        if i_n is not None and len(i_n) == len(i_arr):
            i_ph = i_ph + k0 * _fundamental_phasor(i_n, s, win, freq, sr, inception_idx)
        if not np.isfinite(abs(v_ph)) or not np.isfinite(abs(i_ph)) or abs(i_ph) < 1e-9:
            return None
        return v_ph / i_ph

    z_by_phase: dict[str, Optional[complex]] = {}
    if have_voltage:
        z_by_phase = {"A": _loop_z(va, ia), "B": _loop_z(vb, ib), "C": _loop_z(vc, ic)}

    peak_current = {
        phase: float(np.max(np.abs(arr[inception_idx: inception_idx + win])))
        for phase, arr in (("A", ia), ("B", ib), ("C", ic))
        if arr is not None and len(arr) > inception_idx
    }
    digital_trip = set(row.get("digital_trip_phases") or [])
    digital_cb_open = set(row.get("digital_cb_open_phases") or [])
    digital_cb_close = set(row.get("digital_cb_close_phases") or [])  # reclose corroboration only

    support: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
    STRONG, MODERATE, WEAK = 3.0, 1.5, 0.5

    for phase in "ABC":
        if phase in digital_trip:
            support[phase] += STRONG
        if phase in digital_cb_open:
            support[phase] += STRONG
        if phase in digital_cb_close:
            # Corroboration only — reclosing this phase means it was the one
            # that opened, reinforcing (never substituting for) the trip
            # evidence above. Absence of this (no AR, or record truncated
            # before reclose) intentionally contributes nothing either way.
            support[phase] += WEAK

    z_mags = {p: abs(z) for p, z in z_by_phase.items() if z is not None}
    if len(z_mags) >= 2:
        min_z = min(z_mags.values())
        for phase, mag in z_mags.items():
            # A loop within 2x the smallest |Z| is "also consistent with
            # seeing the fault" (matches the source analysis's "both ground
            # loops consistent -> DLG candidate" case); anything further out
            # reads as the healthy phase's mutual-coupling residue.
            if mag <= min_z * 2.0:
                support[phase] += STRONG if mag <= min_z * 1.2 else MODERATE

    if peak_current:
        max_i = max(peak_current.values())
        if max_i > 0:
            for phase, i_mag in peak_current.items():
                if i_mag >= max_i * 0.5:
                    support[phase] += MODERATE
                elif i_mag >= max_i * 0.15:
                    support[phase] += WEAK

    # A phase needs at least one piece of real (non-corroboration-only)
    # evidence to be included — pure reclose evidence with nothing else
    # backing a phase isn't enough on its own.
    has_primary_evidence = {
        phase: (phase in digital_trip or phase in digital_cb_open or z_mags.get(phase) is not None
                or phase in peak_current)
        for phase in "ABC"
    }
    threshold = STRONG  # requires at least one strong signal, or two moderate ones
    faulted = sorted(
        phase for phase in "ABC"
        if has_primary_evidence[phase] and support[phase] >= threshold
    )

    if not faulted:
        # Not enough combined evidence to improve on the plain threshold —
        # fall back rather than report nothing.
        phases_str = row.get("faulted_phases", "") or "A"
        return [p for p in phases_str.split("+") if p], to_ground

    return faulted, to_ground


def _compute_fault_classification(payload: dict) -> dict:
    """Derive fault type code, phases, zone, trip and timing for the Jenis Gangguan panel."""
    payload = scope_payload(payload)
    time = np.array(payload.get("time", []))
    empty = {
        "fault_code": "Unknown",
        "phases": [],
        "phases_label": "-",
        "to_ground": False,
        "trip_type": None,
        "zone": None,
        "prefault_ms": 0.0,
        "fault_ms": 0.0,
        "total_ms": 0.0,
        "ar_status": None,
    }
    if len(time) < 4:
        return empty

    # Shared no-fault gate: if no real fault is present, don't report a fault
    # type/phase/timing — that would contradict the AI panel and the physics.
    det = detect_fault_presence(payload)
    if det.no_fault:
        return {
            **empty,
            "fault_code": "NONE",
            "no_fault": True,
            "total_ms": round(float((time[-1] - time[0]) * 1000), 1),
        }

    row = extract_ml_features(payload, "21")
    window = build_event_window(payload)
    total_ms = round(float((time[-1] - time[0]) * 1000), 1)
    fault_ms = float(row.get("fault_duration_ms", 0) or 0)
    prefault_ms = max(0.0, round(total_ms - fault_ms, 1))

    # F4 is the shared reasoning chain used by the record, incident and report.
    # Import locally: fault_reasoning itself uses our phasor helpers.
    from ..record_analysis import build_record_analysis

    reasoning = build_record_analysis("fault-classification", payload).reasoning
    phase_conclusion = next(
        (c for c in reasoning.get("conclusions", []) if c.get("key") == "phases"), {}
    )
    phase_value = phase_conclusion.get("value") or {}
    phases = [p for p in phase_value.get("phases", []) if p in ("A", "B", "C")]
    to_ground = bool(phase_value.get("ground", False))
    n_phases = len(phases)

    if n_phases >= 3:
        fault_code = "3Ph"
    elif n_phases == 2 and to_ground:
        fault_code = "DLG"
    elif n_phases == 2:
        fault_code = "LL"
    elif n_phases == 1 and to_ground:
        fault_code = "SLG"
    else:
        fault_code = "SL" if n_phases == 1 else "Unknown"

    trip_type = row.get("trip_type") or None
    if trip_type == "unknown":
        trip_type = None
    zone = row.get("zone_operated") or None

    ar_ok = row.get("reclose_successful")
    ar_status = "successful" if ar_ok is True else ("failed" if ar_ok is False else None)
    phases_label = "+".join(phases) + ("-N" if to_ground and n_phases < 3 else "")

    return {
        "fault_code": fault_code,
        "phases": phases,
        "phases_label": phases_label if phases_label else "-",
        "to_ground": to_ground,
        "trip_type": trip_type,
        "zone": zone,
        "prefault_ms": prefault_ms,
        "fault_ms": fault_ms,
        "total_ms": total_ms,
        "ar_status": ar_status,
        "no_fault": False,
        "timing_source": window.method,
        "timing_confidence": round(window.confidence, 3),
    }


def _inception_idx_from_payload(payload: dict) -> tuple[np.ndarray, int]:
    """Canonical fault-inception index, shared with electrical-params,
    extract-features, and AI feature extraction so locus-event / SOE rel_ms
    always aligns with the inception marker shown on the waveform plot."""
    time = np.array(payload.get("time", []))
    if len(time) < 4:
        return time, 0
    inception_idx, _timing_source, _timing_confidence = _canonical_inception_idx(payload, time)
    return time, inception_idx


# Curated key-event classification. Only protection-relevant channels become
# locus markers — full SOE stays in the dedicated SOE table.
_LOCUS_EVENT_RULES = [
    # (category, short label, substring keywords, regex keywords) — first rule
    # whose substring OR regex matches the upper-cased channel name wins.
    ("zone",     "Zone",  ["ZONE 1", "ZONE1", "ZONE 2", "ZONE2", "ZONE 3", "ZONE3"],
                          [r"\bZ[123]\b"]),
    ("trip",     "Trip",  ["TRIP", "OPRT", "OPERATE", "TRIPPING"], []),
    ("comms",    "Chan",  ["CHAN RECV", "CHAN. RECV", "SIG. SEND", "SIG SEND",
                           "DIST. CHAN", "CARRIER", "POTT", "PUTT"], []),
    ("reclose",  "AR",    ["A/R", "AR ", "RECLOS", "A R ", "AUTORECLOSE"],
                          [r"\b79\b"]),
    ("breaker",  "CB",    ["CB AUX", "52-A", "52A", "52-B", "52B",
                           "CB OPEN", "CB CLOSE", "PMT", "BREAKER"], []),
]


def _classify_locus_event(name: str) -> Optional[tuple[str, str]]:
    import re as _re
    upper = name.upper()
    for category, label, keywords, patterns in _LOCUS_EVENT_RULES:
        if any(kw in upper for kw in keywords):
            return category, label
        if any(_re.search(p, upper) for p in patterns):
            return category, label
    return None


def _compute_full_soe_events(payload: dict) -> dict:
    """All digital channel state transitions, anchored to fault inception.

    Unlike _compute_locus_events (which only keeps protection-relevant
    channels for the impedance overlay), this emits every 0<->1 transition
    across every digital channel — used by the PDF report's SOE table.
    On a multi-line DFR record only the analysed line's (and common) channels
    are listed.
    """
    payload = scope_payload(payload)
    time, inception_idx = _inception_idx_from_payload(payload)
    if len(time) == 0:
        return {"inception_time_ms": None, "events": [], "timing_source": "insufficient_data", "timing_confidence": 0.0}

    window = build_event_window(payload)
    inception_s = float(time[inception_idx]) if inception_idx < len(time) else float(time[0])
    events: list[dict] = []

    for sch in payload.get("status_channels", []):
        name = str(sch.get("name", "") or "").strip()
        if not name:
            continue
        samples = sch.get("samples") or []
        n = min(len(samples), len(time))
        if n < 2:
            continue

        classified = _classify_locus_event(name)
        if classified is not None:
            category, label = classified
        else:
            category, label = "other", ""

        prev = int(samples[0])
        for idx in range(1, n):
            val = int(samples[idx])
            if val != prev:
                t_s = float(time[idx])
                events.append({
                    "time_ms": round(t_s * 1000.0, 2),
                    "rel_ms": round((t_s - inception_s) * 1000.0, 2),
                    "channel": name,
                    "state": val,
                    "category": category,
                    "label": label,
                })
            prev = val

    events.sort(key=lambda e: (e["time_ms"], e["channel"]))
    return {
        "inception_time_ms": round(inception_s * 1000.0, 2),
        "events": events,
        "timing_source": window.method,
        "timing_confidence": round(window.confidence, 3),
    }


def _compute_locus_events(payload: dict) -> dict:
    """Curated digital events with timestamps, anchored to fault inception."""
    payload = scope_payload(payload)
    time, inception_idx = _inception_idx_from_payload(payload)
    if len(time) == 0:
        return {"inception_time_ms": None, "events": [], "timing_source": "insufficient_data", "timing_confidence": 0.0}

    window = build_event_window(payload)
    inception_s = float(time[inception_idx]) if inception_idx < len(time) else float(time[0])
    events: list[dict] = []

    for sch in payload.get("status_channels", []):
        name = str(sch.get("name", "") or "")
        classified = _classify_locus_event(name)
        if classified is None:
            continue
        category, label = classified
        samples = sch.get("samples") or []
        n = min(len(samples), len(time))
        if n < 2:
            continue
        prev = int(samples[0])
        for idx in range(1, n):
            val = int(samples[idx])
            if val != prev:
                t_s = float(time[idx])
                events.append({
                    "time_ms": round(t_s * 1000.0, 2),
                    "rel_ms": round((t_s - inception_s) * 1000.0, 2),
                    "channel": name,
                    "state": val,
                    "category": category,
                    "label": label,
                })
            prev = val

    events.sort(key=lambda e: (e["time_ms"], e["channel"]))
    return {
        "inception_time_ms": round(inception_s * 1000.0, 2),
        "events": events,
        "timing_source": window.method,
        "timing_confidence": round(window.confidence, 3),
    }


@router.get("/fault-classification")
async def fault_classification(analysis_id: str):
    """Classify fault type, phases, zone and trip for the Jenis Gangguan panel."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_fault_classification, payload)
    return result


@router.get("/electrical-params")
async def electrical_params(analysis_id: str):
    """Compute extended electrical parameters for the fault analysis panel."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    params = await loop.run_in_executor(None, _compute_electrical_params, payload)
    return params


@router.get("/extract-features")
async def extract_features(analysis_id: str):
    """Auto-extract fault features from stored COMTRADE data."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    features = await loop.run_in_executor(None, _extract_features_from_payload, payload)
    return features


@router.get("/locus-events", response_model=LocusEventsResponse)
async def locus_events(analysis_id: str):
    """Curated key digital events (trip/zone/AR/CB/comms) with timestamps,
    used to overlay the protection sequence along the impedance locus."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_locus_events, payload)
    return result


@router.get("/full-soe", response_model=LocusEventsResponse)
async def full_soe(analysis_id: str):
    """All digital channel transitions for the PDF report SOE table."""
    payload = load_analysis(analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_full_soe_events, payload)
    return result


@router.post("/locus", response_model=LocusResponse)
async def compute_locus(body: LocusAnalysisRequest):
    """Compute impedance locus (R-X trajectory) for the selected loop."""
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    if detect_fault_presence(payload).no_fault:
        return LocusResponse(loop=body.loop, points=[], zones=body.zones, fault_inception_idx=None)

    loop = asyncio.get_event_loop()
    points, diagnostics = await loop.run_in_executor(
        None, partial(
            _compute_locus, payload, body.loop,
            body.k0, body.k0_angle_deg, body.invert_i,
            body.ct_ratio_override, body.vt_ratio_override,
        )
    )
    return LocusResponse(
        loop=body.loop,
        points=[LocusPoint(**p) for p in points],
        zones=body.zones,
        fault_inception_idx=None,
        diagnostics=LocusDiagnostics(**diagnostics),
    )


def _compute_locus_batch(
    payload: dict,
    loops: list[str],
    k0: float,
    k0_angle_deg: float,
    invert_i: bool,
    ct_ratio_override: Optional[float],
    vt_ratio_override: Optional[float],
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Compute every requested loop from a single already-loaded payload."""
    out: dict[str, list[dict]] = {}
    diagnostics_out: dict[str, dict] = {}
    for loop in loops:
        try:
            out[loop], diagnostics_out[loop] = _compute_locus(
                payload, loop, k0, k0_angle_deg, invert_i,
                ct_ratio_override, vt_ratio_override,
            )
        except HTTPException:
            # A loop whose voltage/current channel is absent shouldn't fail the
            # whole batch — just return no points for it.
            out[loop] = []
            diagnostics_out[loop] = {}
    return out, diagnostics_out


@router.post("/locus-batch", response_model=LocusBatchResponse)
async def compute_locus_batch(body: LocusBatchRequest):
    """Compute all requested loops in one request. Loads/parses the stored
    COMTRADE payload once instead of once per loop — avoids the 6× redundant
    load that pushed large records past the client timeout."""
    payload = load_analysis(body.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    if detect_fault_presence(payload).no_fault:
        return LocusBatchResponse(points_by_loop={loop_name: [] for loop_name in body.loops})

    loop = asyncio.get_event_loop()
    points_by_loop, diagnostics_by_loop = await loop.run_in_executor(
        None, partial(
            _compute_locus_batch, payload, body.loops,
            body.k0, body.k0_angle_deg, body.invert_i,
            body.ct_ratio_override, body.vt_ratio_override,
        )
    )
    return LocusBatchResponse(
        points_by_loop={
            loop_name: [LocusPoint(**p) for p in pts]
            for loop_name, pts in points_by_loop.items()
        },
        diagnostics_by_loop={
            loop_name: LocusDiagnostics(**d) if d else LocusDiagnostics()
            for loop_name, d in diagnostics_by_loop.items()
        },
    )


@router.post("/ai-analysis", response_model=AIFaultResult)
async def ai_fault_analysis(features: AIFaultFeatures):
    """Run LightGBM fault cause analysis for relay 21 (distance protection)."""
    if not features.analysis_id:
        raise HTTPException(status_code=422, detail="analysis_id is required for AI analysis.")
    payload = load_analysis(features.analysis_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Analysis session not found or expired.")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_ml_prediction, payload, "21")
    return AIFaultResult(**result)
