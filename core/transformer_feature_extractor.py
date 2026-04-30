"""
Transformer Feature Extractor
==============================
Extracts features from transformer COMTRADE recordings for the 87T event
classifier (Inrush / InternalFault / ThroughFault / Overexcitation / MalOperate).

Key features (derived from IEEE C37.91-2008, IEC 60255-111, relay manuals):

  Harmonic analysis (FFT on a 2-cycle window post-fault-inception):
    - 2nd harmonic ratio (I2f / I1f %) per phase — primary inrush indicator
    - 5th harmonic ratio (I5f / I1f %) per phase — overexcitation indicator
    - THD of differential current

  Differential / restraint:
    - |Idiff| / |Irestraint| ratio → slope characteristic point
    - Max differential magnitude (% of rated)
    - Differential vs restraint slope comparison against 20% / 40% dual-slope

  Waveform shape:
    - DC-offset index (asymmetry) — strong inrush indicator
    - Peak-to-peak asymmetry ratio (Ip_pos / Ip_neg)
    - Zero-crossing interval variance — inrush waveforms miss crossings

  Current balance:
    - HV vs LV current magnitude ratio (CT-ratio corrected)
    - Phase-angle difference between HV and LV (should be ~180° normal)

  Context:
    - Energisation flag (inferred from pre-fault current = 0 on one winding)
    - Fault duration, inception angle, pre-fault MVA loading estimate

References used for threshold table (no data — knowledge-based):
  - ABB Application Guide 1MRK504049: 2nd harmonic > 15–20% → inrush
  - Siemens 7UT8 Manual: 5th harmonic > 10% → overexcitation
  - IEEE C37.91-2008 §6.3: slope char 20% / 80%
  - Horowitz & Phadke, "Power System Relaying" 4th ed. §9
  - PLN SPLN D3.012-1:2012 transformer protection requirements
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Fundamental frequency (Hz) — PLN system
SYSTEM_FREQ_HZ = 50.0

# Harmonic thresholds (knowledge-based, no field data)
H2_INRUSH_THRESHOLD_PCT    = 15.0   # 2nd harmonic ≥ 15% → inrush likely
H5_OVEREXCIT_THRESHOLD_PCT = 10.0   # 5th harmonic ≥ 10% → overexcitation likely
SLOPE1_PCT                 = 20.0   # Dual-slope first knee (typical relay setting)
SLOPE2_PCT                 = 80.0   # Dual-slope saturation knee
IDIFF_OPERATE_PU           = 0.20   # Minimum operate threshold (20% of tap)
ASYMMETRY_INRUSH_THRESHOLD = 0.35   # DC-offset asymmetry index ≥ 0.35 → inrush

# Standard transformer ratings (MVA) commonly used in Indonesia (incl. IBT)
STANDARD_MVA_RATINGS = [30, 60, 100, 150, 200, 250, 300, 315, 500]


# ─────────────────────────────────────────────────────────────────────────────
# Feature dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerFeatures:
    """
    All features extracted for transformer event classification.
    None = channel not available or computation failed.
    """

    # ── Harmonic ratios (per-phase, averaged if multi-phase) ─────────────────
    h2_ratio_a_pct: Optional[float] = None   # 2nd harmonic / fundamental × 100, phase A
    h2_ratio_b_pct: Optional[float] = None
    h2_ratio_c_pct: Optional[float] = None
    h2_ratio_max_pct: Optional[float] = None  # max across phases

    h5_ratio_a_pct: Optional[float] = None   # 5th harmonic / fundamental × 100, phase A
    h5_ratio_b_pct: Optional[float] = None
    h5_ratio_c_pct: Optional[float] = None
    h5_ratio_max_pct: Optional[float] = None

    # THD of differential current (first 2 cycles)
    thd_diff_pct: Optional[float] = None

    # ── Differential / restraint features ────────────────────────────────────
    idiff_max_a_pu: Optional[float] = None   # |Idiff_A| in per-unit of rated
    idiff_max_b_pu: Optional[float] = None
    idiff_max_c_pu: Optional[float] = None
    idiff_max_pu: Optional[float] = None     # worst phase

    irstr_max_a_pu: Optional[float] = None
    irstr_max_b_pu: Optional[float] = None
    irstr_max_c_pu: Optional[float] = None
    irstr_max_pu: Optional[float] = None

    slope_worst_pct: Optional[float] = None  # |Idiff| / |Irestraint| × 100 worst phase
    above_slope1:    Optional[bool]  = None  # True if Idiff > SLOPE1% × Irstr
    above_slope2:    Optional[bool]  = None  # True if Idiff > SLOPE2% × Irstr

    # ── Waveform shape ────────────────────────────────────────────────────────
    dc_offset_index_a: Optional[float] = None   # Asymmetry index 0-1 (0=symmetric)
    dc_offset_index_b: Optional[float] = None
    dc_offset_index_c: Optional[float] = None
    dc_offset_index_max: Optional[float] = None

    # Peak-to-peak asymmetry: (|Ip_pos| - |Ip_neg|) / max(|Ip_pos|, |Ip_neg|)
    pp_asymmetry_a: Optional[float] = None
    pp_asymmetry_b: Optional[float] = None
    pp_asymmetry_c: Optional[float] = None

    # Zero-crossing interval variance (normalised)
    zc_interval_variance: Optional[float] = None

    # ── Current balance (HV vs LV) ────────────────────────────────────────────
    hv_lv_ratio_a: Optional[float] = None    # |I_HV_A| / |I_LV_A| (should ≈ 1 if CTs normalised)
    hv_lv_phase_diff_a_deg: Optional[float] = None  # phase-angle diff HV vs LV (≈180° normal)

    # ── Energisation context ──────────────────────────────────────────────────
    energisation_flag: bool = False   # LV pre-fault current ≈ 0 (energising from HV)
    lv_prefault_irms_pu: Optional[float] = None  # pre-fault LV RMS in per-unit
    estimated_mva: Optional[float] = None        # snapped to standard rating list
    hv_base_current_a: Optional[float] = None   # rated HV current used as pu base
    lv_base_current_a: Optional[float] = None   # rated LV current from MVA estimate

    # ── Timing / general ─────────────────────────────────────────────────────
    inception_angle_deg: Optional[float] = None  # voltage angle at event inception
    fault_duration_ms: Optional[float] = None
    peak_idiff_a: Optional[float] = None    # absolute peak primary amps
    sampling_rate_hz: float = 0.0
    n_cycles_analysed: float = 2.0

    # ── Meta ──────────────────────────────────────────────────────────────────
    station_name: str = ""
    relay_family: str = "UNKNOWN"
    channels_available: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_transformer_features(
    record,
    ch_map,          # TransformerChannelMap
    fault_event=None,   # FaultEvent from fault_detector (optional)
    rated_current_a: float = 0.0,   # rated primary current (A); 0 = auto-estimate
) -> TransformerFeatures:
    """
    Extract all transformer-specific features from a ComtradeRecord.

    Args:
        record:          ComtradeRecord (parsed COMTRADE file)
        ch_map:          TransformerChannelMap (from transformer_channel_mapper)
        fault_event:     FaultEvent if available (provides inception index)
        rated_current_a: Rated primary current for per-unit base.
                         If 0, estimated from pre-fault RMS.

    Returns:
        TransformerFeatures dataclass
    """
    from core.transformer_channel_mapper import get_mapped_samples

    features = TransformerFeatures()
    features.station_name = getattr(record, 'station_name', '')
    features.relay_family = ch_map.relay_family

    # Sample axis
    time = getattr(record, 'time', None)
    if time is None or len(time) < 4:
        features.warnings.append("No time axis — cannot extract features")
        return features

    fs = _get_sampling_rate(record, time)
    features.sampling_rate_hz = fs

    # Samples per cycle
    if fs <= 0:
        features.warnings.append("Cannot determine sampling rate")
        return features
    spc = int(round(fs / SYSTEM_FREQ_HZ))

    # Fault inception index
    inception_idx = _get_inception_idx(fault_event, time, spc)

    if fault_event is not None:
        features.fault_duration_ms = getattr(fault_event, 'duration_ms', None)

    # Get all waveforms
    samples = get_mapped_samples(record, ch_map)
    features.channels_available = {k: (v is not None) for k, v in samples.items()}

    # ── Per-unit base ─────────────────────────────────────────────────────────
    i_base = _estimate_base_current(samples, rated_current_a, inception_idx, spc)
    features.hv_base_current_a = float(i_base) if i_base > 0 else None
    # Estimate MVA from HV side and derive LV base current for per-unit scaling.
    est_mva = _estimate_mva_from_hv(samples, inception_idx, spc)
    if est_mva is not None:
        features.estimated_mva = _snap_to_standard_mva(est_mva)
        features.lv_base_current_a = _estimate_lv_base_current(
            samples, features.estimated_mva, inception_idx, spc
        )

    # ── Compute differential / restraint ─────────────────────────────────────
    diff_a, rstr_a = _get_diff_rstr(samples, 'a', ch_map)
    diff_b, rstr_b = _get_diff_rstr(samples, 'b', ch_map)
    diff_c, rstr_c = _get_diff_rstr(samples, 'c', ch_map)

    # Analysis window: 2 cycles starting at inception
    n_win = min(2 * spc, len(time) - inception_idx)
    end_idx = inception_idx + n_win

    # ── Harmonic analysis on differential (or HV if no diff) ─────────────────
    for phase, sig_arr in [('a', diff_a), ('b', diff_b), ('c', diff_c)]:
        if sig_arr is None:
            # Fallback to HV winding current
            sig_arr = samples.get(f'i_hv_{phase}')
        if sig_arr is None or len(sig_arr) <= end_idx:
            continue

        window = sig_arr[inception_idx:end_idx]
        h2, h5, thd = _compute_harmonics(window, fs)
        setattr(features, f'h2_ratio_{phase}_pct', h2)
        setattr(features, f'h5_ratio_{phase}_pct', h5)

    # Set max harmonic ratios
    h2_vals = [v for v in [features.h2_ratio_a_pct, features.h2_ratio_b_pct, features.h2_ratio_c_pct] if v is not None]
    h5_vals = [v for v in [features.h5_ratio_a_pct, features.h5_ratio_b_pct, features.h5_ratio_c_pct] if v is not None]
    if h2_vals:
        features.h2_ratio_max_pct = max(h2_vals)
    if h5_vals:
        features.h5_ratio_max_pct = max(h5_vals)

    # THD on combined differential
    if diff_a is not None and len(diff_a) > end_idx:
        _, _, features.thd_diff_pct = _compute_harmonics(diff_a[inception_idx:end_idx], fs)

    # ── Differential / restraint magnitudes ──────────────────────────────────
    for phase, diff_sig, rstr_sig, attr_d, attr_r in [
        ('a', diff_a, rstr_a, 'idiff_max_a_pu', 'irstr_max_a_pu'),
        ('b', diff_b, rstr_b, 'idiff_max_b_pu', 'irstr_max_b_pu'),
        ('c', diff_c, rstr_c, 'idiff_max_c_pu', 'irstr_max_c_pu'),
    ]:
        if diff_sig is not None and len(diff_sig) > end_idx and i_base > 0:
            window = diff_sig[inception_idx:end_idx]
            setattr(features, attr_d, float(np.max(np.abs(window))) / i_base)
        if rstr_sig is not None and len(rstr_sig) > end_idx and i_base > 0:
            window = rstr_sig[inception_idx:end_idx]
            setattr(features, attr_r, float(np.max(np.abs(window))) / i_base)

    idiff_pus = [v for v in [features.idiff_max_a_pu, features.idiff_max_b_pu, features.idiff_max_c_pu] if v is not None]
    irstr_pus = [v for v in [features.irstr_max_a_pu, features.irstr_max_b_pu, features.irstr_max_c_pu] if v is not None]
    if idiff_pus:
        features.idiff_max_pu = max(idiff_pus)
    if irstr_pus:
        features.irstr_max_pu = max(irstr_pus)

    # Slope characteristic
    if features.idiff_max_pu is not None and features.irstr_max_pu is not None and features.irstr_max_pu > 0:
        slope = (features.idiff_max_pu / features.irstr_max_pu) * 100.0
        features.slope_worst_pct = slope
        features.above_slope1 = slope > SLOPE1_PCT
        features.above_slope2 = slope > SLOPE2_PCT
    elif features.idiff_max_pu is not None:
        # No restraint channel — use idiff vs rated threshold
        features.above_slope1 = features.idiff_max_pu > IDIFF_OPERATE_PU

    # ── Waveform shape analysis ───────────────────────────────────────────────
    for phase, diff_sig, attr_dc, attr_pp in [
        ('a', diff_a, 'dc_offset_index_a', 'pp_asymmetry_a'),
        ('b', diff_b, 'dc_offset_index_b', 'pp_asymmetry_b'),
        ('c', diff_c, 'dc_offset_index_c', 'pp_asymmetry_c'),
    ]:
        if diff_sig is None:
            diff_sig = samples.get(f'i_hv_{phase}')
        if diff_sig is None or len(diff_sig) <= end_idx:
            continue
        window = diff_sig[inception_idx:end_idx]
        dc_idx = _dc_offset_index(window)
        pp_asym = _pp_asymmetry(window)
        setattr(features, attr_dc, dc_idx)
        setattr(features, attr_pp, pp_asym)

    dc_vals = [v for v in [features.dc_offset_index_a, features.dc_offset_index_b, features.dc_offset_index_c] if v is not None]
    if dc_vals:
        features.dc_offset_index_max = max(dc_vals)

    # Zero-crossing variance (on phase-A differential / HV)
    sig_zc = diff_a if diff_a is not None else samples.get('i_hv_a')
    if sig_zc is not None and len(sig_zc) > end_idx:
        features.zc_interval_variance = _zc_interval_variance(
            sig_zc[inception_idx:end_idx], spc)

    # ── Current balance ───────────────────────────────────────────────────────
    i_hv_a_arr = samples.get('i_hv_a')
    i_lv_a_arr = samples.get('i_lv_a')
    if i_hv_a_arr is not None and i_lv_a_arr is not None and len(i_hv_a_arr) > end_idx:
        win_hv = i_hv_a_arr[inception_idx:end_idx]
        win_lv = i_lv_a_arr[inception_idx:end_idx]
        rms_hv = _rms(win_hv)
        rms_lv = _rms(win_lv)
        if rms_lv > 0.01:
            features.hv_lv_ratio_a = rms_hv / rms_lv
            features.hv_lv_phase_diff_a_deg = _phase_diff_deg(win_hv, win_lv, fs)

    # ── Energisation detection ────────────────────────────────────────────────
    prefault_samples = max(0, inception_idx - spc)
    if i_lv_a_arr is not None and inception_idx > spc:
        prefault_lv = i_lv_a_arr[prefault_samples:inception_idx]
        rms_lv_pre = _rms(prefault_lv)
        lv_base = features.lv_base_current_a if features.lv_base_current_a else i_base
        features.lv_prefault_irms_pu = (rms_lv_pre / lv_base) if lv_base > 0 else None
        features.energisation_flag = rms_lv_pre < 0.05 * lv_base if lv_base > 0 else False

    # ── Peak differential (absolute) ─────────────────────────────────────────
    if diff_a is not None and len(diff_a) > end_idx:
        features.peak_idiff_a = float(np.max(np.abs(diff_a[inception_idx:end_idx])))

    # ── Inception angle ───────────────────────────────────────────────────────
    v_hv_a = samples.get('v_hv_a')
    if v_hv_a is not None and inception_idx < len(v_hv_a):
        features.inception_angle_deg = _estimate_inception_angle(v_hv_a, inception_idx, fs)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_harmonics(window: np.ndarray, fs: float) -> tuple:
    """
    Compute 2nd harmonic ratio, 5th harmonic ratio, and THD for a waveform window.

    Returns:
        (h2_pct, h5_pct, thd_pct)  — all relative to fundamental magnitude.
        Returns (None, None, None) if signal is too short or zero.
    """
    n = len(window)
    if n < 4:
        return None, None, None

    # Apply Hann window to reduce spectral leakage
    win = np.hanning(n)
    sig = window * win

    # FFT
    spec = np.abs(np.fft.rfft(sig, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Find fundamental component
    f1_idx = _nearest_freq_idx(freqs, SYSTEM_FREQ_HZ)
    i1 = spec[f1_idx] if f1_idx < len(spec) else 0.0

    if i1 < 1e-9:
        return None, None, None

    f2_idx = _nearest_freq_idx(freqs, 2 * SYSTEM_FREQ_HZ)
    f5_idx = _nearest_freq_idx(freqs, 5 * SYSTEM_FREQ_HZ)

    i2 = spec[f2_idx] if f2_idx < len(spec) else 0.0
    i5 = spec[f5_idx] if f5_idx < len(spec) else 0.0

    # THD: RMS of all harmonics 2..10 / fundamental
    thd_sq = 0.0
    for k in range(2, 11):
        fk_idx = _nearest_freq_idx(freqs, k * SYSTEM_FREQ_HZ)
        if fk_idx < len(spec):
            thd_sq += spec[fk_idx] ** 2
    thd_pct = (np.sqrt(thd_sq) / i1) * 100.0

    return (i2 / i1) * 100.0, (i5 / i1) * 100.0, thd_pct


def _nearest_freq_idx(freqs: np.ndarray, target_hz: float) -> int:
    """Return FFT bin index closest to target_hz."""
    return int(np.argmin(np.abs(freqs - target_hz)))


def _dc_offset_index(window: np.ndarray) -> float:
    """
    DC-offset (asymmetry) index.

    For a purely sinusoidal waveform the mean is 0.
    For an inrush waveform with DC offset, the mean is significantly non-zero.

    Index = |mean(window)| / rms(window)   (0 = symmetric, 1 = full DC)
    """
    rms_val = _rms(window)
    if rms_val < 1e-9:
        return 0.0
    return float(abs(np.mean(window)) / rms_val)


def _pp_asymmetry(window: np.ndarray) -> float:
    """
    Peak-to-peak asymmetry: difference between positive and negative peaks.

    Value near 0 → symmetric (through-fault, internal-fault).
    Value near 1 → highly asymmetric (inrush).
    """
    ip = float(np.max(window))
    in_ = float(np.min(window))
    denom = max(abs(ip), abs(in_))
    if denom < 1e-9:
        return 0.0
    return abs(ip + in_) / denom  # asymmetry between positive and negative peaks


def _zc_interval_variance(window: np.ndarray, spc: int) -> float:
    """
    Variance of zero-crossing intervals, normalised by expected interval.

    For a pure sinusoid: zero crossings every spc/2 samples → variance ≈ 0.
    For inrush (missing half-cycles): variance is large.

    Returns normalised variance (dimensionless).
    """
    sign = np.sign(window)
    # Find zero crossings (sign changes)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) < 3:
        return 1.0  # Very few crossings → likely inrush / missing half-cycle

    intervals = np.diff(crossings)
    expected = spc / 2.0
    variance = float(np.var(intervals)) / (expected ** 2)
    return min(variance, 5.0)  # Cap at 5 to avoid outlier explosion


def _rms(arr: np.ndarray) -> float:
    """Root mean square of array."""
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def _phase_diff_deg(sig_a: np.ndarray, sig_b: np.ndarray, fs: float) -> float:
    """
    Estimate phase difference between two signals using cross-correlation.
    Returns angle in degrees (-180 to +180).
    """
    if len(sig_a) < 2 or len(sig_b) < 2:
        return 0.0
    n = min(len(sig_a), len(sig_b))
    corr = np.correlate(sig_a[:n], sig_b[:n], mode='full')
    lag = int(np.argmax(np.abs(corr))) - (n - 1)
    deg = (lag / (fs / SYSTEM_FREQ_HZ)) * 360.0
    # Wrap to -180..+180
    deg = ((deg + 180) % 360) - 180
    return float(deg)


def _estimate_inception_angle(v_sig: np.ndarray, inception_idx: int, fs: float) -> float:
    """Estimate voltage phase angle at inception (degrees, 0-360)."""
    if inception_idx >= len(v_sig):
        return 0.0
    spc = fs / SYSTEM_FREQ_HZ
    # Use one cycle before inception to estimate phase
    start = max(0, inception_idx - int(spc))
    pre_cycle = v_sig[start:inception_idx]
    if len(pre_cycle) < 4:
        return 0.0
    # Find last zero-crossing before inception
    sign = np.sign(pre_cycle)
    zc_idx = np.where(np.diff(sign) > 0)[0]  # positive-going crossings
    if len(zc_idx) == 0:
        return 0.0
    last_zc = zc_idx[-1]
    angle = ((inception_idx - (start + last_zc)) / spc) * 360.0
    return float(angle % 360.0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_sampling_rate(record, time: np.ndarray) -> float:
    """Get sampling rate from record metadata or estimate from time axis."""
    rates = getattr(record, 'sampling_rates', [])
    if rates:
        first = rates[0]
        if isinstance(first, dict):
            # comtrade library may use keys like 'samp' or 'rate'
            return float(first.get('samp', first.get('rate', 0)) or 0)
        if isinstance(first, (list, tuple)) and len(first) >= 1:
            # Our parser stores (rate_hz, end_sample)
            return float(first[0] or 0)
        return float(first or 0)
    if len(time) > 1:
        dt = float(np.median(np.diff(time[:min(100, len(time))])))
        return 1.0 / dt if dt > 0 else 0.0
    return 0.0


def _get_inception_idx(fault_event, time: np.ndarray, spc: int) -> int:
    """Return sample index for fault inception, with fallback."""
    if fault_event is not None:
        idx = getattr(fault_event, 'inception_idx', None)
        if idx is not None and 0 <= idx < len(time):
            return int(idx)
    # Default: use 2 cycles as pre-fault margin
    return min(2 * spc, max(0, len(time) // 4))


def _prefault_rms(arr: np.ndarray, inception_idx: int, spc: int) -> Optional[float]:
    """RMS over one-cycle pre-fault window."""
    if arr is None or inception_idx < spc:
        return None
    pre = arr[max(0, inception_idx - spc):inception_idx]
    if len(pre) == 0:
        return None
    return _rms(pre)


def _estimate_mva_from_hv(samples: dict, inception_idx: int, spc: int) -> Optional[float]:
    """
    Estimate transformer MVA from HV side pre-fault RMS V/I.
    Assumes phase-to-ground voltage channels (kV) and phase current (A).
    """
    v_candidates = [samples.get('v_hv_a'), samples.get('v_hv_b'), samples.get('v_hv_c')]
    i_candidates = [samples.get('i_hv_a'), samples.get('i_hv_b'), samples.get('i_hv_c')]

    v_rms = [v for v in (_prefault_rms(v, inception_idx, spc) for v in v_candidates) if v]
    i_rms = [i for i in (_prefault_rms(i, inception_idx, spc) for i in i_candidates) if i]
    if not v_rms or not i_rms:
        return None

    v_ph_kv = float(np.mean(v_rms))
    i_a = float(np.mean(i_rms))
    if v_ph_kv <= 0.1 or i_a <= 0.1:
        return None

    # S_MVA = 3 * V_phase(kV) * I(A) / 1000
    return (3.0 * v_ph_kv * i_a) / 1000.0


def _snap_to_standard_mva(mva: float) -> float:
    """Snap estimated MVA to nearest standard rating."""
    if not STANDARD_MVA_RATINGS:
        return float(mva)
    return float(min(STANDARD_MVA_RATINGS, key=lambda x: abs(x - mva)))


def _estimate_lv_base_current(samples: dict, mva: float, inception_idx: int, spc: int) -> Optional[float]:
    """
    Compute LV base current (A) from MVA and LV phase voltage.
    Falls back to 20 kV if LV voltage channels are not available.
    """
    if not mva or mva <= 0:
        return None
    v_candidates = [samples.get('v_lv_a'), samples.get('v_lv_b'), samples.get('v_lv_c')]
    v_rms = [v for v in (_prefault_rms(v, inception_idx, spc) for v in v_candidates) if v]
    v_ph_kv = float(np.mean(v_rms)) if v_rms else 20.0
    if v_ph_kv <= 0.1:
        v_ph_kv = 20.0
    # I_base = S_MVA * 1000 / (3 * V_phase_kV)
    return (mva * 1000.0) / (3.0 * v_ph_kv)


def _estimate_base_current(samples: dict, rated_a: float, inception_idx: int, spc: int) -> float:
    """
    Estimate per-unit base current.
    Uses rated_a if provided, otherwise estimates from pre-fault HV RMS.
    """
    if rated_a > 10.0:
        return rated_a

    # Estimate from pre-fault HV phase-A current
    i_hv = samples.get('i_hv_a')
    if i_hv is not None and inception_idx >= spc:
        pre = i_hv[max(0, inception_idx - spc):inception_idx]
        rms = _rms(pre)
        if rms > 1.0:
            # Rated ≈ 1.1× pre-fault (assume ~90% loading pre-fault)
            return rms * 1.1

    # Last resort: use peak of differential divided by √2
    i_diff = samples.get('i_diff_a')
    if i_diff is not None and len(i_diff) > inception_idx + spc:
        peak = np.max(np.abs(i_diff[inception_idx:inception_idx + spc]))
        if peak > 1.0:
            return peak / np.sqrt(2)

    return 100.0  # fallback: 100 A rated (avoids division by zero)


def _get_diff_rstr(samples: dict, phase: str, ch_map) -> tuple:
    """
    Return (differential_array, restraint_array) for a given phase.

    If relay-measured differential not available, compute:
      Idiff = I_HV + I_LV  (after CT polarity normalisation: I_LV should be ~anti-phase)
      Irestr = (|I_HV| + |I_LV|) / 2
    """
    diff = samples.get(f'i_diff_{phase}')
    rstr = samples.get(f'i_rstr_{phase}')

    if diff is None:
        i_hv = samples.get(f'i_hv_{phase}')
        i_lv = samples.get(f'i_lv_{phase}')
        if i_hv is not None and i_lv is not None:
            n = min(len(i_hv), len(i_lv))
            # LV current is anti-phase (transforming → reversed polarity)
            diff = i_hv[:n] + i_lv[:n]   # differential = sum (should ≈ 0 through-fault)
            rstr = (np.abs(i_hv[:n]) + np.abs(i_lv[:n])) / 2.0
        elif i_hv is not None:
            diff = i_hv   # single-ended: only HV available
            rstr = None
        else:
            diff = None
            rstr = None

    return diff, rstr


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper for display / export
# ─────────────────────────────────────────────────────────────────────────────

def features_to_dict(f: TransformerFeatures) -> dict:
    """Convert TransformerFeatures to a flat dict for CSV / JSON export."""
    return {
        'station_name':           f.station_name,
        'relay_family':           f.relay_family,
        'h2_ratio_a_pct':         f.h2_ratio_a_pct,
        'h2_ratio_b_pct':         f.h2_ratio_b_pct,
        'h2_ratio_c_pct':         f.h2_ratio_c_pct,
        'h2_ratio_max_pct':       f.h2_ratio_max_pct,
        'h5_ratio_a_pct':         f.h5_ratio_a_pct,
        'h5_ratio_b_pct':         f.h5_ratio_b_pct,
        'h5_ratio_c_pct':         f.h5_ratio_c_pct,
        'h5_ratio_max_pct':       f.h5_ratio_max_pct,
        'thd_diff_pct':           f.thd_diff_pct,
        'idiff_max_pu':           f.idiff_max_pu,
        'irstr_max_pu':           f.irstr_max_pu,
        'slope_worst_pct':        f.slope_worst_pct,
        'above_slope1':           f.above_slope1,
        'above_slope2':           f.above_slope2,
        'dc_offset_index_max':    f.dc_offset_index_max,
        'pp_asymmetry_a':         f.pp_asymmetry_a,
        'zc_interval_variance':   f.zc_interval_variance,
        'hv_lv_ratio_a':          f.hv_lv_ratio_a,
        'hv_lv_phase_diff_a_deg': f.hv_lv_phase_diff_a_deg,
        'energisation_flag':      f.energisation_flag,
        'lv_prefault_irms_pu':    f.lv_prefault_irms_pu,
        'estimated_mva':          f.estimated_mva,
        'hv_base_current_a':      f.hv_base_current_a,
        'lv_base_current_a':      f.lv_base_current_a,
        'channels_available':     f.channels_available,
        'inception_angle_deg':    f.inception_angle_deg,
        'fault_duration_ms':      f.fault_duration_ms,
        'peak_idiff_a':           f.peak_idiff_a,
        'sampling_rate_hz':       f.sampling_rate_hz,
    }
