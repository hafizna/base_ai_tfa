"""
Ekstraktor Fitur Rele Diferensial Saluran (87L)
================================================
Jalur fitur terpisah untuk rele diferensial saluran transmisi.

Rele 87L bekerja pada prinsip perbandingan arus ujung lokal vs. ujung remote
(arus diferensial Id vs. arus restraint Ir).  Rekaman COMTRADE yang dihasilkan
biasanya hanya memuat arus lokal (IA/IB/IC) — tanpa tegangan VT dan tanpa
perhitungan impedansi zona.

Referensi literatur:
  • Ahmadimanesh & Shahrtash (2013): DWT-based fault classification for transmission
    lines — high-frequency energy ratio (detail coeff D1/D2) discriminates impulsive
    (petir) vs. slow-rise (pohon/vegetasi) faults.
  • Guillen et al. (2019): Wavelet scalogram + CNN, 100% accuracy current-only.
  • Rai et al. (2021): Statistical coherence — rise_time, dc_offset, oscillation freq
    as primary discriminators between fault causes.
  • Gopakumar et al. (2018): DWT energy distribution separates lightning from
    conductor faults (arcing) and vegetation contacts.

Fitur baru (tidak ada di distance extractor):
  rise_time_ms         — 10%-90% current rise time (ms); lightning <1ms, pohon 10-100ms
  dc_offset_index      — half-cycle asymmetry; proxy for inception angle without VT
  transient_osc_freq_hz— dominant non-fundamental frequency via FFT
  dwt_energy_detail_1  — FFT band energy D1 (fs/4 – fs/2): very high freq
  dwt_energy_detail_2  — FFT band energy D2 (fs/8 – fs/4): high freq
  dwt_energy_approx    — FFT band energy A  (0 – fs/8): low freq (fundamental)
  dwt_hf_ratio         — (D1+D2)/(D1+D2+A): fraction of high-freq energy

Jalur ini menyimpan fitur tegangan sebagai NaN (bukan 0.0) sehingga LightGBM
memperlakukannya sebagai nilai hilang, bukan pengukuran nyata.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Import helper functions shared with the distance extractor ────────────────
# These are module-level private helpers — import explicitly to avoid reimpl.
from core.feature_extractor import (
    _get_channel,
    _get_sampling_rate,
    _extract_line_tag,
    _detect_active_line_tag,
    _detect_operated_line_tag_from_status,
    _calculate_di_dt,
    _calculate_peak_current,
    _calculate_i0_i1_ratio,
    _calculate_thd,
    _calculate_symmetrical_magnitudes,
    _determine_fault_type,
)

nan = float("nan")


@dataclass
class DifferentialLineFeatures:
    """
    Features extracted from an 87L line differential relay recording.

    Voltage-derived fields (inception_angle, voltage_sag, v_prefault, v_fault,
    r_x_ratio, z_magnitude, z_angle) are intentionally absent — they would be
    float('nan') and are already absent from 87L recordings by design.

    The schema is a SUPERSET of the DistanceFeatures fields that the Tier 2
    classifier reads, plus new 87L-specific morphology features.
    """

    # ── Universal current features ─────────────────────────────────────────────
    di_dt_max: float                        # Max |dI/dt| first cycle (A/s)
    di_dt_phase: str                        # Phase with highest dI/dt
    peak_fault_current_a: float             # Max instantaneous fault current (A)
    peak_fault_phase: str                   # Phase with highest peak
    i0_i1_ratio: float                      # Zero-seq / positive-seq current ratio
    thd_percent: float                      # THD of fault current, first 2 cycles
    i0_magnitude_a: Optional[float]
    i1_magnitude_a: Optional[float]
    i2_magnitude_a: Optional[float]

    # ── Fault morphology (87L-specific, literature-based) ─────────────────────
    rise_time_ms: Optional[float]           # 10%-90% rise time of leading edge
    dc_offset_index: Optional[float]        # Half-cycle asymmetry [0-1]
    transient_osc_freq_hz: Optional[float]  # Dominant non-50Hz oscillation (Hz)
    dwt_energy_detail_1: Optional[float]    # FFT band D1 energy (fs/4–fs/2)
    dwt_energy_detail_2: Optional[float]    # FFT band D2 energy (fs/8–fs/4)
    dwt_energy_approx: Optional[float]      # FFT band A energy (0–fs/8)
    dwt_hf_ratio: Optional[float]          # (D1+D2)/(D1+D2+A), high-freq fraction

    # ── Optional 87L-relay-native channels ────────────────────────────────────
    has_differential_channels: bool         # True if relay exported Id/Ir channels
    id_peak_a: Optional[float]             # Peak differential current from relay
    id_ir_ratio_max: Optional[float]       # Max Id/Ir ratio (slope characteristic)

    # ── Protection context ─────────────────────────────────────────────────────
    trip_type: str                          # "single_pole" or "three_pole"
    reclose_attempted: bool
    reclose_successful: Optional[bool]
    reclose_time_ms: Optional[float]
    fault_count: int

    # ── Fault type ────────────────────────────────────────────────────────────
    faulted_phases: List[str]
    fault_type: str                         # "SLG", "DLG", "LL", "3PH"
    is_ground_fault: bool

    # ── Metadata ──────────────────────────────────────────────────────────────
    station_name: str
    relay_model: str
    sampling_rate_hz: float
    record_duration_ms: float


# ── Private helper functions (87L-specific) ───────────────────────────────────

def _calculate_rise_time(ia, ib, ic, inception_idx: int,
                          sampling_rate: float, system_freq: float) -> Optional[float]:
    """
    Calculate 10%-90% rise time of the fault current leading edge.

    Returns time in milliseconds.  Uses the phase with the highest peak current
    as the representative signal.

    Lightning: typically < 1 ms.
    Pohon / vegetasi: typically 10–100 ms.
    Layang-layang / benda asing: 1–20 ms.
    """
    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        # Search 3 cycles after inception for the peak
        window_end = min(inception_idx + 3 * samples_per_cycle, len(ia.samples))

        peaks = {
            'A': np.max(np.abs(ia.samples[inception_idx:window_end])),
            'B': np.max(np.abs(ib.samples[inception_idx:window_end])),
            'C': np.max(np.abs(ic.samples[inception_idx:window_end])),
        }
        phase = max(peaks, key=peaks.get)
        samples = {
            'A': ia.samples, 'B': ib.samples, 'C': ic.samples
        }[phase]

        peak_val = peaks[phase]
        if peak_val < 1e-6:
            return None

        lvl_10 = 0.10 * peak_val
        lvl_90 = 0.90 * peak_val

        # Find first crossing of 10% and 90% thresholds after inception
        seg = np.abs(samples[inception_idx:window_end])
        idx_10 = np.argmax(seg >= lvl_10)  # first sample at or above 10%
        idx_90 = np.argmax(seg >= lvl_90)  # first sample at or above 90%

        if idx_10 == 0 and seg[0] < lvl_10:
            return None   # never crossed 10% — can't measure
        if idx_90 <= idx_10:
            # Already at 90% at inception (e.g. clipped recording)
            # Return sub-sample estimate
            return 0.0

        rise_samples = idx_90 - idx_10
        rise_ms = (rise_samples / sampling_rate) * 1000.0
        return float(rise_ms)

    except Exception as e:
        logger.debug(f"Rise time calculation failed: {e}")
        return None


def _calculate_dc_offset_index(ia, ib, ic, inception_idx: int,
                                 sampling_rate: float, system_freq: float) -> Optional[float]:
    """
    Calculate half-cycle asymmetry as a proxy for fault inception angle.

    Method: compare RMS of positive half-cycle vs. negative half-cycle in the
    first cycle after inception.  A large asymmetry (high DC offset) indicates
    fault occurred near voltage peak — characteristic of lightning.

    dc_offset_index = |RMS_pos - RMS_neg| / (RMS_pos + RMS_neg)
      → 0.0: perfectly symmetric (fault at voltage zero crossing)
      → 1.0: maximally asymmetric (pure DC offset, fault at voltage peak)
    """
    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_end = min(inception_idx + samples_per_cycle, len(ia.samples))

        # Use phase with highest peak
        peak_a = np.max(np.abs(ia.samples[inception_idx:window_end]))
        peak_b = np.max(np.abs(ib.samples[inception_idx:window_end]))
        peak_c = np.max(np.abs(ic.samples[inception_idx:window_end]))
        if peak_a >= peak_b and peak_a >= peak_c:
            seg = ia.samples[inception_idx:window_end]
        elif peak_b >= peak_c:
            seg = ib.samples[inception_idx:window_end]
        else:
            seg = ic.samples[inception_idx:window_end]

        pos = seg[seg > 0]
        neg = seg[seg < 0]

        rms_pos = float(np.sqrt(np.mean(pos**2))) if len(pos) > 2 else 0.0
        rms_neg = float(np.sqrt(np.mean(neg**2))) if len(neg) > 2 else 0.0

        denom = rms_pos + rms_neg
        if denom < 1e-9:
            return 0.0

        return float(abs(rms_pos - rms_neg) / denom)

    except Exception as e:
        logger.debug(f"DC offset index calculation failed: {e}")
        return None


def _calculate_transient_oscillation_freq(ia, ib, ic, inception_idx: int,
                                           sampling_rate: float,
                                           system_freq: float) -> Optional[float]:
    """
    Find the dominant non-fundamental oscillation frequency via FFT.

    Uses a 2-cycle window after inception.  Suppresses the fundamental
    (system_freq ± 5 Hz) and its 2nd/3rd harmonics, then returns the
    frequency bin with the highest magnitude.

    Lightning: 1–10 kHz (travelling wave transients).
    Pohon / layang-layang: 100–1000 Hz (slow resistive arc).
    Hewan: 50–500 Hz (brief metallic contact).
    """
    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_end = min(inception_idx + 2 * samples_per_cycle, len(ia.samples))

        # Use phase with highest peak
        peak_a = np.max(np.abs(ia.samples[inception_idx:window_end]))
        peak_b = np.max(np.abs(ib.samples[inception_idx:window_end]))
        peak_c = np.max(np.abs(ic.samples[inception_idx:window_end]))
        if peak_a >= peak_b and peak_a >= peak_c:
            seg = ia.samples[inception_idx:window_end].copy()
        elif peak_b >= peak_c:
            seg = ib.samples[inception_idx:window_end].copy()
        else:
            seg = ic.samples[inception_idx:window_end].copy()

        if len(seg) < 8:
            return None

        # Remove DC and window
        seg -= np.mean(seg)
        seg *= np.hanning(len(seg))

        N = len(seg)
        freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
        magnitudes = np.abs(np.fft.rfft(seg))

        # Suppress DC, fundamental and its harmonics (up to 4th)
        suppress_bw = 5.0   # ±5 Hz band around each harmonic
        for h in range(0, 5):  # 0 Hz, 50, 100, 150, 200
            center = h * system_freq
            mask = (freqs >= center - suppress_bw) & (freqs <= center + suppress_bw)
            magnitudes[mask] = 0.0

        # Also suppress frequencies above Nyquist/2 (less reliable)
        if sampling_rate > 0:
            magnitudes[freqs > sampling_rate / 2.0 * 0.95] = 0.0

        if np.max(magnitudes) < 1e-12:
            return None

        peak_freq = float(freqs[np.argmax(magnitudes)])
        return peak_freq if peak_freq > 0 else None

    except Exception as e:
        logger.debug(f"Transient oscillation frequency calculation failed: {e}")
        return None


def _calculate_fft_band_energies(ia, ib, ic, inception_idx: int,
                                   sampling_rate: float,
                                   system_freq: float
                                   ) -> tuple:
    """
    Compute FFT-based frequency band energies as DWT approximations.

    Three bands (mimicking 2-level DWT decomposition):
      D1 (Detail 1): fs/4  – fs/2  (very high frequency)
      D2 (Detail 2): fs/8  – fs/4  (high frequency)
      A  (Approx)  : 0     – fs/8  (low frequency, includes fundamental)

    Returns (energy_d1, energy_d2, energy_approx, hf_ratio)
    All energies are normalised to the total FFT energy to be scale-invariant.
    """
    try:
        samples_per_cycle = int(sampling_rate / system_freq)
        window_end = min(inception_idx + 2 * samples_per_cycle, len(ia.samples))

        # Use phase with highest peak
        peak_a = np.max(np.abs(ia.samples[inception_idx:window_end]))
        peak_b = np.max(np.abs(ib.samples[inception_idx:window_end]))
        peak_c = np.max(np.abs(ic.samples[inception_idx:window_end]))
        if peak_a >= peak_b and peak_a >= peak_c:
            seg = ia.samples[inception_idx:window_end].copy()
        elif peak_b >= peak_c:
            seg = ib.samples[inception_idx:window_end].copy()
        else:
            seg = ic.samples[inception_idx:window_end].copy()

        if len(seg) < 8:
            return None, None, None, None

        seg -= np.mean(seg)
        N = len(seg)
        freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
        power = np.abs(np.fft.rfft(seg)) ** 2  # power spectrum

        fs = sampling_rate
        mask_d1 = (freqs >= fs / 4.0) & (freqs < fs / 2.0)
        mask_d2 = (freqs >= fs / 8.0) & (freqs < fs / 4.0)
        mask_a  = (freqs >= 0)         & (freqs < fs / 8.0)

        e_d1 = float(np.sum(power[mask_d1]))
        e_d2 = float(np.sum(power[mask_d2]))
        e_a  = float(np.sum(power[mask_a]))

        total = e_d1 + e_d2 + e_a
        if total < 1e-30:
            return 0.0, 0.0, 0.0, 0.0

        # Normalise
        e_d1_n = e_d1 / total
        e_d2_n = e_d2 / total
        e_a_n  = e_a  / total
        hf_ratio = (e_d1 + e_d2) / total

        return float(e_d1_n), float(e_d2_n), float(e_a_n), float(hf_ratio)

    except Exception as e:
        logger.debug(f"FFT band energy calculation failed: {e}")
        return None, None, None, None


def _look_for_differential_channels(record, inception_idx: int,
                                     sampling_rate: float,
                                     system_freq: float) -> tuple:
    """
    Search COMTRADE record for relay-native differential (Id) and restraint (Ir)
    channels exported by 87L relays (ABB REL670, SEL-387L, GE L90, Siemens 7SD).

    Returns (has_diff_channels, id_peak_a, id_ir_ratio_max).
    """
    # Canonical name patterns used by major relay manufacturers
    IDIFF_PATTERNS = (
        "I-DIFF", "IDIFF", "ID", "ID1", "ID2",
        "DIFF:I", "DIFFERENTIAL", "I DIFF",
        # ABB REL670 COMTRADE channel names
        "Ln1:87L:I-DIFF",
        # SEL-387L
        "IDIF", "DIFF",
        # GE L90
        "IL-DIFF", "ILD",
        # Siemens 7SD
        "IDIFF_A", "IDIFF_B", "IDIFF_C",
    )
    IREST_PATTERNS = (
        "I-REST", "IREST", "IR", "RESTRAIN", "IRESTR",
        "Ln1:87L:I-REST",
        "STAB", "BIAS",
    )

    samples_per_cycle = max(1, int(sampling_rate / system_freq))
    window_end_fn = lambda ch: min(inception_idx + 3 * samples_per_cycle, len(ch.samples))

    id_channel = None
    ir_channel = None

    for ch in record.analog_channels:
        name_up = (getattr(ch, "name", "") or "").upper()
        canon   = (getattr(ch, "canonical_name", "") or "").upper()
        combined = name_up + " " + canon

        if id_channel is None:
            for p in IDIFF_PATTERNS:
                if p.upper() in combined:
                    id_channel = ch
                    break
        if ir_channel is None:
            for p in IREST_PATTERNS:
                if p.upper() in combined:
                    ir_channel = ch
                    break
        if id_channel and ir_channel:
            break

    if id_channel is None:
        return False, None, None

    # Extract Id peak
    try:
        we = window_end_fn(id_channel)
        id_peak = float(np.max(np.abs(id_channel.samples[inception_idx:we])))
    except Exception:
        return True, None, None

    # Compute Id/Ir ratio if restraint channel present
    id_ir_ratio_max = None
    if ir_channel is not None:
        try:
            we_r = window_end_fn(ir_channel)
            ir_seg = np.abs(ir_channel.samples[inception_idx:we_r])
            id_seg = np.abs(id_channel.samples[inception_idx:we])

            min_len = min(len(id_seg), len(ir_seg))
            if min_len > 0:
                ir_pos = ir_seg[:min_len]
                id_pos = id_seg[:min_len]
                # Avoid division by zero; use 1% of peak as minimum denominator
                ir_min = max(np.max(ir_pos) * 0.01, 1e-6)
                ratio = id_pos / np.maximum(ir_pos, ir_min)
                id_ir_ratio_max = float(np.max(ratio))
        except Exception:
            pass

    return True, float(id_peak), id_ir_ratio_max


# ── Public API ────────────────────────────────────────────────────────────────

def extract_87l_features(record, fault, protection) -> Optional["DifferentialLineFeatures"]:
    """
    Extract current-only fault-cause features from an 87L line differential
    relay COMTRADE recording.

    This is a completely separate extraction path from distance relays.
    Voltage features are not computed — use NaN in flatten to signal missing
    data to LightGBM rather than feeding in misleading 0.0 values.

    Args:
        record    : ComtradeRecord (from comtrade_parser)
        fault     : FaultEvent    (from fault_detector)
        protection: ProtectionResult (from protection_router)

    Returns:
        DifferentialLineFeatures or None if extraction fails
    """
    try:
        inception_idx = fault.inception_idx
        active_line_tag = (
            _detect_operated_line_tag_from_status(record, inception_idx)
            or _detect_active_line_tag(record, inception_idx)
        )

        ia = _get_channel(record, 'IA', 'current', active_line_tag)
        ib = _get_channel(record, 'IB', 'current', active_line_tag)
        ic = _get_channel(record, 'IC', 'current', active_line_tag)

        if not all([ia, ib, ic]):
            logger.error("[87L] Missing required current channels (IA/IB/IC)")
            return None

        sampling_rate = _get_sampling_rate(record)
        if sampling_rate == 0:
            logger.error("[87L] Cannot determine sampling rate")
            return None

        system_freq = float(record.frequency) if record.frequency else 50.0

        # ── Standard current features (shared helpers) ────────────────────────
        di_dt_max, di_dt_phase = _calculate_di_dt(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        peak_current, peak_phase = _calculate_peak_current(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        i0_i1_ratio = _calculate_i0_i1_ratio(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        thd_percent = _calculate_thd(
            ia, ib, ic, inception_idx, sampling_rate, system_freq, fault.faulted_phases
        )
        i0_mag, i1_mag, i2_mag = _calculate_symmetrical_magnitudes(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        fault_type, is_ground_fault = _determine_fault_type(fault.faulted_phases, i0_i1_ratio)

        # ── 87L morphology features ───────────────────────────────────────────
        rise_time_ms = _calculate_rise_time(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        dc_offset_index = _calculate_dc_offset_index(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        transient_osc_freq_hz = _calculate_transient_oscillation_freq(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )
        e_d1, e_d2, e_a, hf_ratio = _calculate_fft_band_energies(
            ia, ib, ic, inception_idx, sampling_rate, system_freq
        )

        # ── Optional 87L relay-native differential channels ───────────────────
        has_diff_ch, id_peak, id_ir_ratio = _look_for_differential_channels(
            record, inception_idx, sampling_rate, system_freq
        )

        # ── Reclose / protection context ──────────────────────────────────────
        dead_time_ms = None
        if fault.reclose_events:
            first_reclose_s = float(fault.reclose_events[0].get('time', 0.0) or 0.0)
            if fault.clearing_time is not None:
                dt_ms = (first_reclose_s - float(fault.clearing_time)) * 1000.0
                dead_time_ms = dt_ms if dt_ms >= 0 else None
            if dead_time_ms is None:
                dead_time_ms = first_reclose_s * 1000.0

        record_dur_ms = (
            float(record.time[-1]) * 1000.0 if len(record.time) > 0 else 0.0
        )

        return DifferentialLineFeatures(
            # Standard current
            di_dt_max=di_dt_max,
            di_dt_phase=di_dt_phase,
            peak_fault_current_a=peak_current,
            peak_fault_phase=peak_phase,
            i0_i1_ratio=i0_i1_ratio,
            thd_percent=thd_percent,
            i0_magnitude_a=i0_mag,
            i1_magnitude_a=i1_mag,
            i2_magnitude_a=i2_mag,
            # Morphology
            rise_time_ms=rise_time_ms,
            dc_offset_index=dc_offset_index,
            transient_osc_freq_hz=transient_osc_freq_hz,
            dwt_energy_detail_1=e_d1,
            dwt_energy_detail_2=e_d2,
            dwt_energy_approx=e_a,
            dwt_hf_ratio=hf_ratio,
            # Optional relay channels
            has_differential_channels=has_diff_ch,
            id_peak_a=id_peak,
            id_ir_ratio_max=id_ir_ratio,
            # Protection context
            trip_type=protection.trip_type,
            reclose_attempted=(protection.auto_reclose_attempted or bool(fault.reclose_events)),
            reclose_successful=(
                protection.auto_reclose_successful
                if protection.auto_reclose_successful is not None
                else (fault.reclose_events[0].get('success') if fault.reclose_events else None)
            ),
            reclose_time_ms=dead_time_ms,
            fault_count=len(fault.reclose_events) + 1,
            # Fault type
            faulted_phases=fault.faulted_phases,
            fault_type=fault_type,
            is_ground_fault=is_ground_fault,
            # Metadata
            station_name=record.station_name,
            relay_model=record.rec_dev_id,
            sampling_rate_hz=sampling_rate,
            record_duration_ms=record_dur_ms,
        )

    except Exception as exc:
        logger.error(f"[87L] Feature extraction failed: {exc}", exc_info=True)
        return None
