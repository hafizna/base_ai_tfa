"""
Synthetic Transformer COMTRADE Generator
==========================================
Generates realistic synthetic COMTRADE (.cfg + .dat) files for transformer
protection events — specifically designed for Phase 2 development when NO
real transformer event recordings are available.

Generates all 5 event classes:
  1. INRUSH          — magnetising inrush (2nd harmonic dominant)
  2. INTERNAL_FAULT  — winding fault (no harmonic restraint, slope exceeded)
  3. THROUGH_FAULT   — external fault, balanced differential
  4. OVEREXCITATION  — 5th harmonic dominant, elevated voltage
  5. MAL_OPERATE     — CT saturation pattern

Physics models:
  - Inrush: superimposed DC offset + 2nd harmonic using Fourier synthesis
    (Blume & Boyajian model, 1919; updated by Holcomb, 1961)
  - Overexcitation: V/Hz ratio generates 5th harmonic via transformer core model
    (Dolinar et al., 1993 IEEE TPWRD)
  - Through-fault: equal HV/LV currents, differential ≈ 0
  - CT saturation: Jiles-Atherton B-H curve approximation for mal-operate

Output:
  data/synthetic/transformer/{event_class}/{station}_{timestamp}.cfg
  data/synthetic/transformer/{event_class}/{station}_{timestamp}.dat

Usage:
  python data/generate_synthetic_transformer.py
  python data/generate_synthetic_transformer.py --n 20 --seed 42
"""

from __future__ import annotations

import argparse
import os
import math
import random
import struct
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FREQ       = 50.0       # Hz — PLN system
SRATE      = 4800.0     # samples/s (96 samples/cycle — typical transformer relay)
SPC        = int(SRATE / FREQ)   # 96 samples per cycle
PRE_FAULT  = 3          # cycles of pre-fault
EVENT      = 5          # cycles of event window
POST_FAULT = 2          # cycles of post-event
TOTAL_CYC  = PRE_FAULT + EVENT + POST_FAULT
N_SAMPLES  = TOTAL_CYC * SPC

V_RATED_KV  = 150.0     # kV (primary HV, PLN 150 kV)
I_RATED_HV  = 400.0     # A  (HV rated current — typical 100 MVA transformer)
I_RATED_LV  = 2000.0    # A  (LV rated current)
CT_RATIO_HV = 400.0     # A primary / 1 A secondary
CT_RATIO_LV = 2000.0    # A primary / 1 A secondary

NOISE_A     = 0.5       # additive Gaussian noise amplitude (A)

OUT_DIR = Path(__file__).parent / "synthetic" / "transformer"

# ─────────────────────────────────────────────────────────────────────────────
# Waveform generators
# ─────────────────────────────────────────────────────────────────────────────

def _sine(t, amp, freq=FREQ, phase_deg=0.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * t + np.radians(phase_deg))


def _noise(n: int, sigma: float = NOISE_A) -> np.ndarray:
    return np.random.normal(0, sigma, n)


def _time_axis(n: int = N_SAMPLES) -> np.ndarray:
    return np.arange(n) / SRATE


def _fault_idx(cycle_offset: int = PRE_FAULT) -> int:
    return cycle_offset * SPC


# ─────────────────────────────────────────────────────────────────────────────
# 1. INRUSH — magnetising inrush model
#    Based on Equation from "Transformer Engineering" (Kulkarni & Khaparde, 2004)
#    i_inrush(t) = Ip * [sin(ωt + α - φ) - sin(α - φ)·exp(-t/τ)] + 2Ip·sin(ωt/2)
#    where:
#      α = voltage phase at switching (inception angle)
#      φ = impedance angle (large for transformers)
#      τ = L/R time constant (large → slow decay)
# ─────────────────────────────────────────────────────────────────────────────

def _gen_inrush(rng: np.random.Generator) -> dict:
    """Generate inrush current waveforms for HV and LV sides."""
    t = _time_axis()
    fi = _fault_idx()

    # Inrush parameters
    ip       = rng.uniform(3.0, 8.0) * I_RATED_HV   # Peak inrush (3–8× rated)
    tau_s    = rng.uniform(0.08, 0.25)               # DC decay time constant (s)
    alpha    = rng.uniform(0, math.pi / 4)           # Switching angle (worst: α ≈ 0)
    phi      = math.pi / 2 - rng.uniform(0, 0.2)    # Impedance angle ≈ 90°

    # Inrush waveform (only on HV side — transformer was de-energised on LV)
    i_hv_a = np.zeros(N_SAMPLES)
    for k in range(fi, N_SAMPLES):
        tk = (k - fi) / SRATE
        # Fundamental + DC offset
        fund = ip * math.sin(2 * math.pi * FREQ * tk + alpha - phi)
        dc   = -ip * math.sin(alpha - phi) * math.exp(-tk / tau_s)
        # 2nd harmonic (naturally present in inrush ≈ 30–60% of fundamental)
        h2_amp = rng.uniform(0.30, 0.60) * ip
        h2   = h2_amp * math.sin(4 * math.pi * FREQ * tk + alpha)
        i_hv_a[k] = fund + dc + h2

    # LV side current ≈ 0 (de-energised before switching)
    i_lv_a = _noise(N_SAMPLES, sigma=0.1 * I_RATED_LV * 0.02)

    # Phase B, C: similar with 120° shift, slightly less DC offset
    def _inrush_phase(phase_deg):
        arr = np.zeros(N_SAMPLES)
        alpha_ph = alpha + math.radians(phase_deg)
        dc_factor = rng.uniform(0.3, 0.7)  # phases have less symmetric inrush
        for k in range(fi, N_SAMPLES):
            tk = (k - fi) / SRATE
            fund = ip * math.sin(2 * math.pi * FREQ * tk + alpha_ph - phi)
            dc   = -ip * dc_factor * math.sin(alpha_ph - phi) * math.exp(-tk / tau_s)
            h2_amp = rng.uniform(0.20, 0.50) * ip
            h2   = h2_amp * math.sin(4 * math.pi * FREQ * tk + alpha_ph)
            arr[k] = fund + dc + h2
        return arr

    i_hv_b = _inrush_phase(-120)
    i_hv_c = _inrush_phase(+120)
    i_lv_b = _noise(N_SAMPLES, sigma=0.5)
    i_lv_c = _noise(N_SAMPLES, sigma=0.5)

    # Pre-fault: normal load current
    for arr, amp, ph in [(i_hv_a, I_RATED_HV, 0), (i_hv_b, I_RATED_HV, -120), (i_hv_c, I_RATED_HV, 120)]:
        arr[:fi] = _sine(t[:fi], amp * 0.9, phase_deg=ph) + _noise(fi)

    # Voltage (HV side, used for inception angle)
    v_hv_a = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=0)
    v_hv_b = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=-120)
    v_hv_c = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=120)

    return {
        'event_class': 'INRUSH',
        'i_hv_a': i_hv_a, 'i_hv_b': i_hv_b, 'i_hv_c': i_hv_c,
        'i_lv_a': i_lv_a, 'i_lv_b': i_lv_b, 'i_lv_c': i_lv_c,
        'v_hv_a': v_hv_a, 'v_hv_b': v_hv_b, 'v_hv_c': v_hv_c,
        'fault_idx': fi, 'comment': f'Inrush Ip={ip/I_RATED_HV:.1f}pu tau={tau_s:.2f}s',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTERNAL FAULT — winding-to-winding or turn-to-turn fault
# ─────────────────────────────────────────────────────────────────────────────

def _gen_internal_fault(rng: np.random.Generator) -> dict:
    """Internal fault: high differential, low harmonic content."""
    t = _time_axis()
    fi = _fault_idx()

    fault_pct = rng.uniform(0.3, 1.0)   # fault severity (fraction of winding)
    ip_fault  = rng.uniform(2.0, 10.0) * I_RATED_HV  # peak fault current

    # HV current: pre-fault load + fault contribution
    i_hv_a = _sine(t, I_RATED_HV * 0.9, phase_deg=0)
    i_hv_b = _sine(t, I_RATED_HV * 0.9, phase_deg=-120)
    i_hv_c = _sine(t, I_RATED_HV * 0.9, phase_deg=120)

    # During fault: large HV current (mostly fundamental, low harmonic)
    for k in range(fi, min(fi + EVENT * SPC, N_SAMPLES)):
        tk = (k - fi) / SRATE
        i_hv_a[k] += ip_fault * math.sin(2 * math.pi * FREQ * tk) * fault_pct
        i_hv_b[k] += ip_fault * math.sin(2 * math.pi * FREQ * tk - 2*math.pi/3) * fault_pct * 0.3
        i_hv_c[k] += ip_fault * math.sin(2 * math.pi * FREQ * tk + 2*math.pi/3) * fault_pct * 0.3

    # LV current: partially collapsed (fault consumes energy internally)
    i_lv_a = _sine(t, I_RATED_LV * 0.9, phase_deg=180) * (1 - fault_pct * 0.7)
    i_lv_b = _sine(t, I_RATED_LV * 0.9, phase_deg=60)  * (1 - fault_pct * 0.7)
    i_lv_c = _sine(t, I_RATED_LV * 0.9, phase_deg=-60) * (1 - fault_pct * 0.7)

    # Add noise
    for arr in [i_hv_a, i_hv_b, i_hv_c, i_lv_a, i_lv_b, i_lv_c]:
        arr += _noise(N_SAMPLES, sigma=NOISE_A)

    v_hv_a = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=0) * (1 - 0.3 * fault_pct)
    v_hv_b = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=-120)
    v_hv_c = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=120)

    return {
        'event_class': 'INTERNAL_FAULT',
        'i_hv_a': i_hv_a, 'i_hv_b': i_hv_b, 'i_hv_c': i_hv_c,
        'i_lv_a': i_lv_a, 'i_lv_b': i_lv_b, 'i_lv_c': i_lv_c,
        'v_hv_a': v_hv_a, 'v_hv_b': v_hv_b, 'v_hv_c': v_hv_c,
        'fault_idx': fi, 'comment': f'Internal fault {fault_pct:.0%} severity',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. THROUGH FAULT — external fault, balanced HV ↔ LV
# ─────────────────────────────────────────────────────────────────────────────

def _gen_through_fault(rng: np.random.Generator) -> dict:
    """Through fault: HV and LV currents balanced, differential ≈ 0."""
    t = _time_axis()
    fi = _fault_idx()

    imult = rng.uniform(3.0, 8.0)  # fault current multiplier

    i_hv_a = _sine(t, I_RATED_HV, phase_deg=0)
    i_hv_b = _sine(t, I_RATED_HV, phase_deg=-120)
    i_hv_c = _sine(t, I_RATED_HV, phase_deg=120)

    i_lv_a = _sine(t, I_RATED_LV, phase_deg=180)   # anti-phase (through transformer)
    i_lv_b = _sine(t, I_RATED_LV, phase_deg=60)
    i_lv_c = _sine(t, I_RATED_LV, phase_deg=-60)

    # Fault: both sides increase proportionally → differential stays zero
    for k in range(fi, min(fi + EVENT * SPC, N_SAMPLES)):
        tk = (k - fi) / SRATE
        delta_hv = I_RATED_HV * imult * math.sin(2 * math.pi * FREQ * tk)
        delta_lv = I_RATED_LV * imult * math.sin(2 * math.pi * FREQ * tk + math.pi)
        i_hv_a[k] += delta_hv
        i_lv_a[k] += delta_lv

    for arr in [i_hv_a, i_hv_b, i_hv_c, i_lv_a, i_lv_b, i_lv_c]:
        arr += _noise(N_SAMPLES, sigma=NOISE_A)

    # Voltage sag on HV side
    v_sag = rng.uniform(0.3, 0.6)
    v_hv_a = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=0)
    v_hv_a[fi:fi + EVENT * SPC] *= (1 - v_sag)
    v_hv_b = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=-120)
    v_hv_c = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=120)

    return {
        'event_class': 'THROUGH_FAULT',
        'i_hv_a': i_hv_a, 'i_hv_b': i_hv_b, 'i_hv_c': i_hv_c,
        'i_lv_a': i_lv_a, 'i_lv_b': i_lv_b, 'i_lv_c': i_lv_c,
        'v_hv_a': v_hv_a, 'v_hv_b': v_hv_b, 'v_hv_c': v_hv_c,
        'fault_idx': fi, 'comment': f'Through fault {imult:.1f}x rated',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. OVEREXCITATION — V/Hz condition, 5th harmonic dominant
# ─────────────────────────────────────────────────────────────────────────────

def _gen_overexcitation(rng: np.random.Generator) -> dict:
    """Overexcitation: elevated flux, 5th harmonic dominant in magnetising current."""
    t = _time_axis()
    fi = _fault_idx()

    vh_factor = rng.uniform(1.1, 1.3)  # overvoltage: 110–130%
    i5_factor = rng.uniform(0.10, 0.25)  # 5th harmonic as fraction of fundamental

    # HV current: magnetising current with 5th harmonic
    i_mag = I_RATED_HV * rng.uniform(0.05, 0.15)  # light load (magnetising only)

    i_hv_a = _sine(t, I_RATED_HV * 0.9, phase_deg=0)
    i_hv_b = _sine(t, I_RATED_HV * 0.9, phase_deg=-120)
    i_hv_c = _sine(t, I_RATED_HV * 0.9, phase_deg=120)

    for k in range(fi, N_SAMPLES):
        tk = (k - fi) / SRATE
        h5 = i_mag * i5_factor * math.sin(10 * math.pi * FREQ * tk)  # 5th harmonic
        i_hv_a[k] += h5 * vh_factor
        i_hv_b[k] += h5 * vh_factor
        i_hv_c[k] += h5 * vh_factor

    i_lv_a = _sine(t, I_RATED_LV * 0.9, phase_deg=180)
    i_lv_b = _sine(t, I_RATED_LV * 0.9, phase_deg=60)
    i_lv_c = _sine(t, I_RATED_LV * 0.9, phase_deg=-60)

    for arr in [i_hv_a, i_hv_b, i_hv_c, i_lv_a, i_lv_b, i_lv_c]:
        arr += _noise(N_SAMPLES, sigma=NOISE_A)

    # Elevated voltage
    v_amp = V_RATED_KV * 1000 * math.sqrt(2/3) * vh_factor
    v_hv_a = _sine(t, v_amp, phase_deg=0)
    v_hv_b = _sine(t, v_amp, phase_deg=-120)
    v_hv_c = _sine(t, v_amp, phase_deg=120)

    return {
        'event_class': 'OVEREXCITATION',
        'i_hv_a': i_hv_a, 'i_hv_b': i_hv_b, 'i_hv_c': i_hv_c,
        'i_lv_a': i_lv_a, 'i_lv_b': i_lv_b, 'i_lv_c': i_lv_c,
        'v_hv_a': v_hv_a, 'v_hv_b': v_hv_b, 'v_hv_c': v_hv_c,
        'fault_idx': fi, 'comment': f'Overexcitation V={vh_factor:.2f}pu H5={i5_factor:.0%}',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAL_OPERATE — CT saturation during external fault
# ─────────────────────────────────────────────────────────────────────────────

def _gen_mal_operate(rng: np.random.Generator) -> dict:
    """CT saturation: high through-current causes CT to saturate → false differential."""
    t = _time_axis()
    fi = _fault_idx()

    imult = rng.uniform(6.0, 12.0)  # very high through-fault current

    i_hv_a = _sine(t, I_RATED_HV, phase_deg=0)
    i_lv_a = _sine(t, I_RATED_LV, phase_deg=180)

    # High fault current — CT on LV saturates
    for k in range(fi, min(fi + int(EVENT * SPC * 0.7), N_SAMPLES)):
        tk = (k - fi) / SRATE
        # True current
        true_hv = I_RATED_HV * imult * math.sin(2 * math.pi * FREQ * tk)
        true_lv = I_RATED_LV * imult * math.sin(2 * math.pi * FREQ * tk + math.pi)
        # LV CT saturates — clips at saturation level + residual flux
        sat_lv = _ct_saturation_model(true_lv, sat_level=I_RATED_LV * 5.0, rng=rng)
        i_hv_a[k] = true_hv
        i_lv_a[k] = sat_lv

    i_hv_b = _sine(t, I_RATED_HV, phase_deg=-120) + _noise(N_SAMPLES, NOISE_A)
    i_hv_c = _sine(t, I_RATED_HV, phase_deg=120)  + _noise(N_SAMPLES, NOISE_A)
    i_lv_b = _sine(t, I_RATED_LV, phase_deg=60)   + _noise(N_SAMPLES, NOISE_A)
    i_lv_c = _sine(t, I_RATED_LV, phase_deg=-60)  + _noise(N_SAMPLES, NOISE_A)

    for arr in [i_hv_a, i_lv_a]:
        arr += _noise(N_SAMPLES, sigma=NOISE_A)

    v_hv_a = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=0)
    v_hv_b = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=-120)
    v_hv_c = _sine(t, V_RATED_KV * 1000 * math.sqrt(2/3), phase_deg=120)
    v_hv_a[fi:fi + EVENT * SPC] *= 0.5  # voltage sag from external fault

    return {
        'event_class': 'MAL_OPERATE',
        'i_hv_a': i_hv_a, 'i_hv_b': i_hv_b, 'i_hv_c': i_hv_c,
        'i_lv_a': i_lv_a, 'i_lv_b': i_lv_b, 'i_lv_c': i_lv_c,
        'v_hv_a': v_hv_a, 'v_hv_b': v_hv_b, 'v_hv_c': v_hv_c,
        'fault_idx': fi, 'comment': f'CT saturation {imult:.0f}x rated',
    }


def _ct_saturation_model(i_true: float, sat_level: float, rng: np.random.Generator) -> float:
    """
    Simple CT saturation model: clamps output at saturation level.
    Real CT behaviour is more complex (Jiles-Atherton), this is a first approximation.
    """
    if abs(i_true) > sat_level:
        return math.copysign(sat_level * (1.0 + rng.uniform(0, 0.05)), i_true)
    return i_true


# ─────────────────────────────────────────────────────────────────────────────
# COMTRADE file writer
# ─────────────────────────────────────────────────────────────────────────────

_STATIONS = [
    "GIS_CAWANG_150KV", "GI_BEKASI_150KV", "GIS_SURALAYA_150KV",
    "GI_CILEGON_150KV", "GI_DEPOK_150KV", "GI_TAMBUN_150KV",
    "GI_CIBATU_150KV",  "GI_GANDUL_150KV",
]

_RELAY_MODELS = [
    "ABB_RET615", "ABB_RET670", "SIEMENS_7UT85",
    "SIEMENS_7UT87", "SEL_387E", "GE_T60",
]


def _write_comtrade(data: dict, out_dir: Path, station: str, relay: str) -> Path:
    """Write a synthetic COMTRADE (.cfg + .dat) pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    name = f"{station}_{data['event_class']}_{ts}"
    cfg_path = out_dir / f"{name}.cfg"
    dat_path = out_dir / f"{name}.dat"

    # Channels to write
    channels: List[Tuple[str, np.ndarray, str, float]] = [
        # (name, array, unit, CT_primary)
        ('IW1A', data['i_hv_a'], 'A', CT_RATIO_HV),
        ('IW1B', data['i_hv_b'], 'A', CT_RATIO_HV),
        ('IW1C', data['i_hv_c'], 'A', CT_RATIO_HV),
        ('IW2A', data['i_lv_a'], 'A', CT_RATIO_LV),
        ('IW2B', data['i_lv_b'], 'A', CT_RATIO_LV),
        ('IW2C', data['i_lv_c'], 'A', CT_RATIO_LV),
        ('VW1A', data['v_hv_a'], 'kV', 150.0 / (150.0 / math.sqrt(3))),
        ('VW1B', data['v_hv_b'], 'kV', 150.0 / (150.0 / math.sqrt(3))),
        ('VW1C', data['v_hv_c'], 'kV', 150.0 / (150.0 / math.sqrt(3))),
    ]

    n_analog  = len(channels)
    n_status  = 2
    n_total   = n_analog + n_status
    n_samples = len(channels[0][1])

    # Time of first sample
    t0_str = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")[:26]

    # Write .cfg
    with open(cfg_path, 'w') as f:
        # Line 1: station, device id, version
        f.write(f"{station},{relay},1999\n")
        # Line 2: number of channels
        f.write(f"{n_total},{n_analog}A,{n_status}D\n")
        # Analog channel definitions
        for idx, (ch_name, arr, unit, ct_p) in enumerate(channels, start=1):
            a_val = 1.0 / ct_p   # COMTRADE scale factor
            f.write(f"{idx},{ch_name},,{unit},{a_val:.8f},0.000000,-99999,99999,{ct_p:.1f},1.0,S\n")
        # Status channel definitions
        f.write(f"{n_analog+1},87T_OPERATE,,0\n")
        f.write(f"{n_analog+2},87T_TRIP,,0\n")
        # Line freq
        f.write(f"{FREQ:.1f}\n")
        # Sampling rate
        f.write("1\n")
        f.write(f"{SRATE:.1f},{n_samples}\n")
        # Time
        f.write(f"{t0_str}\n")   # start of file
        f.write(f"{t0_str}\n")   # trigger time
        f.write("BINARY\n")

    # Normalise to int16 (COMTRADE binary)
    scale_factors = []
    normalised = []
    for _, arr, _, ct_p in channels:
        max_abs = np.max(np.abs(arr))
        scale = max_abs / 32000.0 if max_abs > 0 else 1.0
        scale_factors.append(scale)
        normalised.append((arr / scale).astype(np.int16))

    fi = data['fault_idx']

    with open(dat_path, 'wb') as f:
        for k in range(n_samples):
            # Sample number (1-indexed, uint32)
            f.write(struct.pack('<I', k + 1))
            # Timestamp in microseconds
            t_us = int(k * 1e6 / SRATE)
            f.write(struct.pack('<I', t_us))
            # Analog samples (int16 × n_analog)
            for ch_arr in normalised:
                f.write(struct.pack('<h', int(ch_arr[k])))
            # Status bits (uint16): 87T operates after fault inception
            if k >= fi:
                f.write(struct.pack('<H', 0b11))  # both bits high
            else:
                f.write(struct.pack('<H', 0b00))

    return cfg_path


# ─────────────────────────────────────────────────────────────────────────────
# Batch generator
# ─────────────────────────────────────────────────────────────────────────────

_GENERATORS = {
    'INRUSH':         _gen_inrush,
    'INTERNAL_FAULT': _gen_internal_fault,
    'THROUGH_FAULT':  _gen_through_fault,
    'OVEREXCITATION': _gen_overexcitation,
    'MAL_OPERATE':    _gen_mal_operate,
}


def generate_synthetic_dataset(
    n_per_class: int = 10,
    seed: int = 42,
    out_dir: Path = OUT_DIR,
) -> dict:
    """
    Generate n_per_class synthetic events for each of the 5 transformer event classes.

    Returns:
        dict mapping event_class → list of .cfg file paths
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    output = {}

    for event_class, gen_fn in _GENERATORS.items():
        class_dir = out_dir / event_class
        paths = []
        for i in range(n_per_class):
            station = random.choice(_STATIONS)
            relay   = random.choice(_RELAY_MODELS)
            data    = gen_fn(rng)
            cfg_path = _write_comtrade(data, class_dir, station, relay)
            paths.append(str(cfg_path))
            print(f"  [{event_class}] {cfg_path.name}")
        output[event_class] = paths

    total = sum(len(v) for v in output.values())
    print(f"\nGenerated {total} synthetic transformer events in {out_dir}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CSV label file writer (for training scaffold)
# ─────────────────────────────────────────────────────────────────────────────

def write_labels_csv(generated: dict, out_dir: Path = OUT_DIR) -> Path:
    """Write a labeled_transformer_events.csv from generated output dict."""
    import csv
    csv_path = out_dir / "labeled_transformer_events.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'event_class', 'source', 'confirmed'])
        for event_class, paths in generated.items():
            for p in paths:
                writer.writerow([p, event_class, 'SYNTHETIC', 'True'])
    print(f"Labels written to {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Generate synthetic transformer COMTRADE files for Phase 2 development.
            Produces n_per_class files for each of 5 event classes.
        """)
    )
    parser.add_argument('--n',    type=int, default=10, help='Files per class (default 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    parser.add_argument('--out',  type=str, default=str(OUT_DIR), help='Output directory')
    args = parser.parse_args()

    print(f"Generating {args.n} × 5 = {args.n * 5} synthetic transformer events...")
    generated = generate_synthetic_dataset(
        n_per_class=args.n,
        seed=args.seed,
        out_dir=Path(args.out),
    )
    write_labels_csv(generated, Path(args.out))
