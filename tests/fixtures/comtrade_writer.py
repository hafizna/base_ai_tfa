"""Minimal ASCII COMTRADE (.cfg + .dat) writer for Stage 2 batch-upload tests.

Only implements the small subset of IEEE C37.111 needed to round-trip
through ``core.comtrade_parser.parse_comtrade``: 3 analog current channels,
1 status channel, ASCII data format, single sample rate. Not a general
COMTRADE writer — just enough to exercise real file-pairing/parsing code
paths in tests without hand-maintained binary fixtures.
"""

from __future__ import annotations

import numpy as np


def write_ascii_comtrade(
    station_name: str,
    rec_dev_id: str,
    samples: dict[str, np.ndarray],
    status: dict[str, np.ndarray],
    sample_rate_hz: float,
    freq: float = 50.0,
) -> tuple[str, str]:
    """Return (cfg_text, dat_text) for a single-sample-rate ASCII COMTRADE record.

    ``samples`` maps analog channel name -> array (current channels, unit A).
    ``status`` maps status channel name -> 0/1 array.
    """
    n = len(next(iter(samples.values())))
    analog_names = list(samples.keys())
    status_names = list(status.keys())

    lines = []
    lines.append(f"{station_name},{rec_dev_id},1999")
    lines.append(f"{len(analog_names) + len(status_names)},{len(analog_names)}A,{len(status_names)}D")
    for i, name in enumerate(analog_names, start=1):
        # An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS
        lines.append(f"{i},{name},,,A,1,0,0,-32767,32767,1,1,P")
    for i, name in enumerate(status_names, start=1):
        lines.append(f"{i},{name},,,0")
    lines.append(f"{freq}")
    lines.append("1")
    lines.append(f"{sample_rate_hz},{n}")
    lines.append("01/01/2026,00:00:00.000000")
    lines.append("01/01/2026,00:00:00.000000")
    lines.append("ASCII")
    lines.append("1")
    cfg_text = "\r\n".join(lines) + "\r\n"

    dat_lines = []
    for idx in range(n):
        row = [str(idx + 1), str(int(idx * (1_000_000 / sample_rate_hz)))]
        for name in analog_names:
            row.append(f"{samples[name][idx]:.4f}")
        for name in status_names:
            row.append(str(int(status[name][idx])))
        dat_lines.append(",".join(row))
    dat_text = "\r\n".join(dat_lines) + "\r\n"

    return cfg_text, dat_text


def synthetic_cfg_dat_bytes(
    fault_idx_fraction: float = 0.3,
    sr: float = 1200.0,
    dur_s: float = 1.0,
    station_name: str = "GOLDEN TEST",
    rec_dev_id: str = "SYNTH",
) -> tuple[bytes, bytes]:
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    fault_idx = int(fault_idx_fraction * sr)

    def sine(amp, phase_deg):
        base = amp * np.sin(2 * np.pi * 50.0 * t + np.deg2rad(phase_deg))
        return base

    ia = sine(3.7, 0)
    ia[fault_idx:] = sine(40.0, 0)[fault_idx:]
    ib = sine(3.7, -120)
    ic = sine(3.7, 120)

    trip = np.zeros(n, dtype=int)
    trip[fault_idx + 10:] = 1

    cfg_text, dat_text = write_ascii_comtrade(
        station_name, rec_dev_id,
        {"IA": ia, "IB": ib, "IC": ic},
        {"TRIP": trip},
        sample_rate_hz=sr,
    )
    return cfg_text.encode("ascii"), dat_text.encode("ascii")
