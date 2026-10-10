"""Evidence-based fault-phase classification — synthetic tests.

Locks in the behavior added after a real Kebumen-Gombong SLG-R record was
misclassified as DLG (A+C): a healthy phase's small mutual-coupling current
crossed the old plain-amplitude threshold and got reported as a second
faulted phase. _evidence_based_fault_phases (webapp/api/routers/relay_21.py)
combines digital trip/pole-open evidence, per-phase ground-loop impedance
(the loop actually seeing the fault has small |Z|), and peak current — with
reclose evidence used ONLY as an optional corroborating boost, never a
requirement, since a record can be truncated before reclose completes and
still be an unambiguous single-phase fault.
"""

import math

import numpy as np

from webapp.api.routers.relay_21 import _compute_fault_classification
from webapp.api.ml_predict import _build_narrative_evidence, extract_ml_features


FREQ = 50.0
SR = 5000.0
N = 3000
PRE_FAULT_CYCLES = 10


def _wave(mag: float, ang_rad: float, t: np.ndarray) -> np.ndarray:
    return mag * np.cos(2 * math.pi * FREQ * t + ang_rad)


def _build_payload(
    *,
    fault_mag_a: float,
    fault_mag_b: float,
    fault_mag_c: float,
    fault_ang_a: float = 0.0,
    fault_ang_b: float = -2 * math.pi / 3,
    fault_ang_c: float = 2 * math.pi / 3,
    in_mag: float,
    status_channels: list[dict] | None = None,
) -> dict:
    """One synthetic record: clean 3-phase pre-fault load, then a step at a
    fixed inception sample to the given per-phase fault current magnitudes
    (fault_mag_b/c small relative to fault_mag_a models a real SLG-A record's
    mutual-coupling residue on the healthy phases, not a second fault)."""
    t = np.arange(N) / SR
    inception = int(round(SR * (PRE_FAULT_CYCLES / FREQ)))

    v_pre = 220.0
    i_pre = 5.0
    va = _wave(v_pre, 0.0, t)
    vb = _wave(v_pre, -2 * math.pi / 3, t)
    vc = _wave(v_pre, 2 * math.pi / 3, t)
    ia = _wave(i_pre, 0.0, t)
    ib = _wave(i_pre, -2 * math.pi / 3, t)
    ic = _wave(i_pre, 2 * math.pi / 3, t)

    # Post-fault: phase A collapses toward ground (large current, depressed
    # voltage); B/C keep near-nominal voltage but pick up a small coupling
    # current — this is the exact shape that used to fool the plain
    # amplitude threshold into calling a healthy phase "faulted".
    va_f = _wave(v_pre * 0.25, fault_ang_a, t)
    vb_f = _wave(v_pre * 0.95, -2 * math.pi / 3, t)
    vc_f = _wave(v_pre * 0.95, 2 * math.pi / 3, t)
    ia_f = _wave(fault_mag_a, fault_ang_a, t)
    ib_f = _wave(fault_mag_b, fault_ang_b, t)
    ic_f = _wave(fault_mag_c, fault_ang_c, t)
    in_f = _wave(in_mag, fault_ang_a, t)

    va[inception:] = va_f[inception:]
    vb[inception:] = vb_f[inception:]
    vc[inception:] = vc_f[inception:]
    ia[inception:] = ia_f[inception:]
    ib[inception:] = ib_f[inception:]
    ic[inception:] = ic_f[inception:]
    in_arr = np.zeros(N)
    in_arr[inception:] = in_f[inception:]

    status = [0] * N
    for i in range(inception, N):
        status[i] = 1

    channels = [
        {"name": "VA", "canonical_name": "VA", "unit": "V", "phase": "A", "measurement": "voltage", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": va.tolist()},
        {"name": "VB", "canonical_name": "VB", "unit": "V", "phase": "B", "measurement": "voltage", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": vb.tolist()},
        {"name": "VC", "canonical_name": "VC", "unit": "V", "phase": "C", "measurement": "voltage", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": vc.tolist()},
        {"name": "IA", "canonical_name": "IA", "unit": "A", "phase": "A", "measurement": "current", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": ia.tolist()},
        {"name": "IB", "canonical_name": "IB", "unit": "A", "phase": "B", "measurement": "current", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": ib.tolist()},
        {"name": "IC", "canonical_name": "IC", "unit": "A", "phase": "C", "measurement": "current", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": ic.tolist()},
        {"name": "IN", "canonical_name": "IN", "unit": "A", "phase": "N", "measurement": "current", "ct_primary": 1.0, "ct_secondary": 1.0, "samples": in_arr.tolist()},
    ]

    return {
        "station_name": "TEST",
        "frequency": FREQ,
        "time": t.tolist(),
        "trigger_time_iso": None,
        "start_time_iso": None,
        "analog_channels": channels,
        "status_channels": status_channels if status_channels is not None else [{"name": "TRIP", "samples": status}],
    }


def _status(name: str, on_from_frac: float = PRE_FAULT_CYCLES / FREQ + 0.01, off_at_frac: float | None = None) -> dict:
    """A digital channel that goes high at ``on_from_frac`` seconds and
    (optionally) back low at ``off_at_frac`` seconds — models a trip,
    pole-open, or reclose-close edge at an arbitrary time within the record."""
    t = np.arange(N) / SR
    samples = (t >= on_from_frac).astype(int)
    if off_at_frac is not None:
        samples[t >= off_at_frac] = 0
    return {"name": name, "samples": samples.tolist()}


def test_slg_with_small_healthy_phase_coupling_is_not_misread_as_dlg():
    """The exact real-world shape: phase A collapses hard, B/C keep a small
    non-zero current (mutual coupling, not a real fault) — must classify as
    SLG on A, not DLG(A+C), given real digital trip evidence (CB1.TrpA)."""
    payload = _build_payload(
        fault_mag_a=6000.0, fault_mag_b=500.0, fault_mag_c=700.0,
        in_mag=7000.0,
        status_channels=[_status("CB1.TrpA")],
    )
    result = _compute_fault_classification(payload)
    assert result["fault_code"] == "SLG"
    assert result["phases"] == ["A"]
    assert result["to_ground"] is True


def test_ai_diagnostic_uses_single_pole_relay_phase_instead_of_analog_dlg():
    """The AI diagnostic must use the same actual relay phase as the fault
    panel.  A single-pole trip on A overrides the false A+C analog threshold
    caused by healthy-phase mutual coupling, including in the FCT sentence."""
    payload = _build_payload(
        fault_mag_a=6000.0, fault_mag_b=500.0, fault_mag_c=700.0,
        in_mag=7000.0,
        status_channels=[_status("CB1.TrpA")],
    )

    row = extract_ml_features(payload, "21")
    evidence = _build_narrative_evidence(
        row,
        ranking=[],
        pred="PETIR",
        confidence=0.91,
        margin=0.8,
    )
    texts = [item["text"] for item in evidence]

    assert row["digital_trip_type"] == "single_pole"
    assert row["faulted_phases"] == "A"
    assert "Fasa A-N (Single Line to Ground)" in texts[0]
    assert any("fasa terganggu A-N" in text for text in texts)
    assert all("A+C" not in text and "Double Line to Ground" not in text for text in texts)


def test_slg_with_no_reclose_evidence_at_all_still_resolves_to_slg():
    """Central claim from the underlying analysis: absence of reclose
    evidence (record truncated before AR completes, or no AR configured at
    all) must NOT be read as evidence against the tripped phase, and must
    not push the classification toward a multi-phase fault. Only the trip
    bit is present here — no CB-close/reclose channel at all."""
    payload = _build_payload(
        fault_mag_a=6000.0, fault_mag_b=500.0, fault_mag_c=700.0,
        in_mag=7000.0,
        status_channels=[_status("CB1.TrpA", off_at_frac=None)],  # trip stays latched, no reclose channel present
    )
    result = _compute_fault_classification(payload)
    assert result["fault_code"] == "SLG"
    assert result["phases"] == ["A"]


def test_phase_select_a_channel_recognized():
    """Regression guard for the "Phase Select A" naming (no separator issue
    for this one — kept as a sanity check the evidence path still honors
    plain digital phase-select evidence, not just CB1.Trp{X})."""
    payload = _build_payload(
        fault_mag_a=6000.0, fault_mag_b=500.0, fault_mag_c=700.0,
        in_mag=7000.0,
        status_channels=[_status("Phase Select A")],
    )
    result = _compute_fault_classification(payload)
    assert result["phases"] == ["A"]


def test_genuine_dlg_both_phases_have_real_evidence():
    """A genuine double line-to-ground fault (both A and C carry large,
    comparable fault current and both trip) must still classify as DLG —
    the evidence combination must not suppress a real multi-phase fault,
    only a false one."""
    payload = _build_payload(
        fault_mag_a=6000.0, fault_mag_b=500.0, fault_mag_c=5800.0,
        in_mag=7000.0,
        status_channels=[_status("CB1.TrpA"), _status("CB1.TrpC")],
    )
    # Both faulted phases must have a voltage collapse too. The base SLG
    # fixture keeps C healthy; two trip poles alone do not establish DLG (F4).
    inception = int(round(SR * (PRE_FAULT_CYCLES / FREQ)))
    vc = next(ch for ch in payload["analog_channels"] if ch["canonical_name"] == "VC")
    vc["samples"][inception:] = [value * 0.25 / 0.95 for value in vc["samples"][inception:]]
    result = _compute_fault_classification(payload)
    assert result["fault_code"] == "DLG"
    assert set(result["phases"]) == {"A", "C"}


def test_ll_bc_badge_agrees_with_ai_reasoning_despite_three_pole_trip():
    """A three-pole trip describes breaker operation, not three faulted phases."""
    from tests.test_fault_reasoning import _record
    from webapp.api.record_analysis import build_record_analysis

    payload = _record(faulted=("B", "C"), ground=False, status=[
        ("TRIP R", [(40, 80)]), ("TRIP S", [(40, 80)]), ("TRIP T", [(40, 80)]),
    ])
    chain = build_record_analysis("test", payload).reasoning
    phases = next(c for c in chain["conclusions"] if c["key"] == "phases")["value"]
    result = _compute_fault_classification(payload)
    assert result["phases"] == phases["phases"]
    assert result["phases"] == ["B", "C"]
    assert result["fault_code"] == "LL"
    assert result["to_ground"] is False
