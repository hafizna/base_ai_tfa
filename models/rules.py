"""
Tier 1 Rule Engine
==================
Deterministic rules applied BEFORE the ML classifier.
These rules handle cases with strong structural signals
that do not require training data.

Rules fire in priority order.  First match wins.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RuleResult:
    label: str                # e.g. "KONDUKTOR", "PERMANENT_UNKNOWN"
    confidence: float         # 0–1
    rule_name: str            # which rule fired
    evidence: str             # human-readable explanation


def apply_rules(row: dict) -> Optional[RuleResult]:
    """
    Apply all Tier 1 rules to a flat feature dict (one row of labeled_features.csv,
    or the equivalent dict produced by flatten_features in batch_extract.py).

    Returns a RuleResult if a rule fires, else None (→ pass to Tier 2).

    Args:
        row: dict with keys matching labeled_features.csv columns
    """

    fault_count      = int(row.get("fault_count", 1))
    faulted_phases   = str(row.get("faulted_phases", ""))
    duration_ms      = float(row.get("fault_duration_ms", 0))
    reclose_ok       = row.get("reclose_successful")   # True / False / None
    trip_type        = str(row.get("trip_type", ""))
    zone             = str(row.get("zone_operated", ""))
    peak_i           = float(row.get("peak_fault_current_a", 0) or 0)
    fault_type       = str(row.get("fault_type", "") or "").upper()
    soe_hint         = str(row.get("soe_faulted_phases", "") or "")
    soe_source       = str(row.get("soe_phase_hint_source", "") or "")
    soe_mismatch     = bool(row.get("soe_phase_mismatch", False))
    v_ratio_spread   = float(row.get("voltage_phase_ratio_spread_pu", 0) or 0)
    healthy_v_ratio  = float(row.get("healthy_phase_voltage_ratio", 0) or 0)
    v2_v1_ratio      = float(row.get("v2_v1_ratio", 0) or 0)
    ct_anomaly       = bool(row.get("ct_anomaly_detected", False))
    ct_anomaly_note  = str(row.get("ct_anomaly_evidence", "") or "")

    # ------------------------------------------------------------------
    # Rule 0 - CT / measurement anomaly
    #   Sustained, nearly flat current elevation on a phase whose voltage
    #   remains healthy is not a lightning impulse signature. It points to
    #   CT secondary / wiring / measurement trouble and must override PETIR.
    # ------------------------------------------------------------------
    if ct_anomaly:
        return RuleResult(
            label="PERALATAN / PROTEKSI",
            confidence=0.82,
            rule_name="ct_measurement_anomaly",
            evidence=(
                f"Indikasi anomali CT/pengukuran: {ct_anomaly_note}. "
                "Pola ini lebih konsisten dengan gangguan rangkaian CT/peralatan "
                "daripada sambaran petir impulsif."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 1 - KONDUKTOR: Fault On Reclose with phase change
    #   Evidence: multiple phases across the recording AND multiple fault
    #   events detected.  Phase change (A→B, B→A, etc.) on reclose is the
    #   structural-damage signature of broken tower / conductor.
    #
    #   Guard conditions:
    #   - reclose_ok must NOT be True: a successful reclose proves the fault
    #     cleared and the line is intact — cannot be a broken conductor.
    #   - fault_count capped at 20: values above this indicate a detection
    #     artefact (e.g. oscillation mis-counted as sub-faults).
    # ------------------------------------------------------------------
    phase_count = faulted_phases.count("+") + 1 if faulted_phases else 1
    multi_phase_for = phase_count == 2   # exactly 2 different phases = likely change on reclose

    reclose_succeeded = (reclose_ok is True or reclose_ok == "True")
    if (fault_count >= 2 and fault_count <= 20 and multi_phase_for
            and duration_ms > 80 and not reclose_succeeded):
        return RuleResult(
            label="KONDUKTOR / TOWER",
            confidence=0.85,
            rule_name="fault_on_reclose_phase_change",
            evidence=(
                f"fault_count={fault_count}, fasa={faulted_phases}, "
                f"dur={duration_ms:.0f}ms - perubahan fasa saat AR, "
                "diduga kerusakan mekanik pada konduktor/tower. "
                "Verifikasi dengan rekaman AR dan catatan operasi."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 1b - SOE / waveform phase mismatch
    #   If digital SOE explicitly selects a two-phase loop (L12/L23/L31)
    #   but waveform extraction says 3PH and the voltages are strongly
    #   asymmetric, treat PETIR as unsafe. This pattern often points to
    #   measurement/protection/equipment-origin anomalies or at least a
    #   waveform phase interpretation issue that needs manual review.
    # ------------------------------------------------------------------
    if (
        soe_mismatch
        and fault_type == "3PH"
        and duration_ms > 20
        and peak_i > 200
        and (v_ratio_spread >= 0.15 or healthy_v_ratio >= 0.92 or v2_v1_ratio >= 0.15)
    ):
        return RuleResult(
            label="PERALATAN / PROTEKSI",
            confidence=0.78,
            rule_name="soe_waveform_phase_mismatch",
            evidence=(
                f"SOE memilih loop {soe_hint} ({soe_source}), tetapi ekstraksi waveform "
                f"membaca {fault_type}. Asimetri tegangan tinggi "
                f"(spread={v_ratio_spread:.2f}, healthy_phase={healthy_v_ratio:.2f}, "
                f"V2/V1={v2_v1_ratio:.2f}) — indikasi anomali pengukuran/proteksi/peralatan; "
                "jangan terima label PETIR tanpa verifikasi manual."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 2 - Three-pole final trip with failed reclose
    #   Guard: peak_i > 50A ensures recording contains real fault current.
    #   Near-zero current (<50A) means this file captured dead time or the
    #   other-end relay recording — not reliable for AR outcome assessment.
    # ------------------------------------------------------------------
    if ((reclose_ok is False or reclose_ok == "False")
            and trip_type == "three_pole"
            and peak_i > 50):
        return RuleResult(
            label="GANGGUAN PERMANEN",
            confidence=0.75,
            rule_name="three_pole_failed_reclose",
            evidence=(
                f"trip_type=three_pole, reclose_successful=False, "
                f"peak_i={peak_i:.0f}A - "
                "gangguan permanen 3-fasa, AR gagal. "
                "Diduga kerusakan mekanik/konduktor — verifikasi kondisi jalur diperlukan."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 3 - Explicit failed reclose, cause unknown
    #   Guards:
    #   - duration_ms > 10: faults shorter than ~10ms are detection artefacts
    #     (real CB-cleared faults last at least ¼ cycle ≈ 5ms at 50Hz).
    #   - peak_i > 100A: recordings with near-zero current are dead-time or
    #     remote-end files that do not contain the fault current waveform;
    #     the AR outcome from such a file is not reliable.
    # ------------------------------------------------------------------
    if ((reclose_ok is False or reclose_ok == "False")
            and duration_ms > 10
            and peak_i > 100):
        return RuleResult(
            label="GANGGUAN PERMANEN",
            confidence=0.90,
            rule_name="explicit_failed_reclose",
            evidence=(
                f"reclose_successful=False, dur={duration_ms:.0f}ms, "
                f"peak_i={peak_i:.0f}A - gangguan permanen, AR gagal. "
                "Penyebab spesifik belum dapat ditentukan dari rekaman ini saja."
            ),
        )

    # No rule fired → pass to Tier 2
    return None
