"""
Transformer Event Classifier — Phase 2
========================================
Knowledge-based rule engine for transformer 87T event classification.

Event classes:
  INRUSH          — Magnetising inrush current (normal operation, no fault)
  INTERNAL_FAULT  — True internal transformer fault (windings, core, bushing)
  THROUGH_FAULT   — External fault with high through-current (CT saturation possible)
  OVEREXCITATION  — Transformer overexcitation / V/Hz condition
  MAL_OPERATE     — Maloperation: false differential trip (CT saturation, mismatch)
  UNKNOWN         — Cannot determine from available features

Decision logic is derived entirely from published standards and relay manuals
(NO field data used) — designed to be replaced/augmented by a trained ML model
once sufficient labeled events are collected.

Key references:
  - IEEE C37.91-2008: "Guide for Protective Relay Applications to Power Transformers"
  - IEC 60255-111:2017: Transformer differential protection
  - ABB Application Guide 1MRK504049-UEN: RET615 commissioning
  - Siemens 7UT85/86/87 Manual v04 §5: protection functions
  - Horowitz & Phadke, "Power System Relaying" 4th ed. §9.3–9.6
  - PLN SPLN D3.012-1:2012

Threshold summary (conservative, suitable for unknown CT ratios):
  H2 ≥ 15%      → strong inrush indicator
  H2 ≥ 10%      → moderate inrush indicator (combined with DC offset)
  H5 ≥ 10%      → overexcitation indicator
  DC offset ≥ 0.35 → inrush waveform shape
  Slope > 20%   → differential element would operate (slope 1 knee)
  Slope > 80%   → high-set element would operate (slope 2 knee)
  Energisation + H2 → confirms inrush (one winding was de-energised)
  Phase diff ≠ 180° → winding asymmetry / internal fault indicator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (knowledge-based — update when field data available)
# ─────────────────────────────────────────────────────────────────────────────

# 2nd harmonic
H2_STRONG_PCT   = 15.0   # ≥ this → strong inrush signal
H2_MODERATE_PCT = 10.0   # ≥ this (+ DC offset) → moderate inrush signal

# 5th harmonic
H5_OVEREXCIT_PCT = 10.0  # ≥ this → overexcitation signal

# DC offset / waveform asymmetry
DC_OFFSET_STRONG  = 0.35  # ≥ this → strong inrush waveform
DC_OFFSET_MODERATE = 0.20  # ≥ this → moderate asymmetry

# Differential / restraint slope
SLOPE_OP_PCT  = 20.0  # > this → differential element operates
SLOPE_HI_PCT  = 80.0  # > this → high-set / instantaneous element

# Through-fault indicator: slope very low (differential ≈ 0)
SLOPE_THROUGH_MAX_PCT = 10.0

# HV-LV phase difference — normal is ~180° (current enters HV, exits LV)
PHASE_DIFF_NORMAL_DEG  = 180.0
PHASE_DIFF_TOLERANCE_DEG = 25.0   # ± this around 180° → normal / through-fault

# Minimum idiff for any classification (below this → no meaningful event)
IDIFF_MIN_PU = 0.05

# Zero-crossing variance threshold
ZC_VAR_INRUSH = 0.5   # ≥ this → irregular zero-crossings (inrush / mal-operate)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerClassificationResult:
    """Result of transformer event classification."""

    # Primary classification
    event_class: str = "UNKNOWN"   # one of the 6 classes above
    confidence: float = 0.0        # 0-1

    # Layered output (README: Arah Berikutnya #1)
    fault_origin: str = "UNKNOWN"
    # INTERNAL_TRAFO | EXTERNAL_DISTRIBUSI | EXTERNAL_TRANSMISI | PROTEKSI_PERALATAN | UNKNOWN
    fault_origin_reason: str = ""

    protection_assessment: str = "EVIDENCE_KURANG"
    # BENAR | TIDAK_SELEKTIF | EVIDENCE_KURANG
    protection_assessment_reason: str = ""

    # Evidence
    rule_name: str = ""
    evidence: str = ""

    # Secondary: probability estimates per class (sum approx 1)
    class_probabilities: dict = field(default_factory=lambda: {
        'INRUSH':         0.0,
        'INTERNAL_FAULT': 0.0,
        'THROUGH_FAULT':  0.0,
        'OVEREXCITATION': 0.0,
        'MAL_OPERATE':    0.0,
        'UNKNOWN':        0.0,
    })

    # Recommendations for operator
    recommendation: str = ""
    action_required: bool = False

    # Phase-level detail
    worst_h2_phase: str = ""
    worst_h2_pct: Optional[float] = None
    worst_h5_pct: Optional[float] = None
    worst_dc_offset: Optional[float] = None
    slope_point: Optional[float] = None   # |Idiff| / |Irestraint| %

    # Data quality flags
    limited_data: bool = False     # True when key channels missing
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_transformer_event(features) -> TransformerClassificationResult:
    """
    Apply knowledge-based rules to classify a transformer event.

    Args:
        features: TransformerFeatures dataclass from transformer_feature_extractor

    Returns:
        TransformerClassificationResult
    """
    result = TransformerClassificationResult()
    result.warnings = list(features.warnings)

    # ── Safety check: insufficient data ──────────────────────────────────────
    has_harmonic = features.h2_ratio_max_pct is not None
    has_diff     = features.idiff_max_pu is not None
    has_shape    = features.dc_offset_index_max is not None

    if not has_harmonic and not has_diff:
        result.event_class = "UNKNOWN"
        result.confidence  = 0.0
        result.evidence    = "Insufficient channel data for classification"
        result.limited_data = True
        result.class_probabilities['UNKNOWN'] = 1.0
        result.recommendation = ("Pastikan channel diferensial / HV-LV tersedia "
                                 "dalam file COMTRADE untuk analisis lebih lanjut.")
        return result

    if not has_harmonic:
        result.warnings.append("Harmonic analysis unavailable (no differential/HV channel)")
        result.limited_data = True
    if not has_diff:
        result.warnings.append("Differential magnitude unavailable")
        result.limited_data = True

    # ── Collect evidence scores ───────────────────────────────────────────────
    scores = {k: 0.0 for k in result.class_probabilities}

    h2   = features.h2_ratio_max_pct or 0.0
    h5   = features.h5_ratio_max_pct or 0.0
    dc   = features.dc_offset_index_max or 0.0
    slp  = features.slope_worst_pct or 0.0
    enrg = features.energisation_flag
    i_diff_pu = features.idiff_max_pu or 0.0
    i_rstr_pu = features.irstr_max_pu or 0.0
    phase_diff = features.hv_lv_phase_diff_a_deg
    zc_var = features.zc_interval_variance or 0.0

    # Record worst-phase details for report
    result.worst_h2_pct  = h2
    result.worst_h5_pct  = h5
    result.worst_dc_offset = dc
    result.slope_point   = slp

    # ─────────────────────────────────────────────────────────────────────────
    # RULE 1 — INRUSH (highest priority when 2nd harmonic high)
    #
    # Criterion: IEEE C37.91-2008 §6.3
    #   "2nd harmonic content > 15-20% of fundamental is traditionally used
    #    to distinguish inrush from internal faults"
    # ─────────────────────────────────────────────────────────────────────────
    inrush_score = 0.0

    if h2 >= H2_STRONG_PCT:
        inrush_score += 0.6
    elif h2 >= H2_MODERATE_PCT:
        inrush_score += 0.35

    if dc >= DC_OFFSET_STRONG:
        inrush_score += 0.25
    elif dc >= DC_OFFSET_MODERATE:
        inrush_score += 0.12

    if enrg:
        inrush_score += 0.20  # energisation context confirms inrush

    if zc_var >= ZC_VAR_INRUSH:
        inrush_score += 0.10  # missing half-cycles = inrush waveform

    scores['INRUSH'] = min(inrush_score, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # RULE 2 — OVEREXCITATION
    #
    # Criterion: Siemens 7UT8 Manual §5.8
    #   "5th harmonic > 10% with differential current indicates V/Hz condition"
    # ─────────────────────────────────────────────────────────────────────────
    overexcit_score = 0.0
    if h5 >= H5_OVEREXCIT_PCT:
        overexcit_score += 0.55
        if h2 < H2_MODERATE_PCT:  # pure overexcitation has low 2nd harmonic
            overexcit_score += 0.20
    if h5 >= 15.0:  # strong overexcitation
        overexcit_score += 0.15

    scores['OVEREXCITATION'] = min(overexcit_score, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # RULE 3 — THROUGH FAULT
    #
    # Criterion: low differential, high restraint
    #   |Idiff| / |Irestraint| < 10% (well below slope 1 knee)
    #   HV–LV phase difference ≈ 180° (normal load / fault transfer)
    # ─────────────────────────────────────────────────────────────────────────
    through_score = 0.0
    if slp > 0 and slp < SLOPE_THROUGH_MAX_PCT:
        through_score += 0.50
    elif i_rstr_pu > 0.5 and i_diff_pu < 0.1:
        through_score += 0.45

    if phase_diff is not None:
        angle_err = abs(abs(phase_diff) - PHASE_DIFF_NORMAL_DEG)
        if angle_err <= PHASE_DIFF_TOLERANCE_DEG:
            through_score += 0.20

    if h2 < H2_MODERATE_PCT and h5 < H5_OVEREXCIT_PCT:
        through_score += 0.15   # harmonic content consistent with external fault

    scores['THROUGH_FAULT'] = min(through_score, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # RULE 4 — INTERNAL FAULT
    #
    # Criterion: IEEE C37.91-2008 §6.4
    #   Differential > restraint slope (slope_worst > 20%)
    #   AND 2nd harmonic is NOT high (excludes inrush)
    #   AND 5th harmonic is NOT high (excludes overexcitation)
    # ─────────────────────────────────────────────────────────────────────────
    internal_score = 0.0

    if features.above_slope1:
        internal_score += 0.50
    if features.above_slope2:
        internal_score += 0.20  # high-set operated → severe fault

    # Penalise if inrush or overexcitation indicators are present
    if h2 >= H2_STRONG_PCT:
        internal_score -= 0.35
    elif h2 >= H2_MODERATE_PCT:
        internal_score -= 0.15

    if dc >= DC_OFFSET_STRONG:
        internal_score -= 0.20

    if h5 >= H5_OVEREXCIT_PCT:
        internal_score -= 0.15

    # Phase difference asymmetry → winding fault indicator
    if phase_diff is not None:
        angle_err = abs(abs(phase_diff) - PHASE_DIFF_NORMAL_DEG)
        if angle_err > PHASE_DIFF_TOLERANCE_DEG * 2:
            internal_score += 0.15

    scores['INTERNAL_FAULT'] = max(0.0, min(internal_score, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # RULE 5 — MALOPERATION
    #
    # Criterion:
    #   CT saturation: external fault (low H2 for inrush, but asymmetric waveform)
    #   Differential trip during through-fault (slope exceeded due to saturation)
    # ─────────────────────────────────────────────────────────────────────────
    mal_score = 0.0

    # CT saturation: high current but moderate harmonic distortion
    if i_rstr_pu > 1.0 and i_diff_pu < 0.3:
        # High through-current + small differential → likely CT saturation
        mal_score += 0.35

    if zc_var >= ZC_VAR_INRUSH and h2 < H2_MODERATE_PCT and dc >= DC_OFFSET_MODERATE:
        # Distorted waveform (saturation) without 2nd harmonic signature of inrush
        mal_score += 0.25

    if features.above_slope1 and slp < SLOPE_HI_PCT and h2 < H2_MODERATE_PCT:
        # Slope 1 exceeded but no clear inrush harmonic → saturation-driven trip
        mal_score += 0.15

    scores['MAL_OPERATE'] = min(mal_score, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Normalise scores → probabilities
    # ─────────────────────────────────────────────────────────────────────────
    total = sum(scores.values())
    if total > 0:
        probs = {k: v / total for k, v in scores.items()}
    else:
        probs = {k: 0.0 for k in scores}
        probs['UNKNOWN'] = 1.0

    result.class_probabilities = probs

    # ─────────────────────────────────────────────────────────────────────────
    # Pick best class
    # ─────────────────────────────────────────────────────────────────────────
    best_class = max(probs, key=lambda k: probs[k])
    best_prob  = probs[best_class]

    # Minimum confidence threshold — if no class is dominant, return UNKNOWN
    if best_prob < 0.25 or best_class == 'UNKNOWN':
        result.event_class = "UNKNOWN"
        result.confidence  = best_prob
        result.evidence    = "No dominant classification pattern detected. Manual review required."
        result.limited_data = True
    else:
        result.event_class = best_class
        result.confidence  = best_prob
        result.rule_name, result.evidence = _build_evidence(best_class, features, h2, h5, dc, slp)

    # ─────────────────────────────────────────────────────────────────────────
    # Set recommendation
    # ─────────────────────────────────────────────────────────────────────────
    result.recommendation, result.action_required = _recommendation(result.event_class, result.confidence)

    # ── Layered output: fault_origin + protection_assessment ─────────────────
    result.fault_origin, result.fault_origin_reason, \
        result.protection_assessment, result.protection_assessment_reason = \
        _derive_layered_output(result.event_class, result.confidence,
                               result.limited_data, features)

    # Log
    logger.info(
        f"Transformer classification: {result.event_class} "
        f"({result.confidence:.0%}) | H2={h2:.1f}% H5={h5:.1f}% "
        f"DC={dc:.2f} slope={slp:.1f}% | origin={result.fault_origin} "
        f"protection={result.protection_assessment}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layered output derivation (README: Arah Berikutnya #1)
# ─────────────────────────────────────────────────────────────────────────────

def _derive_layered_output(event_class: str, confidence: float,
                            limited_data: bool, features) -> tuple:
    """
    Derive fault_origin and protection_assessment from event_class.

    Returns (fault_origin, fault_origin_reason,
             protection_assessment, protection_assessment_reason).

    fault_origin:
      INTERNAL_TRAFO        — fault/phenomenon inside the transformer
      EXTERNAL_DISTRIBUSI   — external fault, distribution side
      EXTERNAL_TRANSMISI    — external fault, transmission side
      PROTEKSI_PERALATAN    — protection or CT/equipment issue
      UNKNOWN               — cannot determine

    protection_assessment:
      BENAR                 — protection operated correctly
      TIDAK_SELEKTIF        — non-selective (false or unnecessary trip)
      EVIDENCE_KURANG       — insufficient evidence to assess
    """
    if limited_data or event_class == 'UNKNOWN' or confidence < 0.25:
        return (
            'UNKNOWN',
            'Data tidak mencukupi untuk menentukan origin gangguan.',
            'EVIDENCE_KURANG',
            'Confidence rendah atau data terbatas — tidak dapat menilai selektivitas proteksi.',
        )

    if event_class == 'INRUSH':
        return (
            'INTERNAL_TRAFO',
            'Inrush magnetisasi terjadi di dalam transformator saat energisasi. '
            'Bukan gangguan — fenomena normal operasi.',
            'BENAR',
            'Relay mendeteksi second harmonic dengan benar. '
            'Jika rele tidak trip, proteksi bekerja sesuai filosofi restraint harmonik.',
        )

    if event_class == 'INTERNAL_FAULT':
        return (
            'INTERNAL_TRAFO',
            'Arus diferensial melebihi slope-1 tanpa kandungan harmonik tinggi, '
            'konsisten dengan gangguan belitan atau inti transformator.',
            'BENAR',
            '87T bekerja sesuai fungsi — arus diferensial melebihi batas operate '
            'dengan indikasi kuat gangguan internal.',
        )

    if event_class == 'THROUGH_FAULT':
        # Through-fault origin: LV side is usually distribution, HV side is transmission.
        # Without additional relay context we can only say "external" generically.
        irstr = getattr(features, 'irstr_max_pu', None) or 0.0
        if irstr > 2.0:
            origin = 'EXTERNAL_DISTRIBUSI'
            origin_reason = (
                f'Arus restraint sangat tinggi ({irstr:.1f} pu) konsisten dengan '
                'through-fault dari gangguan distribusi sisi LV.'
            )
        else:
            origin = 'EXTERNAL_TRANSMISI'
            origin_reason = (
                f'Arus restraint {irstr:.1f} pu — through-fault kemungkinan '
                'dari sisi transmisi atau bus. Konfirmasi dengan relay feeder/OCR.'
            )
        return (
            origin,
            origin_reason,
            'BENAR',
            'Rele diferensial berhasil restraint — tidak trip saat through-fault. '
            'Perlu verifikasi saturasi CT pada arus tinggi.',
        )

    if event_class == 'OVEREXCITATION':
        return (
            'INTERNAL_TRAFO',
            'Kondisi V/Hz berlebih (harmonic ke-5 tinggi) — sumber di sistem '
            'tegangan atau AVR, berdampak pada inti transformator.',
            'BENAR',
            'Proteksi V/Hz atau IDMT bekerja sesuai kondisi overeksitasi. '
            'Review setting AVR dan tap changer.',
        )

    if event_class == 'MAL_OPERATE':
        return (
            'PROTEKSI_PERALATAN',
            'Indikasi saturasi CT akibat arus gangguan eksternal tinggi yang '
            'menciptakan arus diferensial semu — bukan gangguan internal trafo.',
            'TIDAK_SELEKTIF',
            'Rele trip tidak selektif: gangguan bukan internal trafo tetapi rele '
            '87T operate. Periksa rasio CT, setting slope, dan stabilitas diferensial.',
        )

    # Fallback
    return ('UNKNOWN', '', 'EVIDENCE_KURANG', '')


# ─────────────────────────────────────────────────────────────────────────────
# Evidence builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_evidence(event_class: str, features, h2: float, h5: float, dc: float, slp: float) -> tuple:
    """Return (rule_name, evidence_string) for the given classification."""

    if event_class == 'INRUSH':
        parts = []
        if h2 >= 15:
            parts.append(f"2nd harmonic {h2:.1f}% (threshold 15%)")
        if dc >= 0.35:
            parts.append(f"DC offset index {dc:.2f} (threshold 0.35)")
        if features.energisation_flag:
            parts.append("energisation detected (LV pre-fault current ≈ 0)")
        return ("RULE_INRUSH_H2", "; ".join(parts) or f"2nd harmonic {h2:.1f}%")

    if event_class == 'INTERNAL_FAULT':
        parts = []
        if features.above_slope2:
            parts.append(f"slope {slp:.1f}% exceeds high-set (80%)")
        elif features.above_slope1:
            parts.append(f"slope {slp:.1f}% exceeds slope-1 (20%)")
        if h2 < 10:
            parts.append(f"2nd harmonic {h2:.1f}% (low, consistent with fault)")
        return ("RULE_INTERNAL_FAULT_SLOPE", "; ".join(parts) or f"slope {slp:.1f}%")

    if event_class == 'THROUGH_FAULT':
        parts = []
        if slp > 0:
            parts.append(f"slope {slp:.1f}% (below 10% — differential ≈ 0)")
        if features.irstr_max_pu:
            parts.append(f"restraint current {features.irstr_max_pu:.2f} pu (high)")
        return ("RULE_THROUGH_FAULT_SLOPE", "; ".join(parts) or "low differential vs high restraint")

    if event_class == 'OVEREXCITATION':
        parts = [f"5th harmonic {h5:.1f}% (threshold 10%)"]
        if h2 < 10:
            parts.append(f"2nd harmonic {h2:.1f}% (low, rules out inrush)")
        return ("RULE_OVEREXCITATION_H5", "; ".join(parts))

    if event_class == 'MAL_OPERATE':
        parts = []
        if features.irstr_max_pu and features.irstr_max_pu > 1.0:
            parts.append(f"restraint {features.irstr_max_pu:.2f} pu (CT saturation likely)")
        if slp > 0:
            parts.append(f"slope {slp:.1f}% (false trip from saturation)")
        return ("RULE_MAL_OPERATE_SAT", "; ".join(parts) or "CT saturation pattern")

    return ("RULE_UNKNOWN", "Insufficient evidence for classification")


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────────────────────

_RECOMMENDATIONS = {
    'INRUSH': (
        "Arus inrush magnetisasi — operasi normal transformator. "
        "Tidak diperlukan tindakan korektif. Verifikasi bahwa rele tidak memasukkan "
        "transformator ke dalam lockout. Periksa setting second-harmonic restraint.",
        False
    ),
    'INTERNAL_FAULT': (
        "GANGGUAN INTERNAL — Isolasi transformator segera. "
        "Lakukan pengujian: DGA (dissolved gas analysis), IR test, turns-ratio test, "
        "dan inspeksi fisik sebelum energisasi kembali. "
        "Koordinasi dengan tim pemeliharaan dan proteksi.",
        True
    ),
    'THROUGH_FAULT': (
        "Gangguan eksternal (through-fault) dengan arus tinggi melewati transformator. "
        "Periksa kondisi isolasi pasca-gangguan (thermal aging). "
        "Verifikasi integritas CT — kemungkinan saturasi pada arus tinggi.",
        False
    ),
    'OVEREXCITATION': (
        "Kondisi tegangan lebih / V/Hz berlebih terdeteksi. "
        "Periksa setting AVR (Automatic Voltage Regulator) dan tap changer. "
        "Verifikasi tegangan sistem pasca-gangguan. "
        "Setting rele V/Hz harus dikalibrasi ulang jika sering terjadi.",
        True
    ),
    'MAL_OPERATE': (
        "KEMUNGKINAN MALOPERATE — Periksa rasio dan polaritas CT sebelum energisasi. "
        "Kemungkinan saturasi CT akibat arus gangguan eksternal yang tinggi. "
        "Lakukan studi stabilisasi diferensial dan review setting slope. "
        "Koordinasi dengan departemen proteksi.",
        True
    ),
    'UNKNOWN': (
        "Tidak dapat menentukan jenis event. Diperlukan review manual oleh insinyur proteksi. "
        "Pastikan channel HV, LV, diferensial, dan restraint tersedia dalam file COMTRADE.",
        False
    ),
}


def _recommendation(event_class: str, confidence: float) -> tuple:
    """Return (recommendation_text, action_required)."""
    rec, action = _RECOMMENDATIONS.get(event_class, _RECOMMENDATIONS['UNKNOWN'])
    if confidence < 0.50:
        rec = f"[Kepercayaan rendah: {confidence:.0%}] " + rec
    return rec, action


# ─────────────────────────────────────────────────────────────────────────────
# ML scaffold — placeholder for when labeled data becomes available
# ─────────────────────────────────────────────────────────────────────────────

class TransformerMLClassifier:
    """
    Placeholder for a trained ML classifier.

    Intended workflow once data is available:
      1. Collect labeled transformer events (target: ≥ 30 per class)
      2. Run batch_extract_transformer.py to build features CSV
      3. Train this classifier using train_transformer.py
      4. Replace classify_transformer_event() with this model's predictions
         (keeping rule-based as Tier 1 fallback)

    Feature vector (17 features):
      h2_ratio_max_pct, h5_ratio_max_pct, thd_diff_pct,
      idiff_max_pu, irstr_max_pu, slope_worst_pct,
      dc_offset_index_max, pp_asymmetry_a, zc_interval_variance,
      hv_lv_ratio_a, hv_lv_phase_diff_a_deg,
      energisation_flag, lv_prefault_irms_pu,
      inception_angle_deg, fault_duration_ms,
      above_slope1 (bool→int), above_slope2 (bool→int)

    Recommended models (in order of priority given data scarcity):
      1. Random Forest with class_weight='balanced' (robust to imbalance)
      2. XGBoost with scale_pos_weight tuning
      3. GaussianProcessClassifier (calibrated uncertainty)
      4. 1D-CNN on raw waveform (when ≥ 200 samples per class)
    """

    FEATURE_COLS = [
        'h2_ratio_max_pct', 'h5_ratio_max_pct', 'thd_diff_pct',
        'idiff_max_pu', 'irstr_max_pu', 'slope_worst_pct',
        'dc_offset_index_max', 'pp_asymmetry_a', 'zc_interval_variance',
        'hv_lv_ratio_a', 'hv_lv_phase_diff_a_deg',
        'energisation_flag', 'lv_prefault_irms_pu',
        'inception_angle_deg', 'fault_duration_ms',
        'above_slope1', 'above_slope2',
    ]

    TARGET_COL = 'event_class'

    CLASSES = ['INRUSH', 'INTERNAL_FAULT', 'THROUGH_FAULT',
               'OVEREXCITATION', 'MAL_OPERATE']

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_trained = False

    def build_feature_vector(self, features) -> list:
        """Convert TransformerFeatures to 17-element feature vector."""
        from core.transformer_feature_extractor import features_to_dict
        d = features_to_dict(features)
        row = []
        for col in self.FEATURE_COLS:
            val = d.get(col)
            if val is None:
                row.append(0.0)
            elif isinstance(val, bool):
                row.append(float(val))
            else:
                try:
                    row.append(float(val))
                except (TypeError, ValueError):
                    row.append(0.0)
        return row

    def train(self, features_csv: str) -> None:
        """
        Train classifier from labeled CSV.
        CSV must have columns in FEATURE_COLS + TARGET_COL.
        """
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score
        from imblearn.over_sampling import SMOTE

        df = pd.read_csv(features_csv)
        df = df.dropna(subset=[self.TARGET_COL])
        df[self.FEATURE_COLS] = df[self.FEATURE_COLS].fillna(0)

        X = df[self.FEATURE_COLS].values
        y = df[self.TARGET_COL].values

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        # SMOTE for class balance (needs ≥ 6 samples per class)
        try:
            sm = SMOTE(k_neighbors=3, random_state=42)
            X_res, y_res = sm.fit_resample(X, y_enc)
        except Exception:
            X_res, y_res = X, y_enc

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
        )
        self.model.fit(X_res, y_res)

        # Cross-validation
        scores = cross_val_score(self.model, X, y_enc, cv=min(5, len(y) // 5 or 2))
        logger.info(f"TransformerMLClassifier CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        self.is_trained = True

    def predict(self, features) -> TransformerClassificationResult:
        """Run ML inference (falls back to rules if not trained)."""
        if not self.is_trained or self.model is None:
            logger.warning("TransformerMLClassifier not trained — falling back to rule engine")
            return classify_transformer_event(features)

        import numpy as np
        fv = self.build_feature_vector(features)
        X = np.array(fv).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        classes = self.label_encoder.classes_

        result = TransformerClassificationResult()
        result.class_probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
        best_idx = int(np.argmax(proba))
        result.event_class = classes[best_idx]
        result.confidence  = float(proba[best_idx])
        result.rule_name   = "ML_RANDOM_FOREST"
        result.evidence    = f"Trained model: {result.event_class} ({result.confidence:.0%})"
        result.recommendation, result.action_required = _recommendation(
            result.event_class, result.confidence)
        return result

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({'model': self.model, 'encoder': self.label_encoder}, path)
        logger.info(f"TransformerMLClassifier saved to {path}")

    def load(self, path: str) -> None:
        import joblib
        obj = joblib.load(path)
        self.model = obj['model']
        self.label_encoder = obj['encoder']
        self.is_trained = True
        logger.info(f"TransformerMLClassifier loaded from {path}")
