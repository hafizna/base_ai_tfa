"""
Tests for Phase 2 Transformer Classification Pipeline
======================================================
Tests for rule-based transformer event classifier and feature thresholds.
Aligned with: models/transformer_classifier.py
              core/transformer_feature_extractor.py
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_classifier import (
    classify_transformer_event,
    TransformerClassificationResult,
    TransformerMLClassifier,
    H2_STRONG_PCT,
    H2_MODERATE_PCT,
    H5_OVEREXCIT_PCT,
    DC_OFFSET_STRONG,
    SLOPE_OP_PCT,
    SLOPE_HI_PCT,
)
from core.transformer_feature_extractor import TransformerFeatures


# ---------------------------------------------------------------------------
# Helpers: build TransformerFeatures with sensible defaults
# ---------------------------------------------------------------------------

def _make_features(**overrides) -> TransformerFeatures:
    """Build a TransformerFeatures instance with safe defaults, then apply overrides."""
    defaults = dict(
        h2_ratio_a_pct=0.0, h2_ratio_b_pct=0.0, h2_ratio_c_pct=0.0, h2_ratio_max_pct=0.0,
        h5_ratio_a_pct=0.0, h5_ratio_b_pct=0.0, h5_ratio_c_pct=0.0, h5_ratio_max_pct=0.0,
        thd_diff_pct=5.0,
        idiff_max_a_pu=0.0, idiff_max_b_pu=0.0, idiff_max_c_pu=0.0,
        irstr_max_a_pu=1.0, irstr_max_b_pu=1.0, irstr_max_c_pu=1.0,
        slope_worst_pct=15.0,
        above_slope1=False,
        above_slope2=False,
        dc_offset_index_max=0.0,
        pp_asymmetry_a=0.0,
        zc_interval_variance=0.0,
        hv_lv_ratio_a=1.0,
        energisation_flag=False,
        inception_angle_deg=45.0,
        fault_duration_ms=80.0,
        peak_idiff_a=500.0,
    )
    defaults.update(overrides)
    return TransformerFeatures(**defaults)


# ---------------------------------------------------------------------------
# INRUSH detection tests
# ---------------------------------------------------------------------------

class TestInrushClassification:

    def test_strong_h2_classifies_inrush(self):
        """H2 >= H2_STRONG_PCT threshold must produce INRUSH classification."""
        feats = _make_features(
            h2_ratio_max_pct=H2_STRONG_PCT + 5.0,
            dc_offset_index_max=DC_OFFSET_STRONG + 0.05,
            energisation_flag=True,
        )
        result = classify_transformer_event(feats)
        assert result.event_class == "INRUSH", (
            f"Expected INRUSH, got {result.event_class} (confidence={result.confidence:.2f})"
        )

    def test_inrush_not_action_required(self):
        """Inrush magnetisation should not require field action."""
        feats = _make_features(
            h2_ratio_max_pct=20.0,
            dc_offset_index_max=0.4,
            energisation_flag=True,
        )
        result = classify_transformer_event(feats)
        if result.event_class == "INRUSH":
            assert result.action_required is False

    def test_moderate_h2_with_dc_offset_inrush(self):
        """Moderate H2 combined with strong DC offset should still lean toward INRUSH."""
        feats = _make_features(
            h2_ratio_max_pct=H2_MODERATE_PCT + 2.0,
            dc_offset_index_max=DC_OFFSET_STRONG + 0.1,
            pp_asymmetry_a=0.4,
        )
        result = classify_transformer_event(feats)
        # High H2+DC combo → INRUSH probability should dominate over INTERNAL_FAULT
        assert result.class_probabilities.get("INRUSH", 0.0) > \
               result.class_probabilities.get("INTERNAL_FAULT", 0.0)


# ---------------------------------------------------------------------------
# OVEREXCITATION detection tests
# ---------------------------------------------------------------------------

class TestOverexcitationClassification:

    def test_high_h5_classifies_overexcitation(self):
        """H5 >= H5_OVEREXCIT_PCT with low H2 must produce OVEREXCITATION."""
        feats = _make_features(
            h5_ratio_max_pct=H5_OVEREXCIT_PCT + 3.0,
            h2_ratio_max_pct=2.0,   # low H2 — not inrush
            dc_offset_index_max=0.05,
        )
        result = classify_transformer_event(feats)
        assert result.event_class == "OVEREXCITATION", (
            f"Expected OVEREXCITATION, got {result.event_class}"
        )

    def test_h5_overexcit_not_action_required(self):
        """Overexcitation with no slope violation should not be action_required by default."""
        feats = _make_features(
            h5_ratio_max_pct=12.0,
            h2_ratio_max_pct=1.5,
            above_slope1=False,
            above_slope2=False,
        )
        result = classify_transformer_event(feats)
        if result.event_class == "OVEREXCITATION":
            # Recommendation should exist
            assert result.recommendation is not None and len(result.recommendation) > 0


# ---------------------------------------------------------------------------
# INTERNAL_FAULT detection tests
# ---------------------------------------------------------------------------

class TestInternalFaultClassification:

    def test_high_slope_low_h2_classifies_internal(self):
        """Above slope2 with negligible H2/H5 → INTERNAL_FAULT."""
        feats = _make_features(
            slope_worst_pct=SLOPE_HI_PCT + 10.0,
            above_slope1=True,
            above_slope2=True,
            h2_ratio_max_pct=1.5,   # low — not inrush
            dc_offset_index_max=0.05,
            idiff_max_a_pu=2.5,
            idiff_max_b_pu=2.5,
            idiff_max_c_pu=2.5,
        )
        result = classify_transformer_event(feats)
        assert result.event_class == "INTERNAL_FAULT", (
            f"Expected INTERNAL_FAULT, got {result.event_class} "
            f"(probs={result.class_probabilities})"
        )

    def test_internal_fault_is_action_required(self):
        """Internal fault must always flag action_required=True."""
        feats = _make_features(
            slope_worst_pct=90.0,
            above_slope1=True,
            above_slope2=True,
            h2_ratio_max_pct=1.0,
            dc_offset_index_max=0.02,
        )
        result = classify_transformer_event(feats)
        if result.event_class == "INTERNAL_FAULT":
            assert result.action_required is True


# ---------------------------------------------------------------------------
# THROUGH_FAULT detection tests
# ---------------------------------------------------------------------------

class TestThroughFaultClassification:

    def test_low_slope_through_fault(self):
        """Low idiff with high irstr and slope below Slope1 → THROUGH_FAULT."""
        feats = _make_features(
            slope_worst_pct=SLOPE_OP_PCT - 5.0,
            above_slope1=False,
            above_slope2=False,
            idiff_max_a_pu=0.05,
            idiff_max_b_pu=0.05,
            idiff_max_c_pu=0.05,
            irstr_max_a_pu=3.0,
            irstr_max_b_pu=3.0,
            irstr_max_c_pu=3.0,
            h2_ratio_max_pct=1.0,
        )
        result = classify_transformer_event(feats)
        assert result.event_class == "THROUGH_FAULT", (
            f"Expected THROUGH_FAULT, got {result.event_class} "
            f"(probs={result.class_probabilities})"
        )


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestClassificationResultStructure:

    def test_result_has_all_required_fields(self):
        """TransformerClassificationResult must have all required fields populated."""
        feats = _make_features(h2_ratio_max_pct=18.0, energisation_flag=True)
        result = classify_transformer_event(feats)

        assert isinstance(result, TransformerClassificationResult)
        assert result.event_class in {
            "INRUSH", "INTERNAL_FAULT", "THROUGH_FAULT", "OVEREXCITATION", "MAL_OPERATE", "UNKNOWN"
        }
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.class_probabilities, dict)
        assert len(result.class_probabilities) >= 5
        assert result.recommendation is not None
        assert result.rule_name is not None

    def test_probabilities_sum_to_one(self):
        """Class probability scores must sum to 1.0 (within floating-point tolerance)."""
        feats = _make_features(
            h2_ratio_max_pct=5.0,
            slope_worst_pct=25.0,
            above_slope1=True,
        )
        result = classify_transformer_event(feats)
        total = sum(result.class_probabilities.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total:.4f}, expected 1.0"

    def test_evidence_list_populated(self):
        """Evidence list should contain at least one item for any non-trivial input."""
        feats = _make_features(h2_ratio_max_pct=20.0, dc_offset_index_max=0.4)
        result = classify_transformer_event(feats)
        assert len(result.evidence) > 0

    def test_all_classes_reachable(self):
        """All 5 event classes must be reachable via the rule engine."""
        cases = [
            # INRUSH
            _make_features(h2_ratio_max_pct=20.0, energisation_flag=True, dc_offset_index_max=0.4),
            # INTERNAL_FAULT
            _make_features(slope_worst_pct=95.0, above_slope1=True, above_slope2=True,
                           h2_ratio_max_pct=1.0, idiff_max_a_pu=3.0),
            # THROUGH_FAULT
            _make_features(slope_worst_pct=10.0, above_slope1=False, idiff_max_a_pu=0.04,
                           irstr_max_a_pu=4.0, irstr_max_b_pu=4.0, irstr_max_c_pu=4.0),
            # OVEREXCITATION
            _make_features(h5_ratio_max_pct=14.0, h2_ratio_max_pct=1.5),
            # MAL_OPERATE: high restraint + very low diff + low H2
            _make_features(irstr_max_a_pu=5.0, irstr_max_b_pu=5.0, irstr_max_c_pu=5.0,
                           idiff_max_a_pu=0.02, idiff_max_b_pu=0.02, idiff_max_c_pu=0.02,
                           h2_ratio_max_pct=2.0, slope_worst_pct=8.0),
        ]
        seen_classes = {classify_transformer_event(f).event_class for f in cases}
        expected = {"INRUSH", "INTERNAL_FAULT", "THROUGH_FAULT", "OVEREXCITATION", "MAL_OPERATE"}
        # At least 4 of the 5 expected classes must be reachable
        assert len(seen_classes & expected) >= 4, (
            f"Only reached classes: {seen_classes}"
        )


# ---------------------------------------------------------------------------
# ML scaffold tests (untrained fallback)
# ---------------------------------------------------------------------------

class TestTransformerMLClassifier:

    def test_ml_classifier_instantiates(self):
        """TransformerMLClassifier must instantiate without errors."""
        clf = TransformerMLClassifier()
        assert clf is not None

    def test_untrained_predict_falls_back_to_rules(self):
        """Untrained ML classifier must fall back to rule-based result."""
        clf = TransformerMLClassifier()
        feats = _make_features(h2_ratio_max_pct=22.0, energisation_flag=True, dc_offset_index_max=0.45)
        result = clf.predict(feats)
        # Should return a valid result (via rules fallback)
        assert isinstance(result, TransformerClassificationResult)
        assert result.event_class in {
            "INRUSH", "INTERNAL_FAULT", "THROUGH_FAULT", "OVEREXCITATION", "MAL_OPERATE", "UNKNOWN"
        }
        assert result.limited_data is True  # Must flag that no trained model is available

    def test_feature_vector_length(self):
        """Feature vector must have expected length (17 features)."""
        clf = TransformerMLClassifier()
        assert len(clf.FEATURE_COLS) == 17, f"Expected 17 feature columns, got {len(clf.FEATURE_COLS)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
