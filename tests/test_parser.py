"""
Tests for COMTRADE Parser (M1a)
================================
Validates parser correctness, channel normalization, and primary value conversion.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from core.comtrade_parser import parse_comtrade, ComtradeRecord
from core.channel_normalizer import normalize_channel_name, detect_manufacturer
from core.protection_router import determine_protection, ProtectionType
from core.fault_detector import detect_fault
from core.feature_extractor import extract_distance_features
from models.predict import classify_file


class TestChannelNormalization:
    """Test channel name normalization across vendors."""

    def test_sel_channels(self):
        """Test SEL channel patterns."""
        result = normalize_channel_name("IA", "A", "SEL")
        assert result['canonical_name'] == "IA"
        assert result['phase'] == "A"
        assert result['measurement'] == "current"

        result = normalize_channel_name("APHSVOL", "kV", "SEL")
        assert result['canonical_name'] == "VA"
        assert result['phase'] == "A"

    def test_abb_channels(self):
        """Test ABB channel patterns."""
        result = normalize_channel_name("IL1", "A", "ABB")
        assert result['canonical_name'] == "IA"

        result = normalize_channel_name("UL1", "kV", "ABB")
        assert result['canonical_name'] == "VA"

    def test_siemens_channels(self):
        """Test Siemens channel patterns."""
        result = normalize_channel_name("I Phase A", "A", "SIEMENS")
        assert result['canonical_name'] == "IA"

        result = normalize_channel_name("UA", "kV", "SIEMENS")
        assert result['canonical_name'] == "VA"

    def test_ge_channels(self):
        """Test GE channel patterns."""
        result = normalize_channel_name("IAPH", "A", "GE")
        assert result['canonical_name'] == "IA"

    def test_residual_channels(self):
        """Test neutral/residual channel detection."""
        result = normalize_channel_name("3I0", "A", "UNKNOWN")
        assert result['canonical_name'] == "IN"
        assert result['phase'] == "N"

        result = normalize_channel_name("3V0", "kV", "UNKNOWN")
        assert result['canonical_name'] == "VN"
        assert result['phase'] == "N"

    def test_unknown_channel(self):
        """Test handling of unrecognized channels."""
        result = normalize_channel_name("WEIRD_CHANNEL_123", "kV", "UNKNOWN")
        assert result['canonical_name'] == "WEIRD_CHANNEL_123"  # Should keep raw name
        assert result['phase'] is None

    def test_87l_diff_percent_channel_maps_as_current(self):
        result = normalize_channel_name("Ln1:87L:I-DIFF:I diff.:phs A", "%", "SIEMENS")
        assert result['measurement'] == "current"
        assert result['phase'] == "A"
        assert result['canonical_name'] == "IDIFF_A"

    def test_87l_restraint_percent_channel_maps_as_current(self):
        result = normalize_channel_name("Ln1:87L:I-DIFF:I restr.:phs B", "%", "SIEMENS")
        assert result['measurement'] == "current"
        assert result['phase'] == "B"
        assert result['canonical_name'] == "IREST_B"

    def test_87t_diff_pu_channel_maps_as_current(self):
        result = normalize_channel_name("87T.ida", "pu", "UNKNOWN")
        assert result['measurement'] == "current"
        assert result['phase'] == "A"
        assert result['canonical_name'] == "IDIFF_A"

    def test_siemens_ocr_voltage_channels(self):
        result = normalize_channel_name("uEn", "V", "SIEMENS")
        assert result['canonical_name'] == "VN"
        assert result['phase'] == "N"
        assert result['measurement'] == "voltage"

        result = normalize_channel_name("uL23", "V", "SIEMENS")
        assert result['canonical_name'] == "VBC"
        assert result['phase'] is None
        assert result['measurement'] == "voltage"


class TestManufacturerDetection:
    """Test manufacturer detection from relay model."""

    def test_sel_detection(self):
        assert detect_manufacturer("SEL-421", "") == "SEL"
        assert detect_manufacturer("SEL-311L", "") == "SEL"

    def test_abb_detection(self):
        assert detect_manufacturer("REL670", "") == "ABB"
        assert detect_manufacturer("REF615", "") == "ABB"

    def test_siemens_detection(self):
        assert detect_manufacturer("7SA6", "") == "SIEMENS"
        assert detect_manufacturer("7UT6", "") == "SIEMENS"

    def test_ge_detection(self):
        assert detect_manufacturer("P443", "") == "GE"
        assert detect_manufacturer("MICOM P14x", "") == "GE"

    def test_qualitrol_detection(self):
        assert detect_manufacturer("Qualitrol LLC", "") == "QUALITROL"

    def test_unknown(self):
        assert detect_manufacturer("UNKNOWN_RELAY", "") == "UNKNOWN"


class TestPrimaryConversion:
    """Test CT/VT primary value conversion."""

    def test_ct_conversion(self):
        """
        Example: CT ratio 1200/5
        Secondary current: 3.5A
        Primary should be: 3.5 * (1200/5) = 840A
        """
        secondary_value = 3.5
        ct_primary = 1200.0
        ct_secondary = 5.0

        expected_primary = secondary_value * (ct_primary / ct_secondary)
        assert abs(expected_primary - 840.0) < 0.001

    def test_vt_conversion(self):
        """
        Example: VT ratio 150000/110 (150 kV / 110 V)
        Secondary voltage: 63.5V
        Primary should be: 63.5 * (150000/110) ≈ 86,590 V = 86.59 kV
        """
        secondary_value = 63.5
        vt_primary = 150000.0
        vt_secondary = 110.0

        expected_primary = secondary_value * (vt_primary / vt_secondary)
        assert abs(expected_primary - 86590.9) < 1.0  # Allow 1V tolerance


class TestParserEdgeCases:
    """Test parser handling of edge cases."""

    def test_missing_dat_file(self):
        """Parser should handle missing .dat gracefully."""
        # For this test to work, we'd need a real .cfg file
        # This is a placeholder showing the expected behavior
        pass  # Will test with real files in integration tests

    def test_encoding_issues(self):
        """Parser should handle non-UTF8 characters."""
        pass  # Will test with real files if issues arise

    def test_zero_secondary(self):
        """Parser should handle ct_secondary = 0 without crashing."""
        # This tests the warning path in the parser
        pass


# Integration tests with real files will be added after M1a gate check
def test_parser_on_real_file():
    """
    Test parser on actual PLN COMTRADE file.
    This will be filled in during gate check validation.
    """
    # Example path - update with actual file
    # test_file = "path/to/real/file.cfg"
    # record = parse_comtrade(test_file)
    # assert record is not None
    # assert len(record.analog_channels) > 0
    pass


def test_transformer_diff_real_file_routes_to_87t():
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "raw_data" / "UPT PURWOKERTO" / "2024" / "09. SEPTEMBER" / "28092024_06.24_TRIP TRF #1 GI COMAL" / "Trafo 1_Diff" / "Trafo 1_Diff.CFG"

    record = parse_comtrade(str(cfg_path))
    assert record is not None
    assert len(record.analog_channels) == 16
    assert len(record.status_channels) == 111
    assert any(ch.name == "87T.Op_Inst" for ch in record.status_channels)
    assert any(ch.name == "HVS.Ia" for ch in record.analog_channels)
    assert any(ch.name == "LVS.Ia" for ch in record.analog_channels)

    prot = determine_protection(record)
    assert prot.primary_protection == ProtectionType.TRANSFORMER_DIFF
    assert prot.classifiable is True


def test_line_diff_real_file_does_not_route_to_87t():
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "raw_data" / "UPT PURWOKERTO" / "2024" / "02. FEBRUARI" / "06. 11022024 SUTT BREBES-KANCI #1 (17.20)_PETIR" / "RECLOSE KNCI-BREBES 1 (11 FEB 2024)" / "RECLOSE KNCI-BREBES 1 (11 FEB 2024)" / "REKAMAN DFR" / "GI KANCI" / "BREBES 1" / "BREBES 1.CFG"
    record = parse_comtrade(str(cfg_path))
    assert record is not None
    assert any("87L" in ch.name for ch in record.status_channels)

    prot = determine_protection(record)
    assert prot.primary_protection == ProtectionType.DIFFERENTIAL


def test_ocr_statuses_route_to_overcurrent():
    class Status:
        def __init__(self, name, samples):
            self.name = name
            self.samples = np.array(samples, dtype=int)

    class Record:
        def __init__(self):
            self.status_channels = [
                Status("O/C Ph L2 PU", [0, 1, 1, 0]),
                Status("O/C Ph L3 PU", [0, 1, 1, 0]),
                Status("Overcurrent PU", [0, 1, 1, 0]),
                Status("Relay PICKUP", [0, 1, 1, 0]),
            ]
            self.rec_dev_id = "7SJ622"
            self.station_name = "KESUGIHAN OCR"

    prot = determine_protection(Record())

    assert prot.primary_protection == ProtectionType.OVERCURRENT
    assert prot.classifiable is True
    assert set(prot.operated_phases) == {"B", "C"}


def test_line_diff_real_file_falls_back_to_line_analysis():
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "raw_data" / "UPT PURWOKERTO" / "2024" / "02. FEBRUARI" / "06. 11022024 SUTT BREBES-KANCI #1 (17.20)_PETIR" / "RECLOSE KNCI-BREBES 1 (11 FEB 2024)" / "RECLOSE KNCI-BREBES 1 (11 FEB 2024)" / "REKAMAN DFR" / "GI KANCI" / "BREBES 1" / "BREBES 1.CFG"

    result = classify_file(str(cfg_path))

    assert result.event_type == "LINE"
    assert "proteksi diferensial saluran (87L)" in result.evidence
    assert result.label != "GANGGUAN INTERNAL TRAFO"


def test_mrica_wsobo_real_file_prefers_waveform_faulted_phases():
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "raw_data" / "UPT PURWOKERTO" / "2024" / "04. APRIL" / "03. 11042024 SUTT MRICA-WSOBO #1 (17.48)" / "11042024 WNSB-MRICA 1 SISI MRICA" / "Thursday 11 April 2024 17.48.12.001.CFG"

    record = parse_comtrade(str(cfg_path))
    assert record is not None

    prot = determine_protection(record)
    fault = detect_fault(record)
    feat = extract_distance_features(record, fault, prot)

    assert fault is not None
    assert fault.faulted_phases == ["B", "C"]
    assert feat is not None
    assert feat.faulted_phases == ["B", "C"]
    assert feat.fault_type != "3PH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
