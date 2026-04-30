"""
Tests for Protection Router and Transformer Channel Mapper
==========================================================
Validates routing decisions (87T vs 87L vs distance vs UNKNOWN) and
vendor-specific channel name mapping for transformer differential relays.

Aligned with:
  core/protection_router.py   (replaces misaligned stages.stage2_context)
  core/transformer_channel_mapper.py
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.protection_router import (
    determine_protection,
    ProtectionType,
    _check_auto_reclose_attempted,
    _check_auto_reclose_successful,
)
from core.transformer_channel_mapper import (
    detect_transformer_relay_family,
    map_transformer_channels,
    TransformerChannelMap,
)


# ---------------------------------------------------------------------------
# Mock COMTRADE objects
# ---------------------------------------------------------------------------

@dataclass
class MockChannel:
    name: str
    samples: list = field(default_factory=list)


@dataclass
class MockStatusChannel(MockChannel):
    pass


@dataclass
class MockAnalogChannel(MockChannel):
    measurement: str = "current"
    canonical_name: str = ""


@dataclass
class MockRecord:
    rec_dev_id: str = ""
    station_name: str = ""
    analog_channels: List[MockAnalogChannel] = field(default_factory=list)
    status_channels: List[MockStatusChannel] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    time: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_status(names: List[str], triggered: bool = False) -> List[MockStatusChannel]:
    """Create status channels; first channel has a rising edge if triggered=True."""
    import numpy as np
    channels = []
    for i, name in enumerate(names):
        # Simple mock: ch 0 has a 0→1 transition if triggered, rest are flat 0
        if triggered and i == 0:
            s = np.array([0, 0, 1, 1, 1, 0])
        else:
            s = np.zeros(6, dtype=int)
        channels.append(MockStatusChannel(name=name, samples=s))
    return channels


# ---------------------------------------------------------------------------
# Protection router — 87T routing
# ---------------------------------------------------------------------------

class TestTransformerDiffRouting:

    def test_87t_status_channel_routes_transformer(self):
        """'87T TRIP' status channel must route to TRANSFORMER_DIFF."""
        record = MockRecord(
            rec_dev_id="RET670",
            status_channels=_make_status(["87T TRIP", "PDIF"], triggered=True),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.TRANSFORMER_DIFF

    def test_tr_diff_status_routes_transformer(self):
        """'TR DIFF OPERATE' must also route to TRANSFORMER_DIFF."""
        record = MockRecord(
            rec_dev_id="7UT613",
            status_channels=_make_status(["TR DIFF OPERATE"], triggered=True),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.TRANSFORMER_DIFF

    def test_known_transformer_relay_generic_diff_trip_routes_transformer(self):
        record = MockRecord(
            rec_dev_id="7UT612",
            status_channels=_make_status(["Diff> TRIP"], triggered=True),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.TRANSFORMER_DIFF

    def test_transformer_diff_is_classifiable(self):
        """87T events must be flagged classifiable=True for Phase 2 routing."""
        record = MockRecord(
            status_channels=_make_status(["87T ACT TRIP"], triggered=True),
        )
        prot = determine_protection(record)
        if prot.primary_protection == ProtectionType.TRANSFORMER_DIFF:
            assert prot.classifiable is True

    def test_87t_detected_before_87l(self):
        """When both 87T and 87L channels are present, 87T must win."""
        record = MockRecord(
            status_channels=_make_status(["87T TRIP", "87L TRIP"], triggered=True),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.TRANSFORMER_DIFF


# ---------------------------------------------------------------------------
# Protection router — 87L / distance / unknown
# ---------------------------------------------------------------------------

class TestProtectionRouterGeneral:

    def test_zone1_trip_routes_distance(self):
        """Distance zone1 trip must route to DISTANCE."""
        record = MockRecord(
            rec_dev_id="SEL-421",
            status_channels=_make_status(["ZONE 1 TRIP", "TRIP"], triggered=True),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.DISTANCE

    def test_no_status_channels_unknown(self):
        """No recognisable status channels → UNKNOWN protection."""
        record = MockRecord(
            rec_dev_id="",
            status_channels=[],
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.UNKNOWN

    def test_87l_trip_routes_line_differential(self):
        """'87L TRIP' must route to LINE_DIFF (not TRANSFORMER_DIFF)."""
        record = MockRecord(
            status_channels=_make_status(["87L TRIP"], triggered=True),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.DIFFERENTIAL

    def test_ln1_87l_idiff_operate_does_not_route_transformer(self):
        """Line differential IDIFF signals must stay on the 87L path."""
        record = MockRecord(
            rec_dev_id="7SA522",
            status_channels=_make_status(
                ["Ln1:87L:I-DIFF:Operate:general"],
                triggered=True,
            ),
        )
        prot = determine_protection(record)
        assert prot.primary_protection == ProtectionType.DIFFERENTIAL

    def test_pcs900_successful_reclose_bit_is_recognized(self):
        import numpy as np

        status_dict = {
            "CB1.79.Inprog": np.array([0, 1, 1, 1, 0, 0]),
            "CB1.79.Succ_Rcls": np.array([0, 0, 0, 1, 1, 0]),
            "CB1.79.Fail_Rcls": np.array([0, 0, 0, 0, 0, 0]),
        }
        status_names = list(status_dict.keys())
        assert _check_auto_reclose_attempted(status_names, status_dict) is True
        assert _check_auto_reclose_successful(status_names, status_dict) is True


# ---------------------------------------------------------------------------
# Transformer relay family detection
# ---------------------------------------------------------------------------

class TestRelayFamilyDetection:

    def test_ret670_detected_as_abb(self):
        """RET670 rec_dev_id → ABB family."""
        family = detect_transformer_relay_family("RET670", "GI UNGARAN")
        assert family == "ABB"

    def test_ret615_detected_as_abb(self):
        family = detect_transformer_relay_family("RET615", "")
        assert family == "ABB"

    def test_7ut85_detected_as_siemens(self):
        family = detect_transformer_relay_family("7UT85", "")
        assert family.upper() == "SIEMENS"

    def test_sel387_detected_as_sel(self):
        family = detect_transformer_relay_family("SEL-387", "")
        assert family == "SEL"

    def test_ge_t60_detected_as_ge(self):
        family = detect_transformer_relay_family("T60", "")
        assert family == "GE"

    def test_generic_transformer_relay_detected_as_nr(self):
        family = detect_transformer_relay_family("TRANSFORMER_RELAY", "NR")
        assert family == "NR"

    def test_unknown_relay_returns_generic(self):
        family = detect_transformer_relay_family("UNKNOWN_XYZ", "")
        assert family in {"GENERIC", "UNKNOWN", None, ""}


# ---------------------------------------------------------------------------
# Transformer channel mapping
# ---------------------------------------------------------------------------

class TestTransformerChannelMapping:

    def _make_abb_record(self) -> MockRecord:
        """Minimal ABB RET670 record with standard channel names."""
        analog = [
            MockAnalogChannel(name="IW1A", measurement="current"),
            MockAnalogChannel(name="IW1B", measurement="current"),
            MockAnalogChannel(name="IW1C", measurement="current"),
            MockAnalogChannel(name="IW2A", measurement="current"),
            MockAnalogChannel(name="IW2B", measurement="current"),
            MockAnalogChannel(name="IW2C", measurement="current"),
            MockAnalogChannel(name="IDIFA", measurement="current"),
            MockAnalogChannel(name="IDIFB", measurement="current"),
            MockAnalogChannel(name="IDIFC", measurement="current"),
            MockAnalogChannel(name="IRSTA", measurement="current"),
            MockAnalogChannel(name="IRSTB", measurement="current"),
            MockAnalogChannel(name="IRSTC", measurement="current"),
        ]
        return MockRecord(rec_dev_id="RET670", analog_channels=analog)

    def test_abb_hv_channels_mapped(self):
        """ABB IW1A/B/C must map to HV phase currents."""
        record = self._make_abb_record()
        ch_map = map_transformer_channels(record)
        assert ch_map.has_hv_currents, "ABB IW1x channels should be detected as HV currents"

    def test_abb_lv_channels_mapped(self):
        """ABB IW2A/B/C must map to LV phase currents."""
        record = self._make_abb_record()
        ch_map = map_transformer_channels(record)
        assert ch_map.has_lv_currents, "ABB IW2x channels should be detected as LV currents"

    def test_abb_differential_channels_mapped(self):
        """ABB IDIFA/B/C must map to differential current channels."""
        record = self._make_abb_record()
        ch_map = map_transformer_channels(record)
        assert ch_map.has_differential, "ABB IDIFx channels should be detected as differential"

    def test_abb_restraint_channels_mapped(self):
        """ABB IRSTA/B/C must map to restraint current channels."""
        record = self._make_abb_record()
        ch_map = map_transformer_channels(record)
        assert ch_map.has_restraint, "ABB IRSTx channels should be detected as restraint"

    def test_empty_record_returns_empty_map(self):
        """A record with no analog channels must return a map with all flags False."""
        record = MockRecord(rec_dev_id="UNKNOWN")
        ch_map = map_transformer_channels(record)
        assert isinstance(ch_map, TransformerChannelMap)
        assert not ch_map.has_differential
        assert not ch_map.has_restraint
        assert not ch_map.has_hv_currents
        assert not ch_map.has_lv_currents

    def test_siemens_idiff_channels_mapped(self):
        """Siemens 7UT IDIFF_A / IDIFF_B / IDIFF_C → differential."""
        analog = [
            MockAnalogChannel(name="IDIFF_A", measurement="current"),
            MockAnalogChannel(name="IDIFF_B", measurement="current"),
            MockAnalogChannel(name="IDIFF_C", measurement="current"),
        ]
        record = MockRecord(rec_dev_id="7UT613", analog_channels=analog)
        ch_map = map_transformer_channels(record)
        assert ch_map.has_differential

    def test_siemens_side_suffix_currents_mapped(self):
        analog = [
            MockAnalogChannel(name="iL1-S1", measurement="current"),
            MockAnalogChannel(name="iL2-S1", measurement="current"),
            MockAnalogChannel(name="iL3-S1", measurement="current"),
            MockAnalogChannel(name="iL1-S2", measurement="current"),
            MockAnalogChannel(name="iL2-S2", measurement="current"),
            MockAnalogChannel(name="iL3-S2", measurement="current"),
            MockAnalogChannel(name="3i0-S1", measurement="current"),
            MockAnalogChannel(name="3i0-S2", measurement="current"),
        ]
        record = MockRecord(rec_dev_id="7UT612", analog_channels=analog)
        ch_map = map_transformer_channels(record)
        assert ch_map.has_hv_currents
        assert ch_map.has_lv_currents

    def test_sel_idiff_channels_mapped(self):
        """SEL-387 IAW1/IAW2 channels (winding currents) → HV/LV."""
        analog = [
            MockAnalogChannel(name="IAW1", measurement="current"),
            MockAnalogChannel(name="IBW1", measurement="current"),
            MockAnalogChannel(name="ICW1", measurement="current"),
            MockAnalogChannel(name="IAW2", measurement="current"),
            MockAnalogChannel(name="IBW2", measurement="current"),
            MockAnalogChannel(name="ICW2", measurement="current"),
        ]
        record = MockRecord(rec_dev_id="SEL-387", analog_channels=analog)
        ch_map = map_transformer_channels(record)
        assert ch_map.has_hv_currents
        assert ch_map.has_lv_currents
    
    def test_generic_transformer_relay_87t_channels_mapped(self):
        analog = [
            MockAnalogChannel(name="HVS.Ia", measurement="current"),
            MockAnalogChannel(name="HVS.Ib", measurement="current"),
            MockAnalogChannel(name="HVS.Ic", measurement="current"),
            MockAnalogChannel(name="LVS.Ia", measurement="current"),
            MockAnalogChannel(name="LVS.Ib", measurement="current"),
            MockAnalogChannel(name="LVS.Ic", measurement="current"),
            MockAnalogChannel(name="87T.ida", measurement="current"),
            MockAnalogChannel(name="87T.idb", measurement="current"),
            MockAnalogChannel(name="87T.idc", measurement="current"),
        ]
        record = MockRecord(rec_dev_id="TRANSFORMER_RELAY", station_name="NR", analog_channels=analog)
        ch_map = map_transformer_channels(record)
        assert ch_map.relay_family == "NR"
        assert ch_map.has_hv_currents
        assert ch_map.has_lv_currents
        assert ch_map.has_differential
        assert ch_map.i_diff_a == "87T.ida"
        assert ch_map.i_diff_b == "87T.idb"
        assert ch_map.i_diff_c == "87T.idc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
