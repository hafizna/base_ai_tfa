"""
Transformer Channel Mapper
===========================
Identifies transformer-protection-specific channels from COMTRADE records:
  - HV-side currents (Winding 1)
  - LV-side currents (Winding 2)
  - Tertiary-side currents (Winding 3, if present)
  - Differential currents (computed or measured by relay)
  - Restraint (stabilising) currents

Supports relay families commonly found on PLN 150 kV / 70 kV transformers:
  - ABB RET615 / RED615 / RET670
  - Siemens 7UT612 / 7UT613 / 7UT85 / 7UT86 / 7UT87
  - GE/Multilin T60 / Reason RPV311
  - SEL-387 / SEL-387A / SEL-387E
  - Micom P640 / P643 (Schneider)
  - NR PCS-985T (NARI)

With NO data available, these mappings are derived from:
  - IEC 61850 logical node names (PDIF, TCTR)
  - Relay commissioning manuals (ABB Doc 1MRK505182, Siemens 7UT8 manual)
  - PLN SPLN D3.012-1:2012 (transformer protection standard)
  - IEEE C37.111-2013 COMTRADE naming conventions
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerChannelMap:
    """Identified transformer channels from a ComtradeRecord."""

    # Winding 1 (HV side) phase currents — canonical names after mapping
    i_hv_a: Optional[str] = None   # channel name in record
    i_hv_b: Optional[str] = None
    i_hv_c: Optional[str] = None

    # Winding 2 (LV side) phase currents
    i_lv_a: Optional[str] = None
    i_lv_b: Optional[str] = None
    i_lv_c: Optional[str] = None

    # Winding 3 (tertiary, optional)
    i_tv_a: Optional[str] = None
    i_tv_b: Optional[str] = None
    i_tv_c: Optional[str] = None

    # Differential currents (per-phase, if relay outputs them)
    i_diff_a: Optional[str] = None
    i_diff_b: Optional[str] = None
    i_diff_c: Optional[str] = None

    # Restraint / stabilising currents (per-phase)
    i_rstr_a: Optional[str] = None
    i_rstr_b: Optional[str] = None
    i_rstr_c: Optional[str] = None

    # Neutral / zero-sequence (if present)
    i_n_hv: Optional[str] = None   # HV neutral CT
    i_n_lv: Optional[str] = None   # LV neutral CT

    # Voltage channels (for overexcitation V/Hz detection)
    v_hv_a: Optional[str] = None
    v_hv_b: Optional[str] = None
    v_hv_c: Optional[str] = None

    # Detected relay family
    relay_family: str = "UNKNOWN"

    # Coverage flags
    has_differential: bool = False   # diff channels available
    has_restraint:    bool = False   # restraint channels available
    has_hv_currents:  bool = False
    has_lv_currents:  bool = False

    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Relay-family pattern tables
# Each entry: (regex_pattern, canonical_role)
# Roles: IW1A IW1B IW1C  IW2A IW2B IW2C  IW3A IW3B IW3C
#        IDIFA IDIFB IDIFC  IRSTA IRSTB IRSTC
#        INW1 INW2  VW1A VW1B VW1C
# ─────────────────────────────────────────────────────────────────────────────

# --- ABB RET615 / RET670 ----------------------------------------------------
# IEC 61850 style: W1CTRSA (Winding1 CT R-phase), DIFA (diff A), RSTRA (restraint A)
_ABB_PATTERNS: List[tuple] = [
    # Winding 1 (HV)
    (r'\bW1CTRSA?\b|\bIAW1\b|\bIA_W1\b|\bI_W1_A\b|\bIHVA\b|\bIHV_A\b|\bIW1A\b',  'IW1A'),
    (r'\bW1CTRSB?\b|\bIBW1\b|\bIB_W1\b|\bI_W1_B\b|\bIHVB\b|\bIHV_B\b|\bIW1B\b',  'IW1B'),
    (r'\bW1CTRSC?\b|\bICW1\b|\bIC_W1\b|\bI_W1_C\b|\bIHVC\b|\bIHV_C\b|\bIW1C\b',  'IW1C'),
    # Winding 2 (LV)
    (r'\bW2CTRSA?\b|\bIAW2\b|\bIA_W2\b|\bI_W2_A\b|\bILVA\b|\bILV_A\b|\bIW2A\b',  'IW2A'),
    (r'\bW2CTRSB?\b|\bIBW2\b|\bIB_W2\b|\bI_W2_B\b|\bILVB\b|\bILV_B\b|\bIW2B\b',  'IW2B'),
    (r'\bW2CTRSC?\b|\bICW2\b|\bIC_W2\b|\bI_W2_C\b|\bILVC\b|\bILV_C\b|\bIW2C\b',  'IW2C'),
    # Differential (ABB relay computes and records per IEC 61850 PDIF)
    (r'\bDIFA\b|\bIDIFF_A\b|\bIDIF_A\b|\bIDIFF\.A\b|\bPDIF.*A\b',                 'IDIFA'),
    (r'\bDIFB\b|\bIDIFF_B\b|\bIDIF_B\b|\bIDIFF\.B\b|\bPDIF.*B\b',                 'IDIFB'),
    (r'\bDIFC\b|\bIDIFF_C\b|\bIDIF_C\b|\bIDIFF\.C\b|\bPDIF.*C\b',                 'IDIFC'),
    # Restraint
    (r'\bRSTRA\b|\bIRSTA\b|\bIRSTR_A\b|\bIRST_A\b|\bIDIFF.*RST.*A\b',             'IRSTA'),
    (r'\bRSTRB\b|\bIRSTB\b|\bIRSTR_B\b|\bIRST_B\b|\bIDIFF.*RST.*B\b',             'IRSTB'),
    (r'\bRSTRC\b|\bIRSTC\b|\bIRSTR_C\b|\bIRST_C\b|\bIDIFF.*RST.*C\b',             'IRSTC'),
    # Neutral
    (r'\bINW1\b|\bIN_W1\b|\bINHV\b|\bIN_HV\b',                                    'INW1'),
    (r'\bINW2\b|\bIN_W2\b|\bINLV\b|\bIN_LV\b',                                    'INW2'),
]

# --- Siemens 7UT612/613/85/86/87 -------------------------------------------
# Siemens uses ILA1/ILA2 for line current winding 1/2, IDIFF for differential
_SIEMENS_PATTERNS: List[tuple] = [
    # Winding 1
    (r'\bILA1\b|\bIL_A1\b|\bI_A_W1\b|\bIHV_A\b|\bIA:W1\b|\bIW1:A\b',            'IW1A'),
    (r'\bILB1\b|\bIL_B1\b|\bI_B_W1\b|\bIHV_B\b|\bIB:W1\b|\bIW1:B\b',            'IW1B'),
    (r'\bILC1\b|\bIL_C1\b|\bI_C_W1\b|\bIHV_C\b|\bIC:W1\b|\bIW1:C\b',            'IW1C'),
    (r'\bIL1[-_\s]?S1\b|\bIL1[-_\s]?W1\b|\bIL1[-_\s]?SIDE1\b|\bIL1[-_\s]?SEC1\b', 'IW1A'),
    (r'\bIL2[-_\s]?S1\b|\bIL2[-_\s]?W1\b|\bIL2[-_\s]?SIDE1\b|\bIL2[-_\s]?SEC1\b', 'IW1B'),
    (r'\bIL3[-_\s]?S1\b|\bIL3[-_\s]?W1\b|\bIL3[-_\s]?SIDE1\b|\bIL3[-_\s]?SEC1\b', 'IW1C'),
    # Winding 2
    (r'\bILA2\b|\bIL_A2\b|\bI_A_W2\b|\bILV_A\b|\bIA:W2\b|\bIW2:A\b',            'IW2A'),
    (r'\bILB2\b|\bIL_B2\b|\bI_B_W2\b|\bILV_B\b|\bIB:W2\b|\bIW2:B\b',            'IW2B'),
    (r'\bILC2\b|\bIL_C2\b|\bI_C_W2\b|\bILV_C\b|\bIC:W2\b|\bIW2:C\b',            'IW2C'),
    (r'\bIL1[-_\s]?S2\b|\bIL1[-_\s]?W2\b|\bIL1[-_\s]?SIDE2\b|\bIL1[-_\s]?SEC2\b', 'IW2A'),
    (r'\bIL2[-_\s]?S2\b|\bIL2[-_\s]?W2\b|\bIL2[-_\s]?SIDE2\b|\bIL2[-_\s]?SEC2\b', 'IW2B'),
    (r'\bIL3[-_\s]?S2\b|\bIL3[-_\s]?W2\b|\bIL3[-_\s]?SIDE2\b|\bIL3[-_\s]?SEC2\b', 'IW2C'),
    # Differential (Siemens uses 'IDIFF' + phase suffix)
    (r'\bIDIFF\s*A\b|\bIDIFF_A\b|\bI_DIFF_A\b|\bDIFF.*:.*A\b',                   'IDIFA'),
    (r'\bIDIFF\s*B\b|\bIDIFF_B\b|\bI_DIFF_B\b|\bDIFF.*:.*B\b',                   'IDIFB'),
    (r'\bIDIFF\s*C\b|\bIDIFF_C\b|\bI_DIFF_C\b|\bDIFF.*:.*C\b',                   'IDIFC'),
    # Restraint (Siemens: IREST or IRESTR)
    (r'\bIREST\s*A\b|\bIREST_A\b|\bIRESTR_A\b|\bREST.*A\b',                      'IRSTA'),
    (r'\bIREST\s*B\b|\bIREST_B\b|\bIRESTR_B\b|\bREST.*B\b',                      'IRSTB'),
    (r'\bIREST\s*C\b|\bIREST_C\b|\bIRESTR_C\b|\bREST.*C\b',                      'IRSTC'),
    # Siemens 7UT neutral
    (r'\bIN1\b|\bIE1\b|\bIG1\b',                                                   'INW1'),
    (r'\bIN2\b|\bIE2\b|\bIG2\b',                                                   'INW2'),
    (r'\b3I0[-_\s]?S1\b|\bIN[-_\s]?S1\b|\bIE[-_\s]?S1\b',                         'INW1'),
    (r'\b3I0[-_\s]?S2\b|\bIN[-_\s]?S2\b|\bIE[-_\s]?S2\b',                         'INW2'),
]

# --- SEL-387 / SEL-387A / SEL-387E -----------------------------------------
# SEL uses IW1A…IW2C for winding currents (Winding1, Winding2)
_SEL_PATTERNS: List[tuple] = [
    (r'\bIW1A\b|\bI_W1_A\b|\bIAW1\b',    'IW1A'),
    (r'\bIW1B\b|\bI_W1_B\b|\bIBW1\b',    'IW1B'),
    (r'\bIW1C\b|\bI_W1_C\b|\bICW1\b',    'IW1C'),
    (r'\bIW2A\b|\bI_W2_A\b|\bIAW2\b',    'IW2A'),
    (r'\bIW2B\b|\bI_W2_B\b|\bIBW2\b',    'IW2B'),
    (r'\bIW2C\b|\bI_W2_C\b|\bICW2\b',    'IW2C'),
    (r'\bIW3A\b|\bI_W3_A\b',    'IW3A'),
    (r'\bIW3B\b|\bI_W3_B\b',    'IW3B'),
    (r'\bIW3C\b|\bI_W3_C\b',    'IW3C'),
    # SEL does not typically record differential as analog—computed only
    (r'\bIA87\b|\bIDIF_A\b',     'IDIFA'),
    (r'\bIB87\b|\bIDIF_B\b',     'IDIFB'),
    (r'\bIC87\b|\bIDIF_C\b',     'IDIFC'),
    (r'\bIR87A\b|\bIRST_A\b',    'IRSTA'),
    (r'\bIR87B\b|\bIRST_B\b',    'IRSTB'),
    (r'\bIR87C\b|\bIRST_C\b',    'IRSTC'),
]

# --- GE T60 / Reason RPV311 -------------------------------------------------
_GE_PATTERNS: List[tuple] = [
    (r'\bIA_W1\b|\bIA1\b|\bIA_HV\b',   'IW1A'),
    (r'\bIB_W1\b|\bIB1\b|\bIB_HV\b',   'IW1B'),
    (r'\bIC_W1\b|\bIC1\b|\bIC_HV\b',   'IW1C'),
    (r'\bIA_W2\b|\bIA2\b|\bIA_LV\b',   'IW2A'),
    (r'\bIB_W2\b|\bIB2\b|\bIB_LV\b',   'IW2B'),
    (r'\bIC_W2\b|\bIC2\b|\bIC_LV\b',   'IW2C'),
    (r'\bDIFF-IA\b|\bIDIFF_A\b',        'IDIFA'),
    (r'\bDIFF-IB\b|\bIDIFF_B\b',        'IDIFB'),
    (r'\bDIFF-IC\b|\bIDIFF_C\b',        'IDIFC'),
    (r'\bRST-IA\b|\bIRST_A\b',          'IRSTA'),
    (r'\bRST-IB\b|\bIRST_B\b',          'IRSTB'),
    (r'\bRST-IC\b|\bIRST_C\b',          'IRSTC'),
]

# --- Micom P640 / P643 (Schneider / Alstom) --------------------------------
_MICOM_PATTERNS: List[tuple] = [
    (r'\bIA_HV\b|\bIR_HV\b|\bI1_A\b',   'IW1A'),
    (r'\bIB_HV\b|\bIS_HV\b|\bI1_B\b',   'IW1B'),
    (r'\bIC_HV\b|\bIT_HV\b|\bI1_C\b',   'IW1C'),
    (r'\bIA_LV\b|\bIR_LV\b|\bI2_A\b',   'IW2A'),
    (r'\bIB_LV\b|\bIS_LV\b|\bI2_B\b',   'IW2B'),
    (r'\bIC_LV\b|\bIT_LV\b|\bI2_C\b',   'IW2C'),
    (r'\bIDIF_A\b|\bIDIFF_A\b',           'IDIFA'),
    (r'\bIDIF_B\b|\bIDIFF_B\b',           'IDIFB'),
    (r'\bIDIF_C\b|\bIDIFF_C\b',           'IDIFC'),
    (r'\bIRST_A\b|\bIBIAS_A\b',           'IRSTA'),
    (r'\bIRST_B\b|\bIBIAS_B\b',           'IRSTB'),
    (r'\bIRST_C\b|\bIBIAS_C\b',           'IRSTC'),
]

# --- NR PCS-985T (NARI, used in some PLN substations) ----------------------
_NR_PATTERNS: List[tuple] = [
    (r'\bIA_HV\b|\bIAHV\b|\bI_A_HV\b',   'IW1A'),
    (r'\bIB_HV\b|\bIBHV\b|\bI_B_HV\b',   'IW1B'),
    (r'\bIC_HV\b|\bICHV\b|\bI_C_HV\b',   'IW1C'),
    (r'\bIA_LV\b|\bIALV\b|\bI_A_LV\b',   'IW2A'),
    (r'\bIB_LV\b|\bIBLV\b|\bI_B_LV\b',   'IW2B'),
    (r'\bIC_LV\b|\bICLV\b|\bI_C_LV\b',   'IW2C'),
    (r'\bIDIF_A\b|\bIDIFF_A\b',            'IDIFA'),
    (r'\bIDIF_B\b|\bIDIFF_B\b',            'IDIFB'),
    (r'\bIDIF_C\b|\bIDIFF_C\b',            'IDIFC'),
    (r'\bIRST_A\b|\bIBRK_A\b',             'IRSTA'),
    (r'\bIRST_B\b|\bIBRK_B\b',             'IRSTB'),
    (r'\bIRST_C\b|\bIBRK_C\b',             'IRSTC'),
]

# Generic fallback — works across all vendors when specific ones miss
_GENERIC_PATTERNS: List[tuple] = [
    # Winding 1 by any HV or W1 marker
    (r'(?:W1|HV|WIND1|WINDING1).*[_\s\-]?[AR]$',   'IW1A'),
    (r'(?:W1|HV|WIND1|WINDING1).*[_\s\-]?[BS]$',   'IW1B'),
    (r'(?:W1|HV|WIND1|WINDING1).*[_\s\-]?[CT]$',   'IW1C'),
    (r'^[AR][_\s\-]?(?:W1|HV|WIND1)',               'IW1A'),
    (r'^[BS][_\s\-]?(?:W1|HV|WIND1)',               'IW1B'),
    (r'^[CT][_\s\-]?(?:W1|HV|WIND1)',               'IW1C'),
    # Winding 2 by any LV or W2 marker
    (r'(?:W2|LV|WIND2|WINDING2).*[_\s\-]?[AR]$',   'IW2A'),
    (r'(?:W2|LV|WIND2|WINDING2).*[_\s\-]?[BS]$',   'IW2B'),
    (r'(?:W2|LV|WIND2|WINDING2).*[_\s\-]?[CT]$',   'IW2C'),
    (r'^[AR][_\s\-]?(?:W2|LV|WIND2)',               'IW2A'),
    (r'^[BS][_\s\-]?(?:W2|LV|WIND2)',               'IW2B'),
    (r'^[CT][_\s\-]?(?:W2|LV|WIND2)',               'IW2C'),
    # Transformer differential outputs commonly labeled as 87T.ida/b/c.
    (r'\b87T[._\s\-]*(?:IDA|IA|A)\b',               'IDIFA'),
    (r'\b87T[._\s\-]*(?:IDB|IB|B)\b',               'IDIFB'),
    (r'\b87T[._\s\-]*(?:IDC|IC|C)\b',               'IDIFC'),
    # Differential by keyword
    (r'(?:DIFF|DIF)[_\s\-]?[AR]$',                  'IDIFA'),
    (r'(?:DIFF|DIF)[_\s\-]?[BS]$',                  'IDIFB'),
    (r'(?:DIFF|DIF)[_\s\-]?[CT]$',                  'IDIFC'),
    # Restraint / bias / stabilise
    (r'(?:RSTR|REST|BIAS|STAB)[_\s\-]?[AR]$',       'IRSTA'),
    (r'(?:RSTR|REST|BIAS|STAB)[_\s\-]?[BS]$',       'IRSTB'),
    (r'(?:RSTR|REST|BIAS|STAB)[_\s\-]?[CT]$',       'IRSTC'),
]

# Map relay family name → pattern list
_RELAY_PATTERN_MAP: Dict[str, List[tuple]] = {
    'ABB':     _ABB_PATTERNS,
    'SIEMENS': _SIEMENS_PATTERNS,
    'SEL':     _SEL_PATTERNS,
    'GE':      _GE_PATTERNS,
    'MICOM':   _MICOM_PATTERNS,
    'NR':      _NR_PATTERNS,
}

# Role → channel-map attribute
_ROLE_ATTR: Dict[str, str] = {
    'IW1A': 'i_hv_a',  'IW1B': 'i_hv_b',  'IW1C': 'i_hv_c',
    'IW2A': 'i_lv_a',  'IW2B': 'i_lv_b',  'IW2C': 'i_lv_c',
    'IW3A': 'i_tv_a',  'IW3B': 'i_tv_b',  'IW3C': 'i_tv_c',
    'IDIFA': 'i_diff_a', 'IDIFB': 'i_diff_b', 'IDIFC': 'i_diff_c',
    'IRSTA': 'i_rstr_a', 'IRSTB': 'i_rstr_b', 'IRSTC': 'i_rstr_c',
    'INW1': 'i_n_hv',  'INW2': 'i_n_lv',
}


# ─────────────────────────────────────────────────────────────────────────────
# Relay family detection
# ─────────────────────────────────────────────────────────────────────────────

# Keywords to detect relay family from rec_dev_id / station_name
_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    'ABB':     ['RET615', 'RET670', 'RED615', 'RED670', 'REF615', 'REX',
                'ABB', 'BUSBAR', 'PDIF'],
    'SIEMENS': ['7UT', 'SIPROTEC', 'BM', 'SIEMENS', 'DIGSI'],
    'SEL':     ['SEL-387', 'SEL387', 'SEL 387', 'SEL-487', 'SEL487'],
    'GE':      ['T60', 'GE ', 'GE-', 'MULTILIN', 'RPV', 'GE GRID',
                'D60', 'T35'],
    'MICOM':   ['P640', 'P643', 'P64', 'MICOM', 'ALSTOM', 'SCHNEIDER'],
    'NR':      ['PCS-985', 'PCS985', 'PCS 985', 'NARI', 'NR ELECTRIC', 'TRANSFORMER_RELAY'],
}


def detect_transformer_relay_family(rec_dev_id: str, station_name: str = "") -> str:
    """
    Detect transformer relay family from device ID or station name.

    Returns one of: ABB, SIEMENS, SEL, GE, MICOM, NR, UNKNOWN
    """
    text = f"{rec_dev_id} {station_name}".upper()
    for family, keywords in _FAMILY_KEYWORDS.items():
        if any(kw.upper() in text for kw in keywords):
            return family
    return 'UNKNOWN'


def _is_transformer_current_channel(ch) -> bool:
    """
    Treat standard phase currents as current channels, plus transformer-specific
    differential / restraint channels that some relays record with non-A units.
    """
    if getattr(ch, 'measurement', '') == 'current':
        return True

    raw = f"{getattr(ch, 'name', '')} {getattr(ch, 'canonical_name', '')}".upper()
    if re.search(r'\b87T[._\s\-]*(?:IDA|IDB|IDC|IA|IB|IC)\b', raw):
        return True
    if re.search(r'\b(?:IDIFF|IDIF|DIFF|RSTR|REST|BIAS|STAB|64REF)\b', raw):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main mapping function
# ─────────────────────────────────────────────────────────────────────────────

def map_transformer_channels(record) -> TransformerChannelMap:
    """
    Identify transformer protection channels in a ComtradeRecord.

    Tries:
      1. Relay-family-specific patterns (precise)
      2. Generic cross-vendor patterns (fallback)

    Args:
        record: ComtradeRecord (from comtrade_parser.py)

    Returns:
        TransformerChannelMap with populated channel names.
        Warnings list contains any ambiguity or missing-channel notes.
    """
    ch_map = TransformerChannelMap()
    ch_map.relay_family = detect_transformer_relay_family(
        getattr(record, 'rec_dev_id', ''),
        getattr(record, 'station_name', ''),
    )

    # Gather current channel names (exclude voltage for role matching)
    current_channels = [
        ch for ch in getattr(record, 'analog_channels', [])
        if _is_transformer_current_channel(ch)
    ]
    voltage_channels = [
        ch for ch in getattr(record, 'analog_channels', [])
        if getattr(ch, 'measurement', '') == 'voltage'
    ]

    all_current_names = [ch.name for ch in current_channels]

    # Select pattern list: family-specific first, then generic
    patterns = _RELAY_PATTERN_MAP.get(ch_map.relay_family, []) + _GENERIC_PATTERNS

    assigned: Dict[str, str] = {}   # role → channel_name

    for ch_name in all_current_names:
        ch_upper = ch_name.upper().strip()
        for pattern, role in patterns:
            if role in assigned:
                continue  # already assigned
            try:
                if re.search(pattern, ch_upper):
                    assigned[role] = ch_name
                    logger.debug(f"Transformer channel: '{ch_name}' → {role} (pattern: {pattern})")
                    break
            except re.error:
                continue

    # Populate dataclass
    for role, ch_name in assigned.items():
        attr = _ROLE_ATTR.get(role)
        if attr:
            setattr(ch_map, attr, ch_name)

    # Voltage channels (HV side, for overexcitation detection)
    _map_voltage_channels(ch_map, voltage_channels)

    # Set coverage flags
    ch_map.has_hv_currents  = any([ch_map.i_hv_a, ch_map.i_hv_b, ch_map.i_hv_c])
    ch_map.has_lv_currents  = any([ch_map.i_lv_a, ch_map.i_lv_b, ch_map.i_lv_c])
    ch_map.has_differential = any([ch_map.i_diff_a, ch_map.i_diff_b, ch_map.i_diff_c])
    ch_map.has_restraint    = any([ch_map.i_rstr_a, ch_map.i_rstr_b, ch_map.i_rstr_c])

    # Warn about missing critical channels
    if not ch_map.has_hv_currents:
        ch_map.warnings.append("HV-side (W1) current channels not identified. "
                               "Check channel names or relay family detection.")
    if not ch_map.has_lv_currents:
        ch_map.warnings.append("LV-side (W2) current channels not identified. "
                               "Differential computation will use HV only.")
    if not ch_map.has_differential and not ch_map.has_hv_currents:
        ch_map.warnings.append("No differential or winding current channels found. "
                               "Harmonic analysis will be limited.")

    logger.info(
        f"TransformerChannelMap: family={ch_map.relay_family}, "
        f"HV={ch_map.has_hv_currents}, LV={ch_map.has_lv_currents}, "
        f"DIFF={ch_map.has_differential}, RSTR={ch_map.has_restraint}"
    )
    return ch_map


def _map_voltage_channels(ch_map: TransformerChannelMap, voltage_channels: list) -> None:
    """Map HV-side voltage channels (for V/Hz overexcitation detection)."""
    v_patterns = [
        (r'(?:HV|W1|WIND1).*[_\s\-]?[AR]$|^[AR][_\s\-]?(?:HV|W1)', 'v_hv_a'),
        (r'(?:HV|W1|WIND1).*[_\s\-]?[BS]$|^[BS][_\s\-]?(?:HV|W1)', 'v_hv_b'),
        (r'(?:HV|W1|WIND1).*[_\s\-]?[CT]$|^[CT][_\s\-]?(?:HV|W1)', 'v_hv_c'),
        # Fallback: first 3 voltage channels → HV
        (r'.*', 'v_hv_a'),
        (r'.*', 'v_hv_b'),
        (r'.*', 'v_hv_c'),
    ]
    assigned_v: Dict[str, str] = {}
    for ch in voltage_channels:
        ch_upper = ch.name.upper()
        for pattern, attr in v_patterns:
            if attr in assigned_v:
                continue
            if re.search(pattern, ch_upper):
                assigned_v[attr] = ch.name
                break

    for attr, ch_name in assigned_v.items():
        setattr(ch_map, attr, ch_name)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: extract numpy arrays for a list of channel names
# ─────────────────────────────────────────────────────────────────────────────

def get_channel_samples(record, channel_name: Optional[str]):
    """
    Return the numpy sample array for a named channel in a ComtradeRecord.
    Returns None if channel_name is None or not found.
    """
    if channel_name is None:
        return None
    for ch in getattr(record, 'analog_channels', []):
        if ch.name == channel_name:
            return ch.samples
    return None


def get_mapped_samples(record, ch_map: TransformerChannelMap) -> Dict[str, object]:
    """
    Return a dict of role → numpy_array for all identified channels.
    Missing channels have value None.
    """
    roles = {
        'i_hv_a': ch_map.i_hv_a,  'i_hv_b': ch_map.i_hv_b,  'i_hv_c': ch_map.i_hv_c,
        'i_lv_a': ch_map.i_lv_a,  'i_lv_b': ch_map.i_lv_b,  'i_lv_c': ch_map.i_lv_c,
        'i_tv_a': ch_map.i_tv_a,  'i_tv_b': ch_map.i_tv_b,  'i_tv_c': ch_map.i_tv_c,
        'i_diff_a': ch_map.i_diff_a, 'i_diff_b': ch_map.i_diff_b, 'i_diff_c': ch_map.i_diff_c,
        'i_rstr_a': ch_map.i_rstr_a, 'i_rstr_b': ch_map.i_rstr_b, 'i_rstr_c': ch_map.i_rstr_c,
        'i_n_hv': ch_map.i_n_hv,  'i_n_lv': ch_map.i_n_lv,
        'v_hv_a': ch_map.v_hv_a,  'v_hv_b': ch_map.v_hv_b,  'v_hv_c': ch_map.v_hv_c,
    }
    return {role: get_channel_samples(record, ch_name) for role, ch_name in roles.items()}
