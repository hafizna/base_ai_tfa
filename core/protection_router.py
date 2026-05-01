"""
Protection Type Router
======================
Determines which protection operated (87L, 21, 67N, etc.) by reading status channels.
Routes to appropriate feature extraction: distance-based or differential-based.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


def _normalize_status_name(name: str) -> str:
    """Normalize status channel name for robust matching."""
    if not name:
        return ""
    # Uppercase, replace separators with spaces, collapse whitespace
    s = name.upper()
    # Replace separators (., _, /, -) with spaces.
    s = re.sub(r"[._/\\-]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _name_variants(name: str) -> list[str]:
    """Return matching variants: raw, normalized, and compact (no spaces)."""
    raw = (name or "").upper().strip()
    norm = _normalize_status_name(name)
    compact = norm.replace(" ", "")
    # Preserve order, drop empties
    variants = []
    for v in (raw, norm, compact):
        if v and v not in variants:
            variants.append(v)
    return variants


class ProtectionType(Enum):
    """Protection element types in PLN transmission system."""
    DIFFERENTIAL = "87L"            # Line current differential
    TRANSFORMER_DIFF = "87T"        # Transformer differential (Phase 2)
    DISTANCE = "21"                 # Distance protection (impedance)
    DIRECTIONAL_EF = "67N"          # Directional earth fault
    OVERCURRENT = "50/51"           # Overcurrent
    TRANSFORMER_OVERCURRENT = "51T" # Transformer overcurrent / thermal (Phase 2)
    UNKNOWN = "UNKNOWN"             # Could not determine


class TeleprotectionScheme(Enum):
    """Teleprotection schemes used with distance protection."""
    POTT = "POTT"              # Permissive overreach transfer trip
    PUTT = "PUTT"              # Permissive underreach transfer trip
    DCB = "DCB"                # Directional comparison blocking
    NONE = "NONE"              # No teleprotection (standalone distance or comms failure)
    UNKNOWN = "UNKNOWN"


@dataclass
class ProtectionEvent:
    """Result of protection type detection from status channels."""

    # What operated
    primary_protection: ProtectionType
    operated_zones: List[str]         # ["Z1"], ["Z2"], ["Z1", "Z2"], etc.
    operated_phases: List[str]        # ["A"], ["A", "B"], ["A", "B", "C"]

    # Teleprotection
    teleprotection_scheme: TeleprotectionScheme
    permission_received: bool         # Did remote end send permissive signal?
    comms_failure: bool               # Was communication link broken?

    # Trip details
    trip_type: str                    # "single_pole", "three_pole", "unknown"

    # Auto-reclose
    auto_reclose_attempted: bool
    auto_reclose_successful: Optional[bool]  # None if not attempted

    # Classification routing
    classifiable: bool                # True = distance trip, proceed to classification
    skip_reason: Optional[str]        # Why it's not classifiable (if applicable)

    # Confidence
    confidence: float                 # 0-1, how confident we are in this determination
    warnings: List[str]


def determine_protection(record) -> ProtectionEvent:
    """
    Read status channels to determine which protection operated.

    Args:
        record: ComtradeRecord with parsed status channels

    Returns:
        ProtectionEvent with protection type and routing decision

    Strategy:
    1. Look for 87L operate signals → if found, primary = DIFFERENTIAL
    2. Look for 21/Z1/Z2/Z3/DIST operate signals → if found, primary = DISTANCE
    3. Look for 67N operate signals → if found, primary = DIRECTIONAL_EF
    4. Look for communication failure indicators
    5. Look for teleprotection receive signals (POTT/PUTT)
    6. Look for auto-reclose signals
    7. Look for trip type (single-pole vs three-pole)

    Routing logic:
    - If 87L operated AND 21 did NOT → classifiable = False (differential trip)
    - If 21 operated (with or without 87L) → classifiable = True (distance features available)
    - If 67N operated alone → classifiable = False (directional EF only)
    - If UNKNOWN → classifiable = False (cannot determine protection type)
    """

    # Initialize results
    primary_protection = ProtectionType.UNKNOWN
    operated_zones = []
    operated_phases = []
    teleprotection_scheme = TeleprotectionScheme.UNKNOWN
    permission_received = False
    comms_failure = False
    trip_type = "unknown"
    auto_reclose_attempted = False
    auto_reclose_successful = None
    classifiable = False
    skip_reason = None
    confidence = 0.0
    warnings = []

    # Extract status channel names
    status_names = [ch.name for ch in record.status_channels]
    status_dict = {ch.name: ch.samples for ch in record.status_channels}

    # Search for protection operate signals
    xfmr_diff_operated = _check_transformer_diff_operate(
        status_names,
        status_dict,
        rec_dev_id=getattr(record, 'rec_dev_id', ''),
        station_name=getattr(record, 'station_name', ''),
    )
    diff_operated = _check_differential_operate(status_names, status_dict)
    dist_operated, dist_zones, dist_phases = _check_distance_operate(status_names, status_dict)
    ef_operated = _check_directional_ef_operate(status_names, status_dict)
    oc_operated, oc_phases = _check_overcurrent_operate(status_names, status_dict)

    # Determine primary protection
    # 87T takes priority — if transformer-specific signals found, route to Phase 2 classifier
    if xfmr_diff_operated:
        primary_protection = ProtectionType.TRANSFORMER_DIFF
        confidence = 0.90
        classifiable = True   # Phase 2 transformer classifier
        skip_reason = None

    elif dist_operated:
        primary_protection = ProtectionType.DISTANCE
        operated_zones = dist_zones
        operated_phases = dist_phases
        confidence = 0.9

        # Distance protection → classifiable = True
        classifiable = True
        skip_reason = None

    elif diff_operated:
        primary_protection = ProtectionType.DIFFERENTIAL
        confidence = 0.9

        # 87L line differential — route to waveform-based cause classifier.
        # Zone info is unavailable but current waveform features (di/dt, i0/i1, THD,
        # peak current, inception angle) are still valid discriminators.
        # A separate 87L-specific logic module is not required for cause classification.
        classifiable = True
        skip_reason = None
        warnings.append(
            "Proteksi diferensial saluran (87L) terdeteksi — "
            "gangguan berada di segmen terlindungi. "
            "Klasifikasi penyebab dari gelombang arus; tidak ada data zona impedansi."
        )

    elif ef_operated:
        primary_protection = ProtectionType.DIRECTIONAL_EF
        confidence = 0.8

        # 67N / DEF: sensitive directional earth fault.
        # Fires for high-resistance ground faults (tree branch, kite string, foreign object
        # touching one phase) — cause classification from zero-sequence waveform features
        # is still meaningful.  Zone info absent; i0/i1 ratio is the primary discriminator.
        classifiable = True
        skip_reason = None
        warnings.append(
            "Rele arah gangguan tanah (67N / DEF) terdeteksi — "
            "gangguan hubung tanah resistif tinggi; kemungkinan besar pohon, layang-layang, "
            "atau benda asing yang menyentuh penghantar. "
            "Klasifikasi dari pola gelombang zero-sequence; tidak ada data zona impedansi."
        )

    elif oc_operated:
        primary_protection = ProtectionType.OVERCURRENT
        operated_phases = oc_phases
        confidence = 0.85

        # OCR / 50/51 files still carry valid fault waveforms, but zone info is absent.
        classifiable = True
        skip_reason = None
        warnings.append(
            "Proteksi diidentifikasi sebagai OCR / 50/51 dari status pickup/trip; "
            "analisis penyebab tetap berbasis gelombang karena tidak ada logika zona rele jarak."
        )

    else:
        # Last resort: check for generic trip signals (Siemens group indicator, DFR outputs, etc.)
        # Safe to call here — 87L and 67N have already been excluded above
        grp_operated, grp_phases = _check_generic_trip_fallback(status_names, status_dict)
        if grp_operated:
            primary_protection = ProtectionType.DISTANCE
            operated_phases = grp_phases
            confidence = 0.65
            classifiable = True
            skip_reason = None
            warnings.append(
                "Tipe proteksi diidentifikasi dari sinyal trip generik (tidak ada sinyal zona spesifik) "
                "— kemungkinan rele jarak, konfirmasi dari rekaman osilografi"
            )
            logger.debug(f"Generic trip fallback detected distance: phases={grp_phases}")
        else:
            primary_protection = ProtectionType.UNKNOWN
            confidence = 0.0
            classifiable = False
            skip_reason = "Could not determine protection type from status channels"
            warnings.append("No protection operate signals found in status channels")

    # Detect operated phases (if not already set by distance logic)
    if not operated_phases:
        operated_phases = _detect_operated_phases(status_names, status_dict)

    # Detect teleprotection scheme
    teleprotection_scheme = _detect_teleprotection_scheme(status_names, status_dict)
    permission_received = _check_permission_received(status_names, status_dict)

    # Detect communication failure
    comms_failure = _check_comms_failure(status_names, status_dict)
    if comms_failure:
        warnings.append("Communication link failure detected")

    # Detect trip type
    trip_type = _detect_trip_type(status_names, status_dict)
    # Infer from operated phases if still unknown
    if trip_type == "unknown" and operated_phases:
        if len(operated_phases) == 1:
            trip_type = "single_pole"
        elif len(operated_phases) >= 3:
            trip_type = "three_pole"

    # Detect auto-reclose
    auto_reclose_attempted = _check_auto_reclose_attempted(status_names, status_dict)
    if auto_reclose_attempted:
        auto_reclose_successful = _check_auto_reclose_successful(status_names, status_dict)

    return ProtectionEvent(
        primary_protection=primary_protection,
        operated_zones=operated_zones,
        operated_phases=operated_phases,
        teleprotection_scheme=teleprotection_scheme,
        permission_received=permission_received,
        comms_failure=comms_failure,
        trip_type=trip_type,
        auto_reclose_attempted=auto_reclose_attempted,
        auto_reclose_successful=auto_reclose_successful,
        classifiable=classifiable,
        skip_reason=skip_reason,
        confidence=confidence,
        warnings=warnings
    )


def _check_transformer_diff_operate(
    status_names: List[str],
    status_dict: dict,
    rec_dev_id: str = "",
    station_name: str = "",
) -> bool:
    """
    Check if transformer differential protection (87T) operated.

    Distinguishes 87T (transformer) from 87L (line) by looking for transformer-specific
    channel name patterns from major relay families:
      - ABB RET615/RET670: "87T:Operate", "PDIF:Operate", "TR_DIFF_TRIP"
      - Siemens 7UT:       "87T TRIP", "TRANSFORMER DIFF", "DIFF OPERATE"
      - SEL-387:           "87T", "TR_87T_TRIP", "DIFF OP"
      - GE T60:            "87T OPERATE", "XFMR DIFF", "T60_87T"
      - Micom P640:        "PDIF_TRIP", "87T_OPERATE"
      - NARI PCS-985T:     "87T_ACT", "DIFF_ACT", "TR87T"

    Deliberately conservative — requires explicit transformer keywords
    to avoid false-matching line differential signals.
    """
    line_diff_markers = [
        '87L', 'LN1', 'LN2', 'LN3', 'LINE DIFF', 'LINE DIFFERENTIAL',
        'REMOTE TRIP', 'I DIFF OPERATE',
    ]

    def _looks_like_line_diff(variants: list[str]) -> bool:
        return any(marker in v for v in variants for marker in line_diff_markers)

    # Patterns that are specific to TRANSFORMER differential (not line 87L)
    xfmr_patterns = [
        '87T', 'TR_DIFF', 'XFMR_DIFF', 'TRANSFORMER DIFF',
        'TRDIFF', 'TR87', 'PDIF',          # ABB RET615 IEC 61850 PDIF LN
        'XFMR DIFF', 'XFMR_DIFF',          # GE T60: "XFMR DIFF OPERATE"
        'TRAFO', 'TRANSF',                 # Generic transformer keywords
        '87T_ACT', 'DIFF_ACT',             # NARI PCS-985T
    ]
    # Transformer-only hints that are unlikely for line differential
    xfmr_only_hints = [
        'REF', 'RESTRICTED EARTH',          # REF used for transformer/generator
        'HV', 'LV', 'WINDING', 'TAP',        # Winding / tap indicators
        'BUCHHOLZ', 'OLTC',                 # Transformer-specific signals
    ]
    operate_keywords = ['OPERATE', 'TRIP', 'ACT', 'OP', 'TRIG']
    relay_text = f"{rec_dev_id} {station_name}".upper()
    known_transformer_relay = any(
        key in relay_text
        for key in (
            '7UT', 'RET', 'P64', 'P643', 'SEL-387', 'SEL387',
            'T60', 'PCS-985', 'PCS985', 'MICOM', 'TRAFO',
            'TRANSFORMER', 'XFMR',
        )
    )

    # First pass: explicit 87T / transformer-labeled differential operate
    for name in status_names:
        variants = _name_variants(name)
        if _looks_like_line_diff(variants):
            continue
        has_xfmr = any(p in v for v in variants for p in xfmr_patterns)
        has_operate = any(k in v for v in variants for k in operate_keywords)

        if has_xfmr and has_operate:
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                logger.debug(f"87T transformer differential operated: {name} ({samples.sum()} samples)")
                return True

    # Second pass: REF / LV / HV indicators combined with operate = transformer event
    for name in status_names:
        variants = _name_variants(name)
        if _looks_like_line_diff(variants):
            continue
        has_xfmr_hint = any(h in v for v in variants for h in xfmr_only_hints)
        has_operate = any(k in v for v in variants for k in operate_keywords)
        if has_xfmr_hint and has_operate:
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                logger.debug(f"Transformer-specific operate detected: {name} ({samples.sum()} samples)")
                return True

    # Third pass: known transformer relay family + generic differential trip wording.
    # This covers vendor outputs such as Siemens 7UT "Diff> TRIP" that do not spell out 87T.
    if known_transformer_relay:
        generic_diff_patterns = ['DIFF', 'DIFFERENTIAL', 'IDIFF', 'IDIF']
        for name in status_names:
            variants = _name_variants(name)
            if _looks_like_line_diff(variants):
                continue
            has_diff = any(p in v for v in variants for p in generic_diff_patterns)
            has_operate = any(k in v for v in variants for k in operate_keywords)
            if has_diff and has_operate:
                samples = status_dict.get(name, [])
                if len(samples) > 0 and samples.sum() > 0:
                    logger.debug(
                        "Transformer relay family %s routed generic differential signal as 87T: %s (%s samples)",
                        rec_dev_id or station_name or "UNKNOWN",
                        name,
                        samples.sum(),
                    )
                    return True

    return False


def _check_differential_operate(status_names: List[str], status_dict: dict) -> bool:
    """Check if 87L differential operated (issued trip command)."""
    # Siemens: "87L:I-DIFF*:Operate"
    # ABB REL: "DIFL*:OPERATE", "L3D-TRL1/2/3" (line 3-terminal differential)
    # DFR: "DIFF TRIP", "87 TRIP", "MAIN PROT" (if from differential relay)
    patterns = ['87L', '87 ', 'DIFF', 'DIFFERENTIAL',
                'L3D',        # ABB REL: "L3D-TRL1", "L3D-TRIPRES" (line 3-terminal diff)
                'LT3D',       # ABB/DFR exports: "LT3D-TRL1", "LT3D-IDL1MAG"
                'LDL',        # Line differential local/remote trip: "LDL-TRLOCAL"
                'DIFL',       # ABB: "DIFL:OPERATE"
                ]
    operate_keywords = ['OPERATE', 'TRIP',
                        'TRL',    # ABB REL: "L3D-TRL1" (trip per phase)
                        ]

    for name in status_names:
        variants = _name_variants(name)
        # Must have both diff pattern AND operate keyword
        has_diff = any(p in v for v in variants for p in patterns)
        has_operate = any(k in v for v in variants for k in operate_keywords)

        if has_diff and has_operate:
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                logger.debug(f"87L operated: {name} ({samples.sum()} samples)")
                return True

    return False


def _extract_phase_from_name(name_upper: str) -> Optional[str]:
    """
    Extract faulted phase from a channel name using all known naming conventions.

    Handles:
    - ABC notation: TRIP PHA, A PHASE FAULT, PHASE A FAULT, PHS A, ends with A/B/C
    - RST notation: TRIP (R), LP OPRT R, MPU TRIP (S), ends with R/S/T
    - Parenthesized: (R), (S), (T)
    """
    import re

    # Parenthesized RST: (R), (S), (T)
    if '(R)' in name_upper: return 'A'
    if '(S)' in name_upper: return 'B'
    if '(T)' in name_upper: return 'C'

    # Explicit ABC fault/phase keywords
    if any(k in name_upper for k in ['PHA FAULT', 'A PHASE FAULT', 'PHASE A FAULT', 'TRIP PHA', 'TRIP PH A', 'PHS A']): return 'A'
    if any(k in name_upper for k in ['PHB FAULT', 'B PHASE FAULT', 'PHASE B FAULT', 'TRIP PHB', 'TRIP PH B', 'PHS B']): return 'B'
    if any(k in name_upper for k in ['PHC FAULT', 'C PHASE FAULT', 'PHASE C FAULT', 'TRIP PHC', 'TRIP PH C', 'PHS C']): return 'C'

    # Space-delimited single letter: " A ", " B ", " C " or at end " A"/" B"/" C"
    if re.search(r'\bPHASE A\b|\bPH A\b', name_upper): return 'A'
    if re.search(r'\bPHASE B\b|\bPH B\b', name_upper): return 'B'
    if re.search(r'\bPHASE C\b|\bPH C\b', name_upper): return 'C'

    # RST single-letter: space-delimited " R ", " S ", " T " or suffix " R"/" S"/" T"
    if re.search(r'\bOPRT R\b|\bTRIP R\b|OPRT R$| R$', name_upper): return 'A'
    if re.search(r'\bOPRT S\b|\bTRIP S\b|OPRT S$| S$', name_upper): return 'B'
    if re.search(r'\bOPRT T\b|\bTRIP T\b|OPRT T$| T$', name_upper): return 'C'

    # L1/L2/L3 notation (ABB REL, some Siemens): L1=A, L2=B, L3=C
    if re.search(r'\bL1\b|L1$| L1[^0-9]', name_upper): return 'A'
    if re.search(r'\bL2\b|L2$| L2[^0-9]', name_upper): return 'B'
    if re.search(r'\bL3\b|L3$| L3[^0-9]', name_upper): return 'C'

    # PCS900 PhS notation: PhSA/PhSB/PhSC (phase selector output)
    if name_upper in ('PHSA',) or name_upper.endswith('.PHSA') or 'TRPA' in name_upper: return 'A'
    if name_upper in ('PHSB',) or name_upper.endswith('.PHSB') or 'TRPB' in name_upper: return 'B'
    if name_upper in ('PHSC',) or name_upper.endswith('.PHSC') or 'TRPC' in name_upper: return 'C'

    # PCS900 DZ zone+phase: DZ1R/DZ1S/DZ1T
    if re.search(r'DZ\d+R$', name_upper): return 'A'
    if re.search(r'DZ\d+S$', name_upper): return 'B'
    if re.search(r'DZ\d+T$', name_upper): return 'C'

    # Trailing A/B/C with space
    if name_upper.endswith(' A'): return 'A'
    if name_upper.endswith(' B'): return 'B'
    if name_upper.endswith(' C'): return 'C'

    return None


def _check_distance_operate(status_names: List[str], status_dict: dict) -> tuple:
    """
    Check if distance protection operated.

    Returns:
        (operated: bool, zones: List[str], phases: List[str])
    """
    zones = []
    phases = []
    operated = False

    # Siemens: "21 1:*:Operate", "Z 1:Operate", "Z 2:Operate", "Z 3:Operate"
    # Alstom/GE: "DIST Trip", "Z1", "Z2", "Z3"
    # NARI/CSC: "Zone1 Trip", "Trip PhA", "B Phase Fault"
    # External DFR (Indonesian): "LP OPRT R WTS2", "MPU MAIN 1 TRIP (S) UNGARAN 1"

    for name in status_names:
        variants = _name_variants(name)

        is_distance = any(k in v for v in variants for k in [
            '21 ', '21:', 'DIST', 'ZONE', 'Z1', 'Z2', 'Z3',
            'DISTN', 'DISTANCE',
            'LP OPRT', 'LP OPERATE',    # External DFR Indonesian: "LP OPRT R WTS2"
            'MPU MAIN', 'MPU TRIP',     # External DFR: "MPU MAIN 1 TRIP (S) UNGARAN 1"
            'TRIP PHA', 'TRIP PHB', 'TRIP PHC',  # NARI phase trip
            'CB1.TRP',                  # PCS900 Siemens: "CB1.TrpA", "CB1.TrpB", "CB1.TrpC"
            '21Q', '21D.',              # PCS900: "21Q1.Op", "21D.Op" (distance operate)
            'DZ1', 'DZ2', 'DZ3',       # PCS900: "DZ1R", "DZ1S", "DZ1T" (distance zone 1)
            'DIS.PICKUP', 'DIS. ',      # ABB REL: "Dis.Pickup L1", "Dis. forward"
            'ZM1', 'ZM2', 'ZM3',       # ABB REL: "ZM1_TRIP", "ZM1_TRL1" (zone management)
            'F21',                      # Qualitrol/external DFR: "L1_F21_REC" (function 21 = distance)
            'LINE PICKUP',              # GE D60: "LINE PICKUP OP" (line distance pickup)
        ])
        # SEND alone is not enough — only count operate/trip signals
        is_operate = any(k in v for v in variants for k in [
            'OPERATE', 'TRIP', 'OPRT',
            '.OP', '.TRP',              # PCS900: "21Q1.Op", "CB1.TrpA"
            'PICKUP',                   # ABB: "Dis.Pickup L2"
            'START',                    # GE D60: "START Z1 A On" (zone startup = distance engaged)
            '_REC',                     # Qualitrol DFR: "F21_REC" (function output received)
        ])

        if is_distance and is_operate:
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                operated = True
                logger.debug(f"Distance/line protection operated: {name} ({samples.sum()} samples)")

                # Extract zone
                if any(k in v for v in variants for k in ['Z1', 'Z 1', 'ZONE 1', 'ZONE1', '21Q1', 'DZ1', 'ZM1']):
                    zones.append('Z1')
                elif any(k in v for v in variants for k in ['Z2', 'Z 2', 'ZONE 2', 'ZONE2', '21Q2', 'DZ2', 'ZM2']):
                    zones.append('Z2')
                elif any(k in v for v in variants for k in ['Z3', 'Z 3', 'ZONE 3', 'ZONE3', '21Q3', 'DZ3', 'ZM3']):
                    zones.append('Z3')

                # Extract phase using unified helper
                ph = _extract_phase_from_name(_normalize_status_name(name))
                if ph:
                    phases.append(ph)

    # Standalone zone channels like "Z1", "Z2", "Z3", "Z4" that fired (no TRIP/OPRT suffix)
    # They confirm distance operated AND add zone info
    import re as _re
    for name in status_names:
        name_upper = _normalize_status_name(name)
        if _re.match(r'^Z\\d$', name_upper):
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                operated = True
                zones.append(name_upper)

    # Remove duplicates
    zones = list(set(zones))
    phases = list(set(phases))

    return operated, zones, phases


def _check_directional_ef_operate(status_names: List[str], status_dict: dict) -> bool:
    """Check if directional earth fault operated."""
    # Siemens: "67N*:Operate"
    # Alstom/GE: "DEF*"

    for name in status_names:
        variants = _name_variants(name)
        is_67n = any(k in v for v in variants for k in ['67N', 'DEF'])
        is_operate = any(k in v for v in variants for k in ['OPERATE', 'TRIP'])

        if is_67n and is_operate:
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                logger.debug(f"67N operated: {name} ({samples.sum()} samples)")
                return True

    return False


def _check_overcurrent_operate(status_names: List[str], status_dict: dict) -> tuple:
    """
    Check if OCR / 50-51 overcurrent elements operated.

    Returns:
        (operated: bool, phases: List[str])
    """
    phases = []
    operated = False

    oc_keywords = [
        'O/C', 'OVERCURRENT', 'I>', 'I>>', 'IP ', 'IPU', 'IP TRIP',
        'OCR', '50', '51',
    ]
    operate_keywords = ['PICKUP', 'PICKED UP', 'TRIP', 'PU']
    exclude_keywords = ['87', 'DIFF', 'DIST', 'ZONE', '21', '67N', 'DEF']

    for name in status_names:
        variants = _name_variants(name)
        if any(k in v for v in variants for k in exclude_keywords):
            continue
        has_oc = any(k in v for v in variants for k in oc_keywords)
        has_operate = any(k in v for v in variants for k in operate_keywords)
        if not (has_oc and has_operate):
            continue

        samples = status_dict.get(name, [])
        if len(samples) > 0 and samples.sum() > 0:
            operated = True
            ph = _extract_phase_from_name(_normalize_status_name(name))
            if ph:
                phases.append(ph)

    return operated, list(set(phases))


def _check_generic_trip_fallback(status_names: List[str], status_dict: dict) -> tuple:
    """
    Last-resort detection for transmission line files where only generic trip
    outputs are wired to the DFR — no zone-specific relay signals available.

    Handles:
    - Siemens 7SA DFR recordings: "Ln1:Group indicat.:Operate:phs A/B/C"
    - Qualitrol DFR phase trips: "R1_TRIP1_R", "R2_TRIP1_S", "R3_TRIP1_T"
    - GE D60: "TRIP PHASE A/B/C"
    - External DFR: "ANY TRIP", "RELAY TRIP", "LINE TRIP"

    IMPORTANT: Called ONLY after differential and 67N checks have already failed,
    so a match here is assumed to be distance protection.

    Returns: (operated: bool, phases: List[str])
    """
    phases = []
    operated = False

    # Patterns that indicate a relay operated — only relevant on transmission lines
    # where distance protection is the primary protection (87L already excluded above)
    GENERIC_PATTERNS = [
        'GROUP INDICAT',  # Siemens 7SA/7SL: group indicator fires for any prot. function
        'TRIP1_R',        # Qualitrol DFR: "R1_TRIP1_R" = phase R (A) trip output
        'TRIP1_S',        # Qualitrol DFR: "R2_TRIP1_S" = phase S (B) trip output
        'TRIP1_T',        # Qualitrol DFR: "R3_TRIP1_T" = phase T (C) trip output
        'ANY TRIP',       # External DFR generic trip output
        'ANYTRIP',        # External DFR: "R4_ANYTRIP_CBF"
        'RELAY TRIP',     # External DFR relay trip output
        'LINE TRIP',      # External DFR line trip
        'TRIP PHASE',     # GE D60: "TRIP PHASE A/B/C" (phase trip from distance)
        # CB position / auxiliary-contact channels (DFR-only recordings without relay signals)
        # 52A = normally-open CB aux contact — goes 0→1 when CB opens (trip confirmed)
        '52A',            # IEEE 52A aux contact energised when CB is open
        'CB OPEN',        # Explicit CB open status channel
        'CB TRIP',        # Generic CB trip output on DFR
        'PMT BUKA',       # Indonesian: PMT = Pemutus Tenaga (CB), BUKA = open
        'BUKA PMT',       # Indonesian variant
        'PMT TRIP',       # Indonesian DFR CB trip channel
    ]

    for name in status_names:
        variants = _name_variants(name)
        if any(p in v for v in variants for p in GENERIC_PATTERNS):
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                operated = True
                ph = _extract_phase_from_name(_normalize_status_name(name))
                if ph:
                    phases.append(ph)

    return operated, list(set(phases))


def _detect_operated_phases(status_names: List[str], status_dict: dict) -> List[str]:
    """
    Detect which phases operated from trip/operate/fault status channels.

    Handles all known naming conventions via _extract_phase_from_name.
    Prioritises explicit fault/trip channels over generic ones.
    """
    phases = []

    # Priority 1: explicit phase-fault channels (NARI CSC naming)
    explicit_fault_keywords = [
        'PHASE FAULT', 'PH FAULT', 'PHASE STARTUP',
        'TRIP PHA', 'TRIP PHB', 'TRIP PHC',
        'A PHASE FAULT', 'B PHASE FAULT', 'C PHASE FAULT',
    ]
    for name in status_names:
        name_upper = _normalize_status_name(name)
        if any(k in name_upper for k in explicit_fault_keywords):
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                ph = _extract_phase_from_name(name_upper)
                if ph:
                    phases.append(ph)

    if phases:
        return list(set(phases))

    # Priority 2: trip/operate channels with phase suffix
    for name in status_names:
        name_upper = _normalize_status_name(name)
        has_operate = any(k in name_upper for k in ['OPERATE', 'OPRT', 'TRIP', 'PICKUP'])
        if has_operate:
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                ph = _extract_phase_from_name(name_upper)
                if ph:
                    phases.append(ph)

    return list(set(phases))


def _detect_teleprotection_scheme(status_names: List[str], status_dict: dict) -> TeleprotectionScheme:
    """Detect teleprotection scheme from status channels."""
    for name in status_names:
        name_upper = _normalize_status_name(name)
        if 'POTT' in name_upper:
            return TeleprotectionScheme.POTT
        elif 'PUTT' in name_upper:
            return TeleprotectionScheme.PUTT
        elif 'DCB' in name_upper:
            return TeleprotectionScheme.DCB

    return TeleprotectionScheme.UNKNOWN


def _check_permission_received(status_names: List[str], status_dict: dict) -> bool:
    """Check if permissive signal was received from remote end."""
    # Siemens: "85-21Perm*:Receive"
    # Alstom/GE: "DIST. Chan Recv", "Chan Recv"
    # Generic: "RECV", "RECEIVE"

    for name in status_names:
        name_upper = _normalize_status_name(name)
        if any(k in name_upper for k in ['RECV', 'RECEIVE', 'PERM']):
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                return True

    return False


def _check_comms_failure(status_names: List[str], status_dict: dict) -> bool:
    """Check for communication link failure indicators."""
    # Siemens: "Prot. interf.*:Connection broken"

    for name in status_names:
        name_upper = _normalize_status_name(name)
        if any(k in name_upper for k in ['BROKEN', 'FAIL', 'LOSS', 'COMMS FAIL']):
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                return True

    return False


def _detect_trip_type(status_names: List[str], status_dict: dict) -> str:
    """Detect if trip was single-pole or three-pole."""
    # Siemens: "Circuit break.:Trip only pole A/B/C" vs "Definitive trip"
    # Generic: "1 PHS TRIP", "3 PHS TRIP", "1-POLE", "3-POLE"

    for name in status_names:
        name_upper = _normalize_status_name(name)
        samples = status_dict.get(name, [])

        if len(samples) > 0 and samples.sum() > 0:
            if any(k in name_upper for k in ['TRIP ONLY POLE', '1 PHS', '1-POLE', 'SINGLE POLE']):
                return "single_pole"
            elif any(k in name_upper for k in ['DEFINITIVE', '3 PHS', '3-POLE', 'THREE POLE']):
                return "three_pole"

    return "unknown"


def _check_auto_reclose_attempted(status_names: List[str], status_dict: dict) -> bool:
    """Check if auto-reclose was attempted."""
    # Siemens: "ProMon:Closure*"
    # Alstom/GE: "I A/R Close", "A/R"
    # NARI CSC: "1P Trip Init AR", "3P Trip Init AR", "Reclose", "AR Succ", "AR Fail"
    # External DFR: "AR INPROGRESS", "A/R OPRT"

    import numpy as _np
    ar_keywords = [
        'RECLOSE', 'CLOSURE', 'A R', 'AR ',
        'AR INPROG', 'INPROGRESS',          # External DFR: "AR INPROGRESS UNGARAN 1"
        '1P TRIP INIT', '3P TRIP INIT',     # NARI: "1P Trip Init AR"
        'AR 1POLE', '1POLE IN PROG',        # IED: "AR 1pole in prog"
        'A R OPRT',                          # External DFR: "A/R OPRT WTS2"
        '79 INPROG', 'CB1 79 INPROG',       # PCS900: "CB1.79.Inprog"
        'AR CLOSE CMD', 'LINE CLOSURE',      # ABB REL: "AR CLOSE Cmd.", "Line closure"
        '1POLE OPEN', '1POLE OPEN L',       # ABB REL: "1pole open L2"
        '1-POLE OPEN',                      # Siemens 7SA: "CB1:Circuit break.:Position 1-pole phsA:open"
    ]
    for name in status_names:
        name_upper = _normalize_status_name(name)
        if any(k in name_upper for k in ar_keywords):
            samples = status_dict.get(name, [])
            if len(samples) > 0 and samples.sum() > 0:
                return True

    # "Any Pole Dead" / "All Pole Dead" / "Position*open" / CB position going 0→1→0 (or 1→0→1)
    # = CB opened then reclosed.  Both 52A (0→1→0) and 52B (1→0→1) conventions are handled
    # by checking for BOTH a positive AND a negative edge anywhere in the trace.
    for name in status_names:
        name_upper = _normalize_status_name(name)
        is_cb_position = (
            any(k in name_upper for k in ['POLE DEAD', 'ANY POLE', 'ALL POLE'])
            or ('POSITION' in name_upper and 'OPEN' in name_upper)    # Siemens 7SA
            or ('PMT' in name_upper and ('BUKA' in name_upper or 'OPEN' in name_upper))  # Indonesian DFR
            or ('CB' in name_upper and 'OPEN' in name_upper)           # Generic CB OPEN channel
            or re.search(r'\b52A\b|\b52B\b', name_upper) is not None  # IEEE 52A/52B aux contacts
        )
        if is_cb_position:
            samples = status_dict.get(name, [])
            if len(samples) > 1:
                vals = _np.array(samples, dtype=int)
                edges = _np.diff(vals)
                if (edges > 0).any() and (edges < 0).any():
                    return True

    return False


def _check_auto_reclose_successful(status_names: List[str], status_dict: dict) -> Optional[bool]:
    """
    Check if auto-reclose was successful.

    Returns:
        True = successful reclose (transient fault)
        False = failed reclose (permanent fault)
        None = cannot determine
    """
    # Explicit success channels
    success_keywords = [
        'AR SUCC', 'RECLOSE SUCC', 'A R SUCC',     # normalized variants
        'CB CLOSE', 'BO CLOSE', 'CLOSE CMD',
        'SYN MEET', 'VOL MEET',
        'SUCC RCLS', '79 SUCC',                    # PCS900: "CB1.79.Succ_Rcls"
    ]
    # Explicit failure channels
    # NOTE: PERM_TRP intentionally excluded — in NR PCS900, "CB1.79.Perm_Trp1P/3P"
    # means "permission to trip" (mode flag), NOT a failed reclose. The correct
    # PCS900 failure channel is CB1.79.Fail_Rcls, covered by FAIL_RCLS below.
    failure_keywords = [
        'AR FAIL', 'AR LOCKOUT', 'AR FINAL',
        '79 FINAL', 'FINAL TRIP',
        'TOR', 'TRIP ON RECLOSE',
        'FAIL RCLS', '79 FAIL',                    # PCS900: "CB1.79.Fail_Rcls"
        'SOTF TOR',
        'LOCKOUT',
    ]

    import numpy as _np

    # Collect all matching channels first — success takes priority over failure.
    # (Do not short-circuit on first failure: a success signal later in the list
    #  should override a mode/status signal that superficially looks like failure.)
    found_success = False
    found_failure = False

    for name in status_names:
        name_upper = _normalize_status_name(name)
        samples = status_dict.get(name, [])
        if len(samples) == 0 or samples.sum() == 0:
            continue

        if any(k in name_upper for k in success_keywords):
            found_success = True
        if any(k in name_upper for k in failure_keywords):
            found_failure = True

    if found_success:
        return True      # explicit success overrides any ambiguous failure flags
    if found_failure:
        return False

    # "Any/All Pole Dead", "Position*open", or CB position channel going 0→1→0 (or 1→0→1)
    # means CB reclosed successfully.  Handles 52A (0→1→0) and 52B (1→0→1) conventions.
    for name in status_names:
        name_upper = _normalize_status_name(name)
        is_cb_position = (
            any(k in name_upper for k in ['POLE DEAD', 'ANY POLE', 'ALL POLE'])
            or ('POSITION' in name_upper and 'OPEN' in name_upper)    # Siemens 7SA
            or ('PMT' in name_upper and ('BUKA' in name_upper or 'OPEN' in name_upper))  # Indonesian DFR
            or ('CB' in name_upper and 'OPEN' in name_upper)           # Generic CB OPEN channel
            or re.search(r'\b52A\b|\b52B\b', name_upper) is not None  # IEEE 52A/52B aux contacts
        )
        if is_cb_position:
            samples = status_dict.get(name, [])
            if len(samples) > 1:
                vals = _np.array(samples, dtype=int)
                edges = _np.diff(vals)
                if (edges > 0).any() and (edges < 0).any():
                    return True

    return None
