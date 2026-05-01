"""
Channel Name Normalization
===========================
Maps vendor-specific channel names to canonical names (VA, VB, VC, VN, IA, IB, IC, IN).
Handles SEL, ABB, Siemens, GE, and Qualitrol naming conventions.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cache for loaded mappings
_CHANNEL_MAPPINGS: Optional[dict] = None


def _extract_protection_phase(raw_upper: str) -> Optional[str]:
    """Extract phase tags from protection-analog labels such as IDA or phs A."""
    line_diff_match = re.search(r"\b(?:IDL|IL|ID)([123])(?:D|MAG)?\b|\bIL([123])\s*D\b", raw_upper)
    if line_diff_match:
        phase_no = line_diff_match.group(1) or line_diff_match.group(2)
        return {"1": "A", "2": "B", "3": "C"}.get(phase_no)
    if re.search(r"\bIDNS(?:MAG)?\b", raw_upper):
        return "N"

    patterns = (
        r"\bPHS\s*([ABCN])\b",
        r"\bPHASE\s*([ABCN])\b",
        r"\bID([ABCN])\b",
        r"\bIRST([ABCN])\b",
        r"\bRSTR([ABCN])\b",
        r"\bBIAS([ABCN])\b",
        r"[_:\.\-\s]([ABCN])(?:$|[_:\.\-\s])",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_upper)
        if match:
            return match.group(1)
    return None


def _normalize_protection_analog_channel(raw_name: str, raw_upper: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Recognize line/transformer differential analog channels that often use non-A units
    such as pu, %, or In but still represent current-derived quantities.
    """
    text = re.sub(r"[_:/\-\[\]\(\)]+", " ", raw_upper)
    phase = _extract_protection_phase(raw_upper)

    if re.search(r"\b(?:HM2|H2|HM5|H5|HARM(?:ONIC)?)\b", text):
        return None

    if re.search(r"\b(?:IREST|IRESTR|I RESTR|I RESTRAINT|RSTR|RESTRAINT|BIAS|IBIAS)\b", text):
        return {
            "canonical_name": f"IREST_{phase}" if phase else "IREST",
            "phase": phase,
            "measurement": "current",
        }

    if (
        re.search(r"\b(?:IDIFF|IDIF|I DIFF|I DIFF OPERATE|I DIFF PHS|87L I DIFF)\b", text)
        or re.search(r"\b(?:IDL[123]?(?:MAG)?|IL[123]\s*D|IL[123]D|LT3D IDL[123]?(?:MAG)?|IDNSMAG|LDL)\b", text)
    ):
        return {
            "canonical_name": f"IDIFF_{phase}" if phase else "IDIFF",
            "phase": phase,
            "measurement": "current",
        }

    if re.search(r"\b87T\b", text) and re.search(r"\bID[ABCN]\b", text):
        return {
            "canonical_name": f"IDIFF_{phase}" if phase else raw_name,
            "phase": phase,
            "measurement": "current",
        }

    # Zero-sequence / residual / REF protection currents (e.g. HVS.64REF.3i0d, 3i0Ext)
    # Units 'In' (per-unit nominal) or standard 'A' — always represent current.
    if re.search(r"\b(?:3I0|3IO|64REF)\b", text):
        return {
            "canonical_name": "IN",
            "phase": "N",
            "measurement": "current",
        }

    return None


def load_channel_mappings(config_path: Optional[str] = None) -> dict:
    """Load channel mappings from JSON file (cached)."""
    global _CHANNEL_MAPPINGS

    if _CHANNEL_MAPPINGS is not None:
        return _CHANNEL_MAPPINGS

    if config_path is None:
        # Default to config/channel_mappings.json relative to this file
        config_path = Path(__file__).parent.parent / "config" / "channel_mappings.json"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _CHANNEL_MAPPINGS = json.load(f)
        logger.info(f"Loaded channel mappings from {config_path}")
        return _CHANNEL_MAPPINGS
    except Exception as e:
        logger.error(f"Failed to load channel mappings: {e}")
        return {"manufacturers": {}, "manufacturer_detection": {}}


def detect_manufacturer(rec_dev_id: str, station_name: str = "") -> str:
    """
    Guess manufacturer from relay model ID or station name.

    Args:
        rec_dev_id: Relay device ID from .cfg file
        station_name: Station name from .cfg file

    Returns:
        Manufacturer name ("SEL", "ABB", "SIEMENS", "GE", "QUALITROL") or "UNKNOWN"
    """
    mappings = load_channel_mappings()
    detection_patterns = mappings.get("manufacturer_detection", {})

    # Combine device ID and station name for searching
    search_text = f"{rec_dev_id} {station_name}".upper()

    # ── Regex-based detection first (faster and more precise) ────────────────
    # Siemens SIPROTEC 5: device ID format BM + 10 digits (e.g. BM1906001619)
    # BM = "Bestellnummer Modell" — Siemens DIGSI 5 order number
    if re.match(r'^BM\d{8,12}$', rec_dev_id.strip()):
        logger.debug(f"Detected manufacturer: SIEMENS (BM order number: {rec_dev_id})")
        return "SIEMENS"

    # NARI/NR: station_name = "NR" is too short for safe substring match,
    # so handle it explicitly before the generic loop.
    # rec_dev_id patterns: LINE_DISTANCE_RELAY, LINE_DIFFERENTIAL_RELAY
    # station_name: "NR", "NR ELECTRIC", or file paths containing PCS900
    dev_upper = rec_dev_id.strip().upper()
    sta_upper = station_name.strip().upper()
    if dev_upper in ("LINE_DISTANCE_RELAY", "LINE_DIFFERENTIAL_RELAY",
                     "LINE DISTANCE RELAY", "LINE DIFFERENTIAL RELAY"):
        logger.debug(f"Detected manufacturer: NARI (rec_dev_id: {rec_dev_id})")
        return "NARI"
    if sta_upper in ("NR", "NR ELECTRIC") or re.match(r'^PCS[-_]?9', dev_upper):
        logger.debug(f"Detected manufacturer: NARI (station: {station_name})")
        return "NARI"

    for manufacturer, patterns in detection_patterns.items():
        for pattern in patterns:
            if pattern.upper() in search_text:
                logger.debug(f"Detected manufacturer: {manufacturer} (pattern: {pattern})")
                return manufacturer

    logger.warning(f"Could not detect manufacturer from '{rec_dev_id}' or '{station_name}'")
    return "UNKNOWN"


def normalize_channel_name(raw_name: str, unit: str, manufacturer: str = "UNKNOWN") -> Dict[str, Optional[str]]:
    """
    Normalize a vendor-specific channel name to canonical form.

    Args:
        raw_name: Original channel name from .cfg file
        unit: Channel unit ("kV", "V", "A", "kA", etc.)
        manufacturer: Detected manufacturer

    Returns:
        Dictionary with:
            - canonical_name: Standardized name (VA/VB/VC/VN/IA/IB/IC/IN)
            - phase: Phase identifier ("A", "B", "C", "N", or None)
            - measurement: "voltage" or "current"
    """
    # Normalize inputs
    raw_upper = raw_name.strip().upper()
    unit_upper = unit.strip().upper()

    protection_analog = _normalize_protection_analog_channel(raw_name, raw_upper)
    if protection_analog is not None:
        logger.debug(f"Protection analog match '{raw_name}' -> '{protection_analog['canonical_name']}'")
        return protection_analog

    # Determine measurement type from unit
    if (
        unit_upper.startswith("U/")
        or unit_upper in {"KV", "V", "MV"}
        or re.search(r"\b(?:KV|V|MV)\b", unit_upper)
    ):
        measurement = "voltage"
    elif (
        unit_upper.startswith("I/")
        or unit_upper in {"KA", "A", "IN", "PU", "P.U."}  # 'In' = per-unit nominal current (COMTRADE standard)
        or re.search(r"\b(?:KA|A)\b", unit_upper)
    ):
        measurement = "current"
    else:
        measurement = "unknown"
        logger.warning(f"Unknown unit '{unit}' for channel '{raw_name}'")

    # Try manufacturer-specific patterns first
    mappings = load_channel_mappings()
    manufacturer_patterns = mappings.get("manufacturers", {}).get(manufacturer, {}).get("channel_patterns", {})

    for canonical, patterns in manufacturer_patterns.items():
        for pattern in patterns:
            pat_upper = pattern.upper()
            # Use word-boundary match to avoid false substring hits (e.g. "IN" inside "LINE")
            if pat_upper == raw_upper or re.search(r'\b' + re.escape(pat_upper) + r'\b', raw_upper):
                # Extract phase from canonical name
                phase = canonical[-1] if canonical[-1] in "ABCN" else None
                logger.debug(f"Matched '{raw_name}' → '{canonical}' (manufacturer: {manufacturer})")
                return {
                    "canonical_name": canonical,
                    "phase": phase,
                    "measurement": measurement
                }

    # Fall back to generic pattern matching
    canonical_name, phase = _generic_pattern_match(raw_upper, measurement)

    if canonical_name:
        logger.debug(f"Generic match '{raw_name}' → '{canonical_name}'")
        return {
            "canonical_name": canonical_name,
            "phase": phase,
            "measurement": measurement
        }

    # No match found - use raw name
    logger.warning(f"Could not normalize channel '{raw_name}' (unit: {unit}, mfr: {manufacturer})")
    return {
        "canonical_name": raw_name,
        "phase": None,
        "measurement": measurement
    }


def _generic_pattern_match(raw_upper: str, measurement: str) -> tuple:
    """
    Generic pattern matching for common channel naming conventions.
    Supports both ABC and RST phase naming.

    Returns:
        (canonical_name, phase) tuple
    """
    # Token-aware matching avoids false positives like "IN" in "LINE_A_IL1".
    tokens = set(re.findall(r"[A-Z0-9]+", raw_upper))

    def has_token(*candidates: str) -> bool:
        return any(c in tokens for c in candidates)

    def has_substr(*candidates: str) -> bool:
        return any(c in raw_upper for c in candidates)

    # Neutral/residual patterns first, but token-aware (prevents LINE -> IN mistakes).
    if measurement == "voltage":
        if has_token("VN", "V0", "3V0", "VG", "UE", "U0", "UN", "UEN") or has_substr("V_N", "V_RES", "RESVOL"):
            return ("VN", "N")
    if measurement == "current":
        if has_token("IN", "I0", "3I0", "IG", "IE") or has_substr("I_N", "I_RES", "RESCUR"):
            return ("IN", "N")

    # Common Siemens OCR / feeder line-line voltages.
    if measurement == "voltage":
        if has_token("UL12", "VL12", "U12", "V12", "ULAB", "VAB"):
            return ("VAB", None)
        if has_token("UL23", "VL23", "U23", "V23", "ULBC", "VBC"):
            return ("VBC", None)
        if has_token("UL31", "VL31", "U31", "V31", "ULCA", "VCA"):
            return ("VCA", None)

    # Explicit CT/VT R/S/T naming used in several PLN records.
    if measurement == "voltage":
        if re.search(r"\b(VT|V|U)\s*R\b", raw_upper):
            return ("VA", "A")
        if re.search(r"\b(VT|V|U)\s*S\b", raw_upper):
            return ("VB", "B")
        if re.search(r"\b(VT|V|U)\s*T\b", raw_upper):
            return ("VC", "C")
    if measurement == "current":
        if re.search(r"\b(CT|I)\s*R\b", raw_upper):
            return ("IA", "A")
        if re.search(r"\b(CT|I)\s*S\b", raw_upper):
            return ("IB", "B")
        if re.search(r"\b(CT|I)\s*T\b", raw_upper):
            return ("IC", "C")

    # RST and ABC notations. Phase-to-neutral variants (VRN/VAN etc.) map to same phase.
    if has_token("VR", "UL1", "UA", "VA", "V1", "VRN", "VAN") or has_substr("V_A", "VPHSA", "V PHASE A", "IPHSA"):
        return ("VA", "A")
    if has_token("VS", "UL2", "UB", "VB", "V2", "VSN", "VBN") or has_substr("V_B", "VPHSB", "V PHASE B"):
        return ("VB", "B")
    if has_token("VT", "UL3", "UC", "VC", "V3", "VTN", "VCN") or has_substr("V_C", "VPHSC", "V PHASE C"):
        return ("VC", "C")

    if has_token("IR", "IL1", "IA", "I1", "IRN", "IAN") or has_substr("I_A", "I PHASE A"):
        return ("IA", "A")
    if has_token("IS", "IL2", "IB", "I2", "ISN", "IBN") or has_substr("I_B", "I PHASE B"):
        return ("IB", "B")
    if has_token("IT", "IL3", "IC", "I3", "ITN", "ICN") or has_substr("I_C", "I PHASE C"):
        return ("IC", "C")

    return (None, None)
