"""
Pipeline Analisis Gangguan
===========================
Menghubungkan: parse COMTRADE -> deteksi proteksi -> deteksi gangguan -> ekstraksi fitur.
Mengarahkan ke fitur rele jarak atau rele diferensial sesuai jenis proteksi.
"""

from dataclasses import dataclass
from typing import Optional, Union, List
import logging

from .comtrade_parser import parse_comtrade
from .protection_router import determine_protection, ProtectionEvent
from .fault_detector import detect_fault, FaultEvent
from .feature_extractor import (
    extract_distance_features,
    extract_differential_features,
    DistanceFeatures,
    DifferentialFeatures
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of processing one COMTRADE file."""
    filepath: str
    station_name: str
    relay_model: str
    protection_event: ProtectionEvent
    fault_event: Optional[FaultEvent]
    features: Optional[Union[DistanceFeatures, DifferentialFeatures]]
    classifiable: bool
    skip_reason: Optional[str]
    errors: List[str]
    warnings: List[str]


def process_comtrade(cfg_path: str) -> ExtractionResult:
    """
    Full pipeline: parse → detect protection → detect fault → extract features.

    Steps:
    1. Parse COMTRADE file
    2. Determine protection type from status channels
    3. Detect fault inception
    4. Based on protection type:
       - DISTANCE → extract_distance_features() → classifiable = True
       - DIFFERENTIAL → extract_differential_features() → classifiable = False
       - UNKNOWN → extract minimal universal features → classifiable = False
    5. Return ExtractionResult

    Never crashes - logs errors and returns partial results.

    Args:
        cfg_path: Path to .cfg file

    Returns:
        ExtractionResult with features and routing decision
    """

    errors = []
    warnings = []

    # Step 1: Parse COMTRADE file
    logger.info(f"Processing: {cfg_path}")

    record = parse_comtrade(cfg_path)
    if record is None:
        return ExtractionResult(
            filepath=cfg_path,
            station_name="UNKNOWN",
            relay_model="UNKNOWN",
            protection_event=None,
            fault_event=None,
            features=None,
            classifiable=False,
            skip_reason="Failed to parse COMTRADE file",
            errors=["Failed to parse COMTRADE file"],
            warnings=[]
        )

    # Collect parser warnings
    warnings.extend(record.warnings)

    # Step 2: Determine protection type
    protection = determine_protection(record)
    warnings.extend(protection.warnings)

    logger.info(f"Protection type: {protection.primary_protection.value}, Classifiable: {protection.classifiable}")

    # Step 3: Detect fault inception
    fault = detect_fault(record)
    if fault is None:
        return ExtractionResult(
            filepath=cfg_path,
            station_name=record.station_name,
            relay_model=record.rec_dev_id,
            protection_event=protection,
            fault_event=None,
            features=None,
            classifiable=False,
            skip_reason="No fault detected in waveforms",
            errors=["Fault detection failed"],
            warnings=warnings
        )

    logger.info(f"Fault detected at {fault.inception_time:.4f}s, method: {fault.detection_method}")

    # Step 4: Extract features based on protection type
    features = None
    classifiable = False
    skip_reason = None

    if protection.primary_protection.value == "21":
        # Distance protection → extract distance features (full feature set)
        logger.info("Extracting distance features (impedance + universal)...")
        features = extract_distance_features(record, fault, protection)

        if features is None:
            errors.append("Distance feature extraction failed")
            skip_reason = "Distance feature extraction failed"
        else:
            logger.info("Distance features extracted successfully")
            classifiable = True

    elif protection.primary_protection.value == "87L":
        # Differential protection → extract differential features
        # CHANGED: Now classifiable for PETIR detection using universal features
        logger.info("Extracting differential features (universal only, classifiable for PETIR)...")
        features = extract_differential_features(record, fault, protection)

        if features is None:
            errors.append("Differential feature extraction failed")
            skip_reason = "Differential feature extraction failed"
        else:
            logger.info("Differential features extracted successfully")
            # Mark as classifiable - can use universal features for PETIR detection
            classifiable = True
            skip_reason = None

    else:
        # Unknown protection (likely DFR) → extract universal features
        # CHANGED: Now classifiable for PETIR detection using universal features
        logger.info(f"Unknown protection type (likely DFR), extracting universal features...")
        features = extract_differential_features(record, fault, protection)

        if features is None:
            errors.append("Feature extraction failed for unknown protection type")
            skip_reason = "Feature extraction failed"
        else:
            logger.info("Universal features extracted successfully (DFR-compatible)")
            # Mark as classifiable - DFR files can be classified using universal features
            classifiable = True
            skip_reason = None

    return ExtractionResult(
        filepath=cfg_path,
        station_name=record.station_name,
        relay_model=record.rec_dev_id,
        protection_event=protection,
        fault_event=fault,
        features=features,
        classifiable=classifiable,
        skip_reason=skip_reason,
        errors=errors,
        warnings=warnings
    )


def process_batch(cfg_paths: List[str], output_csv: str) -> None:
    """
    Process multiple COMTRADE files and save features to CSV.

    Output CSV columns for distance cases:
    filepath, station, relay, protection_type, zone_operated,
    r_x_ratio, z_magnitude, z_angle, voltage_sag_depth,
    di_dt_max, peak_current, i0_i1_ratio, thd_percent,
    inception_angle, teleprotection_received, comms_failure,
    trip_type, reclose_attempted, reclose_successful, fault_count,
    faulted_phases, fault_type, is_ground_fault,
    classifiable, skip_reason

    Args:
        cfg_paths: List of paths to .cfg files
        output_csv: Path to output CSV file
    """

    import csv

    results = []
    for cfg_path in cfg_paths:
        try:
            result = process_comtrade(cfg_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {cfg_path}: {e}", exc_info=True)

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'filepath', 'station', 'relay', 'protection_type', 'zone_operated',
            'r_x_ratio', 'z_magnitude', 'z_angle', 'voltage_sag_depth',
            'di_dt_max', 'peak_current', 'i0_i1_ratio', 'thd_percent',
            'inception_angle', 'teleprotection_received', 'comms_failure',
            'trip_type', 'reclose_attempted', 'reclose_successful', 'fault_count',
            'faulted_phases', 'fault_type', 'is_ground_fault',
            'classifiable', 'skip_reason', 'errors', 'warnings'
        ])

        # Data rows
        for result in results:
            if isinstance(result.features, DistanceFeatures):
                f = result.features
                writer.writerow([
                    result.filepath,
                    result.station_name,
                    result.relay_model,
                    result.protection_event.primary_protection.value,
                    f.zone_operated,
                    f.r_x_ratio,
                    f.z_magnitude_ohms,
                    f.z_angle_degrees,
                    f.voltage_sag_depth_pu,
                    f.di_dt_max,
                    f.peak_fault_current_a,
                    f.i0_i1_ratio,
                    f.thd_percent,
                    f.inception_angle_degrees,
                    f.teleprotection_received,
                    f.comms_failure,
                    f.trip_type,
                    f.reclose_attempted,
                    f.reclose_successful,
                    f.fault_count,
                    ','.join(f.faulted_phases),
                    f.fault_type,
                    f.is_ground_fault,
                    result.classifiable,
                    result.skip_reason,
                    ';'.join(result.errors) if result.errors else '',
                    ';'.join(result.warnings[:3]) if result.warnings else ''  # Limit warnings to first 3
                ])
            elif isinstance(result.features, DifferentialFeatures):
                f = result.features
                writer.writerow([
                    result.filepath,
                    result.station_name,
                    result.relay_model,
                    result.protection_event.primary_protection.value,
                    '',  # No zone for differential
                    None,  # No R/X for differential
                    None,  # No Z magnitude
                    None,  # No Z angle
                    None,  # No voltage sag (differential doesn't use voltage)
                    f.di_dt_max,
                    f.peak_fault_current_a,
                    f.i0_i1_ratio,
                    f.thd_percent,
                    f.inception_angle_degrees,
                    result.protection_event.permission_received,
                    result.protection_event.comms_failure,
                    result.protection_event.trip_type,
                    f.reclose_attempted,
                    f.reclose_successful,
                    len(result.fault_event.reclose_events) + 1 if result.fault_event else 1,
                    ','.join(f.faulted_phases),
                    f.fault_type,
                    f.is_ground_fault,
                    result.classifiable,
                    result.skip_reason,
                    ';'.join(result.errors) if result.errors else '',
                    ';'.join(result.warnings[:3]) if result.warnings else ''
                ])
            else:
                # No features extracted
                writer.writerow([
                    result.filepath,
                    result.station_name,
                    result.relay_model,
                    result.protection_event.primary_protection.value if result.protection_event else 'UNKNOWN',
                    '',
                    None, None, None, None, None, None, None, None, None,
                    result.protection_event.permission_received if result.protection_event else False,
                    result.protection_event.comms_failure if result.protection_event else False,
                    result.protection_event.trip_type if result.protection_event else 'unknown',
                    False, None, 0,
                    '', 'UNKNOWN', False,
                    result.classifiable,
                    result.skip_reason,
                    ';'.join(result.errors) if result.errors else '',
                    ';'.join(result.warnings[:3]) if result.warnings else ''
                ])

    logger.info(f"Processed {len(results)} files, output saved to {output_csv}")
    logger.info(f"Classifiable (distance): {sum(1 for r in results if r.classifiable)}")
    logger.info(f"Non-classifiable: {sum(1 for r in results if not r.classifiable)}")
