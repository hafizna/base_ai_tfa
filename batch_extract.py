"""
Ekstraksi Fitur Batch
=====================
Mencari semua file COMTRADE berlabel di raw_data/, menjalankan pipeline
analisis gangguan, dan menulis fitur ke:
  - data/features/labeled_features.csv
  - data/features/labeled_features_87l.csv

Label is inferred from the folder name (PETIR, LAYANG, POHON, HEWAN,
KONDUKTOR, BENDA_ASING).  Files in 'olah' or '_extracted' sub-folders
are skipped (processed copies).

ZIP/RAR archives are extracted permanently in-place before scanning.
Each archive is extracted once; subsequent runs skip already-extracted
archives (tracked via a .extracted marker file beside each archive).

Requirements for RAR support:
    pip install rarfile
    # Windows: install WinRAR or UnRAR and ensure it's on PATH
    # Linux/Mac: sudo apt install unrar  OR  brew install rar

Run from the repo root:
    python batch_extract.py
"""

import os
import sys
import csv
import zipfile
import warnings
import traceback
from pathlib import Path
from dataclasses import asdict

# Optional RAR support
try:
    import rarfile
    _RAR_AVAILABLE = True
except ImportError:
    _RAR_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from core.comtrade_parser import parse_comtrade
from core.protection_router import determine_protection
from core.fault_detector import detect_fault
from core.feature_extractor import extract_distance_features, extract_differential_features

RAW_DATA = Path(__file__).parent.parent / "raw_data"


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _marker_path(archive: Path) -> Path:
    """Return the marker file path that signals an archive was already extracted."""
    return archive.with_suffix(archive.suffix + ".extracted")


def _extract_zip(archive: Path) -> int:
    """Extract a ZIP archive into its parent folder. Returns number of files extracted."""
    dest = archive.parent
    extracted = 0
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            # Skip macOS metadata junk
            if "__MACOSX" in member.filename or member.filename.startswith("."):
                continue
            target = dest / member.filename
            # Prevent path traversal
            if not str(target.resolve()).startswith(str(dest.resolve())):
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
    return extracted


def _extract_rar(archive: Path) -> int:
    """Extract a RAR archive into its parent folder. Returns number of files extracted."""
    if not _RAR_AVAILABLE:
        print(f"  [SKIP] rarfile not installed — cannot extract {archive.name}")
        print("         Run: pip install rarfile  (and install WinRAR/UnRAR on PATH)")
        return 0
    dest = archive.parent
    extracted = 0
    with rarfile.RarFile(str(archive), "r") as rf:
        for member in rf.infolist():
            if member.is_dir():
                continue
            target = dest / member.filename
            if not str(target.resolve()).startswith(str(dest.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            rf.extract(member, str(dest))
            extracted += 1
    return extracted


def _already_extracted(archive: Path) -> bool:
    """
    Return True if this archive should be skipped because its contents
    already exist.  Two signals:
      1. A .extracted marker file beside the archive (written by a prior run).
      2. A folder with the same stem exists beside the archive
         (e.g. archive.zip → archive/ folder already there).
    """
    if _marker_path(archive).exists():
        return True
    stem_folder = archive.with_suffix("")
    if stem_folder.is_dir():
        return True
    return False


def extract_archives(root: Path) -> None:
    """
    Recursively find all .zip and .rar files under root and extract them
    permanently in-place.

    Skipped when:
      - A .extracted marker is already present beside the archive, OR
      - A folder with the same stem already exists beside the archive.

    Corrupt or unreadable archives are skipped with a warning — they
    never cause the overall batch run to abort.
    """
    archives = sorted(root.rglob("*.zip")) + sorted(root.rglob("*.ZIP")) + \
               sorted(root.rglob("*.rar")) + sorted(root.rglob("*.RAR"))

    if not archives:
        return

    print(f"\n=== ARCHIVE EXTRACTION ({len(archives)} found) ===")
    for archive in archives:
        if _already_extracted(archive):
            print(f"  [SKIP] already extracted: {archive.name}")
            continue

        suffix = archive.suffix.lower()
        print(f"  [EXTRACT] {archive.relative_to(root)} ... ", end="", flush=True)
        try:
            if suffix == ".zip":
                count = _extract_zip(archive)
            elif suffix == ".rar":
                count = _extract_rar(archive)
            else:
                count = 0

            if count > 0:
                _marker_path(archive).touch()  # mark so future runs skip it
                print(f"{count} files extracted")
            else:
                print("0 files (skipped or empty)")
        except Exception as e:
            print(f"SKIP (corrupt or unreadable): {e}")
    print()


OUT_DIR  = Path(__file__).parent / "data" / "features"
OUT_CSV  = OUT_DIR / "labeled_features.csv"
OUT_CSV_87L = OUT_DIR / "labeled_features_87l.csv"
ERR_CSV  = OUT_DIR / "extraction_errors.csv"

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# Labels are based on PHYSICAL cause, not PLN's administrative accountability
# categories.  PLN uses administrative labels (APPL, DISTRIBUSI) that describe
# who is responsible, not what physically caused the fault — these are excluded
# because they carry no reliable physical-cause signal for the model.
#
# Classes:
#   PETIR        — lightning (direct strike or induced overvoltage)
#   LAYANG       — kite (layang-layang); kept separate — distinct seasonal/
#                  geographic pattern and very different waveform signature
#                  from generic foreign objects
#   POHON        — tree contact (vegetation encroachment)
#   HEWAN        — any animal (ular/snake, babi/pig, burung/bird, tikus/rat,
#                  biawak/monitor lizard, etc.)
#   BENDA_ASING  — non-living foreign object (other than kite)
#   KONDUKTOR    — conductor / tower structural failure
#   PERALATAN    — isolator / VT-CVT / teleprotection / other equipment-origin
#
# Intentionally excluded:
#   APPL         — "Akibat Pekerjaan Pihak Luar" (external-party work);
#                  administrative accountability label, physical cause unknown
#   DISTRIBUSI   — distribution-side transformer trip; not a transmission
#                  line fault, different protection context
# ---------------------------------------------------------------------------

LABEL_MAP = [
    # Weather / environment
    ("petir",         "PETIR"),
    ("sambaran",      "PETIR"),     # "sambaran petir" = direct lightning strike

    # Human-launched object — kept as its own class
    ("layang",        "LAYANG"),    # layang-layang (kite)

    # Vegetation
    ("pohon",         "POHON"),

    # Animals — all species grouped into HEWAN
    ("hewan",         "HEWAN"),
    ("binatang",      "HEWAN"),     # generic animal (UPT Semarang & others use this)
    ("ular",          "HEWAN"),     # snake
    ("babi",          "HEWAN"),     # pig / wild boar
    ("tikus",         "HEWAN"),     # rat / rodent
    ("burung",        "HEWAN"),     # bird
    ("biawak",        "HEWAN"),     # monitor lizard
    ("kukang",        "HEWAN"),     # slow loris
    ("monyet",        "HEWAN"),     # monkey

    # Foreign object (non-living, non-kite)
    ("benda asing",   "BENDA_ASING"),
    ("benda_asing",   "BENDA_ASING"),

    # Conductor / structural failure
    ("tower roboh",   "KONDUKTOR"),
    ("konduktor",     "KONDUKTOR"),

    # Equipment / protection / telecom-origin cases
    ("alat isolator",       "PERALATAN"),
    ("isolator",            "PERALATAN"),
    ("kerusakan peralatan", "PERALATAN"),
    ("gangguan peralatan",  "PERALATAN"),
    ("pilot wire",          "PERALATAN"),
    ("pilotwire",           "PERALATAN"),
    ("teleprotection",      "PERALATAN"),
    ("teleproteksi",        "PERALATAN"),
    ("plcc",                "PERALATAN"),
    ("peralatan",           "PERALATAN"),
]

# Sub-folder fragments to skip (processed copies, analysis outputs)
SKIP_FRAGMENTS = ["olah", "_extracted", "locus z", "locus_z", "locus\\", "/locus/", "analisa"]


def infer_label(path_str: str) -> str:
    low = path_str.lower()
    for fragment, label in LABEL_MAP:
        if fragment in low:
            return label
    return ""


def should_skip(path_str: str) -> bool:
    low = path_str.lower()
    return any(frag in low for frag in SKIP_FRAGMENTS)


def find_labeled_cfgs(root: Path):
    """Yield (cfg_path, label) for every valid labeled CFG file."""
    seen = set()
    for cfg_path in sorted(root.rglob("*.cfg")) + sorted(root.rglob("*.CFG")):
        # Deduplicate: Windows filesystem is case-insensitive so *.cfg and *.CFG
        # can return the same physical file twice.
        resolved = cfg_path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)

        path_str = str(cfg_path)
        if should_skip(path_str.lower()):
            continue
        label = infer_label(path_str)
        if not label:
            continue
        # Check matching DAT exists
        dat = cfg_path.with_suffix(".dat")
        if not dat.exists():
            dat = cfg_path.with_suffix(".DAT")
        if not dat.exists():
            continue
        yield cfg_path, label


def flatten_features(feat, label, cfg_path, prot, fault):
    """Convert DistanceFeatures to a flat dict for CSV output."""
    d = {}
    d["label"]        = label
    d["cfg_path"]     = str(cfg_path)
    d["station_name"] = feat.station_name
    d["relay_model"]  = feat.relay_model
    d["voltage_kv"]   = feat.voltage_kv

    # Protection context
    d["protection_type"]  = prot.primary_protection.name
    d["zone_operated"]    = feat.zone_operated
    d["trip_type"]        = feat.trip_type
    d["faulted_phases"]   = "+".join(feat.faulted_phases) if feat.faulted_phases else ""
    d["fault_type"]       = feat.fault_type
    d["is_ground_fault"]  = feat.is_ground_fault

    # Reclose
    d["reclose_attempted"]  = feat.reclose_attempted
    d["reclose_successful"] = feat.reclose_successful
    d["reclose_time_ms"]    = feat.reclose_time_ms
    d["fault_count"]        = feat.fault_count

    # Fault duration (from fault detector)
    d["fault_duration_ms"]  = fault.duration_ms
    d["fault_inception_ms"] = round(fault.inception_time * 1000, 2)

    # Waveform features
    d["di_dt_max"]              = feat.di_dt_max
    d["di_dt_phase"]            = feat.di_dt_phase
    d["peak_fault_current_a"]   = feat.peak_fault_current_a
    d["peak_fault_phase"]       = feat.peak_fault_phase
    d["i0_i1_ratio"]            = feat.i0_i1_ratio
    d["thd_percent"]            = feat.thd_percent
    d["inception_angle_degrees"]= feat.inception_angle_degrees
    d["voltage_sag_depth_pu"]   = feat.voltage_sag_depth_pu
    d["voltage_sag_phase"]      = feat.voltage_sag_phase
    d["voltage_phase_ratio_spread_pu"] = feat.voltage_phase_ratio_spread_pu
    d["healthy_phase_voltage_ratio"]   = feat.healthy_phase_voltage_ratio
    d["v2_v1_ratio"]                   = feat.v2_v1_ratio
    d["voltage_thd_max_percent"]       = feat.voltage_thd_max_percent
    d["v_prefault_v"]                  = feat.v_prefault_v
    d["v_fault_v"]                     = feat.v_fault_v

    # Impedance
    d["r_x_ratio"]        = feat.r_x_ratio
    d["z_magnitude_ohms"] = feat.z_magnitude_ohms
    d["z_angle_degrees"]  = feat.z_angle_degrees

    # Metadata
    d["sampling_rate_hz"]   = feat.sampling_rate_hz
    d["record_duration_ms"] = feat.record_duration_ms
    d["teleprotection_rx"]  = feat.teleprotection_received
    d["comms_failure"]      = feat.comms_failure

    # Quality flags — rows failing these should be excluded from training
    # peak < 200A primary means secondary-scaling issue (no CT ratio in file)
    # duration < 5ms means false detection (contact bounce / noise)
    d["scaling_ok"]  = feat.peak_fault_current_a >= 200.0
    d["duration_ok"] = fault.duration_ms >= 5.0

    return d


def flatten_differential_features(feat, label, cfg_path, prot, fault, record):
    """Convert DifferentialFeatures to a flat dict for the 87L feature CSV."""
    d = {}
    d["label"]        = label
    d["cfg_path"]     = str(cfg_path)
    d["station_name"] = feat.station_name
    d["relay_model"]  = feat.relay_model
    d["voltage_kv"]   = feat.voltage_kv

    d["protection_type"]  = prot.primary_protection.name
    d["zone_operated"]    = ""
    d["trip_type"]        = prot.trip_type
    d["faulted_phases"]   = "+".join(feat.faulted_phases) if feat.faulted_phases else ""
    d["fault_type"]       = feat.fault_type
    d["is_ground_fault"]  = feat.is_ground_fault

    d["reclose_attempted"]  = feat.reclose_attempted
    d["reclose_successful"] = feat.reclose_successful
    d["reclose_time_ms"]    = None
    d["fault_count"]        = len(getattr(fault, "reclose_events", []) or []) + 1

    d["fault_duration_ms"]  = fault.duration_ms
    d["fault_inception_ms"] = round(fault.inception_time * 1000, 2)

    d["di_dt_max"]               = feat.di_dt_max
    d["di_dt_phase"]             = feat.di_dt_phase
    d["peak_fault_current_a"]    = feat.peak_fault_current_a
    d["peak_fault_phase"]        = ""
    d["i0_i1_ratio"]             = feat.i0_i1_ratio
    d["thd_percent"]             = feat.thd_percent
    d["inception_angle_degrees"] = feat.inception_angle_degrees
    d["voltage_sag_depth_pu"]    = None
    d["voltage_sag_phase"]       = ""
    d["voltage_phase_ratio_spread_pu"] = None
    d["healthy_phase_voltage_ratio"]   = None
    d["v2_v1_ratio"]                   = None
    d["voltage_thd_max_percent"]       = None
    d["v_prefault_v"]                  = None
    d["v_fault_v"]                     = None

    d["r_x_ratio"]        = None
    d["z_magnitude_ohms"] = None
    d["z_angle_degrees"]  = None

    d["sampling_rate_hz"]   = feat.sampling_rate_hz
    d["record_duration_ms"] = ((record.time[-1] - record.time[0]) * 1000.0) if getattr(record, "time", None) else None
    d["teleprotection_rx"]  = prot.permission_received
    d["comms_failure"]      = prot.comms_failure

    d["scaling_ok"]  = feat.peak_fault_current_a >= 200.0
    d["duration_ok"] = fault.duration_ms >= 5.0

    d["idiff_max_percent"]      = feat.idiff_max_percent
    d["irestraint_max_percent"] = feat.irestraint_max_percent
    d["idiff_rise_rate"]        = feat.idiff_rise_rate
    d["classification_status"]  = feat.classification_status

    return d


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract any zip/rar archives found in raw_data/ before scanning
    extract_archives(RAW_DATA)

    cfg_files = list(find_labeled_cfgs(RAW_DATA))
    print(f"Found {len(cfg_files)} labeled CFG files")

    from collections import Counter
    label_counts = Counter(label for _, label in cfg_files)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print()

    rows = []
    rows_87l = []
    errors = []

    for i, (cfg_path, label) in enumerate(cfg_files):
        short = str(cfg_path).replace(str(RAW_DATA), "")
        print(f"[{i+1:3d}/{len(cfg_files)}] {label:<12} {short[-70:]}", end="  ")

        try:
            record = parse_comtrade(str(cfg_path))
            if record is None:
                print("SKIP (parse failed)")
                errors.append({"cfg": str(cfg_path), "label": label, "reason": "parse returned None"})
                continue

            prot  = determine_protection(record)
            fault = detect_fault(record)

            if fault is None:
                print("SKIP (no fault detected)")
                errors.append({"cfg": str(cfg_path), "label": label, "reason": "no fault detected"})
                continue

            if prot.primary_protection.name == "DISTANCE":
                feat = extract_distance_features(record, fault, prot)
                if feat is None:
                    print("SKIP (distance feature extraction failed)")
                    errors.append({"cfg": str(cfg_path), "label": label, "reason": "extract_distance_features returned None"})
                    continue

                row = flatten_features(feat, label, cfg_path, prot, fault)
                rows.append(row)
                print(f"OK  dist z={row['zone_operated']} ph={row['faulted_phases']} "
                      f"dur={row['fault_duration_ms']:.0f}ms "
                      f"ar={row['reclose_successful']}")
                continue

            if prot.primary_protection.name == "DIFFERENTIAL":
                feat = extract_differential_features(record, fault, prot)
                if feat is None:
                    print("SKIP (87L feature extraction failed)")
                    errors.append({"cfg": str(cfg_path), "label": label, "reason": "extract_differential_features returned None"})
                    continue

                row = flatten_differential_features(feat, label, cfg_path, prot, fault, record)
                rows_87l.append(row)
                print(f"OK  87L ph={row['faulted_phases']} dur={row['fault_duration_ms']:.0f}ms "
                      f"ar={row['reclose_successful']}")
                continue

            print(f"SKIP (prot={prot.primary_protection.name})")
            errors.append({"cfg": str(cfg_path), "label": label,
                            "reason": f"protection={prot.primary_protection.name}"})
            continue

        except Exception as e:
            msg = str(e)[:120]
            print(f"ERROR: {msg}")
            errors.append({"cfg": str(cfg_path), "label": label, "reason": msg,
                           "traceback": traceback.format_exc()[-300:]})

    # Write features CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    if rows_87l:
        fieldnames_87l = list(rows_87l[0].keys())
        with open(OUT_CSV_87L, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_87l)
            writer.writeheader()
            writer.writerows(rows_87l)
        print(f"Wrote {len(rows_87l)} rows -> {OUT_CSV_87L}")

    # Write errors CSV
    if errors:
        with open(ERR_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cfg", "label", "reason", "traceback"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(errors)
        print(f"Wrote {len(errors)} errors -> {ERR_CSV}")

    # Summary by label
    from collections import Counter
    success_labels = Counter(r["label"] for r in rows)
    success_labels_87l = Counter(r["label"] for r in rows_87l)
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"{'Label':<15} {'DIST':>8} {'87L':>8} {'Input':>8} {'Rate':>8}")
    print("-" * 57)
    for label in sorted(label_counts):
        s = success_labels.get(label, 0)
        s87 = success_labels_87l.get(label, 0)
        t = label_counts[label]
        print(f"{label:<15} {s:>8} {s87:>8} {t:>8} {(s + s87)/t*100:>7.0f}%")
    print(f"\nTotal: {len(rows)} distance rows + {len(rows_87l)} 87L rows / {len(cfg_files)} attempted")


if __name__ == "__main__":
    main()
