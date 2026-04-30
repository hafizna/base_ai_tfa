"""
Batch Prediction — Semua File COMTRADE di raw_data/
====================================================
Menjalankan classifier pada SETIAP file CFG yang ditemukan,
berlabel maupun tidak, dan menghasilkan dua CSV:

  data/predictions/all_predictions.csv   — semua hasil klasifikasi
  data/predictions/prediction_errors.csv — file yang gagal + alasannya

Gunakan all_predictions.csv untuk crosscheck dengan stakeholder.
Kolom 'folder_label' adalah label dari nama folder (bisa kosong jika tidak berlabel).
Kolom 'status_data' adalah grouping dugaan berbasis path.
Kolom 'suspected_label' adalah label dugaan yang tidak bersifat fixed.
Kolom 'predicted_label' adalah hasil model.
Kolom 'correct' diisi kosong — untuk diisi stakeholder.

Jalankan dari folder pipeline/:
    python batch_predict.py
"""

import sys
import re
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from models.predict import classify_file
from core.path_heuristics import infer_path_tag, infer_status_data, infer_suspected_label

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA  = Path(__file__).parent.parent / "raw_data"
OUT_DIR   = Path(__file__).parent / "data" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = OUT_DIR / "all_predictions.csv"
ERRORS_CSV      = OUT_DIR / "prediction_errors.csv"

# Fragments that mark a sub-folder as non-primary (analysis/locus output)
SKIP_FRAGS = ["olah", "_extracted", "locus", "analisa"]

# ── Label inference from folder path ──────────────────────────────────────────
LABEL_MAP = [
    ("petir",        "PETIR"),
    ("layang",       "LAYANG-LAYANG"),
    ("pohon",        "POHON"),
    ("tower roboh",  "KONDUKTOR"),
    ("konduktor",    "KONDUKTOR"),
    ("alat isolator", "PERALATAN"),
    ("isolator",     "PERALATAN"),
    ("kerusakan peralatan", "PERALATAN"),
    ("gangguan peralatan", "PERALATAN"),
    ("pilot wire",   "PERALATAN"),
    ("pilotwire",    "PERALATAN"),
    ("teleprotection", "PERALATAN"),
    ("teleproteksi", "PERALATAN"),
    ("plcc",         "PERALATAN"),
    ("hewan",        "HEWAN"),
    ("ular",         "HEWAN"),
    ("babi",         "HEWAN"),
    ("kukang",       "HEWAN"),
    ("monyet",       "HEWAN"),
    ("benda asing",  "BENDA ASING"),
    ("bfo",          "BFO"),
    ("peralatan",    "PERALATAN"),
    ("lain",         "LAIN-LAIN"),
]

def _infer_folder_label(path_str: str) -> str:
    low = path_str.lower()
    for frag, lbl in LABEL_MAP:
        if frag in low:
            return lbl
    return ""   # unlabeled / unknown


def _path_meta(cfg: Path) -> dict:
    """Extract UPT, year, month, event folder from path."""
    parts = cfg.parts
    upt   = next((p for p in parts if p.startswith("UPT ")), "-")
    year  = next((p for p in parts if re.fullmatch(r"\d{4}", p)), "-")
    try:
        rel   = cfg.relative_to(RAW_DATA)
        month = rel.parts[2] if len(rel.parts) > 2 else "-"
        event = rel.parts[3] if len(rel.parts) > 3 else rel.parts[-1]
    except Exception:
        month = "-"
        event = "-"
    return {"upt": upt, "year": year, "month": month, "event": event}


# ── Scan all CFG files ────────────────────────────────────────────────────────
def scan_all_cfgs() -> list[Path]:
    seen = set()
    cfgs = []
    for cfg in sorted(RAW_DATA.rglob("*.cfg")) + sorted(RAW_DATA.rglob("*.CFG")):
        key = str(cfg.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        if any(f in str(cfg).lower() for f in SKIP_FRAGS):
            continue
        dat = cfg.with_suffix(".dat")
        if not dat.exists():
            dat = cfg.with_suffix(".DAT")
        if not dat.exists():
            continue   # no .dat pair → skip
        cfgs.append(cfg)
    return cfgs


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    if not RAW_DATA.exists():
        print(f"ERROR: {RAW_DATA} tidak ditemukan.")
        sys.exit(1)

    cfgs = scan_all_cfgs()
    print(f"Ditemukan {len(cfgs)} file CFG dengan pasangan .dat")
    print(f"Mulai prediksi...\n")

    results = []
    errors  = []
    t0 = datetime.now()

    for i, cfg in enumerate(cfgs, 1):
        meta         = _path_meta(cfg)
        folder_label = _infer_folder_label(str(cfg))
        path_str = str(cfg)
        status_data = infer_status_data(path_str)
        suspected_label = infer_suspected_label(path_str)
        path_tag = infer_path_tag(path_str)

        print(f"[{i:>3}/{len(cfgs)}] {cfg.name[:60]:<60}", end=" ", flush=True)

        try:
            r = classify_file(str(cfg))
            row = {
                "no":              i,
                "upt":             meta["upt"],
                "year":            meta["year"],
                "month":           meta["month"],
                "event":           meta["event"],
                "filename":        cfg.name,
                "folder_label":    folder_label,
                "predicted_label": r.label,
                "confidence":      round(r.confidence, 3),
                "tier":            r.tier,
                "rule_name":       r.rule_name,
                "evidence":        r.evidence,
                # Key features
                "station_name":    r.features.get("station_name", ""),
                "zone":            r.features.get("zone_operated", ""),
                "trip_type":       r.features.get("trip_type", ""),
                "faulted_phases":  r.features.get("faulted_phases", ""),
                "fault_duration_ms": r.features.get("fault_duration_ms", ""),
                "fault_count":     r.features.get("fault_count", ""),
                "peak_current_a":  r.features.get("peak_fault_current_a", ""),
                "i0_i1_ratio":     r.features.get("i0_i1_ratio", ""),
                "reclose_ok":      r.features.get("reclose_successful", ""),
                "voltage_sag_pu":  r.features.get("voltage_sag_depth_pu", ""),
                # For stakeholder crosscheck
                "correct":         "",   # leave blank — to be filled by stakeholder
                "notes":           "",
                "cfg_path":        str(cfg),
                "status_data":     status_data,
                "suspected_label": suspected_label,
                "path_tag":        path_tag,
            }
            results.append(row)
            match = ("OK" if folder_label and folder_label in r.label.upper()
                     else "??" if not folder_label
                     else "XX")
            print(f"-> {r.label:<40} [{match}]  status={status_data:<10} conf={r.confidence:.0%}")

        except ValueError as e:
            err_msg = str(e)
            errors.append({
                "no":           i,
                "upt":          meta["upt"],
                "year":         meta["year"],
                "month":        meta["month"],
                "event":        meta["event"],
                "filename":     cfg.name,
                "folder_label": folder_label,
                "error":        err_msg,
                "cfg_path":     str(cfg),
                "status_data":  status_data,
                "suspected_label": suspected_label,
                "path_tag":     path_tag,
            })
            # Shorten for console
            short = err_msg[:60] if len(err_msg) > 60 else err_msg
            print(f"-> SKIP: {short}  status={status_data}")

        except Exception as e:
            errors.append({
                "no":           i,
                "upt":          meta["upt"],
                "year":         meta["year"],
                "month":        meta["month"],
                "event":        meta["event"],
                "filename":     cfg.name,
                "folder_label": folder_label,
                "error":        f"ERROR: {e}",
                "cfg_path":     str(cfg),
                "status_data":  status_data,
                "suspected_label": suspected_label,
                "path_tag":     path_tag,
            })
            print(f"-> ERROR: {str(e)[:60]}  status={status_data}")

    elapsed = (datetime.now() - t0).total_seconds()

    # ── Save results ──────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_errors  = pd.DataFrame(errors)

    df_results.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    df_errors.to_csv(ERRORS_CSV,       index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Selesai dalam {elapsed:.0f} detik")
    print(f"Berhasil diklasifikasikan : {len(results)}")
    print(f"Gagal / tidak didukung   : {len(errors)}")
    print(f"\nHasil tersimpan di:")
    print(f"  {PREDICTIONS_CSV}")
    print(f"  {ERRORS_CSV}")

    if len(results) > 0:
        print(f"\nDistribusi prediksi:")
        vc = df_results["predicted_label"].value_counts()
        for label, count in vc.items():
            print(f"  {label:<45} {count:>4}")

        print(f"\nDistribusi status data:")
        status_vc = df_results["status_data"].value_counts(dropna=False)
        for status, count in status_vc.items():
            print(f"  {status:<20} {count:>4}")

        print(f"\nLabel dugaan teratas:")
        suspected_vc = df_results["suspected_label"].value_counts(dropna=False).head(10)
        for label, count in suspected_vc.items():
            print(f"  {label:<45} {count:>4}")

        # Accuracy on labeled files only
        labeled = df_results[df_results["folder_label"] != ""]
        if len(labeled) > 0:
            # Transient sub-classes (PETIR/LAYANG/HEWAN/BENDA ASING) all map to
            # GANGGUAN TRANSIEN at Tier 2 — count those as correct.
            TRANSIENT_LABELS = {"PETIR", "LAYANG-LAYANG", "HEWAN", "BENDA ASING", "LAIN-LAIN"}
            correct = labeled.apply(
                lambda r: (
                    r["folder_label"].upper() in r["predicted_label"].upper()
                    or (
                        "TRANSIEN" in r["predicted_label"]
                        and r["folder_label"].upper() in TRANSIENT_LABELS
                    )
                ),
                axis=1
            ).sum()
            print(f"\nPada {len(labeled)} file berlabel:")
            print(f"  Prediksi sesuai label folder : {correct} ({correct/len(labeled):.0%})")
            print(f"  Perlu dicek stakeholder       : {len(labeled) - correct}")
            print(f"  (Catatan: PETIR/LAYANG/HEWAN/BENDA ASING/LAIN-LAIN -> GANGGUAN TRANSIEN dihitung benar)")

    print(f"\nBuka {PREDICTIONS_CSV.name} di Excel untuk crosscheck dengan stakeholder.")
    print(f"Kolom 'correct' dan 'notes' dikosongkan — isi berdasarkan hasil lapangan.")


if __name__ == "__main__":
    run()
