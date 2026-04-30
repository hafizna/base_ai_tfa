"""
extract_all.py  —  One-shot archive extractor using 7-Zip
==========================================================
Extracts all .zip and .rar files under raw_data/ using 7-Zip.
Skips archives that already have a matching extracted folder beside them.

Requirements:
    7-Zip must be installed.  Download: https://www.7-zip.org/
    Default install path: C:\\Program Files\\7-Zip\\7z.exe

Usage (from the pipeline/ directory):
    python extract_all.py
    python extract_all.py --dry-run      # show what would be extracted, don't extract
    python extract_all.py --force        # re-extract even if folder already exists
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# ── locate 7z.exe ────────────────────────────────────────────────────────────
SEVENZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "7z",       # already on PATH
    "7z.exe",
]

def find_7z() -> str:
    for candidate in SEVENZIP_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "i"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, OSError):
            continue
    return None


RAW_DATA = Path(__file__).parent.parent / "raw_data"


def already_extracted(archive: Path) -> bool:
    """True if a .extracted marker or same-stem folder exists."""
    marker = archive.with_suffix(archive.suffix + ".extracted")
    if marker.exists():
        return True
    stem_folder = archive.with_suffix("")
    if stem_folder.is_dir():
        return True
    return False


def extract_with_7z(sevenzip: str, archive: Path, dry_run: bool) -> bool:
    """
    Extract archive to its parent directory using 7z.
    Returns True on success.
    """
    dest = archive.parent
    cmd = [
        sevenzip,
        "x",            # extract with full paths
        str(archive),
        f"-o{dest}",    # output directory
        "-y",           # yes to all prompts
        "-spe",         # no extra top-level folder
    ]

    if dry_run:
        print(f"  [DRY-RUN] would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Write marker so batch_extract.py skips this archive next run
            marker = archive.with_suffix(archive.suffix + ".extracted")
            marker.touch()
            return True
        else:
            # Print first few lines of 7z error output
            err_lines = (result.stderr or result.stdout or "").strip().splitlines()
            short_err = " | ".join(err_lines[:3])
            print(f"  [ERROR] 7z returned {result.returncode}: {short_err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] extraction took >120 s, skipped")
        return False
    except Exception as e:
        print(f"  [EXCEPTION] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract all archives under raw_data/ with 7-Zip")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted")
    parser.add_argument("--force",   action="store_true", help="Re-extract already-extracted archives")
    args = parser.parse_args()

    sevenzip = find_7z()
    if sevenzip is None:
        print("ERROR: 7-Zip not found.")
        print("  Download and install from: https://www.7-zip.org/")
        print("  Default install path: C:\\Program Files\\7-Zip\\7z.exe")
        sys.exit(1)

    print(f"Using 7-Zip: {sevenzip}\n")

    archives = sorted(RAW_DATA.rglob("*.zip")) \
             + sorted(RAW_DATA.rglob("*.ZIP")) \
             + sorted(RAW_DATA.rglob("*.rar")) \
             + sorted(RAW_DATA.rglob("*.RAR"))

    print(f"Found {len(archives)} archives under {RAW_DATA.name}/\n")

    ok = skipped = failed = 0
    for archive in archives:
        short = str(archive.relative_to(RAW_DATA))
        suffix = archive.suffix.lower()

        if not args.force and already_extracted(archive):
            skipped += 1
            continue

        label = suffix[1:].upper()
        print(f"  [{label}] {short[-80:]} ... ", end="", flush=True)

        success = extract_with_7z(sevenzip, archive, args.dry_run)
        if success:
            ok += 1
            print("OK" if not args.dry_run else "")
        else:
            failed += 1

    print(f"\n=== DONE ===")
    print(f"  Extracted : {ok}")
    print(f"  Skipped   : {skipped}  (already extracted)")
    print(f"  Failed    : {failed}")
    print()
    if failed == 0:
        print("All archives extracted. Run batch_extract.py to regenerate labeled_features.csv.")
    else:
        print(f"{failed} archive(s) failed — check errors above.")


if __name__ == "__main__":
    main()
