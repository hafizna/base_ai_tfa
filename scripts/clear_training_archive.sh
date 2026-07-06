#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "--yes" ]; then
  echo "This deletes retained raw uploads and feedback from training-data/." >&2
  echo "Run again with: scripts/clear_training_archive.sh --yes" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TRAINING_DATA_DIR:-"${ROOT_DIR}/training-data"}"

python3 - "${DATA_DIR}" <<'PY'
from pathlib import Path
import shutil
import sys

data_dir = Path(sys.argv[1]).resolve()
raw_dir = data_dir / "raw"
labels_dir = data_dir / "labels"
exports_dir = data_dir / "exports"

removed_raw = 0
if raw_dir.exists():
    for child in raw_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            removed_raw += 1
        elif child.is_file():
            child.unlink()

removed_labels = 0
if labels_dir.exists():
    for child in labels_dir.iterdir():
        if child.is_file():
            child.unlink()
            removed_labels += 1

if exports_dir.exists():
    for child in exports_dir.iterdir():
        if child.is_file():
            child.unlink()

print(f"Removed {removed_raw} raw records and {removed_labels} label files from {data_dir}")
PY
