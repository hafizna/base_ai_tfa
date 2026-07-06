#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TRAINING_DATA_DIR:-"${ROOT_DIR}/training-data"}"
OUT_DIR="${1:-"${ROOT_DIR}/training-exports"}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/base-ai-tfa-training-data-${STAMP}.zip"

mkdir -p "${OUT_DIR}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "Training data directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

python3 - "${DATA_DIR}" "${OUT_FILE}" <<'PY'
from pathlib import Path
import sys
import zipfile

data_dir = Path(sys.argv[1]).resolve()
out_file = Path(sys.argv[2]).resolve()

with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in data_dir.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(data_dir))

print(out_file)
PY
