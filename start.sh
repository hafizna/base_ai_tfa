#!/bin/sh
set -eu

PORT_VALUE="${PORT:-8000}"

exec uvicorn webapp.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT_VALUE}" \
  --workers 1 \
  --timeout-keep-alive 120
