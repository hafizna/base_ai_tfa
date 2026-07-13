#!/bin/sh
set -eu

# WEB_CONCURRENCY controls the worker count for the Amazon EC2/Docker Compose
# deployment. Each worker is a separate Python process: it must load the model
# bundle once at boot, so memory cost is roughly N x ~180 MB. Override upward
# only when the EC2 instance has enough RAM headroom and concurrent traffic.
PORT_VALUE="${PORT:-8000}"
WORKERS_VALUE="${WEB_CONCURRENCY:-2}"

exec uvicorn webapp.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT_VALUE}" \
  --workers "${WORKERS_VALUE}" \
  --timeout-keep-alive 120 \
  --access-log
