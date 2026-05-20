#!/bin/sh
set -eu

# WEB_CONCURRENCY follows the Heroku/Railway convention. Each worker is a
# separate Python process: it must pickle-load fault_classifier.pkl once at
# boot (the FastAPI lifespan in webapp/api/main.py does this eagerly), so
# memory cost is roughly N × ~180 MB. Default to 2 on free dynos, override
# upward only when you have RAM headroom and concurrent traffic.
PORT_VALUE="${PORT:-8000}"
WORKERS_VALUE="${WEB_CONCURRENCY:-2}"

exec uvicorn webapp.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT_VALUE}" \
  --workers "${WORKERS_VALUE}" \
  --timeout-keep-alive 120 \
  --access-log
