#!/bin/sh
# Deploy the current main branch to the production EC2 host.
#
# Run this FROM YOUR LOCAL MACHINE (laptop). It SSHes into EC2 and does the
# git pull + docker rebuild there — `docker compose up --build` only ever
# affects the machine it runs on, so it must execute on the server, not here.
#
# Usage:
#   ./deploy.sh
#   EC2_KEY=/path/to/other-key.pem ./deploy.sh
set -eu

EC2_HOST="${EC2_HOST:-ubuntu@16.176.50.252}"
EC2_KEY="${EC2_KEY:-C:\Users\hafizna.fadhli\Downloads\TFA\base-ai-tfa-key.converted.pem}"
REMOTE_DIR="${REMOTE_DIR:-base_ai_tfa}"

echo "==> Deploying to ${EC2_HOST}:${REMOTE_DIR}"

ssh -i "${EC2_KEY}" "${EC2_HOST}" "set -eu
  cd '${REMOTE_DIR}'
  # Same lock as scripts/auto_deploy.sh, so a cron deploy and this one never build at once.
  exec 9>/tmp/base_ai_tfa_deploy.lock
  echo '--- waiting for any running auto-deploy ---'
  flock 9
  echo '--- git pull ---'
  git fetch origin
  git status --short --branch
  git pull origin main
  echo '--- docker compose build + up ---'
  docker compose -f docker-compose.prod.yml up -d --build
  echo '--- waiting for healthcheck ---'
  sleep 5
  docker compose -f docker-compose.prod.yml ps
  echo '--- health endpoint ---'
  curl -sf http://127.0.0.1:8000/api/health && echo '' && echo 'OK: service healthy'
"

echo "==> Deploy finished."
