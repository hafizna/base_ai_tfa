#!/bin/sh
# Auto-deploy for the production EC2 host: when origin/main has moved, pull it
# and rebuild the container. Runs from cron on the server every 2 minutes (see
# scripts/install_auto_deploy.sh), so a merge to main goes live without anyone
# SSHing in. Pull-based on purpose: no SSH key stored in GitHub and no inbound
# access needed.
#
#   FORCE=1 sh scripts/auto_deploy.sh   # rebuild even when already up to date
#
# Everything lives in main() so the shell has parsed the whole file before the
# fast-forward below can rewrite it.
set -eu

main() {
  cd "$(dirname "$0")/.."

  # One deploy at a time; deploy.sh takes the same lock.
  exec 9>/tmp/base_ai_tfa_deploy.lock
  if ! flock -n 9; then
    exit 0
  fi

  git fetch --quiet origin main
  current=$(git rev-parse HEAD)
  target=$(git rev-parse origin/main)
  if [ "$current" = "$target" ] && [ "${FORCE:-0}" != "1" ]; then
    exit 0
  fi

  echo "$(date -Is) deploying $(git log -1 --format='%h %s' "$target") (was $(git rev-parse --short "$current"))"
  git merge --ff-only --quiet "$target"
  docker compose -f docker-compose.prod.yml up -d --build
  docker image prune -f >/dev/null || true
  # Keep a week of build cache so routine rebuilds stay fast; drop the rest
  # (it had grown to 3.5 GB on the 18 GB disk).
  docker builder prune -f --filter until=168h >/dev/null || true

  for _ in $(seq 1 24); do
    if curl -sf http://127.0.0.1:8000/api/health >/dev/null; then
      echo "$(date -Is) healthy"
      return 0
    fi
    sleep 5
  done
  echo "$(date -Is) WARNING: /api/health did not answer within 2 minutes"
  return 1
}

main "$@"
exit
