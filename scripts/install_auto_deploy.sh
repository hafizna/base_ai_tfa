#!/bin/sh
# Install the cron job that runs scripts/auto_deploy.sh every 2 minutes. Run it
# once on the EC2 host; running it again keeps a single entry.
#
#   sh scripts/install_auto_deploy.sh            # install
#   sh scripts/install_auto_deploy.sh --remove   # uninstall
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
log="$HOME/auto_deploy.log"
entry="*/2 * * * * /bin/sh $repo/scripts/auto_deploy.sh >> $log 2>&1"

others=$(crontab -l 2>/dev/null | grep -v 'scripts/auto_deploy.sh' || true)
if [ "${1:-}" = "--remove" ]; then
  printf '%s\n' "$others" | crontab -
  echo "Auto-deploy removed from crontab."
else
  printf '%s\n%s\n' "$others" "$entry" | crontab -
  echo "Auto-deploy installed: origin/main is checked every 2 minutes."
  echo "Log: $log"
fi
