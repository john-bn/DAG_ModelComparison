#!/bin/bash
#
# Manual restart after deploying new code. Cron only covers reboot/crash
# recovery (see run_dag_server.sh / watchdog_dag_server.sh) — it won't notice
# "I pushed a new revision", so run this by hand after `git pull`.

set -euo pipefail

PIDFILE="$HOME/dag/dag-server.pid"

if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    sleep 1
fi
"$(dirname "$0")/run_dag_server.sh"
