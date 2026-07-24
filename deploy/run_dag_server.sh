#!/bin/bash
#
# Idempotent start script for the comparator web daemon (`compare-web serve`).
#
# Run by cron (@reboot) and by watchdog_dag_server.sh — safe to call
# repeatedly, it's a no-op if the daemon is already running. See
# docs/DEPLOYMENT.md for the full deployment guide.

set -euo pipefail

# --- EDIT THESE for your server --------------------------------------------
REPO_DIR="/home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute"
ENV_ACTIVATE="$HOME/new_comparator/bin/activate"   # the conda-pack'd env
DAG_CONFIG="$HOME/dag/config.yaml"
PORT=8000
# ---------------------------------------------------------------------------

STATE_DIR="$HOME/dag"
PIDFILE="$STATE_DIR/dag-server.pid"
LOGFILE="$STATE_DIR/logs/dag-server.log"
mkdir -p "$STATE_DIR/logs"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null \
   && ps -p "$(cat "$PIDFILE")" -o args= | grep -q "comparator.webserver serve"; then
    exit 0   # already running
fi
rm -f "$PIDFILE"

export TZ=UTC
export MPLBACKEND=Agg
unset HERBIE_SAVE_DIR || true
export DAG_CONFIG

source "$ENV_ACTIVATE"
cd "$REPO_DIR"

nohup python -m comparator.webserver serve --host 127.0.0.1 --port "$PORT" \
    >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
