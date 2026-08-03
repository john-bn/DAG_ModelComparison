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
# micromamba env activation. Cron runs with a minimal environment and does NOT
# source ~/.bashrc, so the `micromamba` shell function and MAMBA_ROOT_PREFIX
# that an interactive login sets up are NOT available here — we recreate them
# explicitly below. Fill these in from a normal shell with:
#   which micromamba ; echo "$MAMBA_ROOT_PREFIX" ; micromamba env list
MICROMAMBA="$HOME/.local/bin/micromamba"     # absolute path to the binary
export MAMBA_ROOT_PREFIX="$HOME/micromamba"  # root prefix that holds your envs
ENV_NAME="rtc"                               # the env's name (from `env list`)
DAG_CONFIG="$REPO_DIR/config.yaml"
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

# Activate the env the micromamba way. This runs the env's activate.d hooks
# (they set GDAL_DATA/PROJ_LIB, which cartopy/pyproj need) — bypassing them by
# calling the env's python directly would break projection lookups. We init the
# shell hook from the absolute binary so it works identically under cron, and
# activate rather than `micromamba run` so the backgrounded python stays the
# direct child: $! below is then the server's own PID, keeping the pidfile /
# watchdog / restart logic exact. `set +u` guards the hook, which may reference
# unset shell vars (e.g. PS1); it's restored right after.
set +u
eval "$("$MICROMAMBA" shell hook -s bash)"
micromamba activate "$ENV_NAME"
set -u
cd "$REPO_DIR"

nohup python -m comparator.webserver serve --host 127.0.0.1 --port "$PORT" \
    >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
