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
# pixi env activation. Cron runs with a minimal environment and does NOT source
# ~/.bashrc, so the PATH entry an interactive login sets up is NOT available
# here — spell out the absolute binary path instead. Find it from a normal
# shell with:  which pixi
PIXI="$HOME/.pixi/bin/pixi"                  # absolute path to the binary
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

# Activate the env the pixi way. This runs the packages' activate.d hooks — in
# particular proj4-activate.sh, which sets PROJ_DATA for cartopy/pyproj; bypass
# them by calling the env's python directly and projection lookups break. The
# hook output is plain `export` lines plus a `.` of each activate.d script, so
# eval'ing it is the direct analog of the old micromamba hook. We eval the hook
# from the absolute binary so it works identically under cron, and activate
# rather than `pixi run` so the backgrounded python stays the direct child: $!
# below is then the server's own PID, keeping the pidfile / watchdog / restart
# logic exact. `--frozen` installs strictly from pixi.lock without re-solving,
# so a boot-time start needs no network. `set +u` guards the hook, which may
# reference unset shell vars (e.g. PS1); it's restored right after.
set +u
eval "$("$PIXI" shell-hook --manifest-path "$REPO_DIR/pixi.toml" \
                           --shell bash --frozen)"
set -u
cd "$REPO_DIR"

nohup python -m comparator.webserver serve --host 127.0.0.1 --port "$PORT" \
    >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
