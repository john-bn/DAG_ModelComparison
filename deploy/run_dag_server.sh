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
# ~/.bashrc, so the PATH entry an interactive login gets (~/.pixi/bin) is NOT
# available here — spell the binary out absolutely. Find it from a normal shell
# with `which pixi`; list the environment names with `pixi info`.
PIXI="$HOME/.pixi/bin/pixi"   # absolute path to the pixi binary
PIXI_ENV="default"            # from [tool.pixi.environments] in pyproject.toml
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

# Activate the env the pixi way. `pixi shell-hook` prints the same activation
# script `pixi shell` would source, including the env's etc/conda/activate.d
# hooks. The one that matters here is proj4-activate.sh, which points PROJ_DATA
# at the env's own share/proj and sets PROJ_NETWORK=OFF — pyproj and cartopy
# resolve projections through that, and it also overrides any stale PROJ_DATA
# inherited from another install. We eval the hook rather than wrapping the server in
# `pixi run` because `pixi run` would sit between this script and python: $!
# below must be the server's own PID for the pidfile / watchdog / restart logic
# to stay exact. `--frozen` installs strictly from pixi.lock, so a boot-time
# start never blocks on a network re-solve. `set +u` guards the hook, which may
# reference unset shell vars (e.g. PS1); it's restored right after.
# We cd first so pixi discovers the manifest in the repo root.
cd "$REPO_DIR"
set +u
eval "$("$PIXI" shell-hook --frozen -e "$PIXI_ENV")"
set -u

nohup python -m comparator.webserver serve --host 127.0.0.1 --port "$PORT" \
    >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
