#!/bin/bash
#
# Manual restart after deploying new code. Cron only covers reboot/crash
# recovery (see run_dag_server.sh / watchdog_dag_server.sh) — it won't notice
# "I pushed a new revision", so run this by hand after `git pull`.

set -euo pipefail

PIDFILE="$HOME/dag/dag-server.pid"
PIXI="$HOME/.pixi/bin/pixi"   # keep in sync with run_dag_server.sh
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Reconcile the env with the (possibly just-pulled) pixi.toml / pixi.lock before
# restarting. This is the deploy-time path, where network is available —
# run_dag_server.sh itself stays `--frozen` so a boot-time start never needs it.
# `--locked` fails loudly if pixi.lock is out of date with pixi.toml rather than
# silently re-solving, so the server can't drift from what was committed.
"$PIXI" install --manifest-path "$REPO_DIR/pixi.toml" --locked

if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    sleep 1
fi
"$(dirname "$0")/run_dag_server.sh"
