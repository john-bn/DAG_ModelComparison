#!/bin/bash
#
# Cron'd every few minutes: restarts the comparator web daemon if it stops
# answering. Substitutes for systemd's Restart=on-failure — this host has no
# usable `systemctl --user` session (no D-Bus user session, no lingering) and
# no sudo for a system-level unit. See docs/DEPLOYMENT.md.

set -euo pipefail

PORT=8000   # keep in sync with run_dag_server.sh

if ! curl -sf -o /dev/null "http://127.0.0.1:${PORT}/"; then
    "$(dirname "$0")/run_dag_server.sh"
fi
