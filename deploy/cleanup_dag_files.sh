#!/bin/bash
#
# Delete aged-out working files for the comparator, so the server's disk does
# not grow without bound. Meant to be run by cron (see crontab.example). Pure
# `find` — deliberately NO micromamba/conda activation, so disk cleanup keeps
# working even if the Python env is broken (a broken env is exactly when the
# disk is most likely filling up). See docs/DEPLOYMENT.md.
#
# What it deletes (older than RETENTION_HOURS, by modification time):
#   in DATA_DIR : *.grib2, *.grib and their *.idx index files (Herbie downloads),
#                 then any now-empty <model>/<date>/ subdirectories.
#   in OUT_DIR  : *.png and *.gif figures.
#
# What it deliberately KEEPS (never matched, regardless of age):
#   DATA_DIR/weights_*_knn.npz  — the regridder weight cache (expensive to
#                                 rebuild, reused across every run).
#   OUT_DIR/runs.jsonl          — the run manifest that `compare list` reads.
#   LOG_DIR/*                   — logs are not touched here.
#
# Usage:
#   cleanup_dag_files.sh [HOURS] [--dry-run]
#     HOURS       retention window in hours (default 24). Also settable via the
#                 DAG_RETENTION_HOURS env var; the positional arg wins.
#     --dry-run   list what WOULD be deleted, delete nothing. Recommended for
#                 the first manual run.

set -euo pipefail

# --- EDIT THESE for your server (or set the matching DAG_* env vars) --------
# These MUST point at the same directories as your config.yaml (data_dir /
# out_dir). The defaults match the layout in docs/DEPLOYMENT.md §2. The DAG_*
# environment variables the rest of the app honors take precedence, so if you
# already export them you can leave this block on its defaults.
DATA_DIR="${DAG_DATA_DIR:-$HOME/dag/data}"
OUT_DIR="${DAG_OUT_DIR:-$HOME/dag/figures}"
LOG_DIR="${DAG_LOG_DIR:-$HOME/dag/logs}"
RETENTION_HOURS="${DAG_RETENTION_HOURS:-24}"
# ---------------------------------------------------------------------------

# --- Parse args (positional HOURS and/or --dry-run, in any order) ----------
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        ''|*[!0-9]*)
            echo "cleanup_dag_files.sh: unrecognized argument '$arg'" >&2
            echo "usage: cleanup_dag_files.sh [HOURS] [--dry-run]" >&2
            exit 2
            ;;
        *) RETENTION_HOURS="$arg" ;;
    esac
done

if ! [[ "$RETENTION_HOURS" =~ ^[0-9]+$ ]] || [[ "$RETENTION_HOURS" -lt 1 ]]; then
    echo "cleanup_dag_files.sh: RETENTION_HOURS must be a positive integer (got '$RETENTION_HOURS')" >&2
    exit 2
fi
MINS=$(( RETENTION_HOURS * 60 ))

# --- Safety: refuse to operate on obviously-wrong roots --------------------
# Guards against an empty/misconfigured variable turning `find -delete` loose on
# the whole home directory or filesystem root.
guard_dir() {
    local d="$1" label="$2"
    if [[ -z "$d" ]]; then
        echo "cleanup_dag_files.sh: $label is empty — refusing to run." >&2
        exit 3
    fi
    # Resolve to an absolute, symlink-free path for the comparison.
    local resolved
    resolved="$(cd "$d" 2>/dev/null && pwd -P || true)"
    if [[ -n "$resolved" ]]; then
        case "$resolved" in
            / | /root | "$HOME" | "$(cd "$HOME" && pwd -P)")
                echo "cleanup_dag_files.sh: $label resolves to '$resolved' — too broad, refusing." >&2
                exit 3
                ;;
        esac
    fi
}
guard_dir "$DATA_DIR" DATA_DIR
guard_dir "$OUT_DIR" OUT_DIR

# --- Logging ---------------------------------------------------------------
# Append to a log if LOG_DIR is writable; always echo to stdout too so cron can
# mail the output. `tee -a` to a best-effort log file.
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOGFILE="$LOG_DIR/cleanup.log"
log() {
    local line
    line="$(date -u '+%Y-%m-%dT%H:%M:%SZ') cleanup: $*"
    if [[ -w "$LOG_DIR" || -w "$LOGFILE" ]]; then
        echo "$line" | tee -a "$LOGFILE"
    else
        echo "$line"
    fi
}

action=$([[ "$DRY_RUN" -eq 1 ]] && echo "DRY-RUN (nothing deleted)" || echo "deleting")
log "start: $action files older than ${RETENTION_HOURS}h (>${MINS} min); DATA_DIR=$DATA_DIR OUT_DIR=$OUT_DIR"

# Run one find, printing every match; delete unless dry-run. Returns the count.
# Args: <dir> <human-label> <find-name-predicates...>
sweep_files() {
    local dir="$1" label="$2"; shift 2
    [[ -d "$dir" ]] || { log "skip: $label dir does not exist ($dir)"; return 0; }

    local -a name_args=()
    local first=1
    for pat in "$@"; do
        if [[ $first -eq 1 ]]; then first=0; else name_args+=(-o); fi
        name_args+=(-name "$pat")
    done

    local -a find_cmd=(find "$dir" -type f \( "${name_args[@]}" \) -mmin +"$MINS" -print)
    [[ "$DRY_RUN" -eq 1 ]] || find_cmd+=(-delete)

    local out count
    out="$("${find_cmd[@]}")" || true
    count="$([[ -z "$out" ]] && echo 0 || printf '%s\n' "$out" | wc -l | tr -d ' ')"
    log "$label: ${count} file(s) matched (${*})"
    [[ -n "$out" ]] && printf '%s\n' "$out" | sed 's/^/  - /' | tee -a "$LOGFILE" >/dev/null 2>&1 || true
    return 0
}

# GRIB2 downloads + their sidecar index files (spares weights_*_knn.npz).
sweep_files "$DATA_DIR" "GRIB/index" '*.grib2' '*.grib' '*.idx'

# Generated figures (spares runs.jsonl).
sweep_files "$OUT_DIR" "figures" '*.png' '*.gif'

# Prune Herbie's now-empty <model>/<date>/ subdirectories left behind by the
# GRIB sweep. -mindepth 1 protects DATA_DIR itself; -empty means a dir still
# holding weights_*_knn.npz is never removed.
if [[ -d "$DATA_DIR" && "$DRY_RUN" -eq 0 ]]; then
    find "$DATA_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
fi

log "done"
