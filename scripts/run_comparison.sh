#!/usr/bin/env bash
#
# Cron wrapper for the comparator CLI.
#
# Cron runs with a minimal environment and $HOME as the working directory, so
# this script makes everything explicit: it pins UTC, cd's into the project,
# activates the Python environment, then execs `compare run` passing through
# whatever arguments the crontab line supplied.
#
# Usage (from crontab):
#   /path/to/scripts/run_comparison.sh --model hrrr --var TMP --verif rtma
#
# Edit PROJECT_DIR (and the activation block) once for your server.

set -euo pipefail
export TZ=UTC

# Herbie lets HERBIE_SAVE_DIR override the save_dir the app passes in, which
# would silently redirect downloads away from your configured data_dir. Clear
# it so the data_dir from config.yaml / --data-dir is authoritative.
unset HERBIE_SAVE_DIR || true

# ---------------------------------------------------------------------------
# 1. Project location — EDIT THIS to the absolute path on your server.
# ---------------------------------------------------------------------------
PROJECT_DIR="${DAG_PROJECT_DIR:-$HOME/DAG_ModelComparison}"
cd "$PROJECT_DIR"

# Point the CLI at the project's config.yaml explicitly, so resolution does not
# depend on the working directory (the cd above already lands us here, but this
# makes it robust if someone overrides PROJECT_DIR or runs from elsewhere).
export DAG_CONFIG="${DAG_CONFIG:-$PROJECT_DIR/config.yaml}"

# ---------------------------------------------------------------------------
# 2. Activate the Python environment — pick ONE block for your server.
# ---------------------------------------------------------------------------

# --- (A) conda / miniconda (default; recommended for the esmf/xesmf stack) ---
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${DAG_CONDA_ENV:-new_comparator}"

# --- (B) HPC environment modules + venv (uncomment to use) -------------------
# module load python/3.11
# source "$PROJECT_DIR/.venv/bin/activate"

# --- (C) plain venv (uncomment to use) ---------------------------------------
# NOTE: esmf/xesmf are very painful to install via pip — prefer conda (A).
# source "$PROJECT_DIR/.venv/bin/activate"

# ---------------------------------------------------------------------------
# 3. Run. All CLI args are passed straight through from the crontab line.
# ---------------------------------------------------------------------------
# Invoke via `python -m` rather than the `compare` console script: it works
# whenever the conda env is active, even if `pip install -e .` has not been run.
exec python -m comparator.cli run "$@"
