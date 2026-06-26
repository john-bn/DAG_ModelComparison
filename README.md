# DAG_ModelComparison

This program uses Herbie & xESMF to analyze temperature, dewpoint, visibility, 10 m wind speed, & wind gust fields from the HRRR, NAM, NBM, GFS, & other models run at NCEP, verified against either the RTMA or URMA analysis. MatPlotLib & Cartopy are then used to plot the data in a map. Wind speed is derived from the model's U/V components when no direct wind-speed field is published.

The driver file is new_comparison.py. The program is run by following;

Select a model prompted by the command line output, (i.e, HRRR)
Then, a valid field to analyze. (i.e, temperature)
Then, the verification analysis source, (i.e, RTMA or URMA)
Then, a valid initiliazation hour, (i.e, 00)
Then, a valid forecast hour, (i.e, 24)

As the data is downloaded from NOMADS & AWS, no special permissions are required.
Data are downloaded automatically via Herbie and cached locally in ./data/.
For the environemnt, I recommend: conda env create -f environment.yml
This program is built for Python 3.11 (see `environment.yml`).

## Command-line interface (`compare`)

For unattended (cron) and scripted use there is a non-interactive CLI built on
the same engine. After installing the package (`pip install -e .` inside the
conda env) the `compare` command is on your `$PATH` (equivalently
`python -m comparator.cli`):

```
# Generate one comparison. With no --date it runs in ROLLING REAL-TIME mode:
# it verifies the most recent analysis (now - lag) using the forecast lead
# closest to --lead. Prints the saved PNG path on success.
compare run --model hrrr --var TMP --verif rtma

# See exactly what a scheduled run would target, without fetching anything:
compare run --model hrrr --var TMP --verif rtma --dry-run

# Pin a specific case (backfill). single mode needs --date --init --fxx:
compare run --model hrrr --var WIND --date 2026-06-18 --init 12 --fxx 24

# Animated GIF over every forecast covering an analysis time:
compare run --model hrrr --var TMP --mode gif --date 2026-06-18 --init 12

# Browse what's been produced (reads the runs.jsonl manifest in out_dir):
compare list --model hrrr --since 2026-06-18
compare latest --model hrrr --var TMP        # prints newest matching path
```

Variables: `TMP`, `DPT`, `VIS`, `WIND`, `GUST`. Models: `hrrr`, `nam5k`,
`nam12k`, `rap`, `nbm`, `arw`, `fv3`, `href`, `gfs`, `ifs`.

### Configuration

Copy `config.example.yaml` to `config.yaml` and set **absolute** `data_dir`,
`out_dir`, and `log_dir` (cron's working directory is `$HOME`, so relative
paths land in the wrong place). Settings resolve in the order
**CLI flag > `DAG_*` env var > `config.yaml` > built-in default**. Each `run`
appends a record (incl. mean / RMSE error stats) to `<out_dir>/runs.jsonl` and
logs to `<log_dir>/comparison.log`.

## Deploying on a Linux server with cron

The cron wrapper (`scripts/run_comparison.sh`) defaults to a project at
`$HOME/DAG_ModelComparison` and a conda install at `$HOME/miniconda3` with the
`new_comparator` env, so if you follow that layout it needs no edits.

1. Put the repo at `~/DAG_ModelComparison` on the server.
2. Install Miniconda in your home directory (no root needed) if it isn't there:
   `bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3`
3. Create the environment and install the CLI:
   `conda env create -f environment.yml` then
   `conda activate new_comparator && pip install -e .`
4. `cp config.example.yaml config.yaml` and set the absolute `data_dir`,
   `out_dir`, `log_dir`; create them with `mkdir -p`. `config.yaml` is
   gitignored — each machine keeps its own.
5. Sanity-check end to end:
   `compare run --model hrrr --var TMP --verif rtma --dry-run`
6. (Only if your layout differs) edit `scripts/run_comparison.sh` — set
   `PROJECT_DIR`/`CONDA_BASE` or pick the module-load / venv activation block.
   The wrapper runs the CLI via `python -m comparator.cli`, so it works as long
   as the env activates (even before `pip install -e .`).
7. Schedule it: `crontab scripts/crontab.example` (edit the paths first), or
   merge its lines into `crontab -e`. One crontab line = one model+variable; the
   example schedules HRRR / NAM12K / GFS / NBM × TMP / DPT / WIND / GUST hourly
   (staggered, each to its own log) plus a daily retrospective GIF. Validate any
   new model+variable with a single `compare run` first — not every model
   publishes every field.

"No data yet" (the target hour is too fresh) is logged and exits 0 so cron does
not email an alarm; genuine failures exit non-zero.

The scheduled CLI and the manual entry points share the same engine, so running
by hand still works exactly as before — the interactive `python new_comparison.py`
and the non-interactive `compare run | list | latest` (see above).
