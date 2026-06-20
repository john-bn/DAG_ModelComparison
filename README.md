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

1. Copy the project to the server and create the environment:
   `conda env create -f environment.yml`
2. Install the CLI: `conda activate new_comparator && pip install -e .`
3. `cp config.example.yaml config.yaml` and edit the absolute paths.
4. Sanity-check end to end:
   `compare run --model hrrr --var TMP --verif rtma --dry-run`
5. Edit `scripts/run_comparison.sh` — set `PROJECT_DIR` and pick the env
   activation block (conda by default; module-load / venv variants included).
6. Schedule it: `crontab scripts/crontab.example` (edit the paths first), or
   merge its lines into `crontab -e`. One crontab line = one model+variable.
   The example staggers start minutes and redirects each job to its own log.

"No data yet" (the target hour is too fresh) is logged and exits 0 so cron does
not email an alarm; genuine failures exit non-zero.
