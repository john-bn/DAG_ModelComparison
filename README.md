# DAG_ModelComparison

This program uses Herbie to fetch temperature, dewpoint, visibility, 10 m wind speed, & wind gust fields from the HRRR, NAM, NBM, GFS, & other models run at NCEP, verified against either the RTMA or URMA analysis. The analysis is regridded onto the model grid with a lightweight SciPy KDTree (k-nearest inverse-distance) regridder. MatPlotLib & Cartopy are then used to plot the data in a map. Wind speed is derived from the model's U/V components when no direct wind-speed field is published.

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

For scripted and terminal use there is a non-interactive CLI built on the same
engine. After installing the package (`pip install -e .` inside the conda env)
the `compare` command is on your `$PATH` (equivalently
`python -m comparator.cli`):

```
# Generate one comparison. With no --date it runs in ROLLING REAL-TIME mode:
# it verifies the most recent analysis (now - lag) using the forecast lead
# closest to --lead. Prints the saved PNG path on success.
compare run --model hrrr --var TMP --verif rtma

# See exactly what a rolling run would target, without fetching anything:
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
`out_dir`, and `log_dir` (the web-server working directory may differ from
the project, so relative paths land in the wrong place). Settings resolve in the
order
**CLI flag > `DAG_*` env var > `config.yaml` > built-in default**. Each `run`
appends a record (incl. mean / RMSE error stats) to `<out_dir>/runs.jsonl` and
logs to `<log_dir>/comparison.log`.

## Web UI (`compare-web`)

Instead of a scheduled job, comparisons are triggered **on demand** from an HTML
form: pick a model, variable, verification source, mode (single frame or GIF),
and target time, click **Build comparison**, and the server downloads the GRIB2
files and renders the image right then. It drives the same engine as the CLI and
appends to the same `runs.jsonl` manifest, so `compare list` still sees web
builds.

Run it locally (no extra dependencies — pure standard library):

```
compare-web serve                 # then open http://127.0.0.1:8000/
compare-web serve --port 9000     # (equivalently: python -m comparator.webserver serve)
```

`config.yaml` / `DAG_*` env vars still govern `data_dir`, `out_dir`, and
`log_dir` exactly as for the CLI.

### Deploying on the intranet server (daemon + reverse proxy)

The form is meant to live behind the company intranet web server: `compare-web
serve` runs as an always-on daemon (cron-supervised — no systemd available),
and Apache httpd reverse-proxies a URL path to it (`ProxyPass`/
`ProxyPassReverse`). Because the scientific stack is conda-managed and the
target box is typically air-gapped, the environment is shipped with
**conda-pack**. The full, step-by-step guide (building the env offline,
installing the daemon + cron supervision, and the httpd proxy config) is in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**, with the deployable scripts and
config in **[deploy/](deploy/)**.

The manual entry points still work unchanged and share the same engine — the
interactive `python new_comparison.py` and the non-interactive
`compare run | list | latest` (see above).
