# DAG_ModelComparison

This program uses Herbie to fetch temperature, dewpoint, visibility, 10 m wind speed, & wind gust fields from the HRRR, NAM, NBM, GFS, & other models run at NCEP, verified against either the RTMA or URMA analysis. The analysis is regridded onto the model grid with a lightweight SciPy KDTree (k-nearest inverse-distance) regridder. MatPlotLib & Cartopy are then used to plot the data in a map. Wind speed is derived from the model's U/V components when no direct wind-speed field is published.

## Streamlit app (`new_comparison.py`)

The driver file is `new_comparison.py`, a Streamlit page. Start it either way —
the second form relaunches itself under Streamlit — and open the URL it prints
(http://localhost:8501 by default):

```
streamlit run new_comparison.py
python new_comparison.py
```

Everything is chosen from drop-down menus in the sidebar:

* **NWP model** — HRRR, NAM5k, NAM12k, RAP, NBM, ARW, FV3, GFS, IFS, HREF
* **Field** — TMP (2 m temperature), DPT (2 m dew point), VIS, WIND (10 m), GUST
* **Verify against** — RTMA or URMA
* **Output** — a single frame (PNG), or an animated GIF of every run covering
  the valid time
* **Valid time** — the most recent analysis available, or a specific one

The time menus only offer runs the model actually produces: GFS lists just its
00/06/12/18Z cycles, an HRRR 03Z cycle stops at F018 while the 00Z cycle runs
out to F048, and picking a valid time lists exactly the forecast runs that reach
it. Prefer to think in cycles instead? Switch **Specify by** to *Init + lead*
and each lead is labelled with the valid time it lands on.

Click **Build comparison** and the GRIB2 files are downloaded and the figure
rendered right then, with the engine's progress streaming into the page. The
result is displayed in the page with a **Download** button, the mean error /
RMSE / grid-point count, and an expander listing recent runs from the manifest.
Where the downloads, figures, and logs land is set by `config.yaml` / `DAG_*`
env vars, exactly as for the CLI (see [Configuration](#configuration)).

As the data is downloaded from NOMADS & AWS, no special permissions are required.
Data are downloaded automatically via Herbie and cached locally in ./data/.

## Environment setup

Dependencies are managed with [pixi](https://pixi.sh), which resolves the
scientific stack from conda-forge. That matters because cartopy, pyproj, and
cfgrib are Python layers over C/C++ libraries (GEOS, PROJ, ecCodes) — conda-forge
ships those as real packages the solver reasons about, instead of relying on
whichever wheel happens to vendor them. Install pixi once:

```
curl -fsSL https://pixi.sh/install.sh | bash
```

Then, from the repo root:

```
pixi install -e dev        # build the environment from pixi.lock
pixi run app               # streamlit run new_comparison.py
pixi run -e dev test       # pytest
```

`pixi run <task>` builds the environment first if it is missing, so the commands
above work from a fresh clone with no separate activation step. `pixi shell -e
dev` drops you into an activated shell if you would rather run things directly.

Two environments are defined in `pyproject.toml`: `default` (runtime only, used
by the deployed daemon) and `dev`, which adds pytest and jupyterlab. Exact
package builds and hashes are pinned in the committed `pixi.lock`, so every
machine and the server resolve identically — commit it along with any dependency
change. This program is built for Python 3.11.

Setting this up on a Linux workstation from scratch — installing pixi, fetching
the zip from Confluence, and running your first comparison — is covered
step by step in **[docs/INSTALL_LINUX.md](docs/INSTALL_LINUX.md)**.

## Command-line interface (`compare`)

For scripted and terminal use there is a non-interactive CLI built on the same
engine. `pixi install` already does an editable install of the package, so the
`compare` command is on the environment's `$PATH` — run it as `pixi run compare
…`, or directly after `pixi shell` (equivalently `python -m comparator.cli`):

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

## Intranet web UI (`compare-web`)

The Streamlit app is the front end to reach for locally. `compare-web` is the
same on-demand form built on nothing but the standard library, for the intranet
server, where the deployment is a cron-supervised daemon behind an Apache httpd
reverse proxy (see below). Both share their validation, target resolution, and
build with the CLI (`comparator.runner`), and append to the same `runs.jsonl`
manifest, so `compare list` sees every build regardless of which one made it.

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
`ProxyPassReverse`). The server builds its environment from the committed
`pixi.lock` with `pixi install --frozen`; where the box is genuinely air-gapped
the environment is shipped with **pixi-pack** instead. The full, step-by-step
guide (building the env offline, installing the daemon + cron supervision, and
the httpd proxy config) is in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**, with the deployable scripts and
config in **[deploy/](deploy/)**.

The other entry points share that same engine — the Streamlit app
(`streamlit run new_comparison.py`) and the non-interactive
`compare run | list | latest` (see above).
