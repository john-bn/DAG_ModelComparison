# Installing and running DAG_ModelComparison on Linux

A start-to-finish setup for a Linux workstation, from an empty shell to a
rendered comparison. Nothing here needs `sudo`, and nothing needs conda —
[pixi](https://pixi.sh) installs into your home directory and builds the entire
scientific stack (including the C/C++ libraries behind cartopy, pyproj, and
cfgrib) from the lockfile committed alongside the code.

Total time is about ten minutes, most of it waiting on the package download.

---

## Before you start

Check your architecture:

```bash
uname -m
```

You want `x86_64`. The committed `pixi.lock` covers `linux-64` and `osx-arm64`
only. If this prints `aarch64` you are on ARM — skip to
[ARM machines](#arm-machines-aarch64) at the bottom before going further.

You will also need roughly **2 GB of free disk space** in your home directory
(the solved environment is ~1.2 GB) and outbound HTTPS access to
`conda.anaconda.org`, plus `nomads.ncep.noaa.gov` and AWS at runtime for the
GRIB2 downloads.

---

## 1. Install pixi

pixi ships as a single static binary. Download and run the official installer:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

It installs the binary to `~/.pixi/bin/pixi` and adds that directory to your
`PATH` by appending a line to your shell's startup file (`~/.bashrc` for bash,
`~/.zshrc` for zsh). That change does not affect the shell you are currently
in, so pick it up with:

```bash
source ~/.bashrc      # or: source ~/.zshrc
```

Confirm it worked:

```bash
pixi --version        # e.g. pixi 0.77.0
```

If `pixi: command not found` persists, your shell may read a different startup
file. Either add it by hand —

```bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

— or just call the binary by its full path, `~/.pixi/bin/pixi`, everywhere
below.

---

## 2. Download the program from Confluence

The code is published as a zip attachment on the project's Confluence page:

> **Confluence page:** `<FILL IN THE CONFLUENCE URL>`
> **Attachment:** `DAG_ModelComparison.zip`

Download it from the **Attachments** section of that page in your browser. If
you are working on a headless box, download it on your laptop and copy it
across:

```bash
scp DAG_ModelComparison.zip <you>@<linux-host>:~/
```

Confluence attachments sit behind authentication, so a bare `curl` of the
attachment URL will return a login page rather than the zip. To pull it
directly on the Linux box, create an Atlassian API token first
(**Profile → Manage account → Security → API tokens**) and use it:

```bash
curl -L -u "<you>@accuweather.com:<API_TOKEN>" \
     -o DAG_ModelComparison.zip \
     "<CONFLUENCE_ATTACHMENT_DOWNLOAD_URL>"
```

Verify you got an archive and not an HTML error page — this should say
`Zip archive data`:

```bash
file DAG_ModelComparison.zip
```

Then unzip it and enter the directory:

```bash
unzip DAG_ModelComparison.zip
cd DAG_ModelComparison
```

If `unzip` is not installed, `python3 -m zipfile -e DAG_ModelComparison.zip .`
does the same job with no extra packages.

Sanity-check that you are in the right place — both of these must exist:

```bash
ls pyproject.toml pixi.lock
```

`pixi.lock` is the important one. It pins every package to an exact build and
hash, so your machine gets byte-identical versions to the ones the program was
tested against. If it is missing from the zip, the archive is incomplete — get
a fresh copy rather than letting pixi re-solve from scratch.

---

## 3. Build the environment

From inside the `DAG_ModelComparison` directory:

```bash
pixi install --frozen
```

`--frozen` means "install exactly what `pixi.lock` says, do not re-solve." That
is what you want: it is faster and it guarantees you get the tested versions.

This downloads about 277 packages into `.pixi/envs/default/` inside the project
directory. Expect a few minutes on a first run. Everything lands under the
project folder — there is no system-wide install, no separate environment root
to remember, and deleting the folder removes every trace.

Confirm the heavy native stack loads:

```bash
pixi run python -c "import herbie, scipy, cartopy, matplotlib, cfgrib, pyproj; print('env OK')"
```

`env OK` means the C libraries (GEOS, PROJ, ecCodes) all resolved and linked
correctly. If you want the test suite and JupyterLab as well, build the dev
environment too — it adds pytest and jupyterlab on top of the same pinned
versions:

```bash
pixi install --frozen -e dev
pixi run -e dev test          # 5 known pre-existing failures, 177 passing
```

---

## 4. Create your config file

Copy the example and edit the three directory paths:

```bash
cp config.example.yaml config.yaml
mkdir -p "$HOME/dag/data" "$HOME/dag/figures" "$HOME/dag/logs"
```

Open `config.yaml` and set **absolute** paths — relative paths resolve against
whatever the working directory happens to be, which is not always the project:

```yaml
data_dir: /home/<you>/dag/data      # GRIB2 downloads + regridder weight cache
out_dir:  /home/<you>/dag/figures   # PNG/GIF output + runs.jsonl manifest
log_dir:  /home/<you>/dag/logs      # comparison.log
verif: rtma
gif_workers: 1
rolling:
  lag_hours: 2
  default_lead: 24
```

Settings resolve in the order **CLI flag > `DAG_*` env var > `config.yaml` >
built-in default**, so you can override any of these per-run without editing
the file.

One trap worth knowing about: if `HERBIE_SAVE_DIR` is set in your environment,
Herbie overrides `data_dir` with it and prints a notice at startup saying so.
If your GRIB2 files are landing somewhere unexpected, that is why:

```bash
echo "$HERBIE_SAVE_DIR"    # should be empty
unset HERBIE_SAVE_DIR      # if it is not
```

---

## 5. Run the program

`pixi run` activates the environment and runs the command in one step — there is
no separate activation to remember.

**The Streamlit app** (the normal way to use it):

```bash
pixi run app
```

That is shorthand for `streamlit run new_comparison.py`. Open the URL it prints,
`http://localhost:8501` by default. Pick a model, field, verification source,
output mode, and valid time from the sidebar, then click **Build comparison**.

On a headless box, forward the port from your laptop instead of running a
browser remotely:

```bash
ssh -L 8501:localhost:8501 <you>@<linux-host>
```

**The command-line interface**, for scripted or batch use:

```bash
pixi run compare run --model hrrr --var TMP --verif rtma
pixi run compare run --model hrrr --var TMP --verif rtma --dry-run
pixi run compare run --model hrrr --var WIND --date 2026-06-18 --init 12 --fxx 24
pixi run compare list --model hrrr --since 2026-06-18
pixi run compare latest --model hrrr --var TMP
```

Models: `hrrr`, `nam5k`, `nam12k`, `rap`, `nbm`, `arw`, `fv3`, `href`, `gfs`.
Variables: `TMP`, `DPT`, `VIS`, `WIND`, `GUST`.

**The intranet web form**, if you want the standard-library UI locally:

```bash
pixi run serve        # then open http://127.0.0.1:8000/
```

If you would rather work in an activated shell — running `python`, `pytest`, or
`streamlit` directly, the way a conda environment behaves — use:

```bash
pixi shell -e dev     # `exit` to leave
```

The first plot is slower than later ones: cartopy downloads Natural Earth
coastline, border, and state shapefiles on demand and caches them in
`~/.local/share/cartopy`. That is a one-time cost per machine.

---

## Updating to a new version

Download the new zip from Confluence and unpack it over a fresh directory, then
rebuild from the new lockfile:

```bash
pixi install --frozen
```

If a dependency changed, the new `pixi.lock` reflects it and `--frozen` picks it
up. Copy your `config.yaml` across from the old directory — it is deliberately
not shipped in the zip so an update cannot clobber your paths.

Your downloaded GRIB2 files, figures, and the `runs.jsonl` manifest live in the
`~/dag` directories from step 4, not in the project folder, so they survive an
update untouched.

---

## Troubleshooting

**`pixi: command not found`** — the `PATH` line has not been sourced. See the
end of step 1.

**`the lock file is not up-to-date with the project`** — `pyproject.toml` was
edited without re-locking. If you did not change dependencies on purpose,
you likely have a mismatched zip; get a clean copy. To accept the change
deliberately, run `pixi lock` and commit the result.

**Solve or download fails with a TLS or proxy error** — the box cannot reach
`conda.anaconda.org`. Point pixi at the internal mirror:
`pixi install --channel https://<internal-mirror>/conda-forge`. For a fully
air-gapped host, see [DEPLOYMENT.md](DEPLOYMENT.md), which covers shipping a
prebuilt environment with `pixi-pack`.

**`ImportError` mentioning `libgeos`, `libproj`, or `libeccodes`** — the
environment is incomplete, usually from an interrupted download. Rebuild it:
`rm -rf .pixi && pixi install --frozen`.

**A projection or CRS error at plot time** — a stale `PROJ_LIB` in your
environment is pointing pyproj at another installation's data files. Unset it
and retry. (`PROJ_DATA` is handled for you: the environment's
`proj4-activate.sh` overrides it with the env's own `share/proj` on
activation, and also sets `PROJ_NETWORK=OFF` so PROJ never reaches for grids
over the network. `PROJ_LIB`, the older variable name, is not overridden.)

**Downloads succeed but the figure is empty or the run errors on a specific
cycle** — that model run probably is not published yet. Add `--dry-run` to see
exactly which init and lead the rolling logic picked, and try an earlier valid
time.

---

## ARM machines (aarch64)

`pixi.lock` does not currently include a `linux-aarch64` entry, so
`pixi install` will refuse to run on an ARM box. Because the stack comes from
conda-forge rather than PyPI wheels, adding it is a two-line change — add the
platform to `[tool.pixi.workspace]` in `pyproject.toml`:

```toml
platforms = ["osx-arm64", "linux-64", "linux-aarch64"]
```

then re-solve and commit the updated lockfile:

```bash
pixi lock
```

This needs a machine with network access to conda-forge, but not an ARM machine
— conda solving is metadata-based, so you can generate the ARM lock entry from
any platform.
