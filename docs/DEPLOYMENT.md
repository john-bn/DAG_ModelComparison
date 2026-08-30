# Deploying the web daemon behind the intranet server

The comparator is driven from a web form: a user picks a model, variable,
verification source, mode, and target time, clicks **Build comparison**, and
the server downloads the GRIB2 files on demand and renders the image.

The server (`compare-web serve`) is a **long-running daemon**, not a
per-request CGI script: the pixi env activation and the scientific-stack
imports (matplotlib/cartopy/scipy) happen once at startup, not on every
click. Apache httpd is configured as a **reverse proxy** — it forwards a URL
path prefix to the daemon over loopback and does nothing scientific-stack
specific itself. Since this host has no usable `systemctl --user` (no D-Bus
user session, no lingering) and no sudo for a system-level systemd unit, the
daemon is supervised by **cron** instead: started at boot, restarted by a
watchdog if it stops answering.

```
Browser ──GET/POST─▶ https://fcinet.accuweather.com/<path>/dag/
                          │  Apache: ProxyPass /dag/ → http://127.0.0.1:8000/
                          ▼
                      compare-web serve   (always running, cron-supervised)
                          │ downloads GRIB2, writes PNG/GIF into out_dir
                          ▼
        ◀──HTML──   results page  <img src="output/…"> + mean/RMSE/N stats
```

---

## 1. Get the Python environment onto the server

The scientific stack (herbie, scipy, cartopy, matplotlib, pyproj, cfgrib/eccodes,
…) is declared in `pixi.toml` and pinned exactly by the committed `pixi.lock`.
It is built with **pixi** — BSD-3-Clause, from prefix.dev, resolving from
conda-forge. No Anaconda-licensed software is involved anywhere in this stack,
which is why the project moved off conda/micromamba.

Install pixi as a normal user (no admin rights, no package manager, lands in
`~/.pixi/bin/pixi`):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL -l          # pick up the PATH entry the installer added
pixi --version
```

Then build the environment from the repo directory. Unlike conda there is no env
*name* and no shared root prefix — pixi creates `.pixi/envs/default/` inside the
repo itself, so nothing outside your checkout is written to:

```bash
cd /home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute
pixi install --locked
```

`--locked` makes pixi build exactly what `pixi.lock` records and fail loudly if
the lock is out of date with `pixi.toml`, instead of silently re-solving. That is
what you want on a server: the environment can never drift from what was
committed and tested.

Verify the heavy stack imports. `pixi run` executes inside the env without
needing shell activation, so it's a self-contained check:

```bash
pixi run env-check
# -> env OK
```

`pixi install` also installs this repo itself as an editable package (the
`[pypi-dependencies]` entry in `pixi.toml`), so the `compare` and `compare-web`
console scripts are on PATH inside the env — the separate `pip install -e .`
step the old conda deploy needed is gone.

**If the server ever loses direct conda-forge access**, point pixi at an internal
mirror by replacing the `channels` list in `pixi.toml` and re-locking, or build
the env on a matching **Linux x86-64** host and ship it with
[`pixi-pack`](https://github.com/quantco/pixi-pack). Neither is needed today —
this host reaches conda-forge directly.

---

## 2. Lay out directories and config.yaml

Unlike the old CGI deploy, `out_dir` no longer needs to live inside a
web-served directory — the daemon serves `output/…` itself, so any path the
daemon's user can write to works.

```bash
mkdir -p "$HOME/dag/data" "$HOME/dag/figures" "$HOME/dag/logs"
```

Create `config.yaml` (copy `config.example.yaml`) with **absolute** paths:

```yaml
data_dir: /home/ACCU/<you>/dag/data        # GRIB downloads + regridder weights
out_dir:  /home/ACCU/<you>/dag/figures     # PNG/GIF outputs + runs.jsonl manifest
log_dir:  /home/ACCU/<you>/dag/logs
verif: rtma
rolling:
  lag_hours: 2
  default_lead: 24
```

---

## 3. Install the daemon + cron supervision

The `deploy/` directory has everything needed:

* `run_dag_server.sh` — idempotent start (safe to call repeatedly).
* `watchdog_dag_server.sh` — cron'd health check + restart.
* `restart_dag_server.sh` — manual restart after a code deploy.
* `cleanup_dag_files.sh` — cron'd disk cleanup (see §7).
* `crontab.example` — the three cron lines below.

Edit the **EDIT THESE** block at the top of `run_dag_server.sh` for your `PIXI`
binary path and your `DAG_CONFIG` path (the `REPO_DIR` default already matches
this host's checkout). Cron does not source `~/.bashrc`, so the pixi path must be
spelled out explicitly rather than inherited from an interactive shell — find it
with `which pixi`. Then:

```bash
chmod +x deploy/*.sh
crontab -e
```

and add (see `deploy/crontab.example` for the exact lines):

```cron
@reboot         /home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute/deploy/run_dag_server.sh
*/5 * * * *     /home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute/deploy/watchdog_dag_server.sh
17 * * * *      /home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute/deploy/cleanup_dag_files.sh
```

(The third line is the disk cleanup job — see §7.)

Start it immediately without waiting for a reboot or the next cron tick:

```bash
deploy/run_dag_server.sh
curl http://127.0.0.1:8000/    # sanity check before wiring up httpd
```

---

## 4. Configure httpd as a reverse proxy

`deploy/dag-comparator.conf` has the `ProxyPass`/`ProxyPassReverse` block.
`mod_proxy` + `mod_proxy_http` are required — confirm with `httpd -M | grep
proxy` (or `apache2ctl -M`). Drop the conf file (or its contents) wherever
your httpd config is included from, adjusting the `/dag/` path if you want a
different URL, then reload httpd:

```bash
apachectl graceful       # or: sudo systemctl reload httpd
```

If you don't have write access to httpd's config yourself, hand
`deploy/dag-comparator.conf` to IT along with the port number the daemon
listens on (8000 by default) — it's a small, self-contained addition, not a
new module install (proxy modules are commonly already loaded, as they were
on this host).

---

## 5. Try it

Open `https://fcinet.accuweather.com/<path>/dag/`, submit **HRRR / TMP /
RTMA / Single frame / Most recent available**, and confirm an image appears
with mean/RMSE stats. `compare list` (from the CLI) will also show the run,
since the web build appends to the same `runs.jsonl` manifest.

---

## 6. Redeploying after a code change

Cron only restarts the daemon on reboot or if the watchdog finds it not
answering — it won't notice a new commit. After `git pull`ing an update:

```bash
deploy/restart_dag_server.sh
```

It runs `pixi install --locked` first, so a commit that changed `pixi.toml` /
`pixi.lock` brings the environment along with the code. (`run_dag_server.sh`
itself uses `--frozen` instead, so the boot-time and watchdog restarts never
need network.)

---

## 7. Purge aged-out files so the disk doesn't fill

Every comparison downloads GRIB2 files into `data_dir` and writes PNG/GIF
figures into `out_dir`. Left alone these grow without bound. `cleanup_dag_files.sh`
deletes them once they age past a retention window (**24 hours by default**):

* in `data_dir` — `*.grib2` / `*.grib` and their `*.idx` index files, then any
  now-empty `<model>/<date>/` subdirectories Herbie leaves behind;
* in `out_dir` — `*.png` and `*.gif` figures.

It deliberately **keeps**, regardless of age: the regridder weight cache
(`weights_*_knn.npz` — expensive to rebuild and reused across every run), the
`runs.jsonl` manifest that `compare list` reads, and everything in `log_dir`.

It's pure `find` with **no pixi env activation** on purpose, so cleanup
keeps working even if the Python env is broken — which is exactly when the disk
is most likely filling up. Edit the **EDIT THESE** block at the top so
`DATA_DIR`/`OUT_DIR` match your `config.yaml` (or just export the same `DAG_DATA_DIR`
/ `DAG_OUT_DIR` the app already honors — those win over the block).

Dry-run it first to see what it would remove without deleting anything:

```bash
deploy/cleanup_dag_files.sh --dry-run          # 24h window, delete nothing
deploy/cleanup_dag_files.sh 48 --dry-run        # 48h window, delete nothing
```

Then add the hourly cron line from §3 (or `deploy/crontab.example`). The cron
*frequency* (hourly) is independent of the *retention* (24h): running hourly
just means a file is removed within an hour of crossing 24h old. To keep files
longer, pass a different hour count as the first argument, e.g.
`cleanup_dag_files.sh 72`, or set `DAG_RETENTION_HOURS`. Each run appends a line
per sweep to `<log_dir>/cleanup.log`.

---

## Caveats

* **GIF timeouts.** `deploy/dag-comparator.conf` sets a 300s proxy timeout for
  this reason: an animated GIF spans every forecast covering the analysis
  time and can take several minutes to render, well past mod_proxy's default
  (60s). If GIF builds still time out, raise the `timeout=` value further.
* **Concurrency is now shared-process, not per-request.** Under the old CGI
  model, each submission got its own OS process — fully isolated, if
  wasteful. The daemon handles concurrent requests on separate threads within
  *one* process, and the plotting code uses matplotlib's global `pyplot`
  state, which isn't thread-safe. `comparator/webserver.py` serializes builds
  with a lock to prevent concurrent submissions from corrupting each other's
  figures, so simultaneous requests queue rather than run in parallel. For a
  handful of internal users this is a non-issue; if usage grows enough that
  queuing becomes noticeable, consider a small worker-process pool instead of
  the single-process lock.
* **GIF parallelism.** GIF rendering uses a process pool (`gif_workers` in
  `config.yaml`) independent of the lock above — keep it low (default 1) on a
  memory-constrained server.
