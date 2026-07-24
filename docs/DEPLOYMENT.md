# Deploying the web daemon behind the intranet server

The comparator is driven from a web form: a user picks a model, variable,
verification source, mode, and target time, clicks **Build comparison**, and
the server downloads the GRIB2 files on demand and renders the image.

The server (`compare-web serve`) is a **long-running daemon**, not a
per-request CGI script: the conda env activation and the scientific-stack
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

## 1. Get the conda environment onto the (air-gapped) server

`conda env create -f environment.yml` needs to reach conda-forge, so it will not
work on a box without internet. Use **conda-pack** to build the environment once
elsewhere and ship a self-contained copy.

> The pack must be built on the **same OS/arch as the server** — Linux x86-64.
> If you only have a Mac, build it in a `linux/amd64` Docker container.

On an internet-connected Linux x86-64 host (or `docker run --platform=linux/amd64
-it continuumio/miniconda3 bash`):

```bash
conda env create -f environment.yml          # creates env "new_comparator"
conda activate new_comparator
conda install -c conda-forge conda-pack       # one-off, for the packer itself
conda pack -n new_comparator -o new_comparator.tar.gz
```

Copy `new_comparator.tar.gz` to the server, then:

```bash
mkdir -p "$HOME/new_comparator"
tar -xzf new_comparator.tar.gz -C "$HOME/new_comparator"
source "$HOME/new_comparator/bin/activate"
conda-unpack        # rewrites the absolute paths baked into the env
```

Verify the heavy stack imports:

```bash
python -c "import herbie, scipy, cartopy, matplotlib; print('env OK')"
```

**Alternative — internal mirror.** If AccuWeather has an internal conda channel
(Artifactory/Nexus), point conda at it in `~/.condarc`
(`channels: [https://<internal-mirror>/conda-forge]`, `channel_priority: strict`)
and run `conda env create -f environment.yml` directly on the server instead of
conda-pack.

The repo lives at `/home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute`
on this host. From the activated env, `pip install -e .` (optional — the
daemon is started with `python -m`, which works as long as the env is active
and the working directory is the repo).

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
* `crontab.example` — the two cron lines below.

Edit the **EDIT THESE** block at the top of `run_dag_server.sh` for your
`ENV_ACTIVATE` and `DAG_CONFIG` paths (the `REPO_DIR` default already matches
this host's checkout). Then:

```bash
chmod +x deploy/*.sh
crontab -e
```

and add (see `deploy/crontab.example` for the exact lines):

```cron
@reboot         /home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute/deploy/run_dag_server.sh
*/5 * * * *     /home/grads/scripts/python/rtc/DAG_ModelComparison-reduced_compute/deploy/watchdog_dag_server.sh
```

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
