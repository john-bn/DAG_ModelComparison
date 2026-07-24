# Deploying the web form on the intranet server

The comparator is driven from a web form: a user picks a model, variable,
verification source, mode, and target time, clicks **Build comparison**, and the
server downloads the GRIB2 files on demand and renders the image. There is **no
scheduled job and no long-running daemon** — the work happens per click via CGI.

```
Browser ──GET──▶ https://fcinet.accuweather.com/<path>/index.html   (static form)
        ──POST─▶ https://fcinet.accuweather.com/<path>/build.cgi     (bash → python)
                     │ activates the conda env, runs the comparator,
                     │ downloads GRIB2, writes PNG/GIF into ./output/
                     ▼
        ◀HTML──  results page  <img src="output/…"> + mean/RMSE/N stats
```

The three moving parts — `index.html`, `build.cgi`, and the `output/` directory
— all live in **one web-served directory** so the browser can reach the form and
the images at the same intranet URL.

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

Also put the repo on the server (e.g. `~/DAG_ModelComparison`) and, from the
activated env, `pip install -e .` (optional — the CGI runs the package with
`python -m`, which works as long as the env is active and you `cd` into the repo).

---

## 2. Check whether CGI is enabled for your directory

Copy `comparator/web/test.cgi` into your web directory and make it executable:

```bash
cp comparator/web/test.cgi ~/public_html/dag/test.cgi   # adjust to your web dir
chmod +x ~/public_html/dag/test.cgi
curl -s https://fcinet.accuweather.com/<path>/dag/test.cgi
```

* Prints **`CGI OK …`** → CGI works. Continue to step 3.
* Prints the **script's text** → CGI is not enabled; the server is serving the
  file statically. Options: add an `.htaccess` (below) if overrides are allowed;
  ask IT to enable `ExecCGI` / a `cgi-bin` for your directory; or fall back to
  the standalone server over an SSH tunnel (step 6).

If your Apache allows per-directory overrides, this `.htaccess` in the web dir
often enables it:

```apache
Options +ExecCGI
AddHandler cgi-script .cgi
```

---

## 3. Lay out the web directory

Pick the directory the intranet server serves for you (commonly
`~/public_html/<something>` for `https://host/~user/…`, or a mapped path). Then:

```bash
WEBDIR=~/public_html/dag           # adjust to your served path
mkdir -p "$WEBDIR/output"

# Backend wrapper (edit its paths — see step 4):
cp comparator/web/build.cgi.example "$WEBDIR/build.cgi"
chmod +x "$WEBDIR/build.cgi"

# The static form, with the dropdowns filled from the registry and its POST
# target pointed at build.cgi:
source "$HOME/new_comparator/bin/activate"
cd ~/DAG_ModelComparison
python -m comparator.webserver render-form --action build.cgi > "$WEBDIR/index.html"
```

Create `config.yaml` (copy `config.example.yaml`) and set **`out_dir` to the
`output/` directory inside the web dir**, so rendered images are web-served:

```yaml
data_dir: /home/ACCU/<you>/dag/data        # GRIB downloads + regridder weights
out_dir:  /home/ACCU/<you>/public_html/dag/output   # <-- inside the web dir
log_dir:  /home/ACCU/<you>/dag/logs
verif: rtma
rolling:
  lag_hours: 2
  default_lead: 24
```

---

## 4. Point `build.cgi` at your paths

Edit the three lines marked *EDIT THESE* in `build.cgi`:

```bash
source "$HOME/new_comparator/bin/activate"   # the conda-pack'd env
export DAG_CONFIG="$HOME/dag/config.yaml"     # the config.yaml from step 3
cd "$HOME/DAG_ModelComparison"                # the repo checkout
```

The wrapper also forces `TZ=UTC` and `MPLBACKEND=Agg` and clears
`HERBIE_SAVE_DIR`, so downloads always land in the configured `data_dir`.

---

## 5. Try it

Open `https://fcinet.accuweather.com/<path>/dag/index.html`, submit **HRRR / TMP
/ RTMA / Single frame / Most recent available**, and confirm an image appears
with mean/RMSE stats. `compare list` (from the CLI) will also show the run, since
the web build appends to the same `runs.jsonl` manifest.

---

## 6. Fallback: standalone server over an SSH tunnel

If CGI cannot be enabled, run the bundled server (no extra dependencies) bound to
loopback and reach it through an SSH tunnel:

```bash
# on the server (keep it running with screen/tmux/nohup if you like):
source "$HOME/new_comparator/bin/activate" && cd ~/DAG_ModelComparison
DAG_CONFIG=~/dag/config.yaml python -m comparator.webserver serve --port 8000

# on your workstation:
ssh -L 8000:127.0.0.1:8000 <you>@<server>
# then open http://localhost:8000/ in your browser
```

The standalone server serves the form, runs the build, and serves the images
itself at `/output/…` — nothing is exposed on the LAN.

---

## Caveats

* **CGI request timeouts.** A single-frame build takes tens of seconds and is
  well within a typical Apache `Timeout` (default 300s). An **animated GIF**
  animates every forecast covering the analysis time and can take several
  minutes — it may exceed the web server's timeout under CGI. For GIFs, prefer
  the CLI (`compare run --mode gif …`) on the server, or raise the server
  `Timeout` for that directory.
* **GIF parallelism.** GIF rendering uses a process pool; running it under CGI is
  heavier than a single frame. The single-frame path is the intended web use.
* **Concurrency.** CGI handles each request in its own process, so concurrent
  submissions are fine, but each still downloads/renders independently.
