# Software stack and local setup

One shared Python environment for the whole team, so code from every challenge
runs on every machine.

## Stack

| Layer | Library | Used for |
|---|---|---|
| Language | Python 3.12 | everything |
| Tables | pandas, pyarrow, numpy, scipy | trait tables, parquet sensor files |
| Excel | openpyxl | `metadata_rwc_par_en.xlsx` |
| Images | tifffile, opencv-python, scikit-image | depth / thermal TIFF, undistortion, masks |
| 3D | open3d, pyvista | point clouds, plane fit, clustering, normals, meshes |
| Statistics | statsmodels, scikit-learn | mixed models, regression, baselines |
| Charts | plotly | all interactive charts, including 3D scatter |
| Web app | dash, dash-bootstrap-components | multi-page app in the browser |
| 3D in the app | plotly `Scatter3d`, `Mesh3d`, `Surface` | point clouds and surfaces in the browser (WebGL) |
| Files | imagecodecs, plyfile, requests | compressed TIFF, PLY point clouds, dataset download |
| Parallel | joblib | batch processing across runs |
| Figure export | kaleido | Plotly figures to PNG / SVG for slides |
| Notebooks | jupyterlab, matplotlib | exploration |
| Dev | pytest, ruff, tqdm | tests, lint + format, progress bars |

Hyperspectral files are read with `hyperio` from the
[ADA 2025 repo](https://github.com/pippowell/ada2025). It and the authors'
plant-mask model need `rasterio`, `spectral` and `torch`, which only the
hyperspectral owner installs, from `requirements-hsi.txt`. First check that
`rasterio` opens a `.jp2` cube in this environment; if it cannot, run the
extraction in the authors' conda environment (the app only needs the table).

## App framework: Dash + Plotly web app

Decided 2026-11-14: the app is a Dash web app. PyQt is dropped from
`requirements.txt`.

- Dash is multi-page. Each person owns one file in `app/pages/` and nobody
  blocks anybody else. This matters most with four people and two days.
- Plotly figures are plain Python objects. The same figure works in a notebook,
  in Dash, and as an exported image for slides.
- A biologist opens a browser tab. Nothing to install on the demo machine
  beyond the environment.

### 3D in Dash

Dash shows 3D through Plotly's WebGL traces inside `dcc.Graph`:

- `Scatter3d` for point clouds, `Mesh3d` for planes and hulls, `Surface` for
  height maps. Rotate, zoom and hover work out of the box.
- One ADA depth frame is at most 38,304 points, which `Scatter3d` handles
  easily. Keep one cloud under roughly 200,000 points; downsample above that.
- Send clouds to the browser as cached arrays (`.npz`), not raw TIFFs, and
  update the figure from a callback when the date slider moves.
- Open3D and PyVista are for processing (plane fits, clustering, normals).
  Their own viewers open desktop windows and are not used inside the app.

## Setup

Requires Python 3.12 and git. Linux needs glibc 2.31 or newer (Ubuntu 20.04+).

### Option A: uv (fastest)

Install uv: <https://docs.astral.sh/uv/getting-started/installation/>

```bash
git clone https://github.com/MasterKinjalk/UiWUE.git
cd UiWUE
uv venv --python 3.12
uv pip install -r requirements.lock.txt
uv pip install -e .
```

`-e .` installs the `uiwue` package from `src/` in editable mode. For model
fine-tuning (Challenge 3), add the ML extras (`requirements-ml.txt`: torch,
transformers, accelerate). `--torch-backend auto` picks the CUDA build that
matches the GPU driver:

```bash
uv pip install -e ".[ml]" --torch-backend auto
```

### Option B: plain venv + pip

```bash
git clone https://github.com/MasterKinjalk/UiWUE.git
cd UiWUE
python3.12 -m venv .venv
```

On Windows use `py -3.12 -m venv .venv` instead.

### Activate

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Option B only, after activating:

```bash
pip install -r requirements.lock.txt
pip install -e .
```

`pip install -e .` makes the shared package importable as `uiwue` from
anywhere in the repo.

For the ML extras with pip, install torch first from
<https://pytorch.org/get-started/locally/> (pick the CUDA build), then run
`pip install -e ".[ml]"`.

### Check the install

```bash
python -c "import open3d, dash, plotly, pyvista, statsmodels, cv2, tifffile, uiwue.core; print('ok')"
python -m pytest -q
```

### Run the app

```bash
cd app
python app.py
```

Then open <http://127.0.0.1:8050>. The app reads only the tables in
`derived/`, so it runs from a fresh clone with no raw data.

## Versions

`requirements.txt` gives minimum versions. `requirements.lock.txt` pins the
exact versions everyone installs; it is compiled for Windows, macOS and Linux
with `uv pip compile requirements.txt --universal --python-version 3.12 -o
requirements.lock.txt`. On 2026-11-14 the lock installed on Linux and the
import check and tests passed (dash 4.5.0, plotly 7.1.0, pandas 3.1.1,
numpy 2.5.4, open3d 0.20.0, opencv-python 5.0.0). Note the new major versions:
pandas 3 (copy-on-write, string dtype), OpenCV 5, Dash 4.

The ML extras are not in the lock. On 2026-11-14 they resolved to torch
2.14.1+cu130 (Windows, RTX 5070 Ti) and 2.15.0+cu132 (Linux GPU machine),
transformers 5.23.0, accelerate 1.15.0.

To add a library: put the minimum in `requirements.txt`, re-run the compile
command, and push both files (shared files go through Yuming).

## Data

Raw data never goes in git. `.gitignore` blocks `data/` and the large file
types. Download with the ADA repo tools into a local `data/` folder, or point
the package at another folder with the `UIWUE_DATA` environment variable.
`scripts/download_data.sh` clones the ADA repo into `external/ada2025` and runs
those tools in stages (`env`, `dev`, `tier1`, `tier2`, `tier3`, `train`,
`unzip`; no argument runs all except `train`). Downloads resume and are
MD5-checked; `DATA_DIR` sets the folder:

```bash
scripts/download_data.sh env tier1 tier2 tier3 unzip
```

To see sizes before downloading, run the ADA tool directly:

```bash
python external/ada2025/utils/download_dataset.py --list-only
```

The full camera set is about 618 GB. The paper subset (`--paper`) is about
29 GB. PhenoBench (7.6 GB, for the leaf model) comes from
`scripts/download_phenobench.sh`. Pretrained weights come from
`scripts/download_models.py`; see [ML_MODELS.md](ML_MODELS.md).

Derived tables do go in git: `derived/<module>/*.parquet` (under 20 MB each)
plus `derived/<module>/catalogue.csv`. That is how teammates and the demo
machine get your results. Larger derived files (frame-level tables, point
clouds, thumbnails) go in `cache/`, which git ignores.

## Conventions

The full contract is section 6 of [`plan/Master-Plan.md`](../plan/Master-Plan.md).
In short:

- Run-level tables have one row per `bed`, `plant`, `date`, `hour`, where
  **`hour` is the scheduled slot** (6, 9, 12, 15, 18), not the folder hour. A
  re-run folder maps to the nearest slot at most 1 h away;
  `uiwue.core.slot_for_hour` does this. Store the folder hour as
  `<prefix>capture_hour`.
- Every other column starts with its module prefix (`thermal_`, `depth_`,
  `rgb_`, `hsi_`, `env_`, `truth_`, `detect_`, `diurnal_`). Do not store
  `treatment`, `days_since_cutoff` or `phase`; `uiwue.core` adds them.
- Each module writes `catalogue.csv` (`column, label, unit, direction,
  description`) next to its table. The app's pickers read it.
- Write tables with `uiwue.core.write_table`, read the merged table with
  `uiwue.core.load_traits()`, and run `python -m uiwue.core.validate` before
  pushing.
- Format and lint with `ruff format .` and `ruff check .` before pushing.

Repo layout:

```
plan/                 work plans and the master plan
docs/                 stack, setup, result cards (docs/results/<name>.md); docs/threed/ for Challenge 3
src/uiwue/core/       shared contract, validator (Yuming)
src/uiwue/<module>/   one folder per module: thermal, env, diurnal, threed, spectral, truth, rgb, stats, detect
src/uiwue/ml/         fine-tuning of pretrained models: promptda/ (depth), leafseg/ (plant + leaf masks)
scripts/              data and model downloads, GPU-machine setup and training
app/app.py            app shell; app/components.py shared look and pickers
app/pages/            one file per page, one owner each
derived/<module>/     small result tables + catalogue.csv, in git
cache/                large derived files, not in git
tests/                pytest
data/                 raw data, local only, not in git
models/, runs/        pretrained weights and training outputs, not in git
external/             third-party clones (ada2025), not in git
```
