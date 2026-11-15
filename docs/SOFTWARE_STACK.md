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
[ADA 2025 repo](https://github.com/pippowell/ada2025). It is not in
`requirements.txt`; whoever owns the hyperspectral challenge should add the
install line once confirmed.

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
uv pip install -e .
```

`-e .` installs the `uiwue` package from `src/` in editable mode, with the
dependencies in `requirements.txt`. For model fine-tuning, add the ML extras
(`requirements-ml.txt`: torch, transformers, accelerate). `--torch-backend auto`
picks the CUDA build that matches the GPU driver:

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
pip install -e .
```

For the ML extras with pip, install torch first from
<https://pytorch.org/get-started/locally/> (pick the CUDA build), then run
`pip install -e ".[ml]"`.

### Check the install

```bash
python -c "import open3d, dash, plotly, pyvista, statsmodels, cv2, tifffile, uiwue; print('ok')"
```

```bash
pytest -q
```

## Versions

`requirements.txt` and `requirements-ml.txt` give minimum versions. On
2026-11-14 a full install on Windows 11 (Python 3.12.14, RTX 5070 Ti) gave,
among others: open3d 0.20.0, dash 4.5.0, plotly 7.1.0, pyvista 0.49.2,
pandas 3.1.1, numpy 2.5.4, statsmodels 0.15.0, opencv-python 5.0.0.93,
torch 2.14.1+cu130, transformers 5.23.0, accelerate 1.15.0.

If two machines behave differently, compare with `pip freeze` and pin the
version that works in `requirements.txt`.

## Data

Data never goes in git. `.gitignore` blocks `data/` and the large file types.
`scripts/download_data.sh` clones the ADA repo into `external/ada2025` and
downloads the subsets from the work plan into `data/` with the ADA tools
(resumable, MD5-checked). Stages: `env`, `dev`, `tier1`, `tier2`, `tier3`,
`train`, `unzip`; no argument runs all except `train`. `DATA_DIR` overrides
the location.

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

## Conventions

- Per-plant results use the keys `bed`, `plant`, `date`, `hour`, matching the
  dataset folder names, so tables from different challenges join directly.
- Tables are saved as parquet (and CSV for export).
- Format and lint with `ruff format .` and `ruff check .` before pushing.
- Repo layout (created 2026-11-14; the master plan can still change it):

```
plan/        work plans and the master plan
docs/        stack, setup, design notes
src/uiwue/   shared package (io, calibration, per-challenge modules)
  threed/    Challenge 3: calib.py (ADA calibration), io.py (runs, loaders)
  ml/        fine-tuning: promptda/ (depth), leafseg/ (plant + leaf masks)
scripts/     data, model and GPU-machine setup scripts
app/         Dash app; one file per page in app/pages/
notebooks/   exploration
tests/       pytest
data/        local only, not in git
models/      pretrained weights, not in git
runs/        training outputs, not in git
external/    third-party clones (ada2025), not in git
```
