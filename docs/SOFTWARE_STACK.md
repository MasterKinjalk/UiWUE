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
| Desktop app | PyQt6, PyQt6-WebEngine, pyqtgraph, pyvistaqt | desktop shell, fast native 2D / 3D views |
| Figure export | kaleido | Plotly figures to PNG / SVG for slides |
| Notebooks | jupyterlab, matplotlib | exploration |
| Dev | pytest, ruff, tqdm | tests, lint + format, progress bars |

Hyperspectral files are read with `hyperio` from the
[ADA 2025 repo](https://github.com/pippowell/ada2025). It is not in
`requirements.txt`; whoever owns the hyperspectral challenge should add the
install line once confirmed.

## App framework: recommendation for the master plan

**Dash + Plotly as the main app. PyQt6 as an optional desktop shell.**

- Dash is multi-page. Each person owns one file in `pages/` and nobody blocks
  anybody else. This matters most with four people and two days.
- Plotly figures are plain Python objects. The same figure works in a notebook,
  in Dash, and as an exported image for slides.
- A biologist opens a browser tab. Nothing to install on the demo machine
  beyond the environment.
- PyQt6 can wrap the Dash app in a desktop window with `QWebEngineView`, so we
  can still ship a desktop app without writing the interface twice.
- Use native PyQt views (`pyvistaqt`, `pyqtgraph`) only where Plotly is too
  slow, for example a point cloud above roughly 200,000 points.

This is a recommendation, not a decision. The master plan decides.

## Setup

Requires Python 3.12 and git. Linux needs glibc 2.31 or newer (Ubuntu 20.04+).

### Option A: uv (fastest)

Install uv: <https://docs.astral.sh/uv/getting-started/installation/>

```bash
git clone https://github.com/MasterKinjalk/UiWUE.git
cd UiWUE
uv venv --python 3.12
uv pip install -r requirements.txt
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
pip install -r requirements.txt
```

### Check the install

```bash
python -c "import open3d, dash, plotly, PyQt6, pyvista, statsmodels, cv2, tifffile; print('ok')"
```

## Versions

`requirements.txt` gives minimum versions. On 2026-11-14 it resolved on
Python 3.12 for Windows, macOS (Intel and Apple Silicon) and Linux to, among
others: open3d 0.20.0, dash 4.5.0, plotly 7.1.0, PyQt6 6.12.1, pyvista 0.49.2,
pandas 3.1.1, numpy 2.5.4, statsmodels 0.15.0. This was a dependency
resolution check only; a full install has not been run yet.

If two machines behave differently, compare with `pip freeze` and pin the
version that works in `requirements.txt`.

## Data

Data never goes in git. `.gitignore` blocks `data/` and the large file types.
Download with the ADA repo tools into a local `data/` folder:

```bash
python utils/download_dataset.py --list-only
```

Run that first; it shows sizes before anything downloads. The full camera set
is about 618 GB. The paper subset (`--paper`) is about 29 GB.

## Conventions

- Per-plant results use the keys `bed`, `plant`, `date`, `hour`, matching the
  dataset folder names, so tables from different challenges join directly.
- Tables are saved as parquet (and CSV for export).
- Format and lint with `ruff format .` and `ruff check .` before pushing.
- Proposed repo layout, to be settled in the master plan:

```
plan/        work plans and the master plan
docs/        stack, setup, design notes
src/uiwue/   shared package (io, calibration, per-challenge modules)
app/         Dash app; one file per page in app/pages/
notebooks/   exploration
tests/       pytest
data/        local only, not in git
```
