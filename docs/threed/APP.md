# UiWUE app: the 3D phenotyping pages

The app is a Dash web app. A biologist opens it in a browser tab. This file
covers the app shell and the three Challenge 3 pages (deliverable D5 in
`plan/Kinjalk-Work-Plan.md`, section 8). It also explains how a teammate adds a page.

## Run

From the repository root:

```bash
.venv/Scripts/python.exe app/app.py                  # http://127.0.0.1:8050
.venv/Scripts/python.exe app/app.py --port 8060      # another port
.venv/Scripts/python.exe app/app.py --debug          # hot reload and the Dash error panel
.venv/Scripts/python.exe app/app.py --host 0.0.0.0   # serve the local network (demo machine)
```

On macOS or Linux, use `.venv/bin/python`.

| Variable | Effect |
|---|---|
| `UIWUE_SYNTHETIC=1` | always use synthetic data, even when the real files exist |
| `UIWUE_ROOT=<dir>` | read `data/` from another repository root (the default is the parent of `app/`) |

The Bootstrap theme (FLATLY) loads from a CDN. Without internet, the pages still
work, but they lose their styling.

## Synthetic data

The pages read the files named in `src/uiwue/threed/schema.py`, the shared
contract. The app uses the synthetic table and clouds from `schema.synthetic_runs()`
and `schema.synthetic_cloud()` in either of these cases:

- `data/derived/threed/traits_runs.parquet` is missing, unreadable, or fails
  `schema.validate_runs`
- `UIWUE_SYNTHETIC=1` is set

Every 3D page then shows an orange **Synthetic data** banner that gives the reason.
With real data, a grey line names the table, its size and when it was last updated.

- **Clouds.** A real cached cloud is always used when its file exists. A synthetic
  cloud appears only when the table is synthetic or no cloud has been cached yet.
  Its title then says "synthetic cloud". When real clouds exist but not the
  one for this run, the view says so. It never shows a made-up cloud.
- **Results.** With a synthetic table, the result card shows rough E1 and E3
  stand-ins computed from that table. They are labelled "synthetic placeholder,
  not a result". With a real table, results come only from the files. A missing
  file is reported in place of its numbers.
- **RGB frames and leaf-clip readings** are always real files.

The loaders cache each file by path and modification time. A file that the
pipeline rewrites is picked up on the next page load or callback, with no restart.

## Pages

### 3D Browser (`/threed-browser`, `app/pages/threed_browser.py`)

- **Controls:**
  - bed and plant
  - a date slider over the plant's run dates (the cutoff and re-watering are marked)
  - hour buttons (only hours that ran that day)
  - colour by RGB, height, plant/background or temperature (temperature is
    disabled when the cloud has no `temp`)
  - "Plant only"
  - "Second view"
- **Main view:**
  - `Scatter3d` of the cached cloud in mm, marker size 2
  - `aspectmode="data"`, so x, y and height share one scale
  - the reference plane at z = 0 as a faint `Mesh3d`
  - the camera is kept while the date changes
  - height colours use one scale per plant (its tallest `height_p95` + 15 %), so
    dates and the two views can be compared
- **Second view:** the same plant on a second date (default 2025-09-03), side by
  side. It uses the same hour, or the nearest hour that ran.
- **Side panel:**
  - run metadata
  - QC badges (red = blocking, amber = not blocking)
  - every trait as median ± MAD with its unit; traits that E1 calls unreliable
    are marked
  - RGB frame 7, downscaled to 480 px
- **Below:** `height_p75` over time, one trace per hour (the selected hour as a
  line), with QC-failed runs as grey crosses. The shown date(s) and the drought
  window are marked. Clicking a point moves the 3D view to that date and hour.
- **Deep link:** `/threed-browser?bed=W2&plant=J5&date=2025-09-08&hour=15`.

### 3D Compare (`/threed-compare`, `app/pages/threed_compare.py`)

- **Controls:**
  - trait (traits that E1 calls unreliable are labelled)
  - group by treatment, bed, or hand-picked plants
  - hour slot and date range
  - baseline-normalise: % change from each plant's mean over 2025-09-01/02, the
    same normalisation as E3
  - include QC-failed runs (excluded by default, as in the statistics)
- **Trajectories tab:** for each group, the daily mean with a ±1 SD band, plus a
  thin line per plant. The drought window is shaded, and the cutoff (2025-09-03)
  and re-watering (2025-09-15) are marked.
- **Diurnal rhythm tab:**
  - a day × hour heatmap per group, computed from the table: either the
    deviation from the plant's daily mean, or the absolute value
  - below it, the midday depression (15 h − 9 h) and the overnight recovery
    (6 h next day − 18 h), from `e4_diurnal.csv`, or computed from the table
    when that file is missing (the page says which)
- **Trait vs turgor tab:**
  - for a plant with a leaf clip: the trait per run (left axis) and the hourly
    median clip reading (right axis; higher = lower turgor), in local time
  - the E5 Spearman rows for that plant and the pooled rows for its bed, from
    `e5_turgor.csv`
- **Result card:**
  - E1 noise floor, signal, SNR and reliable / not reliable
  - E3 deficit − control slope with its CI and p-value, the onset day (Holm, and
    unadjusted when given), and the lead time against the first visible symptom
  - the E2 warning when the beds differed before the cutoff
  - each element says when its file is missing

### 3D Export (`/threed-export`, `app/pages/threed_export.py`)

- **Filters:**
  - beds and hours (an empty choice matches nothing)
  - plants and traits (an empty choice means all)
  - date range
  - QC pass only (on by default)
  - include the MAD columns
- **Preview:** a `dash_table.DataTable` of the first 250 matching rows, with a
  row, plant and date count. QC-failed rows are tinted.
- **Download CSV:** every matching row, with the contract columns of
  `traits_runs.parquet` for the chosen traits. When the data are synthetic, the
  file name ends in `_SYNTHETIC`.
- **Download method summary (Markdown):**
  - data source and filters
  - calendar
  - noise floor per trait (`e1_noise.csv`)
  - QC flags and whether each one blocks
  - the parameters section of `docs/threed/PIPELINE.md` (the whole file when it
    has no section whose heading mentions parameters)
  - E0a parity
  - the result files present

## Files read

| File | Used by |
|---|---|
| `data/derived/threed/traits_runs.parquet` | all three pages |
| `data/derived/threed/clouds/<bed>_<plant>/<YYYY-MM-DD>_<HH>.npz` | Browser (randomly downsampled above 40,000 points) |
| `data/rgb/<bed>_<plant>/<YYYY_MM_DD>/<HH>/rgb_007.jpg` | Browser side panel |
| `data/derived/threed/results/e1_noise.csv` | Browser marks, Compare card, Export summary |
| `data/derived/threed/results/e2_bed_baseline.csv`, `e3_onset.csv` | Compare card |
| `data/derived/threed/results/e4_diurnal.csv` | Compare, diurnal tab |
| `data/derived/threed/results/e5_turgor.csv` | Compare, turgor tab |
| `data/derived/threed/results/e0a_parity.csv` | Export summary |
| `data/leaf_readings.parquet`, `data/metadata_rwc_par_en.xlsx` ("Sensor Guide") | Compare, turgor tab (sensor → plant) |
| `docs/threed/PIPELINE.md` | Export summary |

## Code layout

```
app/
  app.py            shell: Dash(use_pages=True), FLATLY theme, navbar and home page from the page registry
  threed_data.py    data layer for the 3D pages: cached loaders, synthetic fallback, filters,
                    trajectories / diurnal helpers, export, method summary, shared figure helpers
  assets/threed.css hour button group, slider marks
  pages/
    threed_browser.py  threed_compare.py  threed_export.py
tests/app/test_threed_pages.py
```

Run the tests with `.venv/Scripts/python.exe -m pytest -q tests/app`. They cover:

- the data layer, on synthetic data and on a temporary root of real-format files
- every callback, called with realistic inputs
- a smoke test that renders each page through the Flask test client and fires
  its first round of callbacks over HTTP

## Adding a page (teammates)

1. Create one file `app/pages/<challenge>_<name>.py`:

   ```python
   import dash
   from dash import Input, Output, callback, dcc, html

   dash.register_page(
       __name__,
       path="/ndvi-maps",          # URL of the page
       name="NDVI Maps",           # label in the navbar and on the home page
       description="One line for the home page card.",
       order=20,                   # position in the navbar (3D pages use 30-32)
   )

   def layout(**query):            # a function, so the data are re-read on each visit
       return html.Div([dcc.Graph(id="ndvi-graph")])

   @callback(Output("ndvi-graph", "figure"), Input("ndvi-graph", "id"))
   def draw(_):
       ...
   ```

2. Prefix every component id with your page, for example `ndvi-`. All pages
   share one id namespace. The 3D pages use `tb-`, `tc-` and `te-`.
3. Import helpers that sit next to `app.py` by their module name
   (`import threed_data`). `app.py` puts `app/` on `sys.path`.
4. Put CSS in `app/assets/`. Dash loads every file there on every page, so
   scope your selectors to your ids.
5. Do not edit `app.py`. The navbar and the home page list every registered page.
6. Create the app only once per process, because pages register their callbacks
   globally. In tests, load `app/app.py` once (see the `dash_app` fixture in
   `tests/app/test_threed_pages.py`).
