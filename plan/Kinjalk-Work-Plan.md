# Kinjalk — Work Plan: Challenge 3, 3D Phenotyping

- **Owner:** Kinjalk
- **Challenge:** 3 — 3D phenotyping from RGB-D point clouds
- **Dataset:** ADA 2025 ([paper](https://arxiv.org/html/2610.23609v1), [code](https://github.com/pippowell/ada2025), data DOI `10.26249/FK2/DMDLTE`)
- **Time box:** 2026-11-14 and 2026-11-15
- **App framework:** Dash + Plotly web app (agreed)
- **Status:** plan only. Nothing is built or downloaded. Build starts after the master plan.

Numbers marked *(estimate)* are my own calculations from the camera
calibration and file sizes. They must be checked on real data in Block 1.
Parameters marked *(start value)* are tuned on the first two plants.

---

## 1. Deliverables

| # | Deliverable | Acceptance test |
|---|---|---|
| D1 | `uiwue.threed` Python package: images to trait table | one command processes a plant folder and writes parquet with no manual steps |
| D2 | `traits_runs.parquet`: one row per plant per run | covers all processed plants, 2025-08-12 to 2025-09-19, all available hours |
| D3 | Validation report: paper-parity, noise floor, QC agreement | three numbers reported, pass/fail stated |
| D4 | Stress results: onset day per trait, turgor correlation | one ranked table, four figures |
| D5 | Three Dash pages: 3D Browser, Compare, Export | each page works on the full table with no code edits |
| D6 | Five slides | problem, pipeline, validation, result, demo |

## 2. Facts about the 3D data

| Item | Fact |
|---|---|
| Depth camera | Pico Flexx time-of-flight, 224 x 171 px, top-down |
| Depth intrinsics | fx 213.08, fy 213.48, cx 106.31, cy 87.48 |
| Depth file | `depth_NNN.tif`, 2 pages: page 0 depth in metres (float32), page 1 NIR intensity (uint16) |
| Orientation | depth and thermal images are stored rotated 180°; the ADA code applies `np.rot90` twice on load |
| Frames | 15 depth + 15 RGB + 5 thermal per plant per run, each with a robot position in `frames_<sensor>.csv` |
| Runs | 06, 09, 12, 15, 18 h. Re-runs at other hours replace failed runs |
| Dates | 2025-08-12 to 2025-09-22. No depth from 2025-09-19 15 h onward |
| Sensor height | 0.60 m (the ADA code's `DEFAULT_SENSOR_HEIGHT_M`) |
| Pixel size on the ground | about 2.8 mm at 0.60 m *(estimate: 0.60 / 213)* |
| View on the ground | about 0.63 m x 0.48 m *(estimate)* |
| Depth noise, one frame | 6 to 12 mm at 0.60 m *(estimate from the 1–2 % sensor spec)* |
| Plant grid | 18 rows (A–R) x 8 columns, checkerboard, 72 plants per 2 m x 1 m bed |
| Nearest neighbour | about 0.17 m diagonal *(estimate from bed size / grid)* |
| Treatment | W1 control. W2, W3 deficit. Irrigation off 2025-09-03, on again 2025-09-15 |

What the paper already did, so I do not repeat it as "new":

- One frame only (frame 7), 15 h runs only, 10 plants, 2025-09-01 to 2025-09-14.
- Vegetation = ExG > 5 on 8-bit RGB. Height = `0.60 − 25th percentile of depth`.
- No ground fit, no plant isolation (neighbours are included), one trait.
- Result: control +4.30 mm/day; treatment slope differs by −13.08 mm/day.

What I add: all 15 frames, all hours, soil-relative height, single-plant
isolation, eight more traits, a noise floor, automatic QC, and a daily-rhythm
analysis. Depth works in the dark, so it is the only structural signal at 06 h.

## 3. Data to download

Per-plant size *(estimate)*: depth about 0.48 GB, RGB about 0.90 GB.

| Tier | Plants | Why | Size (depth + RGB) |
|---|---|---|---|
| 1 | `A2 A8 J5 R1 R7` in W1 and W2 (10 plants) | the paper's plants; lets me check my numbers against the paper | about 14 GB |
| 2 | `W1_B5 W1_E4 W1_G6 W1_K6 W2_B3 W2_G2 W2_I6 W2_Q6` (8 plants) | the ADA team hand-labelled 1,860 runs of these for image problems; lets me test my QC flags | about 11 GB |
| 3 | turgor-clip plants: sensors 10–18 in W2, 19–21 in W1 (12 plants) | needed for the turgor correlation (E5) | about 17 GB |
| — | environmental sub-dataset | turgor, air, PAR, metadata workbook | 1.5 GB |

Tier 3 plant positions are in the "Sensor Guide" sheet of
`metadata_rwc_par_en.xlsx`, so the environmental data is downloaded first.
Thermal for Tier 1 (about 5 GB *(estimate)*) is optional, only for colouring
the 3D view by temperature.

Commands (from a clone of the ADA repo; `--list-only` shows sizes first):

```bash
python utils/download_dataset.py --modalities environmental --data-dir <DATA> -y
```

```bash
python utils/download_dataset.py --modalities depth rgb --beds W1 W2 --plants A2 A8 J5 R1 R7 --data-dir <DATA> -y
```

```bash
python utils/download_dataset.py --modalities depth rgb --plants W1_B5 W1_E4 W1_G6 W1_K6 W2_B3 W2_G2 W2_I6 W2_Q6 --data-dir <DATA> -y
```

```bash
python utils/unzip_data.py --data-dir <DATA>
```

Development starts on two plants (`W1_J5`, `W2_J5`) as soon as they land.

## 4. Pipeline specification

```
load -> undistort -> unproject -> colour -> reference level -> vegetation mask
     -> isolate centre plant -> traits (per frame) -> aggregate (per run) -> QC -> parquet
```

### 4.1 Load

- Read `depth_NNN.tif` page 0 (metres) and page 1 (NIR). Rotate both 180°.
- Read `rgb_NNN.jpg`. Read `frames_depth.csv` and `frames_rgb.csv`.
- Pair depth frame *k* with RGB frame *k*. If a CSV is missing (known on
  2025-08-29 and 2025-09-15), process depth without colour and set `qc_no_pose`.
- Valid depth: 0.20 m < z < 0.80 m *(start value)*. Everything else is dropped.

### 4.2 Point cloud and colour

- Undistort depth and RGB with the ADA calibration (copied into
  `uiwue/threed/calib.py`, with the source commit noted).
- Unproject: `x = (u − cx)·z / fx`, `y = (v − cy)·z / fy`.
- Colour: project each point into the RGB image with the depth-to-RGB
  transform, shifted by the two robot positions, exactly as the ADA code does.
- Two modes, set by a flag:
  - `parity`: fill depth holes up to 5 px (as the ADA code does). Used only for E0.
  - `traits`: no hole filling. A 3 x 3 median filter on depth to reduce noise. Used for everything else.

### 4.3 Reference level (where "height zero" is)

The paper measures height from a fixed 0.60 m sensor height. That ignores robot
height error and pot differences. I measure from the pot instead.

1. Candidate soil points: not vegetation, inside 6 cm of the plant centre *(start value)*.
2. Fit a plane with RANSAC: 5 mm distance threshold, 500 iterations.
3. Accept if at least 150 inlier points and tilt under 10° *(start values)*.
4. If rejected (plant covers the pot, common after about BBCH 12–14), reuse
   that plant's median reference from accepted runs, and set `qc_ref_carried`.
5. Height of each point = signed distance above the reference plane.

### 4.4 Vegetation mask

| Condition | Rule |
|---|---|
| Normal light | ExG = 2G − R − B > 5 (same as the paper) and height > 5 mm |
| Dark frame (mean RGB brightness < 40, typical at 06 h) | height > 15 mm only; set `qc_dark` |
| No colour available | height > 15 mm only |

All thresholds are *(start values)*. The height-only rule will pick up pot rims
if they stand above the soil; this is checked by eye in Block 1.

### 4.5 Isolate the centre plant

1. Plant centre: the image centre projected to the reference plane, refined to
   the centroid of vegetation points within 4 cm of it on early dates.
2. Cluster vegetation points with DBSCAN (`eps` 8 mm, `min_samples` 10) *(start values)*.
3. Keep clusters whose nearest point is within 3 cm of the centre.
4. Clip to a cylinder of radius 8.5 cm around the centre (half the neighbour
   distance) *(start value)*. Record `clipped_fraction` = share of kept points
   within 5 mm of the cylinder wall.
5. If no cluster qualifies: `qc_no_plant`. If the kept cluster's centroid is
   more than 3 cm from the centre: `qc_not_centred`.

Once leaves of neighbours touch, isolation is approximate. `clipped_fraction`
above 0.10 sets `qc_overlap`, and those runs are shown but excluded from statistics by default.

### 4.6 Traits (per frame)

| Trait | Unit | Exact definition |
|---|---|---|
| `height_p50`, `height_p75`, `height_p95` | mm | percentiles of point height above the reference plane |
| `projected_area` | cm² | sum over plant points of `z² / (fx·fy)` (ground area of each pixel) |
| `leaf_area_est` | cm² | sum of `pixel area / cos(inclination)`, inclination capped at 75° |
| `hull_volume` | cm³ | convex hull of plant points plus their projections onto the reference plane |
| `canopy_radius` | mm | 95th percentile of horizontal distance from the plant centre |
| `inclination_mean`, `inclination_sd` | degrees | angle between each point's surface normal and the reference-plane normal; 0° = flat leaf, 90° = vertical |
| `droop_index` | ratio | `height_p50 / height_p95`; falls when leaf tips sink |
| `compactness` | ratio | `projected_area` / area of the 2D convex hull in top view |
| `n_points` | count | plant points |

Normals: Open3D, neighbours within 12 mm, at most 30 *(start value)*. Depth
noise is larger than the pixel spacing, so the two inclination traits are the
most at risk. E1 decides whether they are kept.

### 4.7 Aggregate and QC (per run)

- Each trait: median over the 15 frames, plus MAD as `<trait>_mad`.
- `n_frames_ok`: frames that passed. A run needs at least 8.
- `qc_few_points`: median `n_points` < 300 *(start value)*.
- `qc_high_spread`: `height_p75_mad` > 5 mm *(start value)*.
- `qc_pass`: true when no blocking flag is set (`no_plant`, `few_points`, `high_spread`, `overlap`).

### 4.8 Files written

| File | Content |
|---|---|
| `data/derived/threed/traits_frames.parquet` | one row per plant, run and frame |
| `data/derived/threed/traits_runs.parquet` | one row per plant and run (the shared table) |
| `data/derived/threed/clouds/<bed>_<plant>/<date>_<hour>.npz` | frame 7 cloud for the viewer: xyz, rgb, height, plant label |
| `data/derived/threed/run_log.csv` | every run attempted, with status and error text |

### 4.9 Compute budget

- Tier 1: 10 plants x about 38 days x 5 runs x 15 frames = about 28,500 frame pairs *(estimate)*.
- Target 0.2 s per frame gives about 1.6 h on one core; run in parallel with `joblib` across runs.
- The 15 h runs are processed first so statistics can start before the full batch ends.

## 5. Code layout

```
src/uiwue/threed/
  calib.py      camera constants from the ADA repo
  io.py         list_runs(data_dir, plants, dates, hours); load_run(run) -> frames
  cloud.py      build_cloud(depth, nir, rgb, pose, mode) -> Cloud(xyz, rgb, nir)
  ground.py     fit_reference(cloud, centre) -> Plane | None
  segment.py    vegetation_mask(cloud, plane); isolate_plant(cloud, mask, centre)
  traits.py     frame_traits(plant_points, plane) -> dict
  qc.py         aggregate_run(frame_rows) -> run_row with flags
  run.py        CLI: python -m uiwue.threed.run --data-dir ... --plants ... --hours ...
  stats.py      baseline_normalise, fit_lmm, onset_day, noise_floor, turgor_corr
app/pages/
  threed_browser.py
  threed_compare.py
  threed_export.py
tests/threed/   synthetic-scene tests (flat plane + tilted disc of known height, area, angle)
```

The synthetic tests check that a disc of known size, height and tilt gives the
right `height`, `projected_area` and `inclination` within 5 %.

## 6. Shared table schema

`traits_runs.parquet`, one row per plant per run:

| Column | Type | Example |
|---|---|---|
| `bed` | str | `W2` |
| `plant` | str | `J5` |
| `date` | date | `2025-09-08` |
| `hour` | int | `15` (folder hour; re-runs keep their own hour) |
| `slot` | int | `15` (scheduled slot the run stands for) |
| `timestamp_utc` | datetime | median frame time |
| `treatment` | str | `control` / `deficit` |
| `days_since_cutoff` | int | `5` (0 on 2025-09-03) |
| `<trait>`, `<trait>_mad` | float | section 4.6 |
| `n_frames_ok` | int | `15` |
| `qc_*`, `qc_pass` | bool | section 4.7 |

**Proposal for the master plan:** every challenge keys its per-plant results by
`bed`, `plant`, `date`, `hour`, so tables join without conversion.

## 7. Experiments

All statistics use `qc_pass` runs only. The treatment period is 2025-09-03 to
2025-09-14. Baseline = each plant's mean over 2025-09-01 and 2025-09-02.

### E0 — Is the pipeline correct?

| Check | Method | Pass |
|---|---|---|
| E0a paper parity | `parity` mode, frame 7, 15 h, Tier 1, height = 0.60 − z_p25, mixed model as in the paper | control slope within 20 % of +4.30 mm/day and interaction within 20 % of −13.08 mm/day |
| E0b QC agreement | my `qc_not_centred`, `qc_dark`, `qc_no_plant` against the ADA hand labels (`Not Centered`, `Dark`, `no plant`) on the 8 Tier 2 plants | report precision and recall per flag; target recall ≥ 0.7 |
| E0c synthetic scene | unit tests in `tests/threed` | all traits within 5 % of truth |

### E1 — How noisy is each trait?

- Data: `traits_frames.parquet`, all Tier 1 runs.
- Noise per trait: median over runs of the frame-to-frame MAD.
- Signal per trait: absolute control-minus-treatment difference on 2025-09-14.
- Output: table of noise, signal, and signal-to-noise ratio.
- Rule: a trait with ratio below 3 is reported as "not reliable at this sensor resolution" and left out of E3–E5.

### E2 — Are the beds alike before the cutoff?

- Data: 15 h runs, 2025-08-20 to 2025-09-02, W1 against W2.
- Model: `trait ~ day * bed + (1 | plant)`.
- Output: the `day:bed` term per trait. If significant, the beds differed before
  treatment, and E3 results for that trait carry a warning.

### E3 — Which trait shows stress first?

- Data: 15 h runs, 2025-09-01 to 2025-09-14, Tier 1 (plus Tier 2 and 3 plants if processed).
- Step 1, trend: mixed model `value ~ day * treatment + (1 | plant)` on
  baseline-normalised values (`statsmodels` MixedLM). Report the interaction
  slope, confidence interval and p-value per trait.
- Step 2, onset: for each day, Mann-Whitney test of deficit against control,
  Holm-corrected over days. Onset = first of two consecutive significant days.
- Step 3, lead time: onset day minus (a) the first written symptom for those
  plants in the "BBCH & Symptom" sheet (scored 2, 6, 8, 10, 12, 15 September),
  and (b) the onset of NDVI and leaf-air temperature computed the same way by
  teammates, if their tables are ready.
- Output: ranked table of traits by onset day; trajectory figure with spread bands.

### E4 — Does the daily rhythm change under stress?

- Data: all five slots, 2025-08-28 to 2025-09-18.
- Per plant per day:
  - midday depression = value at 15 h − value at 09 h
  - overnight recovery = value at 06 h next day − value at 18 h
- Hypothesis: deficit plants show a growing midday depression first, then lose
  overnight recovery, before height falls permanently.
- Covariates: PAR (raw mV) and vapour pressure deficit computed from air
  temperature and humidity. W2 has no air sensor, so the W1 sensor is used for both.
- Model: `depression ~ days_since_cutoff * treatment + VPD + (1 | plant)`.
- Output: day x hour heatmap per bed; onset day of abnormal depression.
- Depends on E1: if 06 h and 18 h runs fail QC too often, E4 is limited to 09, 12 and 15 h.

### E5 — Do the traits track leaf water status?

- Data: Tier 3 plants; `leaf_readings.parquet` (from 2025-09-02, every 2 min; higher value = lower turgor).
- Turgor value per run: median within ±15 min of the run's timestamp.
- Per plant: Spearman correlation of each trait with turgor, at lags 0, ±1 and ±2 days.
- Pooled: repeated-measures correlation across the 9 W2 plants.
- Output: trait x lag table; the best trait is named as the structural proxy for turgor.

Priority if time runs short: E0, E1, E3, E5, E2, E4.

## 8. Dash pages

Plotly `Scatter3d` renders in WebGL and handles these clouds directly (at most
38,304 points each). Open3D is used for processing only, not for display.

### 8.1 3D Browser (`threed_browser.py`)

- Controls: bed, plant, date slider, hour buttons, colour-by (RGB, height, plant/background, temperature if present), "plant only" toggle.
- Main view: `Scatter3d` of the cached cloud, marker size 2, with the reference plane drawn as a faint mesh.
- Side panel: trait values for that run, QC flags, RGB frame 7 as an image.
- Below: `height_p75` over time for that plant with the current date marked; clicking a point moves the 3D view to that date.
- Second view toggle: same plant, two dates side by side (for example 2025-09-03 and 2025-09-12).

### 8.2 Compare (`threed_compare.py`)

- Controls: trait, groups (treatment, bed, or a hand-picked plant list), hour slot, date range, baseline-normalise toggle, include-QC-failed toggle.
- Main view: mean line and ±1 SD band per group, individual plants as thin lines, cutoff (2025-09-03) and re-watering (2025-09-15) marked.
- Result card: interaction slope with interval, onset day, lead time against first visible symptom.
- Tab: diurnal heatmap (day x hour) for the chosen trait, if E4 is done.
- Tab: trait against turgor for a clip plant, two y-axes, if E5 is done.

### 8.3 Export (`threed_export.py`)

- Filters: beds, plants, dates, hours, traits, QC pass only.
- Table preview (`dash_table`), row count, CSV download button.
- Second button: download the method summary (parameters used, noise floor per trait) so the numbers can be cited.

## 9. Schedule

Blocks are in working order. Hours are effort, not clock time.

| Block | Hours | Work | Done when |
|---|---|---|---|
| 0 | 0.5 | environment; clone ADA repo; start environmental + Tier 1 download | two plants (`W1_J5`, `W2_J5`) on disk |
| 1 | 2.5 | `calib`, `io`, `cloud`; view one cloud in Plotly; confirm rotation, colour alignment, real sensor height and pixel size | coloured cloud looks right on 3 dates, estimates in section 2 confirmed or corrected |
| 2 | 2.5 | `ground`, `segment`; tune start values on early, mid and late dates of both plants | isolated plant looks right by eye on 6 runs |
| 3 | 2 | `traits`, `qc`, synthetic tests (E0c) | tests pass |
| 4 | 1 | `run` CLI; batch the 15 h runs of Tier 1; start the all-hours batch and Tier 2/3 downloads overnight | `traits_runs.parquet` exists for 15 h |
| 5 | 2 | E0a, E0b, E1 | validation report numbers |
| 6 | 2 | E3, E5, then E2 | ranked onset table, figures |
| 7 | 3 | Dash pages: Browser, Compare, Export | pages run on the full table |
| 8 | 1.5 | E4 if data allows; slides; integration with the team app | demo rehearsed once |

Total about 17 hours over the two days. Blocks 0–4 are day 1. Blocks 5–8 are day 2.

**Cut order if behind:** E4, then E2, then the side-by-side view, then Tier 3 (E5), then thermal colouring. E0, E1, E3 and the three pages are never cut.

## 10. Risks

| Risk | Sign | Response |
|---|---|---|
| Download slow | fewer than 2 plants after 1 hour | develop on what has landed; ask teammates to share already-downloaded depth + RGB |
| Colour misaligned with depth (robot pose offsets) | green bleeding onto soil in the 3D view | mask with height only for that run; set `qc_no_pose` |
| Reference plane wrong once the pot is covered | height jumps between days | carried reference (4.3 step 4); compare against `0.60 − z` as a cross-check column |
| Neighbours merge with the centre plant | `clipped_fraction` rises late in the series | `qc_overlap`; statistics restricted to runs that pass; say so on the slide |
| Inclination too noisy | E1 ratio below 3 | drop the two inclination traits; `droop_index` and `height_p95` carry the wilting story |
| 06 h and 18 h unusable | high `qc_dark` and `qc_high_spread` | E4 on 09, 12, 15 h only |
| n is small (5 plants per bed in Tier 1) | wide intervals | add Tier 2 and Tier 3 plants to E3 (up to 12 control, 18 deficit) |

## 11. Not in the two days

Listed on a "next steps" slide:

- Per-leaf traits: segment leaves in the 1920 x 1080 RGB and lift the masks onto the 3D points.
- Merging the 15 frames into one dense cloud (pose correction + ICP).
- W3 replication and the RWC regression (RWC was sampled in W3 only).
- Recovery after 2025-09-15 (only about 4 days of depth exist).
- Leaf temperature corrected for leaf angle.
- The same trait code on other sugar beet 3D data:
  [BonnBeetClouds3D](https://bonnbeetclouds3d.ipb.uni-bonn.de/) (field, 48 varieties, labelled leaves) and
  [Eschikon](https://projects.asl.ethz.ch/datasets/plant-stress-phenotyping-2018) (greenhouse drought, stereo point clouds). Download access not yet verified.

## 12. What I need from teammates

| Need | From | Why |
|---|---|---|
| Per-plant NDVI and leaf-air temperature tables with the four keys | hyperspectral and thermal owners | lead-time comparison in E3 |
| Agreement on the plant set (Tier 1 for everyone) | all | overlapping results across challenges |
| One shared `<DATA>` location or drive | all | avoid four separate downloads |
| App shell (`app/app.py`, navigation, theme) | whoever owns integration | my pages plug into it |

## 13. Questions for the master plan

1. Do all challenges accept the keys `bed`, `plant`, `date`, `hour` and the `data/derived/<challenge>/` layout?
2. Who owns the app shell and the final integration?
3. Do we process Tier 1 only as a team, or also Tier 2 and Tier 3?
4. Is one shared download possible, and where?
5. Who presents, and how long is the slot?
