# UiWUE — Master Plan

- **Date:** 2026-11-14, written 16:30 CST
- **Status:** proposed. It takes effect 30 minutes after Yuming posts it in the team chat, unless someone objects there.
- **Architect and integrator:** Yuming
- **Inputs:** the four work plans in this folder ([Kinjalk](Kinjalk-Work-Plan.md), [Sam](Sam-Work-Plan.md), [Jack (Jonathan)](Jonathan-Work-Plan.md), [Yuming](Yuming-Work-Plan.md)), [`docs/SOFTWARE_STACK.md`](../docs/SOFTWARE_STACK.md), `requirements.txt`, and the hackathon brief
- **Hard deadline:** Sunday 2026-11-15, **12:00 PM** (online form plus a 2-minute demo video). That is under 20 hours from now.

## Read this first

1. **The deadline is Sunday noon, not Sunday evening.** Three of the four plans schedule work for "Day 2 afternoon". Every scope below is cut to fit a results freeze at 09:30 and a code freeze at 10:00 on Sunday (section 2).
2. **Work is split by layer, with one owner per artefact.** The four plans split by challenge, but the challenges need the same tables, so the same work appeared in up to four plans. Section 4 gives each person a lane; section 5 lists every overlap and who keeps it.
3. **Nobody waits on anybody.** Every module writes a small table to `derived/` in git. The app shows whatever tables exist. A late module is missing from the demo; it does not block it.
4. **One contract, in section 6:** keys, column names, plant set, dates, baseline, onset rule. If your plan says something different, section 6 wins.
5. **The stack holds.** The shared environment installs and imports cleanly. Six gaps need fixing (section 8), none of them a change of framework.

## 1. What we are building

**Product.** UiWUE is a trait-extraction and stress-analysis workbench for a
plant biologist running a water-deficit experiment with multimodal imaging. They
point it at the image folders. It extracts per-plant traits from each sensor
into one table, shows when stress became detectable in each signal, how the
plant's daily rhythm changed, and lets them export the numbers.

**User and decisions.** A plant biologist or phenotyping researcher who needs
to decide (1) which sensor and trait to trust as an early indicator, (2) when
in the day to image, and (3) when a plant needs water.

**Headline for the video** (Sam's pitch, adopted): *how many days of warning
does each sensor give before a stressed plant visibly wilts?*

**What the brief scores**, and where it is covered: a defensible baseline and a
leak-free split (6.6, 6.7), uncertainty (6.6), single against multi-sensor
(Jack), a reduced-sensor option (Jack, Yuming), a named user and decision
(above), stated limits (Methods & limits page).

## 2. The clock

All times CST. Effort available is roughly 10 hours per person.

| Gate | When | What must be true | Who |
|---|---|---|---|
| G0 foundation | Sat 18:30 | `uiwue` package installs; validator and app shell run; thermal and environment tables on `main`; everyone's raw download is running | Yuming; all |
| G1 first tables | Sat 21:00 | every owner has a table on `main` that passes the validator, even if it has 20 rows; ground-truth tables; statistics library v1 run on thermal; depth traits for the 15 h runs of the 10 paper plants; RGB traits for the paper plants | Sam; Jack; Kinjalk |
| G2 full tables | Sat 24:00 | core plants in every table that will make it (spectral may be paper plants only); onset results v1 for every available trait; overnight batches started | all |
| G3 pages | Sun 08:00 | every must-have page loads from a fresh clone, with no raw data | all |
| Results freeze | Sun 09:30 | final tables and one result card per owner pushed | all |
| Code freeze | Sun 10:00 | integrator tags `v1.0`; one dry run of the demo | Yuming |
| Video and form | Sun 10:00–11:15 | video recorded on the tagged app; form text assembled | Sam; Yuming |
| Submit | Sun 11:30 | 30 minutes of buffer | Yuming |

**Rule for anything not in a must-have list:** it starts only after your
must-haves have passed their gate.

## 3. Architecture

```
raw ADA data (local only)            extraction, one owner each          tables in git: derived/
---------------------------------    --------------------------------    ---------------------------------
thermal TIFF, air + light sensors -> uiwue.thermal, uiwue.env  (Yuming) -> thermal/traits_runs, env/env_runs
depth TIFF + RGB JPEG             -> uiwue.threed              (Kinjalk)-> threed/traits_runs  (+ cloud cache)
RGB JPEG, frame 7                 -> uiwue.rgb                 (Jack)   -> rgb/traits_runs
hyperspectral JP2                 -> uiwue.spectral            (Sam)    -> spectral/traits_runs, spectra_runs
clip, RWC and symptom sheets      -> uiwue.truth               (Sam)    -> truth/turgor_runs, plants, rwc_leaves
                                              |
                 uiwue.core (Yuming): keys, design columns, catalogue,
                 validator, load_traits() = one wide table on the four keys
                                              |
        uiwue.stats + uiwue.detect (Jack)            uiwue.diurnal (Yuming)
        baseline, placebo, onset, benchmark,         daily rhythm, plant alert,
        ground-truth link, recovery                  scan schedule, irrigation rules
                                              |
                     Dash app: shell (Yuming) + one file per page (owners)
```

**Layers and the one rule for each**

| Layer | Rule |
|---|---|
| Raw data | Stays on the owner's disk. Each person downloads only the modality they extract. |
| Extraction | One module per modality, one owner. Output is a run-level table that follows section 6. |
| Core | The only code everyone imports. Small, owned by Yuming, changed on request. |
| Statistics | One implementation of baseline, placebo, AUC and onset, owned by Jack, used by every analysis. |
| Analysis | Reads the merged table only. Writes result tables to `derived/`. |
| App | Pages read `derived/` through `uiwue.core`. No page reads raw images. Heavy items a page needs (point clouds, thumbnails) go in `cache/`. |

**Repo layout**

```
plan/  docs/                     plans, stack, result cards (docs/results/<name>.md)
src/uiwue/core/                  Yuming
src/uiwue/thermal/ env/ diurnal/ Yuming
src/uiwue/threed/                Kinjalk
src/uiwue/spectral/ truth/       Sam
src/uiwue/rgb/ stats/ detect/    Jack
app/app.py  app/components.py    Yuming (shell, shared pickers, figure theme)
app/pages/<page>.py              one owner per file (section 7)
derived/<module>/                small tables + catalogue.csv, in git
cache/                           large derived files, not in git
tests/                           contract test (Yuming); module tests (owners)
data/                            raw, local only
```

**Git.** Push to `main`. Edit only your own folders and pages. Shared files
(`src/uiwue/core/`, `app/app.py`, `requirements*.txt`, `.gitignore`) go through
Yuming. `git pull --rebase` before every push.

## 4. Who builds what

| | Kinjalk | Sam | Jack (Jonathan) | Yuming |
|---|---|---|---|---|
| **Lane** | 3D structure | Spectral traits and ground truth | Detection statistics and sensor benchmark | Platform, thermal and environment, daily rhythm and irrigation |
| **Raw data** | depth + RGB | hyperspectral; sensor sheets; leaf scans | RGB | thermal; air and light sensors (already on disk) |
| **Tables** | `threed/traits_runs` | `spectral/traits_runs`, `spectral/spectra_runs`, `truth/*` | `rgb/traits_runs`, `detect/*` | `thermal/traits_runs`, `env/env_runs`, `diurnal/*` |
| **Analyses** | paper parity, noise floor | paper parity for indices, mask QC | placebo, onset per signal, sensor combinations, ground-truth link, recovery | time of day, scan schedule, plant alert, irrigation rules, false alerts |
| **Pages** | 3D canopy (was 3D browser), Trait explorer, Export | Plant profile, Bed map | Early warning, Sensor comparison, Methods & limits | Home, Daily rhythm, Irrigation advisor |
| **Team role** | deputy integrator | story: video script and recording | owner of the shared statistics | integrator; form submission |

### Kinjalk — 3D structure

- **Keep:** the pipeline in your sections 4.1–4.8, paper parity (E0a), synthetic tests (E0c), noise floor (E1), 3D browser, Compare, Export (now one page, 3D canopy).
- **Hand over:** bed check (E2) and onset ranking (E3) to the shared statistics, which run on your table automatically; daily rhythm (E4) to Yuming, using your definitions of midday depression and overnight recovery; turgor correlation (E5) to Jack's ground-truth link.
- **New:** Compare becomes the app-wide **Trait explorer** (any trait from any module, driven by the catalogue). Export covers the merged table. Process W3's nine clip plants too, so depth traits have a confirmation bed.
- **Must-have:** `depth_height_p50/p75/p95`, `depth_projected_area`, `depth_canopy_radius`, `depth_droop_index` for the core plants, 12 h and 15 h first; E0a; the three pages (now one, 3D canopy).
- **Cut first:** the hand-labelled QC comparison (E0b), inclination and normals, hull volume, leaf area estimate, compactness, the side-by-side view.

### Sam — spectral traits and ground truth

- **Keep:** the hyperspectral pipeline (flat-field, authors' mask, NDVI, CI red-edge, SIPI, PRI, WBI, red-edge position), the paper check for indices, the ground-truth work you have already done, Bed map, Plant profile, the pitch.
- **Hand over:** thermal leaf temperature and leaf − air to Yuming's table (same definition as your `leaf_temp`); the early-warning race (S2) to Jack, built to your chart design; trait against turgor (S3) to Jack; daily response (S4), rule replay (S5) and the irrigation page to Yuming, who uses your three rules; the honesty panel to Jack's Methods & limits page, with your content list.
- **Dropped:** canopy-level RWC prediction (S6), because the 97.7 GB download cannot land in time. Your leaf-scan RWC results go on the Plant profile page.
- **New:** `spectral/spectra_runs` (mean plant spectrum per run, for your profile page and Jack's wavelength analysis); `truth/plants` with each clip's onset date and each plant's first symptom date; the video.
- **Must-have:** the three `truth` tables by G1; indices for the 10 paper plants at 15 h, checked against the authors' workbook; Plant profile.
- **Cut first:** Bed map, then indices beyond the paper plants, then the 09 h and 12 h runs.

### Jack — detection statistics and sensor benchmark

- **Keep:** your ground rules (adopted for the whole team, 6.7), the benchmark track (logistic regression and small boosted trees, leave-one-plant-out), onset by AUC with a confidence interval, sensor combinations and the cost ranking, recovery ratio, RGB features.
- **Hand over:** thermal, hyperspectral and depth features come from their owners' tables (this answers your question 1); "repeat per hour" and "one run a day" go to Yuming's schedule analysis; the single-plant view goes into Sam's Plant profile, which shows your stress probability; Export goes to Kinjalk.
- **Main track, now in scope (changed Sat 14 Nov, no longer deferred):** the fusion model from your plan's section 5. It lives in `src/uiwue/fusion/`, with a run guide in `docs/fusion.md`:
  - DOFA for RGB and hyperspectral, ImageBind for thermal and depth.
  - One token per sensor carrying its change from the plant's own baseline, a 2-layer transformer, and heads for deficit, stage and turgor.
  - Stage 1 uses frozen encoders; stages 2-3 add LoRA.
  - It also covers the Part B hyperspectral resolution sweep, the 3 vs 300 band comparison and the 20 nm window masking.
  - **Status:** the code and its tests are on the `jack` branch. Training runs on a separate GPU machine, because the build machine has no usable GPU. That machine downloads its own raw thermal, depth and hyperspectral data (about 1 GB per plant for hyperspectral).
  - **Output:** results land in `derived/detect/fusion_*`. The Sensor comparison page shows them beside the feature models once pushed.
  - **Fallback:** the benchmark track stays the bar the fusion model must clear. If fusion results are not pushed by the submission freeze, the pages show the benchmark alone, with no other change.
- **New:** you own `uiwue.stats`, the one implementation of baseline, placebo, AUC and onset that every analysis calls (6.6), and you run it over every trait in the catalogue. That produces the Early warning page for all signals at once. You also own the ground-truth link for all traits and the Methods & limits page.
- **Must-have:** `uiwue.stats` v1 by Sat 20:30; RGB traits (greenness, green cover, hue; frame 7, centre crop) as the "visible" baseline; placebo and onset for every available trait; sensor-combination table; Early warning and Sensor comparison pages.
- **Cut first:** severity stages (the clip's daily cycle is only 3–4 raw units and absent in W3, so staging on midday recovery is fragile), then the wavelength analysis, then recovery, then Methods & limits (falls back to a short text page).

### Yuming — platform, thermal and environment, daily rhythm and irrigation

- **Keep:** thermal extraction, the control-bed reference, time-of-day separability, flyover schedules, the plant alert, the false-alert check.
- **New:** the platform (`uiwue.core`, validator, app shell, integration, submission); thermal and environment tables for the whole team; daily-rhythm metrics for any trait in the catalogue (Kinjalk's E4 and Sam's S4 definitions); the Irrigation advisor page (Sam's three rules plus the control-bed and no-control-bed variants); thermal at reduced resolution (160 x 120 and 80 x 60) for Jack's cost ranking.
- **Must-have:** G0 in full; day-by-hour separability; alert day per flyover schedule; false-alert counts; Daily rhythm and Irrigation advisor pages; the tagged release and the form.
- **Cut first:** the demand-since-watering figure. The weather-station radiation (DWD) and the published sugar beet baseline are done (added back 2026-11-14 once the must-haves were on `main`). The Open-Meteo and vineyard items in my plan are dropped. The separate Scan schedule page is folded into Daily rhythm.

## 5. Overlaps and how each is resolved

| Work | Appeared in | Owner now | What the others do |
|---|---|---|---|
| Thermal leaf temperature, leaf − air | Sam, Jack, Yuming | Yuming | read `thermal/traits_runs` |
| Hyperspectral indices | Sam, Jack | Sam | Jack reads `spectral/traits_runs` |
| RGB colour features | Jack (Kinjalk uses RGB only to colour and mask clouds) | Jack | — |
| Depth traits | Kinjalk (Jack reads them) | Kinjalk | — |
| Air, humidity, VPD, light at capture time | all four | Yuming | read `env/env_runs` |
| Turgor shift, clip onset, RWC, symptom dates | all four | Sam | read `truth/*` |
| Own-plant baseline | all four, four different windows | Jack, one function | call `uiwue.stats.baseline_delta` |
| Bed check before the cutoff | Kinjalk E2, Jack placebo, Sam, Yuming Y0 | Jack | results shown on Methods & limits |
| Onset day per signal, the early-warning chart | Kinjalk E3, Sam S2, Jack Part A, Yuming Y1 | Jack | one rule (6.6); owners' traits are included automatically |
| W3 replication | Sam S7 | Jack | onset reported for W2 and W3 separately |
| Trait against turgor | Kinjalk E5, Sam S3, Jack Part C | Jack | — |
| Daily rhythm | Kinjalk E4, Sam S4, Yuming | Yuming | generic over any trait |
| Per-hour results, fewer flyovers | Jack (extras), Yuming Y1–Y2 | Yuming | calls Jack's AUC function one hour at a time |
| Irrigation rules and page | Sam S5, Yuming | Yuming | Sam's three rules are the rule set |
| Sensor cost ranking | Jack Part B, Sam's notes | Jack | — |
| Single-plant page | Sam, Jack, Kinjalk | Sam (Plant profile); Kinjalk keeps the 3D canopy page (was the 3D browser) | Jack supplies the probability column |
| Trait-over-time page | Kinjalk Compare, parts of all | Kinjalk (Trait explorer) | — |
| Export | Kinjalk, Jack | Kinjalk | — |
| Limits and honesty page | Sam, Jack | Jack | Sam's content list |
| App shell, integration | asked by all, owned by none | Yuming | — |
| Video and form | not planned | Sam (video), Yuming (form) | one result card each |

## 6. The shared contract

### 6.1 Keys

Every run-level table has exactly one row per `bed`, `plant`, `date`, `hour`.

| Key | Type | Meaning |
|---|---|---|
| `bed` | str | `W1`, `W2`, `W3` |
| `plant` | str | grid position without the bed, e.g. `J5` |
| `date` | date | local date of the run |
| `hour` | int | **scheduled slot**: 6, 9, 12, 15 or 18 |

`hour` is the scheduled slot, not the folder hour. Kinjalk's and Sam's plans
defined it in opposite ways. The slot is chosen because a re-run can sit in a
different folder for each camera (thermal at 16 h, hyperspectral at 15 h), and
then folder hours would not join. A folder hour maps to the nearest slot (at
most 1 hour away). If two folders map to one slot, use the scheduled folder if
it has data, otherwise the re-run with the most frames.

### 6.2 Columns

- Every other column starts with its module prefix: `thermal_`, `depth_`, `rgb_`, `hsi_`, `env_`, `truth_`, `detect_`, `diurnal_`.
- Spread over frames: `<column>_mad`. Quality flags: `<prefix>qc_<flag>` and `<prefix>qc_pass`. Bookkeeping: `<prefix>capture_hour` (the folder hour used), `<prefix>timestamp_utc`, `<prefix>n_frames_ok`.
- Do **not** store `treatment`, `days_since_cutoff` or `phase`. `uiwue.core` adds them from the keys, so they cannot disagree between tables.
- `env/env_runs` and `truth/turgor_runs` use the same four keys.

### 6.3 Catalogue

Each module ships `derived/<module>/catalogue.csv` beside its table:
`column, label, unit, direction, description`. `direction` is `+1` if the value
rises under water deficit, `-1` if it falls, `0` if it is not a stress trait.
The app's pickers and the onset analysis read the catalogue, so adding a trait
to the product means adding a column and a catalogue row, nothing else.

### 6.4 Where tables live

- `derived/<module>/<table>.parquet`, committed to git, under 20 MB each. This is how teammates and the demo machine get your results.
- Anything larger (frame-level tables, point clouds, thumbnails) goes in `cache/`, which git ignores. If a page needs it for the demo, upload it as a release asset and tell Yuming.
- Before pushing a table: `python -m uiwue.core.validate derived/<module>/<table>.parquet`.

### 6.5 Plants and dates

| Item | Decision |
|---|---|
| Core plants (required in every module) | the 10 paper plants (`A2 A8 J5 R1 R7` in W1 and W2) plus the 21 clip plants: 29 plants, 7 of them controls |
| Extended plants (after core) | the eight hand-labelled plants from Kinjalk's Tier 2: `W1_B5 W1_E4 W1_G6 W1_K6 W2_B3 W2_G2 W2_I6 W2_Q6`. This is the answer to Jack's request for more controls. The thermal table will include them, plus `W1_F5` and `W1_N3`. |
| Dates processed | 2025-08-24 to 2025-09-22. Earlier dates are out of scope. |
| Slots | all five for thermal, depth and RGB; 9, 12 and 15 h for hyperspectral |
| Baseline window | 2025-08-30 to 2025-09-02 |
| Placebo window | 2025-08-24 to 2025-08-29 (healthy data not used for the baseline) |
| Day 0 | 2025-09-03 (last watering 11:00). Stress window: day 0 to 12. Re-watered 2025-09-15 18:35. |
| Daily value of a trait | median of the 12 h and 15 h runs that pass QC |

### 6.6 Shared statistics (`uiwue.stats`, owner Jack)

These four functions are the contract. Nobody writes a second version.

| Function | Result |
|---|---|
| `baseline_delta(df, column)` | the value minus the plant's own median over the baseline window at the same slot |
| `auc_by_day(df, column, hours=(12, 15), group="all")` | per day: AUC of deficit against control plants on baseline-delta values, oriented by the catalogue direction, with a 95% interval from 2,000 resamples of plants |
| `onset_day(auc)` | **the first day whose lower bound is above 0.70 on that day and the next** |
| `placebo(df, column)` | the same AUC over the placebo window; a trait that separates beds there is flagged `bed_effect` on every chart |

- Group onset is reported for all deficit plants and for W2 and W3 separately.
- **Plant alert day** is a different thing: one date per plant, from Yuming's rule (score above a threshold on two scans in a row). It is used for irrigation status and for lead time against that plant's own clip.
- **Lead time** is measured against two anchors from `truth/plants`: the plant's first written symptom, and the onset day of RGB greenness.
- Sensitivity check, if time allows: repeat with the baseline taken from 2 Sep only, because the doors near the plants were closed around then (Sam's notes).

### 6.7 Rules for valid results (from Jack's plan, now team-wide)

1. The plant is the unit. Fitted models use leave-one-plant-out. No row-level splits.
2. Models never see date, bed, hour or grid position.
3. Every detection result is shown next to a size-only baseline (`depth_projected_area`).
4. The placebo runs before any result is reported.
5. Any tuned threshold or chosen setting (alert threshold, best time of day) is picked on W2 with controls `A2 J5 R7 B3`, then confirmed unchanged on W3 with controls `A8 R1 Q6`.
6. Normalisation and tuning happen inside the training fold.

### 6.8 Air temperature reference

The two air sensors disagree in the afternoon (the W3-end sensor reads 2.9 °C
warmer at 15 h on average over 1–16 Sep). So:

- `env_air_t` is the **W1-end sensor for every bed**. `env_air_t_w3` is stored beside it.
- `thermal_leaf_air_dt` uses `env_air_t`.
- For comparisons between beds, prefer `thermal_leaf_ref_dt`: leaf temperature minus the control bed's median leaf temperature in the same run.

## 7. The app

Dash multi-page app. Each page registers itself; the shell builds the
navigation. Pages use the shared pickers and figure theme in
`app/components.py` so the app looks like one product.

| Page (file in `app/pages/`) | Owner | What the biologist learns | Priority |
|---|---|---|---|
| Home (`home.py`) | Yuming | what the pipeline did: modules present, plants, dates, missing runs | must |
| 3D canopy (`threed.py`; replaced the 3D browser) | Kinjalk | how much canopy the deficit plants lost, what came back, and what one plant looks like in 3D | must |
| Trait explorer (`trait_explorer.py`) | Kinjalk | how any trait changes over time, by treatment, bed or chosen plants | must |
| Early warning (`early_warning.py`) | Jack | which signal detects stress first, and how long before visible symptoms | must |
| Sensor comparison (`sensor_comparison.py`) | Jack | what combining sensors adds, and the cheapest setup that comes close | must |
| Daily rhythm (`daily_rhythm.py`) | Yuming | how the daily pattern changes under stress, and when to image | must |
| Irrigation advisor (`irrigation.py`) | Yuming | which plants need water now, with the reason and the confidence | must |
| Plant profile (`plant_profile.py`) | Sam | everything known about one plant: spectrum, traits, turgor, RWC, stress probability | must |
| Export (`export.py`) | Kinjalk | the trait table as CSV, with the method summary | must |
| Bed map (`bed_map.py`) | Sam | where in the bed stress appears, over time | should |
| Methods & limits (`methods.py`) | Jack | placebo results, splits, scope, known confounds | should |

**A page is done when** it loads from a fresh clone with only `derived/`, shows
"not available" instead of failing when a module's table is missing, and has
one sentence at the top saying what it answers.

## 8. Software stack: findings and proposed changes

**Verdict.** There is no disagreement about the framework: all four plans
accept Python 3.12 and Dash + Plotly. I installed `requirements.txt` in a fresh
Python 3.12 environment on Linux today and the import check passed (dash 4.5.0,
plotly 7.1.0, pandas 3.1.1, numpy 2.5.4, open3d 0.20.0, OpenCV 5.0.0;
2.6 GB). The gaps are in what the stack does not yet cover:

| # | Gap | Evidence | Proposed change |
|---|---|---|---|
| 1 | The `uiwue` package cannot be imported | the stack doc proposes `src/uiwue/`, and Kinjalk's plan runs `python -m uiwue.threed.run`, but there is no `pyproject.toml` | add a minimal `pyproject.toml`; setup gains `uv pip install -e .` |
| 2 | No way to share derived tables | `.gitignore` blocks `data/` and every `*.parquet`, yet every page needs every module's table | allow `derived/**/*.parquet` and `derived/**/catalogue.csv`; add `cache/` to the ignore list; move table paths from `data/derived/` to `derived/` |
| 3 | Versions are not pinned | only minimums are given, and they resolved to new major versions (pandas 3, OpenCV 5, dash 4) | commit `requirements.lock.txt` from the verified install; everyone installs from the lock |
| 4 | Hyperspectral dependencies are missing | Sam's pipeline needs `hyperio`, `rasterio`, `spectral` and `torch` (the authors' mask model); none are listed | add `requirements-hsi.txt` for the spectral owner only. `hyperio` declares support for Python 3.10–3.12. **Sam's first check:** that `rasterio` opens a `.jp2` cube in this environment. If not, extract in the authors' conda environment; the app only needs the table. |
| 5 | Deep-learning dependencies and a GPU | Jack's main track needs `torch`, TorchGeo, ImageBind, `hyperio` and a GPU | kept out of `requirements.txt`: they are in `requirements-ml.txt` (shared with Kinjalk's ML extras). ImageBind is installed with `--no-deps` because it pins old torch. Training runs on a GPU machine and only the small `derived/detect/fusion_*` tables are pushed. Tests skip without torch. The ImageBind weights are CC BY-NC 4.0 (research use), stated on Methods & limits. |
| 6 | The `hour` convention conflicts with the contract | the stack doc says keys match "the dataset folder names" | change to the scheduled slot (6.1) and add the column-prefix and catalogue rules |

Left alone on purpose: `pyvista`, `jupyterlab`, `kaleido` and `matplotlib` are
not used by any plan, but they install cleanly, so removing them now buys
nothing. Boosted trees use scikit-learn; no extra library is needed.

Housekeeping: `Jonathan-Work-Plan.md` exists twice (repo root and `plan/`, identical).
Jack, please delete the root copy.

## 9. Handoffs

| From | To | What | Needed by |
|---|---|---|---|
| Yuming | all | `uiwue.core`, validator, app shell, `components.py` | G0 |
| Yuming | Jack, Sam | thermal and environment tables | G0 |
| Sam | Jack, Yuming | `truth/plants`, `truth/turgor_runs` | G1 |
| Jack | Yuming | `uiwue.stats` v1 (function names in 6.6) | Sat 20:30 |
| Kinjalk | Jack | `depth_projected_area` (size baseline) | G1 for paper plants |
| Sam | Jack | `spectral/spectra_runs` | G2, if it exists |
| Jack | Sam | `detect/scores` (stress probability per plant and run) | G2 |
| all | Sam, Yuming | result card: three numbers, one figure, two limits | Sun 09:30 |

If a handoff is late, the receiver carries on without it and the app shows the
gap. Nobody blocks on a teammate.

## 10. Answers to the questions in the plans

| Question | Asked by | Answer |
|---|---|---|
| Shared keys and table layout | Kinjalk, Jack | section 6; tables in `derived/<module>/` |
| What `hour` means; mapping re-runs | Jack, Yuming | scheduled slot (6.1) |
| Column prefix per sensor | Jack | yes (6.2) |
| One onset rule | Sam, Jack | yes, Jack's AUC rule (6.6) |
| Shared plant set; extra controls; the 63 RWC plants | all | 29 core plus 8 extended (6.5); RWC plants are out |
| Who owns thermal and hyperspectral extraction | Jack | Yuming and Sam |
| Shared download and data location | Kinjalk, Sam, Jack | none needed: each owner downloads their own modality; results travel through `derived/` |
| Which air sensor for W2 | Sam, Yuming | the W1-end sensor for every bed (6.8) |
| Team-wide splits | Sam | 6.7; no `splits.yaml` needed beyond the fixed control lists |
| `hyperio` and `torch` in requirements | Sam | in `requirements-hsi.txt` (section 8) |
| GPU; ImageBind licence | Jack | training on a separate GPU machine (docs/fusion.md); ImageBind is CC BY-NC 4.0, research use only, stated in the app |
| App shell, integration | all | Yuming; Kinjalk is deputy |
| Slides, presenter | Kinjalk, Sam, Jack | the brief asks for a form and a video, not slides (section 12) |
| Who owns challenge 4 | Yuming | Yuming; Sam's rules and definitions are built in (section 5) |
| Which wilting date | Yuming | per-plant first symptom from `truth/plants`, not one date for all |
| External data | Yuming | allowed; each source is listed with licence and role in `docs/DATA_SOURCES.md` by whoever adds it |

## 11. Risks

| Risk | Sign | Response |
|---|---|---|
| The hyperspectral module does not land | no table for the paper plants by G2 | show the authors' published bed-level indices (the workbook in the ADA repo) on the Trait explorer; say so on Methods & limits |
| The 3D pipeline runs long | no 15 h table at G1 | ship height from the paper's simpler rule (`0.60 − depth`) first, soil-relative height after |
| Too few controls for a stable AUC | wide intervals | extended plants; report intervals as they are |
| A module's table breaks the merged table | validator fails | the validator runs in the contract test; a failing table is left out of the build |
| Integration slips | pages not loading at G3 | each page also runs alone; the video can cut between pages |
| One person is blocked or offline | missed gate | their "cut first" list applies; the deputy integrator covers Yuming |
| Results that are really bed effects | high placebo AUC | flagged `bed_effect` on every chart; never hidden |

## 12. Submission

- **Form (Yuming):** problem statement and solution overview from sections 1 and 3 (the diagram is the workflow diagram the brief asks for); technical approach and results from the four result cards; limits from Methods & limits.
- **Video (Sam), 2 minutes:** the user and the problem (15 s); Home, the pipeline and what it extracted (20 s); 3D canopy and Trait explorer, a plant wilting and its traits diverging (25 s); Early warning, the headline (25 s); Daily rhythm and Irrigation advisor (20 s); Methods & limits and Export (15 s).
- **Result card (everyone, `docs/results/<name>.md`, by Sun 09:30):** three numbers, one figure, two limits.
- **Slides:** none unless someone confirms a live presentation.

## 13. Open items the team must answer

1. Does anyone know of a live presentation after the noon deadline? If not, nobody makes slides.
2. The form needs a named **domain lead** and **technical lead**. Who?
3. Objections to your lane: raise them in the team chat within the 30-minute window. After that, section 4 stands.
