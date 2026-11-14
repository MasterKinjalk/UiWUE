# Kinjalk — Work Plan: Challenge 3, 3D Phenotyping

- **Owner:** Kinjalk
- **Challenge:** 3 — 3D phenotyping and computer vision (point clouds from RGB-D)
- **Dataset:** ADA 2025 ([paper](https://arxiv.org/html/2610.23609v1), [code](https://github.com/pippowell/ada2025), data DOI `10.26249/FK2/DMDLTE`)
- **Time box:** 2026-11-14 and 2026-11-15
- **Status:** plan only. Nothing is built or downloaded yet. Build starts after the team master plan is agreed.

## 1. What I am building

A tool that turns the ADA 2025 depth images into plant-structure traits over
time, and shows a biologist when and how water-deficit stress changes plant
shape. It has three parts:

1. **Pipeline:** depth + RGB (+ thermal) images to one tidy trait table.
2. **Statistics:** tests a biologist can trust, with the bed confound handled.
3. **App pages:** browse a plant in 3D over time, compare groups, export data.

It is a geometry-first toolkit. It uses classical point-cloud processing and
needs no model training, so it can be finished in two days and every number can
be explained.

## 2. What the dataset gives for 3D

| Item | Fact |
|---|---|
| Depth camera | Pico Flexx time-of-flight, 224 x 171 px, top-down, single view |
| Points per frame | at most 38,304 |
| Frames | 15 depth + 15 RGB + 5 thermal per plant per run |
| Runs | 06, 09, 12, 15, 18 h daily, 2025-08-12 to 2025-09-22 |
| Depth gap | no depth from 2025-09-19 15 h onward |
| Plants | 216 (72 per bed). W1 = control. W2, W3 = water deficit |
| Stress window | irrigation off 2025-09-03, back on 2025-09-15 |
| Depth file | 2-page TIFF: page 0 depth in metres (float32), page 1 NIR intensity (uint16) |
| Layout | `<modality>/<bed>_<plant>/<YYYY_MM_DD>/<HH>/` |
| Calibration | intrinsics and extrinsics in `analysis/combined/camera_calibration.py` of the ADA repo |
| Fusion | RGB and thermal can be attached to 3D points. Hyperspectral cannot |
| Ground truth | turgor leaf clips on 21 plants (9 in W2, 9 in W3, 3 in W1); RWC on W3 only; BBCH and symptom log |
| 3D labels | none (no leaf labels, no manual heights) |

Download sizes: depth 104.5 GB, RGB 194.5 GB, thermal 114.1 GB, hyperspectral
205.3 GB. The paper subset (`--paper`) is 29.0 GB for 10 plants (5 in W1, 5 in
W2) with all four cameras. Depth + RGB only for those 10 plants should be
roughly 14 GB (my estimate from the per-plant averages).

## 3. Constraints that shape the design

1. **Low resolution, single view.** Canopy-level traits are reliable. Per-leaf
   traits from depth alone are not.
2. **One control bed.** Treatment is confounded with bed. Every comparison uses
   each plant's own pre-cutoff period (before 2025-09-03) as its baseline.
3. **Robot poses are offset in places.** I do not merge frames by pose. I
   compute traits per frame and take the median across the 15 frames.
4. **A plant folder is a grid position.** Neighbouring plants are in view, so
   the centre plant must be isolated.
5. **No 3D ground truth.** Validation is indirect: frame-to-frame repeatability,
   agreement with the paper's height result, and correlation with turgor.

## 4. Pipeline

```
depth TIFF + RGB (+ thermal)
  -> undistort (OpenCV, repo calibration)
  -> unproject depth to 3D points
  -> attach RGB (and temperature) to each point
  -> fit soil/pot plane (RANSAC), convert to height above plane
  -> vegetation mask (ExG on attached RGB + height above plane)
  -> isolate centre plant (connected cluster nearest image centre)
  -> traits per frame
  -> median and spread across 15 frames, QC flags
  -> trait table (parquet + CSV)
```

| Step | Input | Output | Method |
|---|---|---|---|
| Load | run folder | depth, NIR, RGB, thermal arrays | tifffile, OpenCV |
| Cloud | arrays + calibration | points (N,3), colour, temperature | pinhole unprojection |
| Ground | points | plane, height per point | RANSAC plane fit (Open3D) |
| Mask | points + colour | vegetation points | ExG threshold, height > small margin |
| Isolate | vegetation points | centre-plant points | DBSCAN, cluster nearest centre |
| Traits | plant points | one row per frame | see section 5 |
| Aggregate | 15 rows | one row per run + QC | median, MAD |

## 5. Traits

| Trait | Definition | Why it matters |
|---|---|---|
| `height_p50`, `height_p75`, `height_p95` | height percentiles above the soil plane (m) | growth and wilting; p75 matches the paper |
| `projected_area` | top-down area of the plant points (m²) | leaf area proxy, growth |
| `hull_volume` | convex-hull volume of plant points (m³) | canopy size |
| `inclination_mean`, `inclination_sd` | leaf surface angle from horizontal, from point normals (degrees) | wilting changes leaf angle before height drops |
| `droop_index` | `height_p50 / height_p95` | canopy collapse, independent of plant size |
| `compactness` | projected area / convex-hull area in the top view | leaf rolling and folding |
| `n_points` | plant point count | QC |

QC flags per run: `few_points`, `plane_fit_failed`, `not_centred`,
`high_frame_spread`, `missing_modality`.

## 6. Shared table schema (interface to teammates)

One row per plant per run. Keys match the dataset folder names, so thermal and
hyperspectral results from the other challenges can be joined directly.

| Column | Type | Example |
|---|---|---|
| `bed` | str | `W2` |
| `plant` | str | `J5` |
| `date` | date | `2025-09-08` |
| `hour` | int | `15` |
| `treatment` | str | `control` / `deficit` |
| `days_since_cutoff` | int | `5` |
| `<trait>` | float | median over frames |
| `<trait>_mad` | float | spread over frames |
| `qc_*` | bool | QC flags |

**Proposal for the master plan:** every challenge writes its per-plant results
with the same four keys (`bed`, `plant`, `date`, `hour`).

## 7. Experiments

| # | Question | Data | Method | Output |
|---|---|---|---|---|
| E1 | How noisy is each trait? | all processed runs | spread across the 15 frames of a run | noise floor per trait; traits below the floor are dropped |
| E2 | Are the beds the same before the cutoff? | 2025-08-12 to 2025-09-02 | growth curves per bed | baseline check for the bed confound |
| E3 | Which trait shows stress first? | 15 h runs, 2025-09-01 to 2025-09-14 | mixed model `trait ~ day * treatment + (1 \| plant)` on baseline-normalised values | onset day per trait, ranked |
| E4 | Does the daily rhythm change under stress? | all 5 runs per day | midday droop amplitude and overnight recovery per plant-day | day x hour heatmap |
| E5 | Do traits track leaf water status? | plants with turgor clips | lagged Spearman correlation per plant | which trait best tracks turgor |

Order of priority: E1, E3, E5, E2, E4. E5 runs only if a turgor-clip plant is in
the downloaded set; otherwise I download those plants' depth + RGB as well.

Onset day is the first day where the treatment-control contrast is significant
and stays significant on later days.

## 8. App pages I will deliver

1. **3D browser:** pick bed, plant, date and hour. Point cloud coloured by RGB,
   height or temperature, with a time slider.
2. **Compare:** pick groups and a trait. Trajectories with spread bands, the
   irrigation cutoff marked, and the onset day reported.
3. **Diurnal:** day x hour heatmap for one trait (if E4 is done).
4. **Export:** filtered trait table as CSV, with QC flags.

The app framework is a team decision; see `docs/SOFTWARE_STACK.md`.

## 9. Schedule

| When | Work | Done when |
|---|---|---|
| Day 1, first | start download (paper subset, depth + RGB); set up environment | data on disk, environment runs |
| Day 1 | pipeline steps Load to Isolate on one plant, checked by eye in 3D | one clean isolated plant cloud |
| Day 1 | traits, aggregation, QC; batch over 15 h runs | trait table for 10 plants |
| Day 1, late | batch over all 5 runs, runs overnight | full table for the subset |
| Day 2, morning | E1, E3, E5 (then E2, E4) | figures and numbers |
| Day 2, midday | app pages 1, 2, 4 | pages run on the table |
| Day 2, afternoon | slides, integration with the master app | demo ready |

## 10. Out of scope for two days

These go on a "next steps" slide:

- Per-leaf segmentation and per-leaf traits (segment leaves in RGB, lift to 3D).
- Merging the 15 frames into one dense cloud (pose correction + ICP).
- Learned stress models on point-cloud sequences.
- Recovery analysis after 2025-09-15.
- Thermal corrected for leaf angle, per leaf.
- Tests on other datasets (below).

## 11. Other sugar beet 3D datasets

Not used in the two days. Download access is not yet verified.

| Dataset | Content | Possible use |
|---|---|---|
| [BonnBeetClouds3D](https://bonnbeetclouds3d.ipb.uni-bonn.de/) | field UAV photogrammetry, 48 varieties, 186 plants and 2,661 leaves labelled, leaf length/width references | leaf segmentation, trait-code check |
| [Eschikon Plant Stress](https://projects.asl.ethz.ch/datasets/plant-stress-phenotyping-2018) | greenhouse sugar beet, drought / nitrogen / weed stress, stereo point clouds | test whether the drought traits generalise |
| Pheno4D, Crops3D, PLANesT-3D | labelled point clouds of other species | pretraining only |

## 12. Risks

| Risk | Effect | Response |
|---|---|---|
| Download is slow | pipeline has no data | start with 2 plants (1 per bed); develop on those |
| Plant isolation fails once canopies overlap | wrong traits late in the series | QC flag; report on the period where isolation holds |
| ExG mask fails on dark 06 h and 18 h runs | missing early and late runs | fall back to height + NIR intensity mask |
| Depth noise hides small changes | weak diurnal signal | E1 sets the floor; E4 is optional |
| No turgor-clip plant in the subset | E5 cannot run | download those plants' depth + RGB |

## 13. Questions for the master plan

1. Which app framework: Dash in the browser, or a PyQt desktop shell? (My
   recommendation is in `docs/SOFTWARE_STACK.md`.)
2. Do all challenges agree on the four join keys in section 6?
3. Who owns the shared data download and where does it live?
4. Which plants does every challenge process, so results overlap? I propose the
   10 paper-subset plants plus the 21 turgor-clip plants.
5. Who owns the final integration and the slides?
