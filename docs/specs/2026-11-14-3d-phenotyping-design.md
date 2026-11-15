# Challenge 3: 3D phenotyping toolkit for ADA 2025 — short design

Date: 2026-11-14. Deadline: end of 2026-11-15 (presentation).

## Intent

Give a biologist a tool that turns the ADA 2025 depth data into plant-structure
traits over time, and shows when and how water-deficit stress changes plant
shape. It must be a usable pipeline plus an app, not only a model.

## Goals

1. A reproducible pipeline: depth + RGB (+ thermal) images to one tidy trait
   table (plant x timestamp x trait, with QC flags).
2. Three statistical results a biologist can trust:
   - E1 noise floor: how much each trait varies across the 15 frames of one run.
   - E3 stress onset: on which day each trait separates treatment from control.
   - E5 water status: how traits track the turgor leaf-clip readings
     (only if a clip plant is in the downloaded subset).
3. A Streamlit app with three flows: browse a plant in 3D over time, compare
   groups on a trait, export the table.

## Non-goals (two-day cut)

- Per-leaf segmentation and per-leaf traits.
- Multi-frame merging with ICP. The per-run median across 15 frames is used
  instead; it also yields E1 for free.
- Learned stress classifiers or 4D models.
- Recovery analysis, cross-dataset tests (Eschikon, BonnBeetClouds3D),
  3D-corrected thermal. These go on a "next steps" slide.

## Anti-goals

- Claiming a treatment effect that is really a bed effect. There is one
  control bed, so every comparison uses each plant's pre-cutoff period
  (before 2025-09-03) as its own baseline.
- A demo that only works on hand-picked plants. QC flags must show failures.

## Constraints

- Depth camera is 224 x 171 px, top-down, single view. Canopy-level traits only.
- Robot poses are offset in places; plant folders are grid positions with
  neighbours in view.
- Depth is missing from 2025-09-19 15 h onward.
- Download time is the critical path. Start with the paper subset
  (10 plants, W1 and W2), depth + RGB first, thermal if time allows.

## Approaches considered

- A. Geometry-first trait toolkit, classical processing, no training. Picked:
  robust at this resolution, explainable, finishable in two days.
- B. Per-leaf traits by segmenting RGB and lifting masks to 3D. Dropped for
  time; no labels to validate against.
- C. Learned 4D stress model. Dropped: 21 labelled plants and a bed confound.

## Design

Pipeline, one function per step:

1. Load depth TIFF (page 0 depth in m, page 1 NIR), RGB, optional thermal.
2. Undistort and unproject with the repo calibration; attach RGB (and
   temperature) to each point.
3. Fit the soil/pot plane with RANSAC; express height above that plane.
4. Vegetation mask: ExG threshold on attached RGB, plus height above plane.
5. Isolate the centre plant: largest connected cluster nearest the image centre.
6. Traits per frame: height p50/p75/p95, projected area, convex-hull volume,
   mean and spread of leaf inclination (from point normals), droop index
   (ratio of p50 to p95 height), point count.
7. Per run: median and spread across frames. QC flags: too few points,
   plane fit failed, plant not centred, high frame-to-frame spread.
8. Write parquet + CSV keyed by bed, plant, date, hour.

Statistics: linear mixed model, trait ~ day x treatment + (1 | plant), on
baseline-normalised values; onset day = first day the daily contrast is
significant and stays so.

App: Streamlit, Open3D/plotly for the 3D view, reading only the parquet table
and cached point clouds.

Stack (builder's call): Python, numpy, Open3D, pandas/pyarrow, statsmodels,
Streamlit.

## Two-day schedule

- Day 1: start download; environment; steps 1-5 on one plant; steps 6-8;
  batch run over the subset at the 15 h runs first, then all runs.
- Day 2 morning: E1, E3, E5; diurnal heatmap if all five runs processed.
- Day 2 afternoon: app flows, figures, slides.

## Team interface

The trait table uses the same plant and timestamp keys as the dataset folders,
so teammates' thermal and hyperspectral results can be joined to it.

## Left to the builder

Thresholds (ExG, cluster distance, minimum points), plot styling, caching layout.
