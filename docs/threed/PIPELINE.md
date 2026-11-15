# 3D trait pipeline (`uiwue.threed`, deliverables D1 and D2)

The pipeline turns ADA 2025 depth (Pico Flexx ToF) and RGB frames into one trait row per plant,
run and frame, and one row per plant and run. Plan: `plan/Kinjalk-Work-Plan.md`, sections 2 and
4-6. Shared contract: `src/uiwue/threed/schema.py` (not changed).

## Run it

```bash
python -m uiwue.threed.run --data-dir data --hours 15 --jobs 20          # 15 h runs, all plants
python -m uiwue.threed.run --data-dir data --jobs 20                     # all hours (resumes)
python -m uiwue.threed.run --data-dir data --parity --jobs 20            # E0a paper parity
python -m uiwue.threed.run --data-dir data --plants W1_J5 W2_J5 --dates 2025-09-01 2025-09-14 --hours 06 15
python -m uiwue.threed.run --data-dir data --diagnose W1_J5:2025-09-03:15 W2_J5:2025-09-12:15   # figures
```

The pipeline writes these files to `--out` (default `data/derived/threed/`):

| File | Content |
|---|---|
| `traits_runs.parquet` | one row per plant and run; contract columns plus the extras listed below; passes `schema.validate_runs` |
| `traits_frames.parquet` | one row per plant, run and frame (all 15 frames, `used` marks the aggregated ones) |
| `traits_parity.parquet` | `schema.PARITY_COLUMNS` plus `height_parity_raw_mm`, `z_p25_m`, `n_veg`, `n_points_total`, `frame` |
| `clouds/<bed>_<plant>/<date>_<HH>.npz` | frame 7, plant-centred, contract format (`xyz`, `rgb`, `height`, `plant`, `frame`), points within 15 cm of the centre; no `temp` |
| `run_log.csv` | every run attempted: keys, status, error text, frames used, seconds |
| `plant_refs.csv` | each plant's carried soil reference (median soil plane of its first 7 dates) |
| `diagnostics/` | verification figures and the batch log |

Runs that are already in `traits_runs.parquet` are skipped. Delete the parquet files to recompute.
Tables are checkpointed every `--flush-every` runs.

Tests (77 in `tests/threed`, including the stats tests): `.venv/Scripts/python.exe -m pytest -q tests/threed`.

## Geometry facts, checked on real data (plan section 2)

| Item | Plan estimate | Measured |
|---|---|---|
| Sensor to soil | 0.60 m | 0.452-0.522 m (1st-99th percentile of the fitted soil level over all runs, median 0.487 m). W1 pots are at about 0.46-0.48 m, W2 pots at about 0.49-0.52 m. |
| Pixel size | 2.8 mm | 2.16 mm at 0.46 m, 2.35 mm at 0.50 m, 1.41 mm at canopy tops (0.30 m) |
| View on the ground | 0.63 x 0.48 m | image 0.48 x 0.37 m at 0.46 m (0.53 x 0.40 m at 0.50 m). The valid ToF area has rounded corners and an invalid right edge: pixels more than 112 px from the principal point are dropped. |
| Depth noise, one frame | 6-12 mm | σ ≈ 4.0 mm raw and 3.6 mm after the 3 x 3 median, on soil. Measured from the frame-to-frame differences of frames 4, 7 and 10 (168 pairs, 2025-08-14..20). |
| RGB coverage | - | about 27 % of the ToF points have no RGB pixel (`has_rgb` 0.69-0.75 per frame) |
| Frames | 15 repeat views | a 3 x 5 grid of robot positions, 0.1 m apart, in 3 sweeps about 40 s apart. Edge plants (A2, A6, ...) have a clamped robot column, so frames repeat (identical images); they are dropped by image hash. |
| Plant centre | image centre | The plant sits where the frame-7 **RGB** optical axis meets the soil, not on the ToF axis. In frame-7 depth coordinates this is (29.5, 61.5) mm (median), 28 px off the ToF centre. It lands on the pot centre in every checked run (`diagnostics/frame7_centring_rgb_axis.png`). |
| ToF on young leaves | - | Bright leaf interiors return no depth (saturation); only the leaf rims are measured (5-16 % of the plant cylinder is invalid). This is handled by the hole fill below. |

## Verified correction: the robot x axis has the wrong sign in the ADA calibration

`calib.R_ROBOT_TO_DEPTH` (ADA, rotz(+90°)) maps robot (x, y) to depth (−y, +x). When the 15 frames
of a run are put in one frame with `p_common = p_frame + M · p_robot`, frames from different robot
columns disagree unless robot x maps to depth **−y**:

| Run (15 h) | rotz(+90°), cross-column median \|Δz\| | (x, y) → (−y, −x) |
|---|---|---|
| W1_J5 2025-08-20 | 14-17 mm | 6-7 mm (same as same-column pairs, 6 mm) |
| W2_J5 2025-08-20 | 9-14 mm | 6-8 mm |
| W1_A2 2025-08-20 | 18-20 mm | 7-9 mm |
| W1_J5 2025-09-03 (closed canopy) | 45-56 mm | 16-22 mm (same-column 15-16 mm) |

This explains the "26-48 mm cross-column disagreement" in `docs/ML_MODELS.md`. It was verified
independently: 14.0 → 5.3 mm on W1_J5 2025-08-20 and 52.2 → 11.1 mm on W2_J5 2025-09-03.
- The corrected matrix `[[0,-1,0],[-1,0,0],[0,0,1]]` now lives in `calib.R_ROBOT_TO_DEPTH`, so
  `calib.depth_to_sensor` and `cloud.M_ROBOT_TO_DEPTH` (`cloud.frame_offset`,
  `cloud.depth_to_rgb`) both use it.
- ADA's original is kept as `calib.R_ROBOT_TO_DEPTH_ADA` and used only by `parity.py`.
- Robot y maps to depth −x in both conventions. RGB and depth poses of one frame differ only in
  robot y for more than 95 % of frames, so depth → RGB colouring is the same as ADA's.

## Method and final parameters

Per run (`run.process_run`):

1. **Load** (`io.run_frames`, `io.load_depth`, `io.load_rgb_reduced`): depth page 0 in metres and
   page 1 NIR, both rotated 180°. RGB is decoded at half resolution (960 x 540, about 3 px per ToF
   pixel). Without `frames_depth.csv` / `frames_rgb.csv` (17 runs, on 2025-08-29 and 2025-09-15),
   `qc_no_pose` is set, the nominal 3 x 5 grid gives the frame offsets, the plant's median centre is
   used, and there is no colour (height-only vegetation rule).
2. **Cloud** (`cloud.build_cloud`, mode `traits`):
   - Enclosed invalid regions of up to 150 px are inpainted from their rim (`fill_small_holes`,
     Telea); 2.8 % of plant points come from filled holes.
   - Masked 3 x 3 median. Pixels with fewer than 5 valid pixels in their window are dropped.
   - Valid depth is 0.20-0.80 m. Each pixel is unprojected along its own undistorted ray.
   - Colour comes from the raw RGB at the distorted projection of each point (shifted extrinsic,
     corrected robot mapping). A z-buffer (8 px cells, 10 mm tolerance) marks points hidden from
     the RGB camera as uncoloured.
3. **Plant centre**: the frame-7 RGB optical axis meets the reference plane. It is refined once
   to the centroid of vegetation within 4 cm in frame 7 (median shift 8.5 mm, p95 27.5 mm). In
   frame k the centre is `c7 − (M·p_robot[k] − M·p_robot[7])`.
4. **Frame selection**: a frame is used when at least 98 % of the plant cylinder footprint
   (r = 8.5 cm on the reference plane) lies inside the valid ToF view and at least 95 % inside the
   RGB view. This depends only on the poses and the plane, not on the image content.
   - Typical sets are 4,7,10 (W1, camera about 0.46 m above the soil) and 0,1,3,4,6,7,9,10 (W2,
     about 0.50 m, wider view). The median is 6 frames per run. The set is in `frames_used`.
   - Cross-column frames are as consistent as same-column ones. The median frame-pair difference
     of `height_p75` is 4.7 mm within the centre column and 4.2 mm between the centre and a side
     column (3,009 runs). So all covered frames are used, not only the centre column.
5. **Reference plane** (`ground.fit_reference`):
   - Non-ExG points within 5 cm of the centre (plan: 6 cm; pots are about 11 cm wide) are pooled
     from all used frames in frame-7 coordinates.
   - RANSAC: 5 mm threshold, 500 hypotheses, only hypotheses within 15° of the camera axis are
     scored, then a least-squares refit.
   - The fit is accepted with at least 150 inliers per pooled frame, tilt ≤ 15° (plan: 10°; the
     soil of edge-row pots A2, A6, A8 and B3 is really tilted by 9.5-13°, with a stable level of
     ±2 mm), and a level within 20 mm of the carried reference.
   - Otherwise the plant's carried reference is used and `qc_ref_carried` is set. The carried
     reference is the median of the accepted fits on its first 7 dates (all hours, at least 3
     fits; `plant_refs.csv`).
   - Results: 4,204 runs use their own fit and 1,766 the carried reference. In 09-01..09-14,
     73 % of W1 runs and 47 % of W2 runs are carried (wilting W2 plants uncover the soil).
   - Over a plant's own fits the soil level varies by 4.0 mm SD (median plant).
6. **Vegetation** (`segment.vegetation_mask`):
   - Normal light: ExG > 5 and height > 5 mm.
   - Dark frame (mean sampled RGB < 10, plan start value 40), or a point without colour:
     height > 15 mm only.
   - `learned=` takes a per-point plant mask instead.
7. **Isolation** (`segment.isolate_plant_multi`):
   - Vegetation of all used frames is pooled in the plant-centred frame on a 2.5 mm voxel grid.
   - DBSCAN with eps 8 mm and min_samples 10. Clusters whose nearest point is within 3 cm of the
     centre are kept.
   - The result is clipped to a cylinder of r = 8.5 cm and mapped back to each frame, so every
     frame keeps the same plant.
   - `clipped_fraction` is the share of kept points within 5 mm of the cylinder wall.
8. **Traits per frame** (`traits.frame_traits`, plan 4.6). Normals come from a PCA over the 30
   nearest neighbours within 12 mm, the same neighbourhood as Open3D's hybrid search. Changes:
   - `projected_area` is the per-pixel footprint `z²·a·cosφ/|n·r̂|` projected on the reference
     plane (`·|n·N|`, clipped to 0.5-2x `z²·a`), where `a` is the per-pixel lens-distortion area
     factor. The plan's `z²/(fx·fy)` is the fronto-parallel case; it underestimates tilted leaves
     seen off-axis (E0c: −4.7 % at 20°, −8.8 % at 35°).
   - `leaf_area_est` = `z²·a·cosφ / max(|n·r̂|, cos 75°)`.
   - Extra columns `cyl_height_p50/75/95`, `cyl_projected_area` and `cyl_n_points` use all
     vegetation inside the cylinder, without clustering. They are more stable on a closed canopy
     but include neighbours' leaves over the pot.
9. **Aggregate and QC** (`qc.aggregate_run`):
   - `<trait>` is the median and `<trait>_mad` the unscaled MAD over the used frames that found
     the plant. `n_frames_ok` counts those frames; a run needs at least 2 (plan: 8 of 15 repeat
     views).
   - Flags:

     | Flag | Rule |
     |---|---|
     | `qc_few_points` | median `n_points` < 300, or fewer than 2 ok frames |
     | `qc_high_spread` | `height_p75_mad` > max(5 mm, 5 % of `height_p75`). 5 mm alone flags 27 % of the 15 h runs in 09-01..09-14: the frames are different viewpoints. |
     | `qc_overlap` | median `clipped_fraction` > 0.10 |
     | `qc_not_centred` | kept centroid > 3 cm from the centre |
     | `qc_no_plant` | no ok frame, or more than half of the used frames find no plant |
     | `qc_dark` | median brightness of the used frames < 10 |

   - `qc_pass` is true when no blocking flag is set (blocking flags are defined in the schema).

Extra run columns: `frames_used`, `n_frames_used`, `ref_source` (fit / carried / rejected_fit /
default), `ref_inliers`, `ref_tilt_deg`, `ref_reject_reason`, `soil_z_mm` (soil depth under the
centre), `centre_x_mm`, `centre_y_mm`, `centre_shift_mm`, `n_frames_on_disk`, `cyl_*`,
`clipped_fraction`, `outside_fraction` (share of the attached clusters cut off by the cylinder),
`centroid_offset_mm`, `brightness`. Extra frame columns: `used`, `built`, `cov_depth`, `cov_rgb`,
`robot_dx`, `robot_dy`, `no_plant`, `n_clusters`, `dark`, `n_veg`, `n_filled`, and the same
QC inputs.

## Parity mode (E0a, `parity.py`)

`parity.py` copies the ADA method verbatim:

- Frame 7 at 15 h, read with `tifffile.imread`. Depth and RGB are undistorted with `cv2.undistort`.
- Holes are filled up to 5 px with the ADA routine.
- Colour uses ADA's own shifted extrinsic with ADA's robot mapping
  (`calib.R_ROBOT_TO_DEPTH_ADA`, rotz(+90°), kept on purpose; `parity.ada_depth_to_rgb`).
- Points without colour are dropped. Vegetation is ExG > 5 on the remapped RGB.
- `height = 0.60 − 25th percentile of depth`.
- Runs with fewer than 1000 vegetation points get `height_parity_mm = NaN`, as the paper drops
  them; `height_parity_raw_mm` keeps the value.

A test checks the output against the ADA functions themselves (`step_01` / `step_02`): `z_p25`
agrees to 1e-9 m and the vegetation count is equal. The table has 140 runs (W1/W2 A2, A8, J5,
R1, R7, 2025-09-01..09-14) and none fall below 1000 points. Median parity height is 273 mm in W1
and 171 mm in W2.

## Verification

- E0c (`tests/threed/test_traits.py`): a ground plane (3° tilt) plus a disc with r = 4 cm at
  10 cm, rendered along the real ToF rays, run through cloud → plane → mask → isolation → traits.
  - Without noise, for tilts of 0-45°, every trait is within 4 % of the truth: the height
    percentiles, projected area, leaf area, hull volume, inclination and compactness.
  - With 5 mm depth noise, heights and projected area stay within 3 %. Leaf area is up to 12 %
    too high and inclination is biased upwards (19° measured on a flat disc), because noise
    tilts the normals. E1 has to judge the inclination traits.
- Diagnostics (`data/derived/threed/diagnostics/`):
  - `run_<plant>_<date>_<hour>.png` shows the coloured cloud, height above reference, isolation,
    side view, the plant across frames and the reference-plane residuals. Runs: W1_J5 and
    W2_J5 on 08-20, 09-03 and 09-12 at 15 h, W2_A6 (edge plant) and W1_J5 09-08 06 h (dark).
  - `clouds3d_*.png` shows 3D views of the cached clouds.
  - What the figures show:
    - The plant falls at the same spot of the plant-centred frame in every used frame.
    - The pot soil lies at 0 mm, the floor between pots at about −60 mm, and pots stand
      out as a checkerboard.
    - Seedlings are isolated cleanly. On the closed canopy (W1 from about 09-01) the kept
      cluster is the leaves connected to the centre, and neighbours' leaves over the pot are
      excluded or included depending on connectivity (`qc_overlap`, `cyl_*`).
    - W2_J5 leaves collapse onto the soil by 09-12.
- Real numbers, 15 h runs that pass QC, bed medians of `height_p75`: W1 rises from 92 mm (09-01)
  to 187 mm (09-14). W2 peaks at 122 mm (09-07) and falls to 71 mm (09-14). `projected_area` in W2
  falls from 121 to 38 cm².
- Typical frame-to-frame spread (median MAD, 15 h runs that pass QC, 09-01..09-14):

  | Trait | Median MAD | Relative |
  |---|---|---|
  | `height_p75` | 2.9 mm | 2.5 % |
  | `projected_area` | 3.9 cm² | 4.1 % |
  | `inclination_mean` | 1.2° | |
  | `droop_index` | 0.02 | |

## Batch (2026-11-15)

- 5,970 runs (all 28 plants, 2025-08-12..09-19, all hours) finished with status `ok`, 0 errors.
- 79,667 frame rows, 5,970 cached clouds (1.2 GB).
- One core takes 0.48 s per run (median), about 0.08 s per built frame. With 20 workers the
  batch takes about 3 minutes (0.03 s per run wall time).

QC rates per slot, all dates (n = 1438, 1137, 1120, 1177, 1098 for 06, 09, 12, 15, 18 h):

| slot | no_plant | few_points | high_spread | overlap | not_centred | dark | ref_carried | no_pose | pass |
|---|---|---|---|---|---|---|---|---|---|
| 06 | 0.259 | 0.421 | 0.095 | 0.114 | 0.127 | 0.607 | 0.318 | 0.000 | 0.401 |
| 09 | 0.081 | 0.331 | 0.118 | 0.157 | 0.136 | 0.104 | 0.266 | 0.025 | 0.427 |
| 12 | 0.057 | 0.306 | 0.125 | 0.148 | 0.131 | 0.000 | 0.309 | 0.025 | 0.444 |
| 15 | 0.054 | 0.297 | 0.124 | 0.158 | 0.126 | 0.059 | 0.321 | 0.014 | 0.455 |
| 18 | 0.046 | 0.303 | 0.118 | 0.131 | 0.117 | 0.247 | 0.256 | 0.000 | 0.474 |

QC rates per slot, 2025-09-01..09-14:

| slot | no_plant | few_points | high_spread | overlap | not_centred | dark | ref_carried | pass |
|---|---|---|---|---|---|---|---|---|
| 06 | 0.004 | 0.037 | 0.141 | 0.217 | 0.130 | 1.000 | 0.573 | 0.640 |
| 09 | 0.003 | 0.036 | 0.196 | 0.293 | 0.074 | 0.194 | 0.556 | 0.533 |
| 12 | 0.003 | 0.019 | 0.237 | 0.227 | 0.064 | 0.000 | 0.629 | 0.557 |
| 15 | 0.004 | 0.022 | 0.185 | 0.256 | 0.046 | 0.059 | 0.606 | 0.577 |
| 18 | 0.002 | 0.031 | 0.202 | 0.230 | 0.054 | 0.415 | 0.467 | 0.582 |

Most `few_points` and `no_plant` flags fall on seedlings: up to 2025-08-20, 96 % of runs have
fewer than 300 plant points and 2 % pass. Every 06 h run in the treatment window is dark, so it is
measured by height only.

## Known data event: leaf inclination rises in both beds from 2025-09-02

`inclination_mean` rises by about 20 % in both beds against the 09-01/09-02 baseline (E3,
`results/e3_trajectories.png`). The cause is a real change in the canopy, not the pipeline and
not the rig.

**How the change looks**
- It is a 3-day ramp, not a one-day step. The bed median at 15 h is about 41-44° until 09-01,
  48-49° on 09-03 and 50-52° on 09-05.
  - W2 starts on 09-02 (47.7°).
  - W1 starts on 09-03 (48.4°) and reaches 52-55° by 09-06.
- It shows at every lit hour and in the next morning's runs.
- In a model with plant and size terms (`ln n_points`), the period from 09-03 adds +13.5° in W1
  and +11.6° in W2 over 08-25..09-02 (09-18 h runs, not dark, about 2,000 runs).

**Ruled out**

| Candidate | Evidence |
|---|---|
| Rig: sensor height | The fitted soil level is flat (W1 466-471 mm, W2 501-503 mm). |
| Rig: robot positions | Frame-7 y varies by less than 1 cm per plant; x is unchanged. |
| Rig: sweep timing | The span (79 s) and the column order are the same. |
| Rig: TIFF / ToF settings | The tags are identical. |
| Rig: ToF signal | The NIR level, the invalid share and the 5 x 5 high-frequency depth noise have no step at 09-03. |
| Reference plane | The inclination against the camera axis gives the same values (48.4° against 48.4° on W1 09-03). The step does not follow fit→carried switches. |
| Normal estimation | Dropping depth-edge points (3 x 3 range > 15 mm, a constant 20-24 % of points) and using normals with r = 20 mm / 60 neighbours both keep the step. |
| Vegetation mask | The height-only rule instead of ExG > 5 gives the same trajectory (±0.5°). 88-93 % of coloured points above 15 mm pass ExG, with no drop. |
| Pipeline at all | The raw ToF depth alone (5 x 5 median, pixels at least 15 mm above the soil, frame 7) shows the median surface slope rising from 43-48° to 51-52° (09-03) and 57-59° (09-05). The flat share (< 20°) falls from 0.14 to 0.07. |
| VPD at run time | It does not explain the shift: the effect stays with VPD in the model, and the level stays high on low-VPD days such as 09-09. |

**Coinciding events** (`metadata_rwc_par_en.xlsx`, sheet "Raw Protocol")
- 2.9.25: "Doors nearest plants closed to prevent wind effect", "Leaf sensors attached" (12
  plants only), and "all plants showing start of nutrient deficiency, nitrogen".
- The daytime climate changes on the same day: 24 °C / 60-68 % RH on 08-31..09-01, then
  27-33 °C / 36-48 % RH from 09-02.
- Leaves look more upright and wavy in the RGB from 09-03 (W1_J5, W1_R1, W2_G2).

**What to do with it**
- The shift is common to both beds, so the between-bed contrast (the day x treatment
  interaction) is still valid.
- Within-bed changes against a 09-01/09-02 baseline are not a treatment effect. For
  `inclination_mean` and `inclination_sd`, test the bed difference, or use a date effect common
  to both beds.
- No QC flag was added: the data are correct, and a flag would remove the whole treatment
  period.

## Deviations from the plan

| Plan | Here | Why |
|---|---|---|
| ADA robot → depth rotz(+90°) | robot x → depth −y | cross-column agreement: 6-7 mm instead of 14-17 mm (table above) |
| 15 repeat views, median and MAD over 15, at least 8 ok | median and MAD over the frames that cover the plant cylinder (median 6), at least 2 ok | the frames are a 3 x 5 grid; most of them do not see the whole plant |
| plant centre = image centre | frame-7 RGB optical axis on the soil, refined within 4 cm | the ToF centre lies at a pot corner (28 px off); the RGB axis lands on the pot centre |
| sensor height 0.60 m | soil plane per run, or carried | the soil lies 0.45-0.52 m below the sensor |
| no hole filling in `traits` mode | enclosed holes of up to 150 px are inpainted before the median | ToF saturates on leaf interiors; only the rims have depth |
| soil window 6 cm, tilt < 10° | 5 cm, tilt ≤ 15°, near-horizontal RANSAC hypotheses, prior check of 20 mm | pots are about 11 cm wide; edge-row pots are tilted 9.5-13°; a sloping leaf otherwise beats a small soil patch |
| carried reference = median of the plant's accepted runs | median of the accepted runs on the plant's first 7 dates, all hours (`plant_refs.csv`) | the reference must exist before the batch starts, so the batch stays resumable and parallel |
| dark if brightness < 40 | < 10 | 40 flags about half of all 15 h frames; ExG works down to about 8 (`diagnostics/dark_sheet.jpg`) |
| DBSCAN per frame | one DBSCAN over the voxelised union of the used frames | per-frame clustering kept different leaves in different frames (`height_p75` MAD up to 28 mm) |
| `projected_area` = Σ z²/(fx·fy) | foreshortening-corrected footprint on the reference plane | E0c: −9 % at 35° leaf tilt otherwise |
| `qc_high_spread` at 5 mm | max(5 mm, 5 % of `height_p75`) | the viewpoints differ by up to 0.1 m |
| Open3D normals | scipy KD-tree PCA, same neighbourhood | no Open3D import or thread pool in 20 worker processes |

## Refined depth (PromptDA)

`--depth-model PATH` replaces the ToF depth of every frame the pipeline builds with the
fine-tuned Prompt Depth Anything map (`runs/promptda-vitl/best`, ViT-L, bf16 on the GPU; read
through `uiwue.ml.promptda.infer.DepthRefiner`). Everything after the depth, from the reference
plane to QC, is unchanged. Outputs go to a separate directory
(default `data/derived/threed_refined/`); the CLI refuses to write refined results into
`data/derived/threed/`.

```bash
python -m uiwue.threed.run --data-dir data --depth-model runs/promptda-vitl/best --hours 15 --jobs 20 \
    --out data/derived/threed_refined
python -m uiwue.threed.run --data-dir data --depth-model runs/promptda-vitl/best \
    --diagnose W2_J5:2025-09-14:15        # diagnostics PNG with refined depth
```

### Method

1. **GPU pre-pass** (`run.precompute_refined`, resumable):
   - Frames are selected exactly as in the raw path (`process_run(plan_only=True)`).
   - `DepthRefiner.refine_frame` predicts a 518 x 700 map on the undistorted RGB crop.
     `points_in_depth_frame()` turns it into about 360k points in the ToF frame of that frame.
   - `cloud.points_to_depth_grid` resamples them onto the ToF pixel grid, and the result is cached
     in `refined_depth/<plant>/<date>_<HH>_<k>.npz`.
2. **CPU pass** (20 workers): `run.RefinedDepth` is the `depth_fn`.
   - Pixels with a refined sample take the refined depth.
   - The other pixels keep the ToF after hole filling and the 3 x 3 median. These are the ~27 % of
     the ToF view without RGB, which mostly lie outside the plant cylinder of the frames used.
   - The refined map is used as is, without filling or median.
   - `refined_frac` (frame and run columns) is the share of plant points with refined depth:
     median 0.99, 5th percentile 0.91.

**Downsampling.** I used one sample per ToF pixel instead of a 3D voxel grid: the points that
project into a pixel (with the lens distortion) are z-buffered (10 mm tolerance) and averaged.
- At the ToF spacing this equals a 2.2 mm grid at the soil and 1.4 mm at canopy tops (0.30 m).
- So the parameters tuned on ToF density stay valid unchanged: DBSCAN (eps 8 mm,
  min_samples 10), normals (12 mm, 30 neighbours), per-pixel areas and the 300-point QC rule.
- The z-buffer drops surfaces the RGB camera sees but the ToF camera does not, such as soil
  under a leaf edge, so each ray keeps one surface.
- E0c through this path passes: dense exact points at 3x the ToF density, resampled, give every
  trait within 5 % (`test_e0c_refined_depth_path_within_5_percent`).
- E0c cannot test the model itself, because it needs real RGB.

### Comparison

Subset: Tier-1 plants (W1/W2 A2, A8, J5, R1, R7), 15 h, 2025-09-01..09-14, 140 runs.
- Raw ToF was rerun with the same code into `data/derived/threed_refined/compare_raw/`; it is
  identical to the contract table for these runs.
- **Noise** is the median frame-to-frame MAD over the 139 runs with at least 2 ok frames in both
  versions.
- **Signal** is \|median W1 − median W2\| on 09-14, 5 plants per bed, all runs.
- **SNR** = signal / noise (the E1 definition).
- **Bias** is the median per-run difference.
- Files: `comparison.csv`, `comparison_qc.csv`.

| Trait | Noise raw | Noise refined | Noise change | Signal raw | Signal refined | SNR raw | SNR refined | Bias, refined − raw |
|---|---|---|---|---|---|---|---|---|
| `height_p50` (mm) | 2.15 | 1.49 | -31 % | 118 | 101 | 54.9 | 67.9 | 0.53 |
| `height_p75` (mm) | 2.67 | 1.91 | -28 % | 118 | 95.97 | 44.3 | 50.3 | 0.07 |
| `height_p95` (mm) | 3.37 | 2.58 | -23 % | 122 | 116 | 36.4 | 44.8 | -1.51 |
| `projected_area` (cm²) | 3.62 | 3.28 | -9 % | 81.26 | 98.54 | 22.4 | 30.0 | 6.63 |
| `leaf_area_est` (cm²) | 7.13 | 7.13 | +0 % | 204 | 248 | 28.6 | 34.8 | 16.74 |
| `hull_volume` (cm³) | 59.01 | 54.24 | -8 % | 2853 | 3084 | 48.3 | 56.9 | -56.32 |
| `canopy_radius` (mm) | 0.24 | 0.22 | -8 % | 3.33 | 1.17 | 13.9 | 5.3 | 0.03 |
| `inclination_mean` (°) | 1.09 | 1.07 | -2 % | 9.57 | 13.15 | 8.8 | 12.3 | -0.21 |
| `inclination_sd` (°) | 0.46 | 0.44 | -4 % | 1.16 | 1.86 | 2.5 | 4.2 | -0.43 |
| `droop_index` | 0.018 | 0.015 | -17 % | 0.159 | 0.159 | 8.9 | 10.7 | 0.012 |
| `compactness` | 0.018 | 0.019 | +4 % | 0.343 | 0.368 | 19.0 | 19.7 | 0.043 |
| `n_points` | 155 | 147 | -5 % | 5182 | 5509 | 33.4 | 37.5 | 203 |
| `cyl_height_p75` (mm) | 2.53 | 1.85 | -27 % | 104 | 98.29 | 40.9 | 53.2 | -0.52 |
| `cyl_projected_area` (cm²) | 5.26 | 4.29 | -18 % | 106 | 123 | 20.0 | 28.8 | 7.81 |

| | Raw ToF | Refined |
|---|---|---|
| `qc_pass` | 0.686 | 0.700 |
| `qc_high_spread` | 0.150 | 0.114 |
| `qc_overlap` | 0.179 | 0.200 |
| `qc_not_centred` | 0.029 | 0.064 |
| `qc_no_plant` | 0.007 | 0.000 |
| `qc_ref_carried` | 0.536 | 0.529 |
| GPU per frame (model incl. image I/O + resampling) | - | 0.070 + 0.044 s |
| CPU per built frame | 0.13 s | 0.18 s |
| Wall time, 140 runs (666 frames) | 26 s | 120 s (81 s GPU) |

What the numbers say:
- **Noise:** refined depth lowers the frame-to-frame noise of the height traits by 23-31 %, of
  `droop_index` by 17 %, of `cyl_projected_area` by 18 % and of `projected_area` by 9 %. It
  leaves the noise of `inclination_mean`, `inclination_sd` and `leaf_area_est` unchanged
  (−4 % to 0 %). So the inclination traits stay as noisy as before.
- **SNR:** it rises for every trait except `canopy_radius`. For the at-risk traits the rise
  (×1.4 `inclination_mean`, ×1.7 `inclination_sd`, ×1.2 `leaf_area_est`, ×1.2 `droop_index`)
  comes from a larger W1−W2 difference, not from lower noise. There is no ground truth to tell
  which version is closer to the true difference.
- **The 09-14 signal rests on one run per plant.** For heights it is lower with refined
  (118 → 96 mm for p75).
  - Most of this is W2_J5: 56 mm raw, 92 mm refined. Both versions flag that run `qc_overlap`;
    DBSCAN keeps a different overlapping leaf in each.
  - With runs that pass QC in both versions, averaged over 09-12..09-14, the `height_p75`
    separation is the same (88.8 vs 89.4 mm). The `projected_area` separation rises (74 → 88 cm²)
    and so does `droop_index` (0.07 → 0.09).
- **Biases:**
  - On the tall W1 canopy after 09-08, refined `height_p75` is 3.8 mm lower and `height_p95`
    5.9 mm lower; W2 is 1.7 mm higher.
  - With σ ≈ 4 mm ToF noise, p95 is inflated by about +6 mm on a flat surface, so this is at
    least partly denoising. Some of it may be smoothing at leaf edges: the refined map
    interpolates across depth jumps ("curtains" in the side views).
  - `projected_area` is 6.6 cm² (+7 %) higher and `leaf_area_est` 16.7 cm² (+9 %) higher. This
    comes from leaf interiors that the ToF loses to saturation and to the fill limit.

### Recommendation

- Use refined depth for the height traits (`height_p50/75/95`, `cyl_height_*`),
  `projected_area`, `cyl_projected_area` and `droop_index`: lower noise floor, fewer
  `qc_high_spread` runs (15.0 → 11.4 %), same or better separation.
- Keep raw ToF as the reference for `inclination_mean`, `inclination_sd` and `leaf_area_est`.
  Refined depth does not lower their noise, and their larger bed difference cannot be validated.
- Never mix raw and refined runs in one analysis, because of the biases above.
- Before switching, run the refined batch on all runs (about 36k frames, roughly 70 min GPU plus
  5 min CPU) and let E1/E3 run on both tables.

## Not done or open

- `temp` (thermal) is not in the cached clouds; it is optional in the contract.
- Refined depth is available with `--depth-model` (next section); it is not used for the
  contract table. The learned plant mask (`process_run(plant_mask_fn=...)`) is not wired yet; it
  waits for the Mask2Former v2 checkpoint.
- Isolation on the closed canopy is approximate (plan section 10). 22-29 % of treatment-window
  runs are `qc_overlap`. The `cyl_*` columns are the cluster-free alternative. Both should be
  compared in E1/E3.
- Per-run figures come from `--diagnose` (`run.render_diagnostics`). The 3D cloud views, the
  dark-frame sheet and the frame-7 centring figure in `diagnostics/` were made with one-off
  scripts.
