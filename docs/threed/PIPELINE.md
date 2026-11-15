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
python -m uiwue.threed.run --data-dir data --plant-model runs/leafseg-m2f/best --jobs 20     # learned mask (GPU)
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

Tests (91 in `tests/threed`, including the stats tests): `.venv/Scripts/python.exe -m pytest -q tests/threed`.

## Geometry facts, checked on real data (plan section 2)

| Item | Plan estimate | Measured |
|---|---|---|
| Sensor to soil | 0.60 m | 0.453-0.522 m (1st-99th percentile of the fitted soil level over all runs, median 0.487 m). W1 pots are at about 0.46-0.48 m, W2 pots at about 0.49-0.52 m. |
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
   - Enclosed invalid regions of up to 150 px whose rim spans at most 25 mm are inpainted from
     their rim (`fill_small_holes`, Telea); 1.1 % of plant points come from filled holes.
     - The rim is the valid 8-neighbours of the hole (`cloud.rim_span`, max − min depth).
     - Most small holes are dropouts at depth edges (leaf over soil, leaf over leaf), not
       saturated leaf interiors. On frame 7 of 244 runs (12 and 15 h, not dark), the median
       filled pixel lies in a hole whose rim spans 90 mm, and 88 % lie in holes above 30 mm.
       Filling those bridges the edge with a steep fake surface.
     - 25 mm covers the noise range of a flat rim (8-30 pixels at σ ≈ 4 mm span about
       11-16 mm) plus a leaf tilted by about 45° (1.6 mm per pixel at 0.35 m). It keeps 9 % of
       the pixels a fill without the limit would fill (19 % of the holes), and bounds the error
       of every filled value.
     - ExG does not separate the two kinds: on the canopy, the filled points of holes below
       60 mm span are 56-62 % green at any span, against 57 % for all canopy points.
   - The inpainting runs on a uint16 copy in 0.1 mm units. With the locked OpenCV 5.0, Telea on
     float32 returns values metres outside the rim. On uint16 (without the span limit), 0.1-0.9 %
     of the filled pixels of four checked frame-7 images lie more than 10 mm outside the valid
     range of their 7 x 7 window, and none more than 100 mm (float32: about 99 %).
   - Masked 3 x 3 median. Pixels with fewer than 5 valid pixels in their window are dropped.
   - Valid depth is 0.20-0.80 m. Each pixel is unprojected along its own undistorted ray.
   - Colour comes from the raw RGB at the distorted projection of each point (shifted extrinsic,
     corrected robot mapping). A z-buffer (8 px cells, 10 mm tolerance) marks points hidden from
     the RGB camera as uncoloured.
3. **Plant centre**: the frame-7 RGB optical axis meets the reference plane. It is refined once
   to the centroid of vegetation within 4 cm in frame 7 (median shift 8.6 mm, p95 27.4 mm). In
   frame k the centre is `c7 − (M·p_robot[k] − M·p_robot[7])`.
4. **Frame selection**: a frame is used when at least 98 % of the plant cylinder footprint
   (r = 8.5 cm on the reference plane) lies inside the valid ToF view and at least 95 % inside the
   RGB view. This depends only on the poses and the plane, not on the image content.
   - Typical sets are 4,7,10 (W1, camera about 0.46 m above the soil) and 0,1,3,4,6,7,9,10 (W2,
     about 0.50 m, wider view). The median is 6 frames per run. The set is in `frames_used`.
   - Cross-column frames are as consistent as same-column ones. The median frame-pair difference
     of `height_p75` is 4.1 mm within the centre column and 4.2 mm between the centre and a side
     column (3,007 runs). So all covered frames are used, not only the centre column.
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
   - Results: 4,179 runs use their own fit and 1,791 the carried reference. In 09-01..09-14,
     73 % of W1 runs and 48 % of W2 runs are carried (wilting W2 plants uncover the soil).
   - Over a plant's own fits the soil level varies by 3.9 mm SD (median plant).
6. **Vegetation** (`segment.vegetation_mask`):
   - Normal light: ExG > 5 and height > 5 mm.
   - Dark frame (mean sampled RGB < 10, plan start value 40), or a point without colour:
     height > 15 mm only.
   - With `--plant-model`, points with colour use the Mask2Former centre-plant mask instead of
     ExG (mask and height > 5 mm); points without colour keep height > 15 mm. Dark frames and
     frames without an instance at the centre keep the rule (section "Learned plant mask").
   - The soil candidates of step 5 always use the ExG rule.
7. **Isolation** (`segment.isolate_plant_multi`):
   - Vegetation of all used frames is pooled in the plant-centred frame on a 2.5 mm voxel grid.
   - DBSCAN with eps 8 mm and min_samples 10. Clusters whose nearest point is within 3 cm of the
     centre are kept.
   - The result is clipped to a cylinder of r = 8.5 cm and mapped back to each frame, so every
     frame keeps the same plant.
   - `clipped_fraction` is the share of kept points within 5 mm of the cylinder wall.
   - With the learned mask, the frames first vote per voxel (`votes=`, section "Learned plant
     mask"), then the same DBSCAN and cylinder run.
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
`centroid_offset_mm`, `brightness`, `mask_source` (`rule`, `model` or `mixed`) and
`mask_model_frac` (share of the used frames with the learned mask). Extra frame columns: `used`,
`built`, `cov_depth`, `cov_rgb`, `robot_dx`, `robot_dy`, `no_plant`, `n_clusters`, `dark`,
`n_veg`, `n_filled`, `mask_source`, `mask_instances`, `mask_score`, `mask_merged`, and the same
QC inputs. Without `--plant-model`, `mask_source` is `rule` everywhere.

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
- Real numbers, 15 h runs that pass QC, bed medians of `height_p75`: W1 rises from 93 mm (09-01)
  to 174 mm (09-14). W2 peaks at 124 mm (09-07) and falls to 69 mm (09-14). `projected_area` in W2
  falls from 123 to 39 cm².
- Hole-fill variants on the same 5,970 runs (bed medians over all runs; E1 noise is the median
  frame-to-frame MAD of `qc_pass` runs, `experiments.e1`):

  | Variant | Plant points filled | W1 / W2 `leaf_area_est` | W1 / W2 `projected_area` | W1 / W2 `height_p95` | E1 noise `leaf_area_est` / `projected_area` / `height_p95` |
  |---|---|---|---|---|---|
  | float32 Telea (before) | 2.8 % | 151.7 / 80.6 cm² | 82.9 / 42.3 cm² | 121.1 / 85.9 mm | 5.48 cm² / 3.08 cm² / 2.65 mm |
  | uint16, no span limit | 5.9 % | 161.8 / 90.2 cm² | 84.9 / 45.8 cm² | 122.8 / 85.2 mm | 5.53 / 2.97 / 2.70 |
  | uint16, span ≤ 25 mm (used) | 1.1 % | 147.9 / 76.9 cm² | 81.1 / 40.4 cm² | 122.4 / 85.3 mm | 5.36 / 3.02 / 2.66 |
  | no hole filling | 0 | 145.1 / 76.0 cm² | 79.7 / 40.1 cm² | 122.4 / 85.2 mm | 5.54 / 3.26 / 2.66 |

  Against no filling, the median run changes by +0.8 % in `leaf_area_est` and +1.0 % in
  `projected_area` with the span limit, and by +11.9 % and +7.9 % without it (runs that pass QC
  in both). Heights change by less than 0.3 % in every variant.
- Typical frame-to-frame spread (median MAD, 15 h runs that pass QC, 09-01..09-14; relative is
  the median MAD over the median trait):

  | Trait | Median MAD | Relative |
  |---|---|---|
  | `height_p75` | 2.8 mm | 2.4 % |
  | `projected_area` | 3.9 cm² | 4.7 % |
  | `inclination_mean` | 1.1° | |
  | `droop_index` | 0.02 | |

## Batch (2026-11-15)

- 5,970 runs (all 28 plants, 2025-08-12..09-19, all hours) finished with status `ok`, 0 errors.
- 79,667 frame rows, 5,970 cached clouds (1.2 GB).
- One core takes 0.48 s per run (median), about 0.08 s per built frame. With 20 workers the
  batch takes about 3 minutes (0.03 s per run wall time).
- Re-run with the hole-fill changes (uint16 inpainting and the rim-span limit, above). The disk
  now holds 12 more plants (W3 and W1_B7, W1_F5, W1_N3), so `data/derived/threed/` has 8,562 runs
  of 40 plants, all `ok`. The previous outputs are kept in `data/derived/threed_float32_backup/`.
  The numbers in this document are on the 5,970 runs of the 28 plants above.
- Cached-cloud points more than 150 mm below the reference: 1,910 in 24 clouds (float32 Telea:
  35,794 in 4,677; no hole filling: 1,972 in 21). The edge pots W2_A2, W2_A6 and W2_A8 have
  another 182,469 in 606 clouds: the floor beyond the bed edge (179,878 without hole filling).

QC rates per slot, all dates (n = 1438, 1137, 1120, 1177, 1098 for 06, 09, 12, 15, 18 h):

| slot | no_plant | few_points | high_spread | overlap | not_centred | dark | ref_carried | no_pose | pass |
|---|---|---|---|---|---|---|---|---|---|
| 06 | 0.261 | 0.426 | 0.097 | 0.117 | 0.124 | 0.607 | 0.323 | 0.000 | 0.396 |
| 09 | 0.082 | 0.336 | 0.118 | 0.139 | 0.142 | 0.103 | 0.277 | 0.025 | 0.434 |
| 12 | 0.054 | 0.311 | 0.127 | 0.139 | 0.124 | 0.000 | 0.304 | 0.025 | 0.448 |
| 15 | 0.048 | 0.300 | 0.122 | 0.153 | 0.127 | 0.059 | 0.320 | 0.014 | 0.460 |
| 18 | 0.044 | 0.311 | 0.117 | 0.123 | 0.124 | 0.246 | 0.268 | 0.000 | 0.478 |

QC rates per slot, 2025-09-01..09-14:

| slot | no_plant | few_points | high_spread | overlap | not_centred | dark | ref_carried | pass |
|---|---|---|---|---|---|---|---|---|
| 06 | 0.006 | 0.046 | 0.153 | 0.236 | 0.120 | 1.000 | 0.582 | 0.629 |
| 09 | 0.003 | 0.036 | 0.224 | 0.258 | 0.092 | 0.194 | 0.566 | 0.536 |
| 12 | 0.003 | 0.019 | 0.253 | 0.211 | 0.061 | 0.000 | 0.619 | 0.563 |
| 15 | 0.004 | 0.022 | 0.187 | 0.240 | 0.046 | 0.057 | 0.606 | 0.588 |
| 18 | 0.002 | 0.033 | 0.195 | 0.202 | 0.066 | 0.413 | 0.479 | 0.610 |

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
| no hole filling in `traits` mode | enclosed holes of up to 150 px whose rim spans at most 25 mm are inpainted before the median | ToF saturates on leaf interiors; only the rims have depth. Holes with a wider rim lie on depth edges and stay invalid |
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

## Learned plant mask (Mask2Former)

`--plant-model PATH` replaces the ExG rule with the Mask2Former centre-plant mask
(`runs/leafseg-m2f/best`, v2 step 8,000, PhenoBench plant AP 0.54 / AP50 0.90; model details
in [`docs/ML_MODELS.md`](../ML_MODELS.md), section 2). Outputs go to a separate directory (default
`data/derived/threed_plantmask/`); the CLI refuses `data/derived/threed/`.

```bash
python -m uiwue.threed.run --data-dir data --plant-model runs/leafseg-m2f/best --jobs 20 \
    --out data/derived/threed_plantmask
python -m uiwue.threed.run --data-dir data --plant-model runs/leafseg-m2f/best \
    --out data/derived/threed_plantmask --diagnose W1_J5:2025-09-03:15     # figure with the mask
python scripts/compare_plant_mask.py --new data/derived/threed_plantmask --base data/derived/threed \
    --plants-from data/derived/threed/results/summary.md --out data/derived/threed_plantmask/compare
python scripts/plant_mask_agreement.py --masks data/derived/threed_plantmask \
    --refs data/derived/threed_plantmask/plant_refs.csv --out data/derived/threed_plantmask/compare
```

### Method

1. **GPU pre-pass** (`plantmask.precompute_masks`, resumable): the frames are selected exactly
   as in the ExG path (`run.plan_frames`); runs without poses have no colour and are skipped.
   - Each raw, distorted RGB frame goes into the model at 0.5x. `infer.instance_masks` keeps the
     plant instances with score ≥ 0.5 (the same masks and scores as transformers'
     post-processing, 200x faster).
   - All plant instances are cached at 540 x 960, bit-packed, in
     `plant_masks/<plant>/<date>_<HH>_<k>.npz` (about 8 kB per frame). A frame without an
     instance gets an empty file.
2. **Mask on the points** (`plantmask.PlantMasks`, the `plant_mask_fn`):
   - Each point with colour reads the instances at its RGB pixel (`Cloud.rgb_uv`, full-resolution
     distorted pixels, halved with the pixel-centre convention).
   - The centre plant is the instance with the most points within 3 cm of the plant axis and
     more than 5 mm above the reference (`select_centre`, the DBSCAN attach rule applied to
     instances).
   - Other instances with at least half of their points inside the plant cylinder are merged in
     (`merge_fragments`): the model splits large W1 plants into several instances.
   - If an instance that is not merged has at least 80 % as many points near the axis, two plants'
     leaves cover the pot centre and the frame keeps the rule (`rule_ambiguous`).
   - Vegetation is then mask and height > 5 mm on points with colour, height > 15 mm on points
     without colour (the mask cannot be read there).
3. **Fallback to the ExG / height rule**, per frame, recorded in `mask_source`:
   `rule_dark` (brightness < 10: the model finds no plant on the 06 h frames),
   `rule_no_instance`, `rule_ambiguous`, `rule_no_colour` (no pose), `rule_no_cache`. A run
   whose used frames all fall back goes through the ExG path unchanged, bit for bit.
4. **Reference plane**: unchanged, from the ExG soil candidates. A centre-plant mask would make
   the neighbours' leaves over the pot soil candidates. So the plane and the frame set are the
   same as in the ExG batch. The centre refinement (step 3 of the method above) uses the learned
   mask, so the centre moves by a median of 0.2 mm (p90 4.6 mm); everything else that differs
   comes from the mask.
5. **Isolation** (`run._isolate`, `segment.isolate_plant_multi(votes=...)`):
   - The candidates of all used frames (points with colour above 5 mm, points without colour
     above 15 mm) are pooled on the 2.5 mm voxel grid.
   - Every frame votes per voxel with its mask (the mean over its points there). Frames that
     fell back vote with the ExG rule; points without colour do not vote.
   - A voxel is plant when at least half of the frame votes in its 3 x 3 x 3 neighbourhood say
     so, or when nobody voted there. The neighbourhood is needed because at ToF density a
     2.5 mm voxel holds 1-3 points per frame and so sees only some of the frames.
   - Then the usual DBSCAN (eps 8 mm, min_samples 10, attach within 3 cm) and the 8.5 cm
     cylinder. The kept voxels go back to every frame.
   - Frame columns `mask_source`, `mask_instances`, `mask_score`, `mask_merged`; run columns
     `mask_source` (`model` / `mixed` / `rule`) and `mask_model_frac`.

### Why DBSCAN still runs on the learned mask

Ablation on the Tier-1 plants (W1/W2 A2, A8, J5, R1, R7), 15 h, 2025-09-01..09-14: the 135 runs
with at least one model frame, same code and reference as `data/derived/threed`. Noise is the
median frame-to-frame MAD of the run (frames with at least 300 points, runs with at least 2 of
them).

| Isolation | Noise `height_p75` (mm) | Noise `projected_area` (cm²) | Noise `leaf_area_est` (cm²) | Noise `inclination_mean` (°) | `projected_area` (cm²) | `qc_overlap` |
|---|---|---|---|---|---|---|
| ExG rule + DBSCAN (contract path) | 2.41 | 3.87 | 5.95 | 1.01 | 99 | 0.185 |
| mask per frame + DBSCAN | 2.72 | 4.45 | 9.66 | 1.12 | 76 | 0.067 |
| mask per frame, cylinder only | 2.85 | 4.42 | 11.01 | 1.20 | 87 | 0.104 |
| voxel vote, cylinder only | 2.54 | 3.92 | 7.46 | 1.10 | 87 | 0.074 |
| **voxel vote + DBSCAN (used)** | **2.40** | **3.81** | **7.67** | **1.07** | **79** | **0.052** |

- DBSCAN stays, for three reasons:
  - Without it, `qc_overlap` rises from 0.052 to 0.074. The noise rises by 6 % for
    `height_p75`, 3 % for `projected_area` and 3 % for `inclination_mean`; only `leaf_area_est`
    is 3 % lower.
  - Without it, the cylinder keeps every point without colour above 15 mm, connected to the
    plant or not, and every isolated mask point. With DBSCAN, points without colour are 5 % of
    the plant points, and they join only through 3D connectivity.
  - It keeps the isolation of the contract path, so the two tables differ only in the mask.
- The cost: DBSCAN removes a median of 6 % of the mask's points inside the cylinder (p90
  23 %). These are mask parts that do not connect in 3D to the plant at the centre.
- The voxel vote matters more than DBSCAN. Per-frame masks disagree between viewpoints: the
  instance boundary moves, and a leaf is in the instance in one frame and not in the next.
  Without the vote the noise is 13 % higher for `height_p75`, 17 % for area and 26 % for leaf
  area. Pooled votes give every frame the same voxels, as the ExG path does.
- Against ExG, the used variant has the same `height_p75` noise, 2 % less `projected_area` noise
  and under a third of the `qc_overlap`. Leaf area and inclination are noisier (+29 %, +6 %).

### Comparison

The batch (2026-11-15):

- 8,899 runs of 42 plants, all `ok`; W1_A4 and W1_A6 are new on disk.
- GPU: 53,415 frames segmented, 0.14 s per frame with the GPU shared, about 2.2 h. The cache
  `plant_masks/` takes 403 MB (about 7.5 kB per frame).
- CPU: 630 s on 20 workers.

The baseline is `data/derived/threed` (rim-span-limited fill). It was made with the same code:
without `--plant-model` the pipeline reproduces it bit for bit (220 runs and 2,724 frames
checked, 06, 09 and 15 h). The comparison uses the 5,970 runs of the 28 plants in `results/`
(`scripts/compare_plant_mask.py`, `scripts/plant_mask_agreement.py`; outputs in
`data/derived/threed_plantmask/compare/`).
- The raw data kept arriving during both batches. W1_B7, outside the 28 plants, has a different
  frame set in 68 runs: its RGB files were written after `data/derived/threed` processed it.
- `data/derived/threed` was later extended to more plants; its 5,970 rows of the 28 plants did
  not change (checked).

**QC rates.**

| Runs | n | `qc_pass` ExG | `qc_pass` model | `qc_overlap` ExG | `qc_overlap` model | `qc_not_centred` ExG | `qc_not_centred` model |
|---|---|---|---|---|---|---|---|
| all dates | 5,970 | 0.441 | 0.486 | 0.134 | 0.067 | 0.128 | 0.132 |
| 09-01..09-14 | 2,130 | 0.588 | 0.675 | 0.230 | 0.121 | 0.078 | 0.105 |
| 09-01..09-14, 06 h | 483 | 0.629 | 0.629 | 0.236 | 0.236 | 0.120 | 0.120 |
| 09-01..09-14, 09 h | 392 | 0.536 | 0.658 | 0.258 | 0.115 | 0.092 | 0.143 |
| 09-01..09-14, 12 h | 375 | 0.563 | 0.667 | 0.211 | 0.056 | 0.061 | 0.085 |
| 09-01..09-14, 15 h | 454 | 0.588 | 0.722 | 0.240 | 0.066 | 0.046 | 0.084 |
| 09-01..09-14, 18 h | 426 | 0.610 | 0.697 | 0.202 | 0.110 | 0.066 | 0.092 |

**Where the model is used.** Share of used frames per mask source, 2025-08-24..09-14:

| Hour | model | `rule_dark` | `rule_no_instance` | `rule_ambiguous` |
|---|---|---|---|---|
| 06 | 0.008 | 0.897 | 0.095 | 0.000 |
| 09 | 0.754 | 0.190 | 0.045 | 0.010 |
| 12 | 0.954 | 0.000 | 0.032 | 0.014 |
| 15 | 0.920 | 0.033 | 0.033 | 0.014 |
| 18 | 0.645 | 0.300 | 0.030 | 0.011 |

- Before 2025-08-24 the seedlings are too small: 39-49 % of the frames at 09-18 h have no
  instance with 10 points at the centre.
- Over all dates, 47 % of the runs are fully model, 17 % mixed and 36 % fully ExG.
- So the method changes with the hour. The 06 h runs are the ExG / height rule; 09 h and 18 h
  are mixed; 12 h and 15 h are mostly the model. A diurnal comparison (E4) on this table
  compares two methods.

**Bed medians, 15 h, `qc_pass` runs of each version.**

| Date | W1 `height_p75` ExG / model (mm) | W2 `height_p75` ExG / model (mm) | W1 `projected_area` ExG / model (cm²) | W2 `projected_area` ExG / model (cm²) |
|---|---|---|---|---|
| 09-01 | 92.6 / 102.0 | 82.8 / 83.8 | 99.0 / 96.1 | 122.9 / 69.5 |
| 09-03 | 134.1 / 137.0 | 100.9 / 105.2 | 118.1 / 81.1 | 120.8 / 94.4 |
| 09-05 | 147.2 / 145.4 | 121.8 / 110.2 | 113.1 / 86.0 | 122.4 / 91.6 |
| 09-07 | 156.0 / 140.9 | 124.1 / 120.3 | 127.7 / 83.9 | 101.0 / 72.2 |
| 09-09 | 150.5 / 149.0 | 107.5 / 105.7 | 103.6 / 103.6 | 63.1 / 56.0 |
| 09-11 | 174.6 / 174.4 | 111.7 / 116.5 | 120.0 / 92.5 | 49.9 / 38.2 |
| 09-13 | 171.2 / 171.8 | 72.7 / 76.0 | 104.7 / 86.3 | 42.1 / 30.8 |
| 09-14 | 174.2 / 171.9 | 69.4 / 72.4 | 119.3 / 94.8 | 38.7 / 25.0 |

All 14 days are in `compare/trajectories.csv`. On a run that passes QC in both versions,
`height_p75` is the same (median ratio 1.00-1.01 in both beds and every period). The bed medians
differ because other runs pass QC.

**E1 noise floor** (28 plants, `qc_pass` runs of each version; signal on 09-14):

| Trait | Noise ExG | Noise model | SNR ExG | SNR model |
|---|---|---|---|---|
| `height_p75` (mm) | 2.11 | 2.15 | 36.7 | 34.2 |
| `height_p95` (mm) | 2.66 | 2.52 | 36.0 | 41.8 |
| `projected_area` (cm²) | 3.02 | 2.76 | 21.9 | 24.8 |
| `leaf_area_est` (cm²) | 5.36 | 4.88 | 33.8 | 35.5 |
| `hull_volume` (cm³) | 41.9 | 36.0 | 51.8 | 68.5 |
| `canopy_radius` (mm) | 0.42 | 0.52 | 7.4 | 23.2 |
| `inclination_mean` (°) | 1.27 | 1.28 | 6.3 | 5.7 |
| `droop_index` | 0.019 | 0.018 | 5.4 | 3.1 |
| `compactness` | 0.023 | 0.024 | 13.8 | 6.1 |
| `inclination_sd` (°) | 0.65 | 0.64 | 2.0 | 0.4 |

- The same 11 traits are reliable (SNR ≥ 3) in both versions; `inclination_sd` is not in either.
- Noise is within ±10 % except `canopy_radius` (+23 %) and `hull_volume` (−14 %).
- The signals of `compactness` and `droop_index` halve, those of `canopy_radius` and
  `hull_volume` rise.

**E3 onset** (days since irrigation off, `qc_pass` runs of each version; ExG matches
`results/e3_onset.csv`):

| Trait | Onset ExG | Onset model | Slope ExG (%/day) | Slope model | Slope on runs passing in both: ExG / model |
|---|---|---|---|---|---|
| `n_points` | 6 | 6 | −10.5 | −13.1 | −10.4 / −15.4 |
| `leaf_area_est` | 6 | 6 | −8.4 | −10.3 | −8.3 / −12.7 |
| `hull_volume` | 7 | 7 | −11.6 | −14.1 | −11.5 / −15.6 |
| `projected_area` | – | 6 | −5.6 | −7.8 | −5.5 / −9.0 |
| `height_p75` | – | 7 | −5.6 | −6.0 | −5.6 / −5.9 |
| `height_p95` | – | 7 | −6.0 | −7.0 | −6.1 / −6.8 |
| `height_p50` | – | 10 | −6.8 | −6.5 | −6.7 / −6.7 |
| `compactness` | 10 | – | −3.7 | −3.0 | −3.6 / −3.4 |
| `droop_index` | – | – | −1.0 | −0.03 | |

- **Heights:** the onsets come from QC, not from the values.
  - With the model, 15 deficit plants have passing baseline runs instead of 9, so the daily
    tests have more power.
  - On the runs that pass in both versions the height slopes are equal, and neither version
    has a height onset.
- **Area-type traits:** the slopes steepen on the same runs, by 35-65 %. The model shrinks the
  deficit bed's area more than the control's as the treatment goes on.
  - `projected_area`, model over ExG (runs passing in both, 09-18 h): W1 0.87 at baseline
    (09-01..09-02) and 0.95 in 09-08..09-14; W2 0.96 and 0.88.
  - So against ExG, the model's W1 area rises about 9 % relative to its own baseline and its
    W2 area falls about 8 %.

**Wilted W2 against the rest** (`plant_mask_agreement.py`; 28 plants, 09-18 h; IoU of the
learned and the ExG plant on points with colour in the cylinder, model frames only):

| Bed | Period | Frames | IoU median | IoU p25 | Points only in ExG | Points only in model | Plant points, model / ExG | No instance at centre |
|---|---|---|---|---|---|---|---|---|
| W1 | 09-01..09-07 | 1,010 | 0.75 | 0.59 | 722 | 94 | 0.80 | 0.9 % |
| W1 | 09-08..09-14 | 861 | 0.77 | 0.62 | 555 | 119 | 0.84 | 1.0 % |
| W2 | 09-01..09-07 | 2,450 | 0.74 | 0.61 | 453 | 73 | 0.81 | 1.0 % |
| W2 | 09-08..09-14 | 2,365 | 0.71 | 0.55 | 208 | 49 | 0.83 | 4.0 % |

- What the model drops on wilted W2 (`diagnostics/wilted_w2_exg_vs_model.jpg`, six random W2
  frames at 12 h and 15 h, 09-12..09-14):
  - mostly thin petioles and the fringes of wilting leaves that run from the centre plant,
    which are likely parts of the plant itself;
  - sometimes a neighbour's leaf at the cylinder wall.
- This fits the model's limits: its mask logits are at 1/8 of the frame, and PhenoBench has no
  wilted plants.
- On the closed W1 canopy the dropped points are mostly neighbours' leaves over the pot
  (`diagnostics/mask_examples.jpg`).

### What this means

- `qc_overlap` falls by half and `qc_pass` rises from 0.59 to 0.68 in the treatment window.
  Heights do not change on a given run, and the E1 noise floor is about the same.
- The area-type treatment effect (`projected_area`, `leaf_area_est`, `hull_volume`,
  `n_points`) is 35-65 % steeper with the model. Part of that is likely a bias: the model
  segments the wilted deficit plants less completely. There is no ground truth on ADA to show
  how much.
- The model is not used at 06 h, and only partly at 09 h and 18 h.
- Recommendation:
  - Do not replace the contract table with this one.
  - Use the learned mask for `qc_overlap` and as a check on neighbour merging, and for heights
    if the larger set of passing runs is wanted.
  - Do not use it for area-type treatment effects or for diurnal comparisons until the masks
    have been checked on labelled wilted ADA frames: SAM 3 or hand masks on about 50 deficit
    frames after 09-08.

## Not done or open

- `temp` (thermal) is not in the cached clouds; it is optional in the contract.
- Refined depth is available with `--depth-model`, and the learned plant mask with
  `--plant-model` (sections above); neither is used for the contract table.
- Isolation on the closed canopy is approximate (plan section 10). 22-29 % of treatment-window
  runs are `qc_overlap` with ExG, fewer with the learned mask (section above). The `cyl_*`
  columns are the cluster-free alternative. Both should be compared in E1/E3.
- Per-run figures come from `--diagnose` (`run.render_diagnostics`). The 3D cloud views, the
  dark-frame sheet and the frame-7 centring figure in `diagnostics/` were made with one-off
  scripts.

## Team table (derived/threed)

`python -m uiwue.threed.contract` publishes the pipeline table in the team contract
(plan/Master-Plan.md, section 6) as `derived/threed/traits_runs.parquet` plus `catalogue.csv`.
It is committed to git, and `uiwue.core.load_traits()` reads it.

- `hour` is the scheduled slot. A re-run folder maps to the nearest slot at most 1 h away. When
  two folders map to one slot, the scheduled folder wins if it has data, otherwise the re-run with
  the most frames. The folder hour is kept as `depth_capture_hour`.
- Every column has the `depth_` prefix. Traits, their `_mad`, the `cyl_*` traits, the QC flags,
  and four diagnostics: soil distance, overlap fraction, off-centre distance, brightness.
- Dates run from 2025-08-24, the team scope.
- The catalogue gives a stress direction per trait from the E3 slopes. `inclination_sd` is 0,
  because it is below the noise floor.

On 2026-11-14 the table had 3,645 runs from 28 plants (2025-08-24 to 09-19, all five slots),
was 1.3 MB, and passed `python -m uiwue.core.validate`. Rerun the converter after every
pipeline batch.
