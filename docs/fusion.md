# Fusion model: running it on the GPU machine

The main track of Challenge 1 ([plan/Jonathan-Work-Plan.md](../plan/Jonathan-Work-Plan.md), section 5).
The code is in `src/uiwue/fusion/`:

| File | What it does |
|---|---|
| `inputs.py` | Loads one run per sensor from the raw ADA folders, crops and resizes to 224 x 224, applies fixed scalings, builds the Part B variants, augments for fine-tuning |
| `encoders.py` | DOFA ViT-B/16 for RGB and hyperspectral (TorchGeo, MAE weights); ImageBind's thermal and depth encoders; LoRA adapters |
| `embed.py` | Stage 1: frozen embeddings for every run and variant, to `cache/fusion/` |
| `labels.py` | Deficit label, severity stage S0-S3 (clip plants), turgor deviation |
| `fusion.py` | The fusion model: one token per sensor from [e(t), e(t) - e(baseline)], a [CLS] token, a 2-layer transformer, and heads for deficit, stage and turgor; modality dropout 0.3 |
| `train.py` | 5 folds grouped by plant; stage 1 (frozen) and stages 2-3 (LoRA); evaluation on every sensor subset; results in `derived/detect/fusion_*`; fold checkpoints in `runs/fusion/<model>/` |
| `predict.py` | External test: the 5 fold models (and placebo models) score plants outside training, averaged; `--oof` rescores the CV for plants without clips and plants with the size trait; `--lockbox` scores the v2 lock-box |
| `v2.py` | Architecture v2 (section 8): change-from-baseline and control-referenced tokens, the hand-made thermal token, the ablation harness (`ablate`), the chosen design (`train`), the benchmark on the lock-box (`bench-lockbox`) |
| `features.py` | The benchmark's thermal features for every plant (from folders or straight from the zips) and RGB green cover (`--rgb-size`) |
| `lockbox.py` | The 45 lock-box plants, fixed before any v2 experiment |

## 1. Install

```bash
pip install -e .
pip install -r requirements-ml.txt           # CUDA torch: add --torch-backend auto with uv
pip install --no-deps git+https://github.com/facebookresearch/ImageBind
```

ImageBind is installed without its dependencies because it pins old versions of torch.
The packages it needs are listed in `requirements-ml.txt`.

**Licences.**
- DOFA weights: see the TorchGeo weights page (`DOFABase16_Weights`); confirm the licence before release.
- ImageBind weights: CC BY-NC 4.0, so research use only. This is stated on the Methods & limits page.

## 2. Data

The fusion model reads raw images from `data/<sensor>/<bed>_<plant>/<YYYY_MM_DD>/<HH>/`. Set
`UIWUE_DATA` if they are somewhere else. RGB can stay zipped (`data/rgb/rgb_<bed>_<plant>.zip`).

```bash
python ../ada2025/utils/download_dataset.py --modalities thermal depth hsi rgb --plants <ids> --data-dir data -y
python ../ada2025/utils/unzip_data.py --modalities thermal depth hsi --data-dir data -y
```

- **Plants:** the 39 used in `derived/` (the 10 paper plants, the 21 clip plants and the extra
  W1 controls). Hyperspectral is about 1 GB per plant.
- **Hyperspectral calibration:** the flat-field correction uses ADA's
  `analysis/hyperspectral/black_reference.npy`. By default it is read from a clone of
  ada2025 next to this repo; set `UIWUE_HSI_DARK` to point elsewhere.

## 3. Run

```bash
# Stage 1a: embeddings (all sensors and Part B variants; --hsi-windows adds the 33 masked windows)
python -m uiwue.fusion.embed --data-dir data --hsi-windows

# Stage 1b: train the head (minutes on a GPU)
python -m uiwue.fusion.train

# Stage 2: LoRA on DOFA for hyperspectral (rank 8, last 4 blocks)
python -m uiwue.fusion.train --stage lora --lora-sensor hsi

# Stage 3: LoRA on the thermal encoder
python -m uiwue.fusion.train --stage lora --lora-sensor thermal

# External test: embed new plants (in batches if they arrive in batches), then score them
python -m uiwue.fusion.embed --plants <ids> --hsi-windows --cache cache/fusion_external/b1
python -m uiwue.fusion.predict --cache cache/fusion_external/b1 cache/fusion_external/b2 --model fusion
python -m uiwue.fusion.predict --cache ... --model fusion+lora_thermal --data-dir <raw thermal>

# CV subsets: plants without turgor clips, plants with the size trait
python -m uiwue.fusion.predict --oof --model fusion

# Check, then commit derived/detect/fusion_* (small) and push
python -m uiwue.core.validate
pytest tests/jack
```

**When to run stage 2:** only if stage 1 beats the benchmark's hyperspectral features on the
placebo-checked folds (plan section 5).

**Where the outputs go:**
- Embeddings: `cache/fusion/`. Checkpoints: `runs/fusion/`. Neither goes in git.
- ImageBind downloads its checkpoint to `.checkpoints/` on first use, also ignored.
- Results: `derived/detect/fusion_*`, which are pushed. The Sensor comparison page then shows
  the fusion rows beside the feature models.

## 4. What the outputs answer

| Table | Question |
|---|---|
| `fusion_benchmark` | Part B: every sensor subset through the one trained model, with low-resolution thermal swapped in; early AUC (days 1-4), full AUC, onset day, placebo |
| `fusion_auc_by_day` | Part A: AUC per day with plant-bootstrap intervals |
| `fusion_hsi` | Part B: spatial vs spectral detail (`s112`, `s64`), 3 vs 300 bands (`rgb3`), 20 nm windows (`mask<nm>`): change in early AUC against all bands |
| `fusion_severity` | Part C: do the stage and turgor heads rank the clip plants correctly within each day? |
| `fusion_scores_runs` | `detect_fusion_prob`, `_stage`, `_turgor` per run, out of fold, for the Plant profile page |
| `fusion_cv_subsets_benchmark`, `_auc_by_day` | The CV's out-of-fold scores for `no_clip` plants and `with_size` plants (beside size only on the same plants) |
| `fusion_external_benchmark`, `_auc_by_day`, `_hsi` | External test, same columns as the CV tables plus `plants` (`all`, `no_clip`, `with_size`) and `n_plants` |
| `fusion_external_scores_runs` | `detect_fusion_ext_prob`, `_stage`, `_turgor` per run of the external plants (own names, so both scores tables join) |
| `fusion_v2_ablation` | Section 8: every v2 variant, the v1 design, the benchmark and size on the same development plants, per sensor setup, group and plant subset, with the pre-cutoff canary days and the 2-Sep-baseline rows; one row per round |
| `fusion_lockbox_benchmark`, `_auc_by_day` | Section 8: the lock-box scored once by v2, the v1 design and the benchmark, all trained on the development plants |
| `fusion_v2_scores_runs` | `detect_fusion_v2_prob`, `_stage`, `_turgor` per run, out of fold, development plants |

## 5. Design choices to know

- **Folds and labels.**
  - Folds deal whole plants out within each bed.
  - Early stopping uses an inner split of the training plants.
  - The label is "deficit bed, after 3 Sep 11:00". Controls, and all plants before the cutoff,
    are negative.
- **Placebo checks.** Every setup is scored twice:
  - A placebo model is trained to tell the beds apart on 24-29 Aug. `bed_effect` is true when
    it can.
  - The main model's own score on those days is reported as `placebo_auc_main`.
  - If a sensor only carries a bed difference (lighting, pots), its early AUC is high for the
    wrong reason. These columns expose that, and the recommendation skips such setups.
- **Normalisation.**
  - Images are scaled with fixed physical constants, never with statistics of the data.
  - Embedding features are standardised with means and spreads from the fold's training plants.
- **Baseline.** Each run is compared with the plant's mean embedding over 30 Aug-2 Sep at the
  same hour, or any hour if that hour is missing.
- **How Part B variants are evaluated.** Low-resolution thermal and the hyperspectral variants
  are swapped into the trained model at evaluation, not retrained. They measure what the
  model loses when it gets the cheaper input.
- **LoRA stages.**
  - The encoder sees raw images with flips, 90° rotations, small crops and per-band gain
    jitter. Other sensors use their cached embeddings.
  - After each fold, every run is re-embedded with that fold's tuned encoder. The fold's
    held-out plants were never used to tune it.
  - Thermal uses its centre frame (frame 2) in every stage; see section 6.
- **Verified on real data and weights** on 2026-11-14 (section 6). The tests still use tiny
  random encoders (`tests/jack/test_fusion.py`).

## 6. Verification (2026-11-14)

Checked on the GPU machine with the real DOFA and ImageBind weights and the raw ADA files.
One line per fix (code in `src/uiwue/fusion/`, tests in `tests/jack/test_fusion.py`):

- **ImageBind import.** `imagebind/__init__.py` imports `imagebind.data`, which needs `pkg_resources` (gone from setuptools >= 81), `torchaudio` and `pytorchvideo`; `encoders._imagebind_model_module` now imports only the model code, which needs none of them.
- **ImageBind checkpoint.** It is loaded from `.checkpoints/imagebind_huge.pth` in the repo (or `UIWUE_IMAGEBIND_CKPT`) with `strict=True`, not from the working directory, so a run started elsewhere does not download 4.8 GB again.
- **DOFA under bf16.** The wavelength-to-kernel generator ran in bf16 under autocast (30 % median error per generated band kernel); `DOFAEncoder.forward` now runs the patch embedding in float32 and the blocks in bf16.
- **Thermal frames.** The 5 thermal frames are an 8 cm sweep: the plant sits at 26-80 % of the image width from frame 0 to 4 (ADA calibration), so only the centre frame 2 is embedded in every stage, as RGB and depth use their centre frame 7.
- **Disparity scaling.** `(1 / depth - 2.5) / 0.75` replaces `(1 / depth - 1.6) / 0.6`: soil (0.45-0.52 m) maps to about -0.7 and the tallest canopy (0.19 m) to +3.6, inside the range of ImageBind's standardised SUN RGB-D disparity; the old constants put every pixel at +0.5 to +7.
- **Hyperspectral hours.** Only the 9, 12 and 15 h cubes are used (Master Plan 6.5); the 06 h cubes are dark and flat-field to values up to 1e8.
- **Hyperspectral guards.** A dark or saturated cube (p99 > 6) fails its run instead of being embedded, reflectance is clipped to 0-2 before scaling, and a missing dark reference raises instead of silently skipping the flat-field correction; the default path now also looks in `external/ada2025`.
- **Hyperspectral memory.** The `rgb3` and `mask<nm>` variants are cut from the full input in `embed.py` (band selection commutes with the resize), and HSI loader batches hold a quarter of `--batch` runs; with `--hsi-windows` the loader previously held about 2 GB per run.
- **Cutoff day.** The deficit runs at 12, 15 and 18 h on 3 Sep are still positive and scored as day 0, but are left out of training: 1-7 h after the last watering only bed cues could separate them.
- **Capacity.** `embed.py` writes the peak GPU memory per sensor into the manifest.
- **LoRA on the GPU.** `LoRALinear` now creates its adapters on the base layer's device and dtype; `LoRAStage` adds LoRA after moving the encoder to the GPU, so stages 2-3 failed there with "mat2 is on cpu" (the CPU tests could not see it).
- **LoRA augmentation.** `_LiveImages` draws flips, crops and the baseline run from a generator per (epoch, item); the one generator made in `__init__` was copied into every DataLoader worker, so workers repeated the same augmentations.
- **Requirements.** `requirements-ml.txt` says which packages the ImageBind model code needs; `setuptools<80`, `torchaudio` and `pytorchvideo` are not needed.

Checked and found correct: the hyperspectral reader equals ADA's `HSI.read(..., dark_reference=...)`
exactly (no Savitzky-Golay smoothing here), wavelengths go to DOFA in µm from `hsi_000.json`,
depth and thermal are rotated by 180° as in ADA's loaders, the RGB centre crop holds the target
plant, folds and early stopping are grouped by plant, and scoring reuses `uiwue.stats`.

## 7. Results (2026-11-15)

Section 8 has the v2 architecture, chosen and trained on 171 plants with a lock-box.

Trained on the 39 plants with all four sensors (26 deficit, 13 control); 5 folds of whole
plants, stratified by bed. Every number below is out of fold. Early AUC is days 1-4 after
the cutoff, plant means of the 12 and 15 h runs, all deficit plants against W1, with a
95 % plant-bootstrap interval. Onset is the first day whose lower bound is above 0.70 on
that day and the next.

### Commands and timings

| Step | Where | Command | Time |
|---|---|---|---|
| Embeddings, 39 plants | local RTX 5070 Ti, 8 workers | `python -m uiwue.fusion.embed --plants <39> --hsi-windows --workers 8` | 23 min (hsi 20 min, 3307 runs x 37 variants, 38 failed the dark/unreadable checks; rgb, thermal, depth 1 min each) |
| Stage 1 (frozen) | GPU server | `python -m uiwue.fusion.train` | 38 s |
| Stage 3 (LoRA thermal) | GPU server, 6 workers | `python -m uiwue.fusion.train --stage lora --lora-sensor thermal --workers 6` | 454 s |
| No-clip and size subsets of the CV | local CPU | `python -m uiwue.fusion.predict --oof --model <model>` | 20 s |

The server needs `cache/fusion`, `derived/` and, for stage 3, the thermal frame-2 TIFFs and
`frames_thermal.csv` of the 39 plants (2.5 GB) and the ImageBind checkpoint. Logs and the
run script are in `runs/logs/`.

### Stage 1 and stage 3 against the benchmark and size

| Setup | Fusion, frozen (stage 1) | Fusion + LoRA thermal (stage 3) | Benchmark (logistic) |
|---|---|---|---|
| Thermal | 0.896 [0.749, 1.000], onset 3 | **0.988 [0.953, 1.000], onset 3** | 0.991 [0.964, 1.000], onset 2 |
| Thermal 80 x 60 | 0.778, onset 5 | 0.879, onset 3 | 0.917, onset 2 |
| Depth | 0.929 [0.825, 0.997], onset 2 | 0.929, onset 0 | 0.630, onset 5 |
| RGB | 0.722, onset 6 | 0.749, onset 6 | 0.790, onset 4 (18 plants) |
| Hyperspectral | 0.607 [0.423, 0.790], onset 5 | 0.624, onset 5 | no HSI features |
| RGB + depth + thermal | 0.947, onset 1 | 0.973, onset 0 | 0.973, onset 2 |
| All four | 0.947 [0.849, 1.000], onset 2 | 0.973 [0.905, 1.000], onset 0 | - |
| Plant size only | - | - | 0.491, onset 6 (26 plants) |

**Placebo.** The placebo bed model (trained on 24-29 Aug to tell W1 from W2/W3) separates
the beds in every fusion setup: AUC 0.71-0.98, so `bed_effect` is true throughout, against
false for the benchmark's thermal model (0.51). The frozen embeddings carry the bed
(lighting, pots, neighbours). Whether the detection model *uses* it is shown by its own
score over the placebo window (`placebo_auc_main`):

- thermal 0.49 (stage 1) and 0.60 (stage 3): near chance, so the thermal result is not a bed
  effect; its per-day curve stays at 0.5 until day 3 (lower bound 0.46 on day 2, 0.90 on day 3);
- depth 0.81 (stage 1) and 0.87 (stage 3): depth's early AUC is largely a bed difference.
  The setups with depth reach onset day 0, the afternoon of the last watering, which only a
  bed difference can explain;
- the fused setups sit between (0.58-0.67).

**Plants without turgor clips.** Clip hardware is on 18 of the 26 deficit plants but on only 3
of the 13 controls, from about 2 Sep, after the placebo window. On the 18 plants without clips
(8 deficit, 10 control; `fusion_cv_subsets_benchmark`, `plants == "no_clip"`):
thermal 0.962 (stage 1) and 0.975 [0.888, 1.000] (stage 3); all four sensors 0.925 and 0.950.
The clip hardware is not what the model detects. With 8 deficit plants the onset days are
not stable (thermal: day 8 in stage 1, day 3 in stage 3).

**Size baseline.** On the 28 plants with `depth_projected_area`, size alone has an early
AUC of 0.509 [0.285, 0.739] and onset day 6; thermal through the fusion model on the same
plants: 0.861 (stage 1) and 0.979 [0.925, 1.000] (stage 3).

**Part B, hyperspectral.** Hyperspectral is the weakest branch (0.607); 112 and 64 px give the
same AUC (0.601), and 3 RGB-like bands drop it to 0.518. The largest drops from masking one
20 nm window are 524-564 nm (0.07-0.11, the PRI region), but from a base this weak they are
within noise. Adding hyperspectral to the other three sensors does not change the fused AUC.

**Part C.** Within-day Spearman between the stage head and the clip stage: median 0.35
(stage 1) and 0.37 (stage 3) over 17 days; turgor head 0.09 and 0.03. The heads rank
severity weakly at best.

**Stage 2 (LoRA on hyperspectral) was not run.** The rule (section 5 of the work plan) runs it
only if stage 1's hyperspectral beats the benchmark's hyperspectral features on placebo-clean
folds. The benchmark has no hyperspectral features, and stage 1's hyperspectral is the
weakest branch, flagged by the placebo, and at 0.45 on the no-clip plants. It would cost about
3-7 h of cube decoding.

## 8. Architecture v2 (2026-11-15)

Code in `src/uiwue/fusion/v2.py` (model, cross-validation, ablation), `features.py` (hand-made
thermal features and green cover for every plant) and `lockbox.py`; tests in
`tests/jack/test_fusion_v2.py` and `test_fusion_lockbox.py`. The run log is
[fusion_progress.md](fusion_progress.md).

### What changed and why

| Change | Why |
|---|---|
| **Tokens are changes, not states.** Each sensor token is [e(t) − e(baseline), (e(t) − e(baseline)) − median of the training-fold controls' change in the same run]; v1's absolute e(t) is dropped. | e(t) carries the pot, the place and the light: the v1 placebo bed models found the beds in it (0.71-0.98). The change against the controls of the same run is a difference in differences: weather, light and camera drift shared by the run cancel, as in the benchmark's `thermal_leaf_ref_dt`. |
| **Control reference without the held-out plants.** The reference is the median over the training fold's W1 plants; a training control is compared with the other controls (leave-one-out median); held-out plants of either bed with all training controls. The fold checkpoint stores the reference per run, so `predict.py` references new plants the same way. | A held-out plant's bed is never used, and the lock-box never enters a reference. |
| **Hand-made thermal token** (`tfeat`): leaf temperature (median of the coolest half of the pixels), its 10th percentile, its spread over the five frames, and leaf minus air, each with the same two changes. It belongs to the thermal camera in every sensor setup; low-resolution thermal swaps in the 160 x 120 and 80 x 60 versions. | The ImageBind thermal embedding sees temperature only through a frozen image encoder; the benchmark's four numbers beat it on the 39 plants (0.991 against 0.896). |
| **Same-day causal window** (`win6`): each run's inputs are the mean over the plant's runs of the previous 6 h on the same day (12 h: the 6, 9 and 12 h runs; 15 h: 9, 12 and 15 h). The scoring is unchanged (mean of the 12 and 15 h outputs). | Part of each run's noise (clouds, the flyover time) is independent between runs; the window uses earlier runs only, so a day's score never sees a later run. |
| Optional, tested: weather token (`env`), bed adversary on the pre-cutoff runs (gradient reversal, weight 0.5 or 2), late pre-cutoff negatives weighted up (`hn`), a head without attention (`mean`), image-encoder tokens with or without the ImageBind thermal token (`fused_tfeat`), the hand-made token alone (`tfeat_only`). | Section 7's open problems: bed cues, weather, overfitting, and whether the encoders add anything to the hand-made thermal numbers. |
| Everything else is v1's: 5 folds of whole plants stratified by bed, inner early-stopping split, labels and the cutoff-day rule, the placebo bed model per fold, modality dropout 0.3, the stage and turgor heads, and the scoring (`train.score`). | The numbers sit beside section 7. |

### Protocol

- **Lock-box.** 45 of the 216 plants (15 per bed, 5 with turgor clips), drawn within bed x clip
  strata with a fixed seed before any v2 experiment (`lockbox.py`, commit ec1ec1d). They were
  never trained, tuned or selected on, and never a control reference. They were scored once,
  after the design was fixed.
- **Development plants.** The other 171. Every number below is out of fold (5 plant folds,
  stratified by bed) with 95 % plant-bootstrap intervals (2,000 resamples).
- **Selection rule, fixed before the second round** (fusion_progress.md, 23:25): among the v2
  variants whose thermal `placebo_auc_main` is at most 0.65, the earliest thermal onset day,
  then the higher early-AUC lower bound, then the lower `placebo_auc_main`; keep v1 if nothing
  beats it. Amended at 23:58, after round 3 showed an onset of day 0: a variant with onset day 0
  is not eligible (day 0 is 1-4 h after the last watering, so it marks a bed difference). The
  rule was applied to the full development set by a script, not by eye.
- **Same plants for every comparison.** "v1" in the tables is v1's design (absolute embedding
  and baseline change, no hand-made token, the same encoder embeddings) run through the v2
  harness on the same folds. The benchmark is the thermal logistic model of section 7 (leaf
  temperature, its 10th percentile, leaf minus air, leaf minus the control bed; leave one
  plant out) rerun on the same plants with features from `features.py`, the control reference
  taken from these plants only.
- **Extra columns** (`derived/detect/fusion_v2_ablation.parquet`): AUC on days 0-3, on the
  three days before the cutoff (`day_m1_auc` = 2 Sep), after re-watering (days 14-19), the
  within-day Spearman of the stage and turgor heads, and the same metrics for plants without
  clips and plants with the size trait.

### The data limits early detection: a shift between the beds from 2 Sep

The beds are imaged 10-25 min apart in each flyover, and the greenhouse is warmer at the W3 end
in the afternoon (Master Plan 6.8). Each run therefore carries a bed-level offset. On the 171
development plants, before the cutoff, the bed median of the leaf-temperature change against the
controls moves from run to run by 1.3 °C (W2) and 1.7 °C (W3) (SD over the 12 and 15 h runs),
and by 0.9 and 1.2 °C for leaf minus air, against a spread of 0.7-1.0 °C between the plants of
one bed. The offset is shared by every plant of a bed, so the per-day AUC follows it, more
plants do not average it out, and the plant bootstrap does not include it in the intervals.

On 2 Sep (day -1, inside the baseline window, before the cutoff) the deficit beds turn warmer
than the controls. The raw leaf-minus-air change against the controls (no model) has a per-day
AUC of 0.81 on 2 Sep and 0.84 on day 0, against 0.31-0.51 on 28 Aug-1 Sep; then 0.60 on day 1,
0.76 on day 2, 0.88 on day 3 and 1.00 from day 4. With a baseline from 2 Sep only (the Master
Plan's sensitivity check: the doors near the plants were closed about then) day 0 is 0.55,
day 1 0.27, day 2 0.51, day 3 0.73 and day 4 0.99. Contrasts inside the plant's own image, which
a shared offset cannot move (canopy spread, leaf minus its 10th percentile), stayed at 0.4-0.6
until day 3 and rose from day 4 (66 plants checked at 23:20). Whether 2 Sep is a change in the
greenhouse or an early start of drying in W2/W3 cannot be settled from these data; either way
it precedes the cutoff. Every model's AUC on 2 Sep is in the tables (`day_m1_auc`), and the main
models are rerun with the 2-Sep baseline (`_base0902`).

### Ablation: every variant on the development plants

171 plants (57 per bed), out of fold, 2,000 plant resamples; `fusion_v2_ablation`,
`round == "final"` (the earlier rounds, on 43-121 plants, are in the same table). Thermal
setup. "Own placebo" is the detection model's score over 24-29 Aug, "bed model" the placebo
bed model's; rows ending `_base0902` use a baseline from 2 Sep only.

| Variant | Early AUC (days 1-4) [95 % CI] | Onset day | AUC day 1 [lower] | AUC day 2 [lower] | AUC 2 Sep (day -1) | AUC day 0 | Own placebo | Bed model | No-clip early AUC | Stage ρ | Turgor ρ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v1` | 0.980 [0.963, 0.993] | 3 | 0.74 [0.66] | 0.74 [0.66] | 0.62 | 0.84 | 0.74 | 0.96 | 0.982 | 0.27 | 0.27 |
| `delta` | 0.954 [0.923, 0.980] | 3 | 0.60 [0.51] | 0.65 [0.56] | 0.42 | 0.68 | 0.63 | 0.90 | 0.955 | 0.43 | 0.19 |
| `ref` | 0.984 [0.968, 0.995] | 2 | 0.74 [0.66] | 0.80 [0.72] | 0.74 | 0.88 | 0.57 | 0.88 | 0.984 | 0.25 | 0.21 |
| `benchmark_logistic` | 0.976 [0.953, 0.994] | 2 | 0.49 [0.41] | 0.88 [0.83] | 0.92 | 0.98 | 0.63 | 0.69 | 0.974 | – | – |
| `benchmark_logistic_prelabels` | 0.984 [0.966, 0.997] | 2 | 0.51 [0.42] | 0.88 [0.83] | 0.91 | 0.98 | 0.67 | 0.69 | 0.983 | – | – |
| `delta+ref` | 0.969 [0.945, 0.988] | 2 | 0.73 [0.65] | 0.82 [0.75] | 0.61 | 0.81 | 0.65 | 0.91 | 0.967 | 0.28 | 0.18 |
| `now+delta+ref` | 0.983 [0.967, 0.995] | 0 | 0.81 [0.74] | 0.86 [0.80] | 0.73 | 0.87 | 0.76 | 0.97 | 0.982 | 0.22 | 0.04 |
| `ref+tfeat` | 0.980 [0.960, 0.994] | 2 | 0.66 [0.57] | 0.82 [0.75] | 0.74 | 0.79 | 0.53 | 0.99 | 0.977 | 0.37 | 0.01 |
| `delta+ref+tfeat` | 0.985 [0.967, 0.996] | 2 | 0.64 [0.56] | 0.85 [0.78] | 0.68 | 0.79 | 0.65 | 0.98 | 0.983 | 0.21 | 0.14 |
| `tfeat_only` | 0.988 [0.974, 0.999] | 2 | 0.77 [0.70] | 0.89 [0.84] | 0.74 | 0.56 | 0.52 | 1.00 | 0.988 | 0.41 | 0.29 |
| `v1_base0902` | 0.968 [0.935, 0.992] | 3 | 0.75 [0.66] | 0.70 [0.61] | 0.57 | 0.80 | 0.78 | 0.97 | 0.965 | 0.15 | -0.06 |
| `delta+ref+tfeat_base0902` | 0.971 [0.948, 0.988] | 2 | 0.73 [0.65] | 0.83 [0.75] | 0.49 | 0.80 | 0.67 | 0.96 | 0.973 | 0.38 | 0.18 |
| `tfeat_only_base0902` | 1.000 [0.999, 1.000] | 0 | 0.99 [0.97] | 0.96 [0.93] | 0.49 | 0.83 | 0.60 | 0.99 | 1.000 | 0.35 | 0.35 |
| `benchmark_logistic_base0902` | 0.856 [0.797, 0.909] | 4 | 0.16 [0.10] | 0.65 [0.56] | 0.49 | 0.68 | 0.14 | 0.90 | 0.847 | – | – |
| `benchmark_logistic_prelabels_base0902` | 0.876 [0.822, 0.924] | 4 | 0.15 [0.10] | 0.67 [0.58] | 0.43 | 0.67 | 0.18 | 0.90 | 0.867 | – | – |
| `delta+ref+tfeat+env` | 0.984 [0.964, 0.997] | 2 | 0.69 [0.62] | 0.87 [0.81] | 0.68 | 0.78 | 0.62 | 0.98 | 0.984 | 0.28 | 0.11 |
| `delta+ref+tfeat+adv` | 0.941 [0.900, 0.972] | 2 | 0.66 [0.58] | 0.81 [0.74] | 0.76 | 0.73 | 0.56 | 0.98 | 0.941 | 0.37 | 0.08 |
| `delta+ref+tfeat+adv2` | 0.987 [0.972, 0.997] | 2 | 0.67 [0.59] | 0.85 [0.79] | 0.66 | 0.81 | 0.57 | 0.98 | 0.986 | 0.31 | 0.21 |
| `delta+ref+tfeat+hn` | 0.959 [0.929, 0.984] | 2 | 0.69 [0.61] | 0.84 [0.78] | 0.67 | 0.81 | 0.58 | 0.98 | 0.958 | 0.34 | 0.11 |
| `delta+ref+tfeat+win6` | 0.968 [0.936, 0.992] | 2 | 0.70 [0.61] | 0.78 [0.71] | 0.64 | 0.82 | 0.70 | 0.98 | 0.969 | 0.29 | 0.12 |
| `tfeat_only+win6` | 0.992 [0.982, 0.999] | 1 | 0.84 [0.78] | 0.85 [0.79] | 0.75 | 0.74 | 0.53 | 0.99 | 0.993 | 0.21 | 0.11 |
| `fused_tfeat` | 0.987 [0.963, 1.000] | 3 | 0.67 [0.58] | 0.78 [0.70] | 0.78 | 0.90 | 0.56 | 0.85 | 0.986 | 0.43 | 0.27 |
| `delta+ref+tfeat+mean` | 0.980 [0.960, 0.994] | 2 | 0.66 [0.57] | 0.81 [0.74] | 0.61 | 0.86 | 0.68 | 0.95 | 0.977 | 0.57 | 0.05 |
| `v1+tfeat` | 0.985 [0.955, 1.000] | 3 | 0.72 [0.64] | 0.73 [0.65] | 0.63 | 0.85 | 0.63 | 0.94 | 0.985 | 0.33 | 0.03 |
| `tfeat_only+win6_base0902` | 0.999 [0.996, 1.000] | 1 | 0.84 [0.78] | 0.82 [0.75] | 0.49 | 0.63 | 0.35 | 0.99 | 0.999 | 0.17 | 0.11 |
| `fused_tfeat_base0902` | 0.963 [0.928, 0.989] | 3 | 0.59 [0.50] | 0.68 [0.59] | 0.51 | 0.65 | 0.45 | 0.86 | 0.971 | 0.39 | 0.17 |

Other setups (early AUC / onset day / own placebo score):

| Variant | Thermal 80 x 60 | Depth | RGB | Hyperspectral | All four | ImageBind thermal token alone | Hand-made token alone |
|---|---|---|---|---|---|---|---|
| `v1` | 0.976 / 3 / 0.67 | 0.994 / 0 / 0.89 | 0.938 / 1 / 0.63 | 0.915 / 1 / 0.71 | 0.999 / 0 / 0.74 | – | – |
| `delta` | 0.921 / 3 / 0.59 | 0.989 / 0 / 0.86 | 0.903 / 3 / 0.51 | 0.789 / 5 / 0.61 | 0.996 / 0 / 0.62 | – | – |
| `ref` | 0.942 / 2 / 0.60 | 0.994 / 0 / 0.62 | 0.869 / 4 / 0.30 | 0.906 / 4 / 0.56 | 0.998 / 0 / 0.51 | – | – |
| `benchmark_logistic` | 0.856 / 2 / 0.31 | – | – | – | – | – | – |
| `delta+ref` | 0.894 / 2 / 0.62 | 0.984 / 0 / 0.78 | 0.947 / 2 / 0.45 | 0.832 / 4 / 0.59 | 0.992 / 0 / 0.71 | – | – |
| `now+delta+ref` | 0.912 / 2 / 0.73 | 0.995 / 0 / 0.84 | 0.933 / 1 / 0.67 | 0.897 / 0 / 0.64 | 1.000 / 0 / 0.77 | – | – |
| `ref+tfeat` | 0.953 / 2 / 0.51 | 0.993 / 0 / 0.65 | 0.863 / 3 / 0.34 | 0.901 / 1 / 0.49 | 0.998 / 0 / 0.52 | 0.942 / 2 / 0.62 | 0.987 / 2 / 0.42 |
| `delta+ref+tfeat` | 0.927 / 2 / 0.61 | 0.989 / 0 / 0.73 | 0.943 / 2 / 0.50 | 0.822 / 4 / 0.58 | 0.997 / 0 / 0.72 | 0.908 / 2 / 0.76 | 0.983 / 2 / 0.40 |
| `tfeat_only` | 0.986 / 2 / 0.46 | – | – | – | – | – | – |
| `v1_base0902` | 0.944 / 0 / 0.76 | 0.983 / 0 / 0.91 | 0.935 / 3 / 0.70 | 0.894 / 0 / 0.74 | 1.000 / 0 / 0.74 | – | – |
| `delta+ref+tfeat_base0902` | 0.966 / 2 / 0.67 | 0.982 / 1 / 0.71 | 0.915 / 1 / 0.60 | 0.918 / 0 / 0.65 | 0.999 / 0 / 0.66 | 0.921 / 3 / 0.76 | 0.923 / 4 / 0.39 |
| `tfeat_only_base0902` | 0.999 / 0 / 0.67 | – | – | – | – | – | – |
| `benchmark_logistic_base0902` | 0.709 / 4 / 0.10 | – | – | – | – | – | – |
| `delta+ref+tfeat+env` | 0.957 / 2 / 0.55 | 0.988 / 0 / 0.75 | 0.946 / 2 / 0.58 | 0.873 / 2 / 0.57 | 0.999 / 0 / 0.67 | 0.949 / 2 / 0.71 | 0.925 / 2 / 0.45 |
| `delta+ref+tfeat+adv` | 0.915 / 2 / 0.55 | 0.987 / 0 / 0.70 | 0.937 / 2 / 0.48 | 0.869 / 4 / 0.53 | 0.998 / 0 / 0.55 | 0.839 / 3 / 0.61 | 0.992 / 2 / 0.33 |
| `delta+ref+tfeat+adv2` | 0.954 / 2 / 0.53 | 0.978 / 0 / 0.74 | 0.881 / 3 / 0.65 | 0.734 / 4 / 0.61 | 0.991 / 0 / 0.71 | 0.915 / 3 / 0.68 | 0.946 / 2 / 0.41 |
| `delta+ref+tfeat+hn` | 0.930 / 2 / 0.53 | 0.983 / 0 / 0.77 | 0.940 / 3 / 0.51 | 0.885 / 4 / 0.56 | 0.996 / 0 / 0.74 | 0.860 / 3 / 0.67 | 0.950 / 2 / 0.43 |
| `delta+ref+tfeat+win6` | 0.959 / 3 / 0.68 | 0.988 / 0 / 0.75 | 0.906 / 3 / 0.39 | 0.853 / 4 / 0.47 | 0.998 / 0 / 0.55 | 0.922 / 3 / 0.70 | 0.951 / 3 / 0.55 |
| `tfeat_only+win6` | 0.992 / 1 / 0.56 | – | – | – | – | – | – |
| `fused_tfeat` | 0.987 / 3 / 0.55 | 0.980 / 0 / 0.74 | 0.907 / 3 / 0.43 | 0.861 / 4 / 0.61 | 0.996 / 0 / 0.71 | – | – |
| `delta+ref+tfeat+mean` | 0.949 / 2 / 0.54 | 0.982 / 0 / 0.80 | 0.863 / 3 / 0.52 | 0.872 / 4 / 0.56 | 0.995 / 0 / 0.67 | 0.957 / 3 / 0.71 | 0.988 / 2 / 0.48 |
| `v1+tfeat` | 0.984 / 3 / 0.51 | 0.995 / 0 / 0.90 | 0.905 / 2 / 0.65 | 0.858 / 1 / 0.71 | 1.000 / 0 / 0.78 | 0.973 / 3 / 0.78 | 0.857 / 3 / 0.32 |
| `tfeat_only+win6_base0902` | 0.997 / 1 / 0.34 | – | – | – | – | – | – |
| `fused_tfeat_base0902` | 0.964 / 3 / 0.45 | 0.978 / 1 / 0.72 | 0.853 / 3 / 0.59 | 0.702 / 4 / 0.48 | 0.996 / 1 / 0.58 | – | – |

### The chosen design: `tfeat_only+win6`

The rule picks `tfeat_only+win6`: the v2 head (2-layer transformer, deficit, stage and turgor
heads) on one token, the hand-made thermal features as [change from the plant's baseline,
change minus the training controls' median change in the same run], each run averaged with the
plant's runs of the previous 6 h of that day. No image encoder is used. Runner-up: the same
without the window (`tfeat_only`), onset day 2 (its day-1 lower bound is 0.699).

On the same 171 development plants (114 deficit, 57 control; out of fold; thermal camera only):

| | **v2** (`tfeat_only+win6`) | v1 design | Benchmark (logistic) |
|---|---|---|---|
| Early AUC, days 1-4 [95 % CI] | **0.992 [0.982, 0.999]** | 0.980 [0.963, 0.993] | 0.976 [0.953, 0.994] |
| Onset day | **1** | 3 | 2 |
| AUC day 1 [lower bound] | 0.84 [0.78] | 0.74 [0.66] | 0.49 [0.41] |
| AUC day 2 [lower bound] | 0.85 [0.79] | 0.74 [0.66] | 0.88 [0.83] |
| AUC on day 0 (1-4 h after the last watering) | 0.74 [0.66] | 0.84 | 0.98 |
| AUC on 2 Sep (day -1, before the cutoff) | 0.75 | 0.62 | 0.92 |
| Own score in the placebo window (`placebo_auc_main`) | **0.53** | 0.74 | 0.63 |
| Placebo bed model (`placebo_auc`) | 0.99 | 0.96 | 0.69 |
| Plants without clips (155: 100 deficit, 55 control) | 0.993, onset 1 | 0.982, onset 3 | 0.974, onset 2 |
| Thermal at 80 x 60 | 0.992, onset 1 | 0.976, onset 3 | 0.856, onset 2 |
| **Baseline from 2 Sep only** | 0.999 [0.996, 1.000], onset 1; 2 Sep 0.49 | 0.968, onset 3 | 0.856 [0.797, 0.909], onset 4 |
| Within-day Spearman, stage head / turgor head | 0.21 / 0.11 | 0.27 / 0.27 | – |

Size alone on the same plants: `depth_projected_area` (Master Plan 6.7.3; the depth traits of
all 216 plants, 15 Nov) 0.545 [0.460, 0.638], onset day 5; RGB green cover 0.656 [0.577, 0.737],
onset day 5. Every model above beats both.

What the comparison says:

- **Earlier and cleaner than the benchmark on the same plants.** v2 detects on day 1, the
  benchmark on day 2, v1 on day 3, with the highest early AUC of the three. On day 0, when no
  water has been withheld for more than 4 h, the benchmark already separates the beds (0.98)
  and v2 much less (0.74); on 2 Sep 0.92 against 0.75.
- **The day-1 detection is not the 2-Sep shift.** With the baseline taken from 2 Sep only, the
  2 Sep separation goes (0.49), v2 keeps onset day 1 (day 1 0.84 [0.78]) and the benchmark falls
  to onset day 4 (day 1 0.16). So the benchmark's day-2 onset leans on the shift; v2's does not.
- **Where the gain comes from.** The benchmark trained with the fusion labels (every pre-cutoff
  day a negative) scores the same as the benchmark (`benchmark_logistic_prelabels`), so the
  labels are not it. What v2 adds: run-level training on all five hours, a nonlinear head, the
  frame spread as a feature, and the same-day window (`tfeat_only` without it: onset 2).
- **Placebo.** The detection model's own placebo-window score is 0.53, near chance (v1 design
  0.74, benchmark 0.63). The placebo bed model, a separate model trained to tell the beds apart
  on 24-29 Aug from the same inputs, still separates them (0.99, `bed_effect` true): with the
  controls as the reference, any bed-level offset of a run makes the deficit beds stand out by
  their distance from zero, which a nonlinear bed classifier uses. The detection model is
  trained with those runs as negatives and does not (0.53).
- **By bed.** W2 alone: onset 2; W3 alone: onset 0. With 57 plants per group the bed-level
  offsets of single runs decide single days (the benchmark: W2 onset 0, W3 onset 2), so the
  pooled onset (all deficit plants against W1) is the one to read.

### The other sensors and the fused setups

From the ablation table (early AUC / onset day / own placebo score, all development plants):

| Setup | v1 design | v2 tokens (`delta+ref+tfeat`) | v2 without the ImageBind thermal token (`fused_tfeat`) |
|---|---|---|---|
| Depth | 0.994 / 0 / 0.89 | 0.989 / 0 / 0.73 | 0.980 / 0 / 0.74 |
| RGB | 0.938 / 1 / 0.63 | 0.943 / 2 / 0.50 | 0.907 / 3 / 0.43 |
| Hyperspectral | 0.915 / 1 / 0.71 | 0.822 / 4 / 0.58 | 0.861 / 4 / 0.61 |
| All four | 0.999 / 0 / 0.74 | 0.997 / 0 / 0.72 | 0.996 / 0 / 0.71 |

- The change tokens lower every sensor's own placebo score (depth 0.89 to 0.73, RGB 0.63 to
  0.50, hyperspectral 0.71 to 0.58). Without the absolute embedding, the v1 design's day-1
  onsets for RGB and hyperspectral go (they came with own placebo scores of 0.63-0.71).
- Depth and every fused setup with depth still reach onset day 0 in every variant: depth's 2 Sep
  score is 0.75-0.90. The weight-2 adversary, the weather token and the late negatives do not
  change that. Depth carries a growth or posture difference between the beds that starts before
  the cutoff; the fused setups should not be read as earlier than thermal.
- Adding the image-encoder tokens to the hand-made thermal token does not help the thermal
  setup (`fused_tfeat` 0.987, onset 3; `delta+ref+tfeat` 0.985, onset 2; `tfeat_only` 0.988,
  onset 2).

### Severity heads (clip plants)

Within-day Spearman of the stage head with the clip stage, median over days: v1 design 0.27,
`tfeat_only` 0.41, `delta+ref+tfeat+mean` 0.57, the chosen `tfeat_only+win6` 0.21; turgor head:
0.27, 0.29, 0.05 and 0.11. The deficit probability of the chosen model ranks turgor at 0.34. With
16 clip plants in the development set (14 deficit) the heads stay weak; the window that helps
detection blurs the heads' day-to-day ranking. Severity is not what the selection rule optimised.

### Lock-box (45 plants, scored once)

Scored at 00:48 on 15 Nov, after the design was frozen (commit 1887795), by the five fold
models of each design averaged (`predict.py --lockbox`); the benchmark was fitted on every
development plant-day and applied to the lock-box (`v2 bench-lockbox`). 30 deficit plants,
15 controls (`fusion_lockbox_benchmark`, `plants == "all"`; the canary and severity columns
are in `fusion_v2_ablation`, `round == "lockbox"`):

| Lock-box, thermal | **v2** (`tfeat_only+win6`) | v1 design | Benchmark (logistic) |
|---|---|---|---|
| Early AUC, days 1-4 [95 % CI] | **1.000 [1.000, 1.000]** | 0.993 [0.973, 1.000] | 0.989 [0.958, 1.000] |
| Onset day | 0 | 0 | 2 |
| AUC day 0 [lower] / day 1 [lower] | 0.87 [0.74] / 0.86 [0.74] | 0.88 [0.76] / 0.90 [0.80] | 0.97 / 0.44 |
| AUC on 2 Sep (day -1) | 0.81 | 0.70 | 0.88 |
| Own score in the placebo window | 0.75 | 0.74 | 0.74 |
| Placebo bed model | 1.00 | 0.92 | 0.62 |
| Plants without clips (40) | 1.000, onset 0 | 0.992, onset 0 | 0.986, onset 2 |

Size alone on the lock-box (`depth_projected_area`): 0.447 [0.274, 0.615], onset day 6.

The lock-box confirms the early AUC (1.000: every deficit plant above every control over
days 1-4; v1 design 0.993, benchmark 0.989) but not the clean early onset. On these 45 plants v2
and the v1 design pass the onset rule already on day 0, which cannot be a water-deficit
response, and all three models score the deficit plants higher in the placebo window (0.74-0.75,
against 0.53 for v2 on the development plants). Because the three models agree on it, the
pre-cutoff bed difference belongs to these plants rather than to v2; with 15 controls one run's
bed-level offset moves a day's AUC a lot, and five averaged fold models sharpen both the
detection and the bed difference. What carries over to new plants: the early AUC, and v2 at or
above v1 and the benchmark on it. What the lock-box does not support: an onset before day 2 that
is clear of bed differences. The benchmark keeps onset day 2 there, with day 1 at 0.44.

### Tried and not adopted, and not tried

- **Not adopted** (all in the ablation table): the absolute embedding (`v1`, `now+delta+ref`,
  `v1+tfeat`: the highest own placebo scores, and `now+delta+ref` reaches onset day 0); the
  baseline change alone (`delta`); the image-encoder tokens with the hand-made token
  (`ref+tfeat`, `delta+ref+tfeat`, `fused_tfeat`: onset 2-3); the weather token (`env`); the bed
  adversary (`adv`, `adv2`); late pre-cutoff negatives weighted up (`hn`); the head without
  attention (`mean`); the window on the fused tokens (`delta+ref+tfeat+win6`).
- **Not tried, and why.**
  - Per-bed normalisation: it needs the bed at inference, which rule 2 forbids.
  - LoRA on the thermal encoder for v2: the hand-made token already reaches the ceiling of the
    early AUC, and the ImageBind thermal token adds nothing to it in the fused variants.
  - Hyperspectral as plant-masked mean spectra or indices: a second decode of every cube
    (about 1 GB per plant) while the embedding pipeline was the bottleneck of the night.
  - Patch tokens, more registered thermal frames, a learned temporal model beyond the fixed
    window, an onset (hazard) formulation. The part of the noise that limits early detection is
    shared by every plant of a bed (a run's offset, the 2-Sep shift). Averaging a plant's own
    runs helps only where the noise is independent between runs, which is what the window does.

### Commands and timings

| Step | Where | Command | Time |
|---|---|---|---|
| Thermal features, 216 plants | local, 4 processes | `python -m uiwue.fusion.features --roots data/thermal cache/raw_external/thermal --plants <ids>` (reads the zip of a plant without a folder) | about 7 s per plant (6 min for 71 plants) |
| Green cover, 216 plants | local, 3-4 processes | `python -m uiwue.fusion.features --rgb-size` | 635 s for 173 plants |
| Ablation, 17 variants and the 2-Sep rows, 171 plants | local RTX 5070 Ti (4 processes) and the GPU server (4 processes) | `python -m uiwue.fusion.v2 ablate --cache cache/fusion cache/fusion_external/<b*, g*> --n-boot 2000 --variants <names>`; `--baseline 2025-09-02 2025-09-02 --tag _base0902` for the sensitivity rows | 23 min wall (one variant: 2-4 min on the server, 6-8 min locally) |
| Chosen design | GPU server | `python -m uiwue.fusion.v2 train --cache ... --variant tfeat_only+win6` | 237 s |
| v1 design on the same plants (for the lock-box) | GPU server | `python -m uiwue.fusion.v2 train --cache ... --variant v1 --name fusion_v1dev --out <scratch>` | about 4 min |
| CV subsets | GPU server | `python -m uiwue.fusion.predict --oof --model fusion_v2` | under 1 min |
| Lock-box (once) | GPU server | `python -m uiwue.fusion.predict --lockbox --cache ... --model fusion_v2` (and `fusion_v1dev`); `python -m uiwue.fusion.v2 bench-lockbox` | 18 s, 39 s, 4 s |
| Size on every plant | local | `python -m uiwue.fusion.v2 size --size-traits <depth traits of all 216 plants>` | seconds |

The selection rule was applied with a short script over the ablation table (the rule is
quoted in "Protocol"); `fusion_v2` in `fusion_benchmark` reproduces the ablation's
`tfeat_only+win6` row exactly (same folds and seed). The run scripts, the selection script and
the logs are in `runs/logs/v2/` (not in git).

Outputs: `derived/detect/fusion_v2_ablation.parquet` (every variant and round),
`fusion_benchmark`, `fusion_auc_by_day`, `fusion_severity` (model `fusion_v2`),
`fusion_cv_subsets_*`, `fusion_lockbox_*` and `fusion_v2_scores_runs` (`detect_fusion_v2_*`,
out of fold, development plants only). Checkpoints in `runs/fusion/fusion_v2/` (not in git).
