# Jonathan — Work Plan: Challenge 1, Early Water-Deficit Stress Detection

- **Owner:** Jonathan
- **Challenge:** 1 — early stress detection, sensor benchmarking, severity and recovery
- **Dataset:** ADA 2025 ([paper](https://arxiv.org/html/2610.23609v1), [code](https://github.com/pippowell/ada2025), data DOI `10.26249/FK2/DMDLTE`)
- **Time box:** 2026-11-14 and 2026-11-15
- **Status (Sat 14 Nov):**
  - **Benchmark track:** built, with results in `derived/detect/`.
  - **Main track:** built in `src/uiwue/fusion/`, with a run guide in `docs/fusion.md`. It still has to be trained on the GPU machine.
  - Both tracks are on the `jack` branch.

## 1. Goal

Answer three questions with the ADA 2025 thermal, hyperspectral, RGB and depth
images:

- **A. Early detection:** how many days after the irrigation cutoff can
  water-deficit stress be detected, and is that before symptoms are visible?
- **B. Sensor benchmark:** does combining sensors beat the best single sensor,
  what does full-resolution hyperspectral add, and what is the cheapest setup
  that comes close?
- **C. Severity and recovery:** can we tell mild from severe stress, and how
  far does each plant come back after rewatering?

**Two model tracks.** Every question is answered twice, on the same folds:

- **Main model:** pretrained image encoders with a fusion head, fine-tuned on
  this data (section 5).
- **Benchmark:** logistic regression and small gradient-boosted trees on
  hand-made features (section 4).

The benchmark is the bar the fine-tuned model has to clear, and the fallback
if fine-tuning doesn't work in two days.

## 2. Data I need

**Plants.** The shared team set: the 10 paper-subset plants (5 in W1, 5 in W2)
and the 21 turgor-clip plants (3 in W1, 9 in W2, 9 in W3). I also want about 10
extra W1 controls, thermal + RGB only, so the models see more than about 8
control plants.

**Cameras.**
- Thermal: FLIR A70, 640 x 480, °C, 5 frames per run.
- Hyperspectral: Resonon Pika L, 300 bands, ~384–1025 nm, 256 x 256, raw
  uint8, 1 cube per run.
- RGB: Logitech C920, 15 frames per run.
- Depth: Pico Flexx, 224 x 171; traits from Challenge 3's table, raw frames for
  the depth encoder.

**Ground truth.**
- Turgor clips: every 2 min from 2025-09-02. A higher value means lower turgor.
- RWC: W3 only, destructive, on days 1, 3, 5, 6, 7, 8, 11 and 12 after the
  cutoff. The `rwc` column is empty, so it is computed from the weights.
- Symptom log and BBCH: every plant, on 7 days from 2025-09-02.
- Last-day photo of every plant.

There is no usable soil moisture data.

**Environment.** Air temperature and humidity at W1 and W3, none at W2. PAR in
raw mV only.

**Size (my estimate).** About 2.9 GB per plant for all four cameras: about 90 GB
for the 31 shared plants, plus about 14 GB for the extra controls.

**Compute.** One GPU with ≥ 24 GB memory for embedding extraction and
fine-tuning. Everything in the benchmark track runs on CPU.

**Key dates.** Irrigation off 2025-09-03 11:00, back on 2025-09-15, last images
2025-09-22. No depth from 2025-09-19 15 h onward.

## 3. Ground rules for valid results

The dataset makes it easy to get high accuracy for the wrong reason. Both
model tracks follow these rules:

1. **The plant is the unit.** Train and test by holding out whole plants, never
   by splitting images:
   - Benchmark: leave-one-plant-out.
   - Fine-tuned model: 5 folds grouped by plant, because each fold is a full
     training run.
   - Confidence intervals come from resampling plants.
2. **No shortcuts as inputs.** Models never see date, bed, hour or grid
   position. Every deficit plant was stressed on the same schedule, so the date
   alone gives the answer.
3. **Beat the size baseline.** Stressed plants stop growing. Every figure shows a
   model that uses plant size only. A result counts only if it beats that.
4. **Run the placebo first.** Both tracks must fail to separate W1 from W2/W3
   before the cutoff (2025-08-30 to 2025-09-02). The placebo AUC is the floor
   that every later result must clear. This matters most for the fine-tuned
   model, which can learn bed differences (lighting, pots, neighbours) that no
   hand-made feature would pick up.
5. **Own baseline.** Features and embeddings are also given as a change from
   the plant's own pre-cutoff values at the same hour, the same rule as
   Challenge 3.
6. **Fit everything inside the fold.** Normalisation statistics,
   fine-tuning and hyperparameter choices use training plants only.

## 4. Benchmark track: hand-made features

```
thermal TIFF + HSI cube/JSON + RGB frames + air sensor data
  -> map re-runs to the scheduled hour (±1 h)
  -> HSI flat-field: C = (R − D) / (W · m − D), m = 2.0; smooth (Savitzky–Golay 31, 3)
  -> plant mask: thermal = coolest 50% of centre region (paper rule)
                 HSI = NDVI threshold in centre region
                 RGB = ExG
  -> features per frame -> median and MAD across frames
  -> join with Challenge 3 depth traits on bed, plant, date, hour
  -> subtract each plant's pre-cutoff baseline
  -> wide table (one row per plant per run) + QC flags
  -> logistic regression / small gradient-boosted trees
```

Features:

- **Thermal:** canopy temperature (p10, p50, p90); canopy minus air; canopy
  minus soil in the same image; an empirical stress index (CWSI) from
  within-image percentiles.
- **Hyperspectral:** the paper's NDVI, CI red-edge and SIPI. Also PRI (531/570
  nm), WBI (900/970 nm) and red-edge position. HSI uses the 09, 12 and 15 h
  runs only, because the 06 and 18 h cubes are dark or saturated.
- **RGB:** top-down area, greenness (ExG), mean hue.
- **Depth (from Challenge 3):** height p75, leaf inclination, droop index.

QC flags: `dark_frame`, `hsi_saturated`, `mask_failed`, `missing_modality`,
`rwc_sampled` (W3 plant after its leaf was taken).

## 5. Main track: fine-tuned fusion model

### Encoders

**Hyperspectral and RGB: DOFA** ([paper](https://arxiv.org/abs/2403.15356),
[code](https://github.com/zhu-xlab/DOFA), also in TorchGeo as
`dofa_base_patch16_224`).

- DOFA is a ViT-B/16 whose patch-embedding weights are generated from each
  band's centre wavelength. That means:
  - It takes all 300 Pika L bands as they are, with their wavelengths from the
    cube metadata. Nothing is reduced to a few bands first.
  - The token count depends on image size, not band count, so 300 bands cost
    little more than 3.
- RGB goes through the same encoder with the webcam's three band centres. This
  makes "HSI vs. RGB" a clean comparison: same model, different bands.
- **Input:** flat-field-corrected reflectance, normalised per band on training
  plants. The 256 x 256 cube is resized to 224 x 224, which is almost native
  resolution. If GPU memory runs short, fall back to 112 x 112.
- **Caveat:** DOFA was pretrained on satellite imagery, not close-range plants.
  That is why it is fine-tuned rather than only used frozen.

**Thermal and depth: ImageBind encoders**
([paper](https://arxiv.org/abs/2305.05665)).

- ImageBind has dedicated single-channel ViT-B/16 encoders for thermal and
  depth, trained to align with images.
- Thermal is long-wave infrared and outside DOFA's wavelength range, and depth
  isn't spectral, so these encoders cover what DOFA can't.
- Depth goes in as disparity (1 / depth), as ImageBind expects.
- **Licence:** the ImageBind weights are, to my knowledge, non-commercial.
  Confirm before using them (see section 13).

### Fusion head

```
HSI cube    -> DOFA        -> embedding ┐
RGB frame   -> DOFA        -> embedding ├─ each: [e(t), e(t) − e(baseline)] -> linear -> 256
thermal     -> ImageBind-T -> embedding │      + learned sensor token
depth       -> ImageBind-D -> embedding ┘
                 -> 2-layer transformer over the sensor tokens + [CLS]
                 -> heads: deficit (binary) · stage (ordinal, Part C) · turgor deviation (Part C)
```

- **Modality dropout (p = 0.3) during training.** Whole sensors are randomly
  removed, so one trained model can be evaluated on any sensor subset (Part B)
  and keeps working when depth is missing after 2025-09-19.
- **Baseline delta.** Each sensor token carries both the current embedding and
  its change from the plant's pre-cutoff embedding. That is rule 5, applied to
  learned features.

### Fine-tuning, in stages

1. **Frozen encoders, train the head only.** Embeddings are extracted once (on
   Day 1, overnight), and the head trains in minutes per fold. This is the
   minimum deliverable.
2. **LoRA on DOFA** (rank 8, last 4 blocks) for the hyperspectral branch, if
   stage 1 beats the benchmark's HSI features on the placebo-checked folds.
   Hyperspectral is where the pretrained encoder is furthest from this data.
3. **LoRA on the thermal encoder:** only if time remains.

Augmentation: flips, 90° rotations, small crops, per-band brightness jitter.
Early stopping uses an inner validation split of training plants.

### Implementation (`src/uiwue/fusion/`)

| Plan step | File |
|---|---|
| Inputs per sensor, Part B variants, augmentation | `inputs.py` |
| DOFA, ImageBind, LoRA (rank 8, last 4 blocks) | `encoders.py` |
| Stage 1 embeddings (`python -m uiwue.fusion.embed`) | `embed.py` |
| Deficit, stage and turgor labels | `labels.py` |
| Fusion head | `fusion.py` |
| 5 plant-grouped folds, stages 1-3, evaluation (`python -m uiwue.fusion.train`) | `train.py` |

Where the build differs from the text above:

- **Normalisation.**
  - Inputs use fixed physical scalings, not per-band statistics.
  - The embeddings are standardised per feature on each fold's training plants. A per-sample
    layer norm would erase offsets shared by all features.
- **ImageBind encoder size.** ImageBind's thermal and depth encoders are ViT-S/16 (projected to
  1024), not ViT-B/16.
- **Part B variants are swapped in at evaluation, not retrained.** Low-resolution thermal and
  the hyperspectral variants (spatial 112 and 64, 3 bands, 20 nm windows) measure what the
  trained model loses with the cheaper input.
- **Placebo.**
  - A placebo model is trained to tell the beds apart on 24-29 Aug.
  - The main model's own pre-cutoff score is reported too.
  - A sensor that only carries a bed difference scores well after the cutoff for the wrong
    reason (controls are never positive), so both checks are needed.
- **Thermal in the LoRA stage** uses its centre frame, so runs stack into batches.

## 6. Part A: Early detection

**Label.** Deficit plants from the cutoff onward are positive. Control plants,
and all plants before the cutoff, are negative.

**Method.** For each day after the cutoff (0–12) and each sensor setup, score
held-out plants with both tracks. Compute the AUC per day with a 95% interval
from resampling plants.

**Onset day.** The first day where the AUC's lower bound is above 0.70 and
stays above it. This matches Challenge 3's rule (first significant day that
stays significant), so onset days line up across challenges.

**Before visible symptoms?** I compare each sensor's onset day with two
"visible" days: the first wilting entry in the symptom log, and the RGB-only
onset day. I also compare it with turgor onset on the clip plants, which is
the earliest physiological signal.

**Headline comparison.** Onset day and early AUC (days 1–4) for the fine-tuned
model vs. the benchmark, per sensor and fused.

**Extra if time allows.** Repeat per hour of day. Thermal should be strongest
at 12–15 h.

## 7. Part B: Sensor benchmark

**Sensor combinations.** Every combination of the four sensors, through two
models:
- The fusion model, with absent sensors dropped (it was trained for this).
- The benchmark, with features concatenated.

The output is a table of early AUC (days 1–4), full AUC and onset day for each
setup and each track.

**What full-resolution hyperspectral adds.** The hyperspectral branch keeps
all 300 bands. Three analyses show where its advantage comes from:

1. **Spectral vs. spatial detail.** Run the HSI branch at 224 x 224, 112 x 112
   and 64 x 64. If AUC holds at low spatial resolution, the signal is spectral,
   which favours cheaper line-scan or low-resolution spectral cameras over
   high-resolution ones.
2. **Which wavelengths matter.** Mask 20 nm windows across 384–1025 nm one at a
   time and record the AUC drop. This gives a spectral importance curve. It
   should line up with known stress regions (531/570 nm PRI, red edge, the
   970 nm water band). If it highlights unexpected regions, check those for
   artefacts.
3. **HSI vs. RGB inside one model.** Feed DOFA only the three cube bands
   closest to red, green and blue, then all 300. The difference is what the
   extra bands buy on the same sensor and the same plants.

**Cheaper setups.** Where downsampling doesn't throw away the sensor's main
advantage:
- Thermal at 160 x 120 and 80 x 60.
- One imaging run per day instead of five.

**Cost.** Sensors are ranked by cost, from webcam RGB up to hyperspectral.
Prices to be filled in by whoever owns deployment.

**Output.** A plot of performance against cost rank. The headline answer is:
"for this budget, this setup detects stress on day X".

## 8. Part C: Severity and recovery

**Stages, anchored to the 18 deficit plants with turgor clips:**
- **S0, healthy:** control, or before the cutoff.
- **S1, mild:** turgor drops at midday but recovers by the 06 h run.
- **S2, moderate:** the turgor loss is still there at 06 h.
- **S3, severe:** persistent loss plus wilting in the symptom log, checked
  against the W3 RWC curve.

**Can the images rank severity?** The fusion model's stage and
turgor-deviation heads, and the benchmark's regression, are tested the same
way: do they rank plants correctly *within the same day*? "Days since cutoff"
alone cannot do that.

**Recovery.** For each plant and feature: (after − end of stress) / (before −
end of stress), using pre-cutoff, 2025-09-14/15 and 2025-09-20/21 values. Depth
uses 2025-09-18/19 because of the gap. Only 3 plants died, so recovery is
measured as an amount, not as survival. With this few plants, the link to
end-of-stress features and embeddings is reported as a correlation, not a
prediction model.

## 9. What I hand to the team

**Tables.** All tables use the four shared keys (`bed`, `plant`, `date`,
`hour`).
- **Features:** one row per plant per run; columns prefixed by sensor
  (`thermal_`, `hsi_`, `rgb_`, `depth_`), each with a `_mad` spread column and
  `qc_*` flags.
- **Embeddings:** frozen DOFA and ImageBind embeddings per plant per run, so
  other challenges can use them without a GPU.
- **Labels:** `y_deficit`, `stage`, `turgor_dev`, `rwc`.
- **Results:** `experiment`, `track` (`finetuned` / `benchmark`), `sensors`,
  `days_since_cutoff`, `auc`, `ci_low`, `ci_high`.

**Figures.**
- AUC by day per sensor, both tracks, with the cutoff, onset days and visible
  day marked.
- The cost-versus-performance plot.
- The hyperspectral importance curve.
- Severity stages over time for the clip plants.
- Recovery per plant.

**App views.**
- Detection timeline.
- Sensor comparison (both tracks).
- Single-plant view: stress probability per sensor, stage band, turgor overlay.
- Export.

The framework follows the team decision in `docs/SOFTWARE_STACK.md`.

## 10. Schedule

**Day 1**
- First thing: start the downloads (paper subset with all four cameras, then
  thermal + RGB for the clip plants and the extra controls). Set up the
  environment, with TorchGeo for DOFA and the ImageBind repository.
- Get loaders, HSI calibration and masks working on 1 control and 1 deficit
  plant. Push one cube through DOFA and check the embedding shape and memory.
- Build the benchmark features for the 09, 12 and 15 h runs. Build the labels
  table. Run the placebo test on the benchmark.
- Late: extract frozen embeddings for every plant-run on the GPU overnight
  (all four encoders). Batch the remaining benchmark features on CPU.

**Day 2**
- Morning: placebo test on the fusion model. Train the fusion head (stage 1).
  Part A for both tracks.
- Midday: Part B (sensor subsets, HSI resolution, band masking). LoRA on DOFA
  (stage 2) if stage 1 beats the benchmark.
- Early afternoon: Part C. App views.
- Late afternoon: slides and integration with the master app.

Priority if time runs short:
1. Placebo.
2. Benchmark Part A.
3. Fusion head Part A.
4. Part B sensor subsets.
5. HSI band masking.
6. Part C stages.
7. LoRA.
8. Recovery.
9. HSI resolution sweep.

## 11. Not doing in two days

These go on a "next steps" slide:

- Full fine-tuning of all encoder weights.
- A close-range hyperspectral foundation model instead of DOFA. A model
  pretrained on proximal hyperspectral data has been described
  ([Laprade et al. 2025](https://arxiv.org/abs/2503.01628)), but its code and
  weights are not released yet. Swap it in when they are.
- All 216 plants with spatial-block cross-validation, and testing on an unseen
  bed (train W2, test W3).
- Models over each plant's full time series.
- Recovery prediction models; this needs more plants and more than one stress
  length.
- Testing on another sugar beet stress dataset (e.g. the
  [Eschikon plant stress set](https://projects.asl.ethz.ch/datasets/plant-stress-phenotyping-2018);
  access not verified).

The longer version of this plan is in `docs/ANALYSIS_PLAN.md`.

## 12. Risks

- **No GPU, or GPU too small:** the benchmark track still delivers every part.
  Run the fusion model on HSI at 112 x 112, frozen encoders only.
- **The fine-tuned model overfits (tens of plants):** frozen encoders first,
  LoRA only after that, early stopping on held-out plants. Report the benchmark
  if fine-tuning doesn't beat it.
- **The fine-tuned model learns bed differences:** the placebo test runs on it
  separately. If its placebo AUC is high, train on baseline-delta embeddings
  only and report the floor.
- **DOFA's satellite pretraining doesn't transfer:** LoRA on the HSI branch. If
  that still loses to the hand-made spectral indices, say so. That is a useful
  result too.
- **Download is slow (hyperspectral is the largest):** start with the paper
  subset. If it stays slow, HSI covers only those 10 plants and thermal + RGB
  cover the rest.
- **Too few controls:** add the extra W1 plants. If they don't arrive, report
  the wide intervals as they are.
- **Model learns plant size:** the size-only baseline is in every figure.
- **Depth table arrives late:** the fusion model runs without depth (modality
  dropout), and the benchmark adds depth when it lands.

## 13. Questions for the master plan

1. Who owns thermal and hyperspectral feature extraction? If another challenge
   does, I use their tables for the benchmark track.
2. Which GPU can I use, and is it available overnight on Day 1?
3. Is the ImageBind licence acceptable for this project? If not, the thermal
   and depth branches fall back to an ImageNet-pretrained ViT.
4. Do all challenges agree on the sensor prefix for columns and on mapping
   re-runs to the scheduled hour?
5. Can the shared plant set include about 10 extra W1 controls?
6. Who owns the shared download, and where does it live?
7. Is "AUC lower bound > 0.70, and stays there" the onset rule for every
   challenge that reports an onset day?
8. Who owns the final integration and the slides?
