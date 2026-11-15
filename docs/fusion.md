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
| `predict.py` | External test: the 5 fold models (and placebo models) score plants outside training, averaged; `--oof` rescores the CV for plants without clips and plants with the size trait |

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
