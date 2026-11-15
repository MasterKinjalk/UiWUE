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
| `train.py` | 5 folds grouped by plant; stage 1 (frozen) and stages 2-3 (LoRA); evaluation on every sensor subset; results in `derived/detect/fusion_*` |

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
  - Thermal uses its centre frame in these stages so that runs stack into batches.
- **Not verified here.**
  - The code was written and tested on CPU with tiny random encoders
    (`tests/jack/test_fusion.py`).
  - The real DOFA weights load (TorchGeo), but ImageBind's real checkpoint has not been
    loaded on this machine. Check `ImageBindEncoder` on the first run.
  - The input scalings for thermal (27.5 ± 8.75 °C) and disparity are assumptions; fine-tuning
    adapts to them.
