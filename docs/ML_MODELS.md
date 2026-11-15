# Pretrained models to fine-tune

Two pretrained models fit the Challenge 3 plan
([plan/Kinjalk-Work-Plan.md](../plan/Kinjalk-Work-Plan.md)). Each one targets a risk listed in
the plan. Neither needs hand labels on ADA data.

| Name | Weights | Size | Licence | Fine-tuned on | Plan risk it addresses |
|---|---|---|---|---|---|
| Prompt Depth Anything ViT-L | [`depth-anything/prompt-depth-anything-vitl-hf`](https://hf.co/depth-anything/prompt-depth-anything-vitl-hf) | 340M | Apache-2.0 | ADA ToF + RGB, self-supervised | ToF is 224 x 171 with 6-12 mm noise, so inclination and leaf-area traits may fail E1 |
| Prompt Depth Anything ViT-S | [`depth-anything/prompt-depth-anything-vits-hf`](https://hf.co/depth-anything/prompt-depth-anything-vits-hf) | 25M | Apache-2.0 | same | same, fast variant for local GPUs |
| Mask2Former Swin-L (COCO instance) | [`facebook/mask2former-swin-large-coco-instance`](https://hf.co/facebook/mask2former-swin-large-coco-instance) | 216M | see model card | [PhenoBench](https://www.phenobench.org) sugar beet plant + leaf instances | ExG fails on dark 06 h frames; neighbours merge with the centre plant; per-leaf traits |
| SAM 3 (optional, gated) | [`facebook/sam3`](https://hf.co/facebook/sam3) | 848M | SAM licence | not set up | text-prompted zero-shot "sugar beet leaf" masks, for comparison |

Weights are in `models/` (not in git). Download them with:

```bash
python scripts/download_models.py
```

## 1. Depth refinement: Prompt Depth Anything

Prompt Depth Anything takes an RGB image and a low-resolution metric depth map (the
"prompt"). It returns dense metric depth at the RGB resolution. It was built for iPhone LiDAR at
256 x 192, which is close to the Pico Flexx at 224 x 171.

How it is used here:

- Image: the undistorted 1920 x 1080 RGB frame, cropped to the ToF footprint (1459 x 1080 px)
  and resized to 518 x 700. This is about 0.9 mm per pixel on the soil, against about 2.8 mm for
  the raw ToF.
- Prompt: the ToF points of the same frame, projected into the RGB camera with the ADA
  calibration and the robot positions, then splatted onto a 120 x 162 grid. Empty cells take
  the nearest value, because the model rescales the prompt by its min and max.
- Supervision: no labels are needed. The 15 depth frames of a run are taken from different
  robot positions. The ToF points of the frames in the same robot column are projected into RGB
  frame *k*, a z-buffer removes occluded points, and an L1 loss is applied at those points. The
  model learns to agree with several independent ToF views. This removes noise and uses the RGB
  to sharpen leaf edges.
- Output (`uiwue.ml.promptda.infer.DepthRefiner`): a dense depth map plus its intrinsics.
  `points_in_depth_frame()` returns the points in the same frame as the raw ToF cloud. The 3D
  pipeline can therefore use it in place of the raw cloud.

Smoke tests on 2026-11-14 (local RTX 5070 Ti; training plant `W1_J5`, validation plant `W2_J5`,
15 h runs, 2025-08-20 to 2025-09-10). Errors are measured on ToF points from depth frames other
than the prompt frame, so the noise is independent:

| Model | Median error | p90 error | MAE | Share within 10 mm |
|---|---|---|---|---|
| Raw ToF, bilinear upsampling (baseline) | 8.75 mm | 81.3 mm | 25.5 mm | 54 % |
| ViT-S, zero-shot | 14.25 mm | 86.8 mm | 30.2 mm | 38 % |
| ViT-S, 300 steps on 110 samples | 8.25 mm | 72.8 mm | 23.0 mm | 56 % |

Zero-shot ViT-L already gives sharp per-leaf depth by eye. Its metric scale is off, and
fine-tuning fixes that.

Full run on the GPU server, 2026-11-14. ViT-L was trained on 19,059 samples (18 plants) and validated
on 2,074 samples from the held-out plants W1_A8 and W2_A8. It runs 11,910 steps at
0.29 s/step:

| Model | Median error | p90 error | MAE | Share within 10 mm |
|---|---|---|---|---|
| Raw ToF, bilinear upsampling (baseline) | 6.75 mm | 68.3 mm | 23.8 mm | 61 % |
| ViT-L, zero-shot | 22.3 mm | 114.8 mm | 43.2 mm | 26 % |
| ViT-L, step 1000 | 6.25 mm | 57.3 mm | 21.6 mm | 64 % |
| ViT-L, best (step 11000 of 11910) | 5.75 mm | 52.3 mm | 20.4 mm | 66 % |

The best checkpoint is in `runs/promptda-vitl/best/` on the workstation (1.36 GB, not in git).
By eye on a held-out frame (W2_A8, 2025-09-10, 15 h), the fine-tuned output is denoised and
metric. Leaf outlines are smoother than in the zero-shot output, because the L1 loss on noisy
targets blurs thin edges. If petioles or leaf edges matter for a trait, compare both. The noise in the target ToF points (about 7 mm) limits how low the median error can go.
Most of the gain is in p90 and MAE, which come from the leaf edges.

### Data facts found while building this (check against the plan)

- **The 15 frames of a run are not repeats of one view.** They form a 3 x 5 grid of robot
  positions: x = 0.78, 0.88 and 0.98 m, and y in steps of 0.1 m. Frame 7 is the centre, and
  each column is captured as one sweep about 40 s apart. Plan section 4.7 (median over the 15
  frames) and E1 (noise floor from frame-to-frame MAD) assume repeat views. Both need
  rethinking: the centre plant moves by about 300 px between neighbouring frames.
- **ADA's robot-to-depth mapping has the sign of robot x wrong.** It is `rotz(+90°)`, which maps
  robot x to depth +y; the correct mapping is robot x to depth −y.
  - With ADA's mapping, frames from different columns disagree by 14-52 mm. With the
    corrected one they agree as well as frames within a column (W1_J5 2025-08-20: 14.0 to
    5.3 mm; W2_J5 2025-09-03: 52.2 to 11.1 mm).
  - Earlier I put this down to leaf motion between sweeps. That was wrong.
  - The fix is in `calib.R_ROBOT_TO_DEPTH` (ADA's original is `R_ROBOT_TO_DEPTH_ADA`).
  - Robot y maps the same way under both, so the same-column PromptDA samples (the default)
    are unaffected. `prep.py --all-views` was wrong before the fix and is valid after it; it
    gives about 3x more supervision points per sample.
- On 2025-08-31 the depth of `W1_J5` ranges from 0.27 m to 0.53 m. The 0.60 m sensor height in
  plan section 2 does not hold. Measure the reference level from the soil instead (plan
  section 4.3).
- About 30 % of the ToF view has no RGB pixel. The cameras are 4-5 cm apart and the ToF
  vertical field of view is wider than the RGB one.

## 2. Plant and leaf instances: Mask2Former on PhenoBench

PhenoBench has 1,407 training and 772 validation UAV images of sugar beet (1024 x 1024, about
1 mm per pixel). It has plant-instance and leaf-instance masks. The model is trained with two
classes: `sugar_beet_plant` and `sugar_beet_leaf`. Weeds are dropped. The training uses scale
augmentation from 0.5x to 2x, because ADA RGB at 0.6 m is about 0.43 mm per pixel. Inference
on ADA frames runs at 0.5x.

Use in the 3D pipeline: segment the undistorted RGB frame, take the plant instance nearest the
centre (`Segmentation.centre_plant()`), and keep only the ToF points that project inside it.
This replaces ExG + DBSCAN isolation when neighbours touch, and it also works on dark frames. The
leaf masks make per-leaf traits possible (plan section 11).

Status, 2026-11-14:

- v1, stopped at step 1000 of 8,750. On PhenoBench val: plant AP 0.46 (AP50 0.80), leaf AP 0.44
  (AP50 0.78). The checkpoint is in `runs/leafseg-m2f-step1000/` on the workstation.
- How v1 does on ADA:
  - Healthy plants: leaves are found well, but plant instances break up on large canopies.
  - Wilted, dark W2 plants after the cutoff (W2_J5, 2025-09-12): most leaves are missed. Leaf
    cover is 0.16 at the default threshold and 0.29 with threshold 0.15 plus gamma 0.6.
    PhenoBench has no wilted or dark images.
- v2 adds colour augmentation towards ADA conditions (`photometric()` in `leafseg/data.py`):
  gamma 0.6-2.2 to darken, hue shifted towards yellow and saturation lowered on green pixels
  only, and light blur. It also centres 80 % of crops on a plant pixel. v2 is queued on the GPU server
  after PromptDA.
- Before trusting the masks on stressed plants, check `python -m uiwue.ml.leafseg.infer ...`
  overlays on dates after 2025-09-08. If wilted leaves are still missed, try SAM 3 text prompts
  or pseudo-labels on ADA frames.

## Running

### Locally (Windows, one GPU)

```bash
python -m uiwue.ml.promptda.prep --data-dir data --out data/derived/promptda --jobs 8
```

```bash
python -m uiwue.ml.promptda.train --model models/promptda-vits --batch-size 4 --epochs 3
```

```bash
python -m uiwue.ml.leafseg.train --size 512 --batch-size 2
```

### GPU machine

Used so far: 1 x RTX PRO 6000 Blackwell (96 GB), 8 CPUs, 31 GB RAM. Its home disk was
nearly full, so all work went into a folder under `/tmp`, which is cleaned after some days
unused: copy checkpoints back. The scripts also run on multi-GPU machines (e.g. 4 x A100).

The workstation builds the depth samples (32 cores, raw data already here) and pushes only
what training needs. The raw ADA data stays on the workstation.

1. Push the code, PhenoBench (about 5.4 GB) and the depth samples, from Git Bash on the
   workstation. `GPU_HOST`, `GPU_DIR` and `GPU_KEY` select another machine:

   ```bash
   scripts/push_gpu.sh code phenobench promptda
   ```

2. On the GPU machine, in the pushed folder, set up the environment and download the weights.
   uv, Python 3.12 and the Hugging Face cache go into `<folder>/.cache`, not `/home`:

   ```bash
   bash scripts/setup_gpu.sh
   ```

3. Train. `leafseg` or `promptda` uses all GPUs. `both` splits the GPUs in half; on one GPU,
   both jobs share it:

   ```bash
   bash scripts/train_gpu.sh leafseg
   ```

   ```bash
   bash scripts/train_gpu.sh promptda
   ```

   To run several models one after the other, detached from the SSH session, use `seq`. Each
   model waits until no other `uiwue` trainer is running:

   ```bash
   bash scripts/train_gpu.sh seq promptda leafseg
   ```

4. Watch progress with `tail -f logs/train_*.log` or `.venv/bin/tensorboard --logdir runs`. The
   best checkpoints go to `runs/<name>/best/` together with `metrics.json`. Copy them back into
   `runs/` on the workstation.

On a machine with about 70 GB free, `bash scripts/setup_gpu.sh data prep` downloads the raw
data and builds the samples there instead.

### SAM 3 (optional)

SAM 3 is gated: accept its licence on the model page with your Hugging Face account, then log in
locally before downloading:

```bash
hf auth login --force
```

```bash
python scripts/download_models.py --only sam3
```
