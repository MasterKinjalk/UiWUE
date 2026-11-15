# Pretrained models to fine-tune

Two pretrained models fit the Challenge 3 plan
([plan/Kinjalk-Work-Plan.md](../plan/Kinjalk-Work-Plan.md)). Each one targets a risk listed in
the plan. Neither needs hand labels on ADA data.

| Name | Weights | Size | Licence | Fine-tuned on | Plan risk it addresses |
|---|---|---|---|---|---|
| Prompt Depth Anything ViT-L | [`depth-anything/prompt-depth-anything-vitl-hf`](https://hf.co/depth-anything/prompt-depth-anything-vitl-hf) | 340M | Apache-2.0 | ADA ToF + RGB, self-supervised | ToF is 224 x 171 with 6-12 mm noise, so inclination and leaf-area traits may fail E1 |
| Prompt Depth Anything ViT-S | [`depth-anything/prompt-depth-anything-vits-hf`](https://hf.co/depth-anything/prompt-depth-anything-vits-hf) | 25M | Apache-2.0 | same | same, fast variant for local GPUs |
| Mask2Former Swin-L (COCO instance) | [`facebook/mask2former-swin-large-coco-instance`](https://hf.co/facebook/mask2former-swin-large-coco-instance) | 216M | see model card | [PhenoBench](https://www.phenobench.org) sugar beet plant + leaf instances | neighbours merge with the centre plant; per-leaf traits (it does not help on dark 06 h frames, section 2) |
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

Use in the 3D pipeline (`run --plant-model`, `uiwue.threed.plantmask`, details and the
comparison in [`docs/threed/PIPELINE.md`](threed/PIPELINE.md), section "Learned plant mask"):

- The raw, distorted RGB frame goes into the model at 0.5x (`seg.forward(rgb, 0.5,
  undistort=False)`). The plant instances are cached at half resolution, and every ToF point
  with colour reads them at its own RGB pixel (`Cloud.rgb_uv`).
- The centre plant is the instance with the most points within 3 cm of the plant axis, plus any
  other instance with at least half of its points inside the plant cylinder (fragments of a
  split plant). Not `Segmentation.centre_plant()`: its image-centre rule picks a neighbour in
  frames other than 7.
- **It does not work on dark frames.** At 06 h (mean brightness about 4.7; 60 % of the 06 h runs
  are `qc_dark`, all of them in 2025-09-01..09-14) it finds no plant instance in 97.6 % of the
  dark frames (W1_J5 2025-09-08 06 h: 0 instances in all 3 used frames). Dark frames at 09 h and
  18 h-19 h (brightness 8-10) do get instances, but they are not checked. The pipeline keeps the
  height rule on every dark frame and wherever no instance reaches the centre, so the method
  changes across the day: in 08-24..09-14 the model is used in 1 % of the 06 h frames, 75 % at
  09 h, 95 % at 12 h, 92 % at 15 h and 65 % at 18 h.
- The leaf masks make per-leaf traits possible (plan section 11); they are not used yet.

`instance_masks` in `infer.py` replaces `post_process_instance_segmentation`: the same masks and
scores (a test checks this against transformers), 8 ms instead of 1.8 s per frame, because only
(query, class) pairs that can pass the score threshold are upsampled. The model itself takes about
0.1 s per frame on the local RTX 5070 Ti in fp32 (fp16/bf16 autocast was slower, and bf16 changed
some masks).

Status, 2026-11-15:

- v1, stopped at step 1000 of 8,750. On PhenoBench val: plant AP 0.46 (AP50 0.80), leaf AP 0.44
  (AP50 0.78). The checkpoint is in `runs/leafseg-m2f-step1000/` on the workstation.
- How v1 does on ADA:
  - Healthy plants: leaves are found well, but plant instances break up on large canopies.
  - Wilted, dark W2 plants after the cutoff (W2_J5, 2025-09-12): most leaves are missed. Leaf
    cover is 0.16 at the default threshold and 0.29 with threshold 0.15 plus gamma 0.6.
    PhenoBench has no wilted or dark images.
- v2 adds colour augmentation towards ADA conditions (`photometric()` in `leafseg/data.py`):
  gamma 0.6-2.2 to darken, hue shifted towards yellow and saturation lowered on green pixels
  only, and light blur. It also centres 80 % of crops on a plant pixel.
- v2 checkpoint: `runs/leafseg-m2f/best`, step 8,000 of about 8,800 (50 epochs, 768 px crops,
  batch 8, 200 queries). PhenoBench val: plant AP 0.54 (AP50 0.90), leaf AP 0.49 (AP50 0.81).
- How v2 does on ADA (frame 7 at 15 h, IoU of its points with the ExG-isolated plant, points with
  colour inside the plant cylinder):

  | Run | Case | IoU | What differs |
  |---|---|---|---|
  | W1_J5 2025-08-20 | seedling | 0.89 | |
  | W1_J5 2025-09-03 | closed canopy | 0.71 | 973 points only in ExG: a neighbour's leaves over the pot |
  | W2_J5 2025-09-12 | wilted | 0.57 | 406 points only in ExG: thin petioles and wilting leaf fringes of the plant itself, and some neighbour leaves |
  | W1_J5 2025-09-08 06 h | dark, brightness 4.4 | – | no plant instance |

  Over 28 plants, 09-18 h, 09-01..09-14 (after the voxel vote and DBSCAN,
  `scripts/plant_mask_agreement.py`), the median IoU is 0.75 / 0.77 for W1 before / after 09-08
  and 0.74 / 0.71 for W2. The wilted W2 frames have the lowest IoU (p25 0.55) and four times as
  many frames without an instance at the centre (4.0 % against 1 %).

- Failure modes seen on ADA, in order of how often they matter:
  - dark frames (above);
  - thin structures: the mask logits are at 1/8 of the frame, so petioles and the thin fringes
    of wilting leaves fall out of the mask. This hits the wilted deficit bed most and steepens
    the area-type treatment slopes (PIPELINE.md, "Learned plant mask");
  - large W1 plants from about 2025-09-04 split into several instances (W1_J5 2025-09-04: the
    instance at the centre is only the heart of the plant). The fragment merge recovers part of
    this;
  - the centre plant and a neighbour merged into one instance (W2_J5 2025-09-06).

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
