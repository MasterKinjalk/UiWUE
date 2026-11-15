"""Build Prompt Depth Anything fine-tuning samples from ADA 2025 depth + RGB runs.

Idea: the Pico Flexx ToF depth (224 x 171, 6-12 mm noise) is the *prompt*; the 1920 x 1080 RGB
frame is the *image*. One run has 15 depth frames on a 3 x 5 grid of robot positions (0.1 m
steps), so every RGB frame k is seen by up to 15 depth frames from different viewpoints.
Supervision for RGB frame k = ToF points of the depth frames j in the same robot column as k
(see COLUMN_TOL; ``--all-views`` uses all 15) projected into RGB camera k
(depth j -> RGB k extrinsic from the robot positions), with a z-buffer against occlusion. The
model learns to turn one noisy low-res ToF frame plus RGB into a dense depth map that agrees
with the multi-view consensus: denoising + 6x upsampling, at no labelling cost.

One sample per (run, RGB frame k):
    <out>/<plant>/<date>_<hour>_<k>.jpg    undistorted RGB crop, resized to (H, W)
    <out>/<plant>/<date>_<hour>_<k>.npz    prompt (Hp, Wp) float16 metres, holes filled
                                           prompt_valid (Hp, Wp) bool
                                           sup (N, 3) float16: u, v (output px), z (m)
                                           sup_src (N,) uint8: depth frame of each point
                                           crop (4,) int: x0, y0, w, h in undistorted RGB px
    <out>/index.csv                        one row per sample

Usage:
    python -m uiwue.ml.promptda.prep --data-dir data --out data/derived/promptda \
        --frames 1 4 7 10 13 --jobs 16
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage

from uiwue.threed import calib
from uiwue.threed.io import Run, list_runs, load_depth, load_rgb, read_frames

# Output geometry. The depth footprint is ~1460 x 1080+ px in the RGB image (depth VFOV 43.6
# deg > RGB VFOV 42.4 deg), so the crop is full RGB height and 1459 px wide (aspect 1.351),
# resized to 518 x 700 (both multiples of the ViT patch size 14).
OUT_H, OUT_W = 518, 700
CROP_H = calib.RGB_H
CROP_W = round(CROP_H * OUT_W / OUT_H)  # 1459
# Prompt grid ~0.75x the ToF resolution so nearly every cell receives a ToF point.
PROMPT_H, PROMPT_W = 120, 162

Z_MIN, Z_MAX = 0.15, 0.90  # valid ToF range (m)
EDGE_JUMP = 0.02  # drop ToF "flying pixels": 3x3 depth range above this (m)
ZBUF_CELL = 2  # z-buffer cell size in output px for occlusion tests
ZBUF_TOL = 0.01  # keep points within this of the nearest surface in their cell (m)
N_SUP_MAX = 80_000
# The 15 frames are 3 sweeps along robot y (columns at robot x = 0.78 / 0.88 / 0.98 m). With
# ADA's robot->depth mapping, frames from another column disagreed by 26-48 mm; the cause was a
# sign error for robot x, fixed in calib.R_ROBOT_TO_DEPTH (2026-11-14). Cross-column frames now
# agree like same-column ones, so ``all_views=True`` is valid. The default stays same-column,
# which is what the first trained checkpoint (runs/promptda-vitl) used.
COLUMN_TOL = 0.02  # m, robot x difference that still counts as the same column

_RAYS: np.ndarray | None = None


def depth_rays() -> np.ndarray:
    """Undistorted normalised camera rays (x/z, y/z) for every depth pixel, (H, W, 2)."""
    global _RAYS
    if _RAYS is None:
        u, v = np.meshgrid(np.arange(calib.DEPTH_W), np.arange(calib.DEPTH_H))
        pix = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        rays = cv2.undistortPoints(pix, calib.K_DEPTH, calib.DEPTH_DISTORTION)
        _RAYS = rays.reshape(calib.DEPTH_H, calib.DEPTH_W, 2)
    return _RAYS


def depth_points(depth: np.ndarray, drop_edges: bool = True) -> np.ndarray:
    """Valid ToF pixels -> (N, 3) points in the depth camera frame.

    ``drop_edges`` removes "flying pixels" (3x3 depth range above EDGE_JUMP); used for the
    supervision points, not for the prompt (inference sees raw ToF too).
    """
    valid = (depth > Z_MIN) & (depth < Z_MAX)
    if drop_edges:
        zmax = ndimage.maximum_filter(np.where(valid, depth, 0), size=3)
        zmin = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=3)
        valid &= (zmax - zmin) < EDGE_JUMP
    rays = depth_rays()[valid]
    z = depth[valid].astype(np.float64)
    return np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z])


def project(points: np.ndarray, t_depth_to_rgb: np.ndarray) -> np.ndarray:
    """Depth-frame points -> (N, 3) undistorted RGB pixel u, v and RGB-frame depth z."""
    p = points @ t_depth_to_rgb[:3, :3].T + t_depth_to_rgb[:3, 3]
    z = p[:, 2]
    k = calib.K_RGB
    u = k[0, 0] * p[:, 0] / z + k[0, 2]
    v = k[1, 1] * p[:, 1] / z + k[1, 2]
    return np.column_stack([u, v, z])


def crop_for(uvz_self: np.ndarray) -> tuple[int, int, int, int]:
    """Fixed-size crop centred on the footprint of the frame's own depth points."""
    cu = float(np.median(uvz_self[:, 0])) if len(uvz_self) else calib.RGB_W / 2
    x0 = int(round(np.clip(cu - CROP_W / 2, 0, calib.RGB_W - CROP_W)))
    return x0, 0, CROP_W, CROP_H


def to_output_px(
    uvz: np.ndarray, crop: tuple[int, int, int, int], out_hw: tuple[int, int]
) -> np.ndarray:
    """RGB pixels -> pixel coordinates of an (h, w) grid over the crop; drops points outside."""
    x0, y0, cw, ch = crop
    h, w = out_hw
    # pixel-centre convention: crop pixel c maps to (c + 0.5) * w / cw - 0.5
    u = (uvz[:, 0] - x0 + 0.5) * (w / cw) - 0.5
    v = (uvz[:, 1] - y0 + 0.5) * (h / ch) - 0.5
    keep = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (uvz[:, 2] > Z_MIN)
    return np.column_stack([u[keep], v[keep], uvz[keep, 2]])


def zbuffer_keep(uvz: np.ndarray, out_hw: tuple[int, int], cell: int, tol: float) -> np.ndarray:
    """Boolean mask of points within ``tol`` of the nearest point in their ``cell`` x ``cell`` bin."""
    h, w = out_hw
    gw = w // cell + 1
    idx = (uvz[:, 1] // cell).astype(np.int64) * gw + (uvz[:, 0] // cell).astype(np.int64)
    zmin = np.full((h // cell + 1) * gw, np.inf)
    np.minimum.at(zmin, idx, uvz[:, 2])
    return uvz[:, 2] <= zmin[idx] + tol


def splat_prompt(uvz_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points on the output grid -> low-res prompt (nearest surface per cell), holes filled
    with the nearest valid value (the model min/max-normalises the prompt, so no zeros)."""
    sx, sy = PROMPT_W / OUT_W, PROMPT_H / OUT_H
    pu = np.clip(((uvz_out[:, 0] + 0.5) * sx).astype(np.int64), 0, PROMPT_W - 1)
    pv = np.clip(((uvz_out[:, 1] + 0.5) * sy).astype(np.int64), 0, PROMPT_H - 1)
    prompt = np.full(PROMPT_H * PROMPT_W, np.inf)
    np.minimum.at(prompt, pv * PROMPT_W + pu, uvz_out[:, 2])
    prompt = prompt.reshape(PROMPT_H, PROMPT_W)
    valid = np.isfinite(prompt)
    if not valid.any():
        return np.zeros_like(prompt, dtype=np.float32), valid
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return prompt[iy, ix].astype(np.float32), valid


def build_run(
    run: Run,
    data_dir: Path,
    out_dir: Path,
    frames: list[int],
    rng_seed: int,
    all_views: bool = False,
) -> list[dict]:
    """All samples of one run. Returns index rows (empty if the run lacks depth/RGB/poses)."""
    ddir, rdir = run.dir(data_dir, "depth"), run.dir(data_dir, "rgb")
    pos_d, pos_r = read_frames(ddir / "frames_depth.csv"), read_frames(rdir / "frames_rgb.csv")
    if not pos_d or not pos_r:
        return []

    clouds, raw = {}, {}
    for j, pos in pos_d.items():
        f = ddir / f"depth_{j:03d}.tif"
        if f.exists():
            depth = load_depth(f)[0]
            pts = depth_points(depth)
            if len(pts) > 500:
                clouds[j] = (pts, pos)
                if j in frames:
                    raw[j] = depth_points(depth, drop_edges=False)
    if not clouds:
        return []

    rng = np.random.default_rng(rng_seed)
    rows = []
    pdir = out_dir / run.plant_id
    pdir.mkdir(parents=True, exist_ok=True)
    for k in frames:
        if k not in pos_r or k not in clouds:
            continue
        rgb = load_rgb(rdir / f"rgb_{k:03d}.jpg")
        if rgb is None:
            continue

        # Own depth frame k -> prompt; all depth frames j -> supervision.
        own = project(raw[k], calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, clouds[k][1], pos_r[k]))
        crop = crop_for(own)
        own_out = to_output_px(own, crop, (OUT_H, OUT_W))
        if len(own_out) < 1000:
            continue
        prompt, prompt_valid = splat_prompt(own_out)

        sup, src = [], []
        for j, (pts, pos) in clouds.items():
            if not all_views and abs(pos[0] - clouds[k][1][0]) > COLUMN_TOL:
                continue
            t = calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, pos, pos_r[k])
            uvz = to_output_px(project(pts, t), crop, (OUT_H, OUT_W))
            sup.append(uvz)
            src.append(np.full(len(uvz), j, np.uint8))
        sup, src = np.concatenate(sup), np.concatenate(src)
        keep = zbuffer_keep(sup, (OUT_H, OUT_W), ZBUF_CELL, ZBUF_TOL)
        sup, src = sup[keep], src[keep]
        if len(sup) > N_SUP_MAX:
            sel = rng.choice(len(sup), N_SUP_MAX, replace=False)
            sup, src = sup[sel], src[sel]

        und = cv2.undistort(rgb, calib.K_RGB, calib.RGB_DISTORTION)
        x0, y0, cw, ch = crop
        img = cv2.resize(
            und[y0 : y0 + ch, x0 : x0 + cw], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA
        )

        stem = f"{run.date}_{run.hour}_{k:02d}"
        cv2.imwrite(
            str(pdir / f"{stem}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        np.savez_compressed(
            pdir / f"{stem}.npz",
            prompt=prompt.astype(np.float16),
            prompt_valid=prompt_valid,
            sup=sup.astype(np.float16),
            sup_src=src,
            crop=np.array(crop),
            frame=np.array(k),
        )
        rows.append(
            dict(
                plant_id=run.plant_id,
                date=run.date,
                hour=run.hour,
                frame=k,
                path=f"{run.plant_id}/{stem}",
                n_prompt=int(prompt_valid.sum()),
                n_sup=len(sup),
                n_depth_frames=len(clouds),
                n_sup_frames=len(np.unique(src)),
            )
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="data/derived/promptda")
    p.add_argument("--plants", nargs="+", help="full IDs, e.g. W1_J5 (default: all on disk)")
    p.add_argument("--dates", nargs=2, metavar=("START", "END"), help="e.g. 2025_08_12 2025_09_19")
    p.add_argument("--hours", nargs="+", help="e.g. 09 12 15 (default: all)")
    p.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[1, 4, 7, 10, 13],
        help="RGB frames per run to use as images (default: middle column of the 3x5 grid)",
    )
    p.add_argument(
        "--all-views",
        action="store_true",
        help="supervise with all 15 depth frames, not only the prompt frame's column",
    )
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--limit", type=int, help="process only the first N runs (smoke test)")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_runs = set(
        list_runs(
            data_dir, "rgb", args.plants, tuple(args.dates) if args.dates else None, args.hours
        )
    )
    runs = [
        r
        for r in list_runs(
            data_dir, "depth", args.plants, tuple(args.dates) if args.dates else None, args.hours
        )
        if r in rgb_runs
    ]
    if args.limit:
        runs = runs[: args.limit]
    print(f"{len(runs)} runs with depth + RGB")

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(build_run)(r, data_dir, out_dir, args.frames, i, args.all_views)
        for i, r in enumerate(runs)
    )
    rows = [row for rs in results for row in rs]
    index = out_dir / "index.csv"
    old = []
    if index.exists():  # merge with earlier batches (other plants)
        done = {row["path"] for row in rows}
        with open(index, newline="") as f:
            old = [r for r in csv.DictReader(f) if r["path"] not in done]
    if rows or old:
        with open(index, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list((rows or old)[0].keys()))
            w.writeheader()
            w.writerows(old + rows)
    print(f"{len(rows)} samples written, index: {index} ({len(old) + len(rows)} rows)")


if __name__ == "__main__":
    main()
