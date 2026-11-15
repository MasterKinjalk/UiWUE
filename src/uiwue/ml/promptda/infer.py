"""Refine one ADA depth frame with a (fine-tuned) Prompt Depth Anything model.

Output is a dense depth map on the RGB crop grid (518 x 700, ~0.9 mm/px at the soil instead of
~2.8 mm/px for the raw ToF), plus the pinhole intrinsics of that grid, so the 3D pipeline can
unproject it like any depth image:

    from uiwue.ml.promptda.infer import DepthRefiner
    ref = DepthRefiner("runs/promptda-vitl/best")          # or models/promptda-vitl (zero-shot)
    out = ref.refine_frame(Run("W2_J5", "2025_09_08", "15"), k=7, data_dir="data")
    pts_depth_frame = out.points_in_depth_frame()           # (N, 3), same frame as the ToF cloud

CLI (writes an .npz per frame and a side-by-side PNG):
    python -m uiwue.ml.promptda.infer --model runs/promptda-vitl/best --plant W2_J5 \
        --date 2025_09_08 --hour 15 --frames 7 --out data/derived/promptda_infer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from uiwue.ml.promptda import prep
from uiwue.ml.promptda.data import normalise_rgb
from uiwue.threed import calib
from uiwue.threed.io import Run, load_depth, load_rgb, read_frames


@dataclass
class Refined:
    depth: np.ndarray  # (OUT_H, OUT_W) metres, RGB camera z
    rgb: np.ndarray  # (OUT_H, OUT_W, 3) uint8, undistorted crop
    prompt: np.ndarray  # (PROMPT_H, PROMPT_W) the ToF prompt
    k_out: np.ndarray  # 3x3 intrinsics of the output grid
    t_depth_to_rgb: np.ndarray  # 4x4 extrinsic used (depth frame k -> RGB frame k)

    def points_rgb_frame(self) -> np.ndarray:
        h, w = self.depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = self.depth.ravel()
        x = (u.ravel() - self.k_out[0, 2]) * z / self.k_out[0, 0]
        y = (v.ravel() - self.k_out[1, 2]) * z / self.k_out[1, 1]
        return np.column_stack([x, y, z])

    def points_in_depth_frame(self) -> np.ndarray:
        """Dense points expressed in the ToF camera frame (same frame as the raw cloud)."""
        p = self.points_rgb_frame()
        t = np.linalg.inv(self.t_depth_to_rgb)
        return p @ t[:3, :3].T + t[:3, 3]


def output_intrinsics(crop: tuple[int, int, int, int]) -> np.ndarray:
    """K of the (OUT_H, OUT_W) grid over ``crop`` of the undistorted RGB image."""
    x0, y0, cw, ch = crop
    sx, sy = prep.OUT_W / cw, prep.OUT_H / ch
    k = calib.K_RGB.copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] - x0 + 0.5) * sx - 0.5
    k[1, 2] = (k[1, 2] - y0 + 0.5) * sy - 0.5
    return k


class DepthRefiner:
    def __init__(self, model: str | Path, device: str | None = None, dtype=torch.bfloat16):
        from transformers import PromptDepthAnythingForDepthEstimation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.model = PromptDepthAnythingForDepthEstimation.from_pretrained(model).to(self.device, self.dtype).eval()

    @torch.no_grad()
    def refine(self, rgb: np.ndarray, depth: np.ndarray, pos_depth, pos_rgb) -> Refined:
        """rgb: raw 1920x1080 RGB; depth: ToF metres (after io.load_depth); poses from frames_*.csv."""
        t = calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, pos_depth, pos_rgb)
        own = prep.project(prep.depth_points(depth, drop_edges=False), t)
        crop = prep.crop_for(own)
        prompt, _ = prep.splat_prompt(prep.to_output_px(own, crop, (prep.OUT_H, prep.OUT_W)))
        und = cv2.undistort(rgb, calib.K_RGB, calib.RGB_DISTORTION)
        x0, y0, cw, ch = crop
        img = cv2.resize(und[y0 : y0 + ch, x0 : x0 + cw], (prep.OUT_W, prep.OUT_H), interpolation=cv2.INTER_AREA)
        px = normalise_rgb(img)[None].to(self.device, self.dtype)
        pr = torch.from_numpy(prompt)[None, None].to(self.device, self.dtype)
        pred = self.model(pixel_values=px, prompt_depth=pr).predicted_depth[0].float().cpu().numpy()
        return Refined(pred, img, prompt, output_intrinsics(crop), t)

    def refine_frame(self, run: Run, k: int, data_dir: str | Path) -> Refined | None:
        ddir, rdir = run.dir(data_dir, "depth"), run.dir(data_dir, "rgb")
        pos_d, pos_r = read_frames(ddir / "frames_depth.csv"), read_frames(rdir / "frames_rgb.csv")
        rgb = load_rgb(rdir / f"rgb_{k:03d}.jpg")
        if k not in pos_d or k not in pos_r or rgb is None:
            return None
        depth, _ = load_depth(ddir / f"depth_{k:03d}.tif")
        return self.refine(rgb, depth, pos_d[k], pos_r[k])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/promptda-vitl")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--plant", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--hour", required=True)
    p.add_argument("--frames", nargs="+", type=int, default=[7])
    p.add_argument("--out", default="data/derived/promptda_infer")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = DepthRefiner(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(args.plant, args.date, args.hour)
    for k in args.frames:
        r = ref.refine_frame(run, k, args.data_dir)
        if r is None:
            print(f"frame {k}: missing data")
            continue
        stem = out / f"{run.plant_id}_{run.date}_{run.hour}_{k:02d}"
        np.savez_compressed(
            f"{stem}.npz",
            depth=r.depth.astype(np.float32),
            k_out=r.k_out,
            t_depth_to_rgb=r.t_depth_to_rgb,
            prompt=r.prompt,
        )
        lo, hi = np.percentile(r.depth, [1, 99])
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].imshow(r.rgb)
        ax[1].imshow(r.prompt, cmap="turbo", vmin=lo, vmax=hi)
        ax[1].set_title("ToF prompt")
        ax[2].imshow(r.depth, cmap="turbo", vmin=lo, vmax=hi)
        ax[2].set_title("refined depth")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(f"{stem}.png", dpi=80)
        plt.close(fig)
        print(f"frame {k}: {stem}.npz")


if __name__ == "__main__":
    main()
