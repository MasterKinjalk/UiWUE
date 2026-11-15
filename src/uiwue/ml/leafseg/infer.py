"""Plant + leaf instance masks on ADA RGB frames with the PhenoBench-fine-tuned Mask2Former.

ADA RGB at ~0.43 mm/px is about 2x finer than PhenoBench (~1 mm/px), so frames are run at
``--scale 0.5`` by default. Masks are returned at full RGB resolution, undistorted with the ADA
calibration so they line up with ``uiwue.threed`` point projections.

    from uiwue.ml.leafseg.infer import LeafSegmenter
    seg = LeafSegmenter("runs/leafseg-m2f/best")
    res = seg.segment(rgb)                     # rgb: (1080, 1920, 3) uint8, raw frame
    res.plants, res.leaves                     # (n, H, W) bool, undistorted pixel grid
    res.centre_plant()                         # mask of the plant nearest the image centre

CLI:
    python -m uiwue.ml.leafseg.infer --model runs/leafseg-m2f/best --plant W2_J5 \
        --date 2025_09_08 --hour 15 --frames 7 --out data/derived/leafseg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from uiwue.ml.leafseg.data import normalise_rgb
from uiwue.threed import calib
from uiwue.threed.io import Run, load_rgb


@dataclass
class Segmentation:
    plants: np.ndarray  # (n_p, H, W) bool
    plant_scores: np.ndarray
    leaves: np.ndarray  # (n_l, H, W) bool
    leaf_scores: np.ndarray

    def centre_plant(self, centre: tuple[float, float] | None = None) -> np.ndarray | None:
        """Plant mask whose centroid is nearest ``centre`` (default: image centre)."""
        if len(self.plants) == 0:
            return None
        h, w = self.plants.shape[1:]
        cx, cy = centre or (w / 2, h / 2)
        d = []
        for m in self.plants:
            ys, xs = np.nonzero(m)
            d.append(np.hypot(xs.mean() - cx, ys.mean() - cy))
        return self.plants[int(np.argmin(d))]


class LeafSegmenter:
    def __init__(self, model: str | Path, device: str | None = None, threshold: float = 0.5):
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Mask2FormerImageProcessor.from_pretrained(model)
        self.model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(model).to(self.device).eval()
        )
        self.threshold = threshold

    @torch.no_grad()
    def segment(self, rgb: np.ndarray, scale: float = 0.5, undistort: bool = True) -> Segmentation:
        img = cv2.undistort(rgb, calib.K_RGB, calib.RGB_DISTORTION) if undistort else rgb
        h, w = img.shape[:2]
        # Mask2Former's pixel decoder needs sizes divisible by 32.
        sh, sw = (int(round(h * scale / 32)) * 32, int(round(w * scale / 32)) * 32)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        out = self.model(pixel_values=normalise_rgb(small)[None].to(self.device))
        res = self.processor.post_process_instance_segmentation(
            out, threshold=self.threshold, target_sizes=[(h, w)], return_binary_maps=True
        )[0]
        masks = res["segmentation"]
        info = res["segments_info"]
        if masks is None or not info:
            empty = np.zeros((0, h, w), bool)
            return Segmentation(empty, np.zeros(0), empty, np.zeros(0))
        masks = masks.cpu().numpy().astype(bool)
        labels = np.array([s["label_id"] for s in info])
        scores = np.array([s["score"] for s in info])
        return Segmentation(
            masks[labels == 0], scores[labels == 0], masks[labels == 1], scores[labels == 1]
        )


def overlay(img: np.ndarray, masks: np.ndarray, alpha: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    for m in masks:
        out[m] = (1 - alpha) * out[m] + alpha * rng.integers(40, 255, 3)
    return out.astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--plant", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--hour", required=True)
    p.add_argument("--frames", nargs="+", type=int, default=[7])
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--out", default="data/derived/leafseg")
    args = p.parse_args()

    seg = LeafSegmenter(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(args.plant, args.date, args.hour)
    for k in args.frames:
        rgb = load_rgb(run.dir(args.data_dir, "rgb") / f"rgb_{k:03d}.jpg")
        if rgb is None:
            print(f"frame {k}: missing")
            continue
        r = seg.segment(rgb, args.scale)
        und = cv2.undistort(rgb, calib.K_RGB, calib.RGB_DISTORTION)
        stem = out / f"{run.plant_id}_{run.date}_{run.hour}_{k:02d}"
        np.savez_compressed(
            f"{stem}.npz",
            plants=r.plants,
            plant_scores=r.plant_scores,
            leaves=r.leaves,
            leaf_scores=r.leaf_scores,
        )
        both = np.concatenate([overlay(und, r.plants), overlay(und, r.leaves, seed=1)], axis=1)
        cv2.imwrite(
            f"{stem}.jpg", cv2.cvtColor(cv2.resize(both, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR)
        )
        print(f"frame {k}: {len(r.plants)} plants, {len(r.leaves)} leaves -> {stem}.npz")


if __name__ == "__main__":
    main()
