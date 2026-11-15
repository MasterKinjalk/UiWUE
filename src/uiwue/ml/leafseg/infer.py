"""Plant + leaf instance masks on ADA RGB frames with the PhenoBench-fine-tuned Mask2Former.

ADA RGB at ~0.43 mm/px is about 2x finer than PhenoBench (~1 mm/px), so frames are run at
``--scale 0.5`` by default. By default masks are returned at full RGB resolution, undistorted with
the ADA calibration. ``undistort=False`` keeps the raw (distorted) pixel grid, which is what
``Cloud.rgb_uv`` indexes; the 3D pipeline uses that (``uiwue.threed.plantmask``).

    from uiwue.ml.leafseg.infer import LeafSegmenter
    seg = LeafSegmenter("runs/leafseg-m2f/best")
    res = seg.segment(rgb)                     # rgb: (1080, 1920, 3) uint8, raw frame
    res.plants, res.leaves                     # (n, H, W) bool, undistorted pixel grid
    res.centre_plant()                         # mask of the plant nearest the image centre
    seg.segment(rgb, undistort=False, out_size=(540, 960))   # raw grid at half resolution

The masks do not hold on dark frames: on the 06 h runs (mean brightness about 4) the model finds
no plant at all, so the 3D pipeline falls back to its height rule there.

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
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model).to(self.device).eval()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, rgb: np.ndarray, scale: float = 0.5, undistort: bool = True):
        """Raw model outputs for one RGB frame (``class_queries_logits``, ``masks_queries_logits``)."""
        img = cv2.undistort(rgb, calib.K_RGB, calib.RGB_DISTORTION) if undistort else rgb
        h, w = img.shape[:2]
        # Mask2Former's pixel decoder needs sizes divisible by 32.
        sh, sw = (int(round(h * scale / 32)) * 32, int(round(w * scale / 32)) * 32)  # noqa: RUF046
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        return self.model(pixel_values=normalise_rgb(small)[None].to(self.device))

    @torch.no_grad()
    def segment(
        self,
        rgb: np.ndarray,
        scale: float = 0.5,
        undistort: bool = True,
        out_size: tuple[int, int] | None = None,
    ) -> Segmentation:
        """Plant and leaf instances of one frame. Masks are on the input pixel grid (undistorted
        when ``undistort``), or resized to ``out_size`` (H, W)."""
        h, w = rgb.shape[:2]
        out = self.forward(rgb, scale, undistort)
        masks, scores, labels = instance_masks(out, out_size or (h, w), self.threshold)
        return Segmentation(masks[labels == 0], scores[labels == 0], masks[labels == 1], scores[labels == 1])


@torch.no_grad()
def instance_masks(outputs, target_size: tuple[int, int], threshold: float = 0.5, label: int | None = None):
    """``Mask2FormerImageProcessor.post_process_instance_segmentation(return_binary_maps=True)``
    for one image, about 200x faster: (masks (n, H, W) bool, scores (n,), labels (n,)).

    Same rule as transformers 5.x: the top ``num_queries`` (query, class) pairs by class
    probability, mask logits upsampled to 384 x 384 (bilinear) and thresholded at 0, score =
    class probability x mean mask probability, kept when ``score >= threshold``, then resized to
    ``target_size`` (nearest). The difference: pairs whose class probability is below
    ``threshold`` are dropped before any mask is upsampled (their score cannot reach it), and the
    per-query Python loop is gone. Masks come out in descending score order. ``label`` keeps one
    class only (0 = plant, 1 = leaf).
    """
    import torch.nn.functional as F

    cls = outputs.class_queries_logits[0].float()
    logits = outputs.masks_queries_logits[0].float()
    n_q, n_c = cls.shape[0], cls.shape[1] - 1
    prob, idx = cls.softmax(-1)[:, :-1].flatten().topk(n_q, sorted=False)
    labels, query = idx % n_c, torch.div(idx, n_c, rounding_mode="floor")
    keep = prob >= threshold
    if label is not None:
        keep &= labels == label
    prob, labels, query = prob[keep], labels[keep], query[keep]
    h, w = target_size
    if len(query) == 0:
        return np.zeros((0, h, w), bool), np.zeros(0, np.float32), np.zeros(0, np.int64)
    m = F.interpolate(logits[query][None], size=(384, 384), mode="bilinear", align_corners=False)[0]
    binary = (m > 0).float()
    area = binary.flatten(1).sum(1)
    score = prob * (m.sigmoid().flatten(1) * binary.flatten(1)).sum(1) / (area + 1e-6)
    ok = (score >= threshold) & (area > 0)
    binary, score, labels = binary[ok], score[ok], labels[ok]
    order = torch.argsort(score, descending=True)
    binary, score, labels = binary[order], score[order], labels[order]
    if len(binary):
        binary = F.interpolate(binary[None], size=(h, w), mode="nearest")[0] > 0.5
        nonempty = binary.flatten(1).any(1)
        binary, score, labels = binary[nonempty], score[nonempty], labels[nonempty]
    return binary.bool().cpu().numpy(), score.cpu().numpy(), labels.cpu().numpy()


def overlay(img: np.ndarray, masks: np.ndarray, alpha: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    for m in masks:
        out[m] = (1 - alpha) * out[m] + alpha * rng.integers(40, 255, 3)
    return out.astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        cv2.imwrite(f"{stem}.jpg", cv2.cvtColor(cv2.resize(both, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
        print(f"frame {k}: {len(r.plants)} plants, {len(r.leaves)} leaves -> {stem}.npz")


if __name__ == "__main__":
    main()
