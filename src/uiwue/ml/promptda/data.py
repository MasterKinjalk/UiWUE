"""Dataset over the samples written by ``uiwue.ml.promptda.prep``."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from uiwue.ml.promptda.prep import PROMPT_H, PROMPT_W

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_index(root: Path) -> list[dict]:
    with open(Path(root) / "index.csv", newline="") as f:
        return list(csv.DictReader(f))


def normalise_rgb(img: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) RGB -> float (3, H, W) ImageNet-normalised."""
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class PromptDADataset(Dataset):
    """Returns dict(pixel_values (3,H,W), prompt_depth (1,Hp,Wp), sup (N,3), sup_other (N,) bool).

    ``sup`` holds u, v (output pixels), z (m) of multi-view ToF points; ``sup_other`` marks points
    from depth frames other than the prompt's own frame (independent noise, used for validation).
    A fixed ``n_points`` per sample is drawn (with replacement if needed) so batches stack.
    """

    def __init__(
        self,
        root: Path,
        plants: set[str] | None = None,
        exclude: set[str] | None = None,
        n_points: int = 40_000,
        min_prompt_fill: float = 0.3,
        augment: bool = False,
        seed: int = 0,
    ):
        self.root = Path(root)
        rows = read_index(self.root)
        if plants:
            rows = [r for r in rows if r["plant_id"] in plants]
        if exclude:
            rows = [r for r in rows if r["plant_id"] not in exclude]
        # Drop broken ToF frames (prompt cells with a ToF point below this share of the grid).
        rows = [r for r in rows if int(r["n_prompt"]) >= min_prompt_fill * PROMPT_H * PROMPT_W]
        self.rows = rows
        self.n_points = n_points
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        base = self.root / row["path"]
        img = cv2.cvtColor(cv2.imread(str(base) + ".jpg"), cv2.COLOR_BGR2RGB)
        z = np.load(str(base) + ".npz")
        prompt = z["prompt"].astype(np.float32)
        sup = z["sup"].astype(np.float32)
        other = z["sup_src"] != int(z["frame"])

        rng = np.random.default_rng(None if self.augment else self.seed + i)
        sel = rng.choice(len(sup), self.n_points, replace=len(sup) < self.n_points)
        sup, other = sup[sel], other[sel]

        if self.augment:
            h, w = img.shape[:2]
            if rng.random() < 0.5:  # horizontal flip
                img, prompt = img[:, ::-1], prompt[:, ::-1]
                sup[:, 0] = (w - 1) - sup[:, 0]
            if rng.random() < 0.5:  # vertical flip
                img, prompt = img[::-1], prompt[::-1]
                sup[:, 1] = (h - 1) - sup[:, 1]
            # mild photometric jitter (brightness / contrast), dark 06 h runs are in the data
            a = rng.uniform(0.7, 1.3)
            b = rng.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        return dict(
            pixel_values=normalise_rgb(np.ascontiguousarray(img)),
            prompt_depth=torch.from_numpy(np.ascontiguousarray(prompt))[None],
            sup=torch.from_numpy(sup),
            sup_other=torch.from_numpy(other),
        )
