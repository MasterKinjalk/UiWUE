"""PhenoBench -> Mask2Former instance-segmentation samples (sugar beet plants and leaves).

PhenoBench v1.1 layout (www.phenobench.org, CC BY-SA 4.0):
    PhenoBench/<split>/images/<name>.png            1024 x 1024 RGB (UAV, ~1 mm/px)
    PhenoBench/<split>/semantics/<name>.png         0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed
    PhenoBench/<split>/plant_instances/<name>.png   instance id per plant (crops and weeds)
    PhenoBench/<split>/leaf_instances/<name>.png    instance id per crop leaf

Classes trained here: 0 = sugar beet plant, 1 = sugar beet leaf. Weeds are dropped (there are
none in the ADA greenhouse beds). Plant and leaf masks overlap; Mask2Former matches each ground
truth mask on its own, so overlap is fine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ID2LABEL = {0: "sugar_beet_plant", 1: "sugar_beet_leaf"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP_LABELS = (1, 3)  # crop, partial crop


def _read(path: Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def instance_masks(sem: np.ndarray, plants: np.ndarray, leaves: np.ndarray, min_px: int = 16):
    """-> masks (n, H, W) uint8, classes (n,) int64."""
    masks, classes = [], []
    crop = np.isin(sem, CROP_LABELS)
    for cls, inst, restrict in ((0, plants, crop), (1, leaves, None)):
        for i in np.unique(inst):
            if i == 0:
                continue
            m = inst == i
            if restrict is not None:
                m &= restrict
            if m.sum() >= min_px:
                masks.append(m)
                classes.append(cls)
    h, w = sem.shape
    if not masks:
        return np.zeros((0, h, w), np.uint8), np.zeros((0,), np.int64)
    return np.stack(masks).astype(np.uint8), np.array(classes, np.int64)


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Colour augmentation towards ADA conditions PhenoBench lacks: dark greenhouse frames
    (06 h, shade), and wilted leaves that turn olive / yellow and lose saturation (W2 after the
    irrigation cut-off; the step-1000 model missed most of them on 2025-09-12)."""
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(0.6, 2.2)  # gamma > 1 darkens
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    if rng.random() < 0.6:
        # Shift only green pixels (OpenCV hue 20-100; green 60, yellow 30) so soil stays brown.
        green = (hsv[..., 0] > 20) & (hsv[..., 0] < 100)
        hsv[..., 0][green] += rng.uniform(-22, 4)
        hsv[..., 1][green] *= rng.uniform(0.35, 1.2)
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    x = x * rng.uniform(0.6, 1.3) + rng.uniform(-25, 25)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    return np.clip(x, 0, 255).astype(np.uint8)


def normalise_rgb(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class PhenoBenchInstances(Dataset):
    """Returns dict(pixel_values (3,S,S), mask_labels (n,S,S) uint8, class_labels (n,) long).

    Training augmentation: random scale (``scale_range``; ADA RGB at 0.6 m is ~0.43 mm/px vs
    ~1 mm/px in PhenoBench, so inference on ADA runs at ~0.5x and training covers both), random
    crop to ``size``, flips, 90-degree rotations, brightness/contrast jitter.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        size: int = 768,
        augment: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ):
        self.dir = Path(root) / split
        self.names = sorted(p.name for p in (self.dir / "images").glob("*.png"))
        self.size, self.augment, self.scale_range = size, augment, scale_range

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> dict:
        n = self.names[i]
        img = cv2.cvtColor(_read(self.dir / "images" / n, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sem = _read(self.dir / "semantics" / n)
        plants = _read(self.dir / "plant_instances" / n)
        leaves = _read(self.dir / "leaf_instances" / n)
        rng = np.random.default_rng() if self.augment else np.random.default_rng(i)

        s = self.size
        if self.augment:
            f = rng.uniform(*self.scale_range)
            h, w = img.shape[:2]
            nh, nw = max(s, round(h * f)), max(s, round(w * f))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sem, plants, leaves = (
                cv2.resize(a, (nw, nh), interpolation=cv2.INTER_NEAREST)
                for a in (sem, plants, leaves)
            )
            ys, xs = np.nonzero(plants)
            if len(ys) and rng.random() < 0.8:  # most crops centred on a plant pixel
                j = rng.integers(len(ys))
                y0 = int(np.clip(ys[j] - s // 2 + rng.integers(-s // 4, s // 4 + 1), 0, nh - s))
                x0 = int(np.clip(xs[j] - s // 2 + rng.integers(-s // 4, s // 4 + 1), 0, nw - s))
            else:
                y0, x0 = rng.integers(0, nh - s + 1), rng.integers(0, nw - s + 1)
            sl = (slice(y0, y0 + s), slice(x0, x0 + s))
            img, sem, plants, leaves = img[sl], sem[sl], plants[sl], leaves[sl]
            if rng.random() < 0.5:
                img, sem, plants, leaves = (a[:, ::-1] for a in (img, sem, plants, leaves))
            k = int(rng.integers(0, 4))
            img, sem, plants, leaves = (np.rot90(a, k) for a in (img, sem, plants, leaves))
            img = photometric(np.ascontiguousarray(img), rng)
        else:
            img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
            sem, plants, leaves = (
                cv2.resize(a, (s, s), interpolation=cv2.INTER_NEAREST)
                for a in (sem, plants, leaves)
            )

        masks, classes = instance_masks(*(np.ascontiguousarray(a) for a in (sem, plants, leaves)))
        return dict(
            pixel_values=normalise_rgb(np.ascontiguousarray(img)),
            mask_labels=torch.from_numpy(masks),  # uint8; float on the GPU (4x less IPC)
            class_labels=torch.from_numpy(classes),
        )


def collate(batch: list[dict]) -> dict:
    return dict(
        pixel_values=torch.stack([b["pixel_values"] for b in batch]),
        mask_labels=[b["mask_labels"] for b in batch],
        class_labels=[b["class_labels"] for b in batch],
    )
