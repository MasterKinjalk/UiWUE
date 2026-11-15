"""Raw images for the fusion encoders: one array per sensor and run, plus variants.

Layout: <data>/<sensor>/<bed>_<plant>/<YYYY_MM_DD>/<HH>/ (RGB may also stay zipped,
see uiwue.rgb.run). Run folders are mapped to the scheduled slot as everywhere else
(the scheduled folder if it has frames, else the re-run with the most frames).

Normalisation uses fixed physical constants, never statistics of the data, so
nothing is fitted outside a fold (rule 6). The fusion head layer-normalises the
embeddings itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uiwue import core

SENSORS = ("rgb", "thermal", "depth", "hsi")
FRAME_GLOB = {"thermal": "thermal_*.tif", "depth": "depth_*.tif", "hsi": "hsi_*.jp2"}
# Depth and RGB: 15 frames on a 3 x 5 grid of robot positions; frame 7 is the centre view.
# Thermal: 5 frames on a 8 cm sweep along the row; projecting the plant (frame-7 RGB optical
# axis) through the ADA calibration puts it at 26-80 % of the image width from frame 0 to 4
# and at 48-61 % in frame 2. Frames 0, 1, 3, 4 are off-centre views that bring neighbours in,
# so only frame 2 is used (as ADA's 3d_viewer and the LoRA stage). Verified 2026-11-14.
FRAMES = {"rgb": (7,), "thermal": (2,), "depth": (7,), "hsi": None}  # None = every frame
# Hyperspectral is used at 9, 12 and 15 h only (Master Plan 6.5): the 06 h cubes are dark
# (flat-field values up to 1e8) and the 18 h cubes are often saturated.
SLOTS = {"hsi": (9, 12, 15)}
SIZE = 224

# Logitech C920 band centres (approximate, µm). DOFA takes wavelengths in µm.
RGB_WAVELENGTHS = (0.61, 0.54, 0.465)
# Fixed scalings to roughly zero mean, unit spread (documented constants, not fitted).
REFLECTANCE_SCALE = (0.5, 0.25)  # (x - 0.5) / 0.25 for RGB in 0-1 and HSI reflectance
REFLECTANCE_RANGE = (0.0, 2.0)  # flat-field reflectance is clipped to this (leaves near the lens exceed 1)
THERMAL_SCALE = (27.5, 8.75)  # °C
# 1 / metres. The Pico Flexx sees soil at 0.45-0.52 m (disparity 1.9-2.2) and canopy tops at
# 0.27 m (3.7), tall controls up to 0.19 m (5.2). Centring on 0.40 m (2.5) maps soil to about
# -0.7 and the tallest canopy to +3.6, the range ImageBind's standardised SUN RGB-D disparity
# covers (about -1.2 to +3.7). The earlier (1.6, 0.6) put every pixel at +0.5 to +7.
DISPARITY_SCALE = (2.5, 0.75)
CROP = {"rgb": 0.5, "thermal": 1.0, "depth": 1.0, "hsi": 1.0}  # centre square, fraction of the short side


def _dark_reference() -> Path:
    """ADA's flat-field dark reference (analysis/hyperspectral/black_reference.npy).

    UIWUE_HSI_DARK, else the ada2025 clone in external/ or next to the repo. The old
    default (only next to the repo) did not exist in this layout, and a missing file
    silently turned the flat-field correction off.
    """
    if "UIWUE_HSI_DARK" in os.environ:
        return Path(os.environ["UIWUE_HSI_DARK"])
    rel = Path("ada2025") / "analysis" / "hyperspectral" / "black_reference.npy"
    candidates = [core.REPO / "external" / rel, core.REPO.parent / rel]
    return next((p for p in candidates if p.exists()), candidates[0])


DARK_REFERENCE = _dark_reference()

# Variants: the same run seen through a cheaper or reduced sensor (Part B). Each
# gets its own embedding table and can replace the full sensor at evaluation.
HSI_WINDOW_NM = 20
HSI_RANGE_NM = (384, 1025)


def hsi_windows() -> list[str]:
    lo, hi = HSI_RANGE_NM
    return [f"mask{w}" for w in range(lo, hi, HSI_WINDOW_NM)]


VARIANTS = {
    "rgb": ["full"],
    "thermal": ["full", "r160", "r80"],
    "depth": ["full"],
    "hsi": ["full", "s112", "s64", "rgb3"],
}


# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #
def list_runs(data_dir: Path, sensor: str, plants: list[str] | None = None) -> pd.DataFrame:
    """bed, plant, date, hour (scheduled slot), path, n_frames: one row per slot."""
    if sensor == "rgb":
        from uiwue.rgb import run as rgb

        r = rgb.assign_slots(rgb.list_runs(Path(data_dir), plants))
        r = r[r["n_images"] > 0]
        return r.rename(columns={"n_images": "n_frames"})[["bed", "plant", "date", "hour", "path", "n_frames"]]
    root = Path(data_dir) / sensor
    rows = []
    if not root.is_dir():
        return pd.DataFrame(columns=["bed", "plant", "date", "hour", "path", "n_frames"])
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and "_" in p.name):
        if plants and pdir.name not in plants:
            continue
        bed, plant = pdir.name.split("_", 1)
        for ddir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            try:
                d = pd.Timestamp(ddir.name.replace("_", "-")).date()
            except ValueError:
                continue
            if not core.DATES[0] <= d <= core.DATES[1]:
                continue
            best = {}
            for hdir in (h for h in ddir.iterdir() if h.is_dir() and h.name.isdigit()):
                slot, n = core.slot_for_hour(int(hdir.name)), len(list(hdir.glob(FRAME_GLOB[sensor])))
                if slot is None or n == 0 or slot not in SLOTS.get(sensor, core.SLOTS):
                    continue
                rank = (int(hdir.name) == slot, n)
                if slot not in best or rank > best[slot][0]:
                    best[slot] = (rank, hdir, n)
            for slot, (_, hdir, n) in sorted(best.items()):
                rows.append(dict(bed=bed, plant=plant, date=d, hour=slot, path=str(hdir), n_frames=n))
    return pd.DataFrame(rows, columns=["bed", "plant", "date", "hour", "path", "n_frames"])


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def centre_square(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(1, round(min(h, w) * frac))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def resize_chw(x: np.ndarray, size: int) -> np.ndarray:
    """(C, H, W) float32 -> (C, size, size), area-averaged down / bilinear up, any channel count."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    mode = "area" if min(t.shape[-2:]) >= size else "bilinear"
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(t, size=(size, size), mode=mode, **kw)[0].numpy()


def block_mean(img: np.ndarray, f: int) -> np.ndarray:
    """Average f x f pixel blocks of an (H, W) image: what a camera with 1/f the resolution sees."""
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _hwc_to_chw(img: np.ndarray) -> np.ndarray:
    return np.moveaxis(img if img.ndim == 3 else img[..., None], -1, 0)


# --------------------------------------------------------------------------- #
# per sensor: frames -> list of normalised (C, H, W) arrays at full size
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One run of one sensor: frames (F, C, H, W) float32 and band wavelengths (µm) if spectral."""

    frames: np.ndarray
    wavelengths: tuple[float, ...] | None = None


def _pick(files: list, frames) -> list:
    if frames is None:
        return files
    from uiwue.rgb.run import pick_frames

    return pick_frames(files, tuple(frames))


def load_rgb(path: str, frames=FRAMES["rgb"]) -> Sample:
    from uiwue.rgb import run as rgb

    imgs = [rgb._read_rgb(path, n) for n in _pick(rgb._images(path), frames)]
    out = [_hwc_to_chw(centre_square(i, CROP["rgb"]).astype(np.float32) / 255.0) for i in imgs]
    return Sample(np.stack([resize_chw(o, SIZE) for o in out]), RGB_WAVELENGTHS)


def read_thermal(path: str, frames=FRAMES["thermal"]) -> list[np.ndarray]:
    """Thermal frames in °C, rotated upright (ADA stores them rotated by 180°)."""
    import tifffile

    files = sorted(Path(path).glob(FRAME_GLOB["thermal"]))
    return [np.rot90(tifffile.imread(f).astype(np.float32), 2) for f in _pick(files, frames)]


def load_thermal(path: str, variant: str = "full", frames=FRAMES["thermal"]) -> Sample:
    out = []
    for img in read_thermal(path, frames):
        if variant in ("r160", "r80"):
            img = block_mean(img, 4 if variant == "r160" else 8)
        img = (centre_square(img, CROP["thermal"]) - THERMAL_SCALE[0]) / THERMAL_SCALE[1]
        out.append(resize_chw(img[None], SIZE))
    return Sample(np.stack(out))


def load_depth(path: str, frames=FRAMES["depth"]) -> Sample:
    """Depth as disparity (1 / metres, 0 where invalid), as ImageBind expects."""
    from uiwue.threed.io import load_depth as read_depth

    files = sorted(Path(path).glob(FRAME_GLOB["depth"]))
    out = []
    for f in _pick(files, frames):
        d, _ = read_depth(f)
        disp = np.where(d > 0, 1.0 / np.maximum(d, 1e-3), 0.0).astype(np.float32)
        disp = np.where(d > 0, (disp - DISPARITY_SCALE[0]) / DISPARITY_SCALE[1], 0.0)
        out.append(resize_chw(centre_square(disp, CROP["depth"])[None], SIZE))
    return Sample(np.stack(out).astype(np.float32))


def read_hsi(path: str, dark_reference: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flat-field-corrected cube (H, W, B) and band wavelengths (nm), via hyperio as in the ADA code.

    Identical to ADA's `HSI.read(jp2, dark_reference=black_reference.npy)` (checked on real
    cubes); ADA's Savitzky-Golay smoothing is not applied. Raises if the dark reference is
    missing or the cube is dark/saturated, so the run is reported as failed, not embedded.
    """
    from hyperio import HSI

    jp2 = min(Path(path).glob(FRAME_GLOB["hsi"]))
    dark_path = Path(dark_reference or DARK_REFERENCE)
    if not dark_path.exists():
        raise FileNotFoundError(f"HSI dark reference not found: {dark_path} (set UIWUE_HSI_DARK)")
    cube = np.asarray(HSI.read(jp2, dark_reference=np.load(dark_path)), dtype=np.float32)
    sample = cube[::8, ::8]
    if not np.isfinite(sample).all() or np.percentile(sample, 99) > 3 * REFLECTANCE_RANGE[1]:
        raise ValueError(f"{jp2}: dark or saturated cube (p99 reflectance {np.percentile(sample, 99):.3g})")
    meta = json.loads(jp2.with_suffix(".json").read_text())
    wl = np.asarray(meta["wavelengths"], dtype=np.float32)
    # A truncated cube (hyperio warns "cube has 256 channels") must fail its run here, in the
    # loader: with 300 wavelengths it would reach DOFA and the band masks and stop the whole job.
    if cube.shape[-1] != len(wl):
        raise ValueError(f"{jp2}: {cube.shape[-1]} channels but {len(wl)} wavelengths (truncated cube)")
    return cube, wl


def hsi_bands(wl_nm: np.ndarray, variant: str) -> np.ndarray | None:
    """Band indices kept by a spectral variant (rgb3, mask<nm>); None for full and spatial variants.

    Band selection commutes with the per-band resize and the per-pixel scaling, so a
    spectral variant equals the full model input with these channels taken (embed.py
    uses this instead of building 33 copies of the cube per run).
    """
    if variant == "rgb3":  # the three bands closest to an RGB camera
        return np.array([int(np.abs(wl_nm - w * 1000).argmin()) for w in RGB_WAVELENGTHS])
    if variant.startswith("mask"):  # one 20 nm window removed
        lo = float(variant[4:])
        return np.flatnonzero(~((wl_nm >= lo) & (wl_nm < lo + HSI_WINDOW_NM)))
    if variant == "full" or (variant.startswith("s") and variant[1:].isdigit()):
        return None
    raise ValueError(variant)


def hsi_variant(cube: np.ndarray, wl_nm: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """(C, H, W) reflectance and wavelengths for one HSI variant (spatial or spectral reduction)."""
    x = np.moveaxis(cube, -1, 0)
    idx = hsi_bands(wl_nm, variant)
    if idx is not None:
        x, wl_nm = x[idx], wl_nm[idx]
    elif variant != "full":  # spatial detail removed, token count kept
        x = resize_chw(x, int(variant[1:]))
    return x, wl_nm


def load_hsi(path: str, variant: str = "full", cube=None) -> Sample:
    cube, wl = read_hsi(path) if cube is None else cube
    x, wl = hsi_variant(centre_square(cube, CROP["hsi"]), wl, variant)
    x = (np.clip(x, *REFLECTANCE_RANGE) - REFLECTANCE_SCALE[0]) / REFLECTANCE_SCALE[1]
    return Sample(resize_chw(x, SIZE)[None], tuple(float(w) / 1000 for w in wl))


def load(sensor: str, path: str, variant: str = "full") -> Sample:
    if sensor == "rgb":
        s = load_rgb(path)
        s.frames = (s.frames - REFLECTANCE_SCALE[0]) / REFLECTANCE_SCALE[1]
        return s
    if sensor == "thermal":
        return load_thermal(path, variant)
    if sensor == "depth":
        return load_depth(path)
    if sensor == "hsi":
        return load_hsi(path, variant)
    raise ValueError(sensor)


# --------------------------------------------------------------------------- #
# augmentation (fine-tuning stages): flips, 90° rotations, small crops, band jitter
# --------------------------------------------------------------------------- #
def augment(x, rng: np.random.Generator, crop: float = 0.85, jitter: float = 0.05):
    """x: torch tensor (F, C, H, W). Same geometric transform for every frame of a run."""
    import torch
    import torch.nn.functional as F

    if rng.random() < 0.5:
        x = x.flip(-1)
    if rng.random() < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, int(rng.integers(4)), dims=(-2, -1))
    h, w = x.shape[-2:]
    s = rng.uniform(crop, 1.0)
    ch, cw = max(8, int(h * s)), max(8, int(w * s))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    x = F.interpolate(x[..., y0 : y0 + ch, x0 : x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
    gain = torch.from_numpy(rng.uniform(1 - jitter, 1 + jitter, size=(1, x.shape[1], 1, 1)).astype(np.float32))
    return x * gain.to(x.device)
