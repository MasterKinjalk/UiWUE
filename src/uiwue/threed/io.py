"""Loading ADA 2025 depth / RGB runs.

Layout (after utils/unzip_data.py): ``<data>/<modality>/<bed>_<plant>/<YYYY_MM_DD>/<HH>/``.
Depth and thermal images are stored rotated by 180 degrees; loaders rotate them back, as the
ADA code does.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile


@dataclass(frozen=True)
class Run:
    """One robot pass over one plant: ``<bed>_<plant>/<date>/<hour>``."""

    plant_id: str  # e.g. "W1_J5"
    date: str  # folder name, e.g. "2025_09_01"
    hour: str  # folder name, e.g. "15"

    @property
    def bed(self) -> str:
        return self.plant_id.split("_", 1)[0]

    @property
    def plant(self) -> str:
        return self.plant_id.split("_", 1)[1]

    def dir(self, data_dir: Path, modality: str) -> Path:
        return Path(data_dir) / modality / self.plant_id / self.date / self.hour


def list_runs(
    data_dir: Path,
    modality: str = "depth",
    plants: list[str] | None = None,
    dates: tuple[str, str] | None = None,
    hours: list[str] | None = None,
) -> list[Run]:
    """All runs on disk for ``modality``. ``dates`` is an inclusive (start, end) pair of folder
    names; ``plants`` are full IDs such as ``W2_J5``."""
    root = Path(data_dir) / modality
    runs = []
    if not root.is_dir():
        return runs
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if plants and pdir.name not in plants:
            continue
        for ddir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            if dates and not (dates[0] <= ddir.name <= dates[1]):
                continue
            for hdir in sorted(p for p in ddir.iterdir() if p.is_dir()):
                if hours and hdir.name not in hours:
                    continue
                runs.append(Run(pdir.name, ddir.name, hdir.name))
    return runs


def load_depth(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Depth TIFF -> (depth in metres float32 with 0 = invalid, NIR intensity float32 or None)."""
    with tifffile.TiffFile(path) as tif:
        depth = np.rot90(tif.pages[0].asarray(), 2).astype(np.float32)
        nir = np.rot90(tif.pages[1].asarray(), 2).astype(np.float32) if len(tif.pages) > 1 else None
    if np.nanmedian(depth) < 0:  # Pico Flexx can store signed depth
        depth = -depth
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    return np.ascontiguousarray(depth), None if nir is None else np.ascontiguousarray(nir)


def load_rgb(path: Path) -> np.ndarray | None:
    """JPEG -> RGB uint8 (H, W, 3), or None if missing/unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_frames(path: Path) -> dict[int, np.ndarray]:
    """frames_<sensor>.csv -> {frame index: robot position (x, y, z) in metres}."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {
            int(r["frame_id"]): np.array(
                [
                    float(r["position_robot_x"]),
                    float(r["position_robot_y"]),
                    float(r["position_robot_z"]),
                ]
            )
            for r in csv.DictReader(f)
        }


# ---------------------------------------------------------------- additions for the trait pipeline
N_FRAMES = 15  # depth / RGB frames per run (3 x 5 grid of robot positions)
CENTRE_FRAME = 7


def read_timestamps(path: Path) -> dict[int, int]:
    """frames_<sensor>.csv -> {frame index: timestamp in ns (UTC epoch)}."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["frame_id"]): int(r["timestamp"]) for r in csv.DictReader(f)}


def load_rgb_reduced(path: Path, factor: int = 2) -> np.ndarray | None:
    """JPEG decoded at 1/``factor`` resolution (2, 4 or 8; much faster than full decode)."""
    flag = {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8,
    }[factor]
    img = cv2.imread(str(path), flag)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_frames(run: Run, data_dir: Path) -> dict:
    """Everything about a run's frames that comes from the CSVs.

    Returns ``{"depth_pos", "rgb_pos", "depth_ts", "depth_files"}``: robot positions of the depth
    and RGB frames (empty dict if the CSV is missing), depth timestamps (ns) and the depth TIFFs
    present on disk by frame index.
    """
    ddir, rdir = run.dir(data_dir, "depth"), run.dir(data_dir, "rgb")
    files = {}
    if ddir.is_dir():
        for f in ddir.glob("depth_*.tif"):
            try:
                files[int(f.stem.split("_")[1])] = f
            except ValueError:
                continue
    return {
        "depth_pos": read_frames(ddir / "frames_depth.csv"),
        "rgb_pos": read_frames(rdir / "frames_rgb.csv"),
        "depth_ts": read_timestamps(ddir / "frames_depth.csv"),
        "depth_files": dict(sorted(files.items())),
    }
