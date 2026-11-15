"""Paper-parity plant height (experiment E0a): the ADA 2025 method, reproduced exactly.

Source: ``external/ada2025/analysis/plant_height/step_01_build_pointclouds.py`` and
``step_02_compute_metrics.py`` (ADA commit ``calib.ADA_COMMIT``). Per run: frame 7 only;
``cv2.undistort`` of depth and RGB; depth holes filled up to 5 px; depth pixels coloured from the
RGB with the ADA shifted extrinsic built with ADA's own robot mapping
(``calib.R_ROBOT_TO_DEPTH_ADA``, rotz(+90 deg), kept on purpose; see :func:`ada_depth_to_rgb`);
points without colour dropped; vegetation = ExG > 5 on the remapped 8-bit RGB; ``height = 0.60 m - 25th percentile of vegetation depth``. Runs with fewer than 1000
vegetation points are dropped by the paper (``DEFAULT_VEGETATION_PT_THRESHOLD``); here they get
``height_parity_mm = NaN`` and keep the raw value in ``height_parity_raw_mm``.

Nothing from :mod:`uiwue.threed.cloud` is used, so the result is independent of the traits path.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from uiwue.threed import calib, schema
from uiwue.threed.io import Run, list_runs, read_frames

SENSOR_HEIGHT_M = 0.60
EXG_THRESHOLD = 5.0
FRAME = 7
HOUR = "15"
MIN_VEG_POINTS = 1000
PAPER_PLANTS = [f"{b}_{p}" for b in ("W1", "W2") for p in ("A2", "A8", "J5", "R1", "R7")]
PAPER_DATES = ("2025_09_01", "2025_09_14")


def ada_depth_to_rgb(pos_depth, pos_rgb) -> np.ndarray:
    """ADA's shifted extrinsic with ADA's own robot mapping (``calib.R_ROBOT_TO_DEPTH_ADA``,
    rotz(+90 deg)); ``calib.depth_to_sensor`` now uses the corrected mapping."""
    sd = calib.shift_matrix(calib.R_ROBOT_TO_DEPTH_ADA @ np.asarray(pos_depth, float))
    sr = calib.shift_matrix(calib.R_ROBOT_TO_DEPTH_ADA @ np.asarray(pos_rgb, float))
    return calib.T_DEPTH_TO_RGB @ np.linalg.inv(sr) @ sd


def load_depth_ada(path: Path) -> np.ndarray:
    """``_load_depth_image`` + the sign / <= 0 handling of ``main()`` (NaN stays NaN)."""
    depth = tifffile.imread(path).astype(np.float32)
    depth = np.rot90(np.rot90(depth)).astype(np.float32)
    if float(np.nanmedian(depth)) < 0.0:
        depth = -depth
    depth[depth <= 0] = 0.0
    return depth


def fill_depth_holes(depth_img: np.ndarray, max_gap_pixels: int = 5) -> np.ndarray:
    """Verbatim copy of ADA ``_fill_depth_holes``."""
    valid = (depth_img > 0) & ~np.isnan(depth_img)
    if np.all(valid):
        return depth_img.copy()

    filled = depth_img.copy()
    filled[~valid] = 0

    kernel_size = 2 * max_gap_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    for _ in range(max_gap_pixels):
        prev_invalid = filled == 0
        shifted = np.stack(
            [
                np.roll(filled, 1, axis=0),
                np.roll(filled, -1, axis=0),
                np.roll(filled, 1, axis=1),
                np.roll(filled, -1, axis=1),
            ]
        )
        counts = np.stack(
            [
                np.roll(valid, 1, axis=0),
                np.roll(valid, -1, axis=0),
                np.roll(valid, 1, axis=1),
                np.roll(valid, -1, axis=1),
            ]
        ).astype(np.float32)

        neighbor_sum = np.sum(shifted, axis=0)
        neighbor_count = np.sum(counts, axis=0)
        fill_mask = prev_invalid & closed_mask & (neighbor_count > 0)
        filled[fill_mask] = neighbor_sum[fill_mask] / neighbor_count[fill_mask]
        valid = filled > 0

    return filled


def colorize(depth_img: np.ndarray, rgb_img: np.ndarray, t_depth_to_rgb: np.ndarray):
    """Verbatim logic of ADA ``_colorize_depth_with_rgb``."""
    depth_undist = cv2.undistort(depth_img, calib.K_DEPTH, calib.DEPTH_DISTORTION)
    rgb_undist = cv2.undistort(rgb_img, calib.K_RGB, calib.RGB_DISTORTION)
    depth_undist = fill_depth_holes(depth_undist, max_gap_pixels=5)

    height, width = depth_undist.shape
    rgb_h, rgb_w = rgb_undist.shape[:2]
    fx_d, fy_d = calib.K_DEPTH[0, 0], calib.K_DEPTH[1, 1]
    cx_d, cy_d = calib.K_DEPTH[0, 2], calib.K_DEPTH[1, 2]
    fx_rgb, fy_rgb = calib.K_RGB[0, 0], calib.K_RGB[1, 1]
    cx_rgb, cy_rgb = calib.K_RGB[0, 2], calib.K_RGB[1, 2]

    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    z_depth = depth_undist
    valid_depth = (z_depth > 0) & ~np.isnan(z_depth)
    x_depth = (u_coords - cx_d) * z_depth / fx_d
    y_depth = (v_coords - cy_d) * z_depth / fy_d
    points_depth_h = np.stack([x_depth, y_depth, z_depth, np.ones_like(z_depth)], axis=0).reshape(
        4, -1
    )
    points_rgb_h = t_depth_to_rgb @ points_depth_h
    z_rgb = points_rgb_h[2].reshape(height, width)
    safe_z = np.where(z_rgb > 0, z_rgb, 1.0)
    u_rgb = (fx_rgb * points_rgb_h[0].reshape(height, width) / safe_z + cx_rgb).astype(np.float32)
    v_rgb = (fy_rgb * points_rgb_h[1].reshape(height, width) / safe_z + cy_rgb).astype(np.float32)
    invalid_rgb = (
        ~valid_depth
        | (z_rgb <= 0)
        | (u_rgb < 0)
        | (u_rgb >= rgb_w)
        | (v_rgb < 0)
        | (v_rgb >= rgb_h)
    )
    u_rgb[invalid_rgb] = -1.0
    v_rgb[invalid_rgb] = -1.0
    colorized = cv2.remap(
        rgb_undist,
        u_rgb,
        v_rgb,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    colorized[invalid_rgb] = 0
    return depth_undist, colorized


def cloud_z_exg(depth_img: np.ndarray, colorized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``_depth_to_colorized_point_cloud`` + the ExG of ``_write_ply_with_exg`` (z and ExG only;
    PLY stores float32)."""
    valid = (depth_img > 0) & ~np.isnan(depth_img) & np.any(colorized != 0, axis=2)
    z = depth_img[valid].astype(np.float32)
    c = colorized[valid].astype(np.float32)
    exg = (2.0 * c[:, 1] - c[:, 0] - c[:, 2]).astype(np.float32)
    return z, exg


def parity_run(run: Run, data_dir: Path, frame: int = FRAME) -> dict | None:
    """One run -> parity row, or None if the paper would skip it (missing CSV / frame)."""
    ddir, rdir = run.dir(data_dir, "depth"), run.dir(data_dir, "rgb")
    pos_d, pos_r = read_frames(ddir / "frames_depth.csv"), read_frames(rdir / "frames_rgb.csv")
    if frame not in pos_d or frame not in pos_r:
        return None
    dpath, rpath = ddir / f"depth_{frame:03d}.tif", rdir / f"rgb_{frame:03d}.jpg"
    if not dpath.exists() or not rpath.exists():
        return None
    bgr = cv2.imread(str(rpath), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = ada_depth_to_rgb(pos_d[frame], pos_r[frame])
    depth_u, col = colorize(load_depth_ada(dpath), rgb, t)
    z, exg = cloud_z_exg(depth_u, col)
    veg = exg > EXG_THRESHOLD
    n_veg = int(veg.sum())
    z_p25 = float(np.percentile(z[veg].astype(np.float64), 25.0)) if n_veg else float("nan")
    raw = 1000 * (SENSOR_HEIGHT_M - z_p25)
    return dict(
        bed=run.bed,
        plant=run.plant,
        date=schema.folder_date(run.date),
        hour=int(run.hour),
        height_parity_mm=raw if n_veg >= MIN_VEG_POINTS else float("nan"),
        height_parity_raw_mm=raw,
        z_p25_m=z_p25,
        n_veg=n_veg,
        n_points_total=int(z.size),
        frame=frame,
    )


def parity_table(
    data_dir: Path,
    plants: list[str] | None = None,
    dates: tuple[str, str] | None = None,
    hours: list[str] | None = None,
    jobs: int = 8,
) -> pd.DataFrame:
    from joblib import Parallel, delayed

    runs = list_runs(
        Path(data_dir),
        "depth",
        plants=plants or PAPER_PLANTS,
        dates=dates or PAPER_DATES,
        hours=hours or [HOUR],
    )
    rows = Parallel(n_jobs=jobs)(delayed(parity_run)(r, Path(data_dir)) for r in runs)
    df = pd.DataFrame([r for r in rows if r is not None])
    if len(df):
        df["bed"] = df["bed"].astype("string")
        df["plant"] = df["plant"].astype("string")
        df = df.sort_values(schema.KEYS).reset_index(drop=True)
    return df


def main_from_run(data_dir, out: Path, plants, dates, hours, jobs) -> int:
    df = parity_table(data_dir, plants, dates, hours, jobs)
    path = Path(out) / schema.TRAITS_PARITY.name
    df.to_parquet(path, index=False)
    print(f"parity: {len(df)} runs -> {path}", flush=True)
    return 0
