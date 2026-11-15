"""Depth + RGB frame -> coloured point cloud in the depth-camera frame (plan sections 4.1-4.2).

Differences to the ADA code (``analysis/combined/3d_viewer.py``), all for the ``traits`` mode:

- Depth is not remapped with ``cv2.undistort``. Each pixel is unprojected along its own
  undistorted ray (``cv2.undistortPoints``), so depth values are never interpolated across leaf
  edges.
- A masked 3 x 3 median filter replaces hole filling. Pixels with fewer than
  ``MIN_NEIGHBOURS`` valid neighbours (speckle, flying pixels) are dropped.
- Colour is sampled from the *raw* RGB image at the distorted projection of each point
  (equivalent to sampling the undistorted image, without remapping 2 MPix). A coarse z-buffer
  marks points hidden from the RGB camera (the cameras are 4-5 cm apart) as uncoloured.

The ADA-exact path (``cv2.undistort`` + hole filling) lives in :mod:`uiwue.threed.parity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from uiwue.threed import calib

Z_MIN, Z_MAX = 0.20, 0.80  # valid ToF range (m), plan 4.1
MIN_NEIGHBOURS = 5  # of the 3 x 3 window incl. the pixel itself; fewer -> dropped as speckle
ZBUF_CELL_PX = 8.0  # z-buffer cell in full-resolution RGB px (one ToF pixel is ~6.5 RGB px)
ZBUF_TOL = 0.010  # m; points further than this behind the nearest point of their cell are hidden
RGB_EDGE_MARGIN = 2.0  # px kept free at the RGB image border

_RAYS: np.ndarray | None = None
_PIXEL_AREA: np.ndarray | None = None


def depth_rays() -> np.ndarray:
    """Undistorted normalised rays (x/z, y/z) for every depth pixel, shape (H, W, 2)."""
    global _RAYS
    if _RAYS is None:
        u, v = np.meshgrid(np.arange(calib.DEPTH_W), np.arange(calib.DEPTH_H))
        pix = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        rays = cv2.undistortPoints(pix, calib.K_DEPTH, calib.DEPTH_DISTORTION)
        _RAYS = rays.reshape(calib.DEPTH_H, calib.DEPTH_W, 2)
    return _RAYS


def pixel_area_factor() -> np.ndarray:
    """Per-pixel footprint in normalised image units, ``|det d(ray)/d(u, v)|``, shape (H, W).

    The ground area of a pixel at depth z (fronto-parallel surface) is ``z**2 * factor``. Near the
    image centre this equals ``1 / (fx * fy)`` (plan 4.6); towards the corners the lens
    distortion changes it by up to ~20 %.
    """
    global _PIXEL_AREA
    if _PIXEL_AREA is None:
        r = depth_rays()
        dxdu = np.gradient(r[..., 0], axis=1)
        dydv = np.gradient(r[..., 1], axis=0)
        dxdv = np.gradient(r[..., 0], axis=0)
        dydu = np.gradient(r[..., 1], axis=1)
        _PIXEL_AREA = np.abs(dxdu * dydv - dxdv * dydu)
    return _PIXEL_AREA


def median3(depth: np.ndarray, min_neighbours: int = MIN_NEIGHBOURS) -> np.ndarray:
    """Masked 3 x 3 median of valid (> 0) pixels. Invalid pixels stay 0 (no hole filling);
    valid pixels with fewer than ``min_neighbours`` valid pixels in their window are set to 0."""
    d = np.where(depth > 0, depth, np.nan).astype(np.float32)
    win = sliding_window_view(np.pad(d, 1, constant_values=np.nan), (3, 3)).reshape(*depth.shape, 9)
    n_valid = np.isfinite(win).sum(-1)
    s = np.sort(np.where(np.isfinite(win), win, np.inf), axis=-1)  # valid values first
    lo = np.clip((n_valid - 1) // 2, 0, 8)
    hi = np.clip(n_valid // 2, 0, 8)
    med = 0.5 * (np.take_along_axis(s, lo[..., None], -1)[..., 0] + np.take_along_axis(s, hi[..., None], -1)[..., 0])
    keep = (depth > 0) & (n_valid >= min_neighbours)
    return np.where(keep, med, 0).astype(np.float32)


@dataclass
class Cloud:
    """Valid ToF points of one frame in that frame's depth-camera coordinates (metres).

    ``rc`` keeps each point's (row, col) so masks can go back to the image grid, and a learned
    2D mask (e.g. a plant instance from the RGB) can be lifted onto the points.
    """

    xyz: np.ndarray  # (N, 3) float64
    rc: np.ndarray  # (N, 2) int32, depth pixel (row, col)
    rgb: np.ndarray  # (N, 3) uint8, 0 where no colour
    has_rgb: np.ndarray  # (N,) bool
    nir: np.ndarray | None = None  # (N,) float32
    rgb_uv: np.ndarray | None = None  # (N, 2) float32, full-resolution distorted RGB pixel
    brightness: float = float("nan")  # mean of the sampled RGB values (nan without colour)
    shape: tuple[int, int] = (calib.DEPTH_H, calib.DEPTH_W)
    meta: dict = field(default_factory=dict)
    filled: np.ndarray | None = None  # (N,) bool, point comes from a filled hole
    refined: np.ndarray | None = None  # (N,) bool, depth comes from a refined (learned) map

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def exg(self) -> np.ndarray:
        """Excess green 2G - R - B on 8-bit values (NaN where no colour)."""
        c = self.rgb.astype(np.float32)
        e = 2 * c[:, 1] - c[:, 0] - c[:, 2]
        return np.where(self.has_rgb, e, np.nan)

    @property
    def pixel_area(self) -> np.ndarray:
        """Ground area of each point's pixel (m^2), fronto-parallel: ``z^2 * |dray/duv|``."""
        f = pixel_area_factor()[self.rc[:, 0], self.rc[:, 1]]
        return self.xyz[:, 2] ** 2 * f

    def image(self, values: np.ndarray, fill=np.nan) -> np.ndarray:
        """Scatter per-point ``values`` back onto the depth image grid."""
        values = np.asarray(values)
        out = np.full(self.shape + values.shape[1:], fill, dtype=np.result_type(values, fill))
        out[self.rc[:, 0], self.rc[:, 1]] = values
        return out

    def from_image(self, img: np.ndarray) -> np.ndarray:
        """Gather a per-pixel image (e.g. a learned plant mask) onto the points."""
        return np.asarray(img)[self.rc[:, 0], self.rc[:, 1]]

    def subset(self, mask: np.ndarray) -> Cloud:
        return Cloud(
            xyz=self.xyz[mask],
            rc=self.rc[mask],
            rgb=self.rgb[mask],
            has_rgb=self.has_rgb[mask],
            nir=None if self.nir is None else self.nir[mask],
            rgb_uv=None if self.rgb_uv is None else self.rgb_uv[mask],
            brightness=self.brightness,
            shape=self.shape,
            meta=dict(self.meta),
            filled=None if self.filled is None else self.filled[mask],
            refined=None if self.refined is None else self.refined[mask],
        )


def distort(xn: np.ndarray, yn: np.ndarray, dist) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV 5-parameter distortion of normalised coordinates."""
    k1, k2, p1, p2, k3 = dist
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd, yd


# Depth pixels that see the scene: the Pico Flexx image has invalid rounded corners and an invalid
# right edge (mean validity of 117 frames < 0.5 beyond ~115 px from the principal point).
FOV_RADIUS_PX = 112.0
FOV_MARGIN_PX = 2


def depth_pixel(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth-frame points -> distorted depth image (u, v) in pixels."""
    z = np.where(xyz[:, 2] > 1e-6, xyz[:, 2], np.nan)
    xd, yd = distort(xyz[:, 0] / z, xyz[:, 1] / z, calib.DEPTH_DISTORTION)
    k = calib.K_DEPTH
    return k[0, 0] * xd + k[0, 2], k[1, 1] * yd + k[1, 2]


def in_depth_fov(xyz: np.ndarray) -> np.ndarray:
    """True for points the ToF camera can see (inside the image and its valid circle)."""
    u, v = depth_pixel(xyz)
    k = calib.K_DEPTH
    m = FOV_MARGIN_PX
    ok = (u >= m) & (u <= calib.DEPTH_W - 1 - 3 * m) & (v >= m) & (v <= calib.DEPTH_H - 1 - m)
    return ok & (np.hypot(u - k[0, 2], v - k[1, 2]) < FOV_RADIUS_PX)


def project_to_rgb(xyz: np.ndarray, t_depth_to_rgb: np.ndarray):
    """Depth-frame points -> (u, v) distorted full-resolution RGB pixels, RGB-frame z, and a
    validity mask (in front of the camera and inside the image after undistortion)."""
    q = xyz @ t_depth_to_rgb[:3, :3].T + t_depth_to_rgb[:3, 3]
    z = q[:, 2]
    ok = z > 1e-3
    zs = np.where(ok, z, 1.0)
    xn, yn = q[:, 0] / zs, q[:, 1] / zs
    k = calib.K_RGB
    # bounds check in undistorted coordinates (the 5-parameter model folds back far outside)
    lim_x = (
        (-RGB_EDGE_MARGIN - k[0, 2]) / k[0, 0],
        (calib.RGB_W - 1 + RGB_EDGE_MARGIN - k[0, 2]) / k[0, 0],
    )
    lim_y = (
        (-RGB_EDGE_MARGIN - k[1, 2]) / k[1, 1],
        (calib.RGB_H - 1 + RGB_EDGE_MARGIN - k[1, 2]) / k[1, 1],
    )
    ok &= (xn > lim_x[0] - 0.05) & (xn < lim_x[1] + 0.05) & (yn > lim_y[0] - 0.05) & (yn < lim_y[1] + 0.05)
    xd, yd = distort(xn, yn, calib.RGB_DISTORTION)
    u = k[0, 0] * xd + k[0, 2]
    v = k[1, 1] * yd + k[1, 2]
    ok &= (u >= RGB_EDGE_MARGIN) & (u <= calib.RGB_W - 1 - RGB_EDGE_MARGIN)
    ok &= (v >= RGB_EDGE_MARGIN) & (v <= calib.RGB_H - 1 - RGB_EDGE_MARGIN)
    return u, v, z, ok, (xn, yn)


def zbuffer_visible(xn: np.ndarray, yn: np.ndarray, z: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """True for points that are the nearest surface (within ZBUF_TOL) in their RGB cell."""
    k = calib.K_RGB
    cu = np.floor((k[0, 0] * xn + k[0, 2]) / ZBUF_CELL_PX).astype(np.int64)
    cv = np.floor((k[1, 1] * yn + k[1, 2]) / ZBUF_CELL_PX).astype(np.int64)
    gw = int(calib.RGB_W / ZBUF_CELL_PX) + 4
    gh = int(calib.RGB_H / ZBUF_CELL_PX) + 4
    cu, cv = np.clip(cu + 2, 0, gw - 1), np.clip(cv + 2, 0, gh - 1)
    idx = cv * gw + cu
    zmin = np.full(gw * gh, np.inf)
    np.minimum.at(zmin, idx[ok], z[ok])
    return ok & (z <= zmin[idx] + ZBUF_TOL)


MAX_HOLE_PX = 150  # enclosed invalid regions up to this size are filled (fill_holes=True)
DEPTH_U16_PER_M = 10_000.0  # hole filling inpaints depth as uint16 in 0.1 mm units (max 6.55 m)


def fill_small_holes(depth: np.ndarray, max_px: int = MAX_HOLE_PX) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid regions of at most ``max_px`` pixels that are enclosed by valid pixels
    (Telea inpainting from their rim). Returns (filled depth, mask of filled pixels).

    The Pico Flexx returns no depth on the bright interior of leaves (saturation), most visibly
    on young plants: only the leaf rims get a depth. Large invalid regions (image corners,
    shadows) touch the border or exceed ``max_px`` and stay invalid.

    The inpainting runs on a uint16 copy in 0.1 mm units. With OpenCV 5.0, Telea on float32
    returns values far outside the rim (metres off on a step edge); on uint16 it stays within
    the rim range.
    """
    inv = (depth <= 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = depth.shape
    x, y, bw, bh, area = stats.T
    small = (area <= max_px) & (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
    sel = np.zeros(n, bool)
    sel[1:] = small[1:]
    mask = sel[lab] & (inv > 0)
    if not mask.any():
        return depth, mask
    d16 = np.clip(np.rint(np.where(depth > 0, depth, 0) * DEPTH_U16_PER_M), 0, 65535).astype(np.uint16)
    out = cv2.inpaint(d16, mask.astype(np.uint8), 2, cv2.INPAINT_TELEA).astype(np.float32) / DEPTH_U16_PER_M
    filled = np.where(mask, out, depth).astype(np.float32)
    return filled, mask


def build_cloud(
    depth: np.ndarray,
    nir: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    t_depth_to_rgb: np.ndarray | None = None,
    *,
    rgb_scale: float = 1.0,
    median: bool = True,
    z_range: tuple[float, float] = (Z_MIN, Z_MAX),
    zbuffer: bool = True,
    fill_holes: bool = True,
) -> Cloud:
    """One depth frame (metres, 0 = invalid, already rotated) -> :class:`Cloud`.

    ``depth`` may be a refined depth map of the same grid (e.g. a learned upsampler resampled to
    224 x 171); pass ``median=False`` if it is already denoised. ``rgb`` may be loaded at reduced
    resolution; ``rgb_scale`` is its size relative to 1920 x 1080 (0.5 for
    ``cv2.IMREAD_REDUCED_COLOR_2``). ``t_depth_to_rgb`` is the shifted extrinsic
    (:func:`calib.depth_to_sensor`); without it the cloud has no colour. ``fill_holes`` fills
    small enclosed invalid regions first (:func:`fill_small_holes`); those points have
    ``filled=True``.
    """
    d = np.where(depth > 0, depth, 0).astype(np.float32)
    filled = np.zeros(d.shape, bool)
    if fill_holes:
        d, filled = fill_small_holes(d)
    d = median3(d) if median else d
    valid = (d > z_range[0]) & (d < z_range[1])
    rows, cols = np.nonzero(valid)
    z = d[rows, cols].astype(np.float64)
    rays = depth_rays()[rows, cols]
    xyz = np.column_stack([rays[:, 0] * z, rays[:, 1] * z, z])
    n = len(xyz)
    rgb_out = np.zeros((n, 3), np.uint8)
    has = np.zeros(n, bool)
    uv = None
    bright = float("nan")
    if rgb is not None and t_depth_to_rgb is not None and n:
        u, v, zr, ok, (xn, yn) = project_to_rgb(xyz, t_depth_to_rgb)
        if zbuffer:
            ok = zbuffer_visible(xn, yn, zr, ok)
        # pixel-centre convention for reduced images: full px c -> (c + 0.5) * s - 0.5
        us = ((u + 0.5) * rgb_scale - 0.5).astype(np.float32)
        vs = ((v + 0.5) * rgb_scale - 0.5).astype(np.float32)
        if ok.any():
            samp = cv2.remap(
                rgb,
                us[ok].reshape(-1, 1),
                vs[ok].reshape(-1, 1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1, 3)
            rgb_out[ok] = samp
            has = ok
            bright = float(samp.mean())
        uv = np.column_stack([u, v]).astype(np.float32)
    nir_pts = None if nir is None else nir[rows, cols].astype(np.float32)
    return Cloud(
        xyz=xyz,
        rc=np.column_stack([rows, cols]).astype(np.int32),
        rgb=rgb_out,
        has_rgb=has,
        nir=nir_pts,
        rgb_uv=uv,
        brightness=bright,
        shape=depth.shape,
        filled=filled[rows, cols],
    )


GRID_ZBUF_TOL = 0.010  # m, points_to_depth_grid keeps points this close to the nearest per pixel


def points_to_depth_grid(xyz: np.ndarray, tol: float = GRID_ZBUF_TOL) -> np.ndarray:
    """Resample a dense point set (depth-camera frame of one frame) onto that frame's ToF pixel
    grid: each point goes to the depth pixel it projects into (with the depth lens distortion);
    per pixel, the points within ``tol`` of the nearest one are averaged (z-buffer, so surfaces
    hidden from the ToF camera, e.g. soil seen by the RGB camera under a leaf edge, are dropped).
    Returns a (H, W) depth image in metres with 0 where no point landed.

    This is the "voxel grid" used for refined depth: one sample per ToF pixel (2.2 mm at the
    soil, 1.4 mm at 0.30 m), so the trait parameters tuned on ToF density (DBSCAN eps and
    min_samples, normal radius and max_nn, per-pixel areas, the 300-point QC rule) stay valid.
    """
    h, w = calib.DEPTH_H, calib.DEPTH_W
    p = np.asarray(xyz, float)
    p = p[np.isfinite(p).all(1) & (p[:, 2] > 1e-3)]
    u, v = depth_pixel(p)
    c, r = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    idx, z = r[ok] * w + c[ok], p[ok, 2]
    zmin = np.full(h * w, np.inf)
    np.minimum.at(zmin, idx, z)
    keep = z <= zmin[idx] + tol
    s = np.bincount(idx[keep], weights=z[keep], minlength=h * w)
    n = np.bincount(idx[keep], minlength=h * w)
    out = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return out.reshape(h, w).astype(np.float32)


# Robot position -> depth-camera frame. ADA's ``calib.R_ROBOT_TO_DEPTH`` (rotz(+90 deg)) maps
# robot (x, y) to depth (-y, +x). Aligning the 15 frames of real runs in a common frame shows that
# robot x maps to depth -y instead: frames of different robot columns then agree to 6-7 mm median
# (the same as frames of one column), against 14-17 mm with rotz(+90 deg) (W1_J5 / W2_J5 / W1_A2,
# 2025-08-20 15 h; 16-22 mm against 45-56 mm on the closed canopy of W1_J5 2025-09-03). Robot y
# maps to depth -x in both, so depth -> RGB colouring (RGB and depth poses differ only in robot y
# for > 95 % of frames) is unaffected. See docs/threed/PIPELINE.md.
M_ROBOT_TO_DEPTH = calib.R_ROBOT_TO_DEPTH  # the corrected mapping, single source in calib.py


def frame_offset(pos_robot) -> np.ndarray:
    """Offset that maps a frame's depth coordinates into the run's common frame:
    ``p_common = p_frame + frame_offset(pos)``."""
    return M_ROBOT_TO_DEPTH @ np.asarray(pos_robot, dtype=np.float64)


def depth_to_rgb(pos_depth, pos_rgb) -> np.ndarray:
    """Shifted depth -> RGB extrinsic for a depth frame captured at robot position ``pos_depth``
    and an RGB frame captured at ``pos_rgb`` (ADA composition ``T @ inv(S_rgb) @ S_depth`` with
    the corrected robot -> depth mapping)."""
    s = np.eye(4)
    s[:3, 3] = frame_offset(pos_depth) - frame_offset(pos_rgb)
    return calib.T_DEPTH_TO_RGB @ s
