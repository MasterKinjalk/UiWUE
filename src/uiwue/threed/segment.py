"""Vegetation mask and centre-plant isolation (plan sections 4.4 and 4.5).

Both functions take plain arrays, so a learned plant mask (lifted onto the points with
``Cloud.from_image``) can replace the ExG rule (``learned=``) without touching the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

EXG_THRESHOLD = 5.0  # 2G - R - B on 8-bit values (the paper's rule)
H_MIN = 0.005  # m, vegetation must stand this far above the reference (normal light)
H_MIN_DARK = 0.015  # m, height-only rule (dark frame, or point without colour)
# Mean 8-bit brightness of the RGB samples below which a frame counts as dark. The plan's start
# value 40 flags about half of all 15 h frames (median frame-7 brightness 44 at 15 h on W1_J5 /
# W2_J5); ExG masks stay usable down to ~8 and are noise below ~6 (diagnostics/dark_sheet.jpg).
DARK_BRIGHTNESS = 10.0

DBSCAN_EPS = 0.008  # m
DBSCAN_MIN_SAMPLES = 10
ATTACH_RADIUS = 0.03  # m, a cluster is the plant's if its nearest point is this close to the centre
CYLINDER_RADIUS = 0.085  # m, half the neighbour distance
WALL_BAND = 0.005  # m, clipped_fraction = share of kept points within this of the cylinder wall
SEARCH_RADIUS = 0.13  # m, only vegetation within this distance is clustered
NOT_CENTRED = 0.03  # m, kept centroid further than this from the centre -> qc_not_centred
REFINE_RADIUS = 0.04  # m, centre refinement window (plan 4.5 step 1)


def is_dark(brightness: float, threshold: float = DARK_BRIGHTNESS) -> bool:
    return bool(np.isfinite(brightness) and brightness < threshold)


def vegetation_mask(
    height: np.ndarray,
    exg: np.ndarray | None = None,
    has_rgb: np.ndarray | None = None,
    *,
    dark: bool = False,
    learned: np.ndarray | None = None,
    exg_threshold: float = EXG_THRESHOLD,
    h_min: float = H_MIN,
    h_min_dark: float = H_MIN_DARK,
) -> np.ndarray:
    """Per-point vegetation mask (plan 4.4).

    - ``learned`` given (bool per point, e.g. the centre-plant mask from the RGB): points with
      colour ``learned & h > h_min``; points without colour (``has_rgb`` False; the mask cannot
      be read there) ``h > h_min_dark``. Without ``has_rgb`` every point is ``learned & h > h_min``.
    - normal light, point has colour: ``ExG > exg_threshold & h > h_min``.
    - dark frame, or point without colour (outside the RGB view or hidden from it):
      ``h > h_min_dark``.
    """
    h = np.asarray(height)
    tall = h > h_min_dark
    if learned is not None:
        plant = np.asarray(learned, bool) & (h > h_min)
        return plant if has_rgb is None else np.where(np.asarray(has_rgb, bool), plant, tall)
    if exg is None or dark:
        return tall
    has = np.isfinite(exg) if has_rgb is None else np.asarray(has_rgb, bool)
    green = (np.nan_to_num(exg, nan=-1e9) > exg_threshold) & (h > h_min)
    return np.where(has, green, tall)


@dataclass
class Isolation:
    mask: np.ndarray  # (N,) bool, points of the isolated centre plant
    n_clusters: int  # clusters attached to the centre
    clipped_fraction: float  # share of kept points within WALL_BAND of the cylinder wall
    outside_fraction: float  # share of the attached clusters' points cut off by the cylinder
    centroid_offset: float  # m, horizontal distance of the kept centroid from the centre
    no_plant: bool
    not_centred: bool


def isolate_plant(
    local: np.ndarray,
    veg: np.ndarray,
    *,
    xyz: np.ndarray | None = None,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    attach_radius: float = ATTACH_RADIUS,
    cylinder_radius: float = CYLINDER_RADIUS,
    wall_band: float = WALL_BAND,
    search_radius: float = SEARCH_RADIUS,
    not_centred: float = NOT_CENTRED,
) -> Isolation:
    """Keep the vegetation clusters that touch the plant centre, clipped to a cylinder.

    ``local`` is (N, 3) plane-local coordinates (a, b, h) with the plant centre at a = b = 0
    (``Plane.to_local``). Clustering runs on ``xyz`` (camera frame) if given, else on ``local``
    (the same metric space).
    """
    n = len(local)
    r = np.hypot(local[:, 0], local[:, 1])
    cand = np.flatnonzero(np.asarray(veg, bool) & (r < search_radius))
    empty = Isolation(np.zeros(n, bool), 0, float("nan"), float("nan"), float("nan"), True, False)
    if len(cand) < min_samples:
        return empty
    pts = (local if xyz is None else xyz)[cand]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    rc = r[cand]
    keep_labels = []
    for lab in np.unique(labels[labels >= 0]):
        if rc[labels == lab].min() < attach_radius:
            keep_labels.append(lab)
    if not keep_labels:
        return empty
    attached = np.isin(labels, keep_labels)
    inside = attached & (rc <= cylinder_radius)
    if not inside.any():
        return empty
    mask = np.zeros(n, bool)
    mask[cand[inside]] = True
    rk = r[mask]
    clipped = float(np.mean(rk > cylinder_radius - wall_band))
    outside = float(1 - inside.sum() / attached.sum())
    off = float(np.hypot(local[mask, 0].mean(), local[mask, 1].mean()))
    return Isolation(mask, len(keep_labels), clipped, outside, off, False, off > not_centred)


def refine_centre(local: np.ndarray, veg: np.ndarray, radius: float = REFINE_RADIUS, min_points: int = 20):
    """(da, db) shift of the plant centre to the centroid of vegetation within ``radius`` of it
    (plan 4.5 step 1); (0, 0) when there are too few points."""
    r = np.hypot(local[:, 0], local[:, 1])
    sel = np.asarray(veg, bool) & (r < radius)
    if sel.sum() < min_points:
        return 0.0, 0.0
    return float(local[sel, 0].mean()), float(local[sel, 1].mean())


VOXEL = 0.0025  # m, merged-frame clustering grid
VOTE_SHARE = 0.5  # a voxel is plant when at least this share of its votes says so (votes=)


def _block_sum(vox: np.ndarray, *values: np.ndarray) -> list[np.ndarray]:
    """Sum of each per-voxel array over every voxel's 3 x 3 x 3 neighbourhood (occupied voxels
    only; ``vox`` are unique integer voxel coordinates)."""
    lo = vox.min(0) - 1
    span = vox.max(0) - lo + 2

    def code(v):
        return ((v[:, 0] - lo[0]) * span[1] + (v[:, 1] - lo[1])) * span[2] + (v[:, 2] - lo[2])

    keys = code(vox)
    order = np.argsort(keys)
    sk = keys[order]
    out = [np.zeros(len(vox)) for _ in values]
    for d in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
        nk = code(vox + d)
        j = np.clip(np.searchsorted(sk, nk), 0, len(sk) - 1)
        hit = sk[j] == nk
        src = order[j[hit]]
        for o, v in zip(out, values, strict=True):
            o[hit] += v[src]
    return out


def isolate_plant_multi(
    locs: list[np.ndarray],
    vegs: list[np.ndarray],
    voxel: float = VOXEL,
    *,
    votes: list[np.ndarray] | None = None,
    min_share: float = VOTE_SHARE,
    **kw,
) -> list[Isolation]:
    """One isolation for all frames of a run, so every frame keeps the same plant.

    ``locs`` are the frames' plane-local coordinates in one shared plant-centred frame (the
    frames are already aligned by the robot poses). Vegetation points of all frames are pooled
    on a ``voxel`` grid, clustered once with :func:`isolate_plant`, and the kept voxels are mapped
    back to each frame's points. Per-frame ``clipped_fraction`` and ``centroid_offset`` are
    computed from the frame's own kept points.

    ``votes`` (learned plant mask, one float array per frame: 1 plant, 0 not, NaN no vote) turns
    ``vegs`` into the candidate points and decides per voxel: each frame with a vote in the voxel
    casts the mean vote of its points there, and the voxel is vegetation when at least
    ``min_share`` of the frames say plant, or when no point in it has a vote (points without
    colour). So every frame keeps the same voxels even where the frames' 2D masks disagree, and a
    point the RGB camera cannot see takes the label of its voxel.
    """
    search = kw.get("search_radius", SEARCH_RADIUS)
    cyl = kw.get("cylinder_radius", CYLINDER_RADIUS)
    band = kw.get("wall_band", WALL_BAND)
    nc = kw.get("not_centred", NOT_CENTRED)
    sels, pts, fid, vts = [], [], [], []
    for i, (loc, veg) in enumerate(zip(locs, vegs, strict=True)):
        s = np.flatnonzero(np.asarray(veg, bool) & (np.hypot(loc[:, 0], loc[:, 1]) < search))
        sels.append(s)
        pts.append(loc[s])
        fid.append(np.full(len(s), i))
        if votes is not None:
            vts.append(np.asarray(votes[i], float)[s])
    out = [
        Isolation(np.zeros(len(loc), bool), 0, float("nan"), float("nan"), float("nan"), True, False) for loc in locs
    ]
    if not pts or sum(len(p) for p in pts) == 0:
        return out
    pts, fid = np.vstack(pts), np.concatenate(fid)
    vox, inv = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    centres = (vox + 0.5) * voxel
    veg_vox = np.ones(len(centres), bool)
    if votes is not None:
        # one vote per frame and voxel (the mean of the frame's points there), so a frame that
        # sees a voxel with more points does not outvote the others
        vt = np.concatenate(vts)
        cast = np.isfinite(vt)
        key, kinv = np.unique(inv[cast] * len(locs) + fid[cast], return_inverse=True)
        frame_vote = np.bincount(kinv, weights=vt[cast]) / np.bincount(kinv)
        kvox = key // len(locs)
        n_frames = np.bincount(kvox, minlength=len(vox)).astype(float)
        n_plant = np.bincount(kvox, weights=frame_vote, minlength=len(vox))
        # at ToF density a 2.5 mm voxel holds 1-3 points per frame, so most voxels see only some
        # of the frames; the share is taken over the voxel's 3 x 3 x 3 block (7.5 mm) instead
        n_frames, n_plant = _block_sum(vox, n_frames, n_plant)
        veg_vox = (n_frames == 0) | (n_plant >= min_share * n_frames)
    iso = isolate_plant(centres, veg_vox, **kw)
    if iso.no_plant:
        return out
    keep_pt = iso.mask[inv] & (np.hypot(pts[:, 0], pts[:, 1]) <= cyl)
    for i, loc in enumerate(locs):
        m = np.zeros(len(loc), bool)
        m[sels[i]] = keep_pt[fid == i]
        if not m.any():
            continue
        rk = np.hypot(loc[m, 0], loc[m, 1])
        off = float(np.hypot(loc[m, 0].mean(), loc[m, 1].mean()))
        out[i] = Isolation(
            m,
            iso.n_clusters,
            float(np.mean(rk > cyl - band)),
            iso.outside_fraction,
            off,
            False,
            off > nc,
        )
    return out
