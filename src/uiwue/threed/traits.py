"""Per-frame plant traits (plan section 4.6).

Input is the isolated plant's points in plane-local coordinates (a, b, h): in-plane position
relative to the plant centre and height above the reference plane, all in metres, plus each
point's pixel ground area. Output units follow ``schema.TRAITS``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

NORMAL_RADIUS = 0.012  # m (plan 4.6: neighbours within 12 mm, at most 30)
NORMAL_MAX_NN = 30
INCLINATION_CAP_DEG = 75.0
MIN_POINTS = 10  # fewer plant points -> traits are NaN


def estimate_normals(xyz: np.ndarray, radius: float = NORMAL_RADIUS, max_nn: int = NORMAL_MAX_NN) -> np.ndarray:
    """PCA normals over the ``max_nn`` nearest neighbours within ``radius`` (the same
    neighbourhood as Open3D's ``KDTreeSearchParamHybrid(radius, max_nn)``). Points with fewer
    than 3 neighbours get NaN."""
    p = np.asarray(xyz, float)
    n = len(p)
    out = np.full((n, 3), np.nan)
    if n < 3:
        return out
    k = min(max_nn, n)
    dist, idx = cKDTree(p).query(p, k=k, distance_upper_bound=radius)
    valid = np.isfinite(dist)
    cnt = valid.sum(1)
    idx = np.where(valid, idx, 0)
    nb = p[idx]  # (n, k, 3)
    w = valid[..., None]
    mean = (nb * w).sum(1) / np.maximum(cnt, 1)[:, None]
    d = (nb - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", d, d) / np.maximum(cnt, 1)[:, None, None]
    _, vec = np.linalg.eigh(cov)
    out = vec[:, :, 0]  # eigenvector of the smallest eigenvalue
    out[cnt < 3] = np.nan
    return out


def inclination_deg(normals: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Angle between each normal and the reference-plane normal, folded to [0, 90] degrees
    (0 = flat leaf, 90 = vertical)."""
    c = np.abs(normals @ np.asarray(plane_normal, float))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def _hull_area_2d(ab: np.ndarray) -> float:
    try:
        return float(ConvexHull(ab).volume)  # 2D "volume" is the area
    except (QhullError, ValueError):
        return float("nan")


def _hull_volume(local: np.ndarray) -> float:
    base = local.copy()
    base[:, 2] = 0.0
    try:
        return float(ConvexHull(np.vstack([local, base])).volume)
    except (QhullError, ValueError):
        return float("nan")


def nan_traits() -> dict:
    from uiwue.threed.schema import TRAITS

    d = {t: float("nan") for t in TRAITS}
    d["n_points"] = 0.0
    return d


def frame_traits(
    local: np.ndarray,
    pixel_area: np.ndarray,
    *,
    xyz: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
) -> dict:
    """Traits of one frame's isolated plant.

    ``local``: (N, 3) plane-local (a, b, h) in metres. ``pixel_area``: (N,) ground area of each
    point's pixel in m^2. Normals are estimated on ``xyz`` (camera frame) with ``plane_normal``
    as reference if both are given, else on ``local`` with (0, 0, 1).
    """
    n = len(local)
    out = nan_traits()
    out["n_points"] = float(n)
    if n < MIN_POINTS:
        return out
    h = local[:, 2]
    p50, p75, p95 = np.percentile(h, [50, 75, 95])
    out["height_p50"] = 1000 * p50
    out["height_p75"] = 1000 * p75
    out["height_p95"] = 1000 * p95
    if xyz is not None and plane_normal is not None:
        normals, ref = estimate_normals(xyz), np.asarray(plane_normal, float)
    else:
        normals, ref = estimate_normals(local), np.array([0.0, 0.0, 1.0])
    inc = inclination_deg(normals, ref)
    ok = np.isfinite(inc)
    proj, surf = pixel_area.copy(), pixel_area.copy()
    if xyz is not None and plane_normal is not None and ok.any():
        # pixel footprint on the surface: z^2 a cos(phi) / |n . ray| (phi: ray to optical axis),
        # projected onto the reference plane: times |n . N|. For a fronto-parallel surface both
        # equal z^2 a (plan 4.6); for tilted leaves seen off-axis this removes the camera's
        # foreshortening (E0c: -9 % at 35 deg without it).
        rng = np.linalg.norm(xyz, axis=1)
        ray = xyz / rng[:, None]
        cos_phi = xyz[:, 2] / rng
        nr = np.abs(np.einsum("ij,ij->i", np.nan_to_num(normals), ray))
        nr = np.maximum(nr, np.cos(np.radians(INCLINATION_CAP_DEG)))
        nn = np.abs(np.nan_to_num(normals) @ ref)
        s_ok = pixel_area * cos_phi / nr
        surf = np.where(ok, s_ok, pixel_area)
        proj = np.where(ok, np.clip(s_ok * nn, 0.5 * pixel_area, 2.0 * pixel_area), pixel_area)
    elif ok.any():
        cap = np.radians(np.minimum(np.where(ok, inc, 0.0), INCLINATION_CAP_DEG))
        surf = pixel_area / np.cos(cap)
    out["projected_area"] = 1e4 * float(proj.sum())
    if ok.any():
        out["inclination_mean"] = float(inc[ok].mean())
        out["inclination_sd"] = float(inc[ok].std())
        out["leaf_area_est"] = 1e4 * float(surf.sum())
    out["hull_volume"] = 1e6 * _hull_volume(local)
    r = np.hypot(local[:, 0], local[:, 1])
    out["canopy_radius"] = 1000 * float(np.percentile(r, 95))
    out["droop_index"] = float(p50 / p95) if p95 > 0 else float("nan")
    hull2d = _hull_area_2d(local[:, :2])
    out["compactness"] = float(proj.sum() / hull2d) if np.isfinite(hull2d) and hull2d > 0 else float("nan")
    return out
