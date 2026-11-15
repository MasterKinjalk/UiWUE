"""Reference level: where "height zero" is (plan section 4.3).

Height is measured from the soil surface of the plant's own pot, fitted as a plane with RANSAC
on non-vegetation points near the plant centre. When the plant covers its pot, the fit fails or
is rejected and the plant's carried reference is used instead (``qc_ref_carried``).

Planes are stored as ``n . p + d = 0`` with the unit normal ``n`` pointing from the soil towards
the camera (``n_z < 0`` in the depth-camera frame), so ``height(p) = n . p + d`` is positive
above the soil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANSAC_DIST = 0.005  # m, inlier distance (plan 4.3)
RANSAC_ITERS = 500
MIN_INLIERS = 150
MAX_TILT_DEG = 15.0  # plan start value 10; soil of edge-row pots (A2, A6, A8, B3) is tilted 9.5-13 deg
SOIL_RADIUS = 0.05  # m, soil candidates within this distance of the plant centre (pots are ~11 cm wide)
MAX_PRIOR_DIFF = 0.020  # m, reject a fit whose level at the centre differs more from the prior


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray  # (3,) unit, towards the camera
    d: float

    @staticmethod
    def from_point_normal(point, normal) -> Plane:
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        if n[2] > 0:
            n = -n
        return Plane(n, float(-n @ np.asarray(point, float)))

    @staticmethod
    def horizontal(z: float) -> Plane:
        """Plane perpendicular to the camera axis at depth ``z``."""
        return Plane(np.array([0.0, 0.0, -1.0]), float(z))

    def height(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p) @ self.normal + self.d

    def project(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        return p - np.multiply.outer(self.height(p), self.normal)

    def z_at(self, x: float, y: float) -> float:
        """Depth of the plane on the camera ray through (x, y, *) (camera frame)."""
        n = self.normal
        return float(-(self.d + n[0] * x + n[1] * y) / n[2])

    def ray_hit(self, ray_xy=(0.0, 0.0)) -> np.ndarray:
        """Intersection with the camera ray (x/z, y/z) = ray_xy."""
        r = np.array([ray_xy[0], ray_xy[1], 1.0])
        t = -self.d / (self.normal @ r)
        return t * r

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(-self.normal[2], -1, 1))))

    def shifted(self, offset) -> Plane:
        """The same plane in a frame where ``p_new = p_old - offset`` (e.g. common -> frame k
        with ``offset = frame_offset(pos_k)``)."""
        return Plane(self.normal, float(self.d + self.normal @ np.asarray(offset, float)))

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Two in-plane unit vectors (e1 close to the camera x axis, e2 = n x e1)."""
        n = self.normal
        e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(n, e1)

    def to_local(self, p: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """Points -> (a, b, h): in-plane coordinates relative to ``origin`` and height."""
        e1, e2 = self.basis()
        q = np.asarray(p, float) - np.asarray(origin, float)
        return np.column_stack([q @ e1, q @ e2, self.height(p)])

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (*map(float, self.normal), self.d)


@dataclass
class PlaneFit:
    plane: Plane | None  # None if no plane could be fitted at all
    accepted: bool
    n_candidates: int
    n_inliers: int
    tilt_deg: float
    reason: str  # "ok" or why it was rejected


def fit_plane_ransac(
    points: np.ndarray,
    dist: float = RANSAC_DIST,
    iters: int = RANSAC_ITERS,
    seed: int = 0,
    max_tilt_deg: float | None = None,
) -> tuple[Plane, np.ndarray] | None:
    """RANSAC plane + least-squares refit on the inliers. Returns (plane, inlier mask).

    With ``max_tilt_deg``, only hypotheses within that angle of the camera axis are scored, so a
    sloping leaf with many points cannot win over the (smaller) visible soil patch.
    """
    p = np.asarray(points, float)
    if len(p) < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(iters, 3))
    a, b, c = p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    good = ln > 1e-12
    if not good.any():
        return None
    nrm, a = nrm[good] / ln[good, None], a[good]
    if max_tilt_deg is not None:
        flat = np.abs(nrm[:, 2]) >= np.cos(np.radians(max_tilt_deg))
        if not flat.any():
            return None
        nrm, a = nrm[flat], a[flat]
    dd = -(nrm * a).sum(1)
    # score in chunks to bound memory: |n . p + d| < dist
    best, best_count = 0, -1
    for s in range(0, len(nrm), 100):
        cnt = (np.abs(p @ nrm[s : s + 100].T + dd[s : s + 100]) < dist).sum(0)
        j = int(np.argmax(cnt))
        if cnt[j] > best_count:
            best, best_count = s + j, int(cnt[j])
    plane = Plane.from_point_normal(a[best], nrm[best])
    inl = np.abs(plane.height(p)) < dist
    for _ in range(2):  # least-squares refit on the inliers, then re-select
        q = p[inl]
        if len(q) < 3:
            break
        cen = q.mean(0)
        _, _, vt = np.linalg.svd(q - cen, full_matrices=False)
        plane = Plane.from_point_normal(cen, vt[2])
        inl = np.abs(plane.height(p)) < dist
    return plane, inl


def soil_candidates(
    xyz: np.ndarray, centre: np.ndarray, not_veg: np.ndarray, radius: float = SOIL_RADIUS
) -> np.ndarray:
    """Mask of points within ``radius`` (camera-xy distance) of ``centre`` that are not
    vegetation."""
    r = np.hypot(xyz[:, 0] - centre[0], xyz[:, 1] - centre[1])
    return not_veg & (r < radius)


def fit_reference(
    xyz: np.ndarray,
    centre: np.ndarray,
    not_veg: np.ndarray,
    prior: Plane | None = None,
    *,
    radius: float = SOIL_RADIUS,
    dist: float = RANSAC_DIST,
    iters: int = RANSAC_ITERS,
    min_inliers: int = MIN_INLIERS,
    max_tilt_deg: float = MAX_TILT_DEG,
    max_prior_diff: float = MAX_PRIOR_DIFF,
    seed: int = 0,
) -> PlaneFit:
    """Soil plane around ``centre`` (plan 4.3 steps 1-3).

    ``not_veg`` excludes vegetation that is known before the plane exists (ExG, or a learned
    mask). With a ``prior`` (the plant's carried reference in this frame), a fit whose level at
    the centre is more than ``max_prior_diff`` away is rejected: when the pot is covered, the
    dominant "plane" is a leaf.
    """
    cand = soil_candidates(xyz, centre, not_veg, radius)
    n_c = int(cand.sum())
    if n_c < min_inliers:
        return PlaneFit(None, False, n_c, 0, float("nan"), "few_candidates")
    res = fit_plane_ransac(xyz[cand], dist=dist, iters=iters, seed=seed, max_tilt_deg=max_tilt_deg)
    if res is None:
        return PlaneFit(None, False, n_c, 0, float("nan"), "degenerate")
    plane, inl = res
    n_in, tilt = int(inl.sum()), plane.tilt_deg
    if n_in < min_inliers:
        return PlaneFit(plane, False, n_c, n_in, tilt, "few_inliers")
    if tilt > max_tilt_deg:
        return PlaneFit(plane, False, n_c, n_in, tilt, "tilt")
    if prior is not None:
        diff = abs(plane.z_at(centre[0], centre[1]) - prior.z_at(centre[0], centre[1]))
        if diff > max_prior_diff:
            return PlaneFit(plane, False, n_c, n_in, tilt, "prior_diff")
    return PlaneFit(plane, True, n_c, n_in, tilt, "ok")


def median_plane(planes: list[Plane]) -> Plane | None:
    """Robust average of planes in one frame: median normal (renormalised) and median ``d``."""
    if not planes:
        return None
    n = np.median(np.array([p.normal for p in planes]), axis=0)
    n /= np.linalg.norm(n)
    d = float(np.median([p.d for p in planes]))
    return Plane(n, d)
