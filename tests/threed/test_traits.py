"""Per-frame traits (plan 4.6) and the E0c synthetic scene.

E0c: a ground plane plus a tilted disc of known height, area and angle, rendered along the depth
camera's own undistorted pixel rays and run through the real chain (cloud -> reference plane ->
vegetation mask -> isolation -> traits). Every trait must be within 5 % of the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from uiwue.threed import calib, ground, segment, traits
from uiwue.threed.cloud import build_cloud, depth_rays, points_to_depth_grid


@dataclass
class DiscScene:
    ground_z: float = 0.47  # m, depth of the ground on the optical axis
    ground_tilt_deg: float = 3.0  # ground tilt about the camera y axis
    centre_xy: tuple[float, float] = (0.02, 0.03)  # disc centre (camera x, y), on the ground
    height: float = 0.10  # m, disc centre above the ground (along the ground normal)
    radius: float = 0.04  # m
    tilt_deg: float = 30.0  # disc tilt relative to the ground
    noise: float = 0.0  # m, Gaussian depth noise

    def ground_normal(self) -> np.ndarray:
        a = np.radians(self.ground_tilt_deg)
        return np.array([np.sin(a), 0.0, -np.cos(a)])  # towards the camera

    def ground_point(self) -> np.ndarray:
        n = self.ground_normal()
        x, y = self.centre_xy
        # point on the ground plane through (0, 0, ground_z) with normal n
        z = self.ground_z - (n[0] * x + n[1] * y) / n[2]
        return np.array([x, y, z])

    def disc_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Disc centre and unit normal (towards the camera)."""
        n = self.ground_normal()
        c = self.ground_point() + self.height * n
        e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
        e1 /= np.linalg.norm(e1)
        t = np.radians(self.tilt_deg)
        nd = np.cos(t) * n + np.sin(t) * e1  # tilt about the in-plane axis e2
        return c, nd / np.linalg.norm(nd)

    def depth_along(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Depth (camera z) of the nearest surface along normalised rays (..., 2), and the disc
        mask."""
        d = np.concatenate([rays, np.ones(rays.shape[:-1] + (1,))], -1)  # z = 1 -> t = depth
        ng, pg = self.ground_normal(), self.ground_point()
        tg = (ng @ pg) / (d @ ng)
        c, nd = self.disc_frame()
        td = (nd @ c) / (d @ nd)
        hit = d * td[..., None]
        on_disc = (np.linalg.norm(hit - c, axis=-1) <= self.radius) & (td > 0) & (td < tg)
        return np.where(on_disc, td, tg), on_disc

    def render(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Depth image (metres) on the ToF grid and a per-pixel disc mask."""
        depth, on_disc = self.depth_along(depth_rays())
        depth = depth.astype(np.float32)
        if self.noise:
            depth += np.random.default_rng(seed).normal(0, self.noise, depth.shape).astype(np.float32)
        return depth, on_disc

    def dense_points(self, factor: int = 3) -> np.ndarray:
        """Exact surface points on a ``factor`` x finer ray grid (a stand-in for a dense refined
        depth map such as Prompt Depth Anything's), in the depth-camera frame."""
        r = depth_rays()
        xs = np.linspace(r[..., 0].min(), r[..., 0].max(), calib.DEPTH_W * factor)
        ys = np.linspace(r[..., 1].min(), r[..., 1].max(), calib.DEPTH_H * factor)
        rays = np.stack(np.meshgrid(xs, ys), -1)
        z, _ = self.depth_along(rays)
        return np.column_stack([rays[..., 0].ravel() * z.ravel(), rays[..., 1].ravel() * z.ravel(), z.ravel()])

    def truth(self, n_mc: int = 400_000, seed: int = 1) -> dict:
        """True traits of the disc (heights in mm, areas in cm2, angle in degrees)."""
        rng = np.random.default_rng(seed)
        rr = self.radius * np.sqrt(rng.uniform(0, 1, n_mc))
        aa = rng.uniform(0, 2 * np.pi, n_mc)
        t = np.radians(self.tilt_deg)
        # height of a uniform disc point = h + s * sin(tilt), s along the tilt direction;
        # projection onto the ground is area-uniform, so ToF pixels sample it uniformly
        h = self.height + rr * np.cos(aa) * np.sin(t)
        p50, p75, p95 = np.percentile(h, [50, 75, 95])
        area = np.pi * self.radius**2
        return {
            "height_p50": 1000 * p50,
            "height_p75": 1000 * p75,
            "height_p95": 1000 * p95,
            "projected_area": 1e4 * area * np.cos(t),
            "leaf_area_est": 1e4 * area,
            "hull_volume": 1e6 * area * np.cos(t) * self.height,
            "inclination_mean": self.tilt_deg,
            "compactness": 1.0,
        }


def run_chain(scene: DiscScene, refined: bool = False):
    """Synthetic depth -> traits, through the same functions as the CLI (no colour).
    ``refined``: dense exact points resampled onto the ToF grid (the --depth-model path)."""
    if refined:
        depth = points_to_depth_grid(scene.dense_points())
        cloud = build_cloud(depth, median=False, fill_holes=False)
    else:
        depth, _ = scene.render()
        cloud = build_cloud(depth)
    # soil: everything around the plant (the disc covers the centre)
    centre_guess = scene.ground_point()
    fit = ground.fit_reference(cloud.xyz, centre_guess, np.ones(len(cloud), bool), radius=0.2, min_inliers=500)
    assert fit.accepted, fit.reason
    plane = fit.plane
    centre = plane.project(centre_guess)
    loc = plane.to_local(cloud.xyz, centre)
    veg = segment.vegetation_mask(loc[:, 2])  # no colour: height > 15 mm
    iso = segment.isolate_plant(loc, veg, xyz=cloud.xyz)
    m = iso.mask
    tr = traits.frame_traits(loc[m], cloud.pixel_area[m], xyz=cloud.xyz[m], plane_normal=plane.normal)
    return tr, fit, iso


@pytest.mark.parametrize("tilt", [0.0, 20.0, 35.0])
def test_e0c_disc_traits_within_5_percent(tilt):
    scene = DiscScene(tilt_deg=tilt)
    tr, fit, iso = run_chain(scene)
    truth = scene.truth()
    assert fit.tilt_deg == pytest.approx(scene.ground_tilt_deg, abs=0.2)
    assert not iso.no_plant and iso.clipped_fraction == 0.0
    for k in [
        "height_p50",
        "height_p75",
        "height_p95",
        "projected_area",
        "leaf_area_est",
        "hull_volume",
        "compactness",
    ]:
        assert tr[k] == pytest.approx(truth[k], rel=0.05), k
    if tilt > 0:
        assert tr["inclination_mean"] == pytest.approx(tilt, rel=0.05)
    else:
        assert tr["inclination_mean"] < 2.0
    assert tr["inclination_sd"] < 3.0
    assert tr["droop_index"] == pytest.approx(truth["height_p50"] / truth["height_p95"], rel=0.05)
    assert tr["canopy_radius"] <= 1000 * scene.radius * 1.02


def test_e0c_with_tof_noise_heights_and_area_hold():
    """5 mm depth noise (the ToF level after the 3 x 3 median): heights and areas stay within
    5 %; inclination is biased upwards by noise (why E1 must judge the inclination traits)."""
    scene = DiscScene(tilt_deg=25.0, noise=0.005)
    tr, _, _ = run_chain(scene)
    truth = scene.truth()
    for k in ["height_p50", "height_p75", "projected_area"]:
        assert tr[k] == pytest.approx(truth[k], rel=0.05), k
    assert tr["inclination_mean"] > 20.0


def test_normals_of_a_tilted_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.05, 0.05, (3000, 2))
    t = np.radians(40)
    p = np.column_stack([xy[:, 0], xy[:, 1], xy[:, 0] * np.tan(t)])
    n = traits.estimate_normals(p)
    inc = traits.inclination_deg(n, np.array([0.0, 0.0, 1.0]))
    assert np.nanmedian(inc) == pytest.approx(40.0, abs=0.5)


def test_too_few_points_gives_nan():
    tr = traits.frame_traits(np.zeros((3, 3)), np.ones(3))
    assert tr["n_points"] == 3 and np.isnan(tr["height_p75"])


@pytest.mark.parametrize("tilt", [0.0, 35.0])
def test_e0c_refined_depth_path_within_5_percent(tilt):
    """The --depth-model path: a dense (3x) exact point set resampled onto the ToF grid by
    points_to_depth_grid (z-buffer per pixel) gives the same traits within 5 %."""
    scene = DiscScene(tilt_deg=tilt)
    tr, _fit, _iso = run_chain(scene, refined=True)
    truth = scene.truth()
    for k in [
        "height_p50",
        "height_p75",
        "height_p95",
        "projected_area",
        "leaf_area_est",
        "hull_volume",
    ]:
        assert tr[k] == pytest.approx(truth[k], rel=0.05), k
    if tilt:
        assert tr["inclination_mean"] == pytest.approx(tilt, rel=0.05)
