"""Reference plane (plan 4.3)."""

import numpy as np
import pytest

from uiwue.threed import ground


def tilted_soil(n=3000, tilt_deg=4.0, z0=0.47, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.06, 0.06, (n, 2))
    t = np.radians(tilt_deg)
    z = z0 + xy[:, 0] * np.tan(t) + rng.normal(0, noise, n)
    return np.column_stack([xy, z])


def test_plane_sign_and_height():
    p = ground.Plane.horizontal(0.5)
    assert p.normal[2] < 0
    assert p.height(np.array([[0, 0, 0.4]]))[0] == pytest.approx(0.1)  # 10 cm above the soil
    assert p.z_at(0.1, 0.2) == pytest.approx(0.5)
    assert p.tilt_deg == pytest.approx(0.0)


def test_plane_shift_between_frames():
    p = ground.Plane.from_point_normal([0, 0, 0.5], [0.05, 0.0, -1.0])
    off = np.array([0.1, -0.05, 0.0])
    pts = np.array([[0.02, 0.03, 0.42], [-0.1, 0.0, 0.47]])
    assert np.allclose(p.shifted(off).height(pts - off), p.height(pts))


def test_to_local_basis():
    p = ground.Plane.from_point_normal([0, 0, 0.5], [0.1, 0.05, -1.0])
    origin = p.project(np.array([0.01, 0.02, 0.5]))
    q = origin + 0.03 * p.basis()[0] + 0.07 * p.normal
    a, b, h = p.to_local(q[None], origin)[0]
    assert (a, b, h) == pytest.approx((0.03, 0.0, 0.07), abs=1e-9)


def test_ransac_recovers_soil_under_leaves():
    soil = tilted_soil()
    rng = np.random.default_rng(1)
    leaf = np.column_stack([rng.uniform(-0.03, 0.03, (1500, 2)), rng.uniform(0.38, 0.44, 1500)])
    pts = np.vstack([soil, leaf])
    res = ground.fit_plane_ransac(pts, max_tilt_deg=15)
    assert res is not None
    plane, inl = res
    assert plane.tilt_deg == pytest.approx(4.0, abs=0.5)
    assert plane.z_at(0, 0) == pytest.approx(0.47, abs=0.001)
    assert inl[: len(soil)].mean() > 0.9 and inl[len(soil) :].mean() < 0.1


def test_fit_reference_accepts_and_rejects():
    soil = tilted_soil()
    c = np.array([0, 0, 0.47])
    ok = ground.fit_reference(soil, c, np.ones(len(soil), bool))
    assert ok.accepted and ok.reason == "ok"
    few = ground.fit_reference(soil[:100], c, np.ones(100, bool))
    assert not few.accepted and few.reason == "few_candidates"
    steep = tilted_soil(tilt_deg=30)
    assert not ground.fit_reference(steep, c, np.ones(len(steep), bool)).accepted
    prior = ground.Plane.horizontal(0.50)  # 3 cm deeper than the fit
    pr = ground.fit_reference(soil, c, np.ones(len(soil), bool), prior=prior)
    assert not pr.accepted and pr.reason == "prior_diff"


def test_median_plane():
    planes = [ground.Plane.horizontal(z) for z in (0.46, 0.47, 0.50)]
    assert ground.median_plane(planes).d == pytest.approx(0.47)
    assert ground.median_plane([]) is None
