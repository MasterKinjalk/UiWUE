"""Point cloud construction (plan 4.1-4.2): median filter, hole filling, unprojection, colour,
robot-pose conventions."""

import numpy as np
import pytest

from uiwue.threed import calib
from uiwue.threed.cloud import (
    M_ROBOT_TO_DEPTH,
    build_cloud,
    depth_rays,
    depth_to_rgb,
    fill_small_holes,
    frame_offset,
    in_depth_fov,
    median3,
    pixel_area_factor,
    points_to_depth_grid,
    project_to_rgb,
)

H, W = calib.DEPTH_H, calib.DEPTH_W


def flat(z=0.5):
    return np.full((H, W), z, np.float32)


def test_median3_drops_speckle_and_keeps_planes():
    d = flat()
    d[50:60, 50:60] = 0  # invalid block
    d[100, 100] = 0.3  # spike
    d[19:22, 19:22] = 0
    d[20, 20] = 0.45  # isolated valid pixel inside an invalid block
    m = median3(d)
    assert m[100, 100] == pytest.approx(0.5)
    assert m[20, 20] == 0  # fewer than MIN_NEIGHBOURS valid pixels
    assert (m[51:59, 51:59] == 0).all()  # no hole filling
    ramp = np.tile(np.linspace(0.4, 0.6, W, dtype=np.float32), (H, 1))
    assert np.allclose(median3(ramp)[1:-1, 1:-1], ramp[1:-1, 1:-1], atol=1e-6)


def test_fill_small_holes_only_fills_small_enclosed_regions():
    d = flat()
    d[80:85, 80:85] = 0  # small enclosed hole (25 px)
    d[0:40, 0:40] = 0  # large hole touching the border
    f, mask = fill_small_holes(d)
    assert mask[80:85, 80:85].all() and np.allclose(f[80:85, 80:85], 0.5, atol=1e-3)
    assert not mask[0:40, 0:40].any() and (f[0:40, 0:40] == 0).all()


def test_build_cloud_unprojects_along_rays():
    c = build_cloud(flat(0.5), fill_holes=False)
    assert len(c) == H * W - 4  # the 4 image corners have only 4 valid pixels in their window
    assert np.allclose(c.xyz[:, 2], 0.5)
    r = depth_rays()[c.rc[:, 0], c.rc[:, 1]]
    assert np.allclose(c.xyz[:, :2], r * 0.5)
    assert not c.has_rgb.any() and np.isnan(c.brightness)


def test_pixel_area_at_centre_matches_pinhole():
    a = pixel_area_factor()[int(calib.K_DEPTH[1, 2]), int(calib.K_DEPTH[0, 2])]
    assert a == pytest.approx(1 / (calib.K_DEPTH[0, 0] * calib.K_DEPTH[1, 1]), rel=0.02)


def test_colour_sampling_of_a_uniform_image():
    rgb = np.zeros((calib.RGB_H // 2, calib.RGB_W // 2, 3), np.uint8)
    rgb[...] = (40, 120, 30)
    pos = np.array([0.88, 0.45, 0.0])
    c = build_cloud(flat(0.47), None, rgb, depth_to_rgb(pos, pos), rgb_scale=0.5)
    # ~30 % of the ToF view has no RGB pixel (cameras 4-5 cm apart)
    assert 0.55 < c.has_rgb.mean() < 0.85
    assert (c.rgb[c.has_rgb] == (40, 120, 30)).all()
    assert np.nanmin(c.exg) == pytest.approx(2 * 120 - 40 - 30)


def test_rgb_projection_lands_inside_the_image():
    pos = np.zeros(3)
    u, v, z, ok, _ = project_to_rgb(np.array([[0.0, 0.0, 0.47]]), depth_to_rgb(pos, pos))
    assert ok[0] and 0 < u[0] < calib.RGB_W and 0 < v[0] < calib.RGB_H


def test_robot_mapping_convention():
    # robot y -> depth -x (as in ADA); robot x -> depth -y (ADA's rotz(+90 deg) gives +y)
    assert np.allclose(frame_offset([0, 0.1, 0]), [-0.1, 0, 0])
    assert np.allclose(frame_offset([0.1, 0, 0]), [0, -0.1, 0])
    assert np.allclose(calib.robot_to_depth_position([0, 0.1, 0]), frame_offset([0, 0.1, 0]))
    assert np.isclose(abs(np.linalg.det(M_ROBOT_TO_DEPTH)), 1.0)
    p = np.array([0.88, 0.45, 0.0])
    assert np.allclose(depth_to_rgb(p, p), calib.T_DEPTH_TO_RGB)
    # identical to ADA whenever the depth and RGB poses differ only in robot y
    q = p + [0, 0.03, 0]
    assert np.allclose(depth_to_rgb(p, q), calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, p, q))


def test_fov_excludes_corners():
    assert in_depth_fov(np.array([[0.0, 0.0, 0.5]]))[0]
    assert not in_depth_fov(np.array([[0.24, 0.19, 0.5]]))[0]


def test_points_to_depth_grid_keeps_the_front_surface():
    r = depth_rays()
    rays = r.reshape(-1, 2)
    soil = np.column_stack([rays * 0.5, np.full(len(rays), 0.5)])
    leaf_rays = r[80:90, 100:110].reshape(-1, 2)
    leaf = np.column_stack([leaf_rays * 0.42, np.full(len(leaf_rays), 0.42)])
    g = points_to_depth_grid(np.vstack([soil, leaf]))
    assert np.allclose(g[80:90, 100:110], 0.42, atol=1e-4)  # soil behind the leaf is dropped
    assert np.allclose(g[40, 40], 0.5, atol=1e-4)
    assert (points_to_depth_grid(soil[:0]) == 0).all()
