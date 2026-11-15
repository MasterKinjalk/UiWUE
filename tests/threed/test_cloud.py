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


def test_fill_small_holes_stays_within_the_local_rim_range():
    # Telea on float32 returned values far outside the rim with OpenCV 5.0 (a 4 x 6 hole on a
    # 0.30 / 0.40 m step filled with -0.86..1.57 m). Every filled value must lie within the
    # min/max of the valid pixels of its 7 x 7 window.
    d = flat(0.40)
    d[:, : W // 2] = 0.30  # step between two levels
    d[80:84, W // 2 - 3 : W // 2 + 3] = 0  # 4 x 6 hole across the step
    d[40:45, 30:36] = 0  # hole on the upper level only
    ramp = np.tile(np.linspace(0.42, 0.52, 20, dtype=np.float32), (20, 1))
    d[130:150, 150:170] = ramp  # tilted patch with a hole in it
    d[137:142, 157:163] = 0
    f, mask = fill_small_holes(d, max_rim_span=np.inf)  # rim-span limit off: checks the values only
    assert mask.sum() == 24 + 30 + 30
    valid = np.where(mask, np.nan, d)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(valid, 3, constant_values=np.nan), (7, 7))
    lo, hi = np.nanmin(win, axis=(-2, -1)), np.nanmax(win, axis=(-2, -1))
    r, c = np.nonzero(mask)
    tol = 2e-4  # 0.2 mm
    assert (f[r, c] >= lo[r, c] - tol).all() and (f[r, c] <= hi[r, c] + tol).all()
    assert np.allclose(f[40:45, 30:36], 0.30, atol=1e-3)


def test_fill_small_holes_skips_holes_on_depth_edges():
    # A hole whose rim spans a depth edge (leaf over soil, leaf over leaf) is a gap, not a
    # saturated leaf interior: inpainting would bridge it with a steep fake surface.
    d = flat(0.40)
    d[:, : W // 2] = 0.30  # 100 mm step
    step = np.s_[80:84, W // 2 - 3 : W // 2 + 3]
    d[step] = 0  # hole across the step
    d[40:45, 30:36] = 0  # hole inside the upper level
    ramp = np.tile(np.linspace(0.42, 0.44, 20, dtype=np.float32), (20, 1))  # about 1 mm per px
    d[130:150, 150:170] = ramp
    tilt = np.s_[137:142, 157:163]
    d[tilt] = 0  # hole inside a gently tilted surface: rim span 7.4 mm
    f, mask = fill_small_holes(d)
    assert not mask[step].any() and (f[step] == 0).all()
    assert mask[40:45, 30:36].all() and np.allclose(f[40:45, 30:36], 0.30, atol=1e-3)
    assert mask[tilt].all() and f[tilt].min() >= 0.42 - 2e-4 and f[tilt].max() <= 0.44 + 2e-4
    # the limit compares the rim span (max - min of the valid 8-neighbours of the hole)
    assert fill_small_holes(d, max_rim_span=0.008)[1][tilt].all()
    assert not fill_small_holes(d, max_rim_span=0.007)[1][tilt].any()


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
    u, v, _z, ok, _ = project_to_rgb(np.array([[0.0, 0.0, 0.47]]), depth_to_rgb(pos, pos))
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
