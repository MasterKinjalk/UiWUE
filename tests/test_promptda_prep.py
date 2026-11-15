"""Geometry checks for the Prompt Depth Anything sample builder (no dataset needed)."""

import numpy as np

from uiwue.ml.promptda import prep
from uiwue.ml.promptda.infer import output_intrinsics
from uiwue.threed import calib


def flat_scene(z: float = 0.45) -> np.ndarray:
    return np.full((calib.DEPTH_H, calib.DEPTH_W), z, np.float32)


def test_flat_plane_projects_to_near_constant_rgb_depth():
    pos = np.array([0.88, 0.45, 0.0])
    t = calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, pos, pos)
    uvz = prep.project(prep.depth_points(flat_scene()), t)
    # The RGB camera is tilted ~3.3 deg against the depth camera (rvec y = 0.058), so a flat
    # plane at 0.45 m spans ~18 mm of RGB depth across the ~0.47 m wide view.
    assert 0.01 < np.ptp(uvz[:, 2]) < 0.03
    crop = prep.crop_for(uvz)
    out = prep.to_output_px(uvz, crop, (prep.OUT_H, prep.OUT_W))
    # The cameras sit 4-5 cm apart and the ToF VFOV is wider than the RGB VFOV, so ~30 % of
    # the ToF view has no RGB pixel (69 % land in the crop at 0.45 m).
    assert len(out) > 0.6 * calib.DEPTH_W * calib.DEPTH_H


def test_prompt_is_filled_and_in_range():
    pos = np.array([0.88, 0.45, 0.0])
    t = calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, pos, pos)
    uvz = prep.project(prep.depth_points(flat_scene(0.40)), t)
    out = prep.to_output_px(uvz, prep.crop_for(uvz), (prep.OUT_H, prep.OUT_W))
    prompt, valid = prep.splat_prompt(out)
    assert prompt.shape == (prep.PROMPT_H, prep.PROMPT_W)
    assert valid.mean() > 0.6
    assert np.all(prompt > 0.35) and np.all(prompt < 0.45)


def test_output_intrinsics_reproject_points():
    """Unprojecting output pixels with k_out and re-projecting with K_RGB lands in the crop."""
    pos = np.array([0.88, 0.45, 0.0])
    t = calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, pos, pos)
    uvz = prep.project(prep.depth_points(flat_scene()), t)
    crop = prep.crop_for(uvz)
    out = prep.to_output_px(uvz, crop, (prep.OUT_H, prep.OUT_W))
    k = output_intrinsics(crop)
    x = (out[:, 0] - k[0, 2]) * out[:, 2] / k[0, 0]
    y = (out[:, 1] - k[1, 2]) * out[:, 2] / k[1, 1]
    u = calib.K_RGB[0, 0] * x / out[:, 2] + calib.K_RGB[0, 2]
    v = calib.K_RGB[1, 1] * y / out[:, 2] + calib.K_RGB[1, 2]
    inside = (u >= 0) & (u < calib.RGB_W) & (v >= 0) & (v < calib.RGB_H)
    assert inside.mean() > 0.99


def test_cross_frame_shift_moves_points():
    """Depth frame 0.1 m away along robot y lands shifted in the RGB image of the other frame."""
    p0, p1 = np.array([0.88, 0.45, 0.0]), np.array([0.88, 0.55, 0.0])
    pts = prep.depth_points(flat_scene())
    a = prep.project(pts, calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, p0, p0))
    b = prep.project(pts, calib.depth_to_sensor(calib.T_DEPTH_TO_RGB, p1, p0))
    shift_px = np.median(np.hypot(*(b[:, :2] - a[:, :2]).T))
    # 0.1 m at 0.45 m with f = 1397 px -> ~310 px
    assert 250 < shift_px < 370
