"""Camera calibration of the ADA 2025 robot camera package.

Copied from the ADA 2025 repository, ``analysis/combined/camera_calibration.py``
(https://github.com/pippowell/ada2025, commit 77770dfa22053e8372be0ed88c66df6313e84d7f,
2026-10-28). Frames, the per-frame robot-shift composition and the projection math are
documented in that repository's ``docs/camera_calibration.md``. Keep the numbers in sync.
"""

from __future__ import annotations

import cv2
import numpy as np

ADA_COMMIT = "77770dfa22053e8372be0ed88c66df6313e84d7f"

# Intrinsics (pinhole) and distortion (OpenCV 5-parameter k1 k2 p1 p2 k3)
K_DEPTH = np.array(
    [
        [213.07864182105573, 0.0, 106.30867219370424],
        [0.0, 213.4773675713787, 87.48242329736486],
        [0.0, 0.0, 1.0],
    ]
)
DEPTH_DISTORTION = np.array(
    [
        0.06563883079346727,
        -2.1462902300917204,
        0.0004996931251144205,
        -0.003320348250271367,
        3.229347799887664,
    ]
)
K_RGB = np.array(
    [
        [1397.1693759311743, 0.0, 932.7568965495196],
        [0.0, 1391.8123785653943, 566.6211724978231],
        [0.0, 0.0, 1.0],
    ]
)
RGB_DISTORTION = np.array(
    [
        0.14612012441410802,
        -0.3064798846037176,
        0.0018885253236924308,
        -0.006218715146850731,
        0.2488477543929147,
    ]
)
K_FLIR = np.array(
    [
        [1235.8577572886502, 0.0, 289.0805485094724],
        [0.0, 1236.4101137173182, 265.5147977940737],
        [0.0, 0.0, 1.0],
    ]
)
FLIR_DISTORTION = np.array(
    [
        0.07142030350380577,
        5.283136483147005,
        0.01630229908552685,
        -0.01628107985657429,
        -40.61658327283593,
    ]
)

# Extrinsics: depth camera frame -> sensor frame (Rodrigues vector + translation in metres)
RVEC_DEPTH_TO_RGB = (-0.010361366746608086, 0.057755279059599554, 3.1366704147707054)
TVEC_DEPTH_TO_RGB = (0.0537813722846987, 0.04292643179470712, -0.0019552023110125907)
RVEC_DEPTH_TO_FLIR = (0.0034610286791345066, 0.07564865751693899, -0.03625049923293348)
TVEC_DEPTH_TO_FLIR = (0.029305079934097405, -0.05172043896398373, -0.01493650923515098)

# The depth camera is mounted rotated 90 degrees about the robot z-axis.
ROBOT_TO_DEPTH_YAW_DEG = 90.0

DEPTH_W, DEPTH_H = 224, 171
RGB_W, RGB_H = 1920, 1080
FLIR_W, FLIR_H = 640, 480


def rotation_matrix(rvec) -> np.ndarray:
    """3x3 rotation matrix from a Rodrigues vector."""
    rmat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rmat


def transformation_matrix(rvec, tvec) -> np.ndarray:
    """4x4 rigid transform from a Rodrigues vector and a translation."""
    tfm = np.eye(4)
    tfm[:3, :3] = rotation_matrix(rvec)
    tfm[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return tfm


def rotz(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


T_DEPTH_TO_RGB = transformation_matrix(RVEC_DEPTH_TO_RGB, TVEC_DEPTH_TO_RGB)
T_DEPTH_TO_FLIR = transformation_matrix(RVEC_DEPTH_TO_FLIR, TVEC_DEPTH_TO_FLIR)

# ADA's robot -> depth mapping, rotz(+90 deg): robot (x, y) -> depth (-y, +x). Kept for reference
# and for exact paper parity only.
R_ROBOT_TO_DEPTH_ADA = rotz(np.deg2rad(ROBOT_TO_DEPTH_YAW_DEG))
# Corrected mapping (deviation from ADA, verified 2026-11-14): robot x maps to depth -y, robot y
# to depth -x. With it, ToF frames from different robot columns agree to the same 5-11 mm median
# as frames within one column (W1_J5 2025-08-20: 14.0 -> 5.3 mm; W2_J5 2025-09-03: 52.2 ->
# 11.1 mm). Robot y maps identically under both, so same-column work and depth -> RGB colouring
# (depth and RGB poses differ almost only in y) are unaffected. docs/threed/PIPELINE.md.
R_ROBOT_TO_DEPTH = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def robot_to_depth_position(position_robot) -> np.ndarray:
    """Map a robot-frame position (frames_<sensor>.csv) into the depth camera frame."""
    return R_ROBOT_TO_DEPTH @ np.asarray(position_robot, dtype=np.float64)


def shift_matrix(position) -> np.ndarray:
    """4x4 translation-only transform."""
    tfm = np.eye(4)
    tfm[:3, 3] = position
    return tfm


def depth_to_sensor(t_depth_to_sensor: np.ndarray, pos_depth_robot, pos_sensor_robot) -> np.ndarray:
    """Extrinsic from depth frame (captured at ``pos_depth_robot``) to a sensor frame captured at
    ``pos_sensor_robot``: ``T @ inv(shift_sensor) @ shift_depth`` (docs/camera_calibration.md).

    Works across frames: depth frame j into the RGB camera of frame k.
    """
    shift_d = shift_matrix(robot_to_depth_position(pos_depth_robot))
    shift_s = shift_matrix(robot_to_depth_position(pos_sensor_robot))
    return t_depth_to_sensor @ np.linalg.inv(shift_s) @ shift_d
