"""End-to-end: the CLI on a tiny real subset, cross-frame geometry, and the parity mode against
the ADA code. Skipped when the raw data (or the ADA repository) is absent."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uiwue.threed import parity, run, schema
from uiwue.threed.io import Run

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUN = ("W1_J5", "2025_09_02", "15")
HAVE_DATA = all((DATA / m / RUN[0] / RUN[1] / RUN[2]).is_dir() for m in ("depth", "rgb"))
ADA = ROOT / "external" / "ada2025" / "analysis" / "plant_height"

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="raw ADA data not on disk")


@needs_data
def test_cli_smoke_and_resume(tmp_path):
    args = [
        "--data-dir",
        str(DATA),
        "--plants",
        "W1_J5",
        "--dates",
        "2025-09-02",
        "2025-09-02",
        "--hours",
        "15",
        "--jobs",
        "1",
        "--out",
        str(tmp_path),
    ]
    assert run.main(args) == 0
    df = pd.read_parquet(tmp_path / "traits_runs.parquet")
    schema.validate_runs(df)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["n_frames_ok"] >= 2 and 50 < r["height_p75"] < 250
    fr = pd.read_parquet(tmp_path / "traits_frames.parquet")
    assert set(schema.FRAME_COLUMNS) <= set(fr.columns) and fr["frame"].nunique() == 15
    cloud = np.load(tmp_path / "clouds" / "W1_J5" / "2025-09-02_15.npz")
    assert all(k in cloud.files for k in schema.CLOUD_KEYS)
    assert cloud["xyz"].dtype == np.float32 and cloud["rgb"].dtype == np.uint8
    assert np.allclose(cloud["height"], cloud["xyz"][:, 2]) and cloud["plant"].any()
    assert (pd.read_csv(tmp_path / "run_log.csv")["status"] == "ok").all()
    assert run.main(args) == 0  # resumable: the run is skipped
    assert len(pd.read_csv(tmp_path / "run_log.csv")) == 1


@needs_data
def test_plant_lands_on_the_same_spot_in_every_frame():
    res = run.process_run(Run(*RUN), DATA, None, debug=True)
    g = res["debug"]
    assert len(g["used"]) >= 3
    cents = np.array([g["loc"][k][g["iso"][k].mask, :2].mean(0) for k in g["used"]])
    assert np.max(np.linalg.norm(cents - np.median(cents, 0), axis=1)) < 0.015


@needs_data
@pytest.mark.skipif(not ADA.is_dir(), reason="ADA repository not present")
def test_parity_matches_ada_code():
    pytest.importorskip("plyfile")
    sys.path.insert(0, str(ADA))
    import step_01_build_pointclouds as s1
    import step_02_compute_metrics as s2

    r = Run(*RUN)
    ddir, rdir = r.dir(DATA, "depth"), r.dir(DATA, "rgb")
    rd = s1._find_row_for_frame(s1._read_csv_rows(ddir / "frames_depth.csv"), 7)
    rr = s1._find_row_for_frame(s1._read_csv_rows(rdir / "frames_rgb.csv"), 7)
    depth = s1._load_depth_image(ddir, 7).astype(np.float32)
    if float(np.nanmedian(depth)) < 0:
        depth = -depth
    depth[depth <= 0] = 0.0
    sh, pos = s1.cc.shift_matrix, s1.cc.robot_to_depth_position
    t = s1.T_DEPTH_TO_RGB @ np.linalg.inv(sh(pos(s1._get_robot_position(rr)))) @ sh(pos(s1._get_robot_position(rd)))
    da, col = s1._colorize_depth_with_rgb(depth, s1._load_rgb_image(rdir, 7), t)
    pts, cols = s1._depth_to_colorized_point_cloud(da, col)
    c = cols.astype(np.float32)
    exg = 2 * c[:, 1] - c[:, 0] - c[:, 2]
    m = s2._compute_metrics(pts[:, 2].astype(np.float64), exg.astype(np.float64), 5.0)
    ours = parity.parity_run(r, DATA)
    assert ours["z_p25_m"] == pytest.approx(m["z_p25"], abs=1e-9)
    assert ours["n_veg"] == m["num_points_vegetation"]
    assert ours["height_parity_mm"] == pytest.approx(1000 * (0.60 - m["z_p25"]))


@needs_data
def test_depth_fn_hook_and_refined_cache(tmp_path):
    """--depth-model plumbing: a cached map on the ToF grid is merged with the ToF fallback, and
    a refined map equal to the filtered ToF reproduces the raw traits."""
    from uiwue.threed.cloud import fill_small_holes, median3

    r = Run(*RUN)
    raw = run.process_run(r, DATA, None, want_cloud=False)
    plan = run.process_run(r, DATA, None, plan_only=True)
    assert 7 in plan["build"] and set(plan["build"]) >= {4, 7, 10}

    def same_as_tof(run_, k, depth):
        d = median3(fill_small_holes(np.where(depth > 0, depth, 0).astype(np.float32))[0])
        return d, np.ones(d.shape, bool)

    res = run.process_run(r, DATA, None, want_cloud=False, depth_fn=same_as_tof)
    assert res["run"]["refined_frac"] == 1.0
    for t in ("height_p75", "projected_area", "n_points"):
        assert res["run"][t] == pytest.approx(raw["run"][t], rel=1e-6), t

    # cache with the left half empty -> ToF fallback there
    from uiwue.threed.io import load_depth, run_frames

    z, _ = load_depth(run_frames(r, DATA)["depth_files"][7])
    half = same_as_tof(r, 7, z)[0]
    half[:, : half.shape[1] // 2] = 0
    f = run.refined_path(tmp_path, r, 7)
    f.parent.mkdir(parents=True)
    np.savez_compressed(f, depth=half)
    merged, mask = run.RefinedDepth(tmp_path)(r, 7, z)
    assert not mask[:, :100].any() and mask[:, 130:].mean() > 0.8
    assert (merged[:, :100] == same_as_tof(r, 7, z)[0][:, :100]).all()
    assert run.RefinedDepth(tmp_path)(r, 4, z) is None  # no cached map -> keep the ToF
