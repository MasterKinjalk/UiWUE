"""uiwue.pipeline: a data folder of one's own through every step, into a workspace, never into derived/."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from uiwue import core
from uiwue import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def folder(tmp_path):
    """A data folder in the ADA layout: thermal and depth for three plants, RGB as a zip, the air file."""
    d = tmp_path / "trial"
    for sensor in ("thermal", "depth"):
        for pid in ("W1_A2", "W2_A2", "W3_B1"):
            for day in ("2025_09_01", "2025_09_04"):
                (d / sensor / pid / day / "15").mkdir(parents=True)
    (d / "rgb").mkdir()
    (d / "rgb" / "rgb_W1_A2.zip").write_bytes(b"")
    (d / "thermal" / "notes").mkdir()  # not a plant folder
    (d / "air_readings.parquet").write_bytes(b"")
    return d


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


def test_scan_counts_plants_and_dates_without_opening_an_image(folder):
    f = pl.scan(folder)
    assert f["thermal"]["plants"] == 3 and f["thermal"]["beds"] == {"W1": 1, "W2": 1, "W3": 1}
    assert (f["thermal"]["first"], f["thermal"]["last"], f["thermal"]["plant_days"]) == ("2025-09-01", "2025-09-04", 6)
    assert f["rgb"]["found"] and f["rgb"]["zipped"] == 1 and f["rgb"]["first"] is None
    assert not f["hsi"]["found"] and f["air"]["found"] and not f["clip"]["found"]
    assert pl.other_beds(f) == []
    (folder / "depth" / "W4_A2").mkdir()
    assert pl.other_beds(pl.scan(folder)) == ["W4"]


def test_a_step_without_its_input_is_skipped_and_says_why(folder, runs_dir):
    steps = {s["key"]: s for s in pl.plan(folder, "t")}
    assert list(steps) == list(pl.CORE)  # every analysis step, no model step
    assert steps["thermal"]["state"] == "pending" and steps["threed"]["state"] == "pending"
    assert steps["env"]["state"] == "skipped" and "metadata_rwc_par_en.xlsx" in steps["env"]["why"]
    assert steps["truth"]["state"] == "skipped" and "leaf_readings.parquet" in steps["truth"]["why"]
    only = pl.plan(folder, "t", steps=["thermal", "detect"])
    assert [s["key"] for s in only] == ["thermal", "detect"]


def test_the_models_are_off_unless_named(folder, runs_dir, monkeypatch, tmp_path):
    plain = pl.plan(folder, "t")
    assert not any(s["ml"] for s in plain) and "fusion" not in [s["key"] for s in plain]
    text = " ".join(c for s in plain for c in s["cmds"])
    assert "--plant-model" not in text and "--depth-model" not in text and "uiwue.fusion" not in text
    w = dict(mask=tmp_path / "m", depth=tmp_path / "d", fusion=tmp_path / "f", imagebind=tmp_path / "i")
    monkeypatch.setattr(pl, "weights", lambda: w)
    monkeypatch.setattr(pl, "ml_python", lambda model="mask": "python-ml")
    steps = {s["key"]: s for s in pl.plan(folder, "t", ml=["mask", "fusion"], plants=["W1_A2"])}
    assert steps["threed"]["ml"] and f"--plant-model {w['mask']}" in steps["threed"]["cmds"][0]
    assert "--depth-model" not in steps["threed"]["cmds"][0] and "--plants W1_A2" in steps["threed"]["cmds"][0]
    assert steps["fusion"]["state"] == "pending" and "--sensors rgb thermal depth" in steps["fusion"]["cmds"][0]
    monkeypatch.setattr(pl, "weights", lambda: w | dict(fusion=None))
    assert pl.ml_ready(pl.scan(folder))["fusion"] == "weights not on this machine"


def test_every_output_path_is_inside_the_workspace(folder, runs_dir):
    ws = pl.workspace("t")
    assert ws == runs_dir / "t" and core.REPO / "derived" not in ws.parents
    for s in pl.plan(folder, "t"):
        for argv in s["argv"]:
            for flag in ("--out", "--source", "--runs", "--frames", "--cache"):
                if flag in argv:
                    assert ws in Path(argv[argv.index(flag) + 1]).parents, argv
    env = pl.child_env(folder, ws)
    assert env["UIWUE_DERIVED"] == str(ws / "derived") and env["UIWUE_DATA"] == str(folder)
    assert pl.safe_name("../my trial!") == "my-trial" and pl.safe_name("  ") == ""


def test_uiwue_derived_moves_the_tables_and_the_default_is_the_repo():
    code = "import uiwue.core as c; print(c.DERIVED)"
    env = {k: v for k, v in os.environ.items() if k != "UIWUE_DERIVED"} | {"PYTHONPATH": str(ROOT / "src")}

    def out(e):
        return subprocess.run(
            [sys.executable, "-c", code], env=e, capture_output=True, text=True, check=False
        ).stdout.strip()

    assert Path(out(env)) == ROOT / "derived"
    assert Path(out(env | {"UIWUE_DERIVED": str(ROOT / "elsewhere")})) == ROOT / "elsewhere"


def test_run_writes_status_step_by_step_and_a_failure_only_stops_what_needs_it(folder, runs_dir, monkeypatch):
    calls = []

    def fake(argv, cwd, env, stdout, stderr, check):
        module = argv[2]
        calls.append(module)
        derived = Path(env["UIWUE_DERIVED"])
        if module == "uiwue.thermal":
            (derived / "thermal").mkdir(parents=True)
            (derived / "thermal" / "traits_runs.parquet").write_bytes(b"")
        return subprocess.CompletedProcess(argv, 1 if module == "uiwue.threed.run" else 0)

    monkeypatch.setattr(pl.subprocess, "run", fake)
    st = pl.run(folder, "t")
    state = {s["key"]: s["state"] for s in st["steps"]}
    assert state == dict(
        env="skipped",
        thermal="done",
        truth="skipped",
        rgb="done",
        threed="failed",
        detect="done",
        diurnal="done",
        canopy="skipped",
        check="done",
    )
    assert "uiwue.threed.contract" not in calls  # the step's second command waits for the first
    assert st["state"] == "partial" and st["finished"]
    ws = pl.workspace("t")
    disk = pl.read_status(ws)
    assert disk["state"] == "partial" and not disk["alive"] and disk["workspace"] == str(ws)
    assert [r["name"] for r in pl.list_runs()] == ["t"] and pl.running() is None
    assert "uiwue.thermal --data-dir" in pl.log_tail(ws, 50)
    assert not pl.stop("t")  # nothing alive to stop


def test_a_run_whose_process_is_gone_reads_as_stopped(runs_dir):
    ws = runs_dir / "gone"
    ws.mkdir(parents=True)
    steps = [dict(key="thermal", label="Thermal", state="running", why="", seconds=None)]
    (ws / "status.json").write_text(json.dumps(dict(name="gone", state="running", started="2026-11-15", steps=steps)))
    st = pl.read_status(ws)
    assert st["state"] == "stopped" and not st["alive"] and st["steps"][0]["state"] == "stopped"


def test_the_command_line_is_what_start_runs(folder):
    assert (
        pl.command_line(folder, "t", pl.CORE, [], [], None) == f"python -m uiwue.pipeline --data-dir {folder} --name t"
    )
    line = pl.command_line("D:/my trial", "t", ["thermal", "detect"], ["mask"], ["W1_A2"], 4)
    assert line.endswith('--data-dir "D:/my trial" --name t --steps thermal detect --ml mask --plants W1_A2 --jobs 4')
