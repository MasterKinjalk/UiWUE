"""Run the whole analysis on someone's own data folder, into a workspace of its own.

    python -m uiwue.pipeline --data-dir D:/my_trial --name my-trial
    python -m uiwue.pipeline --data-dir D:/my_trial --steps thermal detect --plants W1_A2 W2_J5 --jobs 8
    python -m uiwue.pipeline --data-dir D:/my_trial --ml mask fusion      # the learned models (GPU)

The folder follows the ADA 2025 layout (``thermal/``, ``depth/``, ``rgb/`` with one
``<bed>_<plant>/<date>/<hour>/`` folder per run, ``air_readings.parquet``, ``leaf_readings.parquet``,
``metadata_rwc_par_en.xlsx``) and the trial's calendar and beds (``uiwue.core``). Each step is one of the
modules' own command lines, started as a child process; a step whose input is missing is skipped, and a
failed step does not stop the ones that do not need it.

The workspace is ``runs/user/<name>/`` (``UIWUE_RUNS_DIR`` moves ``runs/user``):

    derived/      the team tables, as in the repo's derived/ (the children get UIWUE_DERIVED)
    threed/       the 3D pipeline's own files: traits, clouds, plant references
    cache/        fusion embeddings
    status.json   the steps and their state, read by the app's New run page
    log.txt       the children's output
    beat          touched every few seconds while the run is alive

Nothing is written to the repo's ``derived/``, so the app keeps showing the committed tables. To look at a
run, start a second app on its tables:

    UIWUE_DERIVED=runs/user/<name>/derived UIWUE_CLOUDS=runs/user/<name>/threed/clouds \\
        UIWUE_DATA=<data dir> python app/app.py --port 8061

(the New run page does this with its View button). The learned models never run unless ``--ml`` names them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from uiwue.core import REPO

RUNS_DIR = Path(os.environ.get("UIWUE_RUNS_DIR") or REPO / "runs" / "user")
PLANT_RE = re.compile(r"^(W\d+)_([A-Z]\d+)$")
ZIP_RE = re.compile(r"^rgb_(W\d+)_([A-Z]\d+)\.zip$")
DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
BEDS = ("W1", "W2", "W3")
BEAT_SECONDS = 5
ALIVE_SECONDS = 20  # a run whose beat is older than this is no longer running
ML = ("mask", "depth", "fusion")

# ---------------------------------------------------------------- what the folder holds
INPUTS = {  # key -> (name, what is looked for)
    "thermal": ("Thermal", "thermal/<bed>_<plant>/<date>/<hour>/thermal_*.tif"),
    "depth": ("Depth", "depth/<bed>_<plant>/<date>/<hour>/"),
    "rgb": ("RGB", "rgb/<bed>_<plant>/ or rgb/rgb_<bed>_<plant>.zip"),
    "hsi": ("Hyperspectral", "hsi/<bed>_<plant>/<date>/<hour>/"),
    "air": ("Air sensors", "air_readings.parquet"),
    "clip": ("Leaf clip", "leaf_readings.parquet and metadata_rwc_par_en.xlsx"),
}


def _folder_date(name: str):
    return datetime.strptime(name, "%Y_%m_%d").date() if DATE_RE.match(name) else None


def _scan_plants(root: Path, zips: bool = False) -> dict:
    """Plant folders (and RGB zips) under ``root``: IDs, the first and last date folder, plant-days."""
    ids, zipped, dates, days = set(), 0, set(), 0
    if root.is_dir():
        with os.scandir(root) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if zips and (m := ZIP_RE.match(e.name)) and e.is_file():
                ids.add(f"{m[1]}_{m[2]}")
                zipped += 1
            elif PLANT_RE.match(e.name) and e.is_dir():
                ids.add(e.name)
                with os.scandir(e.path) as sub:
                    found = [d for d in (_folder_date(x.name) for x in sub) if d]
                dates.update(found)
                days += len(found)
    return dict(
        found=bool(ids),
        plants=len(ids),
        ids=sorted(ids),
        beds={b: sum(i.startswith(b + "_") for i in ids) for b in sorted({i.split("_")[0] for i in ids})},
        first=min(dates).isoformat() if dates else None,
        last=max(dates).isoformat() if dates else None,
        plant_days=days,
        zipped=zipped,
    )


def scan(data_dir) -> dict:
    """What a data folder holds, per input: {key: dict(found, plants, beds, first, last, ...)}. Cheap: it lists
    the plant and date folders and opens no image."""
    d = Path(data_dir)
    out = {k: _scan_plants(d / k, zips=k == "rgb") for k in ("thermal", "depth", "rgb", "hsi")}
    out["air"] = dict(found=(d / "air_readings.parquet").is_file())
    out["clip"] = dict(
        found=(d / "leaf_readings.parquet").is_file() and (d / "metadata_rwc_par_en.xlsx").is_file()
    )
    out["light"] = dict(found=(d / "metadata_rwc_par_en.xlsx").is_file())
    for k, (name, where) in INPUTS.items():
        out[k] |= dict(name=name, where=where)
    return out


def other_beds(found: dict) -> list[str]:
    """Beds in the folder that the analysis does not know (it compares W2 and W3 against W1)."""
    beds = {b for k in ("thermal", "depth", "rgb") for b in found.get(k, {}).get("beds", {})}
    return sorted(beds - set(BEDS))


# ---------------------------------------------------------------- the learned models
def ml_python() -> str | None:
    """The interpreter for the model steps: UIWUE_ML_PYTHON, this one when it has torch, else .venv-ml."""
    if os.environ.get("UIWUE_ML_PYTHON"):
        return os.environ["UIWUE_ML_PYTHON"]
    if importlib.util.find_spec("torch") is not None:
        return sys.executable
    for p in (REPO / ".venv-ml" / "Scripts" / "python.exe", REPO / ".venv-ml" / "bin" / "python"):
        if p.exists():
            return str(p)
    return None


def _first(*paths) -> Path | None:
    return next((Path(p) for p in paths if p and Path(p).exists()), None)


def weights() -> dict:
    """Where each model's weights are, None when they are not on this machine."""
    fusion = _first(
        os.environ.get("UIWUE_FUSION_RUNS"), REPO / "runs" / "fusion", REPO / ".worktrees" / "fusion" / "runs" / "fusion"
    )
    if fusion is not None and not (fusion / "fusion" / "fold0.pt").exists():
        fusion = None
    return dict(
        mask=_first(os.environ.get("UIWUE_PLANT_MODEL"), REPO / "runs" / "leafseg-m2f" / "best"),
        depth=_first(os.environ.get("UIWUE_DEPTH_MODEL"), REPO / "runs" / "promptda-vitl" / "best"),
        fusion=fusion,
        imagebind=_first(
            os.environ.get("UIWUE_IMAGEBIND_CKPT"),
            REPO / ".checkpoints" / "imagebind_huge.pth",
            REPO / ".worktrees" / "fusion" / ".checkpoints" / "imagebind_huge.pth",
        ),
    )


def ml_ready(found: dict) -> dict:
    """{model: reason it cannot run, or None when it can}."""
    w, py = weights(), ml_python()
    out = {}
    for key in ML:
        if py is None:
            out[key] = "no Python with torch (set UIWUE_ML_PYTHON)"
        elif w[key] is None:
            out[key] = "weights not on this machine"
        elif key == "fusion" and w["imagebind"] is None:
            out[key] = "ImageBind checkpoint not on this machine"
        elif key in ("mask", "depth") and not found["depth"]["found"]:
            out[key] = "no depth folder"
        elif key in ("mask", "depth") and not found["rgb"]["found"]:
            out[key] = "no RGB folder"
        elif key == "fusion" and not any(found[k]["found"] for k in ("thermal", "depth", "rgb")):
            out[key] = "no camera folder"
        else:
            out[key] = None
    return out


# ---------------------------------------------------------------- the steps
@dataclass(frozen=True)
class Step:
    key: str
    label: str  # 1 or 2 words, as on the page
    reads: tuple  # inputs (``scan`` keys) that must all be found
    any_of: tuple = ()  # ... and at least one of these, when given
    tables: tuple = ()  # workspace files it needs when it starts: one of each inner tuple
    writes: str = ""
    about: str = ""
    ml: bool = False


T = "derived/{}/traits_runs.parquet"
STEPS = (
    Step(
        "env", "Environment", ("air", "light"), writes="env/env_runs",
        about="Air temperature, humidity, VPD and light at every flyover (uiwue.env). Downloads one DWD weather "
        "file (1.5 MB) into <data>/external/ on its first run.",
    ),
    Step(
        "thermal", "Thermal", ("thermal", "air"), writes="thermal/traits_runs",
        about="Leaf temperature per run, against the air and against the control bed (uiwue.thermal).",
    ),
    Step(
        "truth", "Leaf clip", ("clip",), tables=(("derived/env/env_runs.parquet",),), writes="truth/turgor_runs, plants",
        about="Leaf-clip turgor, water content and symptom dates: the ground truth (uiwue.truth).",
    ),
    Step("rgb", "RGB", ("rgb",), writes="rgb/traits_runs", about="Greenness and cover from the centre frame (uiwue.rgb)."),
    Step(
        "threed", "3D structure", ("depth",), writes="threed/traits_runs, clouds",
        about="Height, leaf area and volume from the depth camera, with one cloud per run (uiwue.threed). "
        "The slowest step without a model.",
    ),
    Step(
        "detect", "Detection", (), any_of=("thermal", "rgb", "depth"),
        tables=((T.format("thermal"), T.format("rgb"), T.format("threed")),),
        writes="detect/onsets, benchmark, scores_runs",
        about="First day each signal separates deficit from control, the sensor benchmark and the stress "
        "probability (uiwue.detect).",
    ),
    Step(
        "diurnal", "Daily rhythm", ("thermal",), tables=((T.format("thermal"),),), writes="diurnal/scores_runs, schedules",
        about="Thermal scores by time of day, the alert rules and the best flyover (uiwue.diurnal).",
    ),
    Step(
        "canopy", "3D statistics", ("depth",), tables=(("threed/traits_runs.parquet",),), writes="threed/results",
        about="Noise floor, bed baseline, collapse and recovery of the 3D traits (uiwue.threed.experiments E1 to E5).",
    ),
    Step(
        "fusion", "Fusion score", (), any_of=("thermal", "rgb", "depth"), writes="detect/fusion_external_*", ml=True,
        about="Embeds RGB, thermal and depth with the frozen encoders and scores every run with the trained "
        "fusion model (uiwue.fusion). Plants the model was trained on are left out. GPU.",
    ),
    Step("check", "Table check", (), tables=(("derived",),), about="The contract validator on every table written."),
)
BY_KEY = {s.key: s for s in STEPS}
CORE = tuple(s.key for s in STEPS if not s.ml)
ML_ABOUT = {
    "mask": ("Learned mask", "3D structure uses the Mask2Former plant mask in place of the greenness rule. GPU."),
    "depth": ("Refined depth", "3D structure refines every depth frame with Prompt Depth Anything first. GPU."),
    "fusion": ("Fusion score", BY_KEY["fusion"].about),
}


def missing_input(step: Step, found: dict) -> str | None:
    """Why the step cannot run on this folder (the input it lacks), or None."""
    for k in step.reads:
        if not found[k]["found"]:
            return f"no {INPUTS[k][1] if k in INPUTS else 'metadata_rwc_par_en.xlsx'}"
    if step.any_of and not any(found[k]["found"] for k in step.any_of):
        return "no camera folder"
    if step.ml:
        return ml_ready(found)[step.key]
    return None


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(name or "").strip()).strip("-")[:60]


def workspace(name: str) -> Path:
    return RUNS_DIR / safe_name(name)


def commands(step: Step, data_dir: Path, ws: Path, ml=(), plants=(), jobs=None, found=None) -> list[list[str]]:
    """The step's command lines (argument lists; ``python`` stands for the interpreter that runs it)."""
    d, jobs = str(data_dir), str(jobs or default_jobs())
    some = ["--plants", *plants] if plants else []
    mod = lambda m, *a: ["python", "-m", m, *a]  # noqa: E731
    w = weights()
    if step.key == "env":
        return [mod("uiwue.env", "--data-dir", d)]
    if step.key == "thermal":
        return [mod("uiwue.thermal", "--data-dir", d, *some)]
    if step.key == "truth":
        return [mod("uiwue.truth", "--data-dir", d)]
    if step.key == "rgb":
        return [mod("uiwue.rgb.run", "--data-dir", d, "--jobs", jobs, *some)]
    if step.key == "threed":
        models = (["--plant-model", str(w["mask"])] if "mask" in ml else []) + (
            ["--depth-model", str(w["depth"])] if "depth" in ml else []
        )
        return [
            mod("uiwue.threed.run", "--data-dir", d, "--out", str(ws / "threed"), "--jobs", jobs, *some, *models),
            mod("uiwue.threed.contract", "--source", str(ws / "threed" / "traits_runs.parquet")),
        ]
    if step.key == "detect":
        return [mod("uiwue.detect.run")]
    if step.key == "diurnal":
        return [mod("uiwue.diurnal")]
    if step.key == "canopy":
        return [
            mod(
                "uiwue.threed.experiments", "--exp", "E1", "E2", "E3", "E4", "E5",
                "--runs", str(ws / "threed" / "traits_runs.parquet"),
                "--frames", str(ws / "threed" / "traits_frames.parquet"),
                "--out", str(ws / "derived" / "threed" / "results"),
            )  # fmt: skip
        ]
    if step.key == "fusion":
        sensors = [k for k in ("rgb", "thermal", "depth") if found is None or found[k]["found"]]
        cache = str(ws / "cache" / "fusion")
        return [
            mod("uiwue.fusion.embed", "--data-dir", d, "--sensors", *sensors, "--cache", cache, *some),
            mod(
                "uiwue.fusion.predict", "--cache", cache, "--model", "fusion", "--runs-dir", str(w["fusion"]),
                "--data-dir", d, "--size-traits", str(ws / "derived" / "threed" / "traits_runs.parquet"),
                "--out", str(ws / "derived" / "detect"),
            ),  # fmt: skip
        ]
    if step.key == "check":
        return [mod("uiwue.core.validate")]
    raise KeyError(step.key)


def plan(data_dir, name, steps=None, ml=(), plants=(), jobs=None, found=None) -> list[dict]:
    """The run as a list of steps: key, label, state (``pending``, or ``skipped`` with ``why``), commands."""
    data_dir, ws = Path(data_dir), workspace(name)
    found = found or scan(data_dir)
    ml = [m for m in ML if m in ml]
    want = [k for k in (CORE if steps is None else steps) if k in CORE] + (["fusion"] if "fusion" in ml else [])
    out = []
    for s in STEPS:
        if s.key not in want:
            continue
        why = missing_input(s, found)
        cmds = [] if why else commands(s, data_dir, ws, ml, plants, jobs, found)
        out.append(
            dict(
                key=s.key, label=s.label, state="skipped" if why else "pending", why=why or "",
                cmds=[" ".join(c) for c in cmds], argv=cmds, ml=s.ml or (s.key == "threed" and bool(set(ml) & {"mask", "depth"})),
                seconds=None,
            )  # fmt: skip
        )
    return out


def command_line(data_dir, name, steps=None, ml=(), plants=(), jobs=None) -> str:
    """The one command that does what the page's Start button does."""
    q = lambda s: f'"{s}"' if " " in str(s) else str(s)  # noqa: E731
    parts = ["python -m uiwue.pipeline", "--data-dir", q(data_dir or "<folder>"), "--name", q(safe_name(name) or "<name>")]
    if steps is not None and tuple(k for k in CORE if k in steps) != CORE:
        parts += ["--steps", *[k for k in CORE if k in steps]]
    if ml:
        parts += ["--ml", *[m for m in ML if m in ml]]
    if plants:
        parts += ["--plants", *plants]
    if jobs:
        parts += ["--jobs", str(jobs)]
    return " ".join(parts)


# ---------------------------------------------------------------- status on disk
def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    for _ in range(20):  # a reader may hold the file for a moment on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def read_status(ws: Path) -> dict | None:
    """A workspace's status.json with ``alive`` added; a run whose process is gone reads as ``stopped``."""
    path = Path(ws) / "status.json"
    for _ in range(5):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    else:
        return None
    beat = Path(ws) / "beat"
    st["alive"] = st.get("state") == "running" and beat.exists() and time.time() - beat.stat().st_mtime < ALIVE_SECONDS
    if st.get("state") == "running" and not st["alive"]:
        st["state"] = "stopped"
        for s in st.get("steps", []):
            if s["state"] in ("running", "pending"):
                s["state"] = "stopped" if s["state"] == "running" else "skipped"
    st["workspace"] = str(ws)
    return st


def list_runs() -> list[dict]:
    """Every workspace's status, newest first."""
    if not RUNS_DIR.is_dir():
        return []
    out = [st for p in RUNS_DIR.iterdir() if p.is_dir() and (st := read_status(p))]
    return sorted(out, key=lambda s: s.get("started") or "", reverse=True)


def running() -> dict | None:
    return next((st for st in list_runs() if st["alive"]), None)


def log_tail(ws: Path, lines: int = 30) -> str:
    path = Path(ws) / "log.txt"
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 16_000))
        text = f.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


# ---------------------------------------------------------------- running it
def child_env(data_dir: Path, ws: Path) -> dict:
    env = os.environ | dict(
        UIWUE_DATA=str(data_dir), UIWUE_DERIVED=str(ws / "derived"), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO / "src"), env.get("PYTHONPATH")]))
    if (ck := weights()["imagebind"]) is not None:
        env["UIWUE_IMAGEBIND_CKPT"] = str(ck)
    return env


def _have(ws: Path, step: Step) -> str | None:
    """The table a step needs and the workspace lacks, or None."""
    for group in step.tables:
        paths = [ws / p for p in group]
        if not any(p.exists() and (p.is_file() or any(p.rglob("*.parquet"))) for p in paths):
            return " or ".join(group)
    return None


def run(data_dir, name, steps=None, ml=(), plants=(), jobs=None) -> dict:
    """Run the plan step by step and keep status.json current. Returns the final status."""
    data_dir, ws = Path(data_dir).resolve(), workspace(name)
    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} is not a folder")
    if not safe_name(name):
        raise SystemExit("the run needs a name")
    (ws / "derived").mkdir(parents=True, exist_ok=True)
    found = scan(data_dir)
    todo = plan(data_dir, name, steps, ml, plants, jobs, found)
    st = dict(
        name=safe_name(name), data_dir=str(data_dir), started=datetime.now(UTC).isoformat(timespec="seconds"),
        finished=None, state="running", pid=os.getpid(), ml=[m for m in ML if m in ml], plants=list(plants or ()),
        jobs=jobs or default_jobs(), steps=[{k: v for k, v in s.items() if k != "argv"} for s in todo],
    )  # fmt: skip
    status, beat, log = ws / "status.json", ws / "beat", ws / "log.txt"
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            beat.touch()
            stop.wait(BEAT_SECONDS)

    beat.touch()
    _write_json(status, st)
    threading.Thread(target=heartbeat, daemon=True).start()
    env, py_ml = child_env(data_dir, ws), ml_python()
    try:
        with log.open("a", encoding="utf-8") as out:
            out.write(f"\n===== {st['started']}  {st['name']}  <-  {data_dir}\n")
            for s, row in zip(todo, st["steps"], strict=True):
                if row["state"] == "skipped":
                    continue
                if lack := _have(ws, BY_KEY[s["key"]]):
                    row.update(state="skipped", why=f"needs {lack}")
                    _write_json(status, st)
                    continue
                row["state"], t0 = "running", time.time()
                _write_json(status, st)
                code = 0
                for argv in s["argv"]:
                    exe = py_ml if s["ml"] and py_ml else sys.executable
                    out.write(f"\n--- {s['label']}: {' '.join(argv)}\n")
                    out.flush()
                    code = subprocess.run([exe, *argv[1:]], cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT).returncode  # noqa: S603
                    if code:
                        break
                row.update(state="failed" if code else "done", seconds=round(time.time() - t0, 1))
                if code:
                    row["why"] = f"exit code {code}: see the log"
                _write_json(status, st)
    finally:
        stop.set()
        ran = [r["state"] for r in st["steps"] if r["state"] != "skipped"]
        if st["state"] == "running":
            st["state"] = (
                "failed" if not ran or all(x == "failed" for x in ran) else "partial" if "failed" in ran else "done"
            )
        if any(r["state"] == "running" for r in st["steps"]):  # interrupted
            st["state"] = "stopped"
            for r in st["steps"]:
                r["state"] = {"running": "stopped", "pending": "skipped"}.get(r["state"], r["state"])
        st["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
        _write_json(status, st)
    return st


def start(data_dir, name, steps=None, ml=(), plants=(), jobs=None) -> Path:
    """Start ``run`` as a process of its own (it outlives the caller) and return the workspace."""
    ws = workspace(name)
    ws.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "uiwue.pipeline", "--data-dir", str(data_dir), "--name", safe_name(name)]
    if steps is not None:
        argv += ["--steps", *[k for k in CORE if k in steps]]
    if ml:
        argv += ["--ml", *[m for m in ML if m in ml]]
    if plants:
        argv += ["--plants", *plants]
    if jobs:
        argv += ["--jobs", str(jobs)]
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))}
    for k in ("UIWUE_DERIVED", "UIWUE_CLOUDS", "UIWUE_RUN_NAME"):
        env.pop(k, None)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    (ws / "status.json").unlink(missing_ok=True)  # the new process writes a fresh one
    subprocess.Popen(  # noqa: S603
        argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, start_new_session=os.name != "nt",
    )  # fmt: skip
    return ws


def stop(name) -> bool:
    """Stop a running run and its children; True when there was one to stop."""
    ws = workspace(name)
    st = read_status(ws)
    if not st or not st["alive"]:
        return False
    pid = int(st["pid"])
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)  # noqa: S603, S607
    else:
        import signal

        os.killpg(os.getpgid(pid), signal.SIGTERM)
    (ws / "beat").unlink(missing_ok=True)
    st = json.loads((ws / "status.json").read_text(encoding="utf-8"))
    st.update(state="stopped", finished=datetime.now(UTC).isoformat(timespec="seconds"))
    for r in st["steps"]:
        r["state"] = {"running": "stopped", "pending": "skipped"}.get(r["state"], r["state"])
    _write_json(ws / "status.json", st)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path, help="a folder in the ADA 2025 layout")
    ap.add_argument("--name", help="the workspace, runs/user/<name> (default: the folder's name); a name in use resumes")
    ap.add_argument("--steps", nargs="*", choices=CORE, help="default: every step whose input is found")
    ap.add_argument("--ml", nargs="*", default=[], choices=ML, help="learned models to use (GPU); none by default")
    ap.add_argument("--plants", nargs="*", default=[], help="full IDs such as W1_A2 (default: all found)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print what is found and the steps, run nothing")
    a = ap.parse_args(argv)
    name = a.name or a.data_dir.resolve().name
    if a.dry_run:
        found = scan(a.data_dir)
        for k in INPUTS:
            f = found[k]
            print(f"{f['name']:<14} {'found' if f['found'] else 'missing':<8} {f.get('plants', '')}")
        for s in plan(a.data_dir, name, a.steps, a.ml, a.plants, a.jobs, found):
            print(f"{s['label']:<14} {s['state']:<8} {s['why'] or ' && '.join(s['cmds'])}")
        return 0
    if (other := running()) and other["name"] != safe_name(name):
        raise SystemExit(f"run {other['name']} is still going; one at a time")
    st = run(a.data_dir, name, a.steps, a.ml, a.plants, a.jobs)
    for s in st["steps"]:
        print(f"{s['label']:<14} {s['state']:<8} {s['why']}")
    print(f"{st['state']} -> {workspace(name)}")
    return 0 if st["state"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
