"""Hand-made thermal features for every plant, for fusion v2 (docs/fusion.md section 8).

    python -m uiwue.fusion.features --roots data/thermal cache/raw_external/thermal \
        --out cache/fusion_v2/thermal_features.parquet [--workers 4]

The benchmark's thermal features (uiwue.thermal: leaf temperature as the median of the
coolest half of the pixels, its 10th percentile, the 75th-percentile background, each a
median over the run's five frames) computed for every plant folder under the given roots,
at full resolution and on the image averaged to 160 x 120 and 80 x 60 pixels. The
thermal module's table (derived/thermal) covers only the 39 benchmark plants; these
columns reproduce it on them (tests/jack/test_fusion_v2.py) and extend it to the others.

Leaf minus air uses the W1-end air sensor at the capture time, as uiwue.thermal. The
control-bed reference (leaf minus the control plants' median in the same run) is not
stored: fusion v2 and the benchmark rerun take it from the training plants of each fold.

Per-plant results are cached beside --out (one parquet per plant), so a rerun only reads
the plants that are new.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from uiwue import core

RESOLUTIONS = {"full": 1, "r160": 4, "r80": 8}
STATS = ("leaf_t", "leaf_t_p10", "bg_t")  # per frame; leaf_t_mad is the spread of leaf_t over frames


def _frame_stats(img: np.ndarray) -> list[float]:
    from uiwue.thermal import block_mean, leaf_t

    out = []
    for f in RESOLUTIONS.values():
        x = img if f == 1 else block_mean(img, f)
        out += [leaf_t(x), float(np.percentile(x, 10)), float(np.percentile(x, 75))]
    return out


def run_features(run_dir: Path, frames: list[Path]) -> dict:
    """One run: each statistic at each resolution, median over the frames (uiwue.thermal.run_row)."""
    import tifffile

    s = np.array([_frame_stats(tifffile.imread(f).astype(np.float32)) for f in frames])
    med = np.median(s, axis=0)
    row = {}
    for i, res in enumerate(RESOLUTIONS):
        sfx = "" if res == "full" else f"_{res}"
        leaf = med[3 * i]
        row[f"thermal_leaf_t{sfx}"] = float(leaf)
        row[f"thermal_leaf_t_mad{sfx}"] = float(np.median(np.abs(s[:, 3 * i] - leaf)))
        row[f"thermal_leaf_t_p10{sfx}"] = float(med[3 * i + 1])
        row[f"thermal_bg_t{sfx}"] = float(med[3 * i + 2])
    csv = run_dir / "frames_thermal.csv"
    if csv.exists():
        ts = pd.Timestamp(int(pd.read_csv(csv).timestamp.median()), unit="ns", tz="UTC")
    else:
        ts = pd.Timestamp(f"{run_dir.parent.name.replace('_', '-')} {run_dir.name}:10", tz="Europe/Berlin").tz_convert(
            "UTC"
        )
    return row | dict(thermal_timestamp_utc=ts, thermal_n_frames_ok=len(frames))


def plant_features(plant_dir: Path) -> pd.DataFrame:
    """Every scheduled slot of one plant (the slot rule of uiwue.thermal.plant_rows)."""
    bed, plant = plant_dir.name.split("_")
    rows = []
    for date_dir in sorted(d for d in plant_dir.iterdir() if d.is_dir()):
        try:
            date = pd.Timestamp(date_dir.name.replace("_", "-"))
        except ValueError:
            continue
        if not core.DATES[0] <= date.date() <= core.DATES[1]:
            continue
        by_slot = {}
        for run_dir in (d for d in date_dir.iterdir() if d.is_dir() and d.name.isdigit()):
            slot, frames = core.slot_for_hour(int(run_dir.name)), sorted(run_dir.glob("thermal_*.tif"))
            if slot and frames:
                rank = (int(run_dir.name) == slot, len(frames))
                if slot not in by_slot or rank > by_slot[slot][0]:
                    by_slot[slot] = (rank, run_dir, frames)
        for slot, (_, run_dir, frames) in sorted(by_slot.items()):
            try:
                rows.append(dict(bed=bed, plant=plant, date=date, hour=slot) | run_features(run_dir, frames))
            except Exception as e:  # noqa: BLE001  (one unreadable run is skipped and reported)
                print(f"  {run_dir}: {e}")
    return pd.DataFrame(rows)


def _cached(plant_dir: Path, cache_dir: Path) -> Path:
    out = cache_dir / f"{plant_dir.name}.parquet"
    if not out.exists():
        df = plant_features(plant_dir)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
    return out


def add_air(df: pd.DataFrame, data_dir: Path = core.DATA) -> pd.DataFrame:
    """Leaf minus air (W1-end sensor at the capture time) at each resolution, and the QC flag."""
    from uiwue.env import air, at_times

    a = at_times(air(data_dir).air_t, df["thermal_timestamp_utc"])
    a = np.asarray(a, dtype=float)
    for res in RESOLUTIONS:
        sfx = "" if res == "full" else f"_{res}"
        df[f"thermal_leaf_air_dt{sfx}"] = df[f"thermal_leaf_t{sfx}"].to_numpy() - a
    df["thermal_qc_pass"] = df["thermal_n_frames_ok"] >= 3
    return df


def build(roots: list[Path], out: Path, workers: int = 4, plants: list[str] | None = None) -> pd.DataFrame:
    cache_dir = out.parent / (out.stem + "_plants")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for root in roots:  # first root wins for a plant found twice
        for p in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
            if p.is_dir() and "_" in p.name and p.name not in dirs and (plants is None or p.name in plants):
                dirs[p.name] = p
    todo = [p for p in dirs.values() if not (cache_dir / f"{p.name}.parquet").exists()]
    print(f"{len(dirs)} plants, {len(todo)} to read")
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, path in enumerate(ex.map(_cached, todo, [cache_dir] * len(todo))):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(todo)} plants in {time.time() - t0:.0f} s")
    parts = [pd.read_parquet(cache_dir / f"{name}.parquet") for name in dirs]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = add_air(core.normalize_keys(df).sort_values(core.KEYS, ignore_index=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"{len(df)} runs, {df[['bed', 'plant']].drop_duplicates().shape[0]} plants -> {out} ({time.time() - t0:.0f} s)"
    )
    return df


def _rgb_plant(pid: str, data_dir: Path, cache_dir: Path) -> Path:
    from uiwue.rgb import run as rgb_run

    out = cache_dir / f"{pid}.parquet"
    if not out.exists():
        df = rgb_run.extract(data_dir, [pid], jobs=1)
        cols = [c for c in (*core.KEYS, "rgb_green_cover", "rgb_qc_pass") if c in df]
        tmp = out.with_suffix(".tmp")
        df[cols].to_parquet(tmp, index=False)
        tmp.replace(out)
    return out


def rgb_size(data_dir: Path, out: Path, plants: list[str] | None = None, workers: int = 4) -> pd.DataFrame:
    """Green cover (uiwue.rgb: share of centre-crop pixels classed as vegetation, frame 7) for every
    plant with RGB: the Master Plan's second size baseline, for plants without depth_projected_area."""
    from uiwue.rgb import run as rgb_run

    cache_dir = out.parent / (out.stem + "_plants")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pids = sorted(set(rgb_run.list_runs(Path(data_dir), plants)[["bed", "plant"]].astype(str).agg("_".join, axis=1)))
    todo = [p for p in pids if not (cache_dir / f"{p}.parquet").exists()]
    print(f"{len(pids)} plants with RGB, {len(todo)} to read")
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(_rgb_plant, todo, [data_dir] * len(todo), [cache_dir] * len(todo))):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(todo)} plants in {time.time() - t0:.0f} s")
    parts = [pd.read_parquet(cache_dir / f"{p}.parquet") for p in pids]
    df = core.normalize_keys(pd.concat([p for p in parts if len(p)], ignore_index=True))
    df.to_parquet(out, index=False)
    print(f"{len(df)} runs, {len(pids)} plants -> {out} ({time.time() - t0:.0f} s)")
    return df


def load(path: Path) -> pd.DataFrame:
    return core.normalize_keys(pd.read_parquet(path))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", type=Path, nargs="+", default=[core.DATA / "thermal"])
    ap.add_argument("--out", type=Path, default=core.REPO / "cache" / "fusion_v2" / "thermal_features.parquet")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plants", nargs="*")
    ap.add_argument("--rgb-size", action="store_true", help="instead: green cover from the RGB zips in UIWUE_DATA")
    a = ap.parse_args(argv)
    if a.rgb_size:
        out = a.out if a.out.name != "thermal_features.parquet" else a.out.with_name("rgb_size.parquet")
        rgb_size(core.DATA, out, a.plants, a.workers)
        return
    build(a.roots, a.out, a.workers, a.plants)


if __name__ == "__main__":
    main()
