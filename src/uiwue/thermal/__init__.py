"""Leaf temperature from the thermal camera: derived/thermal/traits_runs.parquet.

    python -m uiwue.thermal [--data-dir DIR]

Reads <data>/thermal/<bed>_<plant>/<YYYY_MM_DD>/<HH>/thermal_00N.tif (FLIR A70,
640 x 480, °C, five frames per run) and writes one row per plant and scheduled
run. Leaf temperature follows the dataset authors' rule: the median of the
coolest half of the pixels, then the median over the run's frames.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from uiwue.core import DATA, DATES, KEYS, slot_for_hour, write_table
from uiwue.env import air, at_times

MIN_FRAMES = 3
MIN_CONTROLS = 3  # control plants needed in a run for thermal_leaf_ref_dt

CATALOGUE = [
    (
        "thermal_leaf_t",
        "Leaf temperature",
        "°C",
        1,
        "median of the coolest half of the pixels (authors' rule), median over frames",
    ),
    (
        "thermal_leaf_t_p10",
        "Leaf temperature, coolest 10 %",
        "°C",
        1,
        "10th percentile of the image; a stricter leaf estimate",
    ),
    (
        "thermal_bg_t",
        "Background temperature",
        "°C",
        0,
        "75th percentile of the image: mostly pots and soil, which also warm as they dry",
    ),
    (
        "thermal_leaf_t_r160",
        "Leaf temperature at 160 x 120",
        "°C",
        1,
        "same rule on the image averaged to 160 x 120 pixels (cheaper camera)",
    ),
    (
        "thermal_leaf_t_r80",
        "Leaf temperature at 80 x 60",
        "°C",
        1,
        "same rule on the image averaged to 80 x 60 pixels (cheaper camera)",
    ),
    (
        "thermal_leaf_air_dt",
        "Leaf minus air temperature",
        "°C",
        1,
        "leaf temperature minus the W1-end air sensor at the capture time",
    ),
    (
        "thermal_leaf_ref_dt",
        "Leaf minus control-bed leaf",
        "°C",
        1,
        "leaf temperature minus the median leaf temperature of the control-bed plants in the same run (other controls for a control plant); cancels weather",
    ),
]


def leaf_t(img):
    """Median of the coolest half of the pixels (ADA analysis/leaf_air_temp rule)."""
    v = img.ravel()
    return float(np.median(v[v <= np.percentile(v, 50)]))


def block_mean(img, f):
    """Average f x f pixel blocks: the image a camera with 1/f the resolution would see."""
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def frame_stats(path):
    img = tifffile.imread(path).astype(np.float32)
    return [
        leaf_t(img),
        np.percentile(img, 10),
        np.percentile(img, 75),
        leaf_t(block_mean(img, 4)),
        leaf_t(block_mean(img, 8)),
    ]


def run_row(run_dir, frames):
    s = np.array([frame_stats(f) for f in frames])
    leaf, p10, bg, r160, r80 = np.median(s, axis=0)
    csv = run_dir / "frames_thermal.csv"
    if csv.exists():
        ts = pd.Timestamp(int(pd.read_csv(csv).timestamp.median()), unit="ns", tz="UTC")
    else:  # no frame log: the folder's start hour plus ten minutes
        ts = pd.Timestamp(f"{run_dir.parent.name.replace('_', '-')} {run_dir.name}:10", tz="Europe/Berlin").tz_convert(
            "UTC"
        )
    return dict(
        thermal_leaf_t=leaf,
        thermal_leaf_t_mad=float(np.median(np.abs(s[:, 0] - leaf))),
        thermal_leaf_t_p10=p10,
        thermal_bg_t=bg,
        thermal_leaf_t_r160=r160,
        thermal_leaf_t_r80=r80,
        thermal_capture_hour=int(run_dir.name),
        thermal_timestamp_utc=ts,
        thermal_n_frames_ok=len(frames),
    )


def plant_rows(plant_dir):
    """One row per scheduled slot: the scheduled folder if it has frames, else the re-run with most frames."""
    bed, plant = plant_dir.name.split("_")
    rows = []
    for date_dir in sorted(d for d in plant_dir.iterdir() if d.is_dir()):
        date = pd.Timestamp(date_dir.name.replace("_", "-"))
        if not DATES[0] <= date.date() <= DATES[1]:
            continue
        by_slot = {}
        for run_dir in (d for d in date_dir.iterdir() if d.is_dir()):
            slot, frames = slot_for_hour(int(run_dir.name)), sorted(run_dir.glob("thermal_*.tif"))
            if slot and frames:
                rank = (int(run_dir.name) == slot, len(frames))
                if slot not in by_slot or rank > by_slot[slot][0]:
                    by_slot[slot] = (rank, run_dir, frames)
        for slot, (_, run_dir, frames) in sorted(by_slot.items()):
            rows.append(dict(bed=bed, plant=plant, date=date, hour=slot) | run_row(run_dir, frames))
    return rows


def control_reference(df, col):
    """Median of `col` over control-bed (W1) plants in the same run; a control plant gets the other controls."""
    ref = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["date", "hour"]):
        ctrl = g.loc[g.bed == "W1", col].dropna()
        if len(ctrl) >= MIN_CONTROLS:
            ref[g.index[g.bed != "W1"]] = ctrl.median()
            for i in ctrl.index:
                ref[i] = ctrl.drop(i).median()
    return ref


def build(data_dir=DATA):
    plants = sorted(p for p in (Path(data_dir) / "thermal").iterdir() if p.is_dir())
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(plant_rows, plants) for r in rows])
    df["thermal_leaf_air_dt"] = df.thermal_leaf_t - at_times(air(data_dir).air_t, df.thermal_timestamp_utc)
    df["thermal_qc_few_frames"] = df.thermal_n_frames_ok < MIN_FRAMES
    df["thermal_qc_pass"] = ~df.thermal_qc_few_frames
    passed = df.assign(leaf=df.thermal_leaf_t.where(df.thermal_qc_pass))
    df["thermal_leaf_ref_dt"] = df.thermal_leaf_t - control_reference(passed, "leaf")
    return df.sort_values(KEYS, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()
    df = build(args.data_dir)
    path = write_table(df, "thermal", "traits_runs", CATALOGUE)
    print(f"{len(df)} runs, {df[['bed', 'plant']].drop_duplicates().shape[0]} plants -> {path}")
    print(df.groupby("bed").plant.nunique().to_dict(), "| missing air:", int(df.thermal_leaf_air_dt.isna().sum()))
