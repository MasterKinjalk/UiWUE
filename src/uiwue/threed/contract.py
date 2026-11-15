"""Publish the 3D trait table in the team contract (plan/Master-Plan.md, section 6).

    python -m uiwue.threed.contract            # writes derived/threed/traits_runs.parquet + catalogue.csv
    python -m uiwue.core.validate derived/threed/traits_runs.parquet

The pipeline (``uiwue.threed.run``) keeps its own working table in data/derived/threed/ with
folder hours and unprefixed columns. This module converts it:

- ``hour`` becomes the scheduled slot (nearest slot at most 1 h away). When two folders map to
  one slot, the scheduled folder wins if it has data, else the re-run with the most frames.
- Every column gets the ``depth_`` prefix; the folder hour is kept as ``depth_capture_hour``.
- Design columns (treatment, days since cutoff) are dropped: ``uiwue.core`` adds them on load.
- Dates before 2025-08-24 are out of the team's scope and dropped.
- ``catalogue.csv`` lists every trait with label, unit, stress direction and definition.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from uiwue.core import KEYS, slot_for_hour, write_table
from uiwue.threed import schema

MODULE = "threed"
PREFIX = "depth_"
FIRST_DATE = date(2025, 8, 24)  # plan/Master-Plan.md 6.5: earlier dates are out of scope

UNITS = {"mm": "mm", "cm2": "cm²", "cm3": "cm³", "deg": "°", "ratio": "ratio", "count": "count"}

# label, direction (+1 rises under deficit, -1 falls, 0 not a stress trait), definition.
# Directions follow the E3 interaction slopes (docs/threed/RESULTS.md): every trait is lower in
# W2 than W1 after the cutoff. inclination_sd is 0: below the noise floor in E1 (SNR 2.4).
TRAITS = {
    "height_p50": ("Canopy height, median", -1, "50th percentile of plant-point height above the soil plane"),
    "height_p75": ("Canopy height, p75", -1, "75th percentile of plant-point height above the soil plane"),
    "height_p95": ("Canopy height, p95", -1, "95th percentile of plant-point height above the soil plane"),
    "projected_area": ("Projected leaf area", -1, "top-view area of the plant points (ground area per pixel)"),
    "leaf_area_est": ("Leaf area (tilt-corrected)", -1, "projected area divided by cos(leaf inclination), capped 75°"),
    "hull_volume": ("Canopy volume", -1, "convex hull of plant points and their projection on the soil plane"),
    "canopy_radius": ("Canopy radius", -1, "95th percentile of horizontal distance from the plant centre"),
    "inclination_mean": (
        "Leaf inclination, mean",
        -1,
        "mean angle between surface normals and the soil normal (0° flat); rises in both beds from 09-03",
    ),
    "inclination_sd": ("Leaf inclination, spread", 0, "SD of leaf inclination; below the noise floor (E1)"),
    "droop_index": ("Droop index", -1, "height_p50 / height_p95; falls when leaf tips sink"),
    "compactness": ("Compactness", -1, "projected area / area of the 2D convex hull in top view"),
    "n_points": ("Plant point count", -1, "ToF points on the isolated plant (scales with visible leaf area)"),
}
CYL = {  # all vegetation inside the 8.5 cm plant cylinder, no clustering: steadier on a closed canopy
    "cyl_height_p50": ("Cylinder height, median", -1, "height_p50 of all vegetation in the plant cylinder"),
    "cyl_height_p75": ("Cylinder height, p75", -1, "height_p75 of all vegetation in the plant cylinder"),
    "cyl_height_p95": ("Cylinder height, p95", -1, "height_p95 of all vegetation in the plant cylinder"),
    "cyl_projected_area": ("Cylinder projected area", -1, "projected area of all vegetation in the plant cylinder"),
    "cyl_n_points": ("Cylinder point count", -1, "vegetation points in the plant cylinder"),
}
CYL_UNITS = {
    "cyl_height_p50": "mm",
    "cyl_height_p75": "mm",
    "cyl_height_p95": "mm",
    "cyl_projected_area": "cm²",
    "cyl_n_points": "count",
}
DIAGNOSTICS = {  # column -> (label, unit, definition); direction 0
    "soil_z_mm": ("Sensor to soil distance", "mm", "depth of the fitted soil plane below the camera"),
    "clipped_fraction": ("Overlap fraction", "ratio", "share of plant points at the 8.5 cm cylinder wall"),
    "centroid_offset_mm": ("Plant off-centre distance", "mm", "isolated plant centroid to expected centre"),
    "brightness": ("RGB brightness", "0-255", "mean RGB brightness of the frames used (dark below 10)"),
}
BOOKKEEPING = {
    "hour": "capture_hour",
    "timestamp_utc": "timestamp_utc",
    "n_frames_ok": "n_frames_ok",
    "n_frames_used": "n_frames_used",
    "frames_used": "frames_used",
    "ref_source": "ref_source",
}


def pick_slot_runs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (bed, plant, date, slot): the scheduled folder if it has data, else the re-run
    with the most frames."""
    df = df.copy()
    df["slot_"] = df["hour"].map(slot_for_hour)
    df = df[df["slot_"].notna()].copy()
    df["slot_"] = df["slot_"].astype(int)
    has_data = df["n_frames_ok"].fillna(0) > 0
    df["_rank_sched"] = ((df["hour"] == df["slot_"]) & has_data).astype(int)
    frames = df["n_frames_used"] if "n_frames_used" in df else df["n_frames_ok"]
    df["_rank_frames"] = frames.fillna(0)
    df = df.sort_values(["_rank_sched", "_rank_frames"], ascending=False, kind="stable")
    df = df.drop_duplicates(["bed", "plant", "date", "slot_"], keep="first")
    return df.drop(columns=["_rank_sched", "_rank_frames"])


def to_team_table(runs: pd.DataFrame) -> tuple[pd.DataFrame, list[list]]:
    """Pipeline table -> (team table keyed by bed, plant, date, slot hour, catalogue rows)."""
    runs = runs.copy()
    runs["date"] = pd.to_datetime(runs["date"]).dt.date
    runs = runs[runs["date"] >= FIRST_DATE]
    runs = pick_slot_runs(runs)

    out = runs[["bed", "plant", "date"]].copy()
    out["hour"] = runs["slot_"].astype(int)
    catalogue = []

    def add(col: str, values: pd.Series) -> None:
        out[PREFIX + col] = values

    for t, (label, direction, desc) in TRAITS.items():
        if t not in runs:
            continue
        add(t, runs[t].astype(float))
        if f"{t}_mad" in runs:
            add(f"{t}_mad", runs[f"{t}_mad"].astype(float))
        catalogue.append([PREFIX + t, label, UNITS[schema.TRAITS[t]], direction, desc])
    for t, (label, direction, desc) in CYL.items():
        if t not in runs:
            continue
        add(t, runs[t].astype(float))
        if f"{t}_mad" in runs:
            add(f"{t}_mad", runs[f"{t}_mad"].astype(float))
        catalogue.append([PREFIX + t, label, CYL_UNITS[t], direction, desc])
    for flag in [*schema.QC_FLAGS, "qc_pass"]:
        if flag in runs:
            add(flag, runs[flag].astype(bool))
    for src, dst in BOOKKEEPING.items():
        if src in runs:
            add(dst, runs[src])
    for c, (label, unit, desc) in DIAGNOSTICS.items():
        if c in runs:
            add(c, runs[c].astype(float))
            catalogue.append([PREFIX + c, label, unit, 0, desc])

    out = out.sort_values(KEYS, ignore_index=True)
    return out, catalogue


def from_team_table(table: pd.DataFrame) -> pd.DataFrame:
    """Team table -> pipeline columns, so the 3D pages run on a fresh clone without data/.

    Inverse of ``to_team_table`` for the columns it keeps: the prefix is dropped, the slot goes
    back to ``slot``, the capture hour to ``hour``, and the design columns are recomputed.
    """
    runs = table.rename(columns=lambda c: c.removeprefix(PREFIX))
    runs = runs.rename(columns={"hour": "slot", "capture_hour": "hour"})
    runs["treatment"] = runs["bed"].map(schema.BED_TREATMENT)
    runs["days_since_cutoff"] = (pd.to_datetime(runs["date"]) - pd.Timestamp(schema.CUTOFF)).dt.days
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(schema.TRAITS_RUNS), help="pipeline table to convert")
    args = p.parse_args()
    runs = pd.read_parquet(Path(args.source))
    table, catalogue = to_team_table(runs)
    path = write_table(table, MODULE, "traits_runs", catalogue=catalogue)
    n_plants = table.groupby(["bed", "plant"]).ngroups
    print(f"{len(table)} runs, {n_plants} plants, {table.date.min()}..{table.date.max()} -> {path}")


if __name__ == "__main__":
    main()
