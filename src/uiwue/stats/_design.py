"""Experiment design constants and helpers shared by the statistics.

These mirror Master Plan 6.1-6.5. When `uiwue.core` exists, its versions of
`add_design` and `load_catalogue` are used instead, so the two cannot drift.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["bed", "plant", "date", "hour"]
SLOTS = (6, 9, 12, 15, 18)

CUTOFF = pd.Timestamp("2025-09-03")  # day 0, last watering 11:00
REWATER = pd.Timestamp("2025-09-15")  # 18:35
BASELINE = (pd.Timestamp("2025-08-30"), pd.Timestamp("2025-09-02"))
PLACEBO = (pd.Timestamp("2025-08-24"), pd.Timestamp("2025-08-29"))
STRESS_DAYS = (0, 12)

CONTROL_BEDS = ("W1",)
DEFICIT_BEDS = ("W2", "W3")

# Rule 6.7.5: tuning on W2 with these controls, confirmation on W3 with these.
TUNE_CONTROLS = ("A2", "J5", "R7", "B3")
CONFIRM_CONTROLS = ("A8", "R1", "Q6")

# Bookkeeping suffixes that are never traits (6.2).
_NOT_TRAIT = ("_mad", "capture_hour", "timestamp_utc", "n_frames_ok")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def derived_dir() -> Path:
    """`derived/` in the repo, or `$UIWUE_DERIVED` if set (used by tests)."""
    env = os.environ.get("UIWUE_DERIVED")
    return Path(env) if env else repo_root() / "derived"


def prefix_of(column: str) -> str:
    """Module prefix of a column, e.g. 'thermal_leaf_air_dt' -> 'thermal'."""
    return column.split("_", 1)[0]


def is_trait_column(column: str) -> bool:
    if column in KEYS or "_" not in column:
        return False
    if "qc_" in column:
        return False
    return not any(column.endswith(s) for s in _NOT_TRAIT)


def normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    """`date` as midnight Timestamps, `hour` as int, `bed`/`plant` as str."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["hour"] = df["hour"].astype(int)
    df["bed"] = df["bed"].astype(str)
    df["plant"] = df["plant"].astype(str)
    return df


def add_design(df: pd.DataFrame) -> pd.DataFrame:
    """Add `treatment`, `days_since_cutoff`, `phase` from the keys (6.2)."""
    try:  # the platform owns this; prefer it when it exists
        from uiwue.core import add_design as core_add_design  # type: ignore
    except ImportError:
        core_add_design = None
    if core_add_design is not None:
        return normalise_keys(core_add_design(df))

    df = normalise_keys(df)
    df["treatment"] = np.where(df["bed"].isin(CONTROL_BEDS), "control", "deficit")
    df["days_since_cutoff"] = (df["date"] - CUTOFF).dt.days
    phase = np.select(
        [df["date"] < BASELINE[0], df["date"] < CUTOFF, df["date"] <= REWATER],
        ["early", "baseline", "stress"],
        default="recovery",
    )
    df["phase"] = phase
    return df


def qc_pass(df: pd.DataFrame, column: str) -> pd.Series:
    """True where the column's module says the run passed QC (missing flag = pass)."""
    flag = f"{prefix_of(column)}_qc_pass"
    if flag not in df.columns:
        return pd.Series(True, index=df.index)
    return df[flag].fillna(True).astype(bool)


@lru_cache(maxsize=4)
def _catalogue_cached(path: str) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(path).glob("*/catalogue.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame(columns=["column", "label", "unit", "direction", "description"])
    cat = pd.concat(frames, ignore_index=True).drop_duplicates("column", keep="last")
    cat["direction"] = pd.to_numeric(cat["direction"], errors="coerce").fillna(0).astype(int)
    return cat.reset_index(drop=True)


def load_catalogue() -> pd.DataFrame:
    """All modules' catalogue rows: column, label, unit, direction, description."""
    try:
        from uiwue.core import load_catalogue as core_load_catalogue  # type: ignore
    except ImportError:
        core_load_catalogue = None
    if core_load_catalogue is not None:
        return core_load_catalogue()
    return _catalogue_cached(str(derived_dir()))


def direction_of(column: str, catalogue: pd.DataFrame | None = None) -> int:
    """Catalogue direction (+1 rises under deficit, -1 falls). Unknown or 0 -> +1."""
    cat = load_catalogue() if catalogue is None else catalogue
    hit = cat.loc[cat["column"] == column, "direction"]
    if hit.empty or int(hit.iloc[0]) == 0:
        return 1
    return int(np.sign(hit.iloc[0]))
