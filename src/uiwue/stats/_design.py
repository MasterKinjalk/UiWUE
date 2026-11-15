"""Design constants and helpers for the statistics, on top of `uiwue.core` (the contract).

Dates are re-exported as Timestamps because the statistics compare them with
the normalised `date` column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uiwue import core

KEYS = core.KEYS
SLOTS = core.SLOTS

CUTOFF = pd.Timestamp(core.CUTOFF)  # day 0, last watering 11:00
REWATER = pd.Timestamp(core.REWATER)  # 18:35, after the 18 h run; day 12 is still stress
BASELINE = tuple(pd.Timestamp(d) for d in core.BASELINE)
PLACEBO = tuple(pd.Timestamp(d) for d in core.PLACEBO)
STRESS_DAYS = (0, 12)

CONTROL_BEDS = ("W1",)
DEFICIT_BEDS = ("W2", "W3")

# Rule 6.7.5: tuning on W2 with these controls, confirmation on W3 with these.
TUNE_CONTROLS = core.DEV_CONTROLS
CONFIRM_CONTROLS = core.CONFIRM_CONTROLS

# Bookkeeping suffixes that are never traits (6.2).
_NOT_TRAIT = ("_mad", "capture_hour", "timestamp_utc", "n_frames_ok")


def derived_dir() -> Path:
    """`derived/`, read from core at call time so tests can point it elsewhere."""
    return Path(core.DERIVED)


def prefix_of(column: str) -> str:
    """Module prefix of a column, e.g. 'thermal_leaf_air_dt' -> 'thermal'."""
    return column.split("_", 1)[0]


def is_trait_column(column: str) -> bool:
    if column in KEYS or "_" not in column or "qc_" in column:
        return False
    return not any(column.endswith(s) for s in _NOT_TRAIT)


def normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    return core.normalize_keys(df)


def add_design(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised keys plus `treatment`, `days_since_cutoff`, `phase` (from core)."""
    return core.add_design(core.normalize_keys(df))


def qc_pass(df: pd.DataFrame, column: str) -> pd.Series:
    """True where the column's module says the run passed QC (missing flag = pass)."""
    flag = f"{prefix_of(column)}_qc_pass"
    if flag not in df.columns:
        return pd.Series(True, index=df.index)
    return df[flag].fillna(True).astype(bool)


def load_catalogue() -> pd.DataFrame:
    """All modules' catalogue rows: column, label, unit, direction, description, module."""
    cat = core.catalogue().drop_duplicates("column", keep="last").copy()
    cat["direction"] = pd.to_numeric(cat["direction"], errors="coerce").fillna(0).astype(int)
    return cat.reset_index(drop=True)


def clear_caches() -> None:
    """Forget cached tables and catalogues (after writing new ones, and in tests)."""
    core.catalogue.cache_clear()
    core.load_traits.cache_clear()


def direction_of(column: str, catalogue: pd.DataFrame | None = None) -> int:
    """Catalogue direction (+1 rises under deficit, -1 falls). Unknown or 0 -> +1."""
    cat = load_catalogue() if catalogue is None else catalogue
    hit = cat.loc[cat["column"] == column, "direction"]
    if hit.empty or int(hit.iloc[0]) == 0:
        return 1
    return int(np.sign(hit.iloc[0]))
