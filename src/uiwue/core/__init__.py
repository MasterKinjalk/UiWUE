"""Shared contract (plan/Master-Plan.md section 6): keys, dates, plant sets,
reading and writing derived tables, and the merged trait table.

Every module writes run-level tables keyed by bed, plant, date, hour (the
scheduled slot) to derived/<module>/, plus a catalogue.csv describing its
columns. Pages and analyses read them back through load_traits().
"""

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
# The tables the app and the analyses read. UIWUE_DERIVED moves it: a run on someone's own data
# (uiwue.pipeline) writes to its own workspace and never to the repo's derived/.
DERIVED = Path(os.environ.get("UIWUE_DERIVED") or REPO / "derived")
DATA = Path(os.environ.get("UIWUE_DATA", REPO / "data"))  # raw ADA data, local only

KEYS = ["bed", "plant", "date", "hour"]
SLOTS = (6, 9, 12, 15, 18)
PREFIXES = ("thermal_", "depth_", "rgb_", "hsi_", "env_", "truth_", "detect_", "diurnal_")
DESIGN = ("treatment", "days_since_cutoff", "phase")  # added on load, never stored

CUTOFF = date(2025, 9, 3)  # last watering of W2/W3 at 11:00; day 0
REWATER = date(2025, 9, 15)  # W2/W3 re-watered 18:35, after the 18 h run; day 12 is still stress
DATES = (date(2025, 8, 24), date(2025, 9, 22))
BASELINE = (date(2025, 8, 30), date(2025, 9, 2))
PLACEBO = (date(2025, 8, 24), date(2025, 8, 29))

# Tuned settings are picked on W2 with DEV_CONTROLS, then checked unchanged on W3 (6.7 rule 5)
DEV_CONTROLS = ("W1_A2", "W1_J5", "W1_R7", "W1_B3")
CONFIRM_CONTROLS = ("W1_A8", "W1_R1", "W1_Q6")

# Columns Sam's truth/plants table is expected to carry (one row per bed, plant)
TRUTH_CLIP_ONSET = "truth_clip_onset_date"
TRUTH_FIRST_SYMPTOM = "truth_first_symptom_date"

CATALOGUE_COLUMNS = ["column", "label", "unit", "direction", "description"]


def all_plants():
    """The 72 grid positions per bed: rows A-R, checkerboard columns (A2 A4 .., B1 B3 ..)."""
    rows = "ABCDEFGHIJKLMNOPQR"
    return [f"{r}{c}" for i, r in enumerate(rows) for c in range(2 - i % 2, 9, 2)]


def slot_for_hour(hour):
    """Scheduled slot a run-folder hour stands for (nearest slot at most 1 h away), else None."""
    d = np.abs(np.array(SLOTS) - int(hour))
    return SLOTS[d.argmin()] if d.min() <= 1 else None


def normalize_keys(df):
    """Coerce the shared keys to one dtype each: str, str, datetime64 (midnight), int."""
    df = df.copy()
    for k in ("bed", "plant"):
        if k in df:
            df[k] = df[k].astype(str)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    if "hour" in df:
        df["hour"] = df["hour"].astype(int)
    return df


def add_design(df):
    """Treatment, days since cutoff and phase, derived from the keys so no table can disagree."""
    df = df.copy()
    df["treatment"] = np.where(df.bed == "W1", "control", "deficit")
    df["days_since_cutoff"] = (df.date - pd.Timestamp(CUTOFF)).dt.days
    df["phase"] = pd.cut(
        df.days_since_cutoff, [-1000, -1, 12, 1000], labels=["before cutoff", "stress", "re-watered"]
    ).astype(str)
    return df


def in_window(dates, window):
    return pd.to_datetime(dates).between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))


def table_path(module, name):
    return DERIVED / module / f"{name}.parquet"


def write_table(df, module, name, catalogue=None):
    """Write derived/<module>/<name>.parquet (keys as plain types) and optionally its catalogue rows."""
    out = normalize_keys(df)
    if "date" in out:
        out["date"] = out["date"].dt.date
    path = table_path(module, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    if catalogue is not None:
        pd.DataFrame(catalogue, columns=CATALOGUE_COLUMNS).to_csv(path.parent / "catalogue.csv", index=False)
    return path


def read_table(module, name):
    path = table_path(module, name)
    return normalize_keys(pd.read_parquet(path)) if path.exists() else None


@lru_cache
def catalogue():
    """Every module's catalogue.csv, with a `module` column."""
    parts = [pd.read_csv(p).assign(module=p.parent.name) for p in sorted(DERIVED.glob("*/catalogue.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CATALOGUE_COLUMNS + ["module"])


@lru_cache
def load_traits():
    """One wide table on the four keys, with design columns, or None if nothing is derived yet.

    traits_runs tables define which runs exist (outer join); every other *_runs
    table (environment, scores, turgor) is attached to those runs (left join).
    """
    paths = sorted(DERIVED.glob("*/*_runs.parquet"))
    observed = [normalize_keys(pd.read_parquet(p)) for p in paths if p.stem == "traits_runs"]
    attached = [normalize_keys(pd.read_parquet(p)) for p in paths if p.stem != "traits_runs"]
    if not observed:
        return None
    df = observed[0]
    for t in observed[1:]:
        df = df.merge(t, on=KEYS, how="outer")
    for t in attached:
        df = df.merge(t, on=KEYS, how="left")
    return add_design(df).sort_values(KEYS, ignore_index=True)
