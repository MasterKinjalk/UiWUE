"""Load the merged run-level table and the catalogue, with or without `uiwue.core`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uiwue.stats._design import (
    KEYS,
    add_design,
    derived_dir,
    is_trait_column,
    load_catalogue,
    normalise_keys,
    prefix_of,
)

SENSORS = {"rgb": "RGB", "depth": "Depth", "thermal": "Thermal", "hsi": "Hyperspectral"}
# Ordinal cost tiers (webcam < ToF depth < FLIR thermal < Pika L hyperspectral).
# Doubling weights make a combination's cost dominated by its dearest sensor.
COST_TIER = {"rgb": 1, "depth": 2, "thermal": 3, "hsi": 4}
SIZE_BASELINE = ("depth_projected_area", "rgb_green_cover")  # first available wins (6.7.3)


def cost_rank(sensors) -> int:
    return int(sum(2 ** (COST_TIER[s] - 1) for s in sensors))


def _read_tables(root: Path) -> list[pd.DataFrame]:
    frames = []
    for f in sorted(root.glob("*/traits_runs.parquet")):
        frames.append(normalise_keys(pd.read_parquet(f)))
    for extra in ("env/env_runs.parquet", "truth/turgor_runs.parquet"):
        f = root / extra
        if f.exists():
            frames.append(normalise_keys(pd.read_parquet(f)))
    return frames


def load_merged() -> pd.DataFrame:
    """One wide table on the four keys, with design columns. Empty if nothing exists."""
    try:
        from uiwue.core import load_traits  # type: ignore
    except ImportError:
        load_traits = None
    if load_traits is not None:
        return add_design(load_traits())

    frames = _read_tables(derived_dir())
    if not frames:
        return pd.DataFrame(columns=KEYS)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=KEYS, how="outer")
    return add_design(out)


def trait_columns(df: pd.DataFrame, catalogue: pd.DataFrame | None = None,
                  prefixes=tuple(SENSORS)) -> list[str]:
    """Stress traits to analyse: in the catalogue with direction != 0 when a
    catalogue row exists, numeric, and from a sensor module."""
    cat = load_catalogue() if catalogue is None else catalogue
    known = dict(zip(cat["column"], cat["direction"]))
    cols = []
    for c in df.columns:
        if not is_trait_column(c) or prefix_of(c) not in prefixes:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if c in known and known[c] == 0:
            continue
        cols.append(c)
    return cols


def label_of(column: str, catalogue: pd.DataFrame | None = None) -> str:
    cat = load_catalogue() if catalogue is None else catalogue
    hit = cat.loc[cat["column"] == column, "label"]
    return str(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else column


def size_baseline_column(df: pd.DataFrame) -> str | None:
    for c in SIZE_BASELINE:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def load_result(name: str) -> pd.DataFrame | None:
    """A table from derived/detect/, or None if it does not exist yet."""
    f = derived_dir() / "detect" / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None
