"""The merged run-level table, the catalogue, sensors and their relative cost."""

from __future__ import annotations

import pandas as pd

from uiwue import core
from uiwue.stats._design import KEYS, derived_dir, is_trait_column, load_catalogue, prefix_of

SENSORS = {"rgb": "RGB", "depth": "Depth", "thermal": "Thermal", "hsi": "Hyperspectral"}

# Setups the benchmark compares. Reduced-resolution thermal (Yuming's _r160/_r80
# columns) stands in for a cheaper thermal camera, so it replaces full thermal.
SETUP_LABEL = {
    "rgb": "RGB",
    "depth": "Depth",
    "thermal_r80": "Thermal 80x60",
    "thermal_r160": "Thermal 160x120",
    "thermal": "Thermal",
    "hsi": "Hyperspectral",
}
# Relative cost, ordinal only (webcam < ToF depth < low-res thermal < FLIR A70 < Pika L).
# Not prices; a combination costs the sum of its parts.
SETUP_COST = {"rgb": 1, "depth": 2, "thermal_r80": 3, "thermal_r160": 4, "thermal": 5, "hsi": 12}
SIZE_BASELINE = ("depth_projected_area", "rgb_green_cover")  # first available wins (6.7.3)
# 3D traits left out of the analysis (docs/threed/APP.md, "What was removed"): no stress signal,
# saturated by the 85 mm crop cylinder, or redundant with a kept trait. Kept: leaf area, height p75,
# hull volume, and projected area as the size-only baseline.
DEPTH_EXCLUDED = (
    "depth_droop_index",
    "depth_inclination_mean",
    "depth_inclination_sd",
    "depth_canopy_radius",
    "depth_compactness",
    "depth_n_points",
    "depth_height_p50",
    "depth_height_p95",
)
DEPTH_EXCLUDED_PREFIX = "depth_cyl_"  # every cylinder variant


def excluded(column: str) -> bool:
    """True for a 3D trait the analysis leaves out."""
    return column in DEPTH_EXCLUDED or column.startswith(DEPTH_EXCLUDED_PREFIX)


def setup_of(column: str) -> str:
    """Benchmark setup a feature column belongs to."""
    p = prefix_of(column)
    if p == "thermal":
        for res in ("r80", "r160"):
            if column.endswith(f"_{res}"):
                return f"thermal_{res}"
    return p


def cost_rank(setups) -> int:
    return int(sum(SETUP_COST[s] for s in setups))


def load_merged() -> pd.DataFrame:
    """One wide table on the four keys, with design columns (core.load_traits). Empty if none."""
    df = core.load_traits()
    if df is None:
        return pd.DataFrame(columns=KEYS)
    return df.copy()


def trait_columns(df: pd.DataFrame, catalogue: pd.DataFrame | None = None, prefixes=tuple(SENSORS)) -> list[str]:
    """Stress traits to analyse: numeric, from a sensor module, and not catalogued as
    direction 0 (bookkeeping, background, exposure) or left out as a noisy 3D trait (DEPTH_EXCLUDED)."""
    cat = load_catalogue() if catalogue is None else catalogue
    known = dict(zip(cat["column"], cat["direction"], strict=False))
    cols = []
    for c in df.columns:
        if not is_trait_column(c) or prefix_of(c) not in prefixes:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if known.get(c, 1) == 0 or excluded(c):
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


def load_benchmark() -> pd.DataFrame | None:
    """Both tracks' setups in one table: the feature benchmark and, once trained, the fusion model."""
    parts = [t for t in (load_result("benchmark"), load_result("fusion_benchmark")) if t is not None and not t.empty]
    return pd.concat(parts, ignore_index=True) if parts else None


def load_result(name: str) -> pd.DataFrame | None:
    """A table from derived/detect/, or None if it does not exist yet."""
    f = derived_dir() / "detect" / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None
