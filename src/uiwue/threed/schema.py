"""Shared contract for Challenge 3 (3D phenotyping): file paths, table columns, cloud format.

The pipeline (``uiwue.threed.run``) writes these files; the statistics
(``uiwue.threed.stats`` / ``experiments``) and the Dash pages (``app/pages/threed_*.py``) read
them. Change this module only by agreement, because all three depend on it. Plan:
plan/Kinjalk-Work-Plan.md, sections 4.8 and 6.

``synthetic_runs()`` and ``synthetic_cloud()`` make tables and clouds that follow this contract,
so the statistics and the app can be built and tested before the real batch has run.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths (relative to repo root)
DATA_DIR = Path("data")
DERIVED = DATA_DIR / "derived" / "threed"
TRAITS_FRAMES = DERIVED / "traits_frames.parquet"  # one row per plant, run and frame
TRAITS_RUNS = DERIVED / "traits_runs.parquet"  # one row per plant and run (the shared table)
TRAITS_PARITY = DERIVED / "traits_parity.parquet"  # paper-parity height (E0a), one row per run
CLOUDS_DIR = DERIVED / "clouds"  # <bed>_<plant>/<YYYY-MM-DD>_<HH>.npz, one cached cloud per run
RUN_LOG = DERIVED / "run_log.csv"  # every run attempted: keys, status, error text
RESULTS_DIR = DERIVED / "results"  # experiment tables (CSV) and figures (PNG)

# ---------------------------------------------------------------- experiment calendar
CUTOFF = date(2025, 9, 3)  # irrigation off in W2, W3 (last watering 11:00)
REWATER = date(2025, 9, 15)  # irrigation on again
BASELINE_DATES = (date(2025, 9, 1), date(2025, 9, 2))  # each plant's own baseline
TREATMENT_WINDOW = (CUTOFF, date(2025, 9, 14))
SLOTS = (6, 9, 12, 15, 18)  # scheduled run hours

BED_TREATMENT = {"W1": "control", "W2": "deficit", "W3": "deficit"}

# ---------------------------------------------------------------- columns
KEYS = ["bed", "plant", "date", "hour"]  # every challenge joins on these

TRAITS = {  # name -> unit (definitions: plan section 4.6)
    "height_p50": "mm",
    "height_p75": "mm",
    "height_p95": "mm",
    "projected_area": "cm2",
    "leaf_area_est": "cm2",
    "hull_volume": "cm3",
    "canopy_radius": "mm",
    "inclination_mean": "deg",
    "inclination_sd": "deg",
    "droop_index": "ratio",
    "compactness": "ratio",
    "n_points": "count",
}

QC_FLAGS = {  # name -> blocking (a blocking flag sets qc_pass = False)
    "qc_no_plant": True,
    "qc_few_points": True,
    "qc_high_spread": True,
    "qc_overlap": True,
    "qc_not_centred": False,
    "qc_dark": False,
    "qc_ref_carried": False,
    "qc_no_pose": False,
}

RUN_META = {  # column -> pandas dtype
    "bed": "string",
    "plant": "string",
    "date": "object",  # datetime.date
    "hour": "int64",  # folder hour; re-runs keep their own hour
    "slot": "int64",  # scheduled slot the run stands for (nearest of SLOTS)
    "timestamp_utc": "datetime64[ns, UTC]",  # median frame time
    "treatment": "string",
    "days_since_cutoff": "int64",
    "n_frames_ok": "int64",
}


def run_columns() -> list[str]:
    """Required columns of traits_runs.parquet, in order. Extra columns are allowed."""
    cols = list(RUN_META)
    for t in TRAITS:
        cols += [t, f"{t}_mad"]
    return cols + list(QC_FLAGS) + ["qc_pass"]


FRAME_COLUMNS = KEYS + ["frame"] + list(TRAITS) + ["ok"]  # traits_frames.parquet (extras allowed)

PARITY_COLUMNS = KEYS + ["height_parity_mm"]  # 0.60 m - 25th pct depth of ExG>5 points, frame 7

# ---------------------------------------------------------------- cached cloud (.npz)
# xyz     float32 (N, 3)  metres; plant-centred frame: x, y on the reference plane with the plant
#                         centre at (0, 0); z = height above the reference plane
# rgb     uint8   (N, 3)  colour (0 where no RGB sample)
# height  float32 (N,)    = xyz[:, 2], kept separately for colouring
# plant   bool    (N,)    True for points of the isolated centre plant
# temp    float32 (N,)    optional, leaf temperature in deg C, NaN where absent
# frame   int             which frame the cloud comes from
CLOUD_KEYS = ("xyz", "rgb", "height", "plant")


def cloud_path(bed: str, plant: str, d: date, hour: int) -> Path:
    return CLOUDS_DIR / f"{bed}_{plant}" / f"{d.isoformat()}_{hour:02d}.npz"


# ---------------------------------------------------------------- helpers
def folder_date(name: str) -> date:
    """'2025_09_08' -> date(2025, 9, 8)."""
    return datetime.strptime(name, "%Y_%m_%d").date()


def days_since_cutoff(d: date) -> int:
    return (d - CUTOFF).days


def slot_of(hour: int) -> int:
    return min(SLOTS, key=lambda s: abs(s - hour))


def validate_runs(df: pd.DataFrame) -> None:
    """Raise ValueError if ``df`` misses contract columns or has duplicate keys."""
    missing = [c for c in run_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"traits_runs is missing columns: {missing}")
    dup = df.duplicated(KEYS)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (bed, plant, date, hour) rows")


# ---------------------------------------------------------------- synthetic data for development
def synthetic_runs(
    plants_per_bed: int = 5,
    beds: tuple[str, ...] = ("W1", "W2"),
    start: date = date(2025, 8, 20),
    end: date = date(2025, 9, 19),
    seed: int = 0,
) -> pd.DataFrame:
    """A traits_runs table with plausible growth, a midday dip, and a deficit effect after the
    cutoff in deficit beds. For development and tests only."""
    rng = np.random.default_rng(seed)
    grid = ["A2", "A8", "J5", "R1", "R7", "B3", "G2", "I6", "Q6", "K6"][:plants_per_bed]
    rows = []
    for bed in beds:
        deficit = BED_TREATMENT[bed] == "deficit"
        for p in grid:
            size = rng.uniform(0.8, 1.2)
            d = start
            while d <= end:
                day = (d - start).days
                ds = days_since_cutoff(d)
                stress = max(0, min(ds, (REWATER - CUTOFF).days)) if deficit else 0
                for hour in SLOTS:
                    growth = size * (40 + 4.3 * day) - 13.0 * stress * deficit
                    dip = (-3.0 - 1.5 * stress * deficit) if hour in (12, 15) else 0.0
                    h75 = max(5.0, growth + dip + rng.normal(0, 3))
                    area = max(
                        1.0, size * (20 + 9 * day) - 25 * stress * deficit + rng.normal(0, 8)
                    )
                    vals = {
                        "height_p50": 0.7 * h75,
                        "height_p75": h75,
                        "height_p95": 1.25 * h75 + 2 * stress * deficit,
                        "projected_area": area,
                        "leaf_area_est": 1.3 * area,
                        "hull_volume": area * h75 / 20,
                        "canopy_radius": 10 * np.sqrt(area),
                        "inclination_mean": 35 + 2.5 * stress * deficit + rng.normal(0, 3),
                        "inclination_sd": 18 + rng.normal(0, 2),
                        "droop_index": 0.7 - 0.02 * stress * deficit + rng.normal(0, 0.02),
                        "compactness": 0.6 + rng.normal(0, 0.03),
                        "n_points": float(int(area * 12)),
                    }
                    ts = datetime(d.year, d.month, d.day, hour - 2, 0, tzinfo=UTC)
                    row = {
                        "bed": bed,
                        "plant": p,
                        "date": d,
                        "hour": hour,
                        "slot": hour,
                        "timestamp_utc": pd.Timestamp(
                            ts + timedelta(minutes=int(rng.integers(0, 40)))
                        ),
                        "treatment": BED_TREATMENT[bed],
                        "days_since_cutoff": ds,
                        "n_frames_ok": int(rng.integers(3, 6)),
                    }
                    for t, v in vals.items():
                        row[t] = float(v)
                        row[f"{t}_mad"] = abs(float(v)) * rng.uniform(0.01, 0.06)
                    for f in QC_FLAGS:
                        row[f] = False
                    row["qc_dark"] = hour == 6
                    row["qc_overlap"] = bool(day > 25 and rng.random() < 0.3)
                    row["qc_pass"] = not row["qc_overlap"]
                    rows.append(row)
                d += timedelta(days=1)
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    for c, t in RUN_META.items():
        if t in ("string", "int64"):
            df[c] = df[c].astype(t)
    return df


def synthetic_cloud(n: int = 20_000, height_mm: float = 120.0, seed: int = 0) -> dict:
    """A cloud dict in the CLOUD format: a soil disc plus a rosette of tilted leaf ellipses."""
    rng = np.random.default_rng(seed)
    soil = np.column_stack(
        [
            rng.uniform(-0.2, 0.2, n // 2),
            rng.uniform(-0.15, 0.15, n // 2),
            rng.normal(0, 0.002, n // 2),
        ]
    )
    k = n - n // 2
    ang = rng.uniform(0, 2 * np.pi, k)
    r = rng.uniform(0, 0.12, k)
    z = height_mm / 1000 * (1 - (r / 0.12) ** 2) + rng.normal(0, 0.003, k)
    leaf = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.clip(z, 0.005, None)])
    xyz = np.vstack([soil, leaf]).astype(np.float32)
    plant = np.r_[np.zeros(len(soil), bool), np.ones(k, bool)]
    rgb = np.where(plant[:, None], [70, 140, 60], [60, 45, 35]).astype(np.uint8)
    return {"xyz": xyz, "rgb": rgb, "height": xyz[:, 2].copy(), "plant": plant, "frame": 7}


# ---------------------------------------------------------------- experiment outputs (CSV)
# Written by uiwue.threed.experiments into RESULTS_DIR, read by the Compare page. Extra columns
# are allowed; these are the minimum.
RESULT_FILES = {
    "e0a_parity.csv": ["quantity", "ours", "paper", "rel_diff", "pass"],
    "e0b_qc_agreement.csv": ["flag", "ada_label", "precision", "recall", "n_labelled"],
    "e1_noise.csv": ["trait", "noise", "signal", "snr", "reliable"],
    "e2_bed_baseline.csv": ["trait", "day_bed_coef", "ci_low", "ci_high", "p_value", "warning"],
    "e3_onset.csv": [
        "trait",
        "slope_interaction",
        "ci_low",
        "ci_high",
        "p_value",
        "onset_day",
        "lead_vs_symptom_days",
    ],
    "e3_daily_tests.csv": ["trait", "date", "p_raw", "p_holm", "significant"],
    "e4_diurnal.csv": ["bed", "plant", "date", "trait", "midday_depression", "overnight_recovery"],
    "e5_turgor.csv": ["plant", "trait", "lag_days", "rho", "p_value", "n"],
}
