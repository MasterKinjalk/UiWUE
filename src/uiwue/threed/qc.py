"""Per-run aggregation and QC flags (plan section 4.7, adapted to the 3 x 5 frame grid).

The 15 frames of a run are 15 robot positions, not repeat views. Only frames that see the
plant's whole cylinder footprint in both the depth and the RGB view are used (see
``run.select_frames``); for most plants these are frames 4, 7 and 10 (the centre column, the
plant at +-0.1 m along the row) and, where the robot reached it, the left column at the same
positions. ``<trait>`` is the median over the frames used and ``<trait>_mad`` the median absolute
deviation (unscaled) over the same frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue.threed import schema

MIN_FRAMES_OK = 2  # a run needs at least this many usable frames (plan: 8 of 15 repeat views)
FEW_POINTS = 300  # median n_points below this -> qc_few_points
HIGH_SPREAD_MM = 5.0  # height_p75_mad above max(this, HIGH_SPREAD_REL * height_p75) -> qc_high_spread
# The frames used are different viewpoints (up to 0.1 m apart), so occlusion alone makes p75
# differ by a few % between frames on a tall canopy; 5 mm absolute flags 27 % of the 09-01..09-14
# runs at 15 h. A run is flagged only if the spread is also above 5 % of the height.
HIGH_SPREAD_REL = 0.05
OVERLAP = 0.10  # median clipped_fraction above this -> qc_overlap


def mad(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))))


def _median(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def aggregate_run(frames: pd.DataFrame, meta: dict) -> dict:
    """One traits_runs row from the run's frame rows.

    ``frames`` has one row per processed frame with the trait columns and ``used``, ``ok``,
    ``no_plant``, ``clipped_fraction``, ``centroid_offset_mm``, ``brightness``. ``meta`` carries
    the run keys and run-level facts (``qc_ref_carried``, ``qc_no_pose``, ``timestamp_utc``,
    reference-plane columns...); it is copied into the row.
    """
    row = dict(meta)
    used = frames[frames["used"]] if len(frames) else frames
    good = used[used["ok"]] if len(used) else used
    row["frames_used"] = ",".join(str(int(k)) for k in used["frame"]) if len(used) else ""
    row["n_frames_used"] = len(used)
    row["n_frames_ok"] = len(good)
    for t in schema.TRAITS:
        row[t] = _median(good[t]) if len(good) else float("nan")
        row[f"{t}_mad"] = mad(good[t]) if len(good) > 1 else float("nan")
    for t in [c for c in frames.columns if c.startswith("cyl_")]:
        row[t] = _median(used[t]) if len(used) else float("nan")
        row[f"{t}_mad"] = mad(used[t]) if len(used) > 1 else float("nan")
    if "refined_frac" in frames.columns:  # refined depth (--depth-model): share of plant points
        row["refined_frac"] = _median(good["refined_frac"]) if len(good) else float("nan")
    row["clipped_fraction"] = _median(good["clipped_fraction"]) if len(good) else float("nan")
    row["outside_fraction"] = _median(good["outside_fraction"]) if len(good) else float("nan")
    row["centroid_offset_mm"] = _median(good["centroid_offset_mm"]) if len(good) else float("nan")
    row["brightness"] = _median(used["brightness"]) if len(used) else float("nan")

    no_plant = len(good) == 0 or (len(used) > 0 and used["no_plant"].mean() > 0.5)
    row["qc_no_plant"] = bool(no_plant)
    row["qc_few_points"] = bool(not no_plant and row["n_points"] < FEW_POINTS) or len(good) < MIN_FRAMES_OK
    spread = row["height_p75_mad"]
    limit = (
        max(HIGH_SPREAD_MM, HIGH_SPREAD_REL * row["height_p75"]) if np.isfinite(row["height_p75"]) else HIGH_SPREAD_MM
    )
    row["qc_high_spread"] = bool(np.isfinite(spread) and spread > limit)
    row["qc_overlap"] = bool(np.isfinite(row["clipped_fraction"]) and row["clipped_fraction"] > OVERLAP)
    row["qc_not_centred"] = bool(np.isfinite(row["centroid_offset_mm"]) and row["centroid_offset_mm"] > 1000 * 0.03)
    row["qc_dark"] = bool(meta.get("qc_dark", False))
    row["qc_ref_carried"] = bool(meta.get("qc_ref_carried", False))
    row["qc_no_pose"] = bool(meta.get("qc_no_pose", False))
    row["qc_pass"] = not any(row[f] for f, blocking in schema.QC_FLAGS.items() if blocking)
    return row


def runs_table(rows: list[dict]) -> pd.DataFrame:
    """traits_runs rows -> DataFrame with the contract dtypes, contract columns first."""
    df = pd.DataFrame(rows)
    cols = schema.run_columns()
    df = df[cols + [c for c in df.columns if c not in cols]]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).astype("datetime64[ns, UTC]")
    for c, t in schema.RUN_META.items():
        if t in ("string", "int64"):
            df[c] = df[c].astype(t)
    for f in [*schema.QC_FLAGS, "qc_pass"]:
        df[f] = df[f].astype(bool)
    return df.sort_values(schema.KEYS).reset_index(drop=True)
