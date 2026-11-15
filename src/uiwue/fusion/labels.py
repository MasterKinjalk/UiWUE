"""Training targets per run (Jonathan-Work-Plan sections 6 and 8).

- y_deficit: deficit bed and after the last watering (3 Sep 11:00). Controls,
  and every plant before the cutoff, are negative.
- stage (clip plants only, else -1):
    S0 healthy: control, before the cutoff, or no turgor loss that day
    S1 mild:    midday (12/15 h) clip shift above the threshold, recovered by 06 h
    S2 moderate: the loss is still there at 06 h
    S3 severe:  S2 and a written wilting/yellowing symptom on or before that day
  The threshold is the truth module's onset shift (5 raw units).
- turgor: the clip shift at the run (NaN where there is no clip).
- train: the run is inside the training window (up to the end of stress, day 12), except
  the cutoff day's runs after 11:00 (positive, but too soon after watering to show stress).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue import core

CUTOFF_HOUR = 11  # last watering of W2/W3 on the cutoff day
N_STAGES = 4
TURGOR_SCALE = 10.0  # the turgor head predicts shift / 10


def _turgor_threshold() -> float:
    try:
        from uiwue.truth import ONSET_SHIFT

        return float(ONSET_SHIFT)
    except ImportError:
        return 5.0


def run_labels(
    keys: pd.DataFrame, turgor: pd.DataFrame | None = None, plants: pd.DataFrame | None = None
) -> pd.DataFrame:
    """keys (bed, plant, date, hour) -> keys + y_deficit, stage, turgor, train, days_since_cutoff."""
    turgor = core.read_table("truth", "turgor_runs") if turgor is None else turgor
    plants = core.read_table("truth", "plants") if plants is None else plants
    d = core.add_design(core.normalize_keys(keys[["bed", "plant", "date", "hour"]]))
    cutoff = pd.Timestamp(core.CUTOFF)
    after = (d["date"] > cutoff) | ((d["date"] == cutoff) & (d["hour"] > CUTOFF_HOUR))
    d["y_deficit"] = ((d["treatment"] == "deficit") & after).astype(int)
    # The 12, 15 and 18 h runs of the cutoff day are positive by definition but come 1-7 h
    # after the last watering, so nothing in the image can separate them from the morning's
    # negatives except bed cues. They are scored (day 0 of the curves) but not trained on.
    # Controls on that afternoon are correct negatives and stay in.
    day0_after = (d["date"] == cutoff) & (d["y_deficit"] == 1)
    d["train"] = (d["date"] <= pd.Timestamp(core.REWATER)) & ~day0_after

    d["turgor"] = np.nan
    d["stage"] = -1
    if turgor is not None and "truth_turgor_shift" in turgor:
        t = core.normalize_keys(turgor)[core.KEYS + ["truth_turgor_shift"]]
        d = d.drop(columns="turgor").merge(t.rename(columns={"truth_turgor_shift": "turgor"}), on=core.KEYS, how="left")
        d["stage"] = _stages(d, plants)
    return d


def _stages(d: pd.DataFrame, plants: pd.DataFrame | None) -> np.ndarray:
    thr = _turgor_threshold()
    clip = d.groupby(["bed", "plant"])["turgor"].transform(lambda s: s.notna().any())
    day = d.groupby(["bed", "plant", "date"])
    midday = day["turgor"].transform(lambda s: s[d.loc[s.index, "hour"].isin([12, 15])].median())
    morning = day["turgor"].transform(lambda s: s[d.loc[s.index, "hour"] == 6].median())

    stage = np.full(len(d), -1)
    healthy = clip & ((d["treatment"] == "control") | (d["y_deficit"] == 0))
    stage[healthy.to_numpy()] = 0
    stressed = clip & (d["y_deficit"] == 1) & midday.notna()
    s = np.where(midday > thr, np.where(morning > thr, 2, 1), 0)
    stage[stressed.to_numpy()] = s[stressed.to_numpy()]

    if plants is not None and core.TRUTH_FIRST_SYMPTOM in plants:
        sym = core.normalize_keys(plants)[["bed", "plant", core.TRUTH_FIRST_SYMPTOM]]
        first = d[["bed", "plant"]].merge(sym, on=["bed", "plant"], how="left")[core.TRUTH_FIRST_SYMPTOM]
        severe = (stage == 2) & (pd.to_datetime(first).to_numpy() <= d["date"].to_numpy())
        stage[severe] = 3
    return stage
