"""Onset day per signal: the shared statistics run over every trait in the catalogue.

Output (derived/detect/):
- auc_by_day.parquet:     signal, kind, group, date, day, auc, ci_low, ci_high, n_*
- onsets.parquet:         one row per signal and group, with placebo, pre-cutoff null and lead times
- onsets_by_slot.parquet: the same onset rules per time of day (one slot instead of 12 + 15 h)

Two onsets per signal:
- onset_day:        the contract rule (CI lower bound > 0.70 on two scored days in a row)
- robust_onset_day: the same rule with the bar raised to the pre-cutoff envelope, the
                    largest bed separation seen on any watered day (stats.null_envelope)

Two ground-truth anchors, both on the imaged plants of the group only:
- written symptoms (truth/plants): notes were taken on 2, 6, 8, 10, 12 and 15 Sep,
  so a note dates the symptom to an interval (previous note day, note day]. Most
  imaged deficit plants never got a note, so the anchor is the median over those
  that did, with its n beside it.
- leaf-clip turgor onset (truth/plants, clip plants): one day per plant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue import core
from uiwue.detect.data import SENSORS, label_of, size_baseline_column, trait_columns
from uiwue.stats import (
    CUTOFF,
    GROUPS,
    auc_by_day,
    early_auc,
    fires_before_cutoff,
    null_envelope,
    onset_day,
    placebo,
    robust_onset_day,
)
from uiwue.stats._design import DEFICIT_BEDS, direction_of, prefix_of

VISIBLE_SIGNAL = "rgb_exg"  # "visible" anchor for lead time (6.6); falls back below
VISIBLE_FALLBACK = ("rgb_ngrdi", "rgb_green_cover")
# Notes-sheet dates (2, 6, 8, 10, 12, 15 Sep) as days since cutoff
NOTE_DAYS = (-1, 3, 5, 7, 9, 12)
SLOT_HOURS = (6, 9, 12, 15, 18)


def _days(series) -> pd.Series:
    return (pd.to_datetime(series) - CUTOFF).dt.days


def symptom_days() -> pd.DataFrame | None:
    """Per-plant first written symptom and clip onset, as days since cutoff (truth/plants).

    Columns: bed, plant, symptom_day (the note day, an upper bound), symptom_day_low
    (the previous note day, a lower bound), clip_day. None without the table.
    """
    p = core.read_table("truth", "plants")
    if p is None or core.TRUTH_FIRST_SYMPTOM not in p.columns:
        return None
    out = p[["bed", "plant"]].copy()
    out["symptom_day"] = _days(p[core.TRUTH_FIRST_SYMPTOM])
    prev = {d: lo for lo, d in zip((np.nan, *NOTE_DAYS[:-1]), NOTE_DAYS, strict=True)}
    out["symptom_day_low"] = out["symptom_day"].map(prev)
    out["clip_day"] = _days(p[core.TRUTH_CLIP_ONSET]) if core.TRUTH_CLIP_ONSET in p.columns else np.nan
    return out


def imaged_plants(df: pd.DataFrame, columns) -> pd.DataFrame:
    """bed, plant of every plant with at least one value of a sensor trait."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["bed", "plant"])
    return df.loc[df[cols].notna().any(axis=1), ["bed", "plant"]].drop_duplicates()


def anchors(sym: pd.DataFrame | None, imaged: pd.DataFrame, group: str) -> dict:
    """Ground-truth anchors for one group, on its imaged deficit plants only."""
    out = dict(
        symptom_day=np.nan,
        symptom_day_low=np.nan,
        symptom_first_day=np.nan,
        n_symptom=0,
        clip_day=np.nan,
        n_clip=0,
        n_imaged=0,
    )
    if sym is None:
        return out
    beds = [b for b in GROUPS[group] if b in DEFICIT_BEDS]
    s = sym.merge(imaged, on=["bed", "plant"])
    s = s[s["bed"].isin(beds)]
    out["n_imaged"] = len(s)
    noted = s.dropna(subset=["symptom_day"])
    if len(noted):
        out.update(
            symptom_day=float(noted["symptom_day"].median()),
            symptom_day_low=float(noted["symptom_day_low"].median()),
            symptom_first_day=float(noted["symptom_day"].min()),
            n_symptom=len(noted),
        )
    clip = s["clip_day"].dropna()
    if len(clip):
        out.update(clip_day=float(clip.median()), n_clip=len(clip))
    return out


def _null_columns(a: pd.DataFrame) -> dict:
    pre = fires_before_cutoff(a)
    return dict(
        robust_onset_day=robust_onset_day(a),
        null_auc_max=null_envelope(a),
        precutoff_fire_day=pre,
        fires_before_cutoff=bool(np.isfinite(pre)),
    )


def run_onsets(
    df: pd.DataFrame, columns=None, groups=("all", "W2", "W3"), n_boot: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUC by day and onset for every trait column and group."""
    columns = trait_columns(df) if columns is None else columns
    size_col = size_baseline_column(df)
    sym = symptom_days()
    imaged = imaged_plants(df, columns)
    curves, rows = [], []
    for col in columns:
        sign = direction_of(col)
        for g in groups:
            if not df["bed"].isin(GROUPS[g]).any():
                continue
            a = auc_by_day(df, col, group=g, direction=sign, n_boot=n_boot)
            if a.empty:
                continue
            a = a.rename(columns={"column": "signal"}).assign(kind="trait")
            curves.append(a)
            pb = placebo(df, col, group=g, direction=sign, n_boot=n_boot)
            e, elo, ehi = early_auc(df, col, group=g, direction=sign, n_boot=n_boot)
            rows.append(
                dict(
                    signal=col,
                    label=label_of(col),
                    sensor=SENSORS.get(prefix_of(col), prefix_of(col)),
                    kind="trait",
                    group=g,
                    direction=sign,
                    onset_day=onset_day(a),
                    early_auc=e,
                    early_ci_low=elo,
                    early_ci_high=ehi,
                    max_auc=float(a.loc[a["day"].between(0, 12), "auc"].max()),
                    placebo_auc=pb["auc"],
                    placebo_ci_low=pb["ci_low"],
                    placebo_ci_high=pb["ci_high"],
                    bed_effect=pb["bed_effect"],
                    is_size_baseline=(col == size_col),
                    n_deficit=int(a["n_deficit"].max()),
                    n_control=int(a["n_control"].max()),
                    **_null_columns(a),
                    **anchors(sym, imaged, g),
                )
            )
    onsets = pd.DataFrame(rows)
    auc = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return auc, add_lead_times(onsets)


def run_slot_onsets(
    df: pd.DataFrame, columns=None, groups=("all", "W2", "W3"), hours=SLOT_HOURS, n_boot: int = 2000
) -> pd.DataFrame:
    """Both onsets per signal, group and single time-of-day slot (daily value = that one run)."""
    columns = trait_columns(df) if columns is None else columns
    rows = []
    for col in columns:
        sign = direction_of(col)
        for g in groups:
            if not df["bed"].isin(GROUPS[g]).any():
                continue
            for h in hours:
                a = auc_by_day(df, col, hours=(h,), group=g, direction=sign, n_boot=n_boot)
                if a.empty or a["n_deficit"].max() == 0:
                    continue
                rows.append(
                    dict(
                        signal=col,
                        label=label_of(col),
                        sensor=SENSORS.get(prefix_of(col), prefix_of(col)),
                        group=g,
                        hour=int(h),
                        onset_day=onset_day(a),
                        **_null_columns(a),
                        n_deficit=int(a["n_deficit"].max()),
                        n_control=int(a["n_control"].max()),
                    )
                )
    return pd.DataFrame(rows)


def add_lead_times(onsets: pd.DataFrame) -> pd.DataFrame:
    """Lead time = anchor day - onset day (positive = earlier warning).

    lead_vs_symptom / lead_vs_visible use the contract onset (kept for other pages);
    the *_robust and lead_vs_clip columns use the robust onset.
    """
    if onsets.empty:
        return onsets
    o = onsets.copy()
    # The visible anchor is the first RGB signal that passed the placebo in the pooled group.
    clean = set(o.loc[(o["group"] == "all") & ~o["bed_effect"].astype(bool), "signal"])
    vis = next((c for c in (VISIBLE_SIGNAL, *VISIBLE_FALLBACK) if c in clean), None)
    vis_day = o[o["signal"] == vis].set_index("group")["onset_day"] if vis else pd.Series(dtype=float)
    o["visible_signal"] = vis
    o["visible_onset_day"] = o["group"].map(vis_day)
    o["lead_vs_symptom"] = o["symptom_day"] - o["onset_day"]
    o["lead_vs_visible"] = o["visible_onset_day"] - o["onset_day"]
    o["lead_vs_symptom_robust"] = o["symptom_day"] - o["robust_onset_day"]
    o["lead_vs_clip"] = o["clip_day"] - o["robust_onset_day"]
    return o


__all__ = ["DEFICIT_BEDS", "add_lead_times", "anchors", "run_onsets", "run_slot_onsets", "symptom_days"]
