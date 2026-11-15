"""The four shared statistics (Master Plan 6.6). Nobody writes a second version.

- baseline_delta: value minus the plant's own baseline-window median at the same slot
- auc_by_day:     per-day AUC, deficit vs control plants, with a plant-bootstrap 95% CI
- onset_day:      first day whose CI lower bound is above 0.70 on that day and the next
- placebo:        the same AUC over the pre-baseline placebo window; flags bed effects

Pre-cutoff null (added after review): `null_envelope`, `fires_before_cutoff` and
`robust_onset_day` run the same daily AUC on the days before any plant was
stressed, so a day-level bed effect (sun, doors) cannot pass as early detection.

All functions take a run-level table with the four keys (bed, plant, date, hour)
and the trait column. Design columns are added here if missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue.stats._design import (
    BASELINE,
    CONTROL_BEDS,
    CUTOFF,
    DEFICIT_BEDS,
    PLACEBO,
    STRESS_DAYS,
    add_design,
    direction_of,
    qc_pass,
)

ONSET_THRESHOLD = 0.70
N_BOOT = 2000
SEED = 20250903
EARLY_DAYS = (1, 4)
# Watered days for the pre-cutoff null: placebo start (24 Aug) to day 0 (last watering 11:00).
NULL_DAYS = ((PLACEBO[0] - CUTOFF).days, 0)

GROUPS = {
    "all": DEFICIT_BEDS,
    "W2": ("W2",),
    "W3": ("W3",),
}


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #
def baseline_delta(df: pd.DataFrame, column: str, *, window=BASELINE) -> pd.Series:
    """`column` minus the plant's own median over the baseline window at the same slot.

    Runs that fail the module's QC are excluded from the baseline median (their
    own delta is still returned). Plants with no baseline at a slot get NaN.
    """
    d = add_design(df) if "treatment" not in df.columns else df
    in_win = d["date"].between(*window) & qc_pass(d, column)
    base = d.loc[in_win].groupby(["bed", "plant", "hour"])[column].median().rename("_base")
    joined = d[["bed", "plant", "hour"]].join(base, on=["bed", "plant", "hour"])
    return (d[column] - joined["_base"]).rename(f"{column}_delta")


def daily_values(
    df: pd.DataFrame,
    column: str,
    *,
    hours=(12, 15),
    delta: bool = True,
) -> pd.DataFrame:
    """One value per plant per day: median over the chosen slots of runs passing QC.

    Returns columns bed, plant, date, days_since_cutoff, treatment, value.
    """
    d = add_design(df)
    vals = baseline_delta(d, column) if delta else d[column]
    keep = d["hour"].isin(hours) & qc_pass(d, column) & vals.notna()
    out = d.loc[keep, ["bed", "plant", "date", "days_since_cutoff", "treatment"]].copy()
    out["value"] = vals[keep].to_numpy()
    return out.groupby(["bed", "plant", "date", "days_since_cutoff", "treatment"], as_index=False)["value"].median()


# --------------------------------------------------------------------------- #
# AUC with a plant bootstrap
# --------------------------------------------------------------------------- #
def auc_ci(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Mann-Whitney AUC P(pos > neg) and a percentile CI from resampling plants.

    Positives and negatives are resampled separately (stratified), so each
    bootstrap draw keeps the group sizes. Returns (auc, ci_low, ci_high).
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(SEED) if rng is None else rng
    # pairwise win matrix, computed once; ties count one half
    g = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = float(g.mean())
    if n_boot <= 0:
        return (auc, np.nan, np.nan)
    ip = rng.integers(0, len(pos), size=(n_boot, len(pos)))
    ineg = rng.integers(0, len(neg), size=(n_boot, len(neg)))
    boots = g[ip[:, :, None], ineg[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (auc, float(lo), float(hi))


def _split(daily: pd.DataFrame, group: str) -> tuple[pd.Series, pd.Series]:
    if group not in GROUPS:
        raise ValueError(f"group must be one of {sorted(GROUPS)}, got {group!r}")
    pos = daily.loc[daily["bed"].isin(GROUPS[group]), "value"]
    neg = daily.loc[daily["bed"].isin(CONTROL_BEDS), "value"]
    return pos, neg


def auc_by_day(
    df: pd.DataFrame,
    column: str,
    hours=(12, 15),
    group: str = "all",
    *,
    direction: int | None = None,
    days: tuple[int, int] | None = None,
    window: tuple | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per day: AUC of deficit vs control plants on baseline-delta daily values.

    Values are oriented by the catalogue direction (or `direction`), so AUC > 0.5
    always means "deficit plants moved the stress way". By default every day in
    the table is scored; restrict with `days` (days since cutoff, inclusive) or
    `window` (a pair of dates).

    Returns one row per day: column, group, direction, date, day, auc, ci_low,
    ci_high, n_deficit, n_control.
    """
    sign = direction_of(column) if direction is None else int(np.sign(direction) or 1)
    daily = daily_values(df, column, hours=hours)
    if days is not None:
        daily = daily[daily["days_since_cutoff"].between(*days)]
    if window is not None:
        daily = daily[daily["date"].between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))]
    daily = daily.assign(value=daily["value"] * sign)

    rng = np.random.default_rng(seed)
    rows = []
    for (date, day), sub in daily.groupby(["date", "days_since_cutoff"], sort=True):
        pos, neg = _split(sub, group)
        auc, lo, hi = auc_ci(pos.to_numpy(), neg.to_numpy(), n_boot=n_boot, rng=rng)
        rows.append(
            dict(
                column=column,
                group=group,
                direction=sign,
                date=date,
                day=int(day),
                auc=auc,
                ci_low=lo,
                ci_high=hi,
                n_deficit=len(pos),
                n_control=len(neg),
            )
        )
    cols = ["column", "group", "direction", "date", "day", "auc", "ci_low", "ci_high", "n_deficit", "n_control"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# onset
# --------------------------------------------------------------------------- #
def onset_day(
    auc: pd.DataFrame,
    *,
    threshold: float = ONSET_THRESHOLD,
    days: tuple[int, int] = STRESS_DAYS,
) -> float:
    """First day (since cutoff) whose CI lower bound is above `threshold` on that
    day and on the next day with data. NaN if never.

    The "next day" is the next scored day, so a missing run (e.g. 5 Sep 06 h)
    does not reset the rule; the last day in the window cannot qualify alone.
    """
    a = auc[auc["day"].between(*days)].sort_values("day")
    if a.empty:
        return np.nan
    ok = (a["ci_low"] > threshold).to_numpy()
    d = a["day"].to_numpy()
    for i in range(len(a) - 1):
        if ok[i] and ok[i + 1]:
            return float(d[i])
    return np.nan


def null_envelope(auc: pd.DataFrame, *, days: tuple[int, int] = NULL_DAYS) -> float:
    """Largest bed separation seen while every plant was still watered.

    0.5 + max |AUC - 0.5| over the pre-cutoff days (default: the placebo start to
    day 0, the day of the last watering). Orientation is ignored on purpose: a
    day on which the beds differ the "wrong" way shows how large a day-level bed
    effect (sun, doors, pots) can get just as well. NaN when no such day is scored.
    """
    a = auc[auc["day"].between(*days)]["auc"].dropna()
    return float(0.5 + (a - 0.5).abs().max()) if len(a) else np.nan


def fires_before_cutoff(
    auc: pd.DataFrame, *, threshold: float = ONSET_THRESHOLD, days: tuple[int, int] = NULL_DAYS
) -> float:
    """The onset rule run on the watered days: the day it fires there, or NaN.

    A signal whose rule fires before any plant was stressed cannot be trusted at
    face value on the first stress days.
    """
    return onset_day(auc, threshold=threshold, days=days)


def robust_onset_day(
    auc: pd.DataFrame,
    *,
    threshold: float = ONSET_THRESHOLD,
    days: tuple[int, int] = STRESS_DAYS,
    null_days: tuple[int, int] = NULL_DAYS,
) -> float:
    """Onset with the bar raised to the pre-cutoff envelope.

    First stress day whose CI lower bound is above max(`threshold`,
    `null_envelope`) on that day and the next scored day: the deficit and
    control plants must separate more clearly than the beds ever did while all
    were watered. Equals `onset_day` when no watered day is scored.
    """
    env = null_envelope(auc, days=null_days)
    bar = threshold if not np.isfinite(env) else max(threshold, env)
    return onset_day(auc, threshold=bar, days=days)


def early_auc(
    df: pd.DataFrame,
    column: str,
    hours=(12, 15),
    group: str = "all",
    *,
    days: tuple[int, int] = EARLY_DAYS,
    direction: int | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """One AUC over a range of days: each plant's mean daily value, then deficit vs control."""
    sign = direction_of(column) if direction is None else int(np.sign(direction) or 1)
    daily = daily_values(df, column, hours=hours)
    daily = daily[daily["days_since_cutoff"].between(*days)]
    per_plant = daily.groupby(["bed", "plant"], as_index=False)["value"].mean()
    per_plant["value"] *= sign
    pos, neg = _split(per_plant, group)
    return auc_ci(pos.to_numpy(), neg.to_numpy(), n_boot=n_boot, rng=np.random.default_rng(seed))


# --------------------------------------------------------------------------- #
# placebo
# --------------------------------------------------------------------------- #
def placebo(
    df: pd.DataFrame,
    column: str,
    hours=(12, 15),
    group: str = "all",
    *,
    direction: int | None = None,
    window=PLACEBO,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """The detection AUC computed over the placebo window (healthy, pre-baseline).

    Each plant's mean baseline-delta value over the window is compared between
    the future deficit beds and W1. Every plant was watered then, so any
    separation is a bed difference (light, pots, neighbours, growth rate).

    `bed_effect` is True when the CI excludes 0.5 and the AUC is at least 0.70
    in either orientation, i.e. the beds separate about as well as a detection
    would require.
    """
    sign = direction_of(column) if direction is None else int(np.sign(direction) or 1)
    daily = daily_values(df, column, hours=hours)
    daily = daily[daily["date"].between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))]
    per_plant = daily.groupby(["bed", "plant"], as_index=False)["value"].mean()
    per_plant["value"] *= sign
    pos, neg = _split(per_plant, group)
    auc, lo, hi = auc_ci(pos.to_numpy(), neg.to_numpy(), n_boot=n_boot, rng=np.random.default_rng(seed))
    separated = bool(np.isfinite(lo) and (lo > 0.5 or hi < 0.5) and max(auc, 1 - auc) >= ONSET_THRESHOLD)
    return dict(
        column=column,
        group=group,
        auc=auc,
        ci_low=lo,
        ci_high=hi,
        n_deficit=len(pos),
        n_control=len(neg),
        bed_effect=separated,
    )
