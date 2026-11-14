"""Ground-truth link (traits against clip turgor) and recovery after rewatering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue.detect.data import label_of, trait_columns
from uiwue.stats import BASELINE, daily_values
from uiwue.stats._design import CONTROL_BEDS, DEFICIT_BEDS, prefix_of

MIN_PLANTS_PER_DAY = 5


def turgor_column(df: pd.DataFrame) -> str | None:
    cands = [c for c in df.columns if c.startswith("truth_") and "turgor" in c and "qc" not in c
             and pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in cands if "shift" in c]
    return (pref or cands or [None])[0]


def truth_link(df: pd.DataFrame, columns=None, days=(0, 12)) -> pd.DataFrame:
    """Spearman correlation of each trait (baseline delta) with the clip turgor shift.

    Two numbers per trait, on the clip plants, days 0-12:
    - rho_pooled: all plant-days together. Inflated by the shared time trend.
    - rho_within_day: mean over days of the across-plant correlation on that
      day. "Days since cutoff" alone scores 0 here, so this is the honest one.
    """
    tcol = turgor_column(df)
    if tcol is None:
        return pd.DataFrame()
    turg = daily_values(df, tcol, delta=False).rename(columns={"value": "turgor"})
    turg = turg[turg["days_since_cutoff"].between(*days)]
    rows = []
    for c in trait_columns(df) if columns is None else columns:
        t = daily_values(df, c).rename(columns={"value": "trait"})
        m = t.merge(turg, on=["bed", "plant", "date", "days_since_cutoff", "treatment"])
        if len(m) < 10:
            continue
        pooled = m[["trait", "turgor"]].corr(method="spearman").iloc[0, 1]
        per_day = [
            g[["trait", "turgor"]].corr(method="spearman").iloc[0, 1]
            for _, g in m.groupby("date") if len(g) >= MIN_PLANTS_PER_DAY
        ]
        per_day = [r for r in per_day if np.isfinite(r)]
        rows.append(dict(
            signal=c, label=label_of(c), sensor=prefix_of(c), turgor_column=tcol,
            rho_pooled=float(pooled), rho_within_day=float(np.mean(per_day)) if per_day else np.nan,
            n_days_within=len(per_day), n_plants=m[["bed", "plant"]].drop_duplicates().shape[0],
            n_obs=len(m),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# recovery
# --------------------------------------------------------------------------- #
END = (pd.Timestamp("2025-09-14"), pd.Timestamp("2025-09-15"))
AFTER = (pd.Timestamp("2025-09-20"), pd.Timestamp("2025-09-21"))
AFTER_DEPTH = (pd.Timestamp("2025-09-18"), pd.Timestamp("2025-09-19"))  # no depth from 19 Sep 15 h


def recovery(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Recovery ratio (after - end) / (before - end) per deficit plant and trait.

    1 = back to its own pre-cutoff value, 0 = no recovery. `measurable` is False
    when the plant's change from before to end of stress is within twice the
    control plants' spread of the same change, where the ratio means nothing.
    """
    rows = []
    for c in trait_columns(df) if columns is None else columns:
        d = daily_values(df, c, delta=False)
        after = AFTER_DEPTH if prefix_of(c) == "depth" else AFTER

        def win(w, d=d):
            return d[d["date"].between(*w)].groupby(["bed", "plant"])["value"].median()

        t = pd.DataFrame({"before": win(BASELINE), "end": win(END), "after": win(after)}).dropna()
        if t.empty:
            continue
        t["drop"] = t["before"] - t["end"]
        ctl = t[t.index.get_level_values("bed").isin(CONTROL_BEDS)]["drop"]
        ctl_mid = ctl.median() if len(ctl) else 0.0
        ctl_sd = ctl.std() if len(ctl) > 1 else np.nan
        t = t[t.index.get_level_values("bed").isin(DEFICIT_BEDS)].reset_index()
        t["recovery_ratio"] = (t["after"] - t["end"]) / (t["before"] - t["end"]).replace(0, np.nan)
        t["measurable"] = (
            (t["drop"] - ctl_mid).abs() > 2 * ctl_sd if np.isfinite(ctl_sd) else False
        )
        t["signal"] = c
        t["label"] = label_of(c)
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
