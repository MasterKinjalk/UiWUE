"""Onset day per signal: the shared statistics run over every trait in the catalogue.

Output (derived/detect/):
- auc_by_day.parquet: signal, kind, group, date, day, auc, ci_low, ci_high, n_*
- onsets.parquet:     one row per signal and group, with placebo and lead times
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue.detect.data import SENSORS, label_of, size_baseline_column, trait_columns
from uiwue.stats import CUTOFF, GROUPS, auc_by_day, early_auc, onset_day, placebo
from uiwue.stats._design import DEFICIT_BEDS, derived_dir, direction_of, prefix_of

VISIBLE_SIGNAL = "rgb_exg"  # "visible" anchor for lead time (6.6); falls back below
VISIBLE_FALLBACK = ("rgb_ngrdi", "rgb_green_cover")


def symptom_days() -> pd.DataFrame | None:
    """Per-plant first symptom, as days since cutoff, from truth/plants."""
    f = derived_dir() / "truth" / "plants.parquet"
    if not f.exists():
        return None
    p = pd.read_parquet(f)
    col = next((c for c in p.columns if "symptom" in c and "date" in c), None)
    if col is None:
        return None
    p = p[["bed", "plant", col]].copy()
    p["symptom_day"] = (pd.to_datetime(p[col]) - CUTOFF).dt.days
    return p[["bed", "plant", "symptom_day"]]


def _symptom_anchor(sym: pd.DataFrame | None, group: str) -> float:
    if sym is None:
        return np.nan
    s = sym.loc[sym["bed"].isin(GROUPS[group]), "symptom_day"].dropna()
    return float(s.median()) if len(s) else np.nan


def run_onsets(df: pd.DataFrame, columns=None, groups=("all", "W2", "W3"),
               n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUC by day and onset for every trait column and group."""
    columns = trait_columns(df) if columns is None else columns
    size_col = size_baseline_column(df)
    sym = symptom_days()
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
            rows.append(dict(
                signal=col, label=label_of(col), sensor=SENSORS.get(prefix_of(col), prefix_of(col)),
                kind="trait", group=g, direction=sign, onset_day=onset_day(a),
                early_auc=e, early_ci_low=elo, early_ci_high=ehi,
                max_auc=float(a.loc[a["day"].between(0, 12), "auc"].max()),
                placebo_auc=pb["auc"], placebo_ci_low=pb["ci_low"], placebo_ci_high=pb["ci_high"],
                bed_effect=pb["bed_effect"], is_size_baseline=(col == size_col),
                n_deficit=int(a["n_deficit"].max()), n_control=int(a["n_control"].max()),
                symptom_day=_symptom_anchor(sym, g),
            ))
    onsets = pd.DataFrame(rows)
    auc = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return auc, add_lead_times(onsets)


def add_lead_times(onsets: pd.DataFrame) -> pd.DataFrame:
    """Lead time = anchor day - onset day (positive = earlier warning)."""
    if onsets.empty:
        return onsets
    o = onsets.copy()
    # The visible anchor is the first RGB signal that passed the placebo in the pooled group.
    clean = set(o.loc[(o["group"] == "all") & ~o["bed_effect"].astype(bool), "signal"])
    vis = next((c for c in (VISIBLE_SIGNAL, *VISIBLE_FALLBACK) if c in clean), None)
    vis_day = (
        o[o["signal"] == vis].set_index("group")["onset_day"] if vis else pd.Series(dtype=float)
    )
    o["visible_signal"] = vis
    o["visible_onset_day"] = o["group"].map(vis_day)
    o["lead_vs_symptom"] = o["symptom_day"] - o["onset_day"]
    o["lead_vs_visible"] = o["visible_onset_day"] - o["onset_day"]
    return o


__all__ = ["DEFICIT_BEDS", "add_lead_times", "run_onsets", "symptom_days"]
