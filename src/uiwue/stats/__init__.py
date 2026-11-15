"""Shared statistics (owner: Jack). Master Plan 6.6.

from uiwue.stats import baseline_delta, auc_by_day, onset_day, placebo

auc = auc_by_day(df, "thermal_leaf_air_dt")          # all deficit plants vs W1
day = onset_day(auc)                                 # first day CI low > 0.70 twice
pb  = placebo(df, "thermal_leaf_air_dt")             # pb["bed_effect"]
rob = robust_onset_day(auc)                          # bar raised to the pre-cutoff envelope
"""

from uiwue.stats._design import (
    BASELINE,
    CUTOFF,
    KEYS,
    PLACEBO,
    REWATER,
    add_design,
    direction_of,
    load_catalogue,
)
from uiwue.stats.core import (
    GROUPS,
    NULL_DAYS,
    ONSET_THRESHOLD,
    auc_by_day,
    auc_ci,
    baseline_delta,
    daily_values,
    early_auc,
    fires_before_cutoff,
    null_envelope,
    onset_day,
    placebo,
    robust_onset_day,
)

__all__ = [
    "BASELINE",
    "CUTOFF",
    "GROUPS",
    "KEYS",
    "NULL_DAYS",
    "ONSET_THRESHOLD",
    "PLACEBO",
    "REWATER",
    "add_design",
    "auc_by_day",
    "auc_ci",
    "baseline_delta",
    "daily_values",
    "direction_of",
    "early_auc",
    "fires_before_cutoff",
    "load_catalogue",
    "null_envelope",
    "onset_day",
    "placebo",
    "robust_onset_day",
]
