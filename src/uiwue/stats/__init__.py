"""Shared statistics (owner: Jack). Master Plan 6.6.

from uiwue.stats import baseline_delta, auc_by_day, onset_day, placebo

auc = auc_by_day(df, "thermal_leaf_air_dt")          # all deficit plants vs W1
day = onset_day(auc)                                 # first day CI low > 0.70 twice
pb  = placebo(df, "thermal_leaf_air_dt")             # pb["bed_effect"]
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
    ONSET_THRESHOLD,
    auc_by_day,
    auc_ci,
    baseline_delta,
    daily_values,
    early_auc,
    onset_day,
    placebo,
)

__all__ = [
    "BASELINE",
    "CUTOFF",
    "GROUPS",
    "KEYS",
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
    "load_catalogue",
    "onset_day",
    "placebo",
]
