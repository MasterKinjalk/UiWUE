"""Apply the pre-registered selection rule (docs/fusion_progress.md 23:25, amended 23:58) to the
final ablation table and print the ranking.

Rule: among v2 variants whose thermal own placebo score (placebo_auc_main) is at most 0.65 and
whose thermal onset day is not day 0: earliest thermal onset day, then the higher early-AUC
lower bound, then the lower placebo_auc_main. Keep v1 if the winner does not beat the v1
design on onset or early AUC.
"""

import sys

import numpy as np
import pandas as pd

df = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
t = df[(df["sensors"] == "thermal") & (df["group"] == "all") & (df["plants"] == "all")]
t = t[~t["variant"].str.contains("base0902|benchmark|size")]
t = t.drop_duplicates("variant", keep="last").set_index("variant")
cols = ["early_auc", "early_ci_low", "onset_day", "placebo_auc_main", "day_m1_auc", "day0_auc", "day1_auc",
        "day1_ci_low", "day2_auc", "placebo_auc"]
t = t[cols].copy()
t["eligible"] = (t["placebo_auc_main"] <= 0.65) & (t["onset_day"] != 0) & (t.index != "v1")
t["onset_sort"] = t["onset_day"].fillna(np.inf)
rank = t[t["eligible"]].sort_values(["onset_sort", "early_ci_low", "placebo_auc_main"], ascending=[True, False, True])
pd.set_option("display.width", 200)
print(t.sort_values(["eligible", "onset_sort", "early_ci_low"], ascending=[False, True, False]).round(3).to_string())
print()
if rank.empty:
    print("no eligible variant: keep v1")
    sys.exit()
win = rank.index[0]
v1 = t.loc["v1"]
beats = (rank.loc[win, "onset_sort"] < (v1["onset_day"] if pd.notna(v1["onset_day"]) else np.inf)) or (
    rank.loc[win, "early_auc"] > v1["early_auc"]
)
print(f"winner by the rule: {win}; beats v1 on onset or early AUC: {beats}")
print("CHOSEN:", win if beats else "v1")
