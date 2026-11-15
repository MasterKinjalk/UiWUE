"""Combine the v2 ablation parquet files and print the markdown tables for docs/fusion.md section 8.

    python tables.py <out parquet> <in parquet> [<in parquet> ...]
"""

import sys

import numpy as np
import pandas as pd

out, ins = sys.argv[1], sys.argv[2:]
df = pd.concat([pd.read_parquet(p) for p in ins], ignore_index=True)
df = df.drop_duplicates(subset=["variant", "plants", "sensors", "group"], keep="last")
df.to_parquet(out, index=False)


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def f2(x):
    return "–" if pd.isna(x) else f"{x:.2f}"


def onset(x):
    return "–" if pd.isna(x) else f"{int(x)}"


def ci(r):
    return f"{f3(r['early_auc'])} [{f3(r['early_ci_low'])}, {f3(r['early_ci_high'])}]"


A = df[(df["group"] == "all") & (df["plants"] == "all")]
order = list(dict.fromkeys(A["variant"]))
print("### thermal")
print("| Variant | Early AUC [95 % CI] | Onset | 2 Sep (day -1) | Day 0 | Day 1 [low] | Day 2 [low] | Bed model | Own placebo | No-clip early AUC | Stage ρ | Turgor ρ | Deficit-prob ρ (turgor) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
nc = df[(df["group"] == "all") & (df["plants"] == "no_clip") & (df["sensors"] == "thermal")].set_index("variant")
for v in order:
    r = A[(A["variant"] == v) & (A["sensors"] == "thermal")]
    if r.empty:
        continue
    r = r.iloc[0]
    noclip = f3(nc.loc[v, "early_auc"]) if v in nc.index else "–"
    print(
        f"| {v} | {ci(r)} | {onset(r['onset_day'])} | {f2(r.get('day_m1_auc'))} | {f2(r.get('day0_auc'))} | "
        f"{f2(r.get('day1_auc'))} [{f2(r.get('day1_ci_low'))}] | {f2(r.get('day2_auc'))} [{f2(r.get('day2_ci_low'))}] | "
        f"{f2(r['placebo_auc'])} | {f2(r.get('placebo_auc_main'))} | {noclip} | {f2(r.get('stage_rho_median'))} | "
        f"{f2(r.get('turgor_rho_median'))} | {f2(r.get('turgor_rho_prob_median'))} |"
    )
print()
print("### other setups: early AUC, onset, own placebo")
cols = ["thermal:emb", "thermal:feat", "thermal_r80", "depth", "rgb", "hsi", "rgb+depth+thermal+hsi"]
print("| Variant | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
for v in order:
    cells = []
    for s in cols:
        r = A[(A["variant"] == v) & (A["sensors"] == s)]
        if r.empty:
            cells.append("–")
            continue
        r = r.iloc[0]
        cells.append(f"{f3(r['early_auc'])}, {onset(r['onset_day'])}, {f2(r.get('placebo_auc_main'))}")
    print(f"| {v} | " + " | ".join(cells) + " |")
print()
print("### groups W2 / W3 (thermal)")
for g in ("W2", "W3"):
    G = df[(df["group"] == g) & (df["plants"] == "all") & (df["sensors"] == "thermal")]
    print(g, "; ".join(f"{r.variant}: {f3(r.early_auc)} onset {onset(r.onset_day)}" for r in G.itertuples()))
print()
print("### size")
S = df[df["variant"].str.startswith("size")]
for r in S.itertuples():
    print(r.variant, r.plants, r.group, getattr(r, "n_plants", np.nan), f3(r.early_auc), f"[{f3(r.early_ci_low)}, {f3(r.early_ci_high)}]",
          "onset", onset(r.onset_day), "placebo", f2(r.placebo_auc), "n", r.n_deficit, r.n_control)
print()
print("### with_size subset (thermal)")
W = df[(df["plants"] == "with_size") & (df["group"] == "all") & (df["sensors"] == "thermal")]
for r in W.itertuples():
    print(r.variant, f3(r.early_auc), "onset", onset(r.onset_day), "n", r.n_deficit, r.n_control)
print()
print("n plants:", df.get("n_plants_total", pd.Series([np.nan])).dropna().unique()[:3])
