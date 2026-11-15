"""Markdown for docs/fusion.md section 8 from the final v2 tables.

    python report.py <ablation parquet> [<lockbox summary dir>]
"""

import sys
from pathlib import Path

import pandas as pd

abl = pd.read_parquet(sys.argv[1])
lock_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def f2(x):
    return "–" if pd.isna(x) else f"{x:.2f}"


def on(x):
    return "–" if pd.isna(x) else str(int(x))


def ci(r):
    return f"{f3(r['early_auc'])} [{f3(r['early_ci_low'])}, {f3(r['early_ci_high'])}]"


def row(df, variant, sensors="thermal", group="all", plants="all"):
    r = df[(df["variant"] == variant) & (df["sensors"] == sensors) & (df["group"] == group) & (df["plants"] == plants)]
    return None if r.empty else r.iloc[0]


A = abl[(abl["group"] == "all") & (abl["plants"] == "all")]
variants = [v for v in dict.fromkeys(A["variant"]) if not v.startswith("size")]

print("#### Thermal setup, every variant (development plants, out of fold)\n")
print("| Variant | Early AUC (days 1-4) [95 % CI] | Onset day | AUC day 1 [lower] | AUC day 2 [lower] | AUC 2 Sep (day -1) | AUC day 0 | Own placebo | Bed model | No-clip early AUC | Stage ρ | Turgor ρ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for v in variants:
    r = row(abl, v)
    if r is None:
        continue
    nc = row(abl, v, plants="no_clip")
    print(
        f"| `{v}` | {ci(r)} | {on(r['onset_day'])} | {f2(r.get('day1_auc'))} [{f2(r.get('day1_ci_low'))}] | "
        f"{f2(r.get('day2_auc'))} [{f2(r.get('day2_ci_low'))}] | {f2(r.get('day_m1_auc'))} | {f2(r.get('day0_auc'))} | "
        f"{f2(r.get('placebo_auc_main'))} | {f2(r['placebo_auc'])} | {f3(nc['early_auc']) if nc is not None else '–'} | "
        f"{f2(r.get('stage_rho_median'))} | {f2(r.get('turgor_rho_median'))} |"
    )
print()
print("#### Other setups: early AUC / onset day / own placebo\n")
cols = [("thermal_r80", "Thermal 80 x 60"), ("depth", "Depth"), ("rgb", "RGB"), ("hsi", "Hyperspectral"),
        ("rgb+depth+thermal+hsi", "All four"), ("thermal:emb", "ImageBind thermal token alone"),
        ("thermal:feat", "Hand-made token alone")]
print("| Variant | " + " | ".join(c[1] for c in cols) + " |")
print("|---|" + "---|" * len(cols))
for v in variants:
    cells = []
    for s, _ in cols:
        r = row(abl, v, s)
        cells.append("–" if r is None else f"{f3(r['early_auc'])} / {on(r['onset_day'])} / {f2(r.get('placebo_auc_main'))}")
    if all(c == "–" for c in cells):
        continue
    print(f"| `{v}` | " + " | ".join(cells) + " |")
print()
print("#### Size alone\n")
for r in abl[abl["variant"].str.startswith("size")].itertuples():
    if r.group != "all":
        continue
    print(f"- {r.variant} ({r.plants}, {int(r.n_plants)} plants: {r.n_deficit} deficit, {r.n_control} control): "
          f"early AUC {f3(r.early_auc)} [{f3(r.early_ci_low)}, {f3(r.early_ci_high)}], onset {on(r.onset_day)}, placebo {f2(r.placebo_auc)}")
print()
print("#### Same plants with the size trait (thermal)\n")
for v in variants:
    r = row(abl, v, plants="with_size")
    if r is not None:
        print(f"- {v}: {ci(r)}, onset {on(r['onset_day'])} ({r['n_deficit']} deficit, {r['n_control']} control)")
print()
print("#### W2 and W3 separately (thermal: early AUC, onset)\n")
for v in variants:
    w2, w3 = row(abl, v, group="W2"), row(abl, v, group="W3")
    if w2 is not None:
        print(f"- {v}: W2 {f3(w2['early_auc'])}, {on(w2['onset_day'])}; W3 {f3(w3['early_auc']) if w3 is not None else '–'}, {on(w3['onset_day']) if w3 is not None else '–'}")
if lock_dir is not None:
    print()
    print("#### Lock-box\n")
    for p in sorted(lock_dir.glob("lockbox_*_summary.parquet")):
        d = pd.read_parquet(p)
        d = d[(d["group"] == "all")]
        for r in d.itertuples():
            if r.sensors not in ("thermal", "rgb+depth+thermal+hsi"):
                continue
            print(f"- {p.stem} {r.plants} {r.sensors}: {f3(r.early_auc)} [{f3(r.early_ci_low)}, {f3(r.early_ci_high)}], onset {on(r.onset_day)}, "
                  f"day1 {f2(getattr(r, 'day1_auc', float('nan')))}, 2 Sep {f2(getattr(r, 'day_m1_auc', float('nan')))}, own placebo {f2(r.placebo_auc_main)}, "
                  f"bed model {f2(r.placebo_auc)}, n {r.n_deficit}/{r.n_control}")
print()
print("n dev plants:", abl.get("n_plants_total", pd.Series(dtype=float)).dropna().unique())
