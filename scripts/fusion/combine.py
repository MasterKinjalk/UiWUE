"""derived/detect/fusion_v2_ablation.parquet: every round of the v2 ablation, one row per
(round, variant, plants, sensors, group).

    python combine.py <out> round=file[,file...] ...
"""

import sys

import pandas as pd

ROUND_NOTE = {
    "r1": "43 plants, W1 capped at 18 (18/18/7), 1000 bootstraps",
    "r1_all": "66 plants (41/18/7), 1000 bootstraps",
    "r2": "77 plants, W1 capped at 35 (35/35/7), 1000 bootstraps",
    "r3": "121 plants, W1 capped at 54 (54/54/13), 1000 bootstraps",
    "r4": "161 plants (57/57/47), 1000 bootstraps; 6 W3 plants without the hand-made token",
    "final": "all 171 development plants (57/57/57), 2000 bootstraps",
}
out, specs = sys.argv[1], sys.argv[2:]
parts = []
for spec in specs:
    rnd, files = spec.split("=", 1)
    d = pd.concat([pd.read_parquet(f) for f in files.split(",")], ignore_index=True)
    d = d.drop_duplicates(subset=["variant", "plants", "sensors", "group"], keep="last")
    d.insert(0, "round", rnd)
    d.insert(1, "round_plants", ROUND_NOTE.get(rnd, ""))
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
for c in df.columns:  # mixed object columns (e.g. size_column) must be strings for parquet
    if df[c].dtype == object:
        df[c] = df[c].astype("string")
df.to_parquet(out, index=False)
print(df.groupby("round").size().to_dict(), "->", out)
