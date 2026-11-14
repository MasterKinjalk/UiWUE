"""Synthetic run-level tables with known answers, for testing Jack's lane.

Not real data. Never write these into the repo's `derived/`.

Plants: 7 W1 controls, 11 in W2, 11 in W3 (the core-set shape).
Known effects (days since cutoff, d):
- thermal_leaf_air_dt (+1): deficit rises 0.6 C/day from d=3 (W3 from d=2)
- rgb_green_cover     (-1): deficit falls 0.02/day from d=7
- depth_projected_area (-1): all grow; deficit growth stops from d=4
- hsi_ndvi            (-1): no effect at all
- rgb_bedfx           (+1): deficit beds trend upward the whole time (a bed effect)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2025-09-03")


def plants() -> pd.DataFrame:
    rows = [("W1", p) for p in ["A2", "A8", "J5", "R1", "R7", "B3", "Q6"]]
    rows += [("W2", f"C{i}") for i in range(1, 12)]
    rows += [("W3", f"D{i}") for i in range(1, 12)]
    return pd.DataFrame(rows, columns=["bed", "plant"])


def make_runs(seed: int = 0, hours=(6, 9, 12, 15, 18)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-08-24", "2025-09-22", freq="D")
    p = plants()
    grid = p.merge(pd.DataFrame({"date": dates}), how="cross").merge(
        pd.DataFrame({"hour": list(hours)}), how="cross"
    )
    n = len(grid)
    d = (grid["date"] - CUTOFF).dt.days.to_numpy()
    deficit = grid["bed"].isin(["W2", "W3"]).to_numpy()
    w3 = (grid["bed"] == "W3").to_numpy()
    plant_off = grid.groupby(["bed", "plant"]).ngroup().to_numpy()
    plant_lvl = rng.normal(0, 1, plant_off.max() + 1)[plant_off]

    start = np.where(w3, 2, 3)
    stress = np.clip(np.minimum(d, 12) - start, 0, None) * deficit

    g = pd.DataFrame(grid)
    g["thermal_leaf_air_dt"] = -2 + 0.3 * plant_lvl + 0.6 * stress + rng.normal(0, 0.4, n)
    g["rgb_green_cover"] = (
        0.4 + 0.03 * plant_lvl - 0.02 * np.clip(d - 7, 0, 12) * deficit + rng.normal(0, 0.01, n)
    )
    day_idx = (grid["date"] - grid["date"].min()).dt.days.to_numpy()
    growth = np.where(deficit, np.minimum(day_idx, (CUTOFF - grid["date"].min()).days + 4), day_idx)
    g["depth_projected_area"] = 100 + 5 * plant_lvl + 3.0 * growth + rng.normal(0, 1.5, n)
    g["hsi_ndvi"] = 0.8 + 0.02 * plant_lvl + rng.normal(0, 0.02, n)
    g["rgb_bedfx"] = 0.3 * day_idx * deficit + rng.normal(0, 0.3, n)
    g["rgb_qc_pass"] = True
    # one failed run to exercise QC
    g.loc[g.index[5], "rgb_qc_pass"] = False
    g["date"] = g["date"].dt.date
    return g


def catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("thermal_leaf_air_dt", "Leaf - air temperature", "C", 1, "synthetic"),
            ("rgb_green_cover", "Green cover", "fraction", -1, "synthetic"),
            ("depth_projected_area", "Projected area", "cm2", -1, "synthetic"),
            ("hsi_ndvi", "NDVI", "", -1, "synthetic"),
            ("rgb_bedfx", "Bed-effect trait", "", 1, "synthetic"),
        ],
        columns=["column", "label", "unit", "direction", "description"],
    )


def write_derived(root, seed: int = 0) -> None:
    """Write the synthetic table split by module into a fake `derived/` folder."""
    from pathlib import Path

    root = Path(root)
    runs = make_runs(seed)
    cat = catalogue()
    keys = ["bed", "plant", "date", "hour"]
    for mod in ["thermal", "rgb", "depth", "hsi"]:
        cols = [c for c in runs.columns if c.startswith(mod + "_")]
        sub = runs[keys + cols]
        if mod == "hsi":
            sub = sub[sub["hour"].isin([9, 12, 15])]
        (root / mod).mkdir(parents=True, exist_ok=True)
        sub.to_parquet(root / mod / "traits_runs.parquet", index=False)
        cat[cat["column"].str.startswith(mod + "_")].to_csv(root / mod / "catalogue.csv", index=False)

    # truth: clip turgor shift tracks thermal stress on 4 plants per deficit bed + 2 controls
    clip = runs[runs["plant"].isin(["A2", "J5", "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4"])]
    clip = clip[pd.to_datetime(clip["date"]) >= pd.Timestamp("2025-09-02")]
    rng = np.random.default_rng(seed + 1)
    turgor = clip[keys].copy()
    turgor["truth_turgor_shift"] = (clip["thermal_leaf_air_dt"] + 2) * 2 + rng.normal(0, 1, len(clip))
    (root / "truth").mkdir(parents=True, exist_ok=True)
    turgor.to_parquet(root / "truth" / "turgor_runs.parquet", index=False)
    p = plants()
    p["truth_first_symptom_date"] = np.where(
        p["bed"] == "W1", pd.NaT, pd.Timestamp("2025-09-12")
    )
    p["truth_first_symptom_date"] = pd.to_datetime(p["truth_first_symptom_date"])
    p.to_parquet(root / "truth" / "plants.parquet", index=False)
