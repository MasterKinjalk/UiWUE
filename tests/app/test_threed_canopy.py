"""The 3D canopy data layer (app/threed_canopy.py).

Two kinds of test: the real committed tables (derived/threed, derived/detect, and the local cloud cache
when this machine has it), and small synthetic sandboxes for the edge cases: a plant without a baseline,
a day whose runs all failed QC, a missing cloud folder, sparse clouds, a missing table, an odd run, a
detached cluster at the crop edge, W3 plants that must never be pooled. Tests on the real tables assert
properties of the table (it was re-run from 28 to 216 plants), not its size.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import threed_canopy as tc

from uiwue import core

REAL_TABLE = core.table_path("threed", "traits_runs")
REAL_CLOUDS = tc.clouds_dir()
needs_table = pytest.mark.skipif(
    not REAL_TABLE.exists(), reason="derived/threed/traits_runs.parquet not in this checkout"
)
needs_clouds = pytest.mark.skipif(not REAL_CLOUDS.is_dir(), reason="no local cloud cache (data/derived/threed/clouds)")
FLAGS = list(tc.FLAG_COLUMNS)


@pytest.fixture(autouse=True)
def fresh_caches():
    tc.clear_caches()
    yield
    tc.clear_caches()


# ---------------------------------------------------------------- synthetic sandbox
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Empty derived/ and data/ folders the module reads from."""
    derived, data = tmp_path / "derived", tmp_path / "data"
    derived.mkdir()
    data.mkdir()
    monkeypatch.setattr(core, "DERIVED", derived)
    monkeypatch.setattr(core, "DATA", data)
    return tmp_path


def run_row(bed, plant, day, hour, leaf, height=100.0, volume=1000.0, qc=True, n_points=1000, cap=None):
    return dict(
        bed=bed,
        plant=plant,
        day=day,
        hour=hour,
        leaf=leaf,
        height=height,
        volume=volume,
        qc=qc,
        n_points=n_points,
        cap=hour if cap is None else cap,
    )


def write_table(root, rows):
    """derived/threed/traits_runs.parquet in the committed team format (depth_ prefix, hour = slot)."""
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "bed": df["bed"],
            "plant": df["plant"],
            "date": [core.CUTOFF + timedelta(days=int(d)) for d in df["day"]],
            "hour": df["hour"].astype(int),
            "depth_leaf_area_est": df["leaf"].astype(float),
            "depth_height_p75": df["height"].astype(float),
            "depth_hull_volume": df["volume"].astype(float),
            "depth_height_p50": 1.0,  # a removed trait: must not come through
            "depth_n_points": df["n_points"].astype(float),
            "depth_capture_hour": df["cap"].astype(int),
        }
    )
    for f in FLAGS:
        out[f"depth_{f}"] = False
    out["depth_qc_overlap"] = ~df["qc"].astype(bool)
    out["depth_qc_pass"] = df["qc"].astype(bool)
    (root / "derived" / "threed").mkdir(parents=True, exist_ok=True)
    out.to_parquet(root / "derived" / "threed" / "traits_runs.parquet", index=False)
    tc.clear_caches()


def leaf_curve(day, deficit):
    """Leaf area: 100 cm² on baseline days; controls grow 2 %/day, deficit plants fall to 40 by day 10."""
    if day < 0 or not deficit:
        return 100.0 + 2.0 * max(day, 0)
    if day <= 3:
        return 100.0
    if day <= 12:
        return max(40.0, 100.0 - 8.0 * (day - 3))
    return 60.0


def experiment(skip=()):
    """Two control and three deficit plants, days −4 to 16, slots 9–18 h; `skip` = (bed, plant, day) runs dropped."""
    rows = []
    for bed, plants in (("W1", ("A2", "B3")), ("W2", ("A2", "B3", "C4"))):
        for k, plant in enumerate(plants):
            for day in range(-4, 17):
                if (bed, plant, day) in skip:
                    continue
                for hour in tc.DAY_SLOTS:
                    leaf = leaf_curve(day, bed == "W2") * (1 + 0.02 * k)
                    rows.append(run_row(bed, plant, day, hour, leaf, height=leaf, volume=10 * leaf))
    return rows


def write_cloud(root, bed, plant, day, cap, n_plant=600, n_soil=400, top_mm=120.0, seed=0):
    """A real-format cloud: plant points in the 85 mm cylinder (z 5 mm .. top), soil points at z ≈ 0."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 0.08, n_plant)
    a = rng.uniform(0, 2 * np.pi, n_plant)
    z = np.linspace(0.005, top_mm / 1000.0, n_plant)
    plant_xyz = np.c_[r * np.cos(a), r * np.sin(a), z]
    soil = np.c_[rng.uniform(-0.15, 0.15, (n_soil, 2)), rng.normal(0, 0.001, n_soil)]
    xyz = np.vstack([plant_xyz, soil]).astype(np.float32)
    mask = np.r_[np.ones(n_plant, bool), np.zeros(n_soil, bool)]
    date = core.CUTOFF + timedelta(days=day)
    path = root / "data" / "derived" / "threed" / "clouds" / f"{bed}_{plant}" / f"{date:%Y-%m-%d}_{cap:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((len(xyz), 3), np.uint8)
    np.savez_compressed(path, xyz=xyz, rgb=rgb, height=xyz[:, 2].copy(), plant=mask, frame=np.int64(7))
    tc.clear_caches()
    return path


# ---------------------------------------------------------------- registry and runs
def test_three_traits_with_labels_units_and_catalogue_columns():
    assert list(tc.TRAITS) == ["leaf_area_est", "height_p75", "hull_volume"]
    assert [t.label for t in tc.TRAITS.values()] == ["Leaf area", "Height", "Volume"]
    assert [t.unit for t in tc.TRAITS.values()] == ["cm²", "mm", "cm³"]
    cat = core.catalogue()
    assert {t.column for t in tc.TRAITS.values()} <= set(cat["column"])


@needs_table
def test_runs_is_the_committed_table_w1_w2_unless_w3_is_asked_for():
    r = tc.runs()
    raw = core.read_table("threed", "traits_runs")
    assert len(r) == raw["bed"].isin(["W1", "W2"]).sum()
    assert set(r["bed"]) == {"W1", "W2"}
    want = raw[raw["bed"].isin(["W1", "W2"])].groupby("bed")["plant"].nunique().to_dict()
    assert r.groupby("bed")["plant"].nunique().to_dict() == want and min(want.values()) > 0
    shown = tc.runs(tc.SHOWN_BEDS)
    assert len(shown) == raw["bed"].isin(tc.SHOWN_BEDS).sum()  # W3 only when asked for
    assert set(shown.loc[shown["bed"] == "W3", "treatment"]) <= {"deficit"}  # labelled as uiwue.core labels it
    assert set(r.columns) == set(tc.RUN_COLUMNS) | set(tc.TRAITS)  # no other trait leaks through
    assert set(r["slot"]) <= set(core.SLOTS)
    assert (r["day"] == (r["date"] - pd.Timestamp(core.CUTOFF)).dt.days).all()
    assert set(r.loc[r["bed"] == "W1", "treatment"]) == {"control"}
    assert set(r.loc[r["bed"] == "W2", "treatment"]) == {"deficit"}
    assert r["qc_pass"].dtype == bool and r["date"].min() == pd.Timestamp("2025-08-24")
    # a re-run keeps its folder hour, which names its cloud file
    assert (r["capture_hour"] != r["slot"]).any() and (r["capture_hour"] - r["slot"]).abs().max() <= 1


def test_runs_drops_w3_and_removed_traits(sandbox):
    rows = experiment() + [run_row("W3", "Q2", 0, 12, 50.0)]
    write_table(sandbox, rows)
    r = tc.runs()
    assert set(r["bed"]) == {"W1", "W2"} and "height_p50" not in r and "depth_height_p50" not in r


def test_missing_table_gives_none_and_empty(sandbox):
    assert tc.runs() is None and tc.plant_days() is None and tc.gap("height_p75") is None
    assert tc.windows() is None and tc.headline() is None and tc.onsets() is None
    assert tc.structure_status() is None and tc.structure_alerts() is None
    assert tc.cloud_days("W2", "J5").empty and tc.plant_cloud("W2", "J5", 0) is None
    assert tc.default_pair("W2", "J5") is None and tc.default_plant() is None
    assert tc.axis_ranges("W2", "J5") is None and tc.rgb_frame("W2", "J5", 0) is None
    assert all(not v["chip"] and np.isnan(v["placebo_auc"]) for v in tc.bed_effects().values())
    with pytest.raises(KeyError):
        tc.gap("droop_index")


# ---------------------------------------------------------------- plant-day table and baseline
@needs_table
def test_plant_day_is_the_mean_of_qc_pass_daytime_runs():
    r, p = tc.runs(), tc.plant_days()
    one = p[(p["bed"] == "W2") & (p["plant"] == "J5") & (p["day"] == 5)].iloc[0]
    runs = r[(r["bed"] == "W2") & (r["plant"] == "J5") & (r["day"] == 5) & r["slot"].isin(tc.DAY_SLOTS)]
    ok = runs[runs["qc_pass"]]
    assert one["n_runs"] == len(runs) and one["n_pass"] == len(ok)
    assert one["slots"] == ",".join(str(s) for s in sorted(ok["slot"]))
    for k in tc.TRAITS:
        assert one[k] == pytest.approx(ok[k].mean())
    # baseline: median of the plant-day values on days −4 to −1 (30 Aug – 2 Sep)
    mine = p[(p["bed"] == "W2") & (p["plant"] == "J5")]
    base = mine.loc[mine["day"].between(-4, -1), "height_p75"].median()
    assert one["height_p75_base"] == pytest.approx(base)
    assert one["height_p75_pct"] == pytest.approx((one["height_p75"] / base - 1) * 100)


@needs_table
def test_a_plant_has_a_baseline_exactly_when_it_has_a_qc_pass_baseline_day():
    p, r = tc.plant_days(tc.SHOWN_BEDS), tc.runs(tc.SHOWN_BEDS)
    base = p.groupby(["bed", "plant"])["leaf_area_est_base"].first()
    ok = r[r["qc_pass"] & r["slot"].isin(tc.DAY_SLOTS) & r["day"].between(*tc.BASELINE_DAYS)]
    with_value = ok.dropna(subset=["leaf_area_est"]).groupby(["bed", "plant"]).size()
    assert set(base.dropna().index) == set(with_value.index)
    for bed, plant in base[base.isna()].index:  # none in the 216-plant table; the sandbox test covers the case
        mine = p[(p["bed"] == bed) & (p["plant"] == plant)]
        assert mine["leaf_area_est_pct"].isna().all()


def test_no_baseline_and_all_failed_day_in_a_sandbox(sandbox):
    rows = experiment(skip={("W2", "C4", d) for d in range(-4, 0)})
    rows += [run_row("W2", "C4", -4, 12, 90.0, qc=False)]  # its only baseline run failed QC
    for row in rows:  # every run of W1 A2 on day 6 fails QC
        if (row["bed"], row["plant"], row["day"]) == ("W1", "A2", 6):
            row["qc"] = False
    write_table(sandbox, rows)
    p = tc.plant_days()
    c4 = p[(p["bed"] == "W2") & (p["plant"] == "C4")]
    assert len(c4) == 18 and c4["leaf_area_est_pct"].isna().all()  # days −4 (failed run only) and 0 to 16
    assert c4["leaf_area_est"].notna().sum() == 17 and c4["leaf_area_est_base"].isna().all()
    failed = p[(p["bed"] == "W1") & (p["plant"] == "A2") & (p["day"] == 6)].iloc[0]
    assert failed["n_runs"] == 4 and failed["n_pass"] == 0 and failed["slots"] == ""
    assert np.isnan(failed["leaf_area_est"]) and np.isnan(failed["leaf_area_est_pct"])
    a2 = p[(p["bed"] == "W2") & (p["plant"] == "A2")].set_index("day")
    assert a2.loc[-1, "leaf_area_est_base"] == pytest.approx(100.0)
    assert a2.loc[10, "leaf_area_est_pct"] == pytest.approx(leaf_curve(10, True) - 100.0)
    # no baseline -> not in the gap counts; the failed day -> one control plant fewer that day
    g = tc.gap("leaf_area_est")["by_day"].set_index("day")
    assert g.loc[10, "n_deficit"] == 2 and g.loc[6, "n_control"] == 1 and g.loc[5, "n_control"] == 2
    assert np.isnan(g.loc[6, "ci_low"])  # a group with one plant gets no interval
    st = tc.structure_status()
    assert set(st.loc[(st["bed"] == "W2") & (st["plant"] == "C4"), "status"]) == {"no baseline"}
    assert st.loc[(st["bed"] == "W1") & (st["plant"] == "A2") & (st["day"] == 6), "status"].item() == "no scan"


# ---------------------------------------------------------------- gap series
@needs_table
def test_gap_is_deficit_minus_control_with_a_fixed_band():
    g = tc.gap("leaf_area_est")
    d, p = g["by_day"], tc.plant_days()
    day8 = p[p["day"] == 8]
    expect = (
        day8.loc[day8["bed"] == "W2", "leaf_area_est_pct"].mean()
        - day8.loc[day8["bed"] == "W1", "leaf_area_est_pct"].mean()
    )
    row = d[d["day"] == 8].iloc[0]
    assert row["gap"] == pytest.approx(expect) and row["ci_low"] < row["gap"] < row["ci_high"]
    assert row["n_deficit"] == day8.loc[day8["bed"] == "W2", "leaf_area_est_pct"].notna().sum()
    assert (d["gap"] - (d["deficit"] - d["control"])).abs().max() < 1e-9
    pre = d.loc[d["day"].between(-10, 0), "gap"]
    assert g["null_range"] == (pytest.approx(pre.min()), pytest.approx(pre.max())) and g["null_days"] == (-10, 0)
    tc.clear_caches()
    again = tc.gap("leaf_area_est")["by_day"]
    pd.testing.assert_frame_equal(d, again)  # seeded: the same band on every call
    assert row["ci_high"] < 0 and d.loc[d["day"] == 12, "ci_high"].item() < 0  # deficit plants lost leaf area


# ---------------------------------------------------------------- collapse and recovery
@needs_table
def test_windows_pair_each_plant_with_its_own_peak():
    w, p = tc.windows(), tc.plant_days()
    plants = p[["bed", "plant"]].drop_duplicates()
    assert len(w) == len(plants) * 3 and set(w["trait"]) == set(tc.TRAITS) and set(w["bed"]) == {"W1", "W2"}
    shown = tc.plant_days(tc.SHOWN_BEDS)[["bed", "plant"]].drop_duplicates()
    assert len(tc.windows(tc.SHOWN_BEDS)) == len(shown) * 3
    one = w[(w["bed"] == "W2") & (w["plant"] == "J5") & (w["trait"] == "height_p75")].iloc[0]
    mine = p[(p["bed"] == "W2") & (p["plant"] == "J5")]
    peak = mine.loc[mine["day"].between(1, 3), "height_p75"].mean()
    trough = mine.loc[mine["day"].between(10, 12), "height_p75"].mean()
    assert one["peak_to_trough"] == pytest.approx((trough / peak - 1) * 100)
    assert one["trough"] == pytest.approx(mine.loc[mine["day"].between(10, 12), "height_p75_pct"].mean())
    assert one["n_trough"] == mine.loc[mine["day"].between(10, 12), "height_p75"].notna().sum()
    no_trough = w[(w["trait"] == "height_p75") & (w["n_trough"] == 0)]
    assert no_trough["trough"].isna().all() and no_trough["peak_to_trough"].isna().all()  # no usable trough day


@needs_table
def test_headline_is_the_plant_mean_with_a_plant_bootstrap():
    h, w = tc.headline(), tc.windows()
    s = h["height_p75"]["peak_to_trough"]["deficit"]
    v = w.loc[(w["trait"] == "height_p75") & (w["bed"] == "W2"), "peak_to_trough"].dropna()
    assert s["n"] == len(v) and s["mean"] == pytest.approx(v.mean())
    assert s["ci_low"] < s["mean"] < s["ci_high"] and s["min"] == pytest.approx(v.min())
    # the orientation the brief reports (section B): deficit plants collapse, controls keep growing
    assert h["height_p75"]["peak_to_trough"]["deficit"]["mean"] < -25
    assert h["leaf_area_est"]["peak_to_trough"]["deficit"]["mean"] < -50
    assert h["height_p75"]["peak_to_trough"]["control"]["mean"] > 0
    assert h["leaf_area_est"]["peak_to_rewatered"]["deficit"]["ci_high"] < -40  # area does not come back
    assert h["height_p75"]["peak_to_rewatered"]["deficit"]["ci_high"] > 0  # height does
    h["height_p75"]["peak_to_trough"]["deficit"]["mean"] = 0  # a copy: the cache is untouched
    assert tc.headline()["height_p75"]["peak_to_trough"]["deficit"]["mean"] == pytest.approx(v.mean())


def test_windows_in_a_sandbox(sandbox):
    write_table(sandbox, experiment())
    w = tc.windows()
    a2 = w[(w["bed"] == "W2") & (w["plant"] == "A2") & (w["trait"] == "leaf_area_est")].iloc[0]
    trough = np.mean([leaf_curve(d, True) for d in (10, 11, 12)])
    assert a2["peak"] == pytest.approx(0.0) and a2["trough"] == pytest.approx(trough - 100)
    assert a2["peak_to_trough"] == pytest.approx(trough - 100) and a2["peak_to_rewatered"] == pytest.approx(-40)
    h = tc.headline()["leaf_area_est"]["peak_to_trough"]
    assert h["deficit"]["n"] == 3 and h["control"]["n"] == 2 and h["control"]["mean"] > 0


# ---------------------------------------------------------------- onsets and bed checks
@needs_table
def test_onsets_come_from_the_detect_tables():
    o = tc.onsets()
    src = pd.read_parquet(core.DERIVED / "detect" / "onsets.parquet")
    src = src[src["group"] == "W2"].set_index("signal")
    assert {t.column for t in tc.TRAITS.values()} <= set(o["signal"])
    assert {"Thermal", "RGB", "Depth"} <= set(o["sensor"])
    assert o["is_size_baseline"].sum() == 1  # projected area, the size-only reference (DESIGN 7.6)
    for r in o.itertuples():
        assert r.robust_onset_day == src.loc[r.signal, "robust_onset_day"] or np.isnan(r.robust_onset_day)
        assert r.onset_day == src.loc[r.signal, "onset_day"] or np.isnan(r.onset_day)
    thermal = o[o["sensor"] == "Thermal"].iloc[0]
    assert not thermal["bed_effect"] and not thermal["fires_before_cutoff"]
    slots = pd.read_parquet(core.DERIVED / "detect" / "onsets_by_slot.parquet")
    s = slots[(slots["group"] == "W2") & (slots["signal"] == "depth_leaf_area_est")].set_index("hour")
    leaf = o[o["trait"] == "leaf_area_est"].iloc[0]
    for h in tc.DAY_SLOTS:
        assert leaf[f"onset_{h:02d}"] == s.loc[h, "robust_onset_day"]
    per = [leaf[f"onset_{h:02d}"] for h in tc.DAY_SLOTS]
    assert leaf["slot_min"] == np.nanmin(per) and leaf["slot_max"] == np.nanmax(per)
    assert list(o["robust_onset_day"].fillna(99)) == sorted(o["robust_onset_day"].fillna(99))


@needs_table
def test_bed_effects_read_both_checks():
    """The placebo of the page's own comparison, W2 against W1 (as its onsets), not the pooled W2 + W3 row: the
    detect run now has W3 3D plants, whose canopy also lost a leaf to sampling."""
    b = tc.bed_effects()
    placebo = pd.read_parquet(core.DERIVED / "detect" / "placebo.parquet")
    placebo = placebo[placebo["group"] == "W2"].set_index("signal")
    e2 = pd.read_csv(core.DERIVED / "threed" / "results" / "e2_bed_baseline.csv").set_index("trait")
    for k, t in tc.TRAITS.items():
        assert b[k]["placebo"] == bool(placebo.loc[t.column, "bed_effect"])
        assert b[k]["placebo_auc"] == pytest.approx(placebo.loc[t.column, "placebo_auc"])
        assert b[k]["e2"] == (str(e2.loc[k, "warning"]).strip().lower() in {"true", "1", "yes"})
        assert b[k]["chip"] == (b[k]["placebo"] or b[k]["e2"])  # the chip shows when either check flags the trait
    assert b["height_p75"]["chip"]  # height carries the bed check on every table so far
    assert b["height_p75"]["e2_coef"] == pytest.approx(e2.loc["height_p75", "day_bed_coef"])
    assert b["height_p75"]["e2_unit"] == "mm/day"


# ---------------------------------------------------------------- structure status
def test_structure_rule_watch_irrigate_latch_and_no_scan(sandbox):
    leaf = {0: 90, 1: 75, 3: 70, 4: 95, 5: 95, 12: 95, 13: 95, 14: 70, 15: 70, 16: 95}  # day 2: no QC-pass run
    rows = [run_row("W1", "A2", d, h, 100.0) for d in range(-4, 17) for h in tc.DAY_SLOTS]
    for d in range(-4, 17):
        for h in tc.DAY_SLOTS:
            value = 100.0 if d < 0 else float(leaf.get(d, 95))
            rows.append(run_row("W2", "A2", d, h, value, qc=d != 2))
    write_table(sandbox, rows)
    st = tc.structure_status().query("bed == 'W2'").set_index("day")
    assert st.index.min() == 0  # nothing is scored before the cutoff
    assert st.loc[0, "status"] == "fine" and st.loc[1, "status"] == "watch"
    assert st.loc[2, "status"] == "no scan" and st.loc[3, "status"] == "irrigate"  # a missing day does not reset
    assert st.loc[3, "since"] == 3 and st.loc[4, "raw"] == "fine" and st.loc[4, "status"] == "irrigate"
    assert st.loc[12, "status"] == "irrigate" and st.loc[13, "status"] == "fine"  # latched until re-watering
    assert st.loc[14, "status"] == "watch" and st.loc[15, "status"] == "irrigate" and st.loc[16, "status"] == "fine"
    a = tc.structure_alerts().set_index(["bed", "plant"])
    assert a.loc[("W2", "A2"), "first_watch"] == 1 and a.loc[("W2", "A2"), "first_irrigate"] == 3
    assert a.loc[("W2", "A2"), "status_day12"] == "irrigate" and a.loc[("W2", "A2"), "worst_pct"] == pytest.approx(-30)
    assert np.isnan(a.loc[("W1", "A2"), "first_irrigate"])
    # a stricter threshold is a parameter, not a silent change
    assert np.isnan(tc.structure_alerts(-35.0).set_index(["bed", "plant"]).loc[("W2", "A2"), "first_irrigate"])


@needs_table
def test_structure_rule_on_the_committed_table():
    a = tc.structure_alerts()
    plants = tc.plant_days()[["bed", "plant"]].drop_duplicates()
    assert len(a) == len(plants) and set(a["bed"]) == {"W1", "W2"}  # W3 is never scored with the beds
    deficit = a[(a["bed"] == "W2") & a["has_baseline"]]
    control = a[a["bed"] == "W1"]
    hit, false = deficit["first_irrigate"].notna().mean(), control["first_irrigate"].notna().mean()
    assert hit > 0.75 and false < 0.10  # the −20 % two-day rule separates the beds (63 of 72 vs 2 of 72 here)
    assert (a["first_irrigate"].dropna() >= 0).all()


# ---------------------------------------------------------------- clouds
def cloud_experiment(sandbox):
    """W2 A2 with clouds: day −1 12 h, day 3 15 h (12 h file missing), day 5 12 h sparse (60 points) and
    15 h fine, day 6 12 h under half its run's points, day 8 12 h a QC-failed run with a cloud, day 10 9 h."""
    rows = experiment()
    for row in rows:
        if (row["bed"], row["plant"], row["day"]) == ("W2", "A2", 8):
            row["qc"] = False  # every run of day 8 fails QC
    write_table(sandbox, rows)
    write_cloud(sandbox, "W2", "A2", -1, 12, top_mm=150.0)
    write_cloud(sandbox, "W2", "A2", 3, 15, top_mm=181.0)
    write_cloud(sandbox, "W2", "A2", 5, 12, n_plant=60)
    write_cloud(sandbox, "W2", "A2", 5, 15)
    write_cloud(sandbox, "W2", "A2", 6, 12, n_plant=400)  # run median 1000 points: under half
    write_cloud(sandbox, "W2", "A2", 8, 12)
    write_cloud(sandbox, "W2", "A2", 10, 9, top_mm=90.0)


def test_cloud_days_pick_a_drawable_qc_pass_run(sandbox):
    cloud_experiment(sandbox)
    d = tc.cloud_days("W2", "A2").set_index("day")
    assert d.loc[-1, "ok"] and d.loc[-1, "slot"] == 12 and d.loc[-1, "n_plant"] == 600
    assert d.loc[3, "ok"] and d.loc[3, "slot"] == 15  # 12 h file missing: the next preferred slot
    assert d.loc[5, "ok"] and d.loc[5, "slot"] == 15  # 12 h too sparse
    assert not d.loc[6, "ok"] and d.loc[6, "reason"] == "sparse" and d.loc[6, "slot"] == 12  # 12 h names it
    assert d.loc[8, "reason"] == "no scan" and not d.loc[8, "ok"]  # a cloud of a failed run is never offered
    assert d.loc[10, "ok"] and d.loc[10, "slot"] == 9
    assert d.loc[0, "reason"] == "no file"
    assert d.loc[-1, "z_max"] == pytest.approx(150.0, abs=0.01)
    assert d.loc[-1, "share"] == pytest.approx(0.6)
    six = tc.cloud_days("W2", "A2")
    assert set(six.loc[six["ok"], "day"]) == {-1, 3, 5, 10}


def test_sparse_share_is_a_reason(sandbox):
    rows = [run_row("W2", "A2", d, h, 100.0) for d in (-1, 0) for h in tc.DAY_SLOTS]
    write_table(sandbox, rows)
    write_cloud(sandbox, "W2", "A2", -1, 12, n_plant=400)  # run median 1000 points
    write_cloud(sandbox, "W2", "A2", 0, 12, n_plant=60)
    d = tc.cloud_days("W2", "A2").set_index("day")
    assert d.loc[-1, "reason"] == "sparse" and d.loc[-1, "share"] == pytest.approx(0.4)
    assert d.loc[0, "reason"] == "sparse" and d.loc[0, "n_plant"] == 60
    assert tc.plant_cloud("W2", "A2", -1) is None and tc.plant_cloud("W2", "A2", 0, 12) is None


def test_plant_cloud_is_plant_points_in_mm_or_none(sandbox):
    cloud_experiment(sandbox)
    c = tc.plant_cloud("W2", "A2", -1)
    assert c["n"] == 600 and len(c["x"]) == len(c["y"]) == len(c["z"]) == 600  # soil points dropped
    assert c["slot"] == 12 and c["capture_hour"] == 12 and c["frame"] == 7 and c["day"] == -1
    assert c["z"].min() == pytest.approx(5.0, abs=0.01) and c["z"].max() == pytest.approx(150.0, abs=0.01)
    assert np.hypot(c["x"], c["y"]).max() <= tc.CYLINDER_MM
    with pytest.raises(ValueError):
        c["x"][0] = 1.0  # read-only: the cache cannot be corrupted
    assert tc.plant_cloud("W2", "A2", 5)["slot"] == 15
    assert tc.plant_cloud("W2", "A2", 5, 12) is None  # an explicit slot must be drawable too
    assert tc.plant_cloud("W2", "A2", 8) is None and tc.plant_cloud("W2", "A2", 8, 12) is None  # QC-failed
    assert tc.plant_cloud("W2", "A2", 6) is None and tc.plant_cloud("W2", "A2", 99) is None
    assert tc.plant_cloud("W2", "ZZ", -1) is None


def test_no_cloud_folder_means_no_cloud_never_a_synthetic_one(sandbox):
    write_table(sandbox, experiment())
    d = tc.cloud_days("W2", "A2")
    assert len(d) == 21 and set(d["reason"]) == {"no file"} and not d["ok"].any()
    assert tc.plant_cloud("W2", "A2", -1) is None and tc.axis_ranges("W2", "A2") is None
    assert tc.default_pair("W2", "A2") is None and tc.default_plant() is None
    assert tc.plant_days() is not None and tc.gap("height_p75") is not None  # trait charts still work


def test_default_pair_and_plant(sandbox):
    cloud_experiment(sandbox)
    pair = dict(then=-1, then_slot=12, now=10, now_slot=9, then_in_baseline=True, now_in_trough=True)
    assert tc.default_pair("W2", "A2") == pair
    assert tc.default_plant() == ("W2", "A2")  # the only W2 plant with clouds on a baseline and a trough day
    write_cloud(sandbox, "W2", "B3", -3, 12)
    write_cloud(sandbox, "W2", "B3", 4, 12)
    write_cloud(sandbox, "W2", "B3", 13, 12)  # after re-watering: never the NOW default
    pair = dict(then=-3, then_slot=12, now=4, now_slot=12, then_in_baseline=True, now_in_trough=False)
    assert tc.default_pair("W2", "B3") == pair
    write_cloud(sandbox, "W2", "C4", 4, 12)
    assert tc.default_pair("W2", "C4") is None  # no THEN cloud: no pair
    assert tc.default_plant() == ("W2", "A2")  # B3 has no trough-day cloud, C4 no THEN


def test_default_candidates_rank_typical_agreeing_plants_by_the_visible_drop(sandbox):
    """The default is a rule: typical W2 responders whose NOW run agrees with the bed tiles, the largest drop first."""
    rows = experiment()
    for row in rows:  # C4 loses the most height at the trough; B3 has no trough-day run at all
        if (row["bed"], row["plant"]) == ("W2", "C4") and row["day"] >= 10:
            row["height"] *= 0.8
        if (row["bed"], row["plant"]) == ("W2", "B3") and 10 <= row["day"] <= 12:
            row["qc"] = False
    write_table(sandbox, rows)
    c = tc.default_candidates()
    assert list(c["plant"]) == ["C4", "A2"] and c["agrees"].all()  # B3: no trough run, so no candidate
    assert c.loc[0, "drop"] < c.loc[1, "drop"] and c.loc[0, "now_day"] == 12 and c.loc[0, "then_day"] == -1
    assert tc.default_plant() is None  # no clouds on this machine: nothing to draw, no default
    for plant in ("A2", "C4"):
        write_cloud(sandbox, "W2", plant, -2, 12)
        write_cloud(sandbox, "W2", plant, 11, 12)
    assert tc.default_plant() == ("W2", "C4")  # both drawable: the larger visible drop wins


def test_axis_ranges_and_side_profile(sandbox):
    cloud_experiment(sandbox)
    r = tc.axis_ranges("W2", "A2")
    assert r == {"x": (-85.0, 85.0), "y": (-85.0, 85.0), "z": (0.0, 190.0)}  # 181 mm rounded up
    c = {"x": np.array([-84.0, -83.0, 0.1, 0.2, 84.9]), "y": np.zeros(5), "z": np.array([10.0, 30.0, 50.0, 20.0, 5.0])}
    p = tc.side_profile(c)
    assert len(p) == 34 and p["x"].iloc[0] == -82.5 and p["x"].iloc[-1] == 82.5
    assert p["z"].iloc[0] == 30.0 and p["n"].iloc[0] == 2 and p["z"].iloc[-1] == 5.0
    mid = p[p["x"] == 2.5].iloc[0]
    assert mid["z"] == 50.0 and mid["n"] == 2 and p["z"].isna().sum() == 31
    assert tc.side_profile(None)["z"].isna().all()
    real = tc.side_profile(tc.plant_cloud("W2", "A2", -1))
    assert real["z"].max() <= r["z"][1] and real["n"].sum() == 600


def test_rgb_frame(sandbox):
    from PIL import Image

    cloud_experiment(sandbox)
    folder = sandbox / "data" / "rgb" / "W2_A2" / "2025_09_02" / "12"
    folder.mkdir(parents=True)
    Image.new("RGB", (960, 720), (40, 120, 60)).save(folder / "rgb_007.jpg")
    uri = tc.rgb_frame("W2", "A2", -1)
    assert uri.startswith("data:image/jpeg;base64,")
    assert tc.rgb_frame("W2", "A2", 3) is None  # the day's cloud exists, its RGB frame does not


def test_returned_frames_do_not_change_the_cache(sandbox):
    write_table(sandbox, experiment())
    p = tc.plant_days()
    p["leaf_area_est"] = 0.0
    p.loc[0, "plant"] = "ZZ"
    again = tc.plant_days()
    assert again["leaf_area_est"].max() > 0 and "ZZ" not in set(again["plant"])
    r = tc.runs()
    r["extra"] = 1
    assert "extra" not in tc.runs()


# ---------------------------------------------------------------- the real clouds on this machine
@needs_table
@needs_clouds
def test_real_clouds_default_plant_tells_one_story():
    """The default landing: a W2 plant whose THEN and NOW runs and the card beside them all say what the tiles say."""
    bed, plant = tc.default_plant()
    assert bed == "W2"
    pair = tc.default_pair(bed, plant)
    assert pair["then_in_baseline"] and pair["now_in_trough"]
    best = tc.default_candidates().set_index("plant").loc[plant]
    assert best["agrees"] and best["typical"]
    now = tc.cloud_days(bed, plant).set_index("day").loc[pair["now"]]
    h = tc.headline()
    for k in tc.TRAITS:  # the card (the NOW run's own values) has the sign of the bed's loss
        assert now[f"{k}_run_pct"] < 0 and h[k]["peak_to_trough"]["deficit"]["mean"] < 0
    then = tc.plant_cloud(bed, plant, pair["then"], pair["then_slot"])
    drawn = tc.plant_cloud(bed, plant, pair["now"])
    assert drawn["slot"] == pair["now_slot"] and drawn["height_p75_run"] < then["height_p75_run"]  # it shrank
    r = tc.runs()
    for p in (plant, "J5", "R1"):
        d = tc.cloud_days("W2", p)
        ok = d[d["ok"]]
        merged = ok.merge(r[(r["bed"] == "W2") & (r["plant"] == p)], on=["day", "slot"])
        assert len(merged) == len(ok) and merged["qc_pass"].all()  # only QC-pass runs are offered
        assert (ok["n_plant"] >= tc.MIN_PLANT_POINTS).all()
        for k in tc.TRAITS:  # the run values are the table's values of that run
            assert np.allclose(merged[f"{k}_run"], merged[k], equal_nan=True)
    for c in (then, drawn):
        assert c is not None and c["n"] >= tc.MIN_PLANT_POINTS
        assert np.hypot(c["x"], c["y"]).max() <= tc.CYLINDER_MM + 0.5 and c["z"].min() > 0
        assert c["z"][~c["stray"]].max() <= tc.axis_ranges(bed, plant)["z"][1]


@needs_table
@needs_clouds
def test_real_clouds_hide_floating_edge_pieces_but_not_top_leaves():
    """Checked by image: W2 G8 day 12 (15 h) has a piece of a neighbour's leaf floating at the crop wall 35 mm above
    the plant; W2 J5 day 11 (9 h) has one too (the stray of the earlier hero); the tall control W1 J1 day 4 (15 h)
    has a big leaf crossing the wall at its top, which is part of its canopy and must stay."""
    r = tc._table(*tc.table_key())

    def cloud_of(bed, plant, day, slot):
        row = r[(r["bed"] == bed) & (r["plant"] == plant) & (r["day"] == day) & (r["slot"] == slot)].iloc[0]
        f = tc._cloud_path(bed, plant, row["date"], row["capture_hour"])
        return tc._cloud_file(str(f), f.stat().st_mtime_ns)

    g8 = cloud_of("W2", "G8", 12, 15)
    assert 0 < g8["n_stray"] <= tc.STRAY_MAX_SHARE * g8["n"]
    assert g8["z"][g8["stray"]].min() > g8["z"][~g8["stray"]].max()  # what is hidden floats over what stays
    j5 = cloud_of("W2", "J5", 11, 9)
    assert j5["n_stray"] > 0 and j5["z"][j5["stray"]].min() > j5["z_max"]
    j1 = cloud_of("W1", "J1", 4, 15)
    assert j1["n_stray"] == 0 and j1["z_max"] == pytest.approx(float(j1["z"].max()))


# ---------------------------------------------------------------- which run is drawn, and THEN
def test_the_drawn_run_is_the_days_most_typical_never_an_odd_one(sandbox):
    """One run of the day reads a neighbour's leaf (twice the height): the picture is a typical run, and the run's
    own values come with it, so the card beside the picture describes the picture."""
    rows = experiment()
    for row in rows:
        if (row["bed"], row["plant"], row["day"]) == ("W2", "A2", 5) and row["hour"] == 12:
            row.update(height=2.0 * row["height"], volume=2.0 * row["volume"])  # the odd run, first in slot order
    write_table(sandbox, rows)
    for hour in tc.DAY_SLOTS:
        write_cloud(sandbox, "W2", "A2", 5, hour)
    d = tc.cloud_days("W2", "A2").set_index("day")
    assert d.loc[5, "slot"] == 15 and d.loc[5, "n_pass"] == 4  # 12 h is the odd one; the tie goes 15, 9, 18 h
    leaf = leaf_curve(5, True)
    assert d.loc[5, "height_p75_run"] == pytest.approx(leaf) and d.loc[5, "leaf_area_est_run"] == pytest.approx(leaf)
    assert d.loc[5, "leaf_area_est_run_pct"] == pytest.approx((leaf / 100.0 - 1) * 100)  # against its own baseline
    c = tc.plant_cloud("W2", "A2", 5)
    assert c["slot"] == 15 and c["height_p75_run_pct"] == pytest.approx(d.loc[5, "height_p75_run_pct"])
    assert tc.run_values("W2", "A2", 5, 12)["height_p75_run"] == pytest.approx(2.0 * leaf)  # any run, on request
    assert tc.run_values("W2", "A2", 5, 6) is None and tc.run_values("W2", "ZZ", 5, 12) is None
    assert set(tc.RUN_VALUES) <= set(d.columns) and d.loc[0, "reason"] == "no file" and np.isnan(d.loc[0, "z_max"])


def test_then_is_the_drawable_baseline_run_closest_to_the_baseline(sandbox):
    rows = experiment()
    for row in rows:
        if (row["bed"], row["plant"]) == ("W2", "A2") and row["day"] == -1 and row["hour"] == 12:
            row.update(leaf=30.0, height=30.0, volume=300.0)  # a QC-pass run that lost most of the plant
        if (row["bed"], row["plant"]) == ("W2", "A2") and row["day"] == -2 and row["hour"] == 9:
            row.update(leaf=104.0)  # close to the baseline, but its cloud is missing
    write_table(sandbox, rows)
    write_cloud(sandbox, "W2", "A2", -1, 12)
    write_cloud(sandbox, "W2", "A2", -3, 15)
    write_cloud(sandbox, "W2", "A2", -1, 18)
    t = tc.then_run("W2", "A2")
    assert (t["day"], t["slot"], t["in_baseline"]) == (-1, 18) + (True,)  # exact baseline values, latest day first
    assert t["leaf_area_est_run_pct"] == pytest.approx(0.0) and t["n_stray"] == 0
    assert tc.then_run("W2", "B3") is None  # no cloud before day 0
    rows = [r for r in experiment() if not (r["bed"] == "W2" and r["plant"] == "C4" and -4 <= r["day"] <= -1)]
    rows += [run_row("W2", "C4", -6, h, 100.0) for h in tc.DAY_SLOTS]  # no baseline day: the latest day before 0
    write_table(sandbox, rows)
    write_cloud(sandbox, "W2", "C4", -6, 12)
    t = tc.then_run("W2", "C4")
    assert (t["day"], t["slot"], t["in_baseline"]) == (-6, 12, False) and np.isnan(t["leaf_area_est_run_pct"])


def test_now_day_is_the_latest_drawable_day_without_water(sandbox):
    write_table(sandbox, experiment())
    assert tc.now_day("W2", "A2") == 16  # no clouds: the latest day with a QC-pass run (readouts still show)
    for day in (3, 10, 14):
        write_cloud(sandbox, "W2", "A2", day, 12)
    assert tc.now_day("W2", "A2") == 10  # day 14 is after re-watering
    assert tc.now_day("W2", "ZZ") is None


# ---------------------------------------------------------------- detached edge points (display only)
def blob(centre, n, spread, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(centre, float) + rng.uniform(-spread, spread, (n, 3))


def plant_body(seed=0, n=900):
    """A rosette: points on a dome inside the crop, up to 100 mm, densely sampled (spacing well under 12 mm)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 70, n)
    a = rng.uniform(0, 2 * np.pi, n)
    z = 100.0 * (1 - (r / 80.0) ** 2) + rng.uniform(0, 4, n)
    return np.c_[r * np.cos(a), r * np.sin(a), z]


def test_stray_mask_hides_floating_edge_pieces_and_specks_only():
    body = plant_body()
    edge = blob([82.0, 0.0, 150.0], 40, 3.0, 1)  # enters at the crop wall, floats 45 mm over the plant
    speck = blob([0.0, 30.0, 160.0], 4, 1.0, 2)  # a few points floating over the middle
    low = blob([82.0, 20.0, 20.0], 40, 3.0, 3)  # a piece at the wall, low down: it may be the plant's own leaf
    p = np.vstack([body, edge, speck, low])
    m = tc.stray_mask(p[:, 0], p[:, 1], p[:, 2])
    n_body = len(body)
    assert not m[:n_body].any() and m[n_body : n_body + 44].all() and not m[n_body + 44 :].any()
    big = blob([82.0, 0.0, 150.0], 150, 6.0, 4)  # over 10 % of the points: a real top leaf, kept
    p2 = np.vstack([body, big])
    assert not tc.stray_mask(p2[:, 0], p2[:, 1], p2[:, 2]).any()
    near = blob([75.0, 0.0, 110.0], 30, 2.0, 5)  # at the wall above the top but within 15 mm of it: kept
    p3 = np.vstack([body, near])
    assert not tc.stray_mask(p3[:, 0], p3[:, 1], p3[:, 2])[n_body:].any()
    pair = np.vstack([body, edge, blob([84.0, 8.0, 132.0], 3, 0.5, 6)])  # a speck beside the piece: both go
    assert tc.stray_mask(pair[:, 0], pair[:, 1], pair[:, 2])[n_body:].all()
    few = body[: tc.MIN_PLANT_POINTS - 1]
    assert not tc.stray_mask(few[:, 0], few[:, 1], few[:, 2]).any()  # too few points to judge


def test_hidden_points_leave_the_picture_and_the_side_view_not_the_numbers(sandbox):
    rows = experiment()
    write_table(sandbox, rows)
    path = write_cloud(sandbox, "W2", "A2", -1, 12, top_mm=100.0)
    with np.load(path) as z:
        xyz, plant = z["xyz"], z["plant"]
    piece = (blob([82.0, 0.0, 170.0], 20, 2.0, 7) / 1000.0).astype(np.float32)  # floats over the plant at the wall
    xyz = np.vstack([xyz, piece])
    plant = np.r_[plant, np.ones(len(piece), bool)]
    np.savez_compressed(path, xyz=xyz, rgb=np.zeros((len(xyz), 3), np.uint8), height=xyz[:, 2], plant=plant)
    tc.clear_caches()
    d = tc.cloud_days("W2", "A2").set_index("day")
    c = tc.plant_cloud("W2", "A2", -1)
    assert c["n"] == 620 and c["n_stray"] == 20 == d.loc[-1, "n_stray"] and c["stray"][-20:].all()
    assert d.loc[-1, "z_max"] == pytest.approx(100.0, abs=0.1)  # the frame and colour range ignore it
    assert tc.side_profile(c)["z"].max() == pytest.approx(float(c["z"][~c["stray"]].max()))
    assert tc.axis_ranges("W2", "A2")["z"][1] == 100.0
    assert d.loc[-1, "leaf_area_est_run"] == pytest.approx(100.0)  # the table's value, untouched


# ---------------------------------------------------------------- W3: shown plant by plant, never pooled
def test_w3_plants_are_readable_but_never_pooled(sandbox):
    rows = experiment()
    rows += [run_row("W3", "A2", d, h, 5.0 * leaf_curve(d, True)) for d in range(-4, 17) for h in tc.DAY_SLOTS]
    write_table(sandbox, rows)
    without = [r for r in rows if r["bed"] != "W3"]
    assert set(tc.runs()["bed"]) == {"W1", "W2"} and set(tc.runs(tc.SHOWN_BEDS)["bed"]) == {"W1", "W2", "W3"}
    assert set(tc.plant_days()["bed"]) == {"W1", "W2"} and "W3" in set(tc.plant_days(tc.SHOWN_BEDS)["bed"])
    assert set(tc.windows()["bed"]) == {"W1", "W2"} and "W3" in set(tc.windows(tc.SHOWN_BEDS)["bed"])
    pooled = tc.headline(), tc.gap("leaf_area_est")["by_day"]
    w3 = tc.plant_days(tc.SHOWN_BEDS).query("bed == 'W3'").set_index("day")
    assert w3.loc[10, "leaf_area_est_pct"] == pytest.approx(leaf_curve(10, True) - 100.0)  # its own baseline
    assert set(tc.structure_alerts()["bed"]) == {"W1", "W2"}
    write_table(sandbox, without)
    assert tc.headline() == pooled[0]  # W3 changes no bed number ...
    pd.testing.assert_frame_equal(tc.gap("leaf_area_est")["by_day"], pooled[1])  # ... and no band


def test_sampled_leaf_reads_the_rwc_table(sandbox):
    assert tc.sampled_leaf("W3", "A2") is None  # no table: nothing to say
    (sandbox / "derived" / "truth").mkdir(parents=True)
    t = pd.DataFrame({"bed": ["W3", "W3"], "plant": ["A2", "B3"], "date": pd.to_datetime(["2025-09-04", "2025-09-10"])})
    t.to_parquet(sandbox / "derived" / "truth" / "rwc_leaves.parquet")
    assert tc.sampled_leaf("W3", "A2") == 1 and tc.sampled_leaf("W3", "B3") == 7 and tc.sampled_leaf("W1", "A2") is None


@pytest.mark.skipif(not (ROOT / "derived" / "detect" / "onsets.parquet").exists(), reason="no detect onsets")
def test_onsets_info_says_when_the_onsets_are_older_than_the_table(sandbox):
    import os
    import shutil

    write_table(sandbox, experiment())
    assert tc.onsets_info() is None  # no detect run
    (sandbox / "derived" / "detect").mkdir(parents=True)
    path = sandbox / "derived" / "detect" / "onsets.parquet"
    shutil.copy(ROOT / "derived" / "detect" / "onsets.parquet", path)
    real = pd.read_parquet(path)
    leaf = real[(real["signal"] == "depth_leaf_area_est") & (real["group"] == "W2")].iloc[0]
    table = sandbox / "derived" / "threed" / "traits_runs.parquet"
    t0 = table.stat().st_mtime
    os.utime(path, (t0 + 5, t0 + 5))  # written together (a fresh checkout): current
    tc.clear_caches()
    info = tc.onsets_info()
    assert (info["n_deficit"], info["n_control"]) == (int(leaf["n_deficit"]), int(leaf["n_control"]))
    assert (info["table_deficit"], info["table_control"]) == (3, 2) and not info["stale"]
    os.utime(path, (t0 - 3600, t0 - 3600))  # the 3D table was re-run an hour after detect
    tc.clear_caches()
    assert tc.onsets_info()["stale"]
