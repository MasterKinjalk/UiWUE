import numpy as np
import pandas as pd

from uiwue.detect.benchmark import cheapest_close, run_benchmark, stress_scores
from uiwue.detect.data import load_merged, trait_columns
from uiwue.detect.onsets import run_onsets
from uiwue.detect.truth import recovery, truth_link


def test_load_merged_and_traits(fake_derived):
    df = load_merged()
    assert {"treatment", "days_since_cutoff", "truth_turgor_shift"} <= set(df.columns)
    assert df.duplicated(["bed", "plant", "date", "hour"]).sum() == 0
    cols = trait_columns(df)
    assert "thermal_leaf_air_dt" in cols and "rgb_qc_pass" not in cols
    assert not any(c.startswith("truth_") for c in cols)


def test_run_onsets(fake_derived):
    df = load_merged()
    auc, on = run_onsets(df, n_boot=200)
    o = on.set_index(["signal", "group"])
    assert 3 <= o.loc[("thermal_leaf_air_dt", "all"), "onset_day"] <= 6
    assert np.isnan(o.loc[("hsi_ndvi", "all"), "onset_day"])
    assert o.loc[("rgb_bedfx", "all"), "bed_effect"]
    assert o.loc[("depth_projected_area", "all"), "is_size_baseline"]
    # thermal warns before the synthetic symptom day (9) and before RGB
    assert o.loc[("thermal_leaf_air_dt", "all"), "lead_vs_symptom"] > 0
    # the pre-cutoff null: no synthetic day effect, so the rule stays quiet before day 0
    th = o.loc[("thermal_leaf_air_dt", "all")]
    assert not th["fires_before_cutoff"] and th["robust_onset_day"] >= th["onset_day"]
    assert th["null_auc_max"] < 0.9
    # symptom anchor: imaged deficit plants only, censored to (previous note, note]
    assert th["symptom_day"] == 9 and th["symptom_day_low"] == 7
    assert th["n_symptom"] == 22 and th["n_imaged"] == 22
    assert set(auc["group"]) == {"all", "W2", "W3"}


def test_benchmark_and_scores(fake_derived):
    df = load_merged()
    bench, curves = run_benchmark(df, models=("logistic",), groups=("all",), n_boot=100)
    assert bench["is_size_baseline"].any()
    assert set(bench["plant_set"]) == {"own", "common"}
    b = bench[bench["plant_set"] == "own"].set_index("sensors")
    assert b.loc["thermal", "early_auc"] > 0.8
    assert b.loc["hsi", "early_auc"] < 0.8
    assert len(b) == 1 + 15  # size + every combination of 4 sensors
    assert {"robust_onset_day", "null_auc_max", "fires_before_cutoff"} <= set(bench.columns)
    assert (bench["robust_onset_day"].fillna(99) >= bench["onset_day"].fillna(99)).all()
    best = cheapest_close(bench)
    assert "thermal" in best["sensors"]
    assert curves["signal"].str.startswith("model:").all()

    s = stress_scores(df)
    assert {"bed", "plant", "date", "hour", "detect_stress_prob"} <= set(s.columns)
    late = s[pd.to_datetime(s["date"]) == "2025-09-12"]
    assert (
        late.loc[late["bed"] != "W1", "detect_stress_prob"].mean()
        > late.loc[late["bed"] == "W1", "detect_stress_prob"].mean()
    )


def test_truth_and_recovery(fake_derived):
    df = load_merged()
    t = truth_link(df).set_index("signal")
    assert t.loc["thermal_leaf_air_dt", "rho_pooled"] > 0.5
    r = recovery(df)
    assert {"plant_id", "recovery_ratio", "measurable", "signal"} <= set(r.columns)
    assert "bed" not in r.columns  # a result table, so the validator's key rules do not apply
    assert r["plant_id"].str[:2].isin(["W2", "W3"]).all()


def test_low_res_thermal_is_a_substitute_setup():
    from uiwue.detect.benchmark import full_resolution, sensor_sets
    from uiwue.detect.data import cost_rank

    cols = ["rgb_exg", "thermal_leaf_t", "thermal_leaf_t_r160", "thermal_leaf_t_r80"]
    sets = sensor_sets(cols)
    assert sets[("thermal",)] == ["thermal_leaf_t"]
    assert sets[("thermal_r80",)] == ["thermal_leaf_t_r80"]
    assert sets[("rgb", "thermal_r160")] == ["rgb_exg", "thermal_leaf_t_r160"]
    assert ("thermal", "thermal_r80") not in sets
    assert len(sets) == 1 + 3 + 3  # rgb; three thermal cameras alone; each with rgb
    assert cost_rank(["thermal_r80"]) < cost_rank(["thermal_r160"]) < cost_rank(["thermal"])
    assert full_resolution(cols) == ["rgb_exg", "thermal_leaf_t"]


def test_symptom_anchor_uses_imaged_plants_only():
    from uiwue.detect.onsets import anchors

    sym = pd.DataFrame(
        {
            "bed": ["W2", "W2", "W3", "W3", "W1"],
            "plant": ["A1", "A2", "B1", "B2", "C1"],
            "symptom_day": [3.0, np.nan, 9.0, 9.0, np.nan],
            "symptom_day_low": [-1.0, np.nan, 7.0, 7.0, np.nan],
            "clip_day": [4.0, 6.0, np.nan, 5.0, np.nan],
        }
    )
    imaged = pd.DataFrame({"bed": ["W2", "W2", "W3", "W1"], "plant": ["A1", "A2", "B1", "C1"]})
    a = anchors(sym, imaged, "all")
    assert a["n_imaged"] == 3 and a["n_symptom"] == 2  # W3 B2 is not imaged
    assert a["symptom_day"] == 6 and a["symptom_first_day"] == 3
    assert a["clip_day"] == 5 and a["n_clip"] == 2
    w2 = anchors(sym, imaged, "W2")
    assert w2["symptom_day"] == 3 and w2["symptom_day_low"] == -1 and w2["n_symptom"] == 1


def test_benchmark_scores_each_setup_on_plants_with_all_its_sensors(fake_derived):
    df = load_merged()
    df.loc[df["bed"] == "W3", ["rgb_green_cover", "rgb_bedfx"]] = np.nan  # W3 has no RGB
    df = df.drop(columns=["hsi_ndvi"])
    bench, _ = run_benchmark(df, models=("logistic",), groups=("all", "W3"), n_boot=50)
    own = bench[(bench["plant_set"] == "own") & (bench["group"] == "all")].set_index("sensors")
    assert own.loc["thermal", "n_deficit"] == 22
    assert own.loc["rgb", "n_deficit"] == 11 and own.loc["rgb+thermal", "n_deficit"] == 11
    common = bench[(bench["plant_set"] == "common") & (bench["group"] == "all")]
    assert (common["n_deficit"] == 11).all() and (common["n_control"] == 7).all()
    assert not ((bench["plant_set"] == "common") & (bench["group"] == "W3")).any()
    pick = cheapest_close(bench)
    assert pick["plant_set"] == "common"
    assert cheapest_close(bench, plant_set="own")["plant_set"] == "own"


def test_boosted_model_ignores_missingness():
    from uiwue.detect.benchmark import make_model

    X = np.array([[np.nan, 1.0], [np.nan, 2.0], [0.0, 1.0], [0.0, 2.0]] * 5)
    y = np.array([1, 1, 0, 0] * 5)
    p = make_model("boosted").fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(p, p[0])  # a missing column alone must not separate the classes


def test_truth_link_deficit_only(fake_derived):
    t = truth_link(load_merged()).set_index("signal")
    r = t.loc["thermal_leaf_air_dt"]
    assert r["n_plants_deficit"] == 8 and r["reliable"]
    assert np.isfinite(r["rho_within_day_deficit"])


def test_slot_onsets(fake_derived):
    from uiwue.detect.onsets import run_slot_onsets

    s = run_slot_onsets(load_merged(), ["thermal_leaf_air_dt"], groups=("all",), hours=(12, 15), n_boot=100)
    assert list(s["hour"]) == [12, 15]
    assert s["onset_day"].between(2, 7).all()
    assert {"robust_onset_day", "fires_before_cutoff", "null_auc_max"} <= set(s.columns)
