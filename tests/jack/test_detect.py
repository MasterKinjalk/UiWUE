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
    assert set(auc["group"]) == {"all", "W2", "W3"}


def test_benchmark_and_scores(fake_derived):
    df = load_merged()
    bench, curves = run_benchmark(df, models=("logistic",), groups=("all",), n_boot=100)
    assert bench["is_size_baseline"].any()
    b = bench.set_index("sensors")
    assert b.loc["thermal", "early_auc"] > 0.8
    assert b.loc["hsi", "early_auc"] < 0.8
    assert len(bench) == 1 + 15  # size + every combination of 4 sensors
    best = cheapest_close(bench)
    assert "thermal" in best["sensors"]
    assert curves["signal"].str.startswith("model:").all()

    s = stress_scores(df)
    assert {"bed", "plant", "date", "hour", "detect_stress_prob"} <= set(s.columns)
    late = s[pd.to_datetime(s["date"]) == "2025-09-12"]
    assert late.loc[late["bed"] != "W1", "detect_stress_prob"].mean() > late.loc[
        late["bed"] == "W1", "detect_stress_prob"].mean()


def test_truth_and_recovery(fake_derived):
    df = load_merged()
    t = truth_link(df).set_index("signal")
    assert t.loc["thermal_leaf_air_dt", "rho_pooled"] > 0.5
    r = recovery(df)
    assert {"recovery_ratio", "measurable", "signal"} <= set(r.columns)
    assert set(r["bed"]) <= {"W2", "W3"}
