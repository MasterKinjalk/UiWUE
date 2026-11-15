"""The statistics recover known effects from synthetic tables (plan section 7).

``schema.synthetic_runs`` has a deficit effect that starts on 2025-09-04 (one day after the
cutoff), a midday dip that deepens under stress, and a control growth of 4.3 mm/day x size.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from uiwue.threed import envdata, schema, stats
from uiwue.threed import experiments as ex


@pytest.fixture(scope="module")
def runs10():
    """10 plants per bed: enough for a Holm-corrected daily test to reach significance."""
    return ex.prepare_runs(schema.synthetic_runs(plants_per_bed=10, seed=1))


@pytest.fixture(scope="module")
def runs5():
    """5 plants per bed, the size of the Tier-1 subset."""
    return ex.prepare_runs(schema.synthetic_runs(plants_per_bed=5, seed=2))


def synthetic_frames(
    runs: pd.DataFrame, sigma: float = 4.0, column_offset: float = 30.0, seed: int = 0
) -> pd.DataFrame:
    """A traits_frames table: 15 frames per run on the 3 x 5 grid. Frame k is in grid column
    k % 3 (column 1 = centre, frames 1, 4, 7, 10, 13). Each frame = run value + a column offset
    (viewpoint: -offset, 0, +offset for columns 0, 1, 2) + Gaussian noise ``sigma`` (ToF noise),
    for the height traits; other traits get noise only. ``ok`` marks every frame as used."""
    rng = np.random.default_rng(seed)
    base = runs[schema.KEYS + list(schema.TRAITS)].reset_index(drop=True)
    rows = base.loc[base.index.repeat(15)].reset_index(drop=True)
    rows["frame"] = np.tile(np.arange(15), len(base))
    col = rows["frame"] % 3
    for t in schema.TRAITS:
        shift = (col - 1) * column_offset if t.startswith("height") else 0.0
        rows[t] = rows[t] + shift + rng.normal(0, sigma, len(rows))
    rows["ok"] = True
    return rows


# ---------------------------------------------------------------- helpers
def test_vpd_known_value():
    # es(25 degC) = 3.168 kPa; at 50 % RH the deficit is half of it
    assert stats.vpd(25.0, 50.0) == pytest.approx(1.584, abs=0.002)
    assert stats.vpd(20.0, 100.0) == pytest.approx(0.0)
    s = stats.vpd(pd.Series([10.0, 30.0]), pd.Series([80.0, 40.0]))
    assert isinstance(s, pd.Series) and (s > 0).all()


def test_holm_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    adj = stats.holm(p)
    # sorted 0.005, 0.01, 0.03, 0.04 with m = 4 -> 0.02, 0.03, 0.06, 0.06
    assert adj[3] == pytest.approx(0.02)
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[4])


def test_min_mw_p_limits_holm_with_five_plants():
    assert stats.min_mw_p(5, 5) == pytest.approx(2 / 252)
    # Holm over the 12 treatment days needs p < 0.05 / 12 on the first step: not reachable
    assert stats.min_mw_p(5, 5) > 0.05 / 12
    assert stats.min_mw_p(10, 10) < 0.05 / 12


def test_onset_day_needs_two_consecutive_days():
    days = [date(2025, 9, 3) + timedelta(days=i) for i in range(6)]
    daily = pd.DataFrame({"date": days, "p_holm": [0.01, 0.2, 0.01, 0.02, 0.01, 0.5]})
    assert stats.onset_day(daily) == date(2025, 9, 5)
    assert stats.onset_day(daily.assign(p_holm=0.5)) is None


def test_baseline_normalise_modes():
    df = pd.DataFrame(
        {
            "bed": ["W1"] * 3,
            "plant": ["A2"] * 3,
            "date": list(schema.BASELINE_DATES) + [date(2025, 9, 10)],
            "slot": [15, 15, 15],
            "h": [100.0, 110.0, 126.0],
        }
    )
    pct = stats.baseline_normalise(df, ["h"], mode="pct")
    assert pct["h"].tolist() == pytest.approx([-100 / 21, 100 / 21, 20.0])
    diff = stats.baseline_normalise(df, ["h"], mode="diff")
    assert diff["h"].iloc[-1] == pytest.approx(21.0)
    assert (pct["baseline_n"] == 2).all()
    no_base = stats.baseline_normalise(df.assign(plant=["A2", "A2", "B3"]), ["h"])
    assert np.isnan(no_base["h"].iloc[-1])


def test_value_near_takes_median_in_window():
    t = pd.Series(pd.date_range("2025-09-05 10:00", periods=7, freq="10min", tz="UTC"))
    v = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0])
    q = pd.Series(pd.to_datetime(["2025-09-05 10:30", "2025-09-06 10:30"], utc=True))
    out = stats.value_near(t, v, q, window_min=15)
    # 10:15-10:45 holds 10:20, 10:30, 10:40; the median ignores the spike
    assert out[0] == pytest.approx(5.0)
    assert np.isnan(out[1])


def test_rmcorr_ignores_between_plant_offsets():
    """Within each plant y falls with x, but plants with larger x have larger y. A pooled
    Pearson r is positive; the repeated-measures r must be strongly negative."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(6):
        x = np.arange(10) + 10 * k
        y = 50 * k - 0.8 * x + rng.normal(0, 0.5, 10)
        rows.append(pd.DataFrame({"x": x, "y": y, "s": f"p{k}"}))
    d = pd.concat(rows, ignore_index=True)
    assert np.corrcoef(d["x"], d["y"])[0, 1] > 0.5
    r = stats.rmcorr(d, "x", "y", "s")
    assert r["r"] < -0.9 and r["p_value"] < 1e-10
    assert r["dof"] == 60 - 6 - 1


# ---------------------------------------------------------------- E0a
def test_e0a_paper_model_recovers_slopes_and_passes(runs10):
    syn = ex.synthetic_inputs(seed=1, plants_per_bed=10)
    table, mi = ex.e0a(syn["runs"], syn["parity"])
    scored = table[table["pass"].notna()].set_index("quantity")
    assert mi["plant_id"].nunique() == 10  # Tier-1 plants only, as in the paper
    assert mi["date"].min() == pd.Timestamp("2025-09-03")
    # truth: control 4.3 x mean size (sizes ~ U(0.8, 1.2)); interaction about -13 - 1.5
    assert scored.loc["control_slope_mm_per_day", "ours"] == pytest.approx(4.3, rel=0.2)
    assert scored.loc["interaction_slope_mm_per_day", "ours"] == pytest.approx(-14.0, rel=0.2)
    assert scored["pass"].astype(bool).all()
    assert set(schema.RESULT_FILES["e0a_parity.csv"]) <= set(table.columns)


def test_e0a_fails_when_parity_is_wrong():
    syn = ex.synthetic_inputs(seed=1, plants_per_bed=5)
    bad = syn["parity"].copy()
    bad["height_parity_mm"] = bad["height_parity_mm"] * 0.5  # half the true slope
    table, _ = ex.e0a(None, bad)
    assert not table["pass"].astype(bool).any()


# ---------------------------------------------------------------- E0b
def test_e0b_perfect_and_flipped_flags(runs5):
    keys = runs5[schema.KEYS].iloc[:200].reset_index(drop=True)
    rng = np.random.default_rng(0)
    ada = keys.assign(
        n_rgb_dark=np.where(rng.random(200) < 0.3, 15, 0),
        n_rgb_medium_dark=0,
        n_not_centred=rng.integers(0, 3, 200),
        n_depth_not_centred=0,
        n_no_plant=0,
    )
    perfect = runs5.copy()
    j = perfect[schema.KEYS].reset_index().merge(ada, on=schema.KEYS).set_index("index")
    perfect["qc_dark"] = False
    perfect["qc_not_centred"] = False
    perfect.loc[j.index, "qc_dark"] = (j["n_rgb_dark"] >= 8).to_numpy()
    perfect.loc[j.index, "qc_not_centred"] = (j["n_not_centred"] >= 1).to_numpy()
    main, _var = ex.e0b(perfect, ada)
    m = main.set_index("flag")
    assert m.loc["qc_dark", "precision"] == 1 and m.loc["qc_dark", "recall"] == 1
    assert m.loc["qc_not_centred", "recall"] == 1
    assert m.loc["qc_dark", "n_runs"] == 200
    assert set(schema.RESULT_FILES["e0b_qc_agreement.csv"]) <= set(main.columns)
    flipped = perfect.copy()
    flipped["qc_dark"] = ~flipped["qc_dark"]
    m2 = ex.e0b(flipped, ada)[0].set_index("flag")
    assert m2.loc["qc_dark", "recall"] == 0


# ---------------------------------------------------------------- E1
def test_noise_floor_separates_viewpoint_from_tof_noise(runs5):
    sigma, offset = 4.0, 30.0
    frames = synthetic_frames(runs5.iloc[:400], sigma=sigma, column_offset=offset)
    nf = stats.noise_floor(frames, ["height_p75", "projected_area"]).set_index("trait")
    # within one column only the Gaussian noise is left: MAD of 5 normal draws ~ 0.6 sigma
    assert nf.loc["height_p75", "noise_within_column"] == pytest.approx(0.6 * sigma, rel=0.25)
    # across the whole grid the 30 mm column offsets dominate: an upper bound on noise
    assert nf.loc["height_p75", "noise"] > 5 * nf.loc["height_p75", "noise_within_column"]
    # traits without viewpoint offsets: both estimates agree
    assert nf.loc["projected_area", "noise"] == pytest.approx(nf.loc["projected_area", "noise_within_column"], rel=0.3)
    # if the pipeline used only the centre column, the two estimates coincide
    centre = frames.assign(ok=frames["frame"] % 3 == 1)
    nc = stats.noise_floor(centre, ["height_p75"]).set_index("trait")
    assert nc.loc["height_p75", "noise"] == pytest.approx(nc.loc["height_p75", "noise_within_column"])
    assert nc.loc["height_p75", "frames_per_run"] == 5


def test_e1_marks_traits_without_treatment_effect_unreliable(runs10):
    frames = synthetic_frames(runs10, sigma=2.0, column_offset=0.0)
    t = ex.e1(runs10, frames).set_index("trait")
    assert t.loc["height_p75", "reliable"] and t.loc["projected_area", "reliable"]
    # compactness and inclination_sd carry no treatment effect in the synthetic table
    assert not t.loc["compactness", "reliable"]
    assert t.loc["height_p75", "signal_date"] == date(2025, 9, 14)
    assert set(schema.RESULT_FILES["e1_noise.csv"]) <= set(t.reset_index().columns)
    fallback = ex.e1(runs10, None)
    assert fallback["noise_method"].str.contains("_mad").all()


# ---------------------------------------------------------------- E2
def test_e2_detects_a_pre_cutoff_bed_difference(runs10):
    traits = ["height_p75", "compactness"]
    shifted = runs10.copy()
    w2 = (shifted["bed"] == "W2") & (shifted["date"] <= ex.E2_WINDOW[1])
    day = np.array([(d - ex.E2_WINDOW[0]).days for d in shifted.loc[w2, "date"]])
    shifted.loc[w2, "compactness"] += 0.01 * day  # W2 drifts before the cutoff
    t = ex.e2(shifted, traits).set_index("trait")
    assert t.loc["compactness", "warning"]
    assert t.loc["compactness", "day_bed_coef"] == pytest.approx(0.01, rel=0.3)
    base = ex.e2(runs10, ["compactness"]).set_index("trait")
    assert abs(base.loc["compactness", "day_bed_coef"]) < 0.003
    assert set(schema.RESULT_FILES["e2_bed_baseline.csv"]) <= set(t.reset_index().columns)


# ---------------------------------------------------------------- E3
def test_e3_recovers_onset_and_interaction(runs10):
    onset, daily, _ = ex.e3(runs10, ["height_p75", "compactness"], symptoms=None)
    o = onset.set_index("trait")
    # the synthetic deficit effect starts on 2025-09-04 = day 1
    assert o.loc["height_p75", "onset_day"] in (1, 2)
    assert o.loc["height_p75", "slope_interaction"] < 0 and o.loc["height_p75", "p_value"] < 1e-6
    assert np.isnan(o.loc["compactness", "onset_day"])
    assert onset.iloc[0]["trait"] == "height_p75"  # ranked first
    # no written symptom table: bed-wide protocol date 2025-09-08
    assert o.loc["height_p75", "symptom_date"] == date(2025, 9, 8)
    assert o.loc["height_p75", "lead_vs_symptom_days"] == o.loc["height_p75", "onset_day"] - 5
    assert set(schema.RESULT_FILES["e3_onset.csv"]) <= set(onset.columns)
    assert set(schema.RESULT_FILES["e3_daily_tests.csv"]) <= set(daily.columns)


def test_e3_with_five_plants_per_bed_cannot_pass_holm(runs5):
    onset, _, _ = ex.e3(runs5, ["height_p75"], symptoms=None)
    row = onset.iloc[0]
    assert not row["holm_reachable"]
    assert np.isnan(row["onset_day"])
    assert row["onset_day_unadjusted"] in (1, 2)  # the effect is still there
    assert row["slope_interaction"] < 0 and row["p_value"] < 1e-6


def test_e3_uses_written_symptom_when_present(runs10):
    sym = pd.DataFrame(
        {
            "bed": ["W2"],
            "plant": ["A8"],
            "first_symptom_date": [date(2025, 9, 6)],
            "note": ["wilting"],
        }
    )
    onset, _, _ = ex.e3(runs10, ["height_p75"], symptoms=sym)
    assert onset.iloc[0]["symptom_date"] == date(2025, 9, 6)
    assert "W2_A8" in onset.iloc[0]["symptom_source"]


# ---------------------------------------------------------------- E4
def test_e4_detects_growing_midday_depression(runs10):
    days = [ex.E4_WINDOW[0] + timedelta(days=i) for i in range(25)]
    air = pd.DataFrame(
        {
            "utc_timestamp": pd.to_datetime([f"{d} 10:00" for d in days], utc=True),
            "bed": "W1",
            "vpd_kpa": np.random.default_rng(0).uniform(0.5, 2.5, len(days)),
        }
    )
    diurnal, models = ex.e4(runs10, ["height_p75"], air, None)
    assert set(schema.RESULT_FILES["e4_diurnal.csv"]) <= set(diurnal.columns)
    dep = diurnal.groupby(["treatment", "days_since_cutoff"])["midday_depression"].mean()
    assert dep.loc[("control", 5)] == pytest.approx(-3.0, abs=3.0)  # the fixed midday dip
    assert dep.loc[("deficit", 5)] < dep.loc[("control", 5)] - 4  # deeper under stress
    m = models.set_index("response").loc["midday_depression"]
    assert m["slope_interaction"] < 0 and m["p_value"] < 0.05
    assert 1 <= m["onset_day"] <= 6
    rec = diurnal.groupby(["treatment", "days_since_cutoff"])["overnight_recovery"].mean()
    assert rec.loc[("deficit", 3)] < rec.loc[("control", 3)]


def test_e4_skips_without_all_hours(runs10):
    only15 = runs10[runs10["slot"] == 15]
    diurnal, models = ex.e4(only15, ["height_p75"], None, None)
    assert diurnal.empty and models.empty


# ---------------------------------------------------------------- E5
def test_turgor_corr_recovers_coupling(runs10):
    turgor = ex.synthetic_turgor(runs10)
    res = stats.turgor_corr(runs10[runs10["slot"] == 15], turgor, ["height_p75", "compactness"])
    assert set(schema.RESULT_FILES["e5_turgor.csv"]) <= set(res.columns)
    pooled = res[(res["method"] == "rmcorr") & (res["lag_days"] == 0)].set_index("trait")
    assert pooled.loc["height_p75", "rho"] < -0.8  # height falls as the clip value rises
    assert abs(pooled.loc["compactness", "rho"]) < 0.3
    assert set(res["lag_days"]) == {-2, -1, 0, 1, 2}
    per = res[(res["method"] == "spearman") & (res["trait"] == "height_p75") & (res["lag_days"] == 0)]
    assert per.loc[per["bed"] == "W2", "rho"].median() < -0.8


def test_e5_names_best_proxy(runs10):
    turgor = ex.synthetic_turgor(runs10)
    _res, best = ex.e5(runs10, ["height_p75", "compactness"], turgor)
    assert best["trait"] == "height_p75" and best["significant"]
    assert best["r"] < -0.8


# ---------------------------------------------------------------- CLI
def test_cli_synthetic_writes_contract_files(tmp_path):
    rc = ex.main(["--synthetic", "--out", str(tmp_path)])
    assert rc == 0
    for name, cols in schema.RESULT_FILES.items():
        if name == "e0b_qc_agreement.csv" and not envdata.ADA_QC_JSON.exists():
            continue  # E0b needs the ADA authors' labels, which only an external/ada2025 clone has
        f = tmp_path / name
        assert f.exists(), name
        df = pd.read_csv(f)
        assert set(cols) <= set(df.columns), name
        assert (df["source"] == "synthetic").all()
    assert "SYNTHETIC" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "e3_trajectories.png").exists()
    assert b"\r\n" not in (tmp_path / "e3_onset.csv").read_bytes()


def test_cli_without_table_fails_cleanly(tmp_path):
    assert ex.main(["--runs", str(tmp_path / "missing.parquet"), "--out", str(tmp_path)]) == 2


def test_e5_single_clip_plant_falls_back_to_spearman(runs5):
    """Tier 1 holds one clip plant (W1_J5): no pooled correlation, a single-plant answer."""
    turgor = ex.synthetic_turgor(runs5)
    turgor = turgor[turgor["plant_id"] == "W2_J5"]
    res, best = ex.e5(runs5, ["height_p75"], turgor)
    assert set(res["method"]) == {"spearman"}
    assert best["basis"].startswith("single-plant") and best["r"] < -0.8


def test_e5_names_no_proxy_from_a_control_clip_alone(runs5):
    turgor = ex.synthetic_turgor(runs5)
    res, best = ex.e5(runs5, ["height_p75"], turgor[turgor["plant_id"] == "W1_J5"])
    assert len(res) and best == {}


def test_detrend_removes_each_plants_line():
    t = pd.Series(pd.date_range("2025-09-03", periods=10, freq="D", tz="UTC"))
    df = pd.DataFrame(
        {
            "plant_id": ["a"] * 10,
            "timestamp_utc": t,
            "y": 5.0 + 2.0 * np.arange(10) + np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 0]),
        }
    )
    out = stats.detrend_by(df, ["y"], "plant_id", "timestamp_utc")
    assert abs(np.polyfit(np.arange(10), out["y"], 1)[0]) < 1e-9


def test_e3_daytime_slots_and_e4_without_dark_runs(runs10):
    od, _dd, _ = ex.e3(runs10, ["height_p75"], symptoms=None, slots=ex.DAYTIME_SLOTS)
    assert od.iloc[0]["onset_day"] in (1, 2)
    diurnal, _models = ex.e4(runs10, ["height_p75"], None, None, exclude_dark=True)
    # synthetic 06 h runs are dark: without them there is no overnight recovery
    assert diurnal["overnight_recovery"].isna().all()
    assert diurnal["midday_depression"].notna().any()
