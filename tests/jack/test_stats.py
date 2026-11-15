import numpy as np
import pandas as pd
import pytest

from uiwue.stats import auc_by_day, auc_ci, baseline_delta, onset_day, placebo


def test_auc_ci_extremes():
    auc, lo, _hi = auc_ci(np.arange(10) + 100, np.arange(7), n_boot=200)
    assert auc == 1.0 and lo == 1.0
    auc, _, _ = auc_ci(np.ones(5), np.ones(5), n_boot=0)
    assert auc == 0.5
    assert np.isnan(auc_ci([], [1.0])[0])


def test_baseline_delta_is_zero_median_in_window(runs):
    d = baseline_delta(runs, "thermal_leaf_air_dt")
    win = pd.to_datetime(runs["date"]).between("2025-08-30", "2025-09-02")
    med = d[win].groupby([runs["bed"], runs["plant"], runs["hour"]]).median()
    assert np.allclose(med, 0, atol=1e-9)


def test_auc_by_day_shape_and_orientation(runs):
    a = auc_by_day(runs, "rgb_green_cover", direction=-1, n_boot=200)
    assert set(a.columns) >= {"day", "auc", "ci_low", "ci_high", "n_deficit", "n_control"}
    assert a["day"].is_monotonic_increasing
    late = a[a["day"].between(10, 12)]
    assert (late["auc"] > 0.95).all()  # falling cover oriented as "stress up"
    assert (a["n_control"] == 7).all() and (a["n_deficit"] == 22).all()


def test_onset_finds_known_day(runs):
    a = auc_by_day(runs, "thermal_leaf_air_dt", direction=1, n_boot=500)
    day = onset_day(a)
    assert 3 <= day <= 6
    a3 = auc_by_day(runs, "thermal_leaf_air_dt", direction=1, group="W3", n_boot=500)
    assert onset_day(a3) <= day  # W3 starts a day earlier


def test_null_trait_has_no_onset(runs):
    a = auc_by_day(runs, "hsi_ndvi", direction=-1, n_boot=500)
    assert np.isnan(onset_day(a))


def test_onset_needs_two_consecutive_days():
    a = pd.DataFrame({"day": [0, 1, 2, 3, 4], "ci_low": [0.1, 0.8, 0.5, 0.8, 0.9]})
    assert onset_day(a) == 3
    a = pd.DataFrame({"day": [0, 1, 2], "ci_low": [0.1, 0.1, 0.9]})
    assert np.isnan(onset_day(a))


def test_placebo_flags_bed_effect(runs):
    assert placebo(runs, "rgb_bedfx", direction=1, n_boot=500)["bed_effect"]
    assert not placebo(runs, "hsi_ndvi", direction=-1, n_boot=500)["bed_effect"]
    assert not placebo(runs, "thermal_leaf_air_dt", direction=1, n_boot=500)["bed_effect"]


def test_bad_group(runs):
    with pytest.raises(ValueError):
        auc_by_day(runs, "hsi_ndvi", group="W9", n_boot=10)
