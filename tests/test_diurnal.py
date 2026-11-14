"""Alert rule, threshold tuning and the thermal and environment helpers, on synthetic data."""

import numpy as np
import pandas as pd

from uiwue.core import add_design
from uiwue.diurnal import alert_days, status, tune
from uiwue.env import at_times, since_water, vpd_kpa
from uiwue.thermal import block_mean, leaf_t


def plants(values, bed="W2", plant="J5", hour=15, start="2025-08-24"):
    """One plant scanned once a day at `hour`, with the given score per day."""
    df = pd.DataFrame(
        {
            "bed": bed,
            "plant": plant,
            "hour": hour,
            "date": pd.date_range(start, periods=len(values)),
            "score": values,
            "diurnal_evidence": "ok",
        }
    )
    return add_design(df)


def test_irrigate_needs_two_scans_in_a_row():
    df = plants([0, 0, 0, 3, 0, 3, 3, 3], start="2025-09-01")  # days -2 .. 5
    st = status(df, "score", (15,), k=2)
    assert list(st) == ["fine", "fine", "fine", "watch", "fine", "watch", "irrigate", "irrigate"]
    assert alert_days(df, st).loc["W2_J5"] == 4  # first irrigate on 7 Sep = day 4


def test_weak_evidence_scans_are_skipped_not_counted():
    df = plants([3, 3, 3], start="2025-09-03")
    df.loc[1, "diurnal_evidence"] = "weak"
    st = status(df, "score", (15,), k=2)
    assert st.isna()[1] and list(st[[0, 2]]) == ["watch", "irrigate"]


def test_tuned_threshold_clears_every_healthy_alert():
    rng = np.random.default_rng(0)
    days = 30  # 24 Aug .. 22 Sep
    ctrl = [plants(rng.normal(0, 1, days), bed="W1", plant=p) for p in ("A2", "J5")]
    deficit = plants(np.r_[rng.normal(0, 1, 10), np.linspace(1, 8, 20)])
    df = pd.concat(ctrl + [deficit], ignore_index=True)
    k = tune(df, "score", (15,), "W2", ("W1_A2", "W1_J5"))
    st = status(df, "score", (15,), k)
    days_ = alert_days(df, st)
    assert days_[["W1_A2", "W1_J5"]].isna().all() and days_["W2_J5"] >= 0


def test_leaf_rule_and_downsampling():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert leaf_t(img) == np.median(np.arange(8))  # median of the coolest half
    assert block_mean(img, 2).shape == (2, 2) and block_mean(img, 2)[0, 0] == np.mean([0, 1, 4, 5])


def test_environment_helpers():
    assert abs(vpd_kpa(25.0, 50.0) - 1.584) < 0.01  # textbook value at 25 °C, 50 % RH
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-09-01 10:00", "2025-09-01 10:30"], utc=True))
    out = at_times(s, pd.to_datetime(["2025-09-01 10:28", "2025-09-01 12:00", "2025-09-01 10:01"], utc=True))
    assert out[0] == 2.0 and np.isnan(out[1]) and out[2] == 1.0
    # constant 1 kPa all day: demand since the 11:00 local watering grows 1 kPa·h per hour
    idx = pd.date_range("2025-09-01 00:00", "2025-09-01 23:58", freq="2min", tz="Europe/Berlin").tz_convert("UTC")
    t = pd.to_datetime(["2025-09-01 14:00"]).tz_localize("Europe/Berlin").tz_convert("UTC")
    demand, hours = since_water(t, "W1", pd.Series(1.0, index=idx))
    assert abs(demand[0] - 3) < 1e-6 and abs(hours[0] - 3) < 1e-6
