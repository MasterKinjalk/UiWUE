"""Alert rule, threshold tuning and the thermal and environment helpers, on synthetic data."""

import numpy as np
import pandas as pd

from uiwue.core import add_design
from uiwue.diurnal import alert_days, daily_status, pre_check, recommend, status, tune
from uiwue.env import at_times, dwd_frame, since_water, vpd_kpa
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


def test_alert_stays_latched_until_rewatering():
    # days -1 .. 15: alert on day 2, a cool scan on day 4, no usable scan on day 6, re-watered after day 12
    vals = [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    df = plants(vals, start="2025-09-02")
    df.loc[7, "diurnal_evidence"] = "weak"  # day 6
    ds = daily_status(df, status(df, "score", (15,), k=2)).set_index("day")
    assert ds.loc[1, "status"] == "watch" and ds.loc[2, "status"] == "irrigate" and ds.loc[2, "since"] == 2
    assert (ds.loc[2:12, "status"] == "irrigate").all() and (ds.loc[3:12, "since"] == 2).all()
    assert ds.loc[4, "raw"] == "fine" and ds.loc[4, "latched"]
    assert ds.loc[6, "raw"] == "no scan" and not ds.loc[6, "scanned"] and ds.loc[6, "status"] == "irrigate"
    assert ds.loc[13, "status"] == "fine" and ds.loc[13:, "since"].isna().all()  # latch cleared by re-watering


def test_no_scan_is_not_fine():
    df = plants([0, 0, 0], start="2025-09-05")
    df.loc[1, "diurnal_evidence"] = "weak"
    ds = daily_status(df, status(df, "score", (15,), k=2))
    assert list(ds.status) == ["fine", "no scan", "fine"] and list(ds.scanned) == [True, False, True]


def test_watered_day_check_catches_a_bed_wide_morning_effect():
    # watered days -10 .. -1: every deficit plant is hot on two sunny mornings with a cloudy day between,
    # so the two-in-a-row rule never fires, but the watered-day check still fails the schedule
    sunny = [3 if d in (-10, -8) else 0 for d in range(-10, 0)]
    df = pd.concat(
        [plants(sunny, plant=p, start="2025-08-24") for p in ("A2", "B3", "C4")]
        + [plants([0] * 10, bed="W1", plant=p, start="2025-08-24") for p in ("A2", "B3")],
        ignore_index=True,
    )
    st = status(df, "score", (15,), k=2)
    assert not (st == "irrigate").any()
    r = pre_check(df, "score", (15,), 2)
    assert r["pre_alert_plants"] == 0 and r["pre_worst_day"] == -10 and r["pre_worst_day_frac"] == 0.6
    assert abs(r["pre_above_rate"] - 6 / 50) < 1e-9 and not r["pre_pass"]
    assert pre_check(df, "score", (15,), 5)["pre_pass"]


def test_recommendation_skips_schedules_that_fail_the_watered_day_check():
    row = dict(rule="r", set="W2 tuning", n_detected=17, scans_per_day=1)
    sch = pd.DataFrame(
        [
            row | dict(schedule="09 h", median_alert_day=3.0, pre_pass=False),
            row | dict(schedule="12 + 15 h", median_alert_day=5.0, pre_pass=True, scans_per_day=2),
            row | dict(schedule="18 h", median_alert_day=8.0, pre_pass=True),
            row | dict(schedule="09 h", median_alert_day=3.0, pre_pass=False, set="W3 check"),
        ]
    )
    assert recommend(sch) == [False, True, False, False]
    assert recommend(sch.assign(pre_pass=False)) == [True, False, False, True]  # none pass: best regardless


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
    # DWD 10-min solar: 36 J/cm² over the 10 min ending 12:10 UTC = 600 W/m²; 0.1 h of sun = 60 % of the interval
    f = dwd_frame(pd.DataFrame({"MESS_DATUM": ["202509031210"], "GS_10": [36.0], "SD_10": [0.1]}))
    assert f.index[0] == pd.Timestamp("2025-09-03 12:05", tz="UTC")
    assert abs(f.global_rad.iloc[0] - 600) < 1e-9 and abs(f.sunshine.iloc[0] - 0.6) < 1e-9
