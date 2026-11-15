"""Shapes and mappings of the real environmental files (skipped where a file is absent)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from uiwue.threed import envdata

need_xlsx = pytest.mark.skipif(
    not envdata.METADATA_XLSX.exists(), reason="metadata workbook absent"
)
need_leaf = pytest.mark.skipif(not envdata.LEAF_READINGS.exists(), reason="leaf readings absent")
need_air = pytest.mark.skipif(not envdata.AIR_READINGS.exists(), reason="air readings absent")
need_ada = pytest.mark.skipif(not envdata.ADA_QC_JSON.exists(), reason="ADA QC labels absent")


@need_xlsx
def test_leaf_sensor_map():
    m = envdata.leaf_sensor_map().set_index("sensor_number")
    assert len(m) == 21
    assert (m.loc[1:9, "bed"] == "W3").all()
    assert (m.loc[10:18, "bed"] == "W2").all()
    assert (m.loc[19:21, "bed"] == "W1").all()
    assert tuple(m.loc[10]) == ("W2", "R7") and tuple(m.loc[20]) == ("W1", "J5")
    # cross-check with the protocol note of 22.9.25: W2 K2 (14), M2 (12), O6 (11), R7 (10)
    assert m.loc[14, "plant"] == "K2" and m.loc[12, "plant"] == "M2" and m.loc[11, "plant"] == "O6"


@need_xlsx
def test_air_sensor_map():
    m = envdata.air_sensor_map().set_index("sensor_number")["bed"].to_dict()
    assert m == {1: "W3", 2: "W1"}
    assert envdata.AIR_SENSOR_FOR_BED["W2"] == "W1"


@need_xlsx
@need_air
def test_air_and_vpd_and_sensor_mapping_check():
    air = envdata.load_air()
    assert set(air["bed"]) == {"W1", "W3"}
    assert air["utc_timestamp"].dt.tz is not None
    assert air["vpd_kpa"].between(0, 7).all()
    # The deficit bed W3 gets warmer and drier than W1 after the cutoff, not before: this is
    # what confirms that sensor 1 is W3.
    local = air["utc_timestamp"].dt.tz_convert("Europe/Berlin")
    mid = air[(local.dt.hour >= 11) & (local.dt.hour < 16)].assign(d=local.dt.date)
    rh = mid.pivot_table(index="d", columns="bed", values="rh_pct")
    gap = rh["W3"] - rh["W1"]
    before = gap[(gap.index >= date(2025, 8, 20)) & (gap.index <= date(2025, 9, 1))].mean()
    after = gap[(gap.index >= date(2025, 9, 7)) & (gap.index <= date(2025, 9, 14))].mean()
    assert before > -1 and after < -5


@need_xlsx
@need_leaf
def test_turgor_loader():
    t = envdata.load_turgor()
    assert t["sensor_number"].nunique() == 21
    assert t["plant_id"].notna().all()
    assert t["turgor_value"].between(1000, 2500).all()  # drop-outs and spikes removed
    assert t["utc_timestamp"].min() >= pd.Timestamp("2025-09-02", tz="UTC")
    w2 = t[t["bed"] == "W2"]["plant_id"].unique()
    assert len(w2) == 9 and "W2_R7" in w2
    # deficit clips rise (turgor falls) during the drought, control clips stay flat
    t["d"] = t["utc_timestamp"].dt.date
    med = t.groupby(["plant_id", "d"])["turgor_value"].median().unstack("d")
    rise = (med[date(2025, 9, 14)] - med[date(2025, 9, 3)]).groupby(lambda p: p[:2]).median()
    assert rise["W2"] > 5 and abs(rise["W1"]) < 3


@need_xlsx
def test_par_loader_is_utc_and_diurnal():
    par = envdata.load_par()
    assert len(par) > 4000
    assert str(par["utc_timestamp"].dt.tz) == "UTC"
    hourly = par.groupby(par["utc_timestamp"].dt.hour)["par_mv"].mean()
    assert hourly.loc[[0, 1, 2, 22, 23]].max() < 0.05  # night
    assert 9 <= int(hourly.idxmax()) <= 12  # midday in UTC (local clock 11-14 h)
    days = [date(2025, 9, 5), date(2025, 9, 6)]
    m = envdata.daily_window_mean(par, "par_mv", days)
    assert list(m.index) == days and (m > 0).all()


@need_xlsx
def test_bbch_and_first_symptoms():
    b = envdata.load_bbch()
    assert b.groupby(["bed", "plant"]).ngroups == 216
    assert sorted(b["date"].unique()) == [date(2025, 9, d) for d in (2, 6, 8, 10, 12, 15, 22)]
    row = b[(b["bed"] == "W1") & (b["plant"] == "A2") & (b["date"] == date(2025, 9, 2))]
    assert row["bbch"].iloc[0] == 15
    s = envdata.first_symptoms(b).set_index(["bed", "plant"])
    assert s.loc[("W2", "A6"), "first_symptom_date"] == date(2025, 9, 6)
    assert s.loc[("W2", "A8"), "first_symptom_date"] == date(2025, 9, 8)
    assert "W1" not in set(s.index.get_level_values("bed"))  # no drought symptom in control
    assert not envdata.is_drought_note("powdery mildew, aphids")
    assert envdata.is_drought_note("Wilting")


@need_ada
def test_ada_qc_labels():
    a = envdata.load_ada_qc()
    assert len(a) == 1860
    assert a["plant"].nunique() == 4 + 4 or a.groupby(["bed", "plant"]).ngroups == 8
    assert not a.duplicated(["bed", "plant", "date", "hour"]).any()
    # 2,059 Not Centered flags minus the 162 on thermal frames
    assert int(a["n_not_centred"].sum()) == 1897
    assert int(a["ada_no_plant"].sum()) == 1  # W2_G2, 2025-09-16, 09 h
    dark_by_hour = a.groupby("hour")["ada_dark"].mean()
    assert dark_by_hour.loc[6] > 0.7 and dark_by_hour.loc[12] < 0.05
    assert set(envdata.ADA_LABEL_FOR_FLAG) == {"qc_dark", "qc_not_centred", "qc_no_plant"}
    assert np.issubdtype(a["hour"].dtype, np.integer)
