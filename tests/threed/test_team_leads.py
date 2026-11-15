"""E3 step 3b: onsets of teammates' signals against the 3D traits on the same plants."""

import numpy as np

from uiwue.threed import experiments as ex
from uiwue.threed import schema


def _runs():
    return ex.prepare_runs(schema.synthetic_runs(plants_per_bed=8, seed=3))


def _team_table(runs, column, qc_column, source="projected_area"):
    t = runs[["bed", "plant", "date", "slot", source, "qc_pass"]].rename(
        columns={"slot": "hour", source: column, "qc_pass": qc_column}
    )
    return t


def test_team_runs_shape():
    runs = _runs()
    t = ex.team_runs(_team_table(runs, "rgb_green_cover", "rgb_qc_pass"), "rgb_green_cover", "rgb_qc_pass")
    for c in ("plant_id", "slot", "treatment", "days_since_cutoff", "qc_pass"):
        assert c in t
    assert set(t["treatment"]) <= {"control", "deficit"}


def test_same_signal_gives_zero_lead():
    """A teammate signal that is a copy of a 3D trait (same QC, same baseline change) must have
    the same onset as that trait on the same plants."""
    runs = _runs()
    tables = {"rgb": _team_table(runs, "rgb_green_cover", "rgb_qc_pass"), "thermal": None}
    lead = ex.e3_team(runs, ["projected_area"], None, tables)
    assert set(lead["team_signal"]) == {"rgb_green_cover"}
    row = lead.iloc[0]
    assert row["n_shared_plants"] == runs["plant_id"].nunique()
    if np.isfinite(row["onset_day"]):
        assert row["depth_minus_team_days"] == 0
    assert row["onset_day_unadjusted"] == row["team_onset_day_unadjusted"]


def test_missing_tables_give_empty_result():
    assert ex.e3_team(_runs(), ["projected_area"], None, {"thermal": None, "rgb": None}).empty
