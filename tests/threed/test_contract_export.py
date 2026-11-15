"""The 3D table converts to the team contract and passes uiwue.core.validate."""

import pandas as pd

from uiwue.core import CATALOGUE_COLUMNS
from uiwue.core.validate import validate
from uiwue.threed import schema
from uiwue.threed.contract import pick_slot_runs, to_team_table


def test_team_table_passes_validator(tmp_path):
    runs = schema.synthetic_runs()
    table, catalogue = to_team_table(runs)
    out = tmp_path / "threed"
    out.mkdir()
    table.to_parquet(out / "traits_runs.parquet", index=False)
    pd.DataFrame(catalogue, columns=CATALOGUE_COLUMNS).to_csv(out / "catalogue.csv", index=False)
    assert validate(out / "traits_runs.parquet") == []
    assert set(table.hour) <= {6, 9, 12, 15, 18}
    assert "depth_capture_hour" in table and "treatment" not in table


def test_rerun_rule_prefers_scheduled_folder_then_most_frames():
    base = dict(bed="W2", plant="J5", date=pd.Timestamp("2025-09-08").date())
    runs = pd.DataFrame(
        [
            {**base, "hour": 15, "n_frames_ok": 0, "n_frames_used": 0},  # scheduled but empty
            {**base, "hour": 16, "n_frames_ok": 4, "n_frames_used": 4},
            {**base, "hour": 14, "n_frames_ok": 6, "n_frames_used": 6},  # re-run with most frames
            {**base, "hour": 12, "n_frames_ok": 3, "n_frames_used": 3},  # scheduled with data
            {**base, "hour": 13, "n_frames_ok": 9, "n_frames_used": 9},
        ]
    )
    picked = pick_slot_runs(runs).set_index("slot_")["hour"].to_dict()
    assert picked == {15: 14, 12: 12}
