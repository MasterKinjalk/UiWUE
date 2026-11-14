"""The shared contract (Master-Plan section 6): every derived table passes the validator,
and the core helpers that every module relies on behave as documented."""

import pandas as pd
import pytest

from uiwue.core import DERIVED, all_plants, load_traits, slot_for_hour
from uiwue.core.validate import validate


@pytest.mark.parametrize("path", sorted(DERIVED.glob("*/*.parquet")), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_derived_table_follows_contract(path):
    assert validate(path) == []


def test_slot_mapping():
    assert [slot_for_hour(h) for h in (6, 7, 8, 10, 11, 13, 14, 16, 19, 20)] == [6, 6, 9, 9, 12, 12, 15, 15, 18, None]


def test_grid_has_72_checkerboard_plants():
    plants = all_plants()
    assert len(plants) == 72 and plants[:4] == ["A2", "A4", "A6", "A8"] and plants[4] == "B1"


def test_validator_catches_contract_breaks(tmp_path):
    bad = pd.DataFrame(
        {
            "bed": ["W1", "W1"],
            "plant": ["J5", "J5"],
            "date": ["2025-09-01"] * 2,
            "hour": [16, 16],
            "treatment": ["control"] * 2,
            "leaf": [1.0, 2.0],
        }
    )
    (tmp_path / "m").mkdir()
    bad.to_parquet(tmp_path / "m" / "t.parquet")
    problems = " | ".join(validate(tmp_path / "m" / "t.parquet"))
    for expected in ("duplicate", "scheduled slot", "design columns", "module prefix", "catalogue"):
        assert expected in problems


def test_merged_table_has_one_row_per_run():
    df = load_traits()
    if df is None:
        pytest.skip("no derived tables yet")
    assert not df.duplicated(["bed", "plant", "date", "hour"]).any()
    assert {"treatment", "days_since_cutoff", "phase"} <= set(df.columns)
