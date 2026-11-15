"""The shared contract and its synthetic data agree with each other."""

from datetime import date

import numpy as np

from uiwue.threed import schema


def test_synthetic_runs_match_contract():
    df = schema.synthetic_runs()
    schema.validate_runs(df)
    assert set(df["treatment"]) == {"control", "deficit"}
    assert df.loc[df["date"] == schema.CUTOFF, "days_since_cutoff"].eq(0).all()


def test_synthetic_deficit_lowers_height_after_cutoff():
    df = schema.synthetic_runs()
    late = df[(df["date"] == date(2025, 9, 14)) & (df["hour"] == 15)]
    means = late.groupby("treatment")["height_p75"].mean()
    assert means["deficit"] < means["control"]


def test_synthetic_cloud_format():
    c = schema.synthetic_cloud()
    assert all(k in c for k in schema.CLOUD_KEYS)
    assert c["xyz"].dtype == np.float32 and c["rgb"].dtype == np.uint8
    assert np.allclose(c["height"], c["xyz"][:, 2])


def test_helpers():
    assert schema.folder_date("2025_09_08") == date(2025, 9, 8)
    assert schema.slot_of(16) == 15 and schema.slot_of(7) == 6
    assert schema.cloud_path("W2", "J5", date(2025, 9, 8), 15).name == "2025-09-08_15.npz"
