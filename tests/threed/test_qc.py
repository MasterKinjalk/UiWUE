"""Per-run aggregation and QC flags (plan 4.7, adapted)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from uiwue.threed import qc, schema


def frames(n=3, h75=(100.0, 102.0, 98.0), n_points=2000, clipped=0.02, used=True):
    rows = []
    for i in range(n):
        r = dict.fromkeys(schema.TRAITS, 1.0)
        r.update(
            frame=[4, 7, 10, 1, 13][i],
            height_p75=h75[i],
            n_points=n_points,
            used=used,
            ok=True,
            no_plant=False,
            clipped_fraction=clipped,
            outside_fraction=0.0,
            centroid_offset_mm=5.0,
            brightness=50.0,
            cyl_height_p75=h75[i],
        )
        rows.append(r)
    return pd.DataFrame(rows)


META = dict(
    bed="W1",
    plant="J5",
    date=date(2025, 9, 8),
    hour=15,
    slot=15,
    timestamp_utc=pd.Timestamp("2025-09-08 13:00", tz="UTC"),
    treatment="control",
    days_since_cutoff=5,
)


def test_median_mad_and_pass():
    row = qc.aggregate_run(frames(), META)
    assert row["height_p75"] == 100.0 and row["height_p75_mad"] == 2.0
    assert row["cyl_height_p75"] == 100.0 and row["cyl_height_p75_mad"] == 2.0
    assert row["n_frames_ok"] == 3 and row["frames_used"] == "4,7,10"
    assert row["qc_pass"] and not any(row[f] for f in schema.QC_FLAGS)


def test_blocking_flags():
    assert qc.aggregate_run(frames(n_points=100), META)["qc_few_points"]
    spread = qc.aggregate_run(frames(h75=(100.0, 120.0, 80.0)), META)
    assert spread["qc_high_spread"] and not spread["qc_pass"]
    small = qc.aggregate_run(frames(h75=(20.0, 26.0, 14.0)), META)  # 6 mm MAD: above 5 mm
    assert small["qc_high_spread"]
    tall = qc.aggregate_run(frames(h75=(200.0, 207.0, 193.0)), META)  # 7 mm < 5 % of 200 mm
    assert not tall["qc_high_spread"]
    assert qc.aggregate_run(frames(clipped=0.2), META)["qc_overlap"]
    f = frames()
    f["ok"], f["no_plant"] = False, True
    r = qc.aggregate_run(f, META)
    assert r["qc_no_plant"] and not r["qc_pass"] and r["n_frames_ok"] == 0


def test_non_blocking_flags_keep_pass():
    row = qc.aggregate_run(frames(), {**META, "qc_dark": True, "qc_ref_carried": True})
    assert row["qc_dark"] and row["qc_ref_carried"] and row["qc_pass"]


def test_unused_frames_are_ignored():
    f = pd.concat([frames(), frames(n=2, h75=(500.0, 500.0, 0.0), used=False)])
    assert qc.aggregate_run(f, META)["height_p75"] == 100.0


def test_runs_table_meets_contract():
    rows = [qc.aggregate_run(frames(), {**META, "hour": h, "slot": h}) for h in (6, 15)]
    df = qc.runs_table(rows)
    schema.validate_runs(df)
    assert str(df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert df["bed"].dtype == "string" and df["hour"].dtype == "int64"


def test_mad():
    assert qc.mad([1, 2, 3, 100]) == pytest.approx(1.0)
    assert np.isnan(qc.mad([np.nan]))
