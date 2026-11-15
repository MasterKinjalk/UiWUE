"""Plant profile, Bed map and Export: the data and figure helpers on synthetic tables, and the pages themselves
rendered in a fresh process against the real derived tables (a fresh process, because pages register
their callbacks globally and the shell is built once per interpreter)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import profile_views as pv

from uiwue.core import all_plants

TRAIT = "thermal_leaf_ref_dt"  # real catalogue columns, so units, modes and directions are looked up as in the app
SIZE = "depth_projected_area"
DATES = pd.to_datetime(["2025-08-31", "2025-09-01", "2025-09-05", "2025-09-10"])


def traits(plants=("A2", "J5", "R7")):
    rows = [
        {
            "bed": bed,
            "plant": p,
            "date": d,
            "hour": h,
            TRAIT: (0.1 if bed == "W1" else 1.0) * (i + 1) + k,
            SIZE: 100.0 * (1 if bed == "W1" or i < 2 else 0.5),
        }
        for bed in ("W1", "W2", "W3")
        for k, p in enumerate(plants)
        for i, d in enumerate(DATES)
        for h in (9, 12, 15)
    ]
    df = pd.DataFrame(rows)
    df.loc[df.bed == "W3", SIZE] = float("nan")  # the depth camera did not image W3
    return df


@pytest.fixture
def derived(tmp_path, monkeypatch):
    """An empty derived/ folder the helpers read their tables from."""
    monkeypatch.setattr("uiwue.core.DERIVED", tmp_path)
    monkeypatch.setattr(pv, "DERIVED", tmp_path)
    return tmp_path


def write(derived, module, name, df):
    (derived / module).mkdir(exist_ok=True)
    df.to_parquet(derived / module / f"{name}.parquet", index=False)


# ------------------------------------------------------------------ labels and values
def test_blank_units_never_render_as_nan():
    assert pv.label_unit("rgb_exg") == "Greenness (ExG)" and pv.unit_text("rgb_exg") == ""
    assert pv.label_unit(TRAIT).endswith("(°C)")
    assert pv.fmt(-2.04, "°C", 1) == "−2.0 °C" and pv.fmt(None, "°C") == "–" and pv.fmt(12, "%", 0) == "+12 %"


def test_modes_follow_the_kind_of_trait():
    assert pv.mode_of(SIZE) == "pct" and pv.change_unit(SIZE) == "%"
    assert pv.mode_of(TRAIT) == "diff" and pv.mode_of("detect_stress_prob") == "raw"
    assert pv.mode_of("truth_turgor_shift") == "raw"


def test_daily_value_is_the_median_of_the_midday_runs():
    d = pv.daily(traits(), TRAIT)
    one = d[(d.bed == "W2") & (d.plant == "A2") & (d.date == DATES[0])]
    raw = traits()
    mid = raw[(raw.bed == "W2") & (raw.plant == "A2") & (raw.date == DATES[0]) & raw.hour.isin((12, 15))][TRAIT]
    assert len(one) == 1 and one[TRAIT].iloc[0] == mid.median() and one.day.iloc[0] == -3


def test_change_is_from_the_plants_own_baseline():
    s = pv.series(traits(), SIZE)
    w2 = s[(s.bed == "W2") & (s.plant == "J5")].set_index("day").chg
    assert w2[-3] == 0 and w2[2] == pytest.approx(-50)  # size: % of baseline
    t = pv.series(traits(), TRAIT)
    a = t[(t.bed == "W2") & (t.plant == "A2")].set_index("day").chg
    assert a[-3] == pytest.approx(-0.5) and a[7] == pytest.approx(2.5)  # difference from the baseline median 1.5


def test_sensor_states_keep_the_kinds_of_nothing_apart():
    df = traits()
    assert pv.sensor_state(df, "W2", "J5", SIZE)[0] == "ok"
    assert pv.sensor_state(df, "W3", "J5", SIZE)[0] == "notable"  # sensor has no rows for this plant
    assert pv.sensor_state(df, "W2", "J5", "hsi_ndwi")[0] == "notable"
    df.loc[(df.bed == "W2") & (df.plant == "J5"), SIZE] = float("nan")
    df["depth_qc_pass"] = False
    assert pv.sensor_state(df, "W2", "J5", SIZE)[0] == "noscan"  # in the table, no run passed QC


def test_small_multiple_draws_the_control_band_and_the_plant():
    fig = pv.small_multiple(traits(), "W2", "J5", TRAIT)
    names = [t.name for t in fig.data]
    assert names == ["control ±1 SD", "control", "W2 J5"] and fig.data[-1].line.color == pv.C["navy"]
    assert pv.small_multiple(traits(), "W3", "J5", SIZE) is None
    assert len(pv.detail_figure(traits(), "W2", "J5", TRAIT).data) == 5  # control and own deficit bed bands
    assert len(pv.detail_figure(traits(), "W1", "J5", TRAIT).data) == 3


def test_stress_summary_reports_the_peak_while_unwatered():
    df = traits()
    df["detect_stress_prob"] = df.date.map({DATES[0]: 0.1, DATES[1]: 0.2, DATES[2]: 0.9, DATES[3]: 0.4})
    s = pv.stress_summary(df, "W2", "J5")
    assert s["peak"] == (pytest.approx(0.9), 2) and s["now"] == (pytest.approx(0.4), 7)


def test_neighbours_walk_the_imaged_plants_in_grid_order():
    df = traits()
    assert pv.neighbours(df, "W1", "A2") == (("W3", "R7"), ("W1", "J5"))
    assert pv.neighbours(df, "W2", "Q8") == (None, None)


def test_plant_options_star_the_clip_plants(derived):
    write(
        derived,
        "truth",
        "plants",
        pd.DataFrame({"bed": ["W2", "W2"], "plant": ["R7", "A2"], "truth_has_clip": [True, False]}),
    )
    labels = {o["value"]: o["label"] for o in pv.plant_options(traits(), "W2")}
    assert labels == {"A2": "A2", "J5": "J5", "R7": "R7 ★"}


def test_rwc_says_when_the_leaves_came_from_other_plants(derived):
    assert pv.rwc_state("W3", "B3")[0] == "–"
    write(
        derived,
        "truth",
        "rwc_leaves",
        pd.DataFrame({"bed": ["W3"], "plant": ["A2"], "date": ["2025-09-04"], "truth_rwc": [95.0]}),
    )
    value, hover = pv.rwc_state("W3", "B3")
    assert value == "–" and "other plants" in hover
    assert pv.rwc_state("W3", "A2")[0] == "95 %"


# ------------------------------------------------------------------ bed map
def test_every_pot_has_its_own_grid_position():
    pos = [pv.position(p) for p in all_plants()]
    assert len(set(pos)) == 72 and pv.position("A2") == (2, 0) and pv.position("R7") == (7, 17)


def test_bed_map_has_three_distinct_kinds_of_nothing():
    df = traits()
    w3 = pv.pot_states(df, SIZE, -3, 12, "W3").set_index("plant").state
    assert w3["J5"] == "notable" and w3["B1"] == "notimaged"
    w2 = pv.pot_states(df, SIZE, -3, 12, "W2").set_index("plant").state
    assert (w2 == "value").sum() == 3
    assert pv.pot_states(df, SIZE, 5, 12, "W2").set_index("plant").state["J5"] == "noscan"  # no run that day
    fig = pv.bed_figure(df, SIZE, -3, 12, "W2", selected=("W2", "J5"))
    symbols = {t.name: t.marker.symbol for t in fig.data}
    assert symbols == {
        "not imaged": "circle",
        "no table": "square-open",
        "no scan": "circle-open-dot",
        "value": "square",
    }
    value = fig.data[-1]
    assert list(value.marker.line.color).count(pv.C["navy"]) == 1 and "%" in value.customdata[0][2]
    assert len(pv.bed_figure(df, SIZE, -3, 12, "W2", imaged_only=True).data) == 3


def test_ramp_puts_amber_on_the_stressed_side():
    assert pv.ramp_scale(TRAIT)[-1][1] == pv.C["amber-900"]  # warmer leaf = stressed
    assert pv.ramp_scale(SIZE)[0][1] == pv.C["amber-900"]  # smaller plant = stressed


# ------------------------------------------------------------------ export
def test_export_filters_rows_and_sensor_columns():
    df = traits().assign(treatment=lambda d: d.bed.map({"W1": "control"}).fillna("deficit"))
    df["days_since_cutoff"] = (df.date - pd.Timestamp("2025-09-03")).dt.days
    t = pv.filter_export(df, ["W2", "W3"], "deficit", ["thermal_"], (0, 12))
    assert set(t.bed) == {"W2", "W3"} and t.days_since_cutoff.between(0, 12).all()
    assert TRAIT in t and SIZE not in t and "bed" in t
    assert pv.export_counts(t) == (len(t), 6, 2)
    d = pv.data_dictionary(t.columns)
    assert list(d.columns) == ["column", "label", "unit", "direction", "source", "description"]
    assert d.set_index("column").loc[TRAIT, "unit"] == "°C"


# ------------------------------------------------------------------ the pages, as the browser drives them
def test_pages_render_and_their_callbacks_run_on_the_real_tables():
    script = textwrap.dedent(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401  (the shell registers every page)
        pp, bm, ex, home = (sys.modules[f"pages.{n}"] for n in ("plant_profile", "bed_map", "export", "home"))
        home.layout(); bm.layout(); ex.layout()
        pp.layout(); pp.layout(bed="W3", plant="B3"); pp.layout(bed="nope", plant="nope")
        for bed, plant in (("W2", "R7"), ("W3", "B3"), ("W1", "J5"), ("W1", "F5")):
            assert len(pp.show_plant(bed, plant)) == 6
            pp.show_detail(bed, plant, "depth_projected_area")
            assert pp.download(1, bed, plant)["content"].count("\\n") > 50
        options, value = pp.pick_bed("W3", "B3")
        assert value == "B3"  # a deep-linked plant survives the initial bed callback
        options, value = pp.pick_bed("W3", "R7")
        assert value in {o["value"] for o in options}
        out = bm.show_map("thermal_leaf_ref_dt", 15, 9, False, {"bed": "W2", "plant": "R7"})
        assert len(out) == 8
        bm.show_side({"bed": "W3", "plant": "B3"}, "depth_projected_area", 15, 9)
        for key in ("traits", "detect/scores_runs"):
            assert ex.download_csv(1, key, ["W1", "W2", "W3"], "all", ["thermal_", "detect_"], [-10, 19])["content"]
        print("pages ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=180, check=False
    )
    assert done.returncode == 0 and "pages ok" in done.stdout, done.stderr[-2000:]
