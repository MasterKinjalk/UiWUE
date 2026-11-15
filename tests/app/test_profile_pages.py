"""Plant profile and Bed map: the figure and data helpers on synthetic tables, and the pages themselves
rendered in a fresh process against the real derived tables (a fresh process, because pages register
their callbacks globally and the shell is built once per interpreter)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import profile_views as pv

from uiwue.core import all_plants

TRAIT = "thermal_leaf_ref_dt"  # a real catalogue column, so the colour direction is looked up as in the app
DATES = pd.to_datetime(["2025-09-01", "2025-09-05", "2025-09-10"])


def traits(plants=("A2", "J5", "R7")):
    rows = [
        {"bed": bed, "plant": p, "date": d, "hour": h, TRAIT: (0.1 if bed == "W1" else 1.0) * (i + 1) + k}
        for bed in ("W1", "W2", "W3")
        for k, p in enumerate(plants)
        for i, d in enumerate(DATES)
        for h in (9, 12, 15)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def derived(tmp_path, monkeypatch):
    """An empty derived/ folder the helpers read their tables from."""
    monkeypatch.setattr("uiwue.core.DERIVED", tmp_path)
    monkeypatch.setattr(pv, "DERIVED", tmp_path)
    return tmp_path


def write(derived, module, name, df):
    (derived / module).mkdir(exist_ok=True)
    df.to_parquet(derived / module / f"{name}.parquet", index=False)


# ------------------------------------------------------------------ bed map
def test_every_pot_has_its_own_grid_position():
    pos = [pv.position(p) for p in all_plants()]
    assert len(set(pos)) == 72 and pv.position("A2") == (2, 0) and pv.position("R7") == (7, 17)
    assert all(1 <= x <= 8 and 0 <= y <= 17 for x, y in pos)


def test_bed_map_draws_three_beds_and_greys_pots_without_a_reading():
    fig = pv.bed_map_figure(traits(), TRAIT, DATES[1], 12, clips=frozenset({("W2", "A2")}))
    assert len(fig.data) == 6  # a grey layer and a value layer per bed
    for bed, (grey, coloured) in zip(("W1", "W2", "W3"), (fig.data[0:2], fig.data[2:4], fig.data[4:6])):
        assert len(coloured.x) == 3 and len(grey.x) == 72 - 3, bed
    assert list(fig.data[3].marker.line.color).count("#0b0b0b") == 1  # only the clip pot W2 A2 has a dark ring
    lo, hi = fig.layout.coloraxis.cmin, fig.layout.coloraxis.cmax
    assert lo < hi and "05 Sep 12 h" in fig.layout.title.text and "day 2" in fig.layout.title.text


def test_bed_map_with_no_runs_on_that_day_shows_only_grey_pots():
    fig = pv.bed_map_figure(traits(), TRAIT, pd.Timestamp("2025-09-20"), 15)
    assert all(len(fig.data[i].x) == 0 for i in (1, 3, 5)) and len(fig.data[0].x) == 72


# ------------------------------------------------------------------ plant profile
def test_daily_value_is_the_median_of_the_midday_runs():
    d = pv.daily(traits(), TRAIT)
    one = d[(d.bed == "W2") & (d.plant == "A2") & (d.date == DATES[0])]
    raw = traits()
    mid = raw[(raw.bed == "W2") & (raw.plant == "A2") & (raw.date == DATES[0]) & raw.hour.isin((12, 15))][TRAIT]
    assert len(one) == 1 and one[TRAIT].iloc[0] == mid.median() and one.day.iloc[0] == -2


def test_trace_figure_compares_with_the_beds_medians():
    df = traits()
    assert len(pv.trace_figure(df, "W2", "J5", TRAIT).data) == 3  # W1 median, W2 median, the plant
    assert len(pv.trace_figure(df, "W1", "J5", TRAIT).data) == 2
    assert pv.trace_figure(df, "W2", "Q8", TRAIT) is None


def test_turgor_figure_marks_clip_onset_and_first_symptom():
    turgor = pd.DataFrame(
        {
            "bed": "W3",
            "plant": "B3",
            "date": DATES.repeat(2),
            "hour": [12, 15] * 3,
            "truth_turgor_shift": np.arange(6.0),
        }
    )
    row = pd.Series({"truth_clip_onset_date": pd.Timestamp("2025-09-07"), "truth_first_symptom_date": pd.NaT})
    fig = pv.turgor_figure(turgor, "W3", "B3", row)
    texts = [a.text for a in fig.layout.annotations]
    assert "clip shift begins" in texts and "first symptom noted" not in texts
    assert pv.turgor_figure(turgor, "W3", "Q8", row) is None


def test_rwc_figure_is_for_sampled_w3_plants_only():
    rwc = pd.DataFrame(
        {
            "bed": "W3",
            "plant": ["D7", "D7", "Q6"],
            "date": pd.to_datetime(["2025-09-15"] * 2 + ["2025-09-04"]),
            "truth_rwc": [84.0, 83.0, 92.0],
            "truth_rwc_leaf2": [83.0, np.nan, np.nan],
        }
    )
    fig = pv.rwc_figure(rwc, "W3", "D7")
    assert len(fig.data) == 3 and list(fig.data[1].y) == [84.0, 83.0]
    assert (
        pv.rwc_figure(rwc, "W2", "D7") is None
        and pv.rwc_figure(rwc, "W3", "A2") is None
        and pv.rwc_figure(None, "W3", "D7") is None
    )


def test_spectrum_figure_draws_a_line_per_date_with_the_water_band_marked():
    wl = np.linspace(400, 1000, 300)
    dips = [0.0, 0.05, 0.12]  # the 970 nm dip deepens as the plant dries
    rows = [
        {
            "bed": "W2",
            "plant": "R7",
            "date": d,
            "hour": 15,
            "hsi_spectrum": list(0.5 - dip * np.exp(-(((wl - 970) / 15) ** 2))),
        }
        for d, dip in zip(DATES, dips)
    ]
    fig = pv.spectrum_figure(pd.DataFrame(rows), wl, "W2", "R7", [str(d.date()) for d in DATES[:2]])
    assert len(fig.data) == 2 and len(fig.layout.shapes) == 2 and fig.data[1].y[np.argmin(abs(wl - 970))] < 0.5
    assert pv.spectrum_figure(pd.DataFrame(rows), wl, "W2", "J5", ["2025-09-01"]) is None


def test_spectra_are_missing_until_the_spectral_module_pushes_them(derived):
    assert pv.load_spectra() is None
    write(
        derived,
        "spectral",
        "spectra_runs",
        pd.DataFrame(
            {"bed": ["W2"], "plant": ["R7"], "date": ["2025-09-01"], "hour": [15], "hsi_spectrum": [[0.1, 0.2]]}
        ),
    )
    assert pv.load_spectra() is None  # still no wavelengths
    (derived / "spectral" / "wavelengths.csv").write_text("wavelength_nm\n400\n401\n")
    spec, wl = pv.load_spectra()
    assert list(wl) == [400, 401] and len(spec) == 1


def test_plant_options_star_the_clip_plants(derived):
    write(
        derived,
        "truth",
        "plants",
        pd.DataFrame({"bed": ["W2", "W2"], "plant": ["R7", "A2"], "truth_has_clip": [True, False]}),
    )
    labels = {o["value"]: o["label"] for o in pv.plant_options(traits(), "W2")}
    assert labels == {"A2": "A2", "J5": "J5", "R7": "R7 ★"}


# ------------------------------------------------------------------ the pages, as the browser drives them
def test_pages_render_and_their_callbacks_run_on_the_real_tables():
    script = textwrap.dedent(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401  (the shell registers every page)
        pp, bm = sys.modules["pages.plant_profile"], sys.modules["pages.bed_map"]
        pp.layout(); bm.layout()
        for bed, plant in (("W2", "R7"), ("W3", "B3"), ("W1", "J5"), ("W2", "A2")):
            out = pp.show_plant(bed, plant, "thermal_leaf_ref_dt")
            assert len(out) == 4, out
        options, value = pp.pick_bed("W3")
        assert value in {o["value"] for o in options}
        pp.show_spectrum("W2", "R7", ["2025-09-08"])  # an alert now, a graph once the spectral tables are pushed
        fig = bm.show_map("thermal_leaf_ref_dt", 15, 20)
        assert len(fig.data) == 6
        print("pages ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0 and "pages ok" in done.stdout, done.stderr[-2000:]
