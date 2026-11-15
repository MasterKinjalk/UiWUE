"""The 3D phenotyping pages (plan section 8): data layer, callbacks, and rendering through Flask.

The data layer is tested on synthetic data (``UIWUE_SYNTHETIC=1``) and on a temporary repository
root holding real-format files. Callbacks are called directly with the values their pages lay
out; the smoke tests render every page through the Flask test client and fire each page's
initial callbacks over HTTP, as the browser does.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from contextvars import copy_context
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import threed_data as td

from uiwue.threed import schema

PAGES = ["/", "/threed-browser", "/threed-compare", "/threed-export"]
DAY = date(2025, 9, 8)


# ---------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("UIWUE_SYNTHETIC", "1")
    monkeypatch.delenv("UIWUE_ROOT", raising=False)
    td.clear_caches()
    yield
    td.clear_caches()


@pytest.fixture
def real_root(tmp_path, monkeypatch):
    """A repository root with real-format files: trait table, one large cloud with temperature,
    an RGB frame, E1/E3/E5 results, a sensor sheet and leaf readings."""
    monkeypatch.delenv("UIWUE_SYNTHETIC", raising=False)
    monkeypatch.setenv("UIWUE_ROOT", str(tmp_path))
    (tmp_path / schema.RESULTS_DIR).mkdir(parents=True)
    runs = schema.synthetic_runs(plants_per_bed=3, seed=1)
    runs.to_parquet(tmp_path / schema.TRAITS_RUNS)

    cloud = schema.synthetic_cloud(n=60_000, seed=3)
    temp = np.where(cloud["plant"], 24.0, np.nan).astype(np.float32)
    path = tmp_path / schema.cloud_path("W2", "J5", DAY, 15)
    path.parent.mkdir(parents=True)
    np.savez(path, **cloud, temp=temp)

    from PIL import Image

    rgb = tmp_path / "data/rgb/W2_J5/2025_09_08/15/rgb_007.jpg"
    rgb.parent.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (60, 120, 50)).save(rgb)

    res = tmp_path / schema.RESULTS_DIR
    pd.DataFrame(
        {
            "trait": list(schema.TRAITS),
            "noise": 1.0,
            "signal": 10.0,
            "snr": [10.0] * 10 + [1.5, 10.0],
            "reliable": [True] * 10 + [False, True],
        }
    ).to_csv(res / "e1_noise.csv", index=False)
    pd.DataFrame(
        [
            {
                "trait": "height_p75",
                "slope_interaction": -2.5,
                "ci_low": -3.1,
                "ci_high": -1.9,
                "p_value": 0.0004,
                "onset_day": 4,
                "lead_vs_symptom_days": -3,
                "unit": "% of baseline per day",
                "onset_day_unadjusted": 3,
            }
        ]
    ).to_csv(res / "e3_onset.csv", index=False)
    pd.DataFrame(
        [
            {
                "plant": "W2_J5",
                "bed": "W2",
                "trait": "height_p75",
                "lag_days": 0,
                "rho": -0.6,
                "p_value": 0.01,
                "n": 12,
                "method": "spearman",
            },
            {
                "plant": "pooled_W2",
                "bed": "W2",
                "trait": "height_p75",
                "lag_days": 0,
                "rho": -0.4,
                "p_value": 0.03,
                "n": 40,
                "method": "rmcorr",
            },
        ]
    ).to_csv(res / "e5_turgor.csv", index=False)

    sheet = pd.DataFrame(
        [
            ["Leaf Sensors", None, None],
            ["Sensor #", "Cable #", "Location"],
            [10, 10, "W2 J5"],
            [11, 11, "W3 Q8"],
            [None, None, None],
        ]
    )
    sheet.to_excel(tmp_path / td.SENSOR_SHEET, sheet_name="Sensor Guide", header=False, index=False)
    times = pd.date_range("2025-09-02 08:00", "2025-09-16 08:00", freq="10min", tz="UTC")
    pd.DataFrame(
        {
            "utc_timestamp": times,
            "sensor_number": "10",
            "turgic_value": np.linspace(100, 300, len(times)),
        }
    ).to_parquet(tmp_path / td.LEAF_READINGS)
    td.clear_caches()
    return tmp_path


@pytest.fixture(scope="session")
def dash_app():
    """The app shell with all pages, built once (pages register callbacks globally)."""
    old = os.environ.get("UIWUE_SYNTHETIC")
    os.environ["UIWUE_SYNTHETIC"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("uiwue_app_shell", APP_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("UIWUE_SYNTHETIC", None)
        else:
            os.environ["UIWUE_SYNTHETIC"] = old
    return module.app


def page(name: str):
    """The module Dash imported for ``app/pages/<name>.py``."""
    import dash

    return sys.modules[dash.page_registry[f"pages.{name}"]["module"]]


def with_trigger(prop_ids: list[str], fn, *args):
    """Call a callback that reads ``dash.ctx``, as if ``prop_ids`` had triggered it."""
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    def run():
        context_value.set(AttributeDict(triggered_inputs=[{"prop_id": p, "value": None} for p in prop_ids]))
        return fn(*args)

    return copy_context().run(run)


def traces(fig: go.Figure, kind: str) -> list:
    return [t for t in fig.data if t.type == kind]


# ---------------------------------------------------------------- data layer: synthetic
def test_synthetic_table_follows_contract():
    src = td.runs_source()
    assert src.synthetic and "UIWUE_SYNTHETIC" in src.reason
    df = td.load_runs()
    schema.validate_runs(df)
    assert {"plant_id", "day"} <= set(df.columns)
    assert isinstance(df["date"].iloc[0], date)
    assert td.load_runs() is df  # cached


def test_missing_table_falls_back_to_synthetic(tmp_path, monkeypatch):
    monkeypatch.delenv("UIWUE_SYNTHETIC")
    monkeypatch.setenv("UIWUE_ROOT", str(tmp_path))
    src = td.runs_source()
    assert src.synthetic and "does not exist" in src.reason
    assert len(td.load_runs()) > 0


def test_broken_table_falls_back_with_reason(tmp_path, monkeypatch):
    monkeypatch.delenv("UIWUE_SYNTHETIC")
    monkeypatch.setenv("UIWUE_ROOT", str(tmp_path))
    path = tmp_path / schema.TRAITS_RUNS
    path.parent.mkdir(parents=True)
    pd.DataFrame({"bed": ["W1"]}).to_parquet(path)
    src = td.runs_source()
    assert src.synthetic and "missing columns" in src.reason
    assert not td.load_runs().empty


def test_options_and_lookup():
    df = td.load_runs()
    assert td.beds(df) == ["W1", "W2"]
    assert "J5" in td.plants_in_bed(df, "W2")
    dates = td.run_dates(df, "W2", "J5")
    assert dates[0] < dates[-1] and DAY in dates
    assert td.run_hours(df, "W2", "J5", "2025-09-08") == list(schema.SLOTS)
    row = td.get_run(df, "W2", "J5", DAY, 15)
    assert row["treatment"] == "deficit" and row["days_since_cutoff"] == 5
    assert td.get_run(df, "W2", "J5", DAY, 7) is None
    assert td.nearest(dates, date(2025, 1, 1)) == dates[0]
    assert td.nearest([6, 9, 12], 10) == 9


def test_filter_runs():
    df = td.load_runs()
    f = td.filter_runs(df, beds=["W2"], start="2025-09-01", end="2025-09-02", hours=[15], qc_pass_only=True)
    assert set(f["bed"]) == {"W2"} and set(f["hour"]) == {15}
    assert f["date"].min() >= date(2025, 9, 1) and f["date"].max() <= date(2025, 9, 2)
    assert f["qc_pass"].all()
    assert td.filter_runs(df, beds=[]).empty
    assert len(td.filter_runs(df)) == len(df)
    assert set(td.filter_runs(df, plant_ids=["W1_J5"])["plant_id"]) == {"W1_J5"}


def test_synthetic_cloud_and_thumbnail_paths():
    c = td.load_cloud("W2", "J5", DAY, 15)
    assert c["source"] == "synthetic" and c["xyz"].dtype == np.float32
    assert len(c["xyz"]) <= td.MAX_POINTS
    assert td.load_cloud("W2", "J5", DAY, 15) is c
    row = td.get_run(td.load_runs(), "W2", "J5", DAY, 15)
    assert c["xyz"][:, 2].max() * 1000 == pytest.approx(row["height_p95"], rel=0.15)
    p = td.rgb_frame_path("W2", "J5", DAY, 6)
    assert p.as_posix().endswith("data/rgb/W2_J5/2025_09_08/06/rgb_007.jpg")


def test_synthetic_results_are_labelled():
    e1 = td.load_result("e1_noise.csv")
    assert e1.ok and e1.synthetic and "synthetic" in e1.note
    assert set(e1.df.columns) >= set(schema.RESULT_FILES["e1_noise.csv"])
    e3 = td.load_result("e3_onset.csv")
    assert e3.ok and set(e3.df["trait"]) == set(schema.TRAITS)
    assert e3.df.set_index("trait").loc["height_p75", "slope_interaction"] < 0
    assert not td.load_result("e5_turgor.csv").ok
    assert "compactness" in td.unreliable_traits()  # no treatment effect in the synthetic data
    with pytest.raises(KeyError):
        td.load_result("nope.csv")


def test_trajectories_baseline_and_qc():
    df = td.load_runs()
    _, s = td.trajectories(df, "height_p75", "treatment", normalise=True)
    base = s[s["day"].isin(pd.to_datetime(list(schema.BASELINE_DATES)))]
    assert base["mean"].abs().max() < 10  # % change is near zero on the baseline days
    late = s[s["day"] == pd.Timestamp("2025-09-14")].set_index("group")["mean"]
    assert late["deficit"] < late["control"]
    _, all_runs = td.trajectories(df, "height_p75", include_failed=True)
    _, passing = td.trajectories(df, "height_p75", include_failed=False)
    assert all_runs["n"].sum() >= passing["n"].sum()
    _, s = td.trajectories(df, "height_p75", "plants", plant_ids=["W1_J5", "W2_J5"])
    assert set(s["group"]) == {"W1_J5", "W2_J5"} and (s["n"] == 1).all()
    _, s = td.trajectories(df, "height_p75", "bed", start="2025-09-01", end="2025-09-05")
    assert s["day"].min() >= pd.Timestamp("2025-09-01") and set(s["group"]) == {"W1", "W2"}


def test_diurnal_matrix_and_effects():
    df = td.load_runs()
    m = td.diurnal_matrix(df, "height_p75", "treatment", include_failed=True)
    assert set(m) == {"control", "deficit"}
    assert list(m["control"].columns) == list(schema.SLOTS)
    assert m["control"].mean(axis=1).abs().max() < 1e-6  # deviations from the daily mean
    e = td.diurnal_effects(df, "height_p75")
    assert list(e.columns) == schema.RESULT_FILES["e4_diurnal.csv"]
    assert e["midday_depression"].notna().any() and e["overnight_recovery"].notna().any()


def test_export_frame_and_method_summary():
    df = td.load_runs()
    out = td.export_frame(df.head(5), ["height_p75"], include_mad=False)
    assert "height_p75" in out and "height_p75_mad" not in out and "plant_id" not in out
    assert out["date"].iloc[0] == df["date"].iloc[0].isoformat()
    text = td.method_summary(df, {"beds": "W1, W2"}, ["height_p75", "compactness"])
    assert "SYNTHETIC DATA" in text and "height_p75" in text and "qc_overlap" in text
    assert "docs/threed/PIPELINE.md" in text


# ---------------------------------------------------------------- data layer: real files
def test_real_files_are_used(real_root):
    src = td.runs_source()
    assert not src.synthetic and src.reason == schema.TRAITS_RUNS.as_posix()
    c = td.load_cloud("W2", "J5", DAY, 15)
    assert c["source"] == "file" and c["n_total"] == 60_000 and len(c["xyz"]) == td.MAX_POINTS
    assert c["temp"] is not None and len(c["temp"]) == td.MAX_POINTS
    # real clouds exist, but not this one: no made-up cloud
    assert td.load_cloud("W2", "J5", DAY, 9) is None
    uri = td.rgb_thumbnail("W2", "J5", DAY, 15)
    assert uri.startswith("data:image/jpeg;base64,") and len(uri) < 60_000
    assert td.rgb_thumbnail("W2", "J5", DAY, 9) is None
    assert td.unreliable_traits() == {"compactness"}
    row, res = td.result_row("e3_onset.csv", "height_p75")
    assert row["onset_day"] == 4 and "e3_onset.csv" in res.note and not res.synthetic
    assert not td.load_result("e4_diurnal.csv").ok
    assert "does not exist" in td.load_result("e4_diurnal.csv").note
    assert td.sensor_map() == {"10": "W2_J5", "11": "W3_Q8"}
    assert td.turgor_plants(td.load_runs()) == ["W2_J5"]
    t = td.load_turgor("W2_J5")
    assert list(t.columns) == ["time_utc", "turgor"] and len(t) > 100
    assert td.load_turgor("W1_J5") is None


def test_result_with_missing_columns(real_root):
    pd.DataFrame({"trait": ["height_p75"]}).to_csv(real_root / schema.RESULTS_DIR / "e2_bed_baseline.csv", index=False)
    res = td.load_result("e2_bed_baseline.csv")
    assert not res.ok and "lacks the columns" in res.note


def test_pipeline_parameters_section(real_root):
    doc = real_root / td.PIPELINE_DOC
    doc.parent.mkdir(parents=True)
    doc.write_text("# Pipeline\n\nintro\n\n## Parameters\n\n- radius 12 mm\n\n## Other\n\nx\n")
    text = td.method_summary(td.load_runs(), {})
    assert "radius 12 mm" in text and "## Other" not in text and "intro" not in text
    assert "e1_noise.csv" in text


# ---------------------------------------------------------------- 3D Browser callbacks
def by_id(component) -> dict:
    return {c.id: c for c in component._traverse() if isinstance(getattr(c, "id", None), str)}


def test_browser_layout_from_query(dash_app):
    b = page("threed_browser")
    ids = by_id(b.layout(bed="W1", plant="A8", date="2025-09-03", hour="12"))
    assert ids["tb-bed"].value == "W1" and ids["tb-plant"].value == "A8"
    assert ids["tb-dates"].data[ids["tb-date"].value] == "2025-09-03"
    assert ids["tb-hour"].value == 12 and ids["tb-date2"].value == "2025-09-03"
    ids = by_id(b.layout())  # defaults: W2_J5 on the last drought day at 15 h
    assert (ids["tb-bed"].value, ids["tb-plant"].value, ids["tb-hour"].value) == ("W2", "J5", 15)
    assert ids["tb-dates"].data[ids["tb-date"].value] == "2025-09-14"
    options, value = b.update_plants("W1", "A8")
    assert value == "A8" and "J5" in options
    options, value = b.update_plants("W1", "ZZ")
    assert value == options[0]


def test_browser_dates_keep_and_click(dash_app):
    b = page("threed_browser")
    dates = [d.isoformat() for d in td.run_dates(td.load_runs(), "W2", "J5")]
    idx = dates.index("2025-09-08")
    out = with_trigger(["tb-plant.value"], b.update_dates, "W2", "J5", None, idx, dates, "2025-09-03")
    mx, marks, value, _store, opts, d2 = out
    assert dates[value] == "2025-09-08" and mx == len(dates) - 1 and d2 == "2025-09-03"
    assert any("off" in str(m) for m in marks.values())
    click = {"points": [{"x": "2025-08-25", "customdata": ["2025-08-25", 9]}]}
    out = with_trigger(["tb-series.clickData"], b.update_dates, "W2", "J5", click, idx, dates, "2025-09-03")
    assert out[3][out[2]] == "2025-08-25"
    opts, hour = with_trigger(["tb-series.clickData"], b.update_hours, out[2], out[3], click, "W2", "J5", 15)
    assert hour == 9 and [o["value"] for o in opts] == list(schema.SLOTS)
    opts, hour = with_trigger(["tb-date.value"], b.update_hours, idx, dates, None, "W2", "J5", 12)
    assert hour == 12


def test_browser_render_cloud_modes(dash_app):
    b = page("threed_browser")
    dates = [d.isoformat() for d in td.run_dates(td.load_runs(), "W2", "J5")]
    idx = dates.index("2025-09-08")
    for colour in ("rgb", "height", "plant", "temp"):
        fig, options = b.render_cloud("W2", "J5", idx, dates, 15, colour, False)
        sc = traces(fig, "scatter3d")[-1]
        assert sc.marker.size == 2 and len(sc.x) > 1000
        assert traces(fig, "mesh3d")[0].z == (0, 0, 0, 0)
        assert fig.layout.scene.aspectmode == "data"
        assert "(mm)" in fig.layout.scene.zaxis.title.text
        assert "synthetic cloud" in fig.layout.title.text
        assert next(o for o in options if o["value"] == "temp")["disabled"]  # no temp in synth
    full, _ = b.render_cloud("W2", "J5", idx, dates, 15, "rgb", False)
    plant, _ = b.render_cloud("W2", "J5", idx, dates, 15, "rgb", True)
    assert len(traces(plant, "scatter3d")[0].x) < len(traces(full, "scatter3d")[0].x)
    fig, _ = b.render_cloud("W2", "J5", None, dates, 15, "rgb", False)
    assert not fig.data  # message figure


def test_browser_rgb_palette_is_exact(dash_app):
    b = page("threed_browser")
    rgb = np.array([[10, 20, 30], [200, 100, 0], [10, 20, 30]], np.uint8)
    idx, scale, cmax = b.rgb_colour_index(rgb)
    colours = [c for _, c in scale]
    assert colours[idx[0]] == "rgb(10,20,30)" and colours[idx[1]] == "rgb(200,100,0)"
    assert idx[0] == idx[2] and cmax == len(scale) - 1


def test_browser_second_view_and_side(dash_app):
    b = page("threed_browser")
    fig, md, style, disabled = b.render_second(False, "2025-09-03", "W2", "J5", 15, "rgb", False)
    assert md == 12 and style == {"display": "none"} and disabled
    fig, md, style, disabled = b.render_second(True, "2025-09-03", "W2", "J5", 15, "height", False)
    assert md == 6 and style == {} and not disabled and "2025-09-03" in fig.layout.title.text
    dates = [d.isoformat() for d in td.run_dates(td.load_runs(), "W2", "J5")]
    idx = dates.index("2025-09-08")
    panel, series, label = b.render_side("W2", "J5", idx, dates, 15, True, "2025-09-03")
    text = str(panel)
    assert "height_p75" in text and "QC pass" in text and "rgb_007.jpg" in text
    assert "+5" in label
    assert series.layout.yaxis.title.text == "height_p75 (mm)"
    assert len(series.layout.shapes) >= 4  # drought window, cutoff, re-watering, shown dates
    clickable = [t for t in series.data if t.customdata is not None]
    assert clickable and len(clickable[0].customdata[0]) == 2


def test_browser_real_cloud_with_temperature(real_root, dash_app):
    b = page("threed_browser")
    dates = [d.isoformat() for d in td.run_dates(td.load_runs(), "W2", "J5")]
    idx = dates.index("2025-09-08")
    fig, options = b.render_cloud("W2", "J5", idx, dates, 15, "temp", False)
    assert not next(o for o in options if o["value"] == "temp")["disabled"]
    assert "synthetic" not in fig.layout.title.text and "of 60,000" in fig.layout.title.text
    assert fig.data[-1].marker.colorbar.title.text == "leaf temp. (°C)"
    fig, _ = b.render_cloud("W2", "J5", dates.index("2025-09-08"), dates, 9, "rgb", False)
    assert not fig.data  # real clouds exist, this one does not: message, not a fake cloud
    panel, _, _ = b.render_side("W2", "J5", idx, dates, 15, False, None)
    assert "data:image/jpeg;base64" in str(panel) and "not reliable (E1)" in str(panel)


# ---------------------------------------------------------------- Compare callbacks
def test_compare_trajectories(dash_app):
    c = page("threed_compare")
    fig, count = c.update_trajectories("height_p75", "treatment", [], 15, "2025-08-20", "2025-09-19", False, False)
    names = [t.name for t in fig.data if t.showlegend is not False]
    assert any(n.startswith("control") for n in names) and any(n.startswith("deficit") for n in names)
    assert any(t.fill == "toself" for t in fig.data)  # SD band
    assert fig.layout.yaxis.title.text == "height_p75 (mm)"
    assert {s.x0 for s in fig.layout.shapes if s.type == "line"} >= {"2025-09-03", "2025-09-15"}
    assert "10 plants" in count
    fig, _ = c.update_trajectories("projected_area", "plants", ["W1_J5", "W2_J5"], 12, None, None, True, True)
    assert "change from baseline (%)" in fig.layout.yaxis.title.text
    assert {t.legendgroup for t in fig.data} == {"W1_J5", "W2_J5"}
    fig, count = c.update_trajectories("height_p75", "bed", [], 15, "2025-10-01", "2025-10-02", False, False)
    assert not fig.data and count.startswith("0 plants")
    assert c.toggle_plants("plants") is False and c.toggle_plants("bed") is True


def test_compare_result_card_synthetic_and_missing(dash_app, tmp_path, monkeypatch):
    c = page("threed_compare")
    text = str(c.update_card("height_p75"))
    assert "synthetic placeholder" in text and "slope" in text
    assert "not reliable" in str(c.update_card("compactness"))
    monkeypatch.delenv("UIWUE_SYNTHETIC")
    monkeypatch.setenv("UIWUE_ROOT", str(tmp_path))
    path = tmp_path / schema.TRAITS_RUNS
    path.parent.mkdir(parents=True)
    schema.synthetic_runs(plants_per_bed=2).to_parquet(path)
    td.clear_caches()
    text = str(c.update_card("height_p75"))
    assert "not available yet" in text and "e3_onset.csv" in text and "e1_noise.csv" in text
    heat, _depr, note = c.update_diurnal("height_p75", "treatment", [], None, None, False, "dev")
    assert traces(heat, "heatmap") and "computed from the table" in note
    _fig, rho = c.update_turgor("height_p75", "W2_J5", False, None, None)
    assert "not available yet" in str(rho)


def test_compare_result_card_from_files(real_root, dash_app):
    c = page("threed_compare")
    text = str(c.update_card("height_p75"))
    assert "% of baseline per day" in text and "-2.5" in text and "p < 0.001" in text
    assert "day +4" in text and "-3 days" in text and "e3_onset.csv" in text
    assert "not reliable" in str(c.update_card("compactness"))
    assert "No row for droop_index" in str(c.update_card("droop_index"))


def test_compare_diurnal_and_turgor(real_root, dash_app):
    c = page("threed_compare")
    heat, depr, _note = c.update_diurnal("height_p75", "bed", [], "2025-08-28", "2025-09-18", False, "abs")
    hm = traces(heat, "heatmap")
    assert len(hm) == 2 and list(hm[0].y) == [f"{h:02d} h" for h in schema.SLOTS]
    assert len(traces(depr, "scatter")) >= 4
    fig, rho = c.update_turgor("height_p75", "W2_J5", False, "2025-09-01", "2025-09-19")
    axes = {t.yaxis for t in fig.data}
    assert axes == {"y", "y2"}
    assert "W2 pooled" in str(rho) and "-0.60" in str(rho)
    fig, rho = c.update_turgor("height_p75", None, False, None, None)
    assert not fig.data


# ---------------------------------------------------------------- Export callbacks
def test_export_preview_and_downloads(dash_app):
    e = page("threed_export")
    args = (["W2"], [], "2025-09-01", "2025-09-05", [15], ["height_p75"], True, True)
    data, columns, count = e.update_preview(*args)
    ids = [c["id"] for c in columns]
    assert "height_p75" in ids and "height_p75_mad" in ids and "projected_area" not in ids
    assert {r["bed"] for r in data} == {"W2"} and {r["hour"] for r in data} == {15}
    assert data[0]["qc_pass"] in ("true", "false")
    assert "runs" in count and "5 plants" in count
    sent = e.download_csv(1, *args)
    assert sent["filename"].endswith("_SYNTHETIC.csv")
    csv = pd.read_csv(pd.io.common.StringIO(sent["content"]))
    assert len(csv) == 25 and set(csv["hour"]) == {15}
    sent = e.download_summary(1, *args)
    assert sent["filename"].endswith(".md")
    assert "noise floor" in sent["content"].lower() and "hours: 15" in sent["content"]
    data, _, count = e.update_preview([], [], None, None, [15], [], True, True)
    assert not data and count.startswith("No runs")
    options, value = e.update_plants(["W1"], ["W1_J5", "W2_J5"])
    assert value == ["W1_J5"] and all(o.startswith("W1_") for o in options)


# ---------------------------------------------------------------- smoke: pages through Flask
def _components(node, found: dict) -> None:
    if isinstance(node, dict):
        props = node.get("props")
        if not isinstance(props, dict):  # a plain mapping, e.g. the callback response
            for v in node.values():
                _components(v, found)
            return
        if isinstance(props.get("id"), str):
            found[props["id"]] = props
        for v in props.values():
            _components(v, found)
    elif isinstance(node, list):
        for v in node:
            _components(v, found)


def _outputs(spec: str) -> list[dict]:
    parts = spec.strip(".").split("...") if spec.startswith("..") else [spec]
    return [dict(zip(("id", "property"), p.rsplit(".", 1), strict=True)) for p in parts]


def render_page(client, path: str) -> tuple[dict, list[int], str]:
    """Run the pages router for ``path``, then every callback whose inputs are all on the page,
    with the values in the layout (the browser's first round). Returns the component props by
    id, the HTTP status of each callback and the page layout as JSON text."""
    deps = client.get("/_dash-dependencies").get_json()
    router = next(d for d in deps if "_pages_content.children" in d["output"])

    def post(dep, values, changed):
        outs = _outputs(dep["output"])
        body = {
            "output": dep["output"],
            "outputs": outs if dep["output"].startswith("..") else outs[0],
            "inputs": [dict(i, value=values.get(f"{i['id']}.{i['property']}")) for i in dep["inputs"]],
            "state": [dict(s, value=values.get(f"{s['id']}.{s['property']}")) for s in dep["state"]],
            "changedPropIds": changed,
        }
        return client.post("/_dash-update-component", json=body)

    r = post(
        router,
        {"_pages_location.pathname": path, "_pages_location.search": ""},
        ["_pages_location.pathname"],
    )
    assert r.status_code == 200, r.data[:500]
    found: dict = {}
    page_json = r.get_json()["response"]
    _components(page_json, found)
    values = {f"{cid}.{k}": v for cid, props in found.items() for k, v in props.items()}
    statuses = []
    for dep in deps:
        if dep is router or dep.get("prevent_initial_call"):
            continue
        ids = {i["id"] for i in dep["inputs"]}
        if not ids <= set(found):
            continue
        changed = [f"{i['id']}.{i['property']}" for i in dep["inputs"]]
        resp = post(dep, values, changed)
        assert resp.status_code in (200, 204), (dep["output"], resp.data[:800])
        statuses.append(resp.status_code)
    return found, statuses, json.dumps(page_json)


@pytest.mark.parametrize("path", PAGES)
def test_pages_render_through_flask(dash_app, path):
    client = dash_app.server.test_client()
    assert client.get(path).status_code == 200
    found, statuses, _ = render_page(client, path)
    if path == "/":
        assert not statuses
        return
    assert statuses and all(s == 200 for s in statuses)
    expected = {
        "/threed-browser": "tb-cloud",
        "/threed-compare": "tc-traj",
        "/threed-export": "te-table",
    }[path]
    assert expected in found


def test_navigation_lists_the_pages(dash_app):
    client = dash_app.server.test_client()
    layout = json.dumps(client.get("/_dash-layout").get_json())
    for href in PAGES[1:]:
        assert href in layout
    home = render_page(client, "/")[2]
    for href in PAGES[1:]:
        assert href in home


def test_browser_views_share_the_height_scale(dash_app):
    b = page("threed_browser")
    dates = [d.isoformat() for d in td.run_dates(td.load_runs(), "W2", "J5")]
    first, _ = b.render_cloud("W2", "J5", dates.index("2025-09-08"), dates, 15, "height", False)
    second = b.render_second(True, "2025-09-03", "W2", "J5", 15, "height", False)[0]
    top = first.data[-1].marker.cmax
    assert top == second.data[-1].marker.cmax == pytest.approx(b.colour_top("W2", "J5"))
