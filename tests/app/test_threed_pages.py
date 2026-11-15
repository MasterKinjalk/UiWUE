"""The 3D canopy page (``/threed``, app/pages/threed.py) and its figures (app/threed_figures.py).

Three kinds of test: small sandboxes in the committed table format with real-format clouds (which run is drawn,
THEN and NOW days, empty states), the committed tables of the repository (tiles, charts and readouts computed
live), and the page as Dash drives it: callbacks called with the triggers Dash sets, and the page rendered
through Flask with every initial callback. The cloud cache is local (not in git): tests that draw real clouds
accept the empty state when it is absent, and no test ever sees a made-up cloud.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from contextvars import copy_context
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import threed_canopy as tc
import threed_figures as tf
from components import BAND, C

from uiwue import core

TOKENS = {
    t["value"].lower() for t in json.loads((ROOT / "design" / "tokens.json").read_text("utf-8"))["color"]["tokens"]
}
TOKENS |= {v.lower() for v in C.values()}
FLAGS = list(tc.FLAG_COLUMNS)


@pytest.fixture(autouse=True)
def fresh_caches():
    tc.clear_caches()
    yield
    tc.clear_caches()


# ---------------------------------------------------------------- sandbox: committed table format, real-format clouds
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    derived, data = tmp_path / "derived", tmp_path / "data"
    derived.mkdir()
    data.mkdir()
    monkeypatch.setattr(core, "DERIVED", derived)
    monkeypatch.setattr(core, "DATA", data)
    return tmp_path


def row(bed, plant, day, hour, leaf, height=None, volume=None, qc=True, n_points=1000):
    height = leaf if height is None else height
    volume = 10 * leaf if volume is None else volume
    return dict(bed=bed, plant=plant, day=day, hour=hour, leaf=leaf, height=height, volume=volume, qc=qc, n=n_points)


def write_table(root, rows):
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "bed": df["bed"],
            "plant": df["plant"],
            "date": [core.CUTOFF + timedelta(days=int(d)) for d in df["day"]],
            "hour": df["hour"].astype(int),
            "depth_leaf_area_est": df["leaf"].astype(float),
            "depth_height_p75": df["height"].astype(float),
            "depth_hull_volume": df["volume"].astype(float),
            "depth_n_points": df["n"].astype(float),
            "depth_capture_hour": df["hour"].astype(int),
        }
    )
    for f in FLAGS:
        out[f"depth_{f}"] = False
    out["depth_qc_overlap"] = ~df["qc"].astype(bool)
    out["depth_qc_pass"] = df["qc"].astype(bool)
    (root / "derived" / "threed").mkdir(parents=True, exist_ok=True)
    out.to_parquet(root / "derived" / "threed" / "traits_runs.parquet", index=False)
    tc.clear_caches()


def write_cloud(root, bed, plant, day, hour, n_plant=600, top_mm=120.0, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 0.08, n_plant)
    a = rng.uniform(0, 2 * np.pi, n_plant)
    z = np.linspace(0.005, top_mm / 1000.0, n_plant)
    plant_xyz = np.c_[r * np.cos(a), r * np.sin(a), z]
    soil = np.c_[rng.uniform(-0.15, 0.15, (300, 2)), rng.normal(0, 0.001, 300)]
    xyz = np.vstack([plant_xyz, soil]).astype(np.float32)
    mask = np.r_[np.ones(n_plant, bool), np.zeros(300, bool)]
    date = core.CUTOFF + timedelta(days=day)
    path = root / "data" / "derived" / "threed" / "clouds" / f"{bed}_{plant}" / f"{date:%Y-%m-%d}_{hour:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz=xyz, rgb=np.zeros((len(xyz), 3), np.uint8), height=xyz[:, 2].copy(), plant=mask)
    tc.clear_caches()
    return path


def experiment(skip=(), override=None):
    """W1 A2 (control) and W2 A2, B3 (deficit), days −4 to 16, slots 9–18 h; leaf 100 on baseline days."""
    rows = []
    for bed, plants in (("W1", ("A2",)), ("W2", ("A2", "B3"))):
        for plant in plants:
            for day in range(-4, 17):
                if (bed, plant, day) in skip:
                    continue
                for hour in tc.DAY_SLOTS:
                    leaf = 100.0 if day < 1 or bed == "W1" else max(40.0, 100.0 - 6.0 * day)
                    r = row(bed, plant, day, hour, leaf)
                    if override and (bed, plant, day, hour) in override:
                        r.update(override[(bed, plant, day, hour)])
                    rows.append(r)
    return rows


def cloud(x, y, z) -> dict:
    x, y, z = (np.asarray(v, np.float32) for v in (x, y, z))
    return dict(x=x, y=y, z=z, n=len(z), bed="W2", plant="A2", day=0, slot=12, capture_hour=12)


def colours_in(obj) -> set[str]:
    """Every '#rrggbb' or rgba() colour in a figure or component, lower case."""
    text = json.dumps(obj.to_plotly_json() if hasattr(obj, "to_plotly_json") else obj, default=str)
    hexes = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", text)}
    rgbas = set(re.findall(r"rgba\(\d+,\d+,\d+,[0-9.]+\)", text))
    return hexes | rgbas


def token_rgba() -> set[str]:
    return {
        f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"
        for h in TOKENS
        for a in ("0.14", "0.18")
        if re.fullmatch(r"#[0-9a-f]{6}", h)
    }


# ---------------------------------------------------------------- which run, which days
def test_stage_draws_the_days_typical_run_never_a_failed_one(sandbox):
    outlier = {("W2", "A2", 5, 12): dict(leaf=30.0)}  # the 12 h run, first in slot order, is the odd one out that day
    rows = experiment(override=outlier)
    for r in rows:
        if (r["bed"], r["plant"], r["day"]) == ("W2", "A2", 8):
            r["qc"] = False  # day 8: every run failed QC
    write_table(sandbox, rows)
    write_cloud(sandbox, "W2", "A2", 5, 12, top_mm=40.0)
    write_cloud(sandbox, "W2", "A2", 5, 15, top_mm=110.0)
    write_cloud(sandbox, "W2", "A2", 8, 12)
    c = tf.stage_cloud("W2", "A2", 5)
    assert c["slot"] == 15 == tc.plant_cloud("W2", "A2", 5)["slot"]  # the data layer picks, the page draws
    assert c["z"].max() == pytest.approx(110.0, abs=0.1) and c["leaf_area_est_run"] == pytest.approx(70.0)
    assert tf.stage_cloud("W2", "A2", 5, 12)["slot"] == 12  # an explicit run (THEN) is drawn as asked
    assert tf.stage_cloud("W2", "A2", 8) is None  # a cloud of a failed run is never drawn
    assert tf.stage_cloud("W2", "A2", 6) is None and tf.stage_cloud("W2", "A2", None) is None
    assert tf.stage_cloud("W2", "ZZ", 5) is None


def test_then_is_the_baseline_day_closest_to_the_baseline_and_now_the_trough(sandbox):
    broken = {("W2", "A2", -1, h): dict(leaf=30.0) for h in tc.DAY_SLOTS}  # a QC-pass run that lost the plant
    rows = experiment(override=broken)
    for r in rows:
        if (r["bed"], r["plant"], r["day"]) == ("W2", "A2", 7):
            r["qc"] = False
    write_table(sandbox, rows)
    for day in (-4, -2, -1, 3, 6, 10, 13):
        write_cloud(sandbox, "W2", "A2", day, 12)
    days = tf.strip_days("W2", "A2")
    assert (
        dict(zip(days["day"], days["state"]))[7] == "no scan" and dict(zip(days["day"], days["state"]))[0] == "no file"
    )
    then = tc.then_run("W2", "A2")  # −1 is the latest but far from the baseline; −4 and −2 tie: the later
    assert (then["day"], then["slot"], then["in_baseline"]) == (-2, 12, True)
    assert tc.now_day("W2", "A2") == 10  # the latest drawable day without water (13 is after re-watering)
    assert tf.snap_day(days, 10) == 10 and tf.snap_day(days, 7) == 6 and tf.snap_day(days, 99) == 13
    assert tf.snap_day(days, 8) == 6  # 6 and 10 are both two days away: the earlier
    assert tf.snap_day(days, None) is None


def test_no_cloud_cache_gives_no_days_to_draw_but_keeps_the_readout_days(sandbox):
    write_table(sandbox, experiment())
    days = tf.strip_days("W2", "A2")
    assert len(days) == 21 and set(days["state"]) == {"no file"}
    assert tc.then_run("W2", "A2") is None and tc.now_day("W2", "A2") == 16
    assert tf.snap_day(days, 5) == 5  # no cloud anywhere: any day with a QC-pass run can be shown


# ---------------------------------------------------------------- stage geometry
def ring(radius, z, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cloud(radius * np.cos(a), radius * np.sin(a), np.full(n, float(z)))


def test_reach_is_the_highest_point_any_turn_can_bring_up():
    c = ring(60.0, 120.0)
    sin_e, cos_e = tf._elevation(tf.VIEW)
    assert tf.reach([c]) == pytest.approx(120.0 * cos_e + 60.0 * sin_e, rel=1e-6)
    stray = cloud([*c["x"], 0.0], [*c["y"], 0.0], [*c["z"], 400.0])  # one point far above the rest
    assert tf.reach([stray]) < 400.0 * cos_e and tf.reach([c, None]) == tf.reach([c])
    assert tf.reach([None]) > 0


def test_framing_fills_the_panel_and_centres_the_content():
    """With Plotly's orthographic scale (k · H / 2L px per mm), the content spans the panel's free height (or its
    width, for a low plant) and sits in the middle."""
    r, span = tc.CYLINDER_MM, 2 * (tc.CYLINDER_MM + tf.PAD_MM)
    sin_e, cos_e = tf._elevation(tf.VIEW)
    for top_mm in (40.0, 120.0, 260.0):
        c = ring(70.0, top_mm)
        top = 300.0
        f = tf.framing([c], top)
        k = f["aspectratio"]["x"]
        assert f["aspectratio"]["z"] == pytest.approx(k * top / span)  # one scale on every axis
        scale = k / (2 * span)  # px per mm in a panel 1 px high
        high, low = tf.reach([c]), -(r + tf.FRONT_MM) * sin_e
        used_h = (high - low) * scale
        used_w = (2 * r + 12.0) * scale / tf.STAGE_ASPECT
        assert max(used_h, used_w) == pytest.approx(1 - 2 * tf.MARGIN, rel=1e-6)
        centre_mm = (f["camera"]["center"]["z"] * span / k + top / 2) * cos_e
        assert centre_mm == pytest.approx((high + low) / 2, rel=1e-6)
        eye, centre = f["camera"]["eye"], f["camera"]["center"]
        d = np.array([eye["x"] - centre["x"], eye["y"] - centre["y"], eye["z"] - centre["z"]])
        assert np.allclose(d / np.linalg.norm(d), np.asarray(tf.VIEW) / np.linalg.norm(tf.VIEW))
        assert f["camera"]["projection"]["type"] == "orthographic"


def test_stage_figure_follows_the_design():
    then, now = ring(70.0, 150.0), ring(50.0, 60.0)
    figs = [tf.stage_figure(c, (then, now), 200.0, None, "W2-A2") for c in (then, now)]
    a, b = (f.layout.scene for f in figs)
    assert a.camera == b.camera and a.aspectratio == b.aspectratio and a.zaxis.range == b.zaxis.range
    assert a.zaxis.range[1] >= 150.0 and a.aspectmode == "manual" and a.dragmode == "turntable"
    assert not any(ax.visible for ax in (a.xaxis, a.yaxis, a.zaxis))
    assert figs[0].layout.uirevision == "W2-A2" and figs[0].layout.paper_bgcolor == C["surface"] == a.bgcolor
    plant = next(t for t in figs[0].data if t.name == "plant")
    assert plant.type == "scatter3d" and plant.mode == "markers" and plant.marker.line.width == 0
    assert [c for _, c in plant.marker.colorscale] == [C["leaf-300"], C["leaf-700"], C["leaf-900"]]
    assert plant.marker.cmin == 0 and plant.marker.cmax == 200.0 and 2.5 <= plant.marker.size <= 4.5
    assert {t.name for t in figs[0].data} == {"soil", "crop ring", "scale bar", "scale label", "plant"}
    soil = next(t for t in figs[0].data if t.name == "soil")
    assert soil.type == "mesh3d" and soil.color == C["soil"] and max(np.hypot(soil.x, soil.y)) == pytest.approx(85.0)
    bar = next(t for t in figs[0].data if t.name == "scale bar")
    ends = np.c_[bar.x, bar.y]
    assert np.linalg.norm(ends[1] - ends[0]) == pytest.approx(tf.SCALE_BAR_MM)  # 50 mm long ...
    view = np.asarray(tf.VIEW[:2]) / np.hypot(*tf.VIEW[:2])
    assert abs(np.dot(ends[1] - ends[0], view)) < 1e-6  # ... and square to the view, so it reads true
    assert "50 mm" in next(t for t in figs[0].data if t.name == "scale label").text
    for f in figs:
        assert colours_in(f) <= TOKENS
    empty = tf.stage_figure(None, (then, now), 200.0)
    assert "plant" not in {t.name for t in empty.data}


def test_a_turned_view_keeps_its_direction_and_rebuilds_the_bar():
    c = ring(70.0, 150.0)
    turned = {"eye": {"x": -1.0, "y": 0.2, "z": 0.5}, "center": {"x": 0.1, "y": 0, "z": 0.1}}
    d = tf.view_of(turned)
    assert np.allclose(d, np.array([-1.1, 0.2, 0.4]) / np.linalg.norm([-1.1, 0.2, 0.4]))
    assert tf.view_of(None) == pytest.approx(tuple(np.asarray(tf.VIEW) / np.linalg.norm(tf.VIEW)))
    f = tf.stage_figure(c, (c,), 200.0, d)
    eye, centre = f.layout.scene.camera.eye, f.layout.scene.camera.center
    got = np.array([eye.x - centre.x, eye.y - centre.y, eye.z - centre.z])
    assert np.allclose(got / np.linalg.norm(got), d)
    bar = next(t for t in f.data if t.name == "scale bar")
    assert abs(np.dot(np.diff(np.c_[bar.x, bar.y], axis=0)[0], np.asarray(d[:2]))) < 1e-6


def test_side_view_outlines_then_and_fills_now():
    then, now = ring(60.0, 140.0), ring(60.0, 70.0)
    fig = tf.side_figure(then, now, 150.0, ("then · day −1", "now · day 11"))
    fill, line = fig.data
    assert fill.fill == "tozeroy" and fill.fillcolor == C["leaf-300"] and fill.line.color == C["forest-700"]
    assert line.line.dash == "dot" and line.line.color == C["ink"] and line.name == "then · day −1"
    assert np.nanmax(line.y) == pytest.approx(140.0) and np.nanmax(fill.y) == pytest.approx(70.0)
    assert fig.layout.yaxis.range[1] >= 150.0 and tuple(fig.layout.xaxis.range) == (-85.0, 85.0)
    assert fig.layout.yaxis.title.text == "mm" and colours_in(fig) <= TOKENS


# ---------------------------------------------------------------- charts and tiles on the committed tables
def test_collapse_chart_shows_each_beds_median_spread_and_plants():
    """72 plants a bed: the median plant (line), the middle half (band), one dot per plant, the selected plant navy."""
    w = tc.windows()
    for trait in ("leaf_area_est", "height_p75"):
        fig = tf.slope_figure(trait, "W2", "J5")
        g = w[w["trait"] == trait]
        for group, bed in (("control", "W1"), ("deficit", "W2")):
            mine = g[g["bed"] == bed]
            median = next(t for t in fig.data if t.name == group)
            assert median.y[1] == pytest.approx(mine["peak_to_trough"].median())
            assert median.y[2] == pytest.approx(mine["peak_to_rewatered"].median())
            band = next(t for t in fig.data if t.name == f"{group} middle half")
            q = [mine[c].quantile([0.25, 0.75]).to_numpy() for c in ("peak_to_trough", "peak_to_rewatered")]
            assert band.fill == "toself" and band.fillcolor == BAND[group] and band.mode == "lines"  # no vertex dots
            assert np.allclose(band.y, [0.0, q[0][1], q[1][1], q[1][0], q[0][0], 0.0])  # the middle half of plants
            dots = [t for t in fig.data if t.name == f"{group} plants"]
            assert (
                sum(len(t.x) for t in dots)
                == mine["peak_to_trough"].notna().sum() + mine["peak_to_rewatered"].notna().sum()
            )  # every plant, at the trough and after re-watering
            assert {c[0] for t in dots for c in t.customdata} == {bed}  # a dot is a click target with its plant
        sel = fig.data[-1]
        assert sel.name == "W2 J5" and sel.line.color == C["navy"] and list(sel.customdata[0][:2]) == ["W2", "J5"]
        assert fig.layout.showlegend is False and colours_in(fig) <= TOKENS | token_rgba()


def test_a_w3_plant_is_drawn_alone_and_never_in_the_bands():
    w = tc.windows(tc.SHOWN_BEDS)
    w3 = w[(w["bed"] == "W3") & w["peak_to_trough"].notna()]
    if w3.empty:
        pytest.skip("no W3 plant in the 3D table")
    plant = str(w3["plant"].iloc[0])
    fig = tf.slope_figure("leaf_area_est", "W3", plant)
    assert fig.data[-1].name == f"W3 {plant}" and fig.data[-1].line.color == C["navy"]
    beds = {c[0] for t in fig.data if t.name.endswith(" plants") for c in t.customdata}
    assert beds == {"W1", "W2"}  # no W3 dot, no W3 band
    gap = tf.gap_figure("leaf_area_est", "W3", plant)
    assert any(t.name == f"W3 {plant}" for t in gap.data)


def test_gap_chart_has_the_bands_the_interval_the_onset_and_the_plant_second():
    o = tc.onsets()
    onset = float(o.loc[o["trait"] == "leaf_area_est", "robust_onset_day"].iloc[0])
    fig = tf.gap_figure("leaf_area_est", "W2", "J5", onset)
    g = tc.gap("leaf_area_est")["by_day"]
    gap = next(t for t in fig.data if t.name == "W2 − W1")
    assert list(gap.x) == list(g["day"]) and gap.line.color == C["amber-700"]
    assert next(t for t in fig.data if t.fill == "toself").fillcolor == BAND["deficit"]
    mine = next(t for t in fig.data if t.name == "W2 J5")
    expect = tf.plant_gap("leaf_area_est", "W2", "J5").set_index("day")["value"]
    assert mine.line.color == C["navy"] and np.allclose(mine.y, expect.loc[list(mine.x)].to_numpy())
    names = [t.name for t in fig.data]
    assert names.index("W2 J5") < names.index("W2 − W1")  # the bed gap is drawn on top: the finding reads first
    assert gap.line.width > 2 * mine.line.width
    rects = {s.fillcolor for s in fig.layout.shapes if s.type == "rect"}
    assert {C["mint"], C["sand"], C["gray"]} <= rects  # baseline, no water, watered-day spread
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert any(s.x0 == onset and s.line.dash == "dash" and s.line.color == C["navy"] for s in lines)
    assert fig.layout.xaxis.range[0] < -10 < 16 < fig.layout.xaxis.range[1] and fig.layout.yaxis.title.text == "pp"
    assert colours_in(fig) <= TOKENS | token_rgba()


def test_short_labels_for_the_onset_hover():
    assert tf.short_label("Leaf minus air temperature") == "Leaf − air temp."
    assert tf.short_label("Projected leaf area") == "Projected area"


# ---------------------------------------------------------------- the page as Dash drives it
@pytest.fixture(scope="session")
def dash_app():
    """The app shell with all pages, built once (pages register their callbacks globally)."""
    spec = importlib.util.spec_from_file_location("uiwue_app_shell", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture
def page(dash_app):
    import dash

    return sys.modules[dash.page_registry["pages.threed"]["module"]]


def with_trigger(prop_id: str, value, fn, *args):
    """Call a callback that reads ``dash.ctx`` as if ``prop_id`` had changed to ``value``."""
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    def run():
        context_value.set(AttributeDict(triggered_inputs=[{"prop_id": prop_id, "value": value}]))
        return fn(*args)

    return copy_context().run(run)


def ids(component) -> dict:
    return {
        c.id if isinstance(c.id, str) else json.dumps(c.id): c for c in component._traverse() if getattr(c, "id", None)
    }


def text(component) -> str:
    """A component, a list of them or a figure as JSON text (real characters, not escapes)."""

    def plain(o):
        return o.to_plotly_json() if hasattr(o, "to_plotly_json") else str(o)

    return json.dumps(component, default=plain, ensure_ascii=False)


def test_the_page_is_one_route_with_the_old_links_on_it(dash_app):
    import dash

    threed = [p for p in dash.page_registry.values() if p["path"].startswith("/threed")]
    assert len(threed) == 1
    p = threed[0]
    assert (p["path"], p["name"], p["order"]) == ("/threed", "3D canopy", 25) and p["description"].endswith("?")
    for old in ("threed_browser", "threed_compare", "threed_export"):
        assert f"pages.{old}" not in dash.page_registry and not (APP_DIR / "pages" / f"{old}.py").exists()
    assert not (APP_DIR / "threed_data.py").exists()
    from dash._pages import _path_to_page

    for path in ("threed", "threed-browser", "threed-compare", "threed-export"):
        found, variables = _path_to_page(path)
        assert found["path"] == "/threed" and variables["legacy"] == path.removeprefix("threed")
    assert _path_to_page("plant-profile")[0]["path"] == "/plant-profile"


def test_layout_reads_the_link_and_never_opens_on_a_failed_scan(page):
    state = ids(page.layout())["td-state"].data
    assert (state["bed"], state["plant"]) == page.resolve(None, None)
    days = tf.strip_days(state["bed"], state["plant"])
    by_day = dict(zip(days["day"], days["state"]))
    assert by_day[state["day"]] != "no scan" and (state["then"] is None or by_day[state["then"]] != "no scan")
    if tc.clouds_dir().is_dir():
        assert (state["bed"], state["plant"]) == tc.default_plant() and by_day[state["day"]] == "ok"
        assert state["then_slot"] == tc.then_run(state["bed"], state["plant"])["slot"]
    s = ids(page.layout(bed="W1", plant="J5", day="5"))["td-state"].data
    assert (s["bed"], s["plant"]) == ("W1", "J5") and s["day"] == tf.snap_day(tf.strip_days("W1", "J5"), 5)
    # a link to a day whose runs all failed QC opens on the nearest drawable day
    failed = next(
        (b, p, int(d))
        for b, p in page.plants()[:40]
        for d in [tf.strip_days(b, p).query("state == 'no scan' and day >= 0")["day"].min()]
        if pd.notna(d)
    )
    b, p, day = failed
    date = (pd.Timestamp(core.CUTOFF) + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
    s = ids(page.layout(bed=b, plant=p, date=date))["td-state"].data
    got = dict(zip(*[tf.strip_days(b, p)[c] for c in ("day", "state")]))
    assert s["day"] != day and got[s["day"]] != "no scan" and abs(s["day"] - day) <= 3
    s = ids(page.layout(bed="W2", plant="G8", legacy="-browser", date="2025-09-13", hour="15"))["td-state"].data
    assert (s["plant"], s["day"]) == ("G8", tf.snap_day(tf.strip_days("W2", "G8"), 10))
    assert ids(page.layout(bed="nope", plant="Z9"))["td-state"].data["plant"] == state["plant"]
    head = ids(page.layout())
    assert head["td-bed"].value == state["bed"] and head["td-plant"].value == state["plant"]
    assert page.asked_day("x", None) is None and page.asked_day(None, "2025-09-03") == 0


def test_navigate_covers_every_way_of_choosing(page):
    state = page.state_for("W2", "J5")
    every = list(page.plants())
    nav = page.navigate
    out = with_trigger("td-bed.value", "W1", nav, "W1", "J5", None, None, None, None, [], None, None, state)
    assert out[0]["bed"] == "W1" and out[4] == "J5" and out[3] == page.plant_options("W1") and out[2] == "W1"
    out = with_trigger("td-plant.value", "G8", nav, "W2", "G8", None, None, None, None, [], None, None, state)
    assert out[0] == page.state_for("W2", "G8") and out[1] == {"bed": "W2", "plant": "G8"}
    first = page.state_for(*every[0])
    out = with_trigger("td-prev.n_clicks", 1, nav, *every[0], 1, None, None, None, [], None, None, first)
    assert (out[0]["bed"], out[0]["plant"]) == every[-1]  # wraps round
    click = {"points": [{"customdata": ["W1", "Q6", "+31 %"]}]}
    out = with_trigger(
        "td-slope-leaf.clickData", click, nav, "W2", "J5", None, None, click, None, [], None, None, state
    )
    assert (out[0]["bed"], out[0]["plant"]) == ("W1", "Q6")
    mean = {"points": [{"customdata": ["W2", "", "−39 %"]}]}  # the bed mean line is not a plant
    out = with_trigger(
        "td-slope-height.clickData", mean, nav, "W2", "J5", None, None, None, mean, [], None, None, state
    )
    assert out[0] is page.no_update
    cell = json.dumps({"day": 5, "type": "td-cell"}) + ".n_clicks"
    out = with_trigger(cell, 1, nav, "W2", "J5", None, None, None, None, [1], None, None, state)
    assert out[0] == {**state, "day": 5} and out[1] is page.no_update
    out = with_trigger(cell, 0, nav, "W2", "J5", None, None, None, None, [0], None, None, state)
    assert out[0] is page.no_update  # cells drawn afresh come back with no clicks
    days = tf.strip_days("W2", "J5")
    ok = sorted(int(d) for d in days.loc[days["state"] == "ok", "day"]) or sorted(
        int(d) for d in days.loc[days["state"] != "no scan", "day"]
    )
    at = {**state, "day": ok[-2]}
    out = with_trigger("td-day-next.n_clicks", 1, nav, "W2", "J5", None, None, None, None, [], None, 1, at)
    assert out[0]["day"] == ok[-1]
    out = with_trigger("td-day-next.n_clicks", 2, nav, "W2", "J5", None, None, None, None, [], None, 2, out[0])
    assert out[0] is page.no_update  # stays put at the end
    out = with_trigger("td-day-prev.n_clicks", 1, nav, "W2", "J5", None, None, None, None, [], 1, None, at)
    assert out[0]["day"] == ok[-3]


def test_stage_shares_one_camera_and_the_card_reads_the_run_drawn(page):
    state = page.state_for("W2", "J5")
    out = page.show_stage(state, None)
    then_fig, now_fig, then_head, now_head = out[:4]
    assert f"Day {state['day']}".replace("-", "−") in text(now_head)
    if tc.clouds_dir().is_dir():
        then = tc.then_run("W2", "J5")
        assert f"Day {then['day']}".replace("-", "−") in text(then_head) and f"{then['slot']:02d} h" in text(then_head)
        assert out[4] == "td-view-head td-base" and out[5] == "td-view-head td-dry"
        a, b = then_fig.layout.scene, now_fig.layout.scene
        assert a.camera == b.camera and a.aspectratio == b.aspectratio
        assert then_fig.data[-1].marker.cmax == now_fig.data[-1].marker.cmax == tc.axis_ranges("W2", "J5")["z"][1]
        assert out[6] is None and out[7] is None
    turned = {"eye": {"x": -1.4, "y": -0.4, "z": 0.6}, "center": {"x": 0, "y": 0, "z": 0}}
    out2 = page.show_stage(state, turned)
    if tc.clouds_dir().is_dir():  # the reader's last turn carries over to the next day
        eye, centre = out2[1].layout.scene.camera.eye, out2[1].layout.scene.camera.center
        d = np.array([eye.x - centre.x, eye.y - centre.y, eye.z - centre.z])
        assert np.allclose(d / np.linalg.norm(d), tf.view_of(turned))
    strip, card, side, key = page.show_plant(state)
    pressed = [c for c in strip if c.to_plotly_json()["props"].get("aria-pressed") == "true"]
    assert len(pressed) == 1 and pressed[0].id == {"type": "td-cell", "day": state["day"]}
    assert all(c.disabled for c in strip if "td-noscan" in c.className)
    row = tf.strip_days("W2", "J5").set_index("day").loc[state["day"]]
    t = text(card)
    if row["state"] == "ok":  # the card reads the run drawn as NOW, so the numbers describe the picture
        assert tf.pct(row["leaf_area_est_run_pct"], 0, "") in t and f"{int(row['slot']):02d} h run" in t
        assert "Day mean of the QC-pass" in t  # the plant-day mean on hover
    else:
        assert tf.pct(row["leaf_area_est_pct"], 0, "") in t and "day mean" in t
    assert "W2 J5" in t and "uw-mark" not in t
    readouts = [c for c in card._traverse() if getattr(c, "className", None) == "td-readout"]
    chips = {r.children[0].children: "bed effect" in text(r) for r in readouts}
    effects = tc.bed_effects()
    assert chips == {t.label: bool(page.trait_chips(k, effects)) for k, t in tc.TRAITS.items()}
    assert f"/plant-profile?bed=W2&plant=J5&day={state['day']}" in t  # the card's links carry the day shown
    assert isinstance(side, go.Figure) and len(side.data) == 2 and side.layout.showlegend is False
    then_day = f"then · day {state['then']}".replace("-", "−")
    assert f"now · day {state['day']}" in text(key) and then_day in text(key)


def test_a_control_flagged_by_the_leaf_area_rule_shows_no_status(page):
    """The −20 % two-day rule also fires on a watered control (W1 Q6): the page shows values, never a status."""
    alerts = tc.structure_alerts()
    false = alerts[(alerts["bed"] == "W1") & alerts["first_irrigate"].notna()]
    for plant in [*false["plant"], "Q6"]:
        _, card, _, _ = page.show_plant(page.state_for("W1", plant))
        assert "uw-mark" not in text(card) and "irrigate" not in text(card)


def test_evidence_follows_the_selected_plant(page):
    slope_leaf, slope_height, gap_leaf, gap_height, context, name = page.show_evidence({"bed": "W1", "plant": "J5"})
    for fig in (slope_leaf, slope_height, gap_leaf, gap_height):
        assert any(t.name == "W1 J5" and t.line.color == C["navy"] for t in fig.data)
    assert name == "W1 J5" and context == []  # the treatment is in the card; only W3 gets a chip here
    tiles = page.tiles()
    h = tc.headline()
    t = text(tiles)
    for trait in ("height_p75", "leaf_area_est"):
        assert tf.pct(h[trait]["peak_to_trough"]["deficit"]["mean"], 0, "") in t
        assert tf.pct(h[trait]["peak_to_trough"]["control"]["mean"]) in t
    assert tf.pct(h["leaf_area_est"]["peak_to_rewatered"]["deficit"]["mean"], 0, "") in t
    o = tc.onsets()
    leaf = o.loc[o["trait"] == "leaf_area_est", "robust_onset_day"].iloc[0]
    thermal = o.loc[(o["sensor"] == "Thermal") & ~o["is_size_baseline"], "robust_onset_day"].iloc[0]
    size = o.loc[o["is_size_baseline"], "robust_onset_day"].iloc[0]
    first = text(tiles[3])
    assert "First seen" in first and f"day {leaf:g}" in first and f"Size only · day {size:g}" in first  # DESIGN 7.6
    assert page._after(leaf, thermal, "thermal") in first and page._after(5, 4, "thermal") == "1 day after thermal"
    assert page._after(4, 4, "thermal") == "same day as thermal" and page._after(3, 5, "x") == "2 days before x"
    for label in ("Height lost", "Leaf area lost", "After re-watering"):
        assert label in t
    effects = tc.bed_effects()
    for tile, trait in ((tiles[0], "height_p75"), (tiles[1], "leaf_area_est")):
        assert ("bed effect" in text(tile)) == bool(page.trait_chips(trait, effects))
    info = tc.onsets_info()
    assert ("older table" in text(tiles[3])) == info["stale"]  # never an onset passed off as the new table's
    if info["stale"]:
        assert f"n = {info['n_deficit']} deficit and {info['n_control']} control" in text(tiles[3])


def test_a_flagged_trait_says_bed_effect_once(page):
    """DESIGN 7.4: a trait the bed checks flag carries the chip on every chart that uses it. When every trait on the
    page is flagged (the committed tables: placebo or E2 flags all three), one header chip names them all: the
    same chip on every tile, card row and chart said nothing the header does not."""
    effects = tc.bed_effects()
    lay = page.layout(bed="W2", plant="J5")
    head = lay.children[0]
    cols = [c for c in lay._traverse() if getattr(c, "className", None) == "td-trait-col"]
    assert len(cols) == 2
    heads = {getattr(c.children[0].children[1].children, "id", None): text(c.children[0].children[0]) for c in cols}
    assert set(heads) == {"td-slope-leaf", "td-slope-height"}
    chips = [c for c in head._traverse() if "uw-chip" in str(getattr(c, "className", ""))]
    assert len(chips) <= 3  # bed effect (or 1 control bed), method, and for W3 not pooled
    if page.every_trait_flagged(effects):
        caveat = next(c for c in chips if c.children == "bed effect")
        assert all(t.label in caveat.title for t in tc.TRAITS.values()) and "One control bed" in caveat.title
        assert not any("bed effect" in h for h in heads.values())
        assert "bed effect" not in text(page.tiles())
    else:
        for graph_id, trait in {"td-slope-leaf": "leaf_area_est", "td-slope-height": "height_p75"}.items():
            assert ("bed effect" in heads[graph_id]) == effects[trait]["chip"], graph_id
    assert effects["leaf_area_est"]["chip"]  # on the committed tables the E2 check flags leaf area (p = 0.003)


def test_next_steps_carry_the_selection(page):
    state = page.state_for("W2", "J5")
    links = page.show_next(state)
    assert [link.children for link in links] == [t for t, _, _ in page.NEXT]
    assert all(link.href.endswith(f"?bed=W2&plant=J5&day={state['day']}") for link in links)
    assert links[0].className == "uw-btn uw-btn-primary" and links[1].className == "uw-btn uw-btn-outline"


def test_no_cloud_cache_shows_the_empty_state_and_the_charts_still_render(sandbox, page):
    write_table(sandbox, experiment())
    state = page.state_for("W2", "A2")
    assert state["then"] is None and state["day"] == 16
    out = page.show_stage(state, None)
    assert out[0] == out[1] == page.blank_figure() and not out[0]["data"]  # nothing drawn, nothing made up
    assert "Point clouds" in text(out[6]) and "no table yet" in text(out[7])  # the shared empty state
    strip, card, side, _key = page.show_plant(state)
    assert len(strip) == 21 and "Leaf area" in text(card)
    assert all(t.y is None or np.isnan(np.asarray(t.y, float)).all() for t in side.data)
    figs = page.show_evidence({"bed": "W2", "plant": "A2"})
    assert all(isinstance(f, go.Figure) and f.data for f in figs[:4])
    lay = page.layout(bed="W2", plant="A2")
    assert ids(lay)["td-state"].data["plant"] == "A2"


def test_a_missing_table_is_the_empty_state(sandbox, page):
    lay = page.layout()
    assert "no table yet" in text(lay) and "td-state" not in ids(lay)


# ---------------------------------------------------------------- through Flask, as the browser does it
def _components(node, found: dict) -> None:
    if isinstance(node, dict):
        props = node.get("props")
        if not isinstance(props, dict):
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


def render(client, path: str, search: str = ""):
    """The pages router for ``path``, then every server callback whose inputs are all on the page, with the
    values in the layout (the browser's first round). Returns the props by id and each callback's status."""
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

    r = post(router, {"_pages_location.pathname": path, "_pages_location.search": search}, ["_pages_location.pathname"])
    assert r.status_code == 200, r.data[:500]
    found: dict = {}
    _components(r.get_json()["response"], found)
    values = {f"{cid}.{k}": v for cid, props in found.items() for k, v in props.items()}
    statuses = {}
    for dep in deps:
        if dep is router or dep.get("prevent_initial_call") or dep.get("clientside_function"):
            continue
        inputs = {i["id"] for i in dep["inputs"] if isinstance(i["id"], str)}
        if not inputs or not inputs <= set(found):
            continue
        resp = post(dep, values, [f"{i['id']}.{i['property']}" for i in dep["inputs"]])
        statuses[dep["output"]] = resp.status_code
    return found, statuses, deps


def test_page_renders_through_flask_and_every_first_callback_runs(dash_app):
    client = dash_app.server.test_client()
    assert client.get("/threed").status_code == 200
    found, statuses, deps = render(client, "/threed")
    assert {"td-then", "td-now", "td-strip", "td-card", "td-side", "td-slope-leaf", "td-gap-height", "td-steps"} <= set(
        found
    )
    assert "td-onset" not in found  # the onset strip is the tile now (Early warning has the full table)
    assert statuses and set(statuses.values()) == {200}, statuses
    assert any("td-then.figure" in o and "td-now.figure" in o for o in statuses)
    client_side = {d["output"] for d in deps if d.get("clientside_function")}
    for out in ("td-url.data", "td-link.data", "td-view.data", "td-views.className", "td-scroll.data"):
        assert any(out in o for o in client_side), out  # URL write-back, camera link and memory, phone toggle
    layout = json.dumps(client.get("/_dash-layout").get_json())
    assert '"/threed"' in layout and not re.search(r"/threed-(browser|compare|export)", layout)


@pytest.mark.parametrize("old", ["/threed-browser", "/threed-compare", "/threed-export"])
def test_old_deep_links_land_on_the_page_with_their_plant(dash_app, old):
    client = dash_app.server.test_client()
    assert client.get(f"{old}?bed=W2&plant=G8").status_code == 200  # served by the app, not a 404
    found, statuses, _ = render(client, old, "?bed=W2&plant=G8")
    assert found["td-state"]["data"]["plant"] == "G8" and set(statuses.values()) == {200}


# ---------------------------------------------------------------- design system
def test_page_code_and_css_use_tokens_only():
    for path in (APP_DIR / "pages" / "threed.py", APP_DIR / "threed_figures.py", APP_DIR / "assets" / "threed.css"):
        src = path.read_text(encoding="utf-8")
        assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", src), path.name  # colours come from tokens
        assert "box-shadow" not in src and "plotly_dark" not in src
    css = (APP_DIR / "assets" / "threed.css").read_text(encoding="utf-8")
    assert all(int(px) >= 12 for px in re.findall(r"font-size:\s*(\d+)px", css))  # nothing below 12 px
    ratio = re.search(r"\.td-canvas \{[^}]*aspect-ratio:\s*([0-9.]+)", css)
    assert ratio and float(ratio.group(1)) == pytest.approx(tf.STAGE_ASPECT)  # the frame is solved for this shape
    assert re.search(r"min-height:\s*(4[4-9]|5\d)px", css)  # day cells and readouts are real targets


def test_w3_is_in_the_picker_with_its_own_chips(page):
    if "W3" not in page.beds():
        pytest.skip("no W3 plant in the 3D table")
    plant = next(p for b, p in page.plants() if b == "W3")
    lay = ids(page.layout(bed="W3", plant=plant))
    assert lay["td-state"].data["bed"] == "W3" and [o["value"] for o in lay["td-bed"].options] == ["W1", "W2", "W3"]
    chips = text(lay["td-context"])
    assert "not pooled" in chips and "left out of every bed number" in chips
    assert "not pooled" not in text(ids(page.layout(bed="W2", plant="J5"))["td-context"])
    out = page.show_evidence({"bed": "W3", "plant": plant})
    assert out[-1] == f"W3 {plant}" and "not pooled" in text(out[-2])


def test_detached_edge_points_are_not_drawn_and_the_head_says_so(page):
    body = np.c_[np.linspace(-40, 40, 300), np.zeros(300), np.linspace(5, 90, 300)]
    piece = np.c_[np.full(12, 84.0), np.linspace(-2, 2, 12), np.linspace(140, 146, 12)]
    xyz = np.vstack([body, piece])
    stray = tc.stray_mask(*xyz.T)
    assert stray[-12:].all() and not stray[:-12].any()
    c = cloud(*xyz.T) | dict(stray=stray, n_stray=12, leaf_area_est_run=50.0, leaf_area_est_run_pct=-50.0)
    c |= dict(height_p75_run=60.0, height_p75_run_pct=-40.0, hull_volume_run=500.0, hull_volume_run_pct=-50.0)
    fig = tf.stage_figure(c, (c,), 150.0)
    plant = next(t for t in fig.data if t.name == "plant")
    assert len(plant.z) == 300 and max(plant.z) == pytest.approx(90.0)  # the floating piece is not drawn
    side = np.asarray(tf.side_figure(c, c, 100.0).data[0].y, float)
    assert tf.frame_height(c) == 90.0 and np.nanmax(side) == pytest.approx(90.0)
    head = text(page.view_head("now", 0, c))
    assert "12 detached edge points not drawn" in head and "leaf area 50 cm² (−50 %)" in head
