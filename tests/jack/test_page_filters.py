"""The app-wide 3D trait filter and the page fixes on Early warning and Methods (fast: hand-made tables).

The detect tables here are written by hand into a temporary derived/, so every expected value is known:
a dropped 3D trait is the earliest signal of all, another dropped one is flagged, and the pages must
rank, count and pick only after leaving them out.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import html

REPO = Path(__file__).resolve().parents[2]
PAGES = REPO / "app" / "pages"
if str(PAGES) not in sys.path:  # _jack_common, as the pages import it
    sys.path.insert(0, str(PAGES))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"jack_filter_{name}", PAGES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text(component) -> str:
    return json.dumps(component.to_plotly_json(), default=str, ensure_ascii=False)


SENSOR = {"thermal": "Thermal", "depth": "Depth", "rgb": "RGB"}
# signal: (robust onset day, bed effect, size baseline); groups all and W2. W3 has thermal plants only.
SIGNALS = {
    "thermal_leaf_air_dt": (4, False, False),
    "depth_cyl_n_points": (2, False, False),  # dropped: would be the earliest signal of all
    "depth_n_points": (3, True, False),  # dropped and flagged
    "depth_droop_index": (5, False, False),  # dropped
    "depth_leaf_area_est": (6, False, False),  # kept: the next sensor once the dropped ones are gone
    "depth_hull_volume": (7, True, False),  # kept and flagged
    "depth_projected_area": (8, False, True),  # kept as the size-only baseline
    "rgb_ngrdi": (9, False, False),
}


def _label(signal):
    return signal.split("_", 1)[1].replace("_", " ")  # depth_leaf_area_est -> "leaf area est"


def _onset_row(signal, group, rob, bed, size):
    w3_empty = group == "W3" and not signal.startswith("thermal")
    return dict(
        signal=signal,
        label=_label(signal),
        sensor=SENSOR[signal.split("_", 1)[0]],
        kind="trait",
        group=group,
        direction=1,
        onset_day=float(rob),
        robust_onset_day=float(rob),
        early_auc=0.8,
        early_ci_low=0.6,
        early_ci_high=0.9,
        max_auc=0.9,
        placebo_auc=0.8 if bed else 0.5,
        placebo_ci_low=0.72 if bed else 0.4,
        placebo_ci_high=0.9 if bed else 0.6,
        bed_effect=bed,
        is_size_baseline=size,
        n_deficit=0 if w3_empty else (11 if signal.startswith("depth") else 20),
        n_control=0 if w3_empty else (7 if signal.startswith("depth") else 13),
        null_auc_max=0.7,
        precutoff_fire_day=np.nan,
        fires_before_cutoff=False,
        symptom_day=9.0,
        symptom_day_low=7.0,
        symptom_first_day=3.0,
        n_symptom=4,
        clip_day=6.0,
        n_clip=5,
        n_imaged=20,
        lead_vs_clip=6.0 - rob,
    )


@pytest.fixture
def detect(tmp_path, monkeypatch):
    """A derived/ with hand-made detect tables (and a 3D table that has W3 plants the results do not)."""
    from conftest import point_core_at

    d = tmp_path / "detect"
    d.mkdir()
    rows = [_onset_row(s, g, *v) for s, v in SIGNALS.items() for g in ("all", "W2", "W3")]
    onsets = pd.DataFrame(rows)
    onsets.to_parquet(d / "onsets.parquet", index=False)
    onsets[
        ["signal", "label", "sensor", "group", "placebo_auc", "placebo_ci_low", "placebo_ci_high", "bed_effect"]
        + ["n_deficit", "n_control", "fires_before_cutoff", "precutoff_fire_day", "null_auc_max"]
    ].to_parquet(d / "placebo.parquet", index=False)
    days = np.arange(-10, 19)
    auc = pd.DataFrame(
        [
            dict(signal=s, group=g, day=int(x), date=pd.Timestamp("2025-09-03") + pd.Timedelta(days=int(x)))
            | dict(auc=0.5, ci_low=0.4, ci_high=0.6, n_deficit=11, n_control=7, kind="trait", direction=1, column=s)
            for s in SIGNALS
            for g in ("all", "W2", "W3")
            for x in days
        ]
    )
    auc.to_parquet(d / "auc_by_day.parquet", index=False)
    slots = pd.DataFrame(
        [
            dict(signal=s, label=s, sensor=SENSOR[s.split("_", 1)[0]], group=g, hour=h, onset_day=5.0)
            | dict(robust_onset_day=6.0, null_auc_max=0.7, precutoff_fire_day=np.nan, fires_before_cutoff=False)
            | dict(n_deficit=11, n_control=7)
            for s in SIGNALS
            for g in ("all", "W2")
            for h in (12, 15)
        ]
    )
    slots.to_parquet(d / "onsets_by_slot.parquet", index=False)
    rec = pd.DataFrame(
        [
            dict(plant_id=f"W2_C{i}", signal=s, label=_label(s), recovery_ratio=0.5, measurable=True)
            for s in (*SIGNALS, "depth_height_p75", "depth_cyl_height_p75")
            for i in range(4)
        ]
    )
    rec.to_parquet(d / "recovery.parquet", index=False)
    traits = [*SIGNALS, "depth_height_p75", "depth_cyl_height_p75"]
    info = dict(finished="2026-11-15T04:10:12+00:00", n_boot=100, traits=traits, plants=30, skipped=[])
    (d / "run_info.json").write_text(json.dumps(info))
    # the 3D trait table now has W3 plants, which the detect results above do not
    (tmp_path / "threed").mkdir()
    pd.DataFrame(
        dict(
            bed=["W1", "W2", "W3", "W3"],
            plant=["A2", "A4", "A6", "A8"],
            date=pd.to_datetime(["2025-09-01"] * 4).date,
            hour=[12] * 4,
            depth_leaf_area_est=[1.0, 2.0, 3.0, 4.0],
        )
    ).to_parquet(tmp_path / "threed" / "traits_runs.parquet", index=False)
    point_core_at(tmp_path, monkeypatch)
    return tmp_path


# --------------------------------------------------------------------------- the shared definition
def test_kept_and_dropped_3d_traits():
    import _jack_common as jc

    for s in (*jc.DEPTH_DROPPED, "depth_cyl_n_points", "depth_cyl_height_p75", "depth_cyl_projected_area"):
        assert not jc.kept_signal(s), s
    for s in (*jc.DEPTH_KEPT, jc.DEPTH_SIZE, "thermal_leaf_air_dt", "rgb_ngrdi", "model:depth:logistic"):
        assert jc.kept_signal(s), s
    assert set(jc.DEPTH_KEPT) == {"depth_leaf_area_est", "depth_height_p75", "depth_hull_volume"}
    assert jc.DEPTH_SIZE == "depth_projected_area"
    df = pd.DataFrame({"signal": ["depth_cyl_n_points", "depth_leaf_area_est", "thermal_leaf_t"], "x": [1, 2, 3]})
    assert list(jc.keep_signals(df)["signal"]) == ["depth_leaf_area_est", "thermal_leaf_t"]
    bench = pd.DataFrame({"setup": ["Depth"], "n_features": [16]})
    assert jc.keep_signals(bench) is bench and jc.keep_signals(None) is None


def test_every_catalogued_3d_stress_trait_has_a_decision():
    """A new depth trait must be put in DEPTH_KEPT or DEPTH_DROPPED before it can show up on a page."""
    import _jack_common as jc

    f = REPO / "derived" / "threed" / "catalogue.csv"
    if not f.exists():
        pytest.skip("no 3D catalogue in derived/")
    cat = pd.read_csv(f)
    stress = set(cat.loc[cat["direction"] != 0, "column"])
    decided = {*jc.DEPTH_KEPT, jc.DEPTH_SIZE, *jc.DEPTH_DROPPED}
    undecided = {c for c in stress if c not in decided and not c.startswith(jc.DEPTH_DROPPED_PREFIX)}
    assert not undecided
    assert set(jc.DEPTH_DROPPED) <= set(cat["column"])  # no typo in the dropped list


def test_load_result_filters_on_read(detect):
    import _jack_common as jc

    on = jc.load_result("onsets")
    assert not on["signal"].str.startswith(jc.DEPTH_DROPPED_PREFIX).any()
    assert not on["signal"].isin(jc.DEPTH_DROPPED).any()
    assert {"depth_leaf_area_est", "depth_projected_area", "thermal_leaf_air_dt"} <= set(on["signal"])
    assert jc.load_result("benchmark") is None
    assert jc.run_time(jc.run_info()) == "2026-11-15 04:10 UTC"


# --------------------------------------------------------------------------- Early warning
def test_early_warning_ranks_and_counts_after_the_filter(detect):
    ew = _load("early_warning")
    text = _text(ew.layout())
    for s in ("depth_cyl_n_points", "depth_n_points", "depth_droop_index"):
        assert s not in text  # not in the signal picker
    assert "size · projected area (Depth)" in text  # the size baseline is tagged in the picker
    assert "detect run 2026-11-15 04:10 UTC" in text  # provenance of the plant counts

    tiles, fig, flags, rec, tbl = ew._update("all")
    tiles_text = _text(html.Div(tiles))
    # the earliest signal overall is a dropped trait (day 2): the headline must be thermal, day 4
    assert "First · Thermal" in tiles_text and "day 4" in tiles_text
    assert "Next · Depth" in tiles_text and "leaf area est" in tiles_text  # not cyl n points
    assert "11 deficit · 7 control plants" in tiles_text  # n on the size tile and the next tile
    # only kept flagged rows are counted: hull volume (n_points is dropped)
    assert [f.children for f in flags] == ["1 flagged"]
    assert "hull volume" in flags[0].title and "n points" not in flags[0].title

    names = [y for tr in fig.data for y in tr.y]
    assert not any("cyl" in y or y in ("n points", "droop index") for y in names)
    assert "size · projected area" in names
    assert fig.layout.yaxis.automargin and fig.layout.showlegend is False
    robust = next(tr for tr in fig.data if tr.name == "robust")
    assert "deficit · %{customdata[3]} control plants" in robust.hovertemplate
    row = list(robust.y).index("leaf area est")
    assert list(robust.customdata[row][2:]) == [11, 7]

    rows = pd.DataFrame(tbl.data)
    assert len(rows) == 5 and not rows["name"].str.contains("cyl|droop|n points").any()  # 8 signals, 3 dropped
    assert "size · projected area" in set(rows["name"])

    labels = [r.children[0].children for r in rec if getattr(r, "title", None)]
    assert labels == ["leaf air dt", "ngrdi", "leaf area est", "height p75"]  # four rows, no projected area

    _curves, _chips, cells = ew._signal("all", "thermal_leaf_air_dt")
    assert "11 deficit · 7 control plants" in _text(html.Div(cells))
    empty, _, _ = ew._signal("all", "depth_cyl_n_points")  # a stale picker value: empty figure
    assert isinstance(empty, go.Figure) and not empty.data


def test_early_warning_ranking_scrolls_in_its_own_box_at_phone_width(detect):
    ew = _load("early_warning")
    lay = ew.layout()
    box = next(c for c in _walk(lay) if getattr(c, "className", None) == "uw-scroll")
    graph = box.children.children
    assert graph.id == "ew-ranking" and graph.style["minWidth"] == f"{ew.RANKING_MIN_PX}px"


def _walk(c):
    yield c
    kids = getattr(c, "children", None)
    for k in kids if isinstance(kids, list | tuple) else [kids] if kids is not None else []:
        if hasattr(k, "to_plotly_json"):
            yield from _walk(k)


# --------------------------------------------------------------------------- Methods & limits
def test_methods_placebo_and_counts_after_the_filter(detect):
    ml = _load("methods")
    fig = ml._placebo("all")
    names = [y for tr in fig.data for y in tr.y]
    assert len(names) == 5  # 8 signals, 3 dropped
    assert not any("cyl" in y or "droop" in y or y.startswith("n points") for y in names)
    assert "size · projected area · 3D" in names and fig.layout.yaxis.automargin

    text = _text(ml.layout())
    assert "10 traits" in text and "(4 noisy 3D traits not shown, 6 shown)" in text  # run chip, after the filter
    # W3: the results have thermal only, and the 3D table now has W3 plants the results lack
    assert "thermal only" in text
    assert "3D: 2 W3 plants in its table, none in the detect run of 2026-11-15 04:10 UTC" in text
    assert "RGB: no W3 table" in text
    value = ml._dates("dates", "24 Aug", "22 Sep").children[1]  # breaks after the dash, not inside a date
    assert value.children[0] == "24 Aug–22 Sep" and value.style == {"whiteSpace": "normal"}


def test_methods_w3_limit_follows_the_results():
    ml = _load("methods")
    on = pd.DataFrame(dict(sensor=["Thermal", "Depth"], group=["W3", "W3"], n_deficit=[9, 30], signal=["t", "d"]))
    value, tip, caveat = ml._w3(on, None)
    assert value == "thermal · 3D" and not caveat and "every sensor" in tip


# --------------------------------------------------------------------------- shared parts
def test_table_and_tile_follow_the_type_and_wrap_rules():
    import _jack_common as jc
    import components

    t = jc.table(pd.DataFrame({"a": ["x"], "b": [1.234]}), {"a": "A", "b": "B"}, id="t")
    assert t.style_cell["fontFamily"] == components.FONT["sans"]  # not the DataTable's monospace
    assert any(r["selector"] == ".export" and "teal-700" in r["rule"] for r in t.css)
    tile = jc.tile("Written notes", "day 3–9", "8 / 26 plants", "sand")
    stat = tile.children[1]  # the unit moves under the value as a whole when the tile is narrow
    assert stat.style["flexWrap"] == "wrap"
    assert stat.children[0].style["whiteSpace"] == "nowrap" and stat.children[1].style["whiteSpace"] == "nowrap"
