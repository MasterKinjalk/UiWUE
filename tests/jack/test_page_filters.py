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
    assert "Size only" in text and "size · " not in text  # the size baseline, in its one name, in the picker
    assert "detect run of 2026-11-15 04:10 UTC" in text  # provenance of the plant counts

    tiles, fig, flags, rec, tbl, _w3 = ew._update("all")
    tiles_text = _text(html.Div(tiles))
    thermal, depth = (_text(t) for t in tiles[:2])
    # the earliest signal overall is a dropped trait (day 2): the thermal tile says day 4, the 3D tile is leaf area
    assert [t.children[0].children[0].children for t in tiles[:2]] == ["Thermal", "3D"]
    assert "day 4" in thermal and "day 6" in depth and "Leaf area" in depth and "cyl" not in depth
    assert "11 deficit · 7 control plants" in tiles_text  # n on the size tile and the 3D tile
    # only kept flagged rows are counted: hull volume (n_points is dropped)
    assert [f.children for f in flags] == ["1 flagged"]
    assert "Volume" in flags[0].title and "n points" not in flags[0].title  # the plain name (components.PLAIN)

    names = [y for tr in fig.data for y in tr.y]
    assert not any("cyl" in y or y in ("n points", "droop index") for y in names)
    assert "Size only" in names
    assert fig.layout.yaxis.automargin and fig.layout.showlegend is False
    robust = next(tr for tr in fig.data if tr.name == "robust")
    assert [tr.name for tr in fig.data] == ["robust"]  # the contract rule is on hover, not a second marker
    assert "contract rule" in robust.hovertemplate
    assert "deficit · %{customdata[4]} control plants" in robust.hovertemplate
    row = list(robust.y).index("Leaf area")
    assert list(robust.customdata[row][3:]) == [11, 7]

    rows = pd.DataFrame(tbl.data)
    assert len(rows) == 5 and not rows["name"].str.contains("cyl|droop|n points").any()  # 8 signals, 3 dropped
    assert "Size only" in set(rows["name"])

    labels = [r.children[0].children for r in _rec_rows(rec)]
    assert labels == ["Leaf − air temperature", "Green–red index", "Leaf area", "Height"]  # no projected area

    _curves, chips = ew._signal("all", "thermal_leaf_air_dt")
    assert "11 deficit · 7 control plants" in _text(html.Div(chips))  # on the 'by hour' chip
    empty, _ = ew._signal("all", "depth_cyl_n_points")  # a stale picker value: empty figure
    assert isinstance(empty, go.Figure) and not empty.data


def test_early_warning_ranking_scrolls_in_its_own_box_at_phone_width(detect):
    ew = _load("early_warning")
    lay = ew.layout()
    box = next(c for c in _walk(lay) if getattr(c, "className", None) == "uw-scroll")
    graph = box.children.children
    assert graph.id == "ew-ranking" and graph.style["minWidth"] == f"{ew.RANKING_MIN_PX}px"


def _rec_rows(rec):
    return [r for r in rec if "dt-rec-row" in (getattr(r, "className", None) or "")]


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
    assert "Size only · 3D" in names and fig.layout.yaxis.automargin

    text = _text(ml.layout())
    assert "10 traits" in text and "(4 noisy 3D traits not shown, 6 shown)" in text  # run chip, after the filter
    # W3: the results have thermal only, and the 3D table now has W3 plants the results lack
    assert "thermal only" in text
    assert "3D: 2 W3 plants in its table, none in the detect run of 2026-11-15 04:10 UTC" in text
    assert "RGB: no W3 table" in text
    value = ml._dates("dates", "24 Aug", "22 Sep").children[1]  # breaks after the dash, not inside a date
    assert value.children[0] == "24 Aug–22 Sep" and value.style == {"whiteSpace": "normal"}


def test_methods_w3_limit_follows_the_results():
    """W3 in every sensor's results is no limit, unless one of them is 3D: each W3 plant lost a leaf to the
    water-content sampling, so its 3D results mix drought and sampling (the detect run now has W3 3D plants)."""
    ml = _load("methods")
    on = pd.DataFrame(dict(sensor=["Thermal", "Depth"], group=["W3", "W3"], n_deficit=[9, 30], signal=["t", "d"]))
    value, tip, caveat = ml._w3(on, None)
    assert value == "thermal · 3D" and caveat and "every sensor" in tip and "leaf cut" in tip
    value, tip, caveat = ml._w3(on[on["sensor"] == "Thermal"], None)
    assert value == "thermal" and not caveat and "leaf cut" not in tip


def test_placebo_chip_reads_the_placebo_table_through_the_kept_trait_filter(detect):
    """components.placebo_chip read detect/placebo unfiltered, so a dropped 3D trait could still drive a 'bed effect'
    chip (here n points is dropped and flagged). It now reads through the one kept-trait definition."""
    import _jack_common as jc
    import components

    assert components.placebo_chip("depth_n_points") == []  # dropped, though the table flags it
    assert [c.children for c in components.placebo_chip("depth_hull_volume")] == ["bed effect"]  # kept and flagged
    assert components.placebo_chip("depth_leaf_area_est") == []  # kept, not flagged
    for s in ("depth_n_points", "depth_cyl_height_p75", "depth_leaf_area_est", "thermal_leaf_air_dt"):
        assert components.kept_trait(s) == jc.kept_signal(s)  # one definition for the chip and the detect pages


def test_early_warning_marks_3d_results_that_pool_the_sampled_w3_plants(detect, monkeypatch):
    """W3 plants each had a leaf cut for water content, so a 3D result of 'W2 + W3' (or W3) carries a caveat chip
    with the W2-only onset on hover; W2 and the other sensors carry none."""
    ew = _load("early_warning")
    for group in ("all", "W3"):  # one chip, in the header, linking to the limit on Methods
        *_, w3 = ew._update(group)
        assert [c.children for c in w3] == ["W3 sampled"]
        assert "W2 alone: 3D leaf area day 6" in w3[0].title and w3[0].href == "/methods#w3-sampled"
    tiles, _fig, _flags, rec, _tbl, w3 = ew._update("W2")
    assert w3 == [] and "W3 sampled" not in _text(html.Div(tiles)) + _text(html.Div(rec))
    # a recovery over 100 % (height came back past its pre-cutoff value) says so on hover
    rec = pd.DataFrame(
        dict(
            plant_id=["W2_A2", "W2_A4"], signal="depth_height_p75", label="height", recovery_ratio=2.1, measurable=True
        )
    )
    monkeypatch.setattr(ew, "load_result", lambda name: rec if name == "recovery" else None)
    rows = _rec_rows(ew.recovery_rows("W2"))
    assert len(rows) == 1 and "210 %" in _text(rows[0]) and "past its own baseline" in rows[0].title


def test_setup_ticks_stack_short_source_tags_and_the_auc_days_stay_level():
    """Sensor comparison's fifteen setup names ran into each other at 1440 px, and the Daily AUC's day labels
    turned on end at phone width."""
    from uiwue.detect import figures

    assert figures.setup_tick("rgb+depth+thermal_r160", 9, 9) == "RGB<br>+ 3D<br>+ IR 160<br>9 · 9"
    assert figures.setup_tick("thermal", 17.0, 13.0) == "IR<br>17 · 13" and figures.setup_tick("thermal") == "IR"
    setups = (("Thermal", "thermal", "logistic", 9), ("RGB + Depth + Thermal", "rgb+depth+thermal", "logistic", 9))
    setups += (("Depth", "depth", "logistic", 9), ("Depth", "depth", "fusion_v2", 114))  # one setup, two plant sets
    bench = pd.DataFrame(
        [
            dict(setup=name, sensors=s, model=model, cost_rank=i, n_sensors=s.count("+") + 1)
            | dict(is_size_baseline=False, group="all", early_auc=0.8, early_ci_low=0.7, early_ci_high=0.9)
            | dict(onset_day=4.0, robust_onset_day=4.0, placebo_auc=0.5, bed_effect=False, fires_before_cutoff=False)
            | dict(n_deficit=n, n_control=9, null_auc_max=0.6)
            for i, (name, s, model, n) in enumerate(setups)
        ]
    )
    fig = figures.cost_performance(bench)
    assert list(fig.layout.xaxis.categoryarray) == ["IR<br>9 · 9", "RGB<br>+ 3D<br>+ IR<br>9 · 9", "3D"]
    assert fig.layout.xaxis.tickangle == 0
    points = [(str(cd[3]), int(cd[4]), int(cd[5])) for tr in fig.data for cd in tr.customdata]
    assert ("RGB + Depth + Thermal", 9, 9) in points  # the full name and each point's own counts on hover
    assert {p for p in points if p[0] == "Depth"} == {("Depth", 9, 9), ("Depth", 114, 9)}
    on = pd.DataFrame([_onset_row("thermal_leaf_air_dt", "all", 4, False, False)])
    days = np.arange(-10, 19)
    auc = pd.DataFrame(
        dict(signal="thermal_leaf_air_dt", group="all", day=days, auc=0.6, ci_low=0.5, ci_high=0.7, n_deficit=9)
        | dict(n_control=9, date=pd.Timestamp("2025-09-03") + pd.to_timedelta(days, unit="D"))
    )
    x = figures.auc_curves(auc, on, ["thermal_leaf_air_dt"]).layout.xaxis
    assert (x.tick0, x.dtick, x.tickangle) == (0, 4, 0)  # −8, −4, 0, 4, 8, 12, 16: day 0 and day 12 labelled


# --------------------------------------------------------------------------- shared parts
def test_table_and_tile_follow_the_type_and_wrap_rules():
    import _jack_common as jc
    import components

    t = jc.table(pd.DataFrame({"a": ["x"], "b": [1.234]}), {"a": "A", "b": "B"}, id="t")
    assert t.style_cell["fontFamily"] == components.FONT["sans"]  # not the DataTable's monospace
    # a flag read as objects (the benchmark and the fusion table concatenated, one without the flag) reads yes / no,
    # as a bool column does, not 'true' / 'false'
    flags = pd.DataFrame({"a": [True, False, True], "b": pd.Series([True, None, False], dtype=object)})
    assert [(r["a"], r["b"]) for r in jc.table(flags, {"a": "A", "b": "B"}).data] == [
        ("yes", "yes"),
        ("no", "–"),
        ("yes", "no"),
    ]
    assert any(r["selector"] == ".export" and "teal-700" in r["rule"] for r in t.css)
    tile = components.stat("Written notes", "day 3–9", "8 / 26 plants", "sand")  # the one tile of every page
    stat = tile.children[1]  # the unit moves under the value as a whole when the tile is narrow (uiwue.css)
    assert tile.className == "uw-tile uw-sand uw-stat-tile" and stat.className == "uw-stat"
    assert [c.children for c in stat.children] == ["day 3–9", "8 / 26 plants"]
    css = (REPO / "app" / "assets" / "uiwue.css").read_text(encoding="utf-8")
    assert ".uw-stat-tile .uw-stat { display: flex; flex-wrap: wrap;" in css
    assert ".uw-stat-tile .uw-stat small { margin-left: 0; line-height: 16px; white-space: nowrap; }" in css
    assert not hasattr(jc, "tile")  # one tile app-wide: the page helpers are gone


# --------------------------------------------------------------------------- the refined pages (finding first)
def _ids(c):
    return {getattr(n, "id", None) for n in _walk(c)} - {None}


def test_early_warning_defaults_to_w2_against_w1_and_reads_the_group_from_the_url(detect):
    """W2 + W3 pooled the leaf-sampled W3 plants, which made 3D leaf area read level with thermal: the page opens on
    W2 against W1 (the comparison the rest of the app uses), and ?group= picks another."""
    ew = _load("early_warning")
    seg = lambda lay: next(n for n in _walk(lay) if getattr(n, "id", None) == "ew-group")
    assert seg(ew.layout()).value == "W2" and next(iter(ew.GROUPS)) == "W2"
    assert seg(ew.layout(group="all")).value == "all"
    assert seg(ew.layout(group="W9")).value == "W2"  # an unknown value falls back to the default


def test_early_warning_tiles_answer_and_link_to_the_signal(detect):
    ew = _load("early_warning")
    tiles, *_ = ew._update("W2")
    labels = [t.children[0].children[0].children for t in tiles]
    assert labels == ["Thermal", "3D", "RGB", "Size only", "Leaf clip"]
    hrefs = {t.children[0].children[0].children: getattr(t, "href", None) for t in tiles}
    assert hrefs["Thermal"] == "/plant-profile?bed=W2&trait=thermal_leaf_air_dt"
    assert hrefs["3D"] == "/threed?bed=W2" and hrefs["RGB"] == "/plant-profile?bed=W2&trait=rgb_ngrdi"
    assert "day 6" in _text(tiles[4]) and "Thermal warns 2 days earlier" in tiles[4].title  # clip day 6, thermal 4
    lay = ew.layout()
    assert {"ew-all", "ew-table"} <= _ids(lay)  # the full table sits behind one disclosure
    nav = next(n for n in _walk(lay) if getattr(n, "className", None) == "uw-next")  # components.next_steps
    assert [a.href for a in nav.children] == ["/sensor-comparison", "/daily-rhythm"]
    nav = next(n for n in _walk(ew.layout(bed="W2", plant="R7", day="9")) if getattr(n, "className", None) == "uw-next")
    assert all(a.href.endswith("?bed=W2&plant=R7&day=9") for a in nav.children)  # the plant chosen earlier rides on


def test_every_methods_anchor_exists_and_the_chips_point_to_them(detect):
    import _jack_common as jc

    ml = _load("methods")
    ids = _ids(ml.layout())
    assert set(jc.METHODS_ANCHORS) <= ids, set(jc.METHODS_ANCHORS) - ids
    ew = _load("early_warning")
    links = [n.href for n in _walk(ew.layout()) if str(getattr(n, "href", "")).startswith("/methods#")]
    links += [c.href for c in ew._update("all")[2] + ew._update("all")[5]]
    assert links and all(h.split("#")[1] in ids for h in links)


def test_setup_scores_put_the_best_on_top_with_size_as_the_floor():
    from uiwue.detect import figures

    rows = (("Thermal", "thermal", 0.96, False), ("Depth", "depth", 0.4, False), ("RGB", "rgb", 0.74, True))
    bench = pd.DataFrame(
        [
            dict(setup=n, sensors=s, model="logistic", early_auc=a, early_ci_low=a - 0.1, early_ci_high=min(a + 0.1, 1))
            | dict(group="W2", is_size_baseline=False, bed_effect=f, fires_before_cutoff=False, placebo_auc=0.5)
            | dict(onset_day=4.0, robust_onset_day=4.0 if not f else np.nan, n_deficit=9, n_control=9)
            for n, s, a, f in rows
        ]
        + [
            dict(setup="Plant size only", sensors="size", model="logistic", early_auc=0.35, early_ci_low=0.1)
            | dict(early_ci_high=0.6, group="W2", is_size_baseline=True, bed_effect=False, fires_before_cutoff=False)
            | dict(placebo_auc=0.5, onset_day=np.nan, robust_onset_day=np.nan, n_deficit=9, n_control=9)
        ]
    )
    fig = figures.setup_scores(bench, "W2", highlight="Thermal")
    assert list(fig.layout.yaxis.categoryarray) == ["3D", "RGB", "Thermal"]  # bottom to top: best on top
    tr = fig.data[0]
    sym = dict(zip(tr.y, tr.marker.symbol, strict=True))
    assert sym["RGB"] == "circle-open" and sym["Thermal"] == "circle"  # flagged drawn open
    col = dict(zip(tr.y, tr.marker.color, strict=True))
    assert col["Thermal"] != col["3D"]  # only the best buy is highlighted
    assert any(s.x0 == 0.35 and s.line.dash == "dash" for s in fig.layout.shapes)  # size only: the floor
    assert "chance" in "".join(fig.layout.xaxis.ticktext)
