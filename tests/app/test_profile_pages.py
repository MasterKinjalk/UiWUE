"""Plant profile, Bed map and Export: the data and figure helpers on synthetic tables, and the pages themselves
rendered in a fresh process against the real derived tables (a fresh process, because pages register
their callbacks globally and the shell is built once per interpreter)."""

import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import profile_views as pv
import threed_canopy as tc

from uiwue import core
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


# ------------------------------------------------------------------ 3D canopy: links, leaf-area readout, export
def leaf(bed, plant, day):
    """Leaf area of the synthetic 3D plants: 100 cm² unless noted."""
    if (bed, plant) == ("W1", "A2") and day in (1, 2):
        return 70.0  # a control plant below −20 % two days running: the rule's false alarm
    if (bed, plant) == ("W2", "J5") and day > 3:
        return 100.0 - 8.0 * (min(day, 12) - 3) if day <= 12 else 50.0  # collapses, partly comes back
    return 100.0


@pytest.fixture
def threed(tmp_path, monkeypatch):
    """derived/ holding only a 3D trait table in the committed format: controls W1 A2 (dips) and B3, deficit
    W2 J5 (collapses) and D3 (its baseline runs all failed QC), days −4 to 16, slots 9–18 h."""
    core.catalogue()  # cache the real catalogue before derived/ moves, so labels and units resolve
    monkeypatch.setattr("uiwue.core.DERIVED", tmp_path)
    monkeypatch.setattr(pv, "DERIVED", tmp_path)
    rows = [
        {
            "bed": bed,
            "plant": plant,
            "date": core.CUTOFF + timedelta(days=day),
            "hour": hour,
            "depth_leaf_area_est": leaf(bed, plant, day),
            "depth_height_p75": 50.0,
            "depth_hull_volume": 500.0,
            "depth_height_p50": 1.0,  # a removed trait: never exported
            "depth_qc_pass": not ((bed, plant) == ("W2", "D3") and day < 0),
        }
        for bed, plant in (("W1", "A2"), ("W1", "B3"), ("W2", "J5"), ("W2", "D3"))
        for day in range(-4, 17)
        for hour in (9, 12, 15, 18)
    ]
    write(tmp_path, "threed", "traits_runs", pd.DataFrame(rows))
    tc.clear_caches()
    yield tmp_path
    tc.clear_caches()


def test_threed_links_carry_bed_and_plant():
    assert pv.threed_href("W2", "J5") == "/threed?bed=W2&plant=J5"
    assert pv.span_text(10, 12) == "Day 10–12 · 13–15 Sep"


def test_leaf_area_readout_is_change_from_the_own_baseline(threed):
    j5 = pv.structure_summary("W2", "J5")
    assert j5["end"]["value"] == pytest.approx((44 + 36 + 28) / 3 - 100) and j5["end"]["n"] == 3
    assert j5["end"]["days"] == (10, 12) and j5["after"]["days"] == (15, 16)
    assert j5["after"]["value"] == pytest.approx(-50.0) and j5["first_below"] == 7  # −24 % on 6, −32 % on 7
    a2 = pv.structure_summary("W1", "A2")
    assert a2["first_below"] == 2 and a2["end"]["value"] == pytest.approx(0.0)
    assert j5["alarms"] == (1, 2)  # the rule flags a control plant: shown as values, not as an alert
    d3 = pv.structure_summary("W2", "D3")
    assert not d3["has_baseline"] and d3["end"]["value"] is None and d3["first_below"] is None
    assert pv.structure_summary("W3", "B3") is None and pv.structure_summary("W2", "ZZ") is None


def test_no_3d_table_means_no_readout_and_no_export_option(derived):
    tc.clear_caches()
    assert pv.structure_summary("W2", "J5") is None
    assert pv.THREED_DAYS not in pv.export_tables() and pv.read_export(pv.THREED_DAYS) is None


def test_export_offers_the_3d_plant_day_table(threed):
    assert list(pv.export_tables())[:2] == ["traits", pv.THREED_DAYS]
    t = pv.read_export(pv.THREED_DAYS)
    kept = [x.column for x in tc.TRAITS.values()]
    assert [c for c in t.columns if c.startswith("depth_")] == [c + s for c in kept for s in ("", "_base", "_pct")]
    assert {"bed", "plant", "treatment", "date", "days_since_cutoff", "n_runs", "n_pass", "slots"} <= set(t.columns)
    day12 = t[(t.plant == "J5") & (t.days_since_cutoff == 12)].iloc[0]
    assert day12.depth_leaf_area_est_pct == pytest.approx(-72.0) and day12.n_pass == 4
    assert t.loc[t.plant == "D3", "depth_leaf_area_est_pct"].isna().all()  # no baseline: kept, change empty
    f = pv.filter_export(t, ["W2", "W3"], "deficit", ["depth_"], (0, 12))
    assert set(f.bed) == {"W2"} and pv.export_counts(f) == (26, 2, 13)
    assert not [c for c in pv.filter_export(t, None, "all", [], None).columns if c.startswith("depth_")]
    d = pv.data_dictionary(t.columns, pv.THREED_DAYS).set_index("column")
    assert (
        d.loc["depth_leaf_area_est_pct", "unit"] == "%"
        and "own baseline" in d.loc["depth_leaf_area_est_pct", "description"]
    )
    assert d.loc["depth_leaf_area_est", "unit"] == "cm²" and "QC-pass" in d.loc["depth_leaf_area_est", "description"]
    assert d.loc["depth_height_p75_base", "unit"] == "mm" and d.loc["n_pass", "unit"] == "count"
    assert not d.description.str.contains("not in the catalogue").any()


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

        # 3D canopy: links in, the leaf-area readout, the Home tile and the export table
        import json
        import dash
        import profile_views as pv
        import threed_canopy as tc
        from components import C
        text = lambda c: json.dumps(c.to_plotly_json(), default=str, ensure_ascii=False)
        df = pv.traits()
        three = [a for a in pp.actions(df, "W2", "R7") if a.children == "3D canopy"]
        assert len(three) == 1 and three[0].href == "/threed?bed=W2&plant=R7"
        links = bm.show_side({"bed": "W2", "plant": "R7"}, "depth_projected_area", 15, 9)[-1].children
        assert "/threed?bed=W2&plant=R7" in [a.href for a in links]
        card = pp.side(df, "W2", "R7")[1]
        assert card.className == "uw-card uw-sand" and "Leaf area" in text(card) and "uw-mark" not in text(card)
        r7 = tc.windows().set_index(["bed", "plant", "trait"]).loc[("W2", "R7", "leaf_area_est")]
        assert pv.fmt(r7.trough, "", 0) in text(card) and pv.fmt(r7.rewatered, "", 0) in text(card)
        alerts = tc.structure_alerts()
        if alerts.loc[alerts.treatment == "control", "first_irrigate"].notna().any():
            assert "not used as an alert" in card.title  # the rule flags controls: values only, caveat on hover
        assert pp.leaf_area_card(df, "W3", "B3", "uw-sand").className == "uw-empty"  # W3 is not in the readout
        base = tc.plant_days().groupby(["bed", "plant"])["leaf_area_est_base"].first()
        for bed, plant in base[base.isna()].index[:1]:  # a plant without a baseline, if the table has one
            card = text(pp.leaf_area_card(df, bed, plant, "uw-sand"))
            assert "no baseline" in card and "–" in card

        home = sys.modules["pages.home"]
        assert not [p for p in home.GROUPS if p.startswith("/threed-")]
        if any(p["path"] == "/threed" for p in dash.page_registry.values()):  # the 3D canopy page is in
            tiles = [t for t in home.page_tiles().children if t.children[0].children == "3D"]
            assert [t.href for t in tiles] == ["/threed"] and tiles[0].className == "uw-pagetile uw-mint"
        real = dict(dash.page_registry)
        try:  # the registry once the 3D pages are one: Home shows one 3D tile, forest on mint (DESIGN 2.1)
            dash.page_registry.clear()
            dash.page_registry.update({k: v for k, v in real.items() if not v["path"].startswith("/threed")})
            dash.page_registry["pages.threed"] = dict(
                real["pages.home"], module="pages.threed", path="/threed", relative_path="/threed",
                name="3D canopy", order=25, description="3D",
            )
            tiles = [t for t in home.page_tiles().children if t.children[0].children == "3D"]
            assert len(tiles) == 1 and tiles[0].href == "/threed" and tiles[0].className == "uw-pagetile uw-mint"
            assert tiles[0].children[0].style["background"] == C["forest-700"]
        finally:
            dash.page_registry.clear()
            dash.page_registry.update(real)

        assert "3D canopy · plant-days" in text(ex.layout())
        days = [-10, 19]
        csv = ex.download_csv(1, pv.THREED_DAYS, ["W1", "W2", "W3"], "all", ["depth_"], days)["content"]
        head = csv.splitlines()[0]
        assert head.startswith("bed,plant,treatment,date,days_since_cutoff") and "depth_leaf_area_est_pct" in head
        assert "depth_droop_index" not in head and "depth_projected_area" not in head  # the three kept traits only
        assert len(csv.splitlines()) == 1 + len(tc.plant_days())
        tiles, _ = ex.preview(pv.THREED_DAYS, ["W1", "W2"], "all", ["depth_"], days)
        n_plants = len(tc.runs()[["bed", "plant"]].drop_duplicates())
        assert tiles[1].children[1].children == str(n_plants)  # every W1 and W2 plant of the 3D table
        dictionary = ex.download_dictionary(1, pv.THREED_DAYS, ["W1", "W2"], "all", ["depth_"], days)["content"]
        assert "own baseline" in dictionary
        print("pages ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=180, check=False
    )
    assert done.returncode == 0 and "pages ok" in done.stdout, done.stderr[-2000:]


# ------------------------------------------------------------------ 3D on the profile: the three kept traits only
def test_profile_3d_small_multiples_are_the_kept_3d_traits():
    """The app shows three 3D traits (threed_canopy.TRAITS: leaf area, height p75, hull volume). Projected area is
    no small multiple: it is the size-only baseline, the `size` chip beside the stress probability (DESIGN 7.6)."""
    kept = [t.column for t in tc.TRAITS.values()]
    assert kept == ["depth_leaf_area_est", "depth_height_p75", "depth_hull_volume"]
    assert [c for tag, c, _ in pv.HEADLINES if tag == "3D"] == kept
    assert SIZE not in {c for _, c, _ in pv.HEADLINES}
    pv._SERIES.clear()  # the series cache is keyed by id(df): start clean
    df = traits()
    for c in kept:
        df[c] = df[SIZE]  # 100 on the baseline days −3 and −2, half of it on days 2 and 7 (W3 not imaged)
    for c in kept:
        assert pv.change_unit(c) == "%"  # sizes: % of the plant's own baseline
        plant = pv.small_multiple(df, "W2", "J5", c).data[-1]
        assert list(plant.x) == [-3, -2, 2, 7] and list(plant.y) == pytest.approx([0, 0, -50, -50])


def test_trait_pickers_offer_only_the_kept_3d_traits():
    df = traits()
    df["depth_leaf_area_est"] = df[SIZE]
    df["depth_droop_index"] = 1.0
    depth = [o["value"] for o in pv.trait_choices(df) if o["value"].startswith("depth_")]
    assert depth == ["depth_leaf_area_est"]  # not projected area (the size baseline), not droop index (dropped)


def test_3d_canopy_link_only_for_plants_on_the_3d_canopy_page():
    """The link opens the plant itself: the 3D canopy page shows every plant of its table, W3 ones on their own
    (threed_canopy.SHOWN_BEDS), while its bed numbers stay W1 against W2 (threed_canopy.BEDS)."""
    assert pv.THREED_BEDS == tc.BEDS and pv.THREED_LINK_BEDS == tc.SHOWN_BEDS
    df = traits()
    assert not pv.has_depth(df, "W2", "J5")  # projected area alone (a dropped trait) is no reason for the link
    df["depth_leaf_area_est"] = 1.0  # every bed imaged
    assert [pv.has_depth(df, bed, "J5") for bed in pv.BEDS] == [bed in tc.SHOWN_BEDS for bed in pv.BEDS]
    assert not pv.has_depth(df, "W2", "Q8")  # no runs


def test_3d_panels_links_readout_and_pickers_on_the_real_tables():
    """Plant profile, Bed map and Daily rhythm on the real tables: the kept 3D traits only, and no 3D canopy link
    that would open another plant."""
    script = textwrap.dedent(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401  (the shell registers every page)
        import profile_views as pv
        pp = sys.modules["pages.plant_profile"]
        df = pv.traits()
        three = [label for tag, _, label in pv.HEADLINES if tag == "3D"]
        for bed, plant in (("W2", "R7"), ("W1", "J5")):
            names = [p.to_plotly_json()["props"]["aria-label"].split(":")[0]
                     for grid in pp.multiples(df, bed, plant) for p in grid.children]
            assert [n for n in names if n in three] == three and "Projected area" not in names, names
        assert [a.children for a in pp.actions(df, "W2", "R7")][0] == "3D canopy"
        card = pp.leaf_area_card(df, "W3", "B3", "uw-sand")
        if "W3" not in pv.THREED_BEDS and pv.sensor_state(df, "W3", "B3", "depth_leaf_area_est")[0] == "ok":
            links = {a.children: a.href for a in pp.actions(df, "W3", "B3")}
            assert links["3D canopy"] == "/threed?bed=W3&plant=B3"  # the 3D canopy page shows W3 plants alone
            assert card.className == "uw-empty" and " only, so none for W3 B3" in card.title, card.title

        def find(c, id_):  # the component with this id in a layout
            if getattr(c, "id", None) == id_:
                return c
            kids = getattr(c, "children", None)
            for k in kids if isinstance(kids, (list, tuple)) else [kids]:
                if hasattr(k, "to_plotly_json") and (hit := find(k, id_)) is not None:
                    return hit
            return None

        bm, dr = sys.modules["pages.bed_map"], sys.modules["pages.daily_rhythm"]
        for page, picker in ((pp, "pp-trait"), (bm, "bm-trait"), (dr, "dr-trait")):
            options = find(page.layout(), picker).options
            depth = {o["value"] for o in options if o["value"].startswith("depth_")}
            assert depth == pv.THREED_COLUMNS, (picker, depth)  # the three kept 3D traits, no dropped ones
        print("3d ok")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=180, check=False
    )
    assert done.returncode == 0 and "3d ok" in done.stdout, done.stderr[-2000:]
