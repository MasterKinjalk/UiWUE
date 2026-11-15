"""Phone width (DESIGN 8): no page scrolls sideways. Side panels stack, toolbars wrap, and wide tables and grids
scroll inside their own box (a ``uw-scroll`` wrapper), never the page.

Checked without a browser: the shared styles (app/assets/uiwue.css), and the component trees of the pages (layouts
and first callbacks) rendered against the real derived tables in a fresh process, because pages register their
callbacks globally and the shell is built once per interpreter. The narrowest phone in use is 320 px wide, so
with its two 16 px gutters a page has 288 px: a fixed width, a minimum or a grid track wider than that (less the
padding of the panels around it) pushes the page sideways, unless it sits in a scroll box or is capped with
``min(…, 100%)``.
"""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
CSS = (ROOT / "app" / "assets" / "uiwue.css").read_text(encoding="utf-8")
PHONE = 320 - 2 * 16  # px of content on the narrowest phone
PHONE_MEDIA = "@media (max-width: 700px)"
INSET = {"uw-panel": 34, "uw-card": 32, "uw-tile": 32, "uw-empty": 36}  # padding + border a box takes off
IN_SCROLL_BOX = {".uw-heat"}  # wide by design; the page test checks it is always drawn inside .uw-scroll


# ------------------------------------------------------------------ helpers
def css_rules(css):
    """[(media or None, selector, {property: value})] of a stylesheet with @media one level deep."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    out, media, head, buf = [], None, None, ""
    for ch in css:
        if ch == "{":
            if buf.strip().startswith("@media"):
                media = buf.strip()
            else:
                head = buf.strip()
            buf = ""
        elif ch == "}":
            if head is None:  # the end of an @media block
                media = None
            else:
                pairs = (p.split(":", 1) for p in buf.split(";") if ":" in p)
                decl = {k.strip(): v.strip() for k, v in pairs}
                out += [(media, s.strip(), decl) for s in head.split(",")]
                head = None
            buf = ""
        else:
            buf += ch
    return out


RULES = css_rules(CSS)


def decl(selector, media=None):
    found = [d for m, s, d in RULES if s == selector and m == media]
    assert found, f"no rule for {selector} ({media or 'every width'})"
    return {k: v for d in found for k, v in d.items()}


def minimums(prop, value):
    """The px widths one declaration forces on its box whatever the room: a width or min-width, a flex basis that
    may not shrink, a grid track's minimum (max-content counts as unbounded). min(…, 100%) caps a value at the
    room there is, so a capped value forces nothing."""
    prop = re.sub(r"(?<!^)(?=[A-Z])", "-", prop).lower()  # minWidth -> min-width
    value = f"{value}px" if isinstance(value, (int, float)) else str(value)  # React reads a bare number as px
    value = re.sub(r"min\([^()]*\)", "", value)
    if prop in ("width", "min-width"):
        if value.strip() == "max-content":
            return [float("inf")]
        m = re.fullmatch(r"\s*([\d.]+)px\s*", value)
        return [float(m.group(1))] if m else []
    if prop == "flex":
        parts = value.split()
        return [float(parts[2][:-2])] if len(parts) == 3 and parts[1] == "0" and parts[2].endswith("px") else []
    if prop == "grid-template-columns":
        return [float(w) for w in re.findall(r"minmax\(\s*([\d.]+)px", value)]
    return []


def nodes(tree, above=()):
    """(component, its ancestors) for every component of a rendered tree (Dash JSON); figures are skipped."""
    if isinstance(tree, list):
        for t in tree:
            yield from nodes(t, above)
    elif isinstance(tree, dict) and {"props", "type"} <= set(tree):
        yield tree, above
        for key, value in tree["props"].items():
            if key not in ("figure", "style", "options", "data", "columns"):
                yield from nodes(value, (*above, tree))


def classes(node):
    return set((node["props"].get("className") or "").split())


def scrolls(node):
    return "uw-scroll" in classes(node) or (node["props"].get("style") or {}).get("overflowX") in ("auto", "scroll")


def too_wide(tree):
    """Fixed widths in a rendered tree that do not fit a 320 px phone and are not in a scroll box of their own."""
    out = []
    for node, above in nodes(tree):
        if any(scrolls(a) for a in above):
            continue
        room = PHONE - sum(INSET.get(c, 0) for a in above for c in classes(a))
        for prop, value in (node["props"].get("style") or {}).items():
            if any(w > room for w in minimums(prop, value)):
                out.append((node["props"].get("id") or node["type"], prop, value, room))
    return out


# ------------------------------------------------------------------ the helpers themselves
def test_the_width_reading_counts_only_what_cannot_shrink():
    assert minimums("minWidth", "280px") == [280] and minimums("minWidth", "min(280px, 100%)") == []
    assert minimums("width", 300) == [300] and minimums("width", "max-content") == [float("inf")]
    assert minimums("flex", "0 0 300px") == [300] and minimums("flex", "1 1 320px") == []
    assert minimums("gridTemplateColumns", "repeat(auto-fit, minmax(320px, 1fr))") == [320]
    assert minimums("gridTemplateColumns", "repeat(auto-fit, minmax(min(320px, 100%), 1fr))") == []
    panel = {"type": "Section", "props": {"className": "uw-panel", "children": []}}
    box = {"type": "Div", "props": {"style": {"minWidth": "280px"}}}
    panel["props"]["children"] = [box]
    assert too_wide(panel) == [("Div", "minWidth", "280px", PHONE - 34)]  # 280 px fits the page, not a panel
    scroll = {"type": "Div", "props": {"className": "uw-scroll", "children": panel}}
    assert too_wide(scroll) == []


# ------------------------------------------------------------------ shared styles
def test_a_select_shrinks_to_its_row_and_never_pushes_the_page():
    """A select is as wide as its longest option (Bed map's signals: 381 px, wider than a 390 px phone's 358 px
    row): the label and the select may shrink to the row, the option text is cut with an ellipsis."""
    for selector in (".uw-label", ".uw-select"):
        d = decl(selector)
        assert (d.get("min-width"), d.get("max-width")) == ("0", "100%"), selector
    assert decl(".uw-select").get("text-overflow") == "ellipsis"


def test_wide_tables_and_grids_scroll_in_their_own_box():
    assert decl(".uw-scroll").get("overflow-x") == "auto"


def test_phone_width_stacks_side_panels_and_reflows_long_readout_grids():
    assert decl(".uw-side", PHONE_MEDIA).get("max-width") == "none"
    # Methods' scope (7 readouts, "24 Aug–22 Sep") takes two columns; one row of three (Daily rhythm's plan
    # cards) keeps three
    grid = decl(".uw-readouts-3:has(> :nth-child(4))", PHONE_MEDIA).get("grid-template-columns")
    assert grid == "repeat(2, minmax(0, 1fr))"
    assert decl(".uw-readouts .uw-section", PHONE_MEDIA).get("white-space") == "normal"  # wraps in its cell


def test_home_coverage_tracks_and_axis_line_up_on_a_phone():
    """A long module name ("Environment Yuming") widened its name column and pushed its track right of the others,
    and "15 Sep" ran into "22 Sep": the name column is fixed (the name wraps, the owner gives way) and the axis keeps
    the three event dates."""
    assert decl(".uw-cover-name", PHONE_MEDIA) == {"flex-basis": "120px", "min-width": "0"}
    assert decl(".uw-cover-name > span:not(.uw-dot)", PHONE_MEDIA) == {"display": "none"}  # the owner, not the dot
    for end in ("first-child", "last-child"):
        assert decl(f".uw-cover-axis > span:{end}", PHONE_MEDIA) == {"display": "none"}


def test_dash_components_draw_with_the_tokens_and_no_shadow():
    """Dash 4 draws its slider, dropdown and tooltip with --Dash-* colours, a purple accent (#7f4bc4) by default:
    the slider on Bed map and Export was purple. They read the tokens instead, and cast no shadow (DESIGN 2.2)."""
    d = decl("html:root")
    assert d["--Dash-Fill-Interactive-Strong"] == "var(--uw-teal-700)"
    assert all(v.startswith("var(--uw-") for k, v in d.items() if not k.startswith("--Dash-Shading"))
    assert d["--Dash-Shading-Strong"] == d["--Dash-Shading-Weak"] == "transparent"
    assert "#7f4bc4" not in CSS.lower()


def test_no_shared_style_forces_a_box_wider_than_a_phone():
    wide = [
        (m, s, p, v)
        for m, s, d in RULES
        for p, v in d.items()
        if s not in IN_SCROLL_BOX and any(w > PHONE for w in minimums(p, v))
    ]
    assert not wide


# ------------------------------------------------------------------ the pages, rendered on the real tables
PAGES = textwrap.dedent(
    """
    import json, sys
    sys.path[:0] = ["app", "src"]
    import app  # noqa: F401  (the shell registers every page)
    from plotly.utils import PlotlyJSONEncoder
    m = {n: sys.modules[f"pages.{n}"] for n in ("home", "plant_profile", "bed_map", "daily_rhythm", "irrigation",
                                                  "sensor_comparison", "export")}
    pp, bm, dr, ir, ex = (m[n] for n in ("plant_profile", "bed_map", "daily_rhythm", "irrigation", "export"))
    out = {f"{n}.layout": mod.layout() for n, mod in m.items()}
    for bed, plant in (("W2", "R7"), ("W3", "B3")):
        out[f"plant_profile.show_plant {bed} {plant}"] = pp.show_plant(bed, plant)
        out[f"bed_map.show_side {bed} {plant}"] = bm.show_side({"bed": bed, "plant": plant}, "thermal_leaf_ref_dt", 15, 9)
    out["plant_profile.show_detail"] = pp.show_detail("W2", "R7", "thermal_leaf_ref_dt")
    out["bed_map.show_map"] = bm.show_map("thermal_leaf_ref_dt", 15, 9, {"bed": "W2", "plant": "R7"})[3:]
    out["daily_rhythm.update_separability"] = dr.update_separability("diurnal_score_ref")
    out["daily_rhythm.update_schedules"] = dr.update_schedules("control_ref")
    rule = ir.best_rule(ir.schedules_table()) or "control_ref"
    out["irrigation.update"] = ir.update(rule, ir.latest_day(rule, ir.recommended(rule)), None)[:5]
    out["export.preview"] = ex.preview("traits", ["W1", "W2", "W3"], ["thermal_", "depth_"], [-10, 19])
    out["profile_views.preview_table"] = ex.pv.preview_table(ex.pv.traits())
    print("JSON " + json.dumps(out, cls=PlotlyJSONEncoder))
    """
)


@pytest.fixture(scope="module")
def pages():
    if not (ROOT / "derived" / "diurnal" / "schedules.parquet").exists():
        pytest.skip("needs the derived tables")
    done = subprocess.run(
        [sys.executable, "-c", PAGES], cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
    )
    line = next((x for x in done.stdout.splitlines() if x.startswith("JSON ")), None)
    assert done.returncode == 0 and line, done.stderr[-3000:]
    return json.loads(line[5:])


def test_no_page_forces_a_box_wider_than_a_phone(pages):
    """Plant profile's small multiples used minmax(320px, 1fr) and its trait picker a 280 px minimum inside a panel:
    both pushed a 320 px phone sideways. Now capped with min(…, 100%)."""
    wide = {name: w for name, tree in pages.items() if (w := too_wide(tree))}
    assert not wide


def test_wide_grids_and_tables_sit_in_their_own_scroll_box(pages):
    found = {"bed grid": 0, "heatmap": 0, "table": 0}
    for name, tree in pages.items():
        for node, above in nodes(tree):
            kind = (
                "bed grid"
                if node["props"].get("id") in ("bm-W1", "bm-W2", "bm-W3")
                else "heatmap"
                if "uw-heat" in classes(node)
                else "table"
                if "uw-table" in classes(node)
                else None
            )
            if kind:
                found[kind] += 1
                assert any(scrolls(a) for a in above), f"{name}: {kind} outside a scroll box"
    assert found["bed grid"] == 3 and found["heatmap"] and found["table"] >= 3, found


def test_charts_too_wide_for_a_phone_scroll_in_their_own_box(pages):
    """Daily rhythm's five day-profile facets ran their labels into each other at phone width: the chart keeps a
    minimum width and scrolls inside its panel instead. (Sensor comparison's setups now run down the page, one row
    each, so that chart fits a phone without a scroll box.)"""
    want = {"dr-profile": "daily_rhythm.layout"}
    for gid, name in want.items():
        hit = [(n, above) for n, above in nodes(pages[name]) if n["props"].get("id") == gid]
        assert len(hit) == 1, gid
        graph, above = hit[0]
        assert float(graph["props"]["style"]["minWidth"].removesuffix("px")) > PHONE, gid
        assert any(scrolls(a) for a in above), f"{gid} outside a scroll box"


def test_plant_profile_header_and_grid_wrap_on_a_phone(pages):
    tree = pages["plant_profile.layout"]
    chips = next(n for n, _ in nodes(tree) if n["props"].get("id") == "pp-chips")
    assert chips["props"]["style"]["flexWrap"] == "wrap"
    grids = [
        n["props"]["style"]["gridTemplateColumns"]
        for n, _ in nodes(pages["plant_profile.show_plant W2 R7"])
        if "gridTemplateColumns" in (n["props"].get("style") or {})
    ]
    assert grids and all("minmax(min(320px, 100%)" in g for g in grids)
