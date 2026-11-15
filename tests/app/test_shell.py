"""The shell: the grouped nav (desktop row and phone menu), Home's live finding, and the shared parts every page
can use (stat tile, next steps, outlined buttons, hover tips)."""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
CSS = (ROOT / "app" / "assets" / "uiwue.css").read_text(encoding="utf-8")
JS = (ROOT / "app" / "assets" / "nav.js").read_text(encoding="utf-8")
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "src")]

import components


def run(script):
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0 and "ok" in done.stdout.split(), done.stderr[-3000:]


def test_nav_groups_name_every_page_once_and_never_drop_a_new_one():
    run(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app, components
        groups = app.nav_groups()
        paths = [p["path"] for _, _, found in groups for p in found]
        want = [p["path"] for p in app.pages if p["path"] != "/"]
        assert sorted(paths) == sorted(want) and len(paths) == len(set(paths)), paths
        assert [g for g, _, _ in groups][:4] == ["See", "Detect", "Act", "Data"]
        order = [path for _, _, items in components.NAV for path, _ in items]
        assert [p for p in paths if p in order] == [p for p in order if p in paths]  # the path's order
        extra = dict(app.pages[0], path="/new", relative_path="/new", name="New page", order=99)
        assert app.nav_groups([*app.pages, extra])[-1][0] == "More"
        print("ok")
        """
    )


def test_nav_is_real_links_with_a_menu_button_for_phones():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app
        tree = json.dumps(app.navbar.to_plotly_json(), default=lambda o: o.to_plotly_json())
        links = [p["relative_path"] for p in app.pages if p["path"] != "/"]
        for href in links:  # once in the row, once in the menu
            assert tree.count(f'"href": "{href}"') == 2, href
        assert '"id": "uw-menu"' in tree and "uw-menu-btn" in tree and "uw-menu-current" in tree
        assert tree.count('"role": "group"') == 2 * len(app.nav_groups())
        print("ok")
        """
    )


def test_nav_links_carry_the_selection_to_pages_that_read_it():
    run(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app
        carry = app.carried()
        assert carry.get("/plant-profile") == ["bed", "plant", "day"] and "/threed" in carry
        assert "/export" not in carry and "/" not in carry and "/methods" not in carry  # filters and Home: no
        fake = [dict(app.pages[0], path="/x", relative_path="/x", layout=lambda bed=None, plant=None, **_: None)]
        assert app.carried(fake) == {"/x": ["bed", "plant"]}
        assert "window.UW_CARRY = " in app.app.index_string
        print("ok")
        """
    )
    assert "UW_CARRY" in JS and "_dashprivate_pushstate" in JS  # a click navigates as dcc.Link does


def test_nav_script_marks_the_page_and_folds_the_menu():
    assert "markPage" in JS and 'setAttribute("aria-current", "page")' in JS
    assert 'removeAttribute("open")' in JS and "Escape" in JS


def test_the_menu_takes_over_below_the_width_where_the_row_wraps():
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    block = css[css.index("@media (max-width: 1040px)") :]
    assert ".uw-navrow { display: none; }" in block and ".uw-menu { display: block; }" in block
    assert ".uw-menu { display: none; }" in css  # hidden at desktop width
    assert re.search(r"\.uw-menu-btn \{[^}]*min-height: 44px", css)  # a 44 px target


def test_home_finding_is_read_live_from_the_tables():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        import threed_canopy as tc
        home = sys.modules["pages.home"]
        text = lambda c: json.dumps(c.to_plotly_json(), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        o = tc.onsets()
        tiles = home.onset_tiles()
        if o is None:
            assert "no table yet" in text(tiles[0])
        else:
            shown = [t.children[1].children[0].children for t in tiles]
            ok = ~o.is_size_baseline & ~o.bed_effect & ~o.fires_before_cutoff & o.robust_onset_day.notna()
            first = o[ok].iloc[0]
            assert shown[0] == f"day {int(first.robust_onset_day)}"
            assert [int(s.split()[1]) for s in shown[:-1]] == sorted(int(s.split()[1]) for s in shown[:-1])
            assert all(t.href in ("/early-warning", "/threed") for t in tiles)
        h = tc.headline()
        leaf = home.leaf_tile()
        if h is not None:
            mean = h["leaf_area_est"]["peak_to_trough"]["deficit"]["mean"]
            assert leaf.children[1].children[0].children == home.signed_pct(mean) and leaf.href == "/threed"
            if tc.bed_effects()["leaf_area_est"]["chip"]:
                assert "bed effect" in text(leaf)
        page = text(home.layout())
        assert "W1" in page and "72 plants" in page and "No water" in page
        cover = home.coverage()  # the table inventory waits behind one closed control
        assert cover.id == "home-coverage" and cover.open is False and "uw-cover" in text(cover)
        print("ok")
        """
    )


def test_home_page_tiles_follow_the_nav_groups_and_say_what_each_page_is_for():
    run(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app, components
        home = sys.modules["pages.home"]
        cols = home.page_tiles().children
        assert [c.children[0].children for c in cols][:4] == ["See", "Detect", "Act", "Data"]
        for t in home.page_links():
            assert t.children[1].children[1].children, t.href  # a few words on what the reader learns
        three = [t for t in home.page_links() if t.href == "/threed"]
        assert three and three[0].children[0].style["background"] == components.C["forest-700"]
        print("ok")
        """
    )


def test_stat_tile_and_next_steps():
    tile = components.stat("Thermal sees stress", "day 4", "7 Sep", "gray", hover="how", sub="leaf − air", href="/x")
    assert tile.href == "/x" and "uw-tile-link" in tile.className and tile.title == "how"
    plain = components.stat("Label", "−61", "%", "sand")
    assert plain.className == "uw-tile uw-sand uw-stat-tile" and plain.children[2] is None
    nav = components.next_steps(("Profile", "/plant-profile?bed=W2&plant=R7"), ("Methods", "/methods", "why"))
    first, second = nav.children
    assert "uw-btn-primary" in first.className and "uw-btn-outline" in second.className and second.title == "why"


def test_buttons_are_outlined_unless_primary_and_tips_wrap():
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    btn = re.search(r"\n\.uw-btn \{([^}]*)\}", css).group(1)
    assert "border: 1px solid var(--uw-teal-700)" in btn and "color: var(--uw-teal-700)" in btn
    tip = re.search(r"\.uw-tip \{([^}]*)\}", css).group(1)
    assert "max-width: min(320px" in tip and "font-size: 13px" in tip
    assert re.search(r"\na \{ color: var\(--uw-teal-700\);", css)  # no off-token theme green on links


# ---------------------------------------------------------------- one of each shared part, app-wide
def test_one_stat_tile_with_pairs_a_missing_state_and_plain_chips_inside_a_link():
    pair = components.stat("After re-watering", None, tint="sand", pairs=[("+6", "% height"), ("−49", "% leaf area")])
    row = pair.children[1]
    assert row.className == "uw-stat-row" and [n.children[0].children for n in row.children] == ["+6", "−49"]
    linked = components.stat("Leaf area lost", "−61", "%", "sand", chips=[components.bed_effect_chip("why")], href="/x")
    chip = linked.children[0].children[1]
    assert type(chip).__name__ == "Span" and "uw-chip-caveat" in chip.className  # no link inside the tile's link
    plain = components.stat("Leaf area lost", "−61", "%", "sand", chips=[components.bed_effect_chip("why")])
    assert plain.children[0].children[1].href == "/methods#bed-effect"  # outside a link the chip opens its entry
    gone = components.stat_missing("Leaf clip", "no clip onset")
    assert "uw-stat-missing" in gone.className and gone.title == "no clip onset" and gone.children[1].children == "–"


def test_chips_link_to_their_methods_entry_and_size_only_has_one_name():
    c = components.chip("W3 sampled", "why", anchor="w3-sampled")
    assert c.href == "/methods#w3-sampled" and c.className == "uw-chip uw-chip-caveat"
    assert components.chip("rule", "why", "rule").className == "uw-chip uw-chip-rule"
    size = components.size_chip()
    assert size.children.startswith("Size only · ") and size.href == "/methods#size-baseline"
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    assert "a.uw-chip-caveat:hover" in css and "a.uw-chip-rule" in css  # a linked chip keeps its colours


def test_one_plain_name_per_signal_with_the_technical_name_on_hover():
    assert (
        components.plain("depth_height_p75") == "Height" and components.tech("depth_height_p75") == "Canopy height, p75"
    )
    assert components.plain("rgb_ngrdi") == "Green–red index" and components.tech("rgb_ngrdi") == "NGRDI"
    assert components.plain("thermal_leaf_air_dt") == "Leaf − air temperature"
    assert components.plain("not_a_signal", "given label") == "given label"
    assert components.label_of("thermal_leaf_ref_dt") == "Leaf − control leaf (°C)"
    assert components.label_of("rgb_ngrdi") == "Green–red index"  # a blank unit is left out, never "(nan)"


def test_links_carry_the_selection_and_plant_ids_link_their_profile():
    assert components.query("W2", "R7", 12) == "?bed=W2&plant=R7&day=12"
    assert components.query(None, "R7", None) == ""  # a plant only with its bed
    assert components.page_href("/threed", "W2", "O8") == "/threed?bed=W2&plant=O8"
    link = components.plant_link("W1", "A2", -3)
    assert link.href == "/plant-profile?bed=W1&plant=A2&day=-3" and link.children == "W1 A2"
    assert link.className == "uw-plant-link" and components.plant_link("W1", "A2", text="A2").children == "A2"
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    rule = re.search(r"\n\.uw-plant-link \{([^}]*)\}", css).group(1)
    assert "text-decoration: none" in rule and ".uw-plant-link:hover" in css  # underlined on hover only
    nav = components.next_steps(("A", "/a"), ("B", "/b"), id="x-next")
    assert nav.id == "x-next" and nav.className == "uw-next"
    assert [a.className for a in components.next_links(("A", "/a"), ("B", "/b"))] == [
        "uw-btn uw-btn-primary",
        "uw-btn uw-btn-outline",
    ]
    assert components.signed(-0.04, "", 1) == "0.0" and components.signed(-1.25, "°C") == "−1.2 °C"


def test_every_page_but_home_ends_with_its_next_steps_and_no_inline_button_styles():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        text = lambda c: json.dumps(c.to_plotly_json(), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        for name in ("bed_map", "plant_profile", "threed", "early_warning", "sensor_comparison", "daily_rhythm",
                     "irrigation", "export", "methods"):
            page = text(sys.modules["pages." + name].layout())
            assert page.count('"className": "uw-next"') == 1, name
            for dead in ("dt-next", "rx-next", "td-more", "dt-btn-outline", "dt-details", "dt-summary", "dt-chip-link",
                         "pl-tile", "td-tile", "rx-tile"):
                assert f'{dead}"' not in page and f"{dead} " not in page, (name, dead)  # a class token
        print("ok")
        """
    )
    pages = ROOT / "app" / "pages"
    for f in [*pages.glob("*.py"), ROOT / "app" / "profile_views.py"]:
        src = f.read_text(encoding="utf-8")
        assert "OUTLINE" not in src, f.name  # outlined buttons are uw-btn uw-btn-outline, never an inline style
        assert "def stat_tile" not in src and "def tile(" not in src, f.name  # one stat tile: components.stat
    detect = (ROOT / "app" / "assets" / "detect.css").read_text(encoding="utf-8")
    assert ".dt-btn-outline" not in detect and ".dt-details" not in detect and "a.dt-chip-link" not in detect
