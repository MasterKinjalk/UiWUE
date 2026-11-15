"""Irrigation advisor and Daily rhythm rendered against the real derived tables, in a fresh process (pages
register their callbacks globally and the shell is built once per interpreter)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (ROOT / "derived" / "diurnal" / "schedules.parquet").exists(), reason="needs derived/diurnal"
)


def run(script):
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0 and "ok" in done.stdout.splitlines()[-1:], done.stderr[-3000:]
    return done.stdout


def test_irrigation_advice_is_latched_and_every_callback_runs():
    run(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        ir = sys.modules["pages.irrigation"]
        from uiwue.diurnal import REWATER_DAY
        ir.layout()
        days = sorted(int(d) for d in ir.load_traits().days_since_cutoff.unique())
        for rule in ir.rule_options():
            schedule = ir.recommended(rule)
            prev = {}
            for day in days:
                out = ir.update(rule, day, None)
                assert len(out) == 10
                m = ir.day_plants(rule, schedule, day)
                now = dict(zip(m.pid, m.status))
                if 0 < day <= REWATER_DAY:  # once irrigate, irrigate until re-watering
                    assert all(now[p] == "irrigate" for p, s in prev.items() if s == "irrigate"), (rule, day)
                prev = now
            ir.update(rule, days[-1], m.pid.iloc[-1])  # a selected plant, the last in the list
        # the advisor uses the rule the library picks; the latest day with a usable scan
        assert ir.latest_day("control_ref", ir.recommended("control_ref")) == max(
            d for d in days if d in set(ir.days_of("control_ref", ir.recommended("control_ref")).query("scanned").day)
        )
        assert ir.best_rule(ir.schedules_table()) == "control_ref"
        w = ir.water_now(ir.day_plants("control_ref", ir.recommended("control_ref"), 7))
        assert len(w) and (w.days_flagged >= 1).all() and w.thermal_leaf_ref_dt.notna().all()
        csv = ir.download(1, "control_ref", 7)
        assert csv["filename"].endswith(".csv") and csv["content"].count("\\n") == len(w) + 1
        print("ok")
        """
    )


def test_irrigation_opens_on_the_decision_and_carries_the_selection():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        ir = sys.modules["pages.irrigation"]
        from uiwue.diurnal import REWATER_DAY
        js = lambda o: json.dumps(o, default=lambda c: c.to_plotly_json(), ensure_ascii=False)
        ids = lambda lay: {c.id: c for c in lay._traverse() if isinstance(getattr(c, "id", None), str)}
        sched = ir.recommended("control_ref")
        # it opens on the last day without water (the re-watering day), the decision a grower faces
        assert ir.default_day("control_ref", sched) == REWATER_DAY
        store = ids(ir.layout())
        assert store["ir-day"].data == REWATER_DAY and store["ir-sel"].data is None
        out = ir.update("control_ref", REWATER_DAY, None)
        water = ir.water_now(ir.day_plants("control_ref", sched, REWATER_DAY))
        tiles = js(out[1])  # the answer first, in the shared stat tiles: water now, then the track record
        assert [t.children[0].children[0].children for t in out[1]] == [
            "Water now", "First alert", "False alarms", "Held-out W3"
        ]
        assert f'"{len(water)}"' in tiles and "uw-sand" in out[1][0].className and "Size only" in tiles
        assert out[8] is False  # the CSV of the water-now list is on
        cards = [c.id["pid"] for c in out[4]]  # the plants to water lead the list of every plant
        assert set(cards[: len(water)]) == set(water.pid)
        legend = js(out[5])
        for status in ir.ORDER:
            assert status in legend
        assert "Threshold" in out[3]  # the rule's definition, on the hover of its picker
        # a link in selects the day and the plant; an unknown day falls back to the default
        store = ids(ir.layout(bed="W1", plant="J5", day="7"))
        assert store["ir-day"].data == 7 and store["ir-sel"].data == "W1_J5"
        assert ids(ir.layout(day="99"))["ir-day"].data == REWATER_DAY
        # the selection goes back into the address bar (ir-url) and on to the next pages and the profile
        out = ir.update("control_ref", 7, "W1_J5")
        assert out[-1] == {"bed": "W1", "plant": "J5", "day": 7}
        assert "/plant-profile?bed=W1&plant=J5&day=7" in js(out[7])  # the selected plant's ID links its profile
        nav = ids(ir.layout())["ir-more"]  # next steps: Export and Methods take no plant, so none rides along
        assert [(link.children, link.href) for link in nav.children] == [(t, p) for t, p, _ in ir.NEXT]
        assert "uw-btn-primary" in nav.children[0].className and "uw-btn-outline" in nav.children[1].className
        # a day with nothing to water says so: 0 on a mint tile, the CSV off
        quiet = ir.update("control_ref", -3, None)
        assert quiet[1][0].children[1].children[0].children == "0" and "uw-mint" in quiet[1][0].className
        assert quiet[8] is True
        print("ok")
        """
    )


def test_daily_rhythm_states_its_finding_and_marks_one_schedule_per_rule():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        dr = sys.modules["pages.daily_rhythm"]
        from components import RULE_SHORT
        from uiwue.core import read_table
        dr.layout()
        sch = read_table("diurnal", "schedules")
        for rule in RULE_SHORT:
            table = str(dr.update_schedules(rule))
            assert table.count("'best'") == 1, rule
        for score in dr.SCORES:
            dr.update_separability(score)
        fig, _ = dr.update_rhythm("thermal_leaf_ref_dt", "treatment")
        assert len(fig.layout.annotations) == 5  # the five phases of the day profile
        _, chips = dr.update_rhythm("depth_height_p75", "treatment")
        assert "W1 + W2 only" in str(chips)  # a 3D trait by treatment never pools W3
        rec = sch[sch.recommended & (sch.set == "W2 tuning")]
        assert rec.groupby("rule").size().eq(1).all()
        # the finding first: midday heat, the 3D midday dip, the best flyover, computed from the tables
        tiles = json.dumps(dr.tiles(sch), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        deficit, control = dr.midday_heat()
        assert deficit[12] > 5 > abs(deficit[6]) and abs(control[12]) < 1  # hot at noon, not at dawn
        assert dr.signed(deficit[12], "") in tiles and dr.signed(deficit[6], "°C") in tiles
        dip = dr.tc.midday_dip()["height_p75"]
        assert "Midday dip · 3D" in tiles and ("none" in tiles) == (dip["ci_low"] <= 0 <= dip["ci_high"])
        best = sch[sch.recommended & (sch.rule == dr.best_rule(sch)) & (sch.set == "W3 check")].iloc[0]
        assert "Best flyover" in tiles and best.schedule.replace("all five", "all 5") in tiles
        # a link's selection is carried on to the next pages
        nav = dr.layout(bed="W2", plant="A8", day="12").children[-1]
        assert [link.children for link in nav.children] == [t for t, _, _ in dr.NEXT] and nav.className == "uw-next"
        assert all(link.href.endswith("?bed=W2&plant=A8&day=12") for link in nav.children)
        assert "Size only · day" in tiles  # beside the best flyover, the one detection result here
        print("ok")
        """
    )
