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
                assert len(out) == 7
                m = ir.day_plants(rule, schedule, day)
                now = dict(zip(m.pid, m.status))
                if 0 < day <= REWATER_DAY:  # once irrigate, irrigate until re-watering
                    assert all(now[p] == "irrigate" for p, s in prev.items() if s == "irrigate"), (rule, day)
                prev = now
            ir.update(rule, days[-1], m.pid.iloc[-1])  # a selected plant, the last in the list
        # the advisor opens on the rule the library picks, on the latest day with a usable scan
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


def test_daily_rhythm_when_to_image_marks_one_schedule_per_rule():
    run(
        """
        import sys
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
        dr.update_rhythm("thermal_leaf_ref_dt", "treatment", "afternoon_minus_morning")
        rec = sch[sch.recommended & (sch.set == "W2 tuning")]
        assert rec.groupby("rule").size().eq(1).all()
        print("ok")
        """
    )
