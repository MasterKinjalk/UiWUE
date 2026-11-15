"""New run: reached from Home only, cannot start anything unless the app was started with --runs, models off."""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("dash")

ROOT = Path(__file__).resolve().parents[2]
CSS = (ROOT / "app" / "assets" / "newrun.css").read_text(encoding="utf-8")


def run(script, **env):
    import os

    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={k: v for k, v in os.environ.items() if not k.startswith("UIWUE_")} | env,
    )
    assert done.returncode == 0 and "ok" in done.stdout.split(), done.stderr[-3000:]


def test_the_page_is_on_home_and_not_in_the_nav():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app
        text = lambda c: json.dumps(c.to_plotly_json(), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        assert "/new-run" in [p["path"] for p in __import__("dash").page_registry.values()]
        assert "/new-run" not in [p["path"] for p in app.pages] and "/new-run" not in text(app.navbar)
        home = sys.modules["pages.home"]
        assert text(home.layout()).count('"href": "/new-run"') == 1
        assert "/new-run" not in [t.href for t in home.page_links()]  # the path's four groups stay as they are
        assert "uw-runchip" not in text(app.navbar)  # no run named: these are the committed tables
        print("ok")
        """
    )


def test_start_is_off_by_default_and_the_server_refuses_it_too():
    run(
        """
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        nr = sys.modules["pages.new_run"]
        text = lambda c: json.dumps(c.to_plotly_json(), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        page = text(nr.layout())
        assert not nr.runs_enabled() and "start off" in page
        assert '"id": "nr-start"' in page and '"disabled": true' in page
        assert nr.command("data", "t", ["thermal"], [], "", 4)[1] is True  # Start stays disabled
        assert page.count('"className": "uw-next"') == 1 and "no table yet" not in page
        print("ok")
        """
    )
    run(
        """
        import sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        nr = sys.modules["pages.new_run"]
        import json
        text = lambda c: json.dumps(c.to_plotly_json(), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        assert nr.runs_enabled() and "start off" not in text(nr.layout())
        line, disabled = nr.command("data", "t", ["thermal"], [], "W1_A2, nonsense", 4)
        assert disabled is False and line.endswith("--steps thermal --plants W1_A2 --jobs 4")
        assert nr.command("no/such/folder", "t", ["thermal"], [], "", 4)[1] is True
        print("ok")
        """,
        UIWUE_RUNS="1",
    )


def test_a_checked_folder_shows_what_it_holds_and_the_models_start_off(tmp_path):
    for pid in ("W1_A2", "W2_A2"):
        (tmp_path / "thermal" / pid / "2025_09_04" / "15").mkdir(parents=True)
    run(
        f"""
        import json, sys
        sys.path[:0] = ["app", "src"]
        import app  # noqa: F401
        nr = sys.modules["pages.new_run"]
        text = lambda c: json.dumps(c.to_plotly_json(), default=lambda o: o.to_plotly_json(), ensure_ascii=False)
        tiles, notes, steps, on, ml, ml_on, name = nr.check(1, r"{tmp_path}", "")
        assert "2" in text(tiles[0]) and "plants" in text(tiles[0]) and "uw-stat-missing" in text(tiles[1])
        off = {{o["value"] for o in steps if o["disabled"]}}
        assert "thermal" in off and "threed" in off and "detect" not in off  # thermal needs the air file too
        assert "detect" in on and "thermal" not in on and ml_on == [] and name == r"{tmp_path.name}"
        assert all(o["disabled"] for o in ml if o["value"] in ("mask", "depth"))  # no depth folder
        assert nr.check(1, "no/such/folder", "")[1][0].children == "no folder"
        print("ok")
        """
    )


def test_styles_use_tokens_only_and_controls_are_44_px():
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|box-shadow|gradient", css)
    assert re.search(r"\.nr-input \{[^}]*min-height: 44px", css)
    for size in re.findall(r"font-size: (\d+)px", css):
        assert int(size) >= 12, size
    assert "@media (max-width: 700px)" in css and "minmax(0, 1fr)" in css
