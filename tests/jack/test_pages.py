import importlib.util
import json
from pathlib import Path

import plotly.graph_objects as go
import pytest

PAGES = Path(__file__).resolve().parents[2] / "app" / "pages"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"jack_page_{name}", PAGES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Synthetic derived/ with detect results, built once (fast settings)."""
    import synthetic
    from conftest import point_core_at

    from uiwue.detect.run import main

    root = tmp_path_factory.mktemp("derived")
    synthetic.write_derived(root)
    with pytest.MonkeyPatch.context() as mp:
        point_core_at(root, mp)
        main(["--n-boot", "100"])
        yield root


def _text(component) -> str:
    return json.dumps(component.to_plotly_json(), default=str)


def test_pages_render(results):
    ew = _load("early_warning")
    assert "ew-curves" in _text(ew.layout())
    curves, _ranking, _tbl, head = ew._update("all", ["thermal_leaf_air_dt", "depth_projected_area"])
    assert isinstance(curves, go.Figure) and len(curves.data) >= 4
    assert "Leaf - air temperature" in head

    sc = _load("sensor_comparison")
    assert "sc-figure" in _text(sc.layout())
    fig, _tbl, head = sc._update("all", "early_auc")
    assert isinstance(fig, go.Figure) and "Thermal" in head

    ml = _load("methods")
    assert "Placebo first" in _text(ml.layout())
    assert isinstance(ml._placebo("all"), go.Figure)


def test_pages_without_results(tmp_path, monkeypatch):
    from conftest import point_core_at

    point_core_at(tmp_path, monkeypatch)
    for name in ("early_warning", "sensor_comparison", "methods"):
        text = _text(_load(name).layout())
        assert "Not available" in text or "not available" in text
