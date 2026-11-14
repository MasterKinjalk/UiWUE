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
    import os

    import synthetic

    from uiwue.stats._design import _catalogue_cached

    root = tmp_path_factory.mktemp("derived")
    synthetic.write_derived(root)
    old = os.environ.get("UIWUE_DERIVED")
    os.environ["UIWUE_DERIVED"] = str(root)
    _catalogue_cached.cache_clear()
    from uiwue.detect.run import main

    main(["--n-boot", "100"])
    yield root
    _catalogue_cached.cache_clear()
    if old is None:
        os.environ.pop("UIWUE_DERIVED")
    else:
        os.environ["UIWUE_DERIVED"] = old


def _text(component) -> str:
    return json.dumps(component.to_plotly_json(), default=str)


def test_pages_render(results):
    ew = _load("early_warning")
    assert "ew-curves" in _text(ew.layout())
    curves, ranking, tbl, head = ew._update("all", ["thermal_leaf_air_dt", "depth_projected_area"])
    assert isinstance(curves, go.Figure) and len(curves.data) >= 4
    assert "Leaf - air temperature" in head

    sc = _load("sensor_comparison")
    assert "sc-figure" in _text(sc.layout())
    fig, tbl, head = sc._update("all", "early_auc")
    assert isinstance(fig, go.Figure) and "thermal" in head

    ml = _load("methods")
    assert "Placebo first" in _text(ml.layout())
    assert isinstance(ml._placebo("all"), go.Figure)


def test_pages_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("UIWUE_DERIVED", str(tmp_path))
    for name in ("early_warning", "sensor_comparison", "methods"):
        text = _text(_load(name).layout())
        assert "Not available" in text or "not available" in text
