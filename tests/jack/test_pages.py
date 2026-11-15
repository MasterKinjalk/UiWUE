import importlib.util
import json
from pathlib import Path

import plotly.graph_objects as go
import pytest
from dash import html

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
    text = _text(ew.layout())
    assert "ew-curves" in text and "ew-ranking" in text and "AUC rule" in text
    for group in ("all", "W2", "W3"):
        tiles, ranking, _flags, _rec, tbl = ew._update(group)
        assert isinstance(ranking, go.Figure) and len(ranking.data) == 2
        assert tiles and "day" in _text(html.Div(tiles))
        if group == "all":
            assert getattr(tbl, "export_format", None) == "csv"
        for sig in ("thermal_leaf_air_dt", "depth_projected_area", "hsi_ndvi"):
            curves, _chips, slots = ew._signal(group, sig)
            assert isinstance(curves, go.Figure)
    curves, chips, slots = ew._signal("all", "thermal_leaf_air_dt")
    assert len(curves.data) >= 3 and len(slots) == 5
    _, chips, _ = ew._signal("all", "rgb_bedfx")
    assert "bed effect" in _text(html.Div(chips))

    sc = _load("sensor_comparison")
    assert "sc-figure" in _text(sc.layout())
    for ps in ("common", "own"):
        for group in ("all", "W2", "W3"):
            for metric in ("early_auc", "full_auc"):
                tiles, fig, _flags, _tbl, _chip = sc._update(ps, group, metric)
                assert isinstance(fig, go.Figure)
    tiles, *_ = sc._update("common", "all", "early_auc")
    assert "Thermal only" in _text(html.Div(tiles))

    ml = _load("methods")
    text = _text(ml.layout())
    assert "placebo first" in text and "One control bed" in text
    for group in ("all", "W2", "W3"):
        assert isinstance(ml._placebo(group), go.Figure)


def test_pages_register_a_question():
    import dash

    for name in ("early_warning", "sensor_comparison", "methods"):
        mod = _load(name)
        assert mod.QUESTION.endswith("?")
    assert dash  # registration itself is exercised by the app smoke test


def test_pages_without_results(tmp_path, monkeypatch):
    from conftest import point_core_at

    point_core_at(tmp_path, monkeypatch)
    for name in ("early_warning", "sensor_comparison", "methods"):
        text = _text(_load(name).layout())
        assert "uw-empty" in text and "no table yet" in text
