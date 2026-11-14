"""Helpers for Jack's pages. The leading underscore keeps Dash from loading it as a page."""

from __future__ import annotations

import sys
from pathlib import Path

import dash
from dash import dash_table, html

try:
    import uiwue  # noqa: F401
except ImportError:  # package not installed yet: run from a clone
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uiwue.detect.data import load_result


def register(module: str, **kw) -> None:
    """Register with the shell's page registry; harmless when the page runs alone."""
    try:
        dash.register_page(module, **kw)
    except dash.exceptions.PageError:
        pass


def not_available(what: str) -> html.Div:
    return html.Div(
        [html.Strong("Not available yet. "),
         html.Span(f"{what} Run "),
         html.Code("python -m uiwue.detect.run"),
         html.Span(" once the trait tables are in derived/.")],
        style={"padding": "1rem", "background": "#f6f6f6", "borderRadius": "6px", "margin": "1rem 0"},
    )


def answer(text: str) -> html.P:
    return html.P(text, style={"fontSize": "1.15rem", "fontWeight": 500, "margin": "0.5rem 0 1rem"})


def table(df, columns: dict, page_size: int = 15, id=None):
    """DataTable from a frame; `columns` maps column -> header. Floats to 2 dp."""
    d = df[list(columns)].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(2)
        elif d[c].dtype.kind == "b":
            d[c] = d[c].map({True: "yes", False: ""})
    kw = {"id": id} if id else {}
    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[{"name": h, "id": c} for c, h in columns.items()],
        page_size=page_size, sort_action="native",
        style_cell={"fontFamily": "inherit", "fontSize": 13, "padding": "4px 8px",
                    "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
        style_header={"fontWeight": 600, "background": "#fafafa"},
        **kw,
    )


def run_alone(layout) -> None:
    """`python app/pages/<page>.py` serves just this page on :8050."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = layout() if callable(layout) else layout
    app.run(debug=True)


__all__ = ["answer", "load_result", "not_available", "register", "run_alone", "table"]
