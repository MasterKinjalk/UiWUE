"""Helpers for Jack's pages. The leading underscore keeps Dash from loading it as a page."""

from __future__ import annotations

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:  # the shell has app/ on the path; a page run alone does not
    sys.path.insert(0, str(APP))

import components

from uiwue.detect.data import load_result

header = components.header


def register_page(module: str, **kw) -> None:
    """Register with the shell's page registry; harmless when the page runs alone."""
    try:
        dash.register_page(module, **kw)
    except dash.exceptions.PageError:
        pass


def not_available(what: str) -> html.Div:
    return html.Div(
        [
            components.not_available(what),
            html.P(
                [
                    "Detection results are made by ",
                    html.Code("python -m uiwue.detect.run"),
                    " from the trait tables in derived/.",
                ],
                className="text-secondary small",
            ),
        ]
    )


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
        page_size=page_size,
        sort_action="native",
        style_cell={
            "fontFamily": "inherit",
            "fontSize": 13,
            "padding": "4px 8px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={"fontWeight": 600, "background": components.INK["surface"]},
        **kw,
    )


def run_alone(layout) -> None:
    """`python app/pages/<page>.py` serves just this page on :8050."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
    app.layout = dbc.Container(layout() if callable(layout) else layout, fluid=True, className="py-3")
    app.run(debug=True)


__all__ = ["header", "load_result", "not_available", "register_page", "run_alone", "table"]
