"""Helpers for Jack's pages. The leading underscore keeps Dash from loading it as a page."""

from __future__ import annotations

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dash_table, html

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:  # the shell has app/ on the path; a page run alone does not
    sys.path.insert(0, str(APP))

import components
from components import C, chip, graph, header, panel, readout, segmented, select

from uiwue.detect.data import load_result

GROUPS = {"all": "W2 + W3", "W2": "W2", "W3": "W3"}
ONE_CONTROL_BED = (
    "One control bed (W1). Intervals resample plants, but plants in a bed share the day's light and air, "
    "so bed and treatment cannot be separated and the intervals are too narrow: read 1.00 [1.00, 1.00] as "
    "'the beds differ that day', not as certainty. The placebo and the pre-cutoff check test the beds when "
    "all plants were watered."
)


def register_page(module: str, **kw) -> None:
    """Register with the shell's page registry; harmless when the page runs alone."""
    try:
        dash.register_page(module, **kw)
    except dash.exceptions.PageError:
        pass


def not_available(what: str) -> html.Div:
    """The shared empty state; how to make the table goes on hover."""
    return html.Div(
        components.not_available(what), title=f"{what}: run python -m uiwue.detect.run to build derived/detect/"
    )


def fmt(x, digits=2, unit="") -> str:
    """Number with unit, or an en dash when missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "–"
    return f"{x:.{digits}f}{(' ' + unit) if unit else ''}"


def day_text(x) -> str:
    return "–" if x is None or pd.isna(x) else f"day {int(x)}".replace("-", "−")


def signed_days(x) -> str:
    return components.signed(x, "d", 0)


def tile(label, value, unit="", tint="gray", hover=None, extra=()):
    """Stat tile (DESIGN 5): label 12 / value display 32 / unit 12 on a role tint."""
    return html.Div(
        [
            html.Div([label, *extra], className="uw-readout-label", style={"display": "flex", "gap": "8px"}),
            html.Div([value, html.Small(unit)] if unit else value, className="uw-stat"),
        ],
        className=f"uw-tile uw-{tint}",
        title=hover,
    )


def missing_tile(label, why):
    return html.Div(
        [html.Div(label, className="uw-readout-label"), html.Div("–", className="uw-stat")],
        className="uw-tile uw-empty",
        title=why,
        style={"display": "block"},
    )


def bed_effect_chip(row) -> list:
    """'bed effect' when the placebo flags the signal, 'day effect' when the onset rule
    already fires on watered days; the numbers go on hover."""
    out = []
    if bool(row.get("bed_effect", False)):
        out.append(
            chip(
                "bed effect",
                f"Placebo (24–29 Aug, all watered): AUC {fmt(row.get('placebo_auc'))}, "
                f"95 % CI {fmt(row.get('placebo_ci_low'))}–{fmt(row.get('placebo_ci_high'))}. "
                "The beds already differed; not ranked.",
            )
        )
    if bool(row.get("fires_before_cutoff", False)):
        fire = row.get("precutoff_fire_day", np.nan)
        out.append(
            chip(
                "day effect",
                "The onset rule already fires on watered days"
                + (f" ({day_text(fire)})" if pd.notna(fire) else "")
                + f". Largest watered-day AUC {fmt(row.get('null_auc_max'))}; the robust onset must clear it.",
            )
        )
    return out


def source_tag(sensor: str):
    from uiwue.detect.figures import SENSOR_TAG

    return html.Span(
        SENSOR_TAG.get(sensor, sensor), className="uw-kicker", style={"color": C["slate-600"], "minWidth": "32px"}
    )


def table(df, columns: dict, page_size: int = 15, id=None, export: str | None = "csv"):
    """DataTable from a frame; `columns` maps column -> header. Floats to 2 dp, missing as '–',
    numbers right-aligned in tabular figures, CSV export button."""
    d = df[list(columns)].copy()
    numeric = []
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(2)
            numeric.append(c)
        elif d[c].dtype.kind in "iu":
            numeric.append(c)
        elif d[c].dtype.kind == "b":
            d[c] = d[c].map({True: "yes", False: "no"})
    d = d.astype(object).where(d.notna(), "–")
    kw = {"id": id} if id else {}
    if export:
        kw.update(export_format=export, export_headers="display")
    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[{"name": h, "id": c} for c, h in columns.items()],
        page_size=page_size,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "inherit",
            "fontSize": 13,
            "padding": "6px 8px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "color": C["ink"],
            "border": f"1px solid {C['rule']}",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "right", "fontVariantNumeric": "tabular-nums"} for c in numeric
        ],
        style_header={"fontWeight": 700, "background": C["navy"], "color": C["surface"]},
        css=[{"selector": ".export", "rule": "min-height: 44px; margin-bottom: 8px;"}],
        **kw,
    )


def run_alone(layout) -> None:
    """`python app/pages/<page>.py` serves just this page on :8050."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
    app.layout = dbc.Container(layout() if callable(layout) else layout, fluid=True, className="py-3")
    app.run(debug=True)


__all__ = [
    "GROUPS",
    "ONE_CONTROL_BED",
    "C",
    "bed_effect_chip",
    "chip",
    "day_text",
    "fmt",
    "graph",
    "header",
    "load_result",
    "missing_tile",
    "not_available",
    "panel",
    "readout",
    "register_page",
    "run_alone",
    "segmented",
    "select",
    "signed_days",
    "source_tag",
    "table",
    "tile",
]
