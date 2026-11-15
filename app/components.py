"""Shared look and building blocks for every page: colours, figure theme, headers, pickers.

Colours follow one rule each (validated for colour-blind readers):
beds and treatment are identity (categorical); alert status is state (status
palette, always paired with a marker shape and a label).
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from dash import html

from uiwue.core import CUTOFF, REWATER, catalogue

INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BED = {"W1": SERIES[0], "W2": SERIES[1], "W3": SERIES[2]}
TREATMENT = {"control": SERIES[0], "deficit": SERIES[1]}
GROUP = BED | TREATMENT
STATUS = {  # state -> (colour, marker symbol)
    "fine": ("#0ca30c", "circle"),
    "watch": ("#fab219", "triangle-up"),
    "irrigate": ("#d03b3b", "diamond"),
    "no reading": ("#898781", "circle-open"),
}
# Separability (AUC): 0.5 = no difference, neutral grey; 1 = deficit plants clearly warmer
DIVERGING = [[0, "#184f95"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f0a3a2"], [1, "#b42f2f"]]

pio.templates["uiwue"] = go.layout.Template(
    layout=dict(
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=INK["primary"]),
        paper_bgcolor=INK["surface"],
        plot_bgcolor=INK["surface"],
        colorway=SERIES,
        xaxis=dict(
            gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False, ticks="", title_font_color=INK["secondary"]
        ),
        yaxis=dict(
            gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False, ticks="", title_font_color=INK["secondary"]
        ),
        legend=dict(title_text=""),
        title=dict(x=0, xanchor="left", font=dict(size=15)),
        hoverlabel=dict(bgcolor="white", font_color=INK["primary"]),
        margin=dict(l=60, r=20, t=50, b=50),
    )
)
pio.templates.default = "uiwue"

DAY0_LABEL = f"irrigation off ({CUTOFF:%d %b})"
REWATER_DAY = (REWATER - CUTOFF).days + 0.5  # re-watered in the evening of day 12


def header(title, question):
    return html.Div([html.H3(title, className="mb-1"), html.P(question, className="text-secondary mb-3")])


def not_available(what):
    return dbc.Alert(f"{what} is not available yet: its table has not been pushed to derived/.", color="light")


def trait_options(columns, stress_only=False):
    """Dropdown options for catalogue traits present in `columns`."""
    cat = catalogue()
    cat = cat[cat.column.isin(columns) & ((cat.direction != 0) if stress_only else True)]
    return [{"label": f"{r.label} ({r.unit})", "value": r.column} for r in cat.itertuples()]


def label_of(column):
    row = catalogue().set_index("column").reindex([column]).iloc[0]
    return f"{row.label} ({row.unit})" if isinstance(row.label, str) else column


def mark_stress_period(fig, **where):
    """Hairlines at the irrigation cutoff and the re-watering, on an x axis of days since cutoff."""
    for x, text in ((0, DAY0_LABEL), (REWATER_DAY, "re-watered")):
        fig.add_vline(
            x=x,
            line_width=1,
            line_color=INK["muted"],
            annotation_text=text,
            annotation_position="top right",
            annotation_font=dict(size=11, color=INK["muted"]),
            **where,
        )
    return fig
