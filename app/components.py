"""Shared look and building blocks for every page: design tokens, figure theme, page header, pickers.

Colours, type and marks follow design/DESIGN.md; the colour values are read from
design/tokens.json, so the tokens live in one place. Page styles are in assets/uiwue.css.
"""

import json
from datetime import timedelta
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html

from uiwue.core import BASELINE, CUTOFF, REWATER, catalogue, read_table

_TOKENS = json.loads((Path(__file__).resolve().parents[1] / "design" / "tokens.json").read_text(encoding="utf-8"))
_raw = {t["name"]: t["value"] for t in _TOKENS["color"]["tokens"]}
C = {k: _raw[v[1:-1]] if v.startswith("{") else v for k, v in _raw.items()}  # resolves "{forest-700}"
FONT = _TOKENS["type"]["families"]  # display, sans

# Names other pages import; the values are now tokens
INK = dict(
    primary=C["ink"],
    secondary=C["ink-muted"],
    muted=C["ink-muted"],
    grid=C["rule"],
    axis=C["gray-500"],
    surface=C["surface"],
)
SERIES = [C["teal-700"], C["amber-700"], C["forest-700"], C["plum-700"], C["gray-500"]]
TREATMENT = {"control": C["teal-700"], "deficit": C["amber-700"]}
BED = {
    "W1": C["teal-700"],
    "W2": C["amber-700"],
    "W3": C["amber-900"],
}  # both deficit beds amber, told apart by lightness
GROUP = BED | TREATMENT
VALUE_RAMP = [C[k] for k in ("teal-600", "teal-300", "gray", "amber-200", "amber-700", "amber-900")]  # amber = stressed


def rgba(hex_, alpha):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


BAND = {"control": rgba(C["teal-700"], 0.14), "deficit": rgba(C["amber-700"], 0.18)}  # ±1 SD bands, DESIGN 6

pio.templates["uiwue"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family=FONT["sans"], size=13, color=C["ink"]),
        colorway=SERIES,
        xaxis=dict(gridcolor=C["rule"], linecolor=C["gray-500"], zeroline=False, ticks=""),
        yaxis=dict(gridcolor=C["rule"], linecolor=C["gray-500"], zeroline=False, ticks=""),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", title_text=""),
        hoverlabel=dict(bgcolor=C["surface"], font_color=C["ink"]),
        margin=dict(l=48, r=16, t=24, b=40),
    )
)
pio.templates.default = "uiwue"

BASELINE_DAYS = ((BASELINE[0] - CUTOFF).days - 0.5, (BASELINE[1] - CUTOFF).days + 0.5)
REWATER_DAY = (REWATER - CUTOFF).days + 0.5  # re-watered in the evening of day 12
DRY_DAYS = (0, REWATER_DAY)  # cutoff (day 0) to re-watering: the band edges are the events
GRAPH_CONFIG = {"displayModeBar": False}


def header(title, question, *right):
    """Page header: the title (the page's question on hover), then chips and controls to its right."""
    return html.Div(
        [html.H1(title, title=question, className="uw-title"), html.Div(className="uw-spacer"), *right],
        className="uw-header",
    )


def not_available(what):
    """Empty state: a dashed box naming the module whose table is missing."""
    return html.Div(
        [html.Strong(what), html.Span("no table yet")],
        className="uw-empty",
        title=f"{what}: its table has not been pushed to derived/ yet",
    )


def chip(text, hover, kind="caveat"):
    """caveat (sand), rule (outlined) or context (caps) chip; the detail goes on hover."""
    return html.Span(text, title=hover, className=f"uw-chip uw-chip-{kind}")


def placebo_chip(signal):
    """'bed effect' chip when Jack's placebo test (detect/placebo) flags the trait: the beds differed before irrigation off."""
    p = read_table("detect", "placebo")
    if p is None or "bed_effect" not in p:
        return []
    r = p[(p.signal == signal) & (p.group == "all") & p.bed_effect.astype(bool)]
    if r.empty:
        return []
    r = r.iloc[0]
    return [
        chip(
            "bed effect",
            f"Before irrigation off the deficit and control beds already differ in {r.label} "
            f"(placebo AUC {r.placebo_auc:.2f}, 95 % CI {r.placebo_ci_low:.2f}–{r.placebo_ci_high:.2f}; detect/placebo)",
        )
    ]


def size_chip():
    """The size-only baseline next to a detection result (DESIGN 7.6): when projected leaf area alone separates."""
    o = read_table("detect", "onsets")
    r = o[o.is_size_baseline.astype(bool) & (o.group == "all")] if o is not None and "is_size_baseline" in o else None
    if r is None or r.empty or pd.isna(r.onset_day.iloc[0]):
        return chip("size · –", "Size-only baseline (depth projected area): no onset in detect/onsets yet", "rule")
    r = r.iloc[0]
    return chip(
        f"size · day {r.onset_day:g}",
        f"Size-only baseline: {r.label} alone separates deficit from control plants from day {r.onset_day:g} "
        f"({r.n_deficit} deficit, {r.n_control} control plants with depth data; onset rule of uiwue.stats, detect/onsets)",
        "rule",
    )


def mark(status, hover=None):
    """Status mark by shape: fine circle, watch triangle, irrigate diamond, no scan dashed circle."""
    return html.Span(className=f"uw-mark uw-mark-{status.replace(' ', '')}", title=hover, **{"aria-hidden": "true"})


def readout(label, value, unit="", size="readout"):
    return html.Div(
        [html.Div(label, className="uw-readout-label"), html.Div([value, html.Small(unit)], className=f"uw-{size}")]
    )


def signed(x, unit="", digits=1):
    """'+4.7 °C' / '−2.0 °C' with a real minus sign; '–' when missing."""
    if x is None or pd.isna(x):
        return "–"
    s = f"{x:+.{digits}f}".replace("-", "−")
    return f"{s} {unit}".strip()


def day_label(day):
    """('Day 4', '7 Sep'): days since irrigation off first, the calendar date second."""
    d = CUTOFF + timedelta(days=int(day))
    return f"Day {int(day)}".replace("-", "−"), f"{d.day} {d:%b}"


RULE_SHORT = {  # alert rules (uiwue.diurnal.RULES) in 1-3 words, for buttons and cards
    "control_ref": "vs control bed",
    "local_line": "no control bed",
    "idso": "published line",
    "leaf_air": "leaf − air",
    "microclimate": "demand",
}


def segmented(id, options, value, label):
    """Segmented group (DESIGN 5): a native radio group drawn as buttons. options: {value: text}."""
    return html.Div(
        dbc.RadioItems(
            id=id,
            options=[{"label": t, "value": v} for v, t in options.items()],
            value=value,
            inline=True,
            className="uw-seg",
            inputClassName="btn-check",
            labelClassName="uw-seg-btn",
            labelCheckedClassName="uw-seg-on",
        ),
        role="group",
        **{"aria-label": label},
    )


def select(id, options, value, label):
    """Labelled native select, 44 px. options: {value: text}."""
    return html.Label(
        [
            label,
            dbc.Select(
                id=id,
                options=[{"label": t, "value": v} for v, t in options.items()],
                value=value,
                className="uw-select",
            ),
        ],
        className="uw-label",
    )


def graph(id, label, height=None):
    """A chart with role img and an aria-label naming what it shows."""
    style = {"height": f"{height}px"} if height else None
    return html.Div(dcc.Graph(id=id, config=GRAPH_CONFIG, style=style), role="img", **{"aria-label": label})


def panel(title, *children, hover=None, extra=()):
    """Bordered panel with a section head; `extra` sits next to the head (chips, controls)."""
    return html.Section(
        [html.Div([html.H2(title, title=hover), *extra], className="uw-panel-head"), *children], className="uw-panel"
    )


def unit_of(column):
    row = catalogue().set_index("column").reindex([column]).iloc[0]
    return row.unit if isinstance(row.unit, str) else ""


def trait_options(columns, stress_only=False):
    """Dropdown options for catalogue traits present in `columns`."""
    cat = catalogue()
    cat = cat[cat.column.isin(columns) & ((cat.direction != 0) if stress_only else True)]
    return [{"label": f"{r.label} ({r.unit})", "value": r.column} for r in cat.itertuples()]


def label_of(column):
    row = catalogue().set_index("column").reindex([column]).iloc[0]
    return f"{row.label} ({row.unit})" if isinstance(row.label, str) else column


def mark_stress_period(fig, **where):
    """Baseline band (mint), no-water band (sand) and dotted hairlines at cutoff and re-watering,
    on an x axis of days since irrigation off."""
    fig.add_vrect(*BASELINE_DAYS, fillcolor=C["mint"], line_width=0, layer="below", **where)
    fig.add_vrect(*DRY_DAYS, fillcolor=C["sand"], line_width=0, layer="below", **where)
    for x in DRY_DAYS:
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=C["ink-muted"], **where)
    return fig
