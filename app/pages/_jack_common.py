"""Helpers for Jack's pages. The leading underscore keeps Dash from loading it as a page."""

from __future__ import annotations

import json
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
from components import C, chip, details, graph, header, next_steps, panel, readout, segmented, select, stat
from components import stat_missing as missing_tile

from uiwue.detect.data import DEPTH_EXCLUDED, DEPTH_EXCLUDED_PREFIX
from uiwue.detect.data import load_result as _load_result
from uiwue.stats._design import derived_dir

GROUPS = {"all": "W2 + W3", "W2": "W2", "W3": "W3"}

# Depth signals left out of the analysis, app-wide (docs/threed/APP.md, "What was removed"): noisy,
# saturated or redundant 3D traits. One list, at the source (uiwue.detect.data.DEPTH_EXCLUDED), so the
# detect run no longer scores them; detect tables written before that still hold them, so the pages also
# drop them on read (components.kept_trait, which the placebo chip uses too).
DEPTH_KEPT = ("depth_leaf_area_est", "depth_height_p75", "depth_hull_volume")
DEPTH_SIZE = "depth_projected_area"  # kept only as the size-only baseline beside every result (DESIGN 7.6)
DEPTH_DROPPED = DEPTH_EXCLUDED
DEPTH_DROPPED_PREFIX = DEPTH_EXCLUDED_PREFIX  # every cylinder trait


def kept_signal(signal) -> bool:
    """False for a 3D trait the app leaves out (DEPTH_DROPPED and every depth_cyl_* column)."""
    return components.kept_trait(signal)


def keep_signals(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rows of a detect table whose `signal` the app keeps; a table without that column is returned as is."""
    if df is None or "signal" not in df.columns:
        return df
    return df[df["signal"].map(kept_signal)].reset_index(drop=True)


def short(signal, label="") -> str:
    """The signal's plain name, the one every page uses (``components.plain``)."""
    return components.plain(signal, label)


def load_result(name: str) -> pd.DataFrame | None:
    """A derived/detect table (None if missing) without the dropped 3D traits. The filter runs on read,
    so every ranking, count and 'best signal' pick on a page is computed after it. Jargon labels read plain."""
    df = keep_signals(_load_result(name))
    if df is not None and {"signal", "label"} <= set(df.columns):
        df = df.assign(label=[components.plain(s, lbl) for s, lbl in zip(df["signal"], df["label"], strict=True)])
    return df


# Plain words for the headline number of the detect pages; the technical name and its meaning go on hover
SCORE_HOVER = (
    "Detection score (AUC): how well the signal separates stressed from watered plants; 0.5 = chance, 1 = perfect."
)
ONSET_HOVER = "Onset: the first day the score beats the largest score seen on any watered day, that day and the next."


# Anchors on Methods & limits: one per limit or term, so a chip or hover on any page can point to it
METHODS_ANCHORS = (
    "one-control-bed",
    "bed-effect",
    "w3-sampled",
    "small-rgb",
    "symptom-notes",
    "hyperspectral",
    "missing-runs",
    "microclimate",
    "fusion-lockbox",
    "detection-score",
    "onset",
    "size-baseline",
    "recovery",
    "lead-time",
)


methods_href = components.methods_href


def chip_link(text, hover, anchor, kind="caveat"):
    """A chip that is also a link to its entry on Methods & limits (``components.chip`` with an anchor)."""
    return chip(text, hover, kind, anchor)


def run_info() -> dict:
    """derived/detect/run_info.json: when and on what the detect tables were computed; {} if missing."""
    try:
        return json.loads((derived_dir() / "detect" / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run_time(info: dict) -> str:
    """'2026-11-15 04:10 UTC' from run_info()['finished'], '' when unknown."""
    t = pd.to_datetime(info.get("finished"), utc=True, errors="coerce")
    return "" if pd.isna(t) else f"{t:%Y-%m-%d %H:%M} UTC"


ONE_CONTROL_BED = (
    "One control bed (W1): bed and treatment cannot be separated, so read a score of 1.00 as 'the beds differ "
    "that day', not as certainty. The watered-day checks test the beds before the cut."
)


def register_page(module: str, **kw) -> None:
    """Register with the shell's page registry; harmless when the page runs alone."""
    try:
        dash.register_page(module, **kw)
    except dash.exceptions.PageError:
        pass


def min_width(g, px: int):
    """components.graph with a minimum width, for a chart that scrolls in its own box (``uw-scroll``) at phone
    width instead of squeezing its labels into each other (DESIGN 8)."""
    g.children.style = {**(g.children.style or {}), "minWidth": f"{px}px"}
    return g


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


def bed_effect_chip(row) -> list:
    """'bed effect' when the placebo flags the signal, 'day effect' when the onset rule
    already fires on watered days; the numbers go on hover."""
    out = []

    def _on(key):  # a missing flag (a table without that check, e.g. the fusion rows) is not a flag
        v = row.get(key, False)
        return bool(v) if pd.notna(v) else False

    if _on("bed_effect"):
        out.append(
            components.bed_effect_chip(
                f"Score {fmt(row.get('placebo_auc'))} on watered days 24–29 Aug (95 % CI "
                f"{fmt(row.get('placebo_ci_low'))}–{fmt(row.get('placebo_ci_high'))}; 0.5 = no difference): not ranked."
            )
        )
    if _on("fires_before_cutoff"):
        fire = row.get("precutoff_fire_day", np.nan)
        out.append(
            chip_link(
                "day effect",
                "The contract rule already fires on a watered day"
                + (f" ({day_text(fire)})" if pd.notna(fire) else "")
                + f"; the onset must beat the largest watered-day score, {fmt(row.get('null_auc_max'))}.",
                "bed-effect",
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
        flags = d[c].dropna()
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(2)
            numeric.append(c)
        elif d[c].dtype.kind in "iu":
            numeric.append(c)
        elif d[c].dtype.kind == "b" or (len(flags) and flags.map(lambda v: isinstance(v, bool | np.bool_)).all()):
            # a flag column read as objects (two tables concatenated, one without the flag) is still yes / no
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
            "fontFamily": components.FONT["sans"],  # "inherit" would pick up the table's own monospace
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
        css=[
            {  # an outlined action (DESIGN 5), not the browser's grey button
                "selector": ".export",
                "rule": "min-height: 44px; margin-bottom: 8px; padding: 0 16px; border: 1px solid var(--uw-teal-700); "
                "border-radius: 8px; background: var(--uw-surface); color: var(--uw-teal-700); font-weight: 700; "
                "font-family: inherit;",
            },
            {"selector": ".page-number, .current-page", "rule": "font-family: inherit;"},
            {"selector": ".previous-next-container button", "rule": "min-width: 44px; min-height: 44px;"},
        ],
        **kw,
    )


def run_alone(layout) -> None:
    """`python app/pages/<page>.py` serves just this page on :8050."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])
    app.layout = dbc.Container(layout() if callable(layout) else layout, fluid=True, className="py-3")
    app.run(debug=True)


__all__ = [
    "DEPTH_DROPPED",
    "DEPTH_DROPPED_PREFIX",
    "DEPTH_KEPT",
    "DEPTH_SIZE",
    "GROUPS",
    "METHODS_ANCHORS",
    "ONE_CONTROL_BED",
    "ONSET_HOVER",
    "SCORE_HOVER",
    "C",
    "bed_effect_chip",
    "chip",
    "chip_link",
    "day_text",
    "details",
    "fmt",
    "graph",
    "header",
    "keep_signals",
    "kept_signal",
    "load_result",
    "methods_href",
    "min_width",
    "missing_tile",
    "next_steps",
    "not_available",
    "panel",
    "readout",
    "register_page",
    "run_alone",
    "run_info",
    "run_time",
    "segmented",
    "select",
    "short",
    "signed_days",
    "source_tag",
    "stat",
    "table",
]
