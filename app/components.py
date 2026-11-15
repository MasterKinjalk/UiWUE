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
from uiwue.detect import figures as _figures
from uiwue.detect.data import excluded as _left_out

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
        # hover boxes match the page's hover tips (assets/uiwue.css .uw-tip): white, a gray-500 edge, sans 13
        hoverlabel=dict(
            bgcolor=C["surface"],
            bordercolor=C["gray-500"],
            font=dict(family=FONT["sans"], size=13, color=C["ink"]),
            align="left",
        ),
        margin=dict(l=48, r=16, t=24, b=40),
    )
)
pio.templates.default = "uiwue"

BASELINE_DAYS = ((BASELINE[0] - CUTOFF).days - 0.5, (BASELINE[1] - CUTOFF).days + 0.5)
REWATER_DAY = (REWATER - CUTOFF).days + 0.5  # re-watered in the evening of day 12
DRY_DAYS = (0, REWATER_DAY)  # cutoff (day 0) to re-watering: the band edges are the events
GRAPH_CONFIG = {"displayModeBar": False}

# The app's pages as a path a newcomer can follow, grouped by the question each answers. app.py draws the nav
# from this and Home its page tiles; a registered page missing here still shows, after these groups.
# (group, its question for hover, ((path, what you learn there in a few words), ...))
NAV = (
    (
        "See",
        "Where is the stress, on which plant, and what does it look like?",
        (
            ("/bed-map", "where stress shows, day by day"),
            ("/plant-profile", "one plant, every sensor"),
            ("/threed", "canopy lost, and what came back"),
        ),
    ),
    (
        "Detect",
        "How early can each sensor see it?",
        (
            ("/early-warning", "which sensor warns first"),
            ("/sensor-comparison", "which sensors to combine"),
            ("/daily-rhythm", "the best hour to image"),
        ),
    ),
    ("Act", "Which plants need water today?", (("/irrigation", "which plants to water"),)),
    (
        "Data",
        "What can I take away, and how far can the results be trusted?",
        (("/export", "tables to download"), ("/methods", "checks and limits")),
    ),
)
BLURB = {path: blurb for _, _, items in NAV for path, blurb in items}


# Plain names of the signals, one source for every page (DESIGN 1): the catalogue's technical name goes on hover
# (``tech``). A signal missing here reads by its catalogue label.
PLAIN = {
    "thermal_leaf_air_dt": "Leaf − air temperature",
    "thermal_leaf_ref_dt": "Leaf − control leaf",
    "thermal_leaf_t": "Leaf temperature",
    "thermal_leaf_t_p10": "Coolest leaves",
    "depth_leaf_area_est": "Leaf area",
    "depth_height_p75": "Height",
    "depth_hull_volume": "Volume",
    "depth_projected_area": "Projected leaf area",
    "rgb_ngrdi": "Green–red index",
    "rgb_exg": "Greenness",
    "rgb_green_cover": "Green cover",
    "rgb_hue": "Hue",
    "diurnal_score_ref": "Thermal score",
    "detect_stress_prob": "Stress probability",
    "detect_fusion_v2_prob": "Fusion probability",
    "truth_turgor_shift": "Leaf-clip shift",
}
SIZE_ONLY = "Size only"  # the size-only baseline (DESIGN 7.6), in these two words on every page


def tech(signal):
    """The catalogue's own name of a signal ('Canopy height, p75'), for hover; the column name when unknown."""
    row = catalogue().drop_duplicates("column").set_index("column").reindex([str(signal)]).iloc[0]
    return row.label if isinstance(row.label, str) else str(signal)


def plain(signal, label=None):
    """The signal's plain name (``PLAIN``), else the label given, else the catalogue's."""
    return PLAIN.get(str(signal)) or (label if isinstance(label, str) and label else tech(signal))


# ---------------------------------------------------------------- links that carry the selection
def query(bed=None, plant=None, day=None) -> str:
    """'?bed=W2&plant=R7&day=12' (the parts there are; a plant only with its bed)."""
    parts = [(k, v) for k, v in (("bed", bed), ("plant", plant if bed else None), ("day", day)) if v not in (None, "")]
    return "?" + "&".join(f"{k}={int(v) if k == 'day' else v}" for k, v in parts) if parts else ""


def page_href(path, bed=None, plant=None, day=None) -> str:
    return f"{path}{query(bed, plant, day)}"


def methods_href(anchor) -> str:
    """The entry on Methods & limits that explains a term or a limit (``/methods#bed-effect``)."""
    return f"/methods#{anchor}"


def plant_link(bed, plant, day=None, cls="", hover=None, text=None):
    """A plant's ID as a link to its profile, carrying the day (DESIGN 4: serif ID, underlined on hover). ``text``
    shortens it where the bed is already said ('A2' after 'W1 controls')."""
    return dcc.Link(
        text or f"{bed} {plant}",
        href=page_href("/plant-profile", bed, plant, day),
        title=hover or f"{bed} {plant}: its profile",
        className=f"uw-plant-link {cls}".strip(),
    )


def _group_of(title):
    """The nav group ('Detect') of the page whose name is ``title``; None for a page outside the groups."""
    import dash  # the registry is filled by the time a layout calls header()

    path = next((p["path"] for p in dash.page_registry.values() if p["name"] == title), None)
    return next((group for group, _, items in NAV if any(path == p for p, _ in items)), None)


def header(title, question, *right, kicker=None):
    """Page header: the title (the page's question on hover), then chips and controls to its right. A caps kicker
    sits over the title: the page's nav group (See, Detect, Act, Data), or ``kicker``. It is drawn by CSS from
    ``data-kicker``, so the header's children stay title, spacer, controls."""
    kicker = kicker or _group_of(title)
    kw = {"data-kicker": kicker} if kicker else {}
    return html.Div(
        [html.H1(title, title=question, className="uw-title", **kw), html.Div(className="uw-spacer"), *right],
        className="uw-header",
    )


def not_available(what):
    """Empty state: a dashed box naming the module whose table is missing."""
    return html.Div(
        [html.Strong(what), html.Span("no table yet")],
        className="uw-empty",
        title=f"{what}: its table has not been pushed to derived/ yet",
    )


def chip(text, hover, kind="caveat", anchor=None):
    """caveat (sand), rule (outlined) or context (caps) chip; the detail goes on hover. With ``anchor`` the chip is
    a link to its entry on Methods & limits (``/methods#<anchor>``)."""
    cls = f"uw-chip uw-chip-{kind}"
    if anchor:
        return dcc.Link(text, href=methods_href(anchor), title=hover, className=cls)
    return html.Span(text, title=hover, className=cls)


def _unlinked(c):
    """A chip inside a linked tile: the same chip as a span (a link inside a link is not allowed)."""
    if isinstance(c, dcc.Link | html.A):
        return html.Span(c.children, title=getattr(c, "title", None), className=getattr(c, "className", None))
    return c


def kept_trait(signal) -> bool:
    """False for a 3D trait the app leaves out: the one definition, ``uiwue.detect.data.DEPTH_EXCLUDED`` and every
    ``depth_cyl_*`` column (noisy, saturated or redundant; docs/threed/APP.md). Every page reads the detect tables
    through it, so a dropped trait never shows up, is never ranked and never drives a chip."""
    return not _left_out(str(signal))


KEY_OTHER = frozenset(  # the non-camera columns the pickers keep: one per source, the one each page leads with
    {
        "detect_stress_prob",
        "detect_fusion_v2_prob",
        "diurnal_score_ref",
        "truth_turgor_shift",
        "env_air_t",
        "env_vpd",
        "env_light_wm2",
    }
)
KEY_SENSORS = ("thermal_", "depth_", "rgb_")  # the cameras detect/onsets scores


def key_traits(columns) -> list:
    """The columns worth a place in a signal picker, in their order. A camera trait: it passes Early warning's
    noise rule (``figures.shown``: not flagged, an onset, no later than plant size alone) for at least one deficit
    bed; every kept camera trait while detect/onsets is missing. Any other source: its lead column (``KEY_OTHER``)."""
    o = read_table("detect", "onsets")
    if o is None or o.empty or not {"signal", "group", "kind", "is_size_baseline"} <= set(o.columns):
        cameras = None
    else:
        cameras = _figures.key_signals(o[o["signal"].map(kept_trait)])

    def keep(c):
        if c.startswith(KEY_SENSORS):
            return kept_trait(c) if cameras is None else c in cameras
        return c in KEY_OTHER or not c.startswith(("detect_", "diurnal_", "truth_", "env_"))

    return [c for c in columns if keep(str(c))]


def placebo_chip(signal):
    """'bed effect' chip when Jack's placebo test (detect/placebo) flags the trait: the beds differed before irrigation
    off. The placebo table is read through the kept-trait filter (``kept_trait``), as the detect pages read theirs."""
    p = read_table("detect", "placebo")
    if p is None or not {"signal", "group", "bed_effect"} <= set(p.columns):
        return []
    p = p[p.signal.map(kept_trait)]
    r = p[(p.signal == signal) & (p.group == "all") & p.bed_effect.fillna(False).astype(bool)]
    if r.empty:
        return []
    r = r.iloc[0]
    return [bed_effect_chip(f"{plain(signal, r.label)}: score {r.placebo_auc:.2f} on watered days 24–29 Aug.")]


def bed_effect_chip(why=""):
    """The one ``bed effect`` chip (DESIGN 7.4): the beds already differed before irrigation off; links to Methods."""
    return chip("bed effect", f"The beds already differed before irrigation off. {why}".strip(), anchor="bed-effect")


def size_onset():
    """(day, row) of the size-only baseline's onset, W2 against W1 as on every page; (None, None) without one."""
    o = read_table("detect", "onsets")
    if o is None or "is_size_baseline" not in o:
        return None, None
    r = o[o.is_size_baseline.astype(bool) & (o.group == "W2")]
    col = "robust_onset_day" if "robust_onset_day" in o else "onset_day"
    if r.empty or pd.isna(r[col].iloc[0]):
        return None, None
    return int(r[col].iloc[0]), r.iloc[0]


def size_chip():
    """The size-only baseline next to a single detection result (DESIGN 7.6): 'Size only · day 6', the day plant size
    alone tells W2 from W1 (robust onset, detect/onsets). It links to its entry on Methods & limits."""
    day, r = size_onset()
    if day is None:
        return chip(f"{SIZE_ONLY} · –", "Plant size alone: no onset in detect/onsets yet", "rule", "size-baseline")
    return chip(
        f"{SIZE_ONLY} · day {day}",
        f"Plant size alone tells W2 from W1 from day {day}: the floor every result must beat "
        f"({int(r.n_deficit)} deficit, {int(r.n_control)} control plants).",
        "rule",
        "size-baseline",
    )


def mark(status, hover=None):
    """Status mark by shape: fine circle, watch triangle, irrigate diamond, no scan dashed circle."""
    return html.Span(className=f"uw-mark uw-mark-{status.replace(' ', '')}", title=hover, **{"aria-hidden": "true"})


def readout(label, value, unit="", size="readout"):
    return html.Div(
        [html.Div(label, className="uw-readout-label"), html.Div([value, html.Small(unit)], className=f"uw-{size}")]
    )


def stat(label, value, unit="", tint="gray", hover=None, sub=None, chips=(), href=None, pairs=None, cls=""):
    """Stat tile (DESIGN 5), the one tile of every page: label 12 in sentence case (with any chips), the value in
    display type with its unit (the unit drops under the value as a whole when the tile is narrow), an optional
    second line.

    ``pairs``: [(value, unit), ...] for a tile with two numbers side by side. With ``href`` the whole tile is a link
    to the page that shows the number (a hairline arrow marks it; its chips stay plain text). The method goes in
    ``hover``. Use it in a ``uw-tiles`` grid. tint: gray, sky, sand, mint or lilac; ``stat_missing`` for none."""
    values = pairs if pairs is not None else [(value, unit)]
    nums = [
        html.Div([html.Span(v), html.Small(u) if u else None], className="uw-stat") for v, u in values if v is not None
    ]
    chips = [_unlinked(c) for c in chips] if href else list(chips)
    body = [
        html.Div([html.Span(label), *chips], className="uw-readout-label uw-stat-label"),
        html.Div(nums, className="uw-stat-row") if len(nums) > 1 else nums[0] if nums else None,
        html.Div(sub, className="uw-stat-sub") if sub else None,
    ]
    cls = f"uw-tile uw-{tint} uw-stat-tile {cls}".strip()
    if href:
        return dcc.Link(
            [*body, html.Span("›", className="uw-tile-go", **{"aria-hidden": "true"})],
            href=href,
            title=hover,
            className=f"{cls} uw-tile-link",
        )
    return html.Div(body, className=cls, title=hover)


def stat_missing(label, why):
    """A stat tile without a number: the dashed 'missing' outline (DESIGN 4), an en dash, the reason on hover."""
    return html.Div(
        [html.Div(html.Span(label), className="uw-readout-label uw-stat-label"), html.Div("–", className="uw-stat")],
        className="uw-tile uw-stat-tile uw-stat-missing",
        title=why,
    )


def details(summary, *children, id=None, open=False):
    """Disclosure: a native details/summary (the summary is a real, 44 px, keyboard control), closed by default.
    For rarely used options and long lists, so the page shows its finding without them."""
    kw = {"id": id} if id else {}
    return html.Details(
        [html.Summary(summary, className="uw-summary"), *children], open=open, className="uw-details", **kw
    )


def next_links(*links):
    """The links of ``next_steps``, for a callback that re-draws them when the selection changes."""
    out = []
    for i, link in enumerate(links):
        text, href, hover = (*link, None)[:3]
        cls = "uw-btn uw-btn-primary" if i == 0 else "uw-btn uw-btn-outline"
        out.append(dcc.Link(text, href=href, title=hover, className=cls))
    return out


def next_steps(*links, label="Next steps", id=None):
    """The page's next one or two steps, as real links at its foot: (text, href[, hover]). The first is the one
    filled action (DESIGN 5 'Actions'), the rest are outlined. Carry bed, plant and day in the hrefs (``page_href``);
    give an ``id`` when a callback re-draws them (``next_links``)."""
    kw = {"id": id} if id else {}
    return html.Nav(next_links(*links), className="uw-next", **{"aria-label": label}, **kw)


def signed(x, unit="", digits=1):
    """'+4.7 °C' / '−2.0 °C' with a real minus sign; '–' when missing; no change is unsigned ('0.0', never '−0.0')."""
    if x is None or pd.isna(x):
        return "–"
    if round(float(x), digits) == 0:
        return f"{0:.{digits}f} {unit}".strip()
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
    """Dropdown options for the key catalogue traits (``key_traits``) present in `columns`, by their plain names."""
    cat = catalogue().drop_duplicates("column")
    cat = cat[cat.column.isin(columns) & ((cat.direction != 0) if stress_only else True)]
    cat = cat[cat.column.isin(key_traits(cat.column))]
    return [{"label": label_of(r.column), "value": r.column} for r in cat.itertuples()]


def label_of(column):
    """'Leaf − control leaf (°C)': the plain name with the unit, '' unit left out."""
    row = catalogue().drop_duplicates("column").set_index("column").reindex([column]).iloc[0]
    unit = row.unit if isinstance(row.unit, str) and row.unit.strip() else ""
    return f"{plain(column)} ({unit})" if unit else plain(column)


def mark_stress_period(fig, **where):
    """Baseline band (mint), no-water band (sand) and dotted hairlines at cutoff and re-watering,
    on an x axis of days since irrigation off."""
    fig.add_vrect(*BASELINE_DAYS, fillcolor=C["mint"], line_width=0, layer="below", **where)
    fig.add_vrect(*DRY_DAYS, fillcolor=C["sand"], line_width=0, layer="below", **where)
    for x in DRY_DAYS:
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=C["ink-muted"], **where)
    return fig
