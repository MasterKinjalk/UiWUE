"""Bed map: the three beds as they stand, one pot per grid cell, coloured by change from each plant's baseline.

Top: one tile per bed (its median change that day) and the plants to look at; then the day strip and the beds;
the selected plant on the right, with its profile and 3D canopy one click away. Deep link:
``/bed-map?bed=W2&plant=R7&day=9`` preselects a pot and a day (``&trait=<column>`` picks the signal).
"""

import dash
import pandas as pd
import profile_views as pv
from components import C, header, next_links, next_steps, page_href, plant_link, readout, select, stat
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from uiwue.core import SLOTS


def _coverage():
    """'39 of 216 pots were imaged', counted from the run table (None when there is none yet)."""
    try:
        df = pv.traits()
    except (OSError, ValueError):
        return None
    return None if df is None else f"{len(pv.imaged_plants(df))} of {72 * len(pv.BEDS)} pots were imaged"


COVERAGE = _coverage()
QUESTION = "Where in the beds does stress show, on which day, and which plants should I look at?" + (
    f" {COVERAGE}." if COVERAGE else ""
)
dash.register_page(__name__, path="/bed-map", name="Bed map", order=21, description=QUESTION)

DEFAULT_TRAIT, DEFAULT_HOUR, DEFAULT_DAY = "thermal_leaf_ref_dt", 15, 9
TINT = {
    "W1": ("sky", "CONTROL", "slate-600"),
    "W2": ("sand", "WATER DEFICIT", "amber-900"),
    "W3": ("sand", "WATER DEFICIT", "amber-900"),
}
DAY_MARKS = {d: str(d).replace("-", "−") for d in (-10, -5, 0, 5, 10, 12, 15, 19)}
ON = {"background": C["navy"], "borderColor": C["navy"], "color": C["surface"]}
OFF = {"background": C["surface-2"], "borderColor": C["gray-500"], "color": C["ink"]}
LOOK = 3  # plants to look at


def bed_block(bed):
    tint, kicker, accent = TINT[bed]
    return html.Section(
        [
            html.Div(
                [
                    html.Span(bed, className="uw-section"),
                    html.Span(kicker, className="uw-kicker", style={"color": C[accent]}),
                ],
                className=f"pl-bed-head uw-{tint}",
            ),
            html.Div(
                dcc.Graph(id=f"bm-{bed}", config={"displayModeBar": False}, style={"height": "560px"}),
                role="img",
                **{"aria-label": f"Bed {bed}: 72 pots, rows A to R, columns 1 to 8"},
            ),
        ],
        style={"flex": "0 0 300px", "display": "flex", "flexDirection": "column", "gap": "6px"},
        **{"aria-label": f"Bed {bed}"},
    )


def mark_legend():
    items = [
        ("noscan", "no scan", "imaged, but no run passed QC for this signal"),
        ("notable", "no table", "imaged, but this sensor has no rows for the plant"),
        ("notimaged", "not imaged", "no sensor imaged this pot"),
    ]
    return [html.Span([pv.empty_mark(k), t], title=h) for k, t, h in items]


def layout(bed=None, plant=None, trait=None, day=None, **_):
    df = pv.traits()
    head = header("Bed map", QUESTION)
    if df is None:
        return html.Div([head, pv.empty_panel("", "Run traits", "notable", "no traits_runs table in derived/ yet")])
    options = pv.trait_choices(df, stress_only=True)
    values = [o["value"] for o in options]
    trait = trait if trait in values else DEFAULT_TRAIT if DEFAULT_TRAIT in values else values[0]
    sel = {"bed": bed, "plant": plant} if bed in pv.BEDS and plant in pv.all_plants() else {"bed": "W2", "plant": "R7"}
    day = pv.day_arg(day)
    head = header(
        "Bed map",
        QUESTION,
        select("bm-trait", {o["value"]: o["label"] for o in options}, trait, "Signal"),
        html.Div(
            select("bm-hour", {h: f"{h:02d} h" for h in SLOTS}, DEFAULT_HOUR, "Run"),
            title="flyover slot: each square is one run at this hour",
        ),
    )
    day_row = html.Div(
        [
            html.Div(id="bm-daytext", className="uw-day", **{"aria-live": "polite"}),
            html.Button("Play", id="bm-play", className="uw-seg-btn", style=OFF, **{"aria-pressed": "false"}),
            html.Div(
                dcc.Slider(
                    -10,
                    19,
                    1,
                    value=DEFAULT_DAY if day is None else day,
                    marks=DAY_MARKS,
                    id="bm-day",
                    included=False,
                    allow_direct_input=False,  # the day chip beside it shows the day
                ),
                style={"flex": "1 1 320px", "minWidth": "260px"},
                title="day since irrigation off (day 0 = 3 Sep 2025)",
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "12px"},
    )
    return html.Div(
        [
            head,
            dcc.Store(id="bm-sel", data=sel),
            dcc.Interval(id="bm-tick", interval=900, disabled=True),
            html.Div(
                [
                    html.Div(id="bm-tiles", className="uw-tiles pl-tiles pl-tiles-3"),
                    html.Div(
                        [
                            html.Main(
                                [
                                    day_row,
                                    html.Div(
                                        html.Div(
                                            [bed_block(b) for b in pv.BEDS],
                                            style={"display": "flex", "gap": "16px", "width": "max-content"},
                                        ),
                                        className="uw-scroll uw-panel",
                                    ),
                                    html.Div(id="bm-legend", className="uw-legend"),
                                ],
                                className="uw-main",
                            ),
                            html.Aside(id="bm-side", className="uw-side", **{"aria-label": "Selected plant"}),
                        ],
                        className="uw-row",
                    ),
                ],
                className="pl-stack",
            ),
            next_steps(id="bm-next"),
        ]
    )


@callback(
    Output("bm-tick", "disabled"),
    Output("bm-play", "children"),
    Output("bm-play", "aria-pressed"),
    Output("bm-play", "style"),
    Input("bm-play", "n_clicks"),
    State("bm-tick", "disabled"),
    prevent_initial_call=True,
)
def play_pause(_, disabled):
    playing = bool(disabled)
    return not playing, "Pause" if playing else "Play", "true" if playing else "false", ON if playing else OFF


@callback(
    Output("bm-day", "value"), Input("bm-tick", "n_intervals"), State("bm-day", "value"), prevent_initial_call=True
)
def advance(_, day):
    return -10 if day is None or day >= 19 else day + 1


@callback(
    Output("bm-sel", "data"),
    Input("bm-W1", "clickData"),
    Input("bm-W2", "clickData"),
    Input("bm-W3", "clickData"),
    prevent_initial_call=True,
)
def pick_pot(*clicks):
    trig = ctx.triggered_id
    data = dict(zip(("bm-W1", "bm-W2", "bm-W3"), clicks)).get(trig)
    if not data or not data.get("points"):
        return no_update
    pt = data["points"][0]
    custom = pt.get("customdata")
    if not custom:
        return no_update
    return {"bed": trig.removeprefix("bm-"), "plant": custom[0]}


def stressed_first(column):
    """Sort key sign: amber (more stressed) is the high end of the change, or the low end for a trait that falls."""
    return -1 if pv.direction_of(column) < 0 and pv.mode_of(column) != "raw" else 1


def bed_tiles(df, trait, day, hour):
    """One tile per bed (the median change of its plants this run) and the plants to look at (most stressed)."""
    unit = pv.change_unit(trait)
    signed = pv.mode_of(trait) != "raw"
    digits = 0 if unit == "%" else 1
    out, values = [], []
    for b in pv.BEDS:
        pots = pv.pot_states(df, trait, day, hour, b)
        v = pots[pots.state == "value"]
        values.append(v.assign(bed=b))
        imaged = int((pots.state != "notimaged").sum())
        tint, kicker, _ = TINT[b]
        med = v.chg.median() if len(v) else None
        out.append(
            stat(
                f"{b} · {kicker.lower().removeprefix('water ')}",
                pv.fmt(med, "", digits, signed),
                unit,
                tint,
                f"Median change from each plant's own baseline, {hour:02d} h run; {len(v)} of {imaged} imaged pots "
                "have a value",
                sub=f"median of {len(v)} plants",
            )
        )
    v = pd.concat(values, ignore_index=True).dropna(subset=["chg"])
    top = v.assign(key=v.chg * stressed_first(trait)).nlargest(LOOK, "key")
    links = [
        html.Span(
            [plant_link(r.bed, r.plant, day, "pl-look-id"), html.Small(pv.fmt(r.chg, unit, digits, signed))],
            className="pl-look-item",
        )
        for r in top.itertuples()
    ]
    out.append(
        html.Div(
            [
                html.Div(html.Span("Look at"), className="uw-readout-label uw-stat-label"),
                html.Div(links or "–", className="pl-look"),
            ],
            className="uw-tile uw-gray uw-stat-tile pl-look-tile",
            title=f"The {LOOK} plants furthest to the stressed side this run, every bed",
        )
    )
    return out


@callback(
    Output("bm-W1", "figure"),
    Output("bm-W2", "figure"),
    Output("bm-W3", "figure"),
    Output("bm-tiles", "children"),
    Output("bm-legend", "children"),
    Output("bm-daytext", "children"),
    Input("bm-trait", "value"),
    Input("bm-hour", "value"),
    Input("bm-day", "value"),
    Input("bm-sel", "data"),
)
def show_map(trait, hour, day, sel):
    df = pv.traits()
    day = DEFAULT_DAY if day is None else int(day)
    hour = int(hour or DEFAULT_HOUR)
    selected = (sel["bed"], sel["plant"]) if sel else None
    figs = [pv.bed_figure(df, trait, day, hour, b, selected, imaged_only=False) for b in pv.BEDS]
    lo, hi = pv.ramp_limits(df, trait, hour)
    unit = pv.change_unit(trait)
    signed = pv.mode_of(trait) != "raw"
    flip = pv.direction_of(trait) < 0 and signed
    legend = [
        html.Span(
            pv.legend_ramp(
                pv.fmt(lo, unit, 0 if unit == "%" else 1, signed),
                pv.fmt(hi, unit, 0 if unit == "%" else 1, signed),
                flip,
            ),
            title=f"{pv.run_hover(trait, hour)} Amber = more stressed.",
        ),
        *mark_legend(),
        *pv.bed_effect_chip(trait),
    ]
    label, date = pv.day_text(day).split(" · ")
    return (*figs, bed_tiles(df, trait, day, hour), legend, [html.B(label), html.Span(date)])


@callback(
    Output("bm-side", "children"),
    Output("bm-next", "children"),
    Input("bm-sel", "data"),
    Input("bm-trait", "value"),
    Input("bm-hour", "value"),
    Input("bm-day", "value"),
)
def show_side(sel, trait, hour, day):
    df = pv.traits()
    if not sel or df is None:
        return no_update, no_update
    bed, plant = sel["bed"], sel["plant"]
    day = DEFAULT_DAY if day is None else int(day)
    hour = int(hour or DEFAULT_HOUR)
    pots = pv.pot_states(df, trait, day, hour, bed).set_index("plant")
    r = pots.loc[plant]
    tint, kicker, accent = TINT[bed]
    imaged = r.state != "notimaged"
    name = f"{bed} {plant}"
    steps = next_links(*next_of(df, bed, plant, day, imaged))
    head = html.Div(
        [
            plant_link(bed, plant, day, "pl-plant-id") if imaged else html.Span(name, className="pl-plant-id"),
            html.Span(kicker, className="uw-kicker", style={"color": C[accent]}),
        ],
        className=f"uw-card uw-{tint}",
        style={"flexDirection": "row", "alignItems": "baseline", "gap": "10px", "flexWrap": "wrap"},
    )
    if not imaged:
        return [
            head,
            html.Div(
                [pv.empty_mark("notimaged"), "not imaged"],
                className="uw-empty",
                style={"alignItems": "center"},
                title="no sensor imaged this pot",
            ),
        ], steps
    raw_unit = pv.unit_text(trait)
    if r.state == "value":
        values = html.Div(
            [
                readout("vs baseline", *pv.value_text(trait, r.chg)),
                readout("value", pv.fmt(r[trait], "", 2, False), raw_unit),
            ],
            className="uw-readouts",
            title=pv.run_hover(trait, hour),
        )
    else:
        kind = r.state
        values = html.Div(
            [pv.empty_mark(kind), pv.EMPTY_TEXT[kind]],
            className="uw-empty",
            style={"alignItems": "center"},
            title=pv.NO_BASE
            if pd.notna(r[trait])
            else pv.sensor_state(df, bed, plant, trait)[1] or "no usable run this slot",
        )
    fig = pv.small_multiple(df, bed, plant, trait, height=140, day=day)
    aria = f"{pv.label_text(trait)}, {name}, daily change by day, against the control bed"
    trend = [] if fig is None else [pv.graph_box(fig, aria, 140)]
    return [
        head,
        html.Div(
            [
                html.Div(pv.label_text(trait), className="uw-readout-label", title=pv.method_hover(trait)),
                values,
                *trend,
            ],
            className="uw-panel",
            style={"display": "flex", "flexDirection": "column", "gap": "8px"},
        ),
    ], steps


def next_of(df, bed, plant, day, imaged=True):
    """The page's next steps for the selected plant: its profile, then it in 3D (or which sensor warns first)."""
    if not imaged:
        return [("Early warning", page_href("/early-warning", bed, plant, day), "which sensor warns first")]
    out = [("Plant profile", pv.profile_href(bed, plant, day), f"every sensor for {bed} {plant}")]
    if pv.has_depth(df, bed, plant):
        out.append(
            (
                "3D canopy",
                pv.threed_href(bed, plant, day),
                f"{bed} {plant} in 3D: leaf area, height and volume against its own baseline",
            )
        )
    else:
        out.append(("Early warning", page_href("/early-warning", bed, plant, day), "which sensor warns first"))
    return out
