"""Bed map: the three beds as they stand, one pot per grid cell, coloured by change from each plant's baseline.

Click a pot to see its reading and open its profile. Deep link: ``/bed-map?bed=W2&plant=R7`` preselects a pot
(``&trait=<column>`` picks the signal).
"""

import dash
import profile_views as pv
from components import C, chip, header, placebo_chip, readout, segmented, select
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
QUESTION = "Where in the beds does stress show up, run by run?" + (f" {COVERAGE}." if COVERAGE else "")
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


def bed_block(bed):
    tint, kicker, accent = TINT[bed]
    return html.Section(
        [
            html.Div(
                [
                    html.Span(bed, className="uw-section"),
                    html.Span(kicker, className="uw-kicker", style={"color": C[accent]}),
                    html.Span(
                        id=f"bm-count-{bed}", style={"fontSize": "13px", "color": C["ink"], "marginLeft": "auto"}
                    ),
                ],
                className=f"uw-{tint}",
                style={
                    "display": "flex",
                    "alignItems": "baseline",
                    "gap": "8px",
                    "padding": "8px 12px",
                    "borderRadius": "8px",
                },
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


def layout(bed=None, plant=None, trait=None, **_):
    df = pv.traits()
    head = header("Bed map", QUESTION)
    if df is None:
        return html.Div([head, pv.empty_panel("", "Run traits", "notable", "no traits_runs table in derived/ yet")])
    options = pv.trait_choices(df, stress_only=True)
    values = [o["value"] for o in options]
    trait = trait if trait in values else DEFAULT_TRAIT if DEFAULT_TRAIT in values else values[0]
    sel = {"bed": bed, "plant": plant} if bed in pv.BEDS and plant in pv.all_plants() else {"bed": "W2", "plant": "R7"}
    head = header(
        "Bed map",
        QUESTION,
        select("bm-trait", {o["value"]: o["label"] for o in options}, trait, "Signal"),
        segmented("bm-hour", {h: f"{h:02d} h" for h in SLOTS}, DEFAULT_HOUR, "Run"),
        html.Button(
            "Imaged only",
            id="bm-imaged",
            className="uw-seg-btn",
            style=OFF,
            **{"aria-pressed": "false"},
            title="hide the pots no sensor imaged",
        ),
    )
    day_row = html.Div(
        [
            html.Div(id="bm-daytext", className="uw-day", **{"aria-live": "polite"}),
            html.Button("Play", id="bm-play", className="uw-seg-btn", style=OFF, **{"aria-pressed": "false"}),
            html.Div(
                dcc.Slider(-10, 19, 1, value=DEFAULT_DAY, marks=DAY_MARKS, id="bm-day", included=False),
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
            dcc.Store(id="bm-on", data=False),
            dcc.Interval(id="bm-tick", interval=900, disabled=True),
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
        ]
    )


@callback(
    Output("bm-on", "data"),
    Output("bm-imaged", "aria-pressed"),
    Output("bm-imaged", "style"),
    Input("bm-imaged", "n_clicks"),
    State("bm-on", "data"),
    prevent_initial_call=True,
)
def toggle_imaged(_, on):
    on = not on
    return on, "true" if on else "false", ON if on else OFF


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


@callback(
    Output("bm-W1", "figure"),
    Output("bm-W2", "figure"),
    Output("bm-W3", "figure"),
    Output("bm-count-W1", "children"),
    Output("bm-count-W2", "children"),
    Output("bm-count-W3", "children"),
    Output("bm-legend", "children"),
    Output("bm-daytext", "children"),
    Input("bm-trait", "value"),
    Input("bm-hour", "value"),
    Input("bm-day", "value"),
    Input("bm-on", "data"),
    Input("bm-sel", "data"),
)
def show_map(trait, hour, day, imaged_only, sel):
    df = pv.traits()
    day = DEFAULT_DAY if day is None else int(day)
    hour = int(hour or DEFAULT_HOUR)
    selected = (sel["bed"], sel["plant"]) if sel else None
    figs = [pv.bed_figure(df, trait, day, hour, b, selected, bool(imaged_only)) for b in pv.BEDS]
    counts = []
    for b in pv.BEDS:
        n = pv.bed_counts(df, trait, day, hour, b)
        imaged = 72 - n["notimaged"]
        counts.append(
            html.Span(
                f"{n['value']} of {imaged} imaged",
                title=f"{n['value']} values · {n['noscan']} no scan · {n['notable']} no table · {n['notimaged']} not imaged",
            )
        )
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
            title=f"{pv.run_hover(trait, hour)}; amber = more stressed",
        ),
        *mark_legend(),
        html.Span(
            [
                html.Span(
                    style={
                        "width": "18px",
                        "height": "18px",
                        "boxSizing": "border-box",
                        "borderRadius": "4px",
                        "border": f"3px solid {C['navy']}",
                    },
                    **{"aria-hidden": "true"},
                ),
                "selected",
            ]
        ),
        *placebo_chip(trait),
        chip("method", pv.run_hover(trait, hour), "rule"),
    ]
    label, date = pv.day_text(day).split(" · ")
    return (*figs, *counts, legend, [html.B(label), html.Span(date)])


@callback(
    Output("bm-side", "children"),
    Input("bm-sel", "data"),
    Input("bm-trait", "value"),
    Input("bm-hour", "value"),
    Input("bm-day", "value"),
)
def show_side(sel, trait, hour, day):
    df = pv.traits()
    if not sel or df is None:
        return no_update
    bed, plant = sel["bed"], sel["plant"]
    day = DEFAULT_DAY if day is None else int(day)
    hour = int(hour or DEFAULT_HOUR)
    pots = pv.pot_states(df, trait, day, hour, bed).set_index("plant")
    r = pots.loc[plant]
    tint, kicker, accent = TINT[bed]
    head = html.Div(
        [
            html.Div(
                [
                    html.Span(f"{bed} {plant}", className="uw-stat"),
                    html.Span(kicker, className="uw-kicker", style={"color": C[accent]}),
                ],
                style={"display": "flex", "alignItems": "baseline", "gap": "10px", "flexWrap": "wrap"},
            ),
        ],
        className=f"uw-card uw-{tint}",
    )
    if r.state == "notimaged":
        return [
            head,
            html.Div(
                [pv.empty_mark("notimaged"), "not imaged"],
                className="uw-empty",
                style={"alignItems": "center"},
                title="no sensor imaged this pot",
            ),
        ]
    unit, raw_unit = pv.change_unit(trait), pv.unit_text(trait)
    signed = pv.mode_of(trait) != "raw"
    if r.state == "value":
        values = html.Div(
            [
                readout("vs baseline", pv.fmt(r.chg, "", 0 if unit == "%" else 2, signed), unit),
                readout("value", pv.fmt(r[trait], "", 2, False), raw_unit),
            ],
            className="uw-readouts",
            title=f"{pv.label_unit(trait)} · {pv.day_text(day)} · {hour:02d} h. {pv.run_hover(trait, hour)}",
        )
    else:
        kind = r.state
        values = html.Div(
            [pv.empty_mark(kind), pv.EMPTY_TEXT[kind]],
            className="uw-empty",
            style={"alignItems": "center"},
            title=pv.sensor_state(df, bed, plant, trait)[1] or "no usable run this slot",
        )
    links = [dcc.Link("Open profile", href=pv.profile_href(bed, plant), className="uw-btn uw-btn-primary")]
    if pv.has_depth(df, bed, plant):
        links.append(
            dcc.Link(
                "3D canopy",
                href=pv.threed_href(bed, plant),
                title="this plant's point clouds, leaf area, height and volume against its own baseline",
                className="uw-btn",
                style={"border": f"1px solid {C['teal-700']}", "color": C["teal-700"]},
            )
        )
    return [
        head,
        html.Div(
            [html.Div(pv.label_text(trait), className="uw-readout-label"), values],
            className="uw-panel",
            style={"display": "flex", "flexDirection": "column", "gap": "8px"},
        ),
        html.Div(links, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
    ]
