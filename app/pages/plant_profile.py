"""Plant profile: one plant across every sensor over time, with links to its 3D view, the bed map and its rows.

Deep link: ``/plant-profile?bed=W2&plant=R7`` (optional ``&trait=<column>`` for the detailed chart). Bed and
plant use the identifiers of the tables (bed W1/W2/W3, plant = grid position such as R7). An unknown bed
or plant falls back to the default plant.
"""

import dash
import pandas as pd
import profile_views as pv
from components import C, chip, header, placebo_chip, readout, select, size_chip
from dash import Input, Output, State, callback, dcc, html, no_update

from uiwue.core import TRUTH_CLIP_ONSET, TRUTH_FIRST_SYMPTOM

dash.register_page(
    __name__,
    path="/plant-profile",
    name="Plant profile",
    order=20,
    description="What do the sensors say about this one plant, and when did it start to show stress?",
)

DEFAULT_BED, DEFAULT_PLANT, DEFAULT_TRAIT = "W2", "R7", "thermal_leaf_ref_dt"
QUESTION = "What do the sensors say about this one plant, and when did it start to show stress?"
GRID = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))", "gap": "12px"}


def resolve(df, bed, plant):
    """The (bed, plant) a link asks for when it exists in the tables, else the default plant."""
    if bed in pv.BEDS and plant in {o["value"] for o in pv.plant_options(df, bed)}:
        return bed, plant
    if (DEFAULT_BED, DEFAULT_PLANT) in set(zip(df.bed, df.plant)):
        return DEFAULT_BED, DEFAULT_PLANT
    return pv.imaged_plants(df)[0]


# ------------------------------------------------------------------ pieces
def treatment_chip(bed):
    if bed == "W1":
        return html.Span(
            "control",
            title="W1: watered throughout",
            className="uw-chip uw-chip-context",
            style={"background": C["sky"], "color": C["slate-600"]},
        )
    return html.Span(
        "water deficit",
        title=f"{bed}: no water from day 0 (3 Sep 11:00) to re-watering on day 12 (15 Sep 18:35)",
        className="uw-chip uw-chip-context",
        style={"background": C["sand"], "color": C["amber-900"]},
    )


def link(text, href, primary=False, hover=None):
    style = {} if primary else {"border": f"1px solid {C['teal-700']}", "color": C["teal-700"]}
    return dcc.Link(
        text, href=href, title=hover, className="uw-btn uw-btn-primary" if primary else "uw-btn", style=style
    )


def sensor_panel(df, bed, plant, tag, column, label):
    state, why = pv.sensor_state(df, bed, plant, column)
    if state != "ok":
        return pv.empty_panel(tag, label, state, why)
    value, day = pv.headline(df, bed, plant, column)
    unit = pv.change_unit(column)
    raw = pv.mode_of(column) == "raw"
    digits = 0 if unit == "%" else 2
    shown = f"{value:.0%}" if column == pv.STRESS else pv.fmt(value, "", digits, not raw)
    fig = pv.small_multiple(df, bed, plant, column)
    return html.Section(
        [
            html.Div(
                [
                    html.Span(tag, className="uw-kicker", style={"color": C["slate-600"], "width": "44px"}),
                    html.B(label, title=pv.method_hover(column), style={"flex": "1 1 auto"}),
                    *placebo_chip(column),
                    html.Span(
                        [shown, html.Small("" if column == pv.STRESS else unit)],
                        className="uw-readout",
                        title=f"{pv.day_text(day)}: last day without water with a value",
                    ),
                ],
                style={"display": "flex", "alignItems": "baseline", "gap": "8px", "flexWrap": "wrap"},
            ),
            pv.graph_box(fig, f"{label}, {bed} {plant}, change from baseline by day, against the control bed", 190),
        ],
        className="uw-panel",
        style={"padding": "12px 14px"},
        **{"aria-label": label},
    )


def multiples(df, bed, plant):
    panels = [sensor_panel(df, bed, plant, tag, col, label) for tag, col, label in pv.HEADLINES]
    if not any(c.startswith("hsi_") for c in df.columns):
        panels.append(
            pv.empty_panel("HSI", "Spectrum", "notable", "hyperspectral camera: no table in derived/spectral/ yet")
        )
    ok = [p for p in panels if p.className == "uw-panel"]
    empty = [p for p in panels if p.className != "uw-panel"]
    return [html.Div(ok, style=GRID), html.Div(empty, style=GRID | {"paddingTop": "12px"})]


def event(label, when, hover=None, muted=False):
    return html.Div(
        [
            html.Span(label, style={"flex": "1 1 auto"}),
            html.B(
                when,
                className="uw-section",
                style={"fontSize": "17px", "color": C["ink-muted"] if muted else C["ink-display"]},
            ),
        ],
        title=hover,
        style={
            "display": "flex",
            "gap": "10px",
            "alignItems": "baseline",
            "padding": "8px 0",
            "borderBottom": f"1px solid {C['rule']}",
        },
    )


def when(d):
    """'Day 4 · 7 Sep' for a date in the tables, '–' when missing."""
    if d is None or (not isinstance(d, str) and pd.isna(d)):
        return "–"
    return pv.day_text((pd.Timestamp(d) - pd.Timestamp(pv.CUTOFF)).days)


def side(df, bed, plant):
    row = pv.plant_row(bed, plant)
    s = pv.stress_summary(df, bed, plant)
    tint = "uw-sky" if bed == "W1" else "uw-sand"
    if s is None:
        stress = pv.empty_panel("MODEL", "Stress probability", "notable", "detection model: no score for this plant")
    else:
        (pk, pk_day), (now, now_day) = s["peak"], s["now"]
        after = now_day > pv.END_DAY
        how = (
            "Stress probability of the detection model (derived/detect/scores_runs), daily median of the 12 h and "
            "15 h runs; peak over day 0–12, the days without water"
        )
        stress = html.Div(
            [
                html.Div(
                    [html.Span("Stress probability", className="uw-kicker"), size_chip()],
                    style={"display": "flex", "justifyContent": "space-between", "gap": "8px", "flexWrap": "wrap"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                readout("peak, no water", "–" if pk is None else f"{pk:.0%}"),
                                html.Div("–" if pk_day is None else pv.day_text(pk_day), className="uw-readout-label"),
                            ],
                        ),
                        html.Div(
                            [
                                readout("after re-watering" if after else "latest", f"{now:.0%}"),
                                html.Div(pv.day_text(now_day), className="uw-readout-label"),
                            ],
                        ),
                    ],
                    className="uw-readouts",
                ),
            ],
            className=f"uw-card {tint}",
            title=how,
        )
    rwc_value, rwc_hover = pv.rwc_state(bed, plant)
    clip = row is not None and bool(row.truth_has_clip)
    note = row.truth_first_symptom_note if row is not None and isinstance(row.truth_first_symptom_note, str) else ""
    events = [
        event("Irrigation off", pv.day_text(0), "W2 and W3: last watering 3 Sep 11:00"),
        event(
            "Clip shift onset",
            when(row[TRUTH_CLIP_ONSET]) if clip else "–",
            "leaf-clip turgor shift onset (derived/truth/plants)" if clip else "no leaf clip on this plant",
            muted=not clip,
        ),
        event(
            "First symptom",
            when(row[TRUTH_FIRST_SYMPTOM]) if row is not None else "–",
            note or "first written stress symptom (derived/truth/plants)",
            muted=row is None or pd.isna(row[TRUTH_FIRST_SYMPTOM]),
        ),
        event(
            "Stress peak",
            "–" if s is None or s["peak"][1] is None else pv.day_text(s["peak"][1]),
            "day of the highest daily stress probability while unwatered",
            muted=s is None,
        ),
        event("Re-watered", pv.day_text(pv.END_DAY), "W2 and W3 re-watered 15 Sep 18:35, after the 18 h run"),
        event("RWC", rwc_value, rwc_hover, muted=rwc_value == "–"),
    ]
    runs = len(pv.plant_rows(df, bed, plant))
    return [
        stress,
        html.Section(events, className="uw-panel", style={"padding": "4px 16px"}, **{"aria-label": "Events"}),
        html.Div(readout("runs", f"{runs}", "in the tables"), title="runs of this plant in the merged run table"),
    ]


def actions(df, bed, plant):
    out = []
    if pv.has_depth(df, bed, plant):
        out.append(link("3D view", pv.threed_href(bed, plant), primary=True, hover="point clouds of this plant"))
    out.append(link("Bed map", f"/bed-map?bed={bed}&plant={plant}"))
    return out


# ------------------------------------------------------------------ page
def layout(bed=None, plant=None, trait=None, **_):
    df = pv.traits()
    head_q = QUESTION
    if df is None:
        return html.Div(
            [
                header("Plant profile", head_q),
                pv.empty_panel("", "Run traits", "notable", "no traits_runs table in derived/ yet"),
            ]
        )
    bed, plant = resolve(df, bed, plant)
    options = pv.trait_choices(df)
    values = {o["value"] for o in options}
    trait = trait if trait in values else DEFAULT_TRAIT if DEFAULT_TRAIT in values else options[0]["value"]
    prev_, next_ = pv.neighbours(df, bed, plant)
    nav = lambda target, sym, name: html.A(  # a real link with an aria-label (dcc.Link takes none)
        sym,
        id=f"pp-{name}",
        href=pv.profile_href(*target),
        className="uw-icon-btn",
        title=f"{name} plant",
        **{"aria-label": f"{name} plant"},
    )
    head = header(
        "Plant profile",
        head_q,
        select("pp-bed", {b: b for b in pv.BEDS}, bed, "Bed"),
        select("pp-plant", {o["value"]: o["label"] for o in pv.plant_options(df, bed)}, plant, "Plant"),
        html.Div([nav(prev_, "‹", "previous"), nav(next_, "›", "next")], className="uw-stepper"),
        html.Div(id="pp-chips", style={"display": "flex", "gap": "8px"}),
        html.Div(id="pp-actions", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        html.Button(
            "Download CSV",
            id="pp-csv",
            className="uw-btn",
            title="every run row of this plant, all columns with a value",
            style={"border": f"1px solid {C['teal-700']}", "color": C["teal-700"], "background": C["surface"]},
        ),
    )
    detail = html.Section(
        [
            html.Div(
                [
                    html.H2("One trait"),
                    html.Div(
                        dcc.Dropdown(options, trait, id="pp-trait", clearable=False, searchable=True),
                        style={"minWidth": "280px", "flex": "1 1 280px"},
                        title="trait for the detailed chart",
                    ),
                    html.Div(id="pp-detail-chips"),
                ],
                className="uw-panel-head",
            ),
            html.Div(id="pp-detail"),
        ],
        className="uw-panel",
    )
    return html.Div(
        [
            head,
            dcc.Download(id="pp-download"),
            html.Div(
                [
                    html.Main([html.Div(id="pp-multiples"), detail], className="uw-main"),
                    html.Aside(id="pp-side", className="uw-side", **{"aria-label": "Plant summary"}),
                ],
                className="uw-row",
            ),
        ]
    )


@callback(
    Output("pp-plant", "options"),
    Output("pp-plant", "value"),
    Input("pp-bed", "value"),
    State("pp-plant", "value"),
)
def pick_bed(bed, plant):
    """Plants of the chosen bed; keeps the current plant when the bed has it (so a deep link survives the load)."""
    options = pv.plant_options(pv.traits(), bed)
    values = [o["value"] for o in options]
    if plant in values:
        return options, plant
    clips = [o["value"] for o in options if o["label"].endswith("★")]
    return options, (clips or values or [None])[0]


@callback(
    Output("pp-chips", "children"),
    Output("pp-actions", "children"),
    Output("pp-multiples", "children"),
    Output("pp-side", "children"),
    Output("pp-previous", "href"),
    Output("pp-next", "href"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
)
def show_plant(bed, plant):
    df = pv.traits()
    if df is None or plant is None or plant not in {o["value"] for o in pv.plant_options(df, bed)}:
        return (no_update,) * 6
    row = pv.plant_row(bed, plant)
    chips = [treatment_chip(bed)]
    if row is not None and bool(row.truth_has_clip):
        chips.append(chip("leaf clip", "this plant carries a leaf-clip turgor sensor", "rule"))
    prev_, next_ = pv.neighbours(df, bed, plant)
    return (
        chips,
        actions(df, bed, plant),
        multiples(df, bed, plant),
        side(df, bed, plant),
        pv.profile_href(*prev_),
        pv.profile_href(*next_),
    )


@callback(
    Output("pp-detail", "children"),
    Output("pp-detail-chips", "children"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
    Input("pp-trait", "value"),
)
def show_detail(bed, plant, trait):
    df = pv.traits()
    if df is None or plant is None or trait is None:
        return no_update, no_update
    state, why = pv.sensor_state(df, bed, plant, trait)
    chips = [*placebo_chip(trait), chip("method", pv.method_hover(trait), "rule")]
    if state != "ok":
        return pv.empty_panel(pv.source_of(trait)[0], pv.label_text(trait), state, why), chips
    fig = pv.detail_figure(df, bed, plant, trait)
    if fig is None:
        return pv.empty_panel(
            pv.source_of(trait)[0], pv.label_text(trait), "noscan", "no midday run with a value"
        ), chips
    aria = f"{pv.label_text(trait)}, {bed} {plant} against the control bed and its own bed, day −10 to 19"
    return pv.graph_box(fig, aria, 340), chips


@callback(
    Output("pp-download", "data"),
    Input("pp-csv", "n_clicks"),
    State("pp-bed", "value"),
    State("pp-plant", "value"),
    prevent_initial_call=True,
)
def download(_, bed, plant):
    df = pv.traits()
    if df is None or plant is None:
        return no_update
    rows = pv.plant_csv_frame(df, bed, plant)
    return dcc.send_data_frame(rows.to_csv, f"uiwue_{bed}_{plant}.csv", index=False)
