"""Plant profile: everything known about one plant, and when it started to show stress."""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from components import header, not_available, trait_options
from dash import Input, Output, callback, dcc, html
from profile_views import (
    PROB_COLUMN,
    fmt_date,
    load_spectra,
    plant_options,
    plant_row,
    rwc_figure,
    spectrum_figure,
    trace_figure,
    turgor_figure,
)

from uiwue.core import TRUTH_CLIP_ONSET, TRUTH_FIRST_SYMPTOM, load_traits, read_table

dash.register_page(
    __name__,
    path="/plant-profile",
    name="Plant profile",
    order=20,
    description="one plant's traits, leaf-clip turgor, water content and spectrum, set against the control bed",
)

DEFAULT_BED, DEFAULT_PLANT, DEFAULT_TRAIT = "W2", "R7", "thermal_leaf_ref_dt"


# ------------------------------------------------------------------ pieces
def card(label, value, note=None):
    body = [html.Div(value, className="fs-5 fw-semibold"), html.Div(label, className="text-secondary small")]
    if note:
        body.append(html.Div(note, className="text-secondary small fst-italic"))
    return dbc.Col(dbc.Card(dbc.CardBody(body)), md=True, className="mb-2")


def summary_cards(df, rwc, bed, plant, row):
    runs = df[(df.bed == bed) & (df.plant == plant)]
    cards = [card("Treatment", "control" if bed == "W1" else "water deficit", f"{len(runs)} runs in the tables")]
    if row is None:
        cards.append(card("Leaf clip and notes", "not available"))
    else:
        cards.append(
            card("Clip shift begins", fmt_date(row[TRUTH_CLIP_ONSET]) if row.truth_has_clip else "no leaf clip")
        )
        symptom = row[TRUTH_FIRST_SYMPTOM]
        cards.append(card("First stress symptom", fmt_date(symptom), row.truth_first_symptom_note or None))
    if rwc is not None and bed == "W3" and (rwc.plant == plant).any():
        r = rwc[rwc.plant == plant].iloc[0]
        cards.append(card("Relative water content", f"{r.truth_rwc:.0f} %", f"sampled {fmt_date(r.date)}"))
    prob = [c for c in df.columns if PROB_COLUMN.match(c)]
    latest = runs.dropna(subset=prob[:1]).sort_values(["date", "hour"]) if prob else runs.iloc[0:0]
    cards.append(
        card(
            "Stress probability",
            f"{latest[prob[0]].iloc[-1]:.0%}" if len(latest) else "not available yet",
            "latest run",
        )
    )
    return dbc.Row(cards, className="g-2 mb-2")


def graph_or(fig, what):
    return dcc.Graph(figure=fig) if fig is not None else not_available(what)


# ------------------------------------------------------------------ page
def layout(**_):
    df = load_traits()
    head = header("Plant profile", "What do we know about this one plant, and when did it start to show stress?")
    if df is None:
        return html.Div([head, not_available("The trait table")])
    options = trait_options(df.columns)
    default_trait = DEFAULT_TRAIT if DEFAULT_TRAIT in df else (options[0]["value"] if options else None)
    plants = plant_options(df, DEFAULT_BED)
    default_plant = DEFAULT_PLANT if DEFAULT_PLANT in {p["value"] for p in plants} else plants[0]["value"]
    return html.Div(
        [
            head,
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("Bed"), dcc.Dropdown(["W1", "W2", "W3"], DEFAULT_BED, id="pp-bed", clearable=False)],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Plant (★ = has a leaf clip)"),
                            dcc.Dropdown(plants, default_plant, id="pp-plant", clearable=False),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [dbc.Label("Trait"), dcc.Dropdown(options, default_trait, id="pp-trait", clearable=False)], md=5
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="pp-cards"),
            html.Div(id="pp-trace"),
            dbc.Row([dbc.Col(html.Div(id="pp-turgor"), md=6), dbc.Col(html.Div(id="pp-rwc"), md=6)]),
            html.H5("Spectrum", className="mt-3"),
            html.P(
                "Mean reflectance of the plant on the chosen dates. A fall at the water band and a shift of the red "
                "edge are the hyperspectral signs of water loss.",
                className="text-secondary small",
            ),
            dcc.Dropdown(id="pp-spec-dates", multi=True, placeholder="Dates to compare"),
            html.Div(id="pp-spectrum", className="mt-2"),
        ]
    )


@callback(Output("pp-plant", "options"), Output("pp-plant", "value"), Input("pp-bed", "value"))
def pick_bed(bed):
    options = plant_options(load_traits(), bed)
    clips = [o["value"] for o in options if o["label"].endswith("★")]
    return options, (clips or [o["value"] for o in options] or [None])[0]


@callback(
    Output("pp-cards", "children"),
    Output("pp-trace", "children"),
    Output("pp-turgor", "children"),
    Output("pp-rwc", "children"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
    Input("pp-trait", "value"),
)
def show_plant(bed, plant, trait):
    df, rwc, turgor = load_traits(), read_table("truth", "rwc_leaves"), read_table("truth", "turgor_runs")
    if df is None or plant is None:
        return (not_available("The trait table"),) * 4
    row = plant_row(bed, plant)
    return (
        summary_cards(df, rwc, bed, plant, row),
        graph_or(trace_figure(df, bed, plant, trait), "This trait for this plant"),
        graph_or(turgor_figure(turgor, bed, plant, row) if turgor is not None else None, "A leaf clip on this plant"),
        graph_or(rwc_figure(rwc, bed, plant), "A sampled leaf (RWC was measured on W3 plants only)"),
    )


@callback(
    Output("pp-spec-dates", "options"),
    Output("pp-spec-dates", "value"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
)
def spectrum_dates(bed, plant):
    loaded = load_spectra()
    if loaded is None:
        return [], []
    spec = loaded[0]
    dates = sorted(spec[(spec.bed == bed) & (spec.plant == plant)].date.unique())
    options = [{"label": f"{pd.Timestamp(d):%d %b}", "value": str(pd.Timestamp(d).date())} for d in dates]
    pick = [dates[i] for i in sorted({0, len(dates) // 2, len(dates) - 1})] if dates else []
    return options, [str(pd.Timestamp(d).date()) for d in pick]


@callback(
    Output("pp-spectrum", "children"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
    Input("pp-spec-dates", "value"),
)
def show_spectrum(bed, plant, dates):
    loaded = load_spectra()
    if loaded is None:
        return not_available("The hyperspectral table")
    if not dates:
        return not_available("A spectrum for this plant")
    spec, wl = loaded
    return graph_or(spectrum_figure(spec, wl, bed, plant, dates), "A spectrum for this plant")
