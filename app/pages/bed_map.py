"""Bed map: where in the three beds stress appears, run by run through the experiment."""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from components import header, not_available, trait_options
from dash import Input, Output, State, callback, dcc, html
from profile_views import bed_map_figure, dates_of

from uiwue.core import SLOTS, load_traits, read_table

dash.register_page(
    __name__,
    path="/bed-map",
    name="Bed map",
    order=21,
    description="the 216 pots in their real positions, coloured by any trait, run by run",
)

DEFAULT_TRAIT, DEFAULT_HOUR = "thermal_leaf_ref_dt", 15


def layout(**_):
    df = load_traits()
    head = header("Bed map", "Where in the beds does stress show up first, and does it spread?")
    if df is None:
        return html.Div([head, not_available("The trait table")])
    options = trait_options(df.columns)
    trait = DEFAULT_TRAIT if DEFAULT_TRAIT in df else (options[0]["value"] if options else None)
    if trait is None:
        return html.Div([head, not_available("A trait with values")])
    dates = dates_of(df)
    last = len(dates) - 1
    marks = {i: f"{pd.Timestamp(d):%d %b}" for i, d in enumerate(dates) if i % 5 == 0 or i == last}
    return html.Div(
        [
            head,
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Trait"), dcc.Dropdown(options, trait, id="bm-trait", clearable=False)], md=5),
                    dbc.Col(
                        [
                            dbc.Label("Run"),
                            dcc.Dropdown(
                                [{"label": f"{h:02d} h", "value": h} for h in SLOTS],
                                DEFAULT_HOUR,
                                id="bm-hour",
                                clearable=False,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Animation"),
                            html.Div(dbc.Button("Play", id="bm-play", color="primary", size="sm")),
                        ],
                        md=2,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dcc.Slider(0, last, 1, value=min(last, len(dates) // 2), marks=marks, id="bm-day", included=False),
            dcc.Interval(id="bm-tick", interval=700, disabled=True),
            html.P(
                "Dark ring: pot with a leaf clip. Grey ring: no reading in that run. Colours share one range across "
                "all days and beds, so a pot that warms stays visibly different.",
                className="text-secondary small mt-2",
            ),
            dcc.Graph(id="bm-map"),
        ]
    )


@callback(
    Output("bm-tick", "disabled"),
    Output("bm-play", "children"),
    Input("bm-play", "n_clicks"),
    State("bm-tick", "disabled"),
    prevent_initial_call=True,
)
def play_pause(_, disabled):
    return not disabled, "Pause" if disabled else "Play"


@callback(
    Output("bm-day", "value"), Input("bm-tick", "n_intervals"), State("bm-day", "value"), prevent_initial_call=True
)
def advance(_, day):
    n = len(dates_of(load_traits()))
    return 0 if day is None or day >= n - 1 else day + 1


def clip_set():
    """(bed, plant) of every pot with a leaf clip, from truth/plants."""
    plants = read_table("truth", "plants")
    if plants is None:
        return frozenset()
    clipped = plants[plants.truth_has_clip]
    return frozenset(zip(clipped.bed, clipped.plant))


@callback(Output("bm-map", "figure"), Input("bm-trait", "value"), Input("bm-hour", "value"), Input("bm-day", "value"))
def show_map(trait, hour, day):
    df = load_traits()
    dates = dates_of(df)
    return bed_map_figure(df, trait, dates[min(day or 0, len(dates) - 1)], hour, clip_set())
