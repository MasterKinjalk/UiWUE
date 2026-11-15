"""Export: any derived table as CSV, filtered by bed, treatment, sensor and day, with its data dictionary.

Tables: the merged run-level traits (load_traits) and every result table found under derived/detect and
derived/diurnal, read as they are on disk (no fixed column list, so new columns export too).
"""

import dash
import dash_bootstrap_components as dbc
import profile_views as pv
from components import C, header, select
from dash import Input, Output, State, callback, dcc, html, no_update

dash.register_page(
    __name__,
    path="/export",
    name="Export",
    order=85,
    description="Which rows and columns do I take away, and what does each column mean?",
)

QUESTION = "Which rows and columns do I take away, and what does each column mean?"
TREATMENT = {"all": "all", "control": "control", "deficit": "deficit"}
OUTLINE = {"border": f"1px solid {C['teal-700']}", "color": C["teal-700"], "background": C["surface"]}


def toggles(id, options, value, label):
    """Multi-select drawn as segmented buttons (native checkboxes)."""
    return html.Fieldset(
        [
            html.Legend(label, className="uw-kicker", style={"color": C["ink-muted"], "fontSize": "11px"}),
            dbc.Checklist(
                id=id,
                options=[{"label": t, "value": v} for v, t in options.items()],
                value=value,
                inline=True,
                className="uw-seg",
                inputClassName="btn-check",
                labelClassName="uw-seg-btn",
                labelCheckedClassName="uw-seg-on",
            ),
        ],
        style={"border": 0, "margin": 0, "padding": 0},
    )


def tile(label, value, tint, hover=None):
    return html.Div(
        [html.Div(label, className="uw-readout-label"), html.Div(value, className="uw-stat")],
        className=f"uw-tile uw-{tint}",
        title=hover,
    )


def sensor_options(t):
    tags = {k: f"{v[0]}" if k != "diurnal_" else "IR score" for k, v in pv.SOURCE.items()}
    return {k: tags[k] for k in pv.SOURCE if t is not None and any(c.startswith(k) for c in t.columns)}


def layout(**_):
    tables = pv.export_tables()
    df = pv.traits()
    if df is None:
        tables.pop("traits")
    if not tables:
        return html.Div(
            [header("Export", QUESTION), pv.empty_panel("", "Derived tables", "notable", "nothing in derived/ yet")]
        )
    first = next(iter(tables))
    sensors = sensor_options(pv.read_export(first))
    head = header(
        "Export",
        QUESTION,
        html.Button(
            "Download CSV", id="ex-csv", className="uw-btn uw-btn-primary", title="the filtered rows and columns"
        ),
        html.Button(
            "Dictionary",
            id="ex-dict",
            className="uw-btn",
            style=OUTLINE,
            title="column, label, unit, direction, source and description of every exported column (from the catalogues)",
        ),
        dcc.Download(id="ex-download"),
        dcc.Download(id="ex-download-dict"),
    )
    filters = html.Aside(
        [
            select("ex-table", {k: v[0] for k, v in tables.items()}, first, "Table"),
            toggles("ex-beds", {b: b for b in pv.BEDS}, list(pv.BEDS), "Beds"),
            html.Fieldset(
                [
                    html.Legend(
                        "Treatment", className="uw-kicker", style={"color": C["ink-muted"], "fontSize": "11px"}
                    ),
                    dbc.RadioItems(
                        id="ex-treatment",
                        options=[{"label": t, "value": v} for v, t in TREATMENT.items()],
                        value="all",
                        inline=True,
                        className="uw-seg",
                        inputClassName="btn-check",
                        labelClassName="uw-seg-btn",
                        labelCheckedClassName="uw-seg-on",
                    ),
                ],
                style={"border": 0, "margin": 0, "padding": 0},
            ),
            toggles("ex-sensors", sensors, list(sensors), "Sensors"),
            html.Fieldset(
                [
                    html.Legend(
                        "Days",
                        className="uw-kicker",
                        style={"color": C["ink-muted"], "fontSize": "11px"},
                        title="days since irrigation off (day 0 = 3 Sep 2025); day 12 = re-watering",
                    ),
                    dcc.RangeSlider(
                        -10,
                        19,
                        1,
                        value=[-10, 19],
                        id="ex-days",
                        marks={d: str(d).replace("-", "−") for d in (-10, -4, 0, 5, 12, 19)},
                        allowCross=False,
                    ),
                ],
                style={"border": 0, "margin": 0, "padding": 0},
            ),
        ],
        className="uw-side",
        style={"gap": "16px"},
        **{"aria-label": "Filters"},
    )
    main = html.Main(
        [
            html.Div(id="ex-tiles", className="uw-tiles"),
            html.Div(id="ex-preview"),
        ],
        className="uw-main",
    )
    return html.Div([head, html.Div([filters, main], className="uw-row")])


def chosen(table, beds, treatment, sensors, days):
    t = pv.read_export(table)
    return pv.filter_export(t, beds or [], treatment, sensors or [], days)


@callback(Output("ex-sensors", "options"), Output("ex-sensors", "value"), Input("ex-table", "value"))
def table_sensors(table):
    opts = sensor_options(pv.read_export(table))
    return [{"label": t, "value": v} for v, t in opts.items()], list(opts)


@callback(
    Output("ex-tiles", "children"),
    Output("ex-preview", "children"),
    Input("ex-table", "value"),
    Input("ex-beds", "value"),
    Input("ex-treatment", "value"),
    Input("ex-sensors", "value"),
    Input("ex-days", "value"),
)
def preview(table, beds, treatment, sensors, days):
    t = chosen(table, beds, treatment, sensors, days)
    if t is None:
        return [], pv.empty_panel("", table or "Table", "notable", "this table is not in derived/ any more")
    rows, plants, n_days = pv.export_counts(t)
    tiles = [
        tile("rows", f"{rows:,}", "sky"),
        tile(
            "plants",
            "–" if plants is None else f"{plants}",
            "mint",
            None if plants is not None else "no bed and plant columns",
        ),
        tile(
            "days",
            "–" if n_days is None else f"{n_days}",
            "mint",
            None if n_days is not None else "no date or day column",
        ),
        tile("columns", f"{t.shape[1]}", "gray"),
    ]
    if rows == 0:
        return tiles, html.Div(
            [pv.empty_mark("noscan"), "no rows"],
            className="uw-empty",
            style={"alignItems": "center"},
            title="no row matches these filters",
        )
    return tiles, html.Div(pv.preview_table(t), title=f"first 12 of {rows:,} rows")


@callback(
    Output("ex-download", "data"),
    Input("ex-csv", "n_clicks"),
    State("ex-table", "value"),
    State("ex-beds", "value"),
    State("ex-treatment", "value"),
    State("ex-sensors", "value"),
    State("ex-days", "value"),
    prevent_initial_call=True,
)
def download_csv(_, table, beds, treatment, sensors, days):
    t = chosen(table, beds, treatment, sensors, days)
    if t is None:
        return no_update
    name = (table or "table").replace("/", "_")
    return dcc.send_data_frame(t.to_csv, f"uiwue_{name}.csv", index=False)


@callback(
    Output("ex-download-dict", "data"),
    Input("ex-dict", "n_clicks"),
    State("ex-table", "value"),
    State("ex-beds", "value"),
    State("ex-treatment", "value"),
    State("ex-sensors", "value"),
    State("ex-days", "value"),
    prevent_initial_call=True,
)
def download_dictionary(_, table, beds, treatment, sensors, days):
    t = chosen(table, beds, treatment, sensors, days)
    if t is None:
        return no_update
    name = (table or "table").replace("/", "_")
    d = pv.data_dictionary(t.columns, table or "traits")
    return dcc.send_data_frame(d.to_csv, f"uiwue_{name}_dictionary.csv", index=False)
