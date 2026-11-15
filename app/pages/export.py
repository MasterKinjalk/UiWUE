"""Export: a derived table as CSV, filtered by bed, day and sensor, with its data dictionary on screen and as CSV.

Tables: the merged run-level traits (load_traits), the 3D canopy plant-day table (the three kept 3D traits as
change from each plant's baseline, profile_views.threed_days) and every result table found under
derived/detect and derived/diurnal, read as they are on disk (no fixed column list, so new columns export too).
The main tables come first under plain names (``TABLES``); the rest follow under their file names.
Deep link: ``/export?table=<key>&bed=W2`` preselects a table and a bed.
"""

import dash
import dash_bootstrap_components as dbc
import profile_views as pv
from components import header, next_steps, segmented, select, stat
from dash import Input, Output, State, callback, dcc, html, no_update

QUESTION = "Which table do I take away, and what does each column mean?"
dash.register_page(__name__, path="/export", name="Export", order=85, description=QUESTION)

TABLES = {  # key: (name, what one row is, what it holds), the tables most readers want, in this order
    "traits": ("All sensors · runs", "plant · run", "Every sensor's traits, one row per plant and scheduled run"),
    pv.THREED_DAYS: (
        "3D canopy · plant-days",
        "plant · day",
        "Leaf area, height and volume per plant and day, with each plant's baseline and % change, as on 3D canopy",
    ),
    "detect/scores_runs": ("Stress probability", "plant · run", "The detection model's stress probability per run"),
    "detect/fusion_v2_scores_runs": ("Fusion probability", "plant · run", "Fusion v2 stress probability per run"),
    "diurnal/scores_runs": ("Thermal scores", "plant · run", "Thermal stress scores per run (Daily rhythm)"),
    "detect/onsets": ("Onsets", "signal", "The first day each signal tells deficit from control (Early warning)"),
    "detect/placebo": ("Bed check", "signal", "Do the beds differ before irrigation off? (placebo test)"),
}
VIEWS = {"rows": "Rows", "columns": "Columns"}


def toggles(id, options, value, label):
    """Multi-select drawn as segmented buttons (native checkboxes)."""
    return html.Fieldset(
        [
            html.Legend(label, className="uw-kicker"),
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
    )


def sensor_options(t):
    tags = {k: f"{v[0]}" if k != "diurnal_" else "IR score" for k, v in pv.SOURCE.items()}
    return {k: tags[k] for k in pv.SOURCE if t is not None and any(c.startswith(k) for c in t.columns)}


def table_names(tables):
    """{key: name}: the main tables first under plain names, then the rest under their file names."""
    main = {k: TABLES[k][0] for k in TABLES if k in tables}
    rest = {k: v[0] for k, v in tables.items() if k not in TABLES}
    return main | rest


def grain(t, key):
    """What one row of a table is: from ``TABLES``, else read off its key columns."""
    if key in TABLES:
        return TABLES[key][1]
    parts = [
        p
        for p, cols in (("plant", ("plant", "plant_id")), ("signal", ("signal",)), ("rule", ("rule",)))
        if any(c in t for c in cols)
    ]
    parts += ["run"] if "hour" in t else ["day"] if pv.day_column(t) or "date" in t else []
    return " · ".join(parts) or "row"


def layout(table=None, bed=None, **_):
    tables = pv.export_tables()
    df = pv.traits()
    if df is None:
        tables.pop("traits")
    if not tables:
        return html.Div(
            [header("Export", QUESTION), pv.empty_panel("", "Derived tables", "notable", "nothing in derived/ yet")]
        )
    names = table_names(tables)
    first = table if table in names else next(iter(names))
    beds = [bed] if bed in pv.BEDS else list(pv.BEDS)
    sensors = sensor_options(pv.read_export(first))
    head = header(
        "Export",
        QUESTION,
        html.Button(
            "Download CSV", id="ex-csv", className="uw-btn uw-btn-primary", title="the rows and columns shown, as CSV"
        ),
        html.Button(
            "Dictionary",
            id="ex-dict",
            className="uw-btn uw-btn-outline",
            title="Download the meaning of every column: label, unit, direction, source and description",
        ),
        dcc.Download(id="ex-download"),
        dcc.Download(id="ex-download-dict"),
    )
    filters = html.Aside(
        [
            select("ex-table", names, first, "Table"),
            toggles("ex-beds", {b: b for b in pv.BEDS}, beds, "Beds"),
            html.Fieldset(
                [
                    html.Legend(
                        "Days",
                        className="uw-kicker",
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
            ),
            html.Div(
                toggles("ex-sensors", sensors, list(sensors), "Sensors"),
                id="ex-sensors-box",
                hidden=len(sensors) < 2,
            ),
        ],
        className="uw-side pl-filters",
        **{"aria-label": "Filters"},
    )
    main = html.Main(
        [
            html.Div(id="ex-tiles", className="uw-tiles pl-tiles"),
            html.Div(
                segmented("ex-view", VIEWS, "rows", "Show"),
                className="pl-view",
                title="the rows, or what each column means",
            ),
            html.Div(id="ex-preview"),
        ],
        className="uw-main",
    )
    steps = next_steps(
        ("Methods & limits", "/methods", "how the numbers were checked, and what they cannot tell you"),
        ("Home", "/", "the experiment and its findings"),
    )
    return html.Div([head, html.Div([filters, main], className="uw-row"), steps])


def chosen(table, beds, sensors, days, treatment="all"):
    t = pv.read_export(table)
    return pv.filter_export(t, beds or [], treatment, sensors or [], days)


@callback(
    Output("ex-sensors", "options"),
    Output("ex-sensors", "value"),
    Output("ex-sensors-box", "hidden"),
    Input("ex-table", "value"),
)
def table_sensors(table):
    opts = sensor_options(pv.read_export(table))
    return [{"label": t, "value": v} for v, t in opts.items()], list(opts), len(opts) < 2


@callback(
    Output("ex-tiles", "children"),
    Output("ex-preview", "children"),
    Input("ex-table", "value"),
    Input("ex-beds", "value"),
    Input("ex-sensors", "value"),
    Input("ex-days", "value"),
    Input("ex-view", "value"),
)
def preview(table, beds, sensors, days, view="rows"):
    t = chosen(table, beds, sensors, days)
    if t is None:
        return [], pv.empty_panel("", table or "Table", "notable", "this table is not in derived/ any more")
    rows, plants, n_days = pv.export_counts(t)
    about = TABLES[table][2] if table in TABLES else f"derived/{table}.parquet"
    tiles = [
        stat("Rows", f"{rows:,}", tint="sky", hover=about, sub=f"1 row = {grain(t, table)}"),
        stat(
            "Plants",
            "–" if plants is None else f"{plants}",
            tint="mint",
            hover=None if plants is not None else "no plant column",
        ),
        stat(
            "Days",
            "–" if n_days is None else f"{n_days}",
            tint="mint",
            hover=None if n_days is not None else "no day column",
        ),
        stat("Columns", f"{t.shape[1]}", tint="gray", hover="every column is described under Columns"),
    ]
    if view == "columns":
        d = pv.data_dictionary(t.columns, table or "traits")[["column", "label", "unit", "description"]]
        return tiles, pv.preview_table(d, n=len(d))
    if rows == 0:
        return tiles, html.Div(
            [pv.empty_mark("noscan"), "no rows"],
            className="uw-empty",
            style={"alignItems": "center"},
            title="no row matches these filters",
        )
    return tiles, html.Div(pv.preview_table(t), title=f"first 12 of {rows:,} rows", className="pl-preview")


@callback(
    Output("ex-download", "data"),
    Input("ex-csv", "n_clicks"),
    State("ex-table", "value"),
    State("ex-beds", "value"),
    State("ex-sensors", "value"),
    State("ex-days", "value"),
    prevent_initial_call=True,
)
def download_csv(_, table, beds, sensors, days):
    t = chosen(table, beds, sensors, days)
    if t is None:
        return no_update
    name = (table or "table").replace("/", "_")
    return dcc.send_data_frame(t.to_csv, f"uiwue_{name}.csv", index=False)


@callback(
    Output("ex-download-dict", "data"),
    Input("ex-dict", "n_clicks"),
    State("ex-table", "value"),
    State("ex-beds", "value"),
    State("ex-sensors", "value"),
    State("ex-days", "value"),
    prevent_initial_call=True,
)
def download_dictionary(_, table, beds, sensors, days):
    t = chosen(table, beds, sensors, days)
    if t is None:
        return no_update
    name = (table or "table").replace("/", "_")
    d = pv.data_dictionary(t.columns, table or "traits")
    return dcc.send_data_frame(d.to_csv, f"uiwue_{name}_dictionary.csv", index=False)
