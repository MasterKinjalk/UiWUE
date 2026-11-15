"""3D Export: filter the run table, preview it, download it as CSV with a citable method summary.

Plan section 8.3. The CSV holds the contract columns of ``traits_runs.parquet`` for the chosen
traits; the method summary (Markdown) lists the data source, the filters, the calendar, the
noise floor per trait (E1), the QC flags and the pipeline parameters (docs/threed/PIPELINE.md).
"""

from __future__ import annotations

from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import threed_data as td
from dash import Input, Output, State, callback, dash_table, dcc, html
from threed_data import schema

dash.register_page(
    __name__,
    path="/threed-export",
    name="3D Export",
    title="3D Export · UiWUE",
    description="Filter the 3D trait table, preview it and download CSV plus a method summary.",
    order=32,
)

PREVIEW_ROWS = 250  # rows sent to the preview table; the download holds every match
FILTER_IDS = [
    "te-beds",
    "te-plants",
    "te-dates",
    "te-dates",
    "te-hours",
    "te-traits",
    "te-qc",
    "te-mad",
]
FILTER_PROPS = ["value", "value", "start_date", "end_date", "value", "value", "value", "value"]


# ---------------------------------------------------------------- selection (pure functions)
def selection(beds, plants, start, end, hours, qc_only) -> pd.DataFrame:
    """Rows of the run table matching the filters. Empty plant list = all plants of the beds."""
    return td.filter_runs(
        td.load_runs(),
        beds=list(beds or []),
        plant_ids=list(plants) if plants else None,
        start=start,
        end=end,
        hours=list(hours or []),
        qc_pass_only=bool(qc_only),
    )


def filter_text(beds, plants, start, end, hours, traits, qc_only, mad) -> dict:
    return {
        "beds": ", ".join(beds or []) or "none",
        "plants": ", ".join(plants) if plants else "all in the chosen beds",
        "dates": f"{start or 'first'} to {end or 'last'}",
        "hours": ", ".join(f"{int(h):02d}" for h in hours or []) or "none",
        "traits": ", ".join(traits) if traits else "all",
        "QC pass only": "yes" if qc_only else "no",
        "MAD columns": "yes" if mad else "no",
    }


def preview(sel: pd.DataFrame, traits, mad) -> tuple[list[dict], list[dict]]:
    """DataTable rows (first ``PREVIEW_ROWS``) and columns, booleans as text, floats rounded."""
    out = td.export_frame(sel.head(PREVIEW_ROWS), traits or None, bool(mad))
    columns = []
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].map({True: "true", False: "false"})
            columns.append({"name": c, "id": c})
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(3)
            columns.append({"name": c, "id": c, "type": "numeric"})
        else:
            columns.append({"name": c, "id": c})
    return out.to_dict("records"), columns


def count_text(sel: pd.DataFrame) -> str:
    if sel.empty:
        return "No runs match these filters."
    shown = min(len(sel), PREVIEW_ROWS)
    return f"{td.table_summary(sel)} match · preview shows the first {shown:,}"


def file_stem(kind: str) -> str:
    tag = "_SYNTHETIC" if td.runs_source().synthetic else ""
    return f"threed_{kind}_{datetime.now():%Y%m%d_%H%M}{tag}"


# ---------------------------------------------------------------- layout
def _label(text: str) -> dbc.Label:
    return dbc.Label(text, class_name="small fw-semibold mb-1")


def layout(**_):
    df = td.load_runs()
    banner = td.source_banner()
    if df.empty:
        return dbc.Container([banner, dbc.Alert("The trait table has no runs.", color="danger")])
    beds = td.beds(df)
    hours = sorted(int(h) for h in df["hour"].unique())
    first, last = df["date"].min(), df["date"].max()
    filters = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _label("Beds"),
                                dbc.Checklist(
                                    id="te-beds",
                                    options=beds,
                                    value=beds,
                                    inline=True,
                                    class_name="small",
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                _label("Plants"),
                                dcc.Dropdown(
                                    id="te-plants",
                                    options=td.plant_ids(df),
                                    value=[],
                                    multi=True,
                                    placeholder="All plants in the chosen beds",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                _label("Dates"),
                                dcc.DatePickerRange(
                                    id="te-dates",
                                    start_date=first.isoformat(),
                                    end_date=last.isoformat(),
                                    min_date_allowed=first.isoformat(),
                                    max_date_allowed=last.isoformat(),
                                    display_format="YYYY-MM-DD",
                                    minimum_nights=0,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                _label("Hours"),
                                dbc.Checklist(
                                    id="te-hours",
                                    options=[{"label": f"{h:02d}", "value": h} for h in hours],
                                    value=hours,
                                    inline=True,
                                    class_name="small",
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    class_name="g-3 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _label("Traits"),
                                dcc.Dropdown(
                                    id="te-traits",
                                    multi=True,
                                    value=[],
                                    placeholder="All traits",
                                    options=[
                                        {"label": td.trait_label(t), "value": t}
                                        for t in schema.TRAITS
                                    ],
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Switch(
                                    id="te-qc",
                                    value=True,
                                    class_name="small mb-1",
                                    label="QC pass only",
                                ),
                                dbc.Switch(
                                    id="te-mad",
                                    value=True,
                                    class_name="small mb-0",
                                    label="Include MAD columns",
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Download CSV",
                                    id="te-csv-btn",
                                    color="primary",
                                    size="sm",
                                    class_name="me-2",
                                ),
                                dbc.Button(
                                    "Download method summary",
                                    id="te-md-btn",
                                    color="primary",
                                    outline=True,
                                    size="sm",
                                ),
                            ],
                            md=4,
                            class_name="text-md-end align-self-end",
                        ),
                    ],
                    class_name="g-3 align-items-end",
                ),
            ]
        ),
        class_name="mb-3",
    )
    table = dash_table.DataTable(
        id="te-table",
        page_size=15,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontSize": "0.8rem",
            "padding": "4px 8px",
            "fontFamily": "inherit",
            "whiteSpace": "nowrap",
        },
        style_header={"fontWeight": "600", "backgroundColor": "#F1F3F5"},
        style_data_conditional=[
            {"if": {"filter_query": '{qc_pass} = "false"'}, "backgroundColor": "#FFF5F5"},
        ],
    )
    return dbc.Container(
        [
            banner,
            filters,
            html.Div(id="te-count", className="small mb-2"),
            dbc.Card(
                dbc.CardBody(dcc.Loading(table, delay_show=300), class_name="p-2"),
                class_name="mb-4",
            ),
            dcc.Download(id="te-csv"),
            dcc.Download(id="te-md"),
        ],
        fluid=True,
        class_name="py-3",
    )


# ---------------------------------------------------------------- callbacks
@callback(
    Output("te-plants", "options"),
    Output("te-plants", "value"),
    Input("te-beds", "value"),
    State("te-plants", "value"),
)
def update_plants(beds, plants):
    options = td.plant_ids(td.load_runs(), list(beds or []))
    return options, [p for p in plants or [] if p in options]


@callback(
    Output("te-table", "data"),
    Output("te-table", "columns"),
    Output("te-count", "children"),
    *[Input(i, p) for i, p in zip(FILTER_IDS, FILTER_PROPS, strict=True)],
)
def update_preview(beds, plants, start, end, hours, traits, qc_only, mad):
    sel = selection(beds, plants, start, end, hours, qc_only)
    data, columns = preview(sel, traits, mad)
    return data, columns, count_text(sel)


@callback(
    Output("te-csv", "data"),
    Input("te-csv-btn", "n_clicks"),
    *[State(i, p) for i, p in zip(FILTER_IDS, FILTER_PROPS, strict=True)],
    prevent_initial_call=True,
)
def download_csv(_, beds, plants, start, end, hours, traits, qc_only, mad):
    sel = selection(beds, plants, start, end, hours, qc_only)
    out = td.export_frame(sel, traits or None, bool(mad))
    return dcc.send_data_frame(out.to_csv, f"{file_stem('traits')}.csv", index=False)


@callback(
    Output("te-md", "data"),
    Input("te-md-btn", "n_clicks"),
    *[State(i, p) for i, p in zip(FILTER_IDS, FILTER_PROPS, strict=True)],
    prevent_initial_call=True,
)
def download_summary(_, beds, plants, start, end, hours, traits, qc_only, mad):
    sel = selection(beds, plants, start, end, hours, qc_only)
    text = td.method_summary(
        sel, filter_text(beds, plants, start, end, hours, traits, qc_only, mad), traits or None
    )
    return dcc.send_string(text, f"{file_stem('method_summary')}.md")
