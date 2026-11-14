"""Home: what the pipeline extracted, for which plants and dates."""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from components import INK, header, not_available
from dash import dcc, html

from uiwue.core import DERIVED, catalogue, load_traits

dash.register_page(__name__, path="/", name="Home", order=0)

SENSORS = {"thermal_": "Thermal camera", "depth_": "Depth (3D)", "rgb_": "RGB camera", "hsi_": "Hyperspectral"}


def tile(value, label):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(f"{value:,}", className="fs-3 fw-semibold"),
                    html.Div(label, className="text-secondary small"),
                ]
            )
        ),
        md=3,
    )


def tables():
    rows = []
    for p in sorted(DERIVED.glob("*/*.parquet")):
        t = pd.read_parquet(p)
        keyed = {"bed", "plant"} <= set(t.columns)
        rows.append(
            {
                "Module": p.parent.name,
                "Table": p.stem,
                "Rows": f"{len(t):,}",
                "Plants": t[["bed", "plant"]].drop_duplicates().shape[0] if keyed else "",
                "Dates": f"{pd.to_datetime(t.date).min():%d %b} – {pd.to_datetime(t.date).max():%d %b}"
                if "date" in t
                else "",
            }
        )
    return pd.DataFrame(rows)


def coverage(df):
    """Plants with a reading per sensor and date."""
    rows = {}
    for prefix, name in SENSORS.items():
        cols = [c for c in df.columns if c.startswith(prefix) and pd.api.types.is_float_dtype(df[c])]
        if cols:
            rows[name] = df[df[cols].notna().any(axis=1)].assign(pid=df.bed + df.plant).groupby("date").pid.nunique()
    cov = pd.DataFrame(rows).T.fillna(0)
    fig = go.Figure(
        go.Heatmap(
            z=cov.values,
            x=cov.columns,
            y=cov.index,
            xgap=2,
            ygap=2,
            colorbar=dict(title="plants"),
            colorscale=[[0, "#cde2fb"], [1, "#184f95"]],
            zmin=0,
            zmax=df[["bed", "plant"]].drop_duplicates().shape[0],
            hovertemplate="%{y}, %{x|%d %b}: %{z} plants<extra></extra>",
        )
    )
    fig.add_vline(
        x=pd.Timestamp("2025-09-03"),
        line_width=1,
        line_color=INK["muted"],
        annotation_text="irrigation off",
        annotation_font=dict(size=11, color=INK["muted"]),
    )
    fig.update_layout(height=140 + 45 * len(cov), title="Plants with data, per sensor and day", margin=dict(l=130))
    return fig


def layout(**_):
    df = load_traits()
    intro = header(
        "UiWUE: a workbench for water-deficit experiments",
        "Point the pipeline at the image folders; it extracts per-plant traits from every sensor into one "
        "table, shows when stress became detectable in each signal, and when a plant needs water.",
    )
    if df is None:
        return html.Div([intro, not_available("Every module")])
    cat = catalogue()
    tiles = dbc.Row(
        [
            tile(df[["bed", "plant"]].drop_duplicates().shape[0], "plants"),
            tile(len(df), "plant scans (plant × run)"),
            tile(int((cat.direction != 0).sum()), "stress traits extracted"),
            tile(df.date.nunique(), "days, 5 scheduled runs each"),
        ],
        className="g-3 mb-3",
    )
    guide = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Pages"),
                html.Ul(
                    [
                        html.Li([dcc.Link(p["name"], href=p["relative_path"]), f": {p.get('description', '')}"])
                        for p in sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
                        if p["path"] != "/"
                    ]
                ),
                html.P(
                    "Beds: W1 is the well-watered control; W2 and W3 had irrigation turned off on 3 Sep and "
                    "back on in the evening of 15 Sep. Each plant's values are compared with its own healthy days "
                    "(30 Aug – 2 Sep) at the same time of day.",
                    className="text-secondary small mb-0",
                ),
            ]
        ),
        className="mb-3",
    )
    return html.Div(
        [
            intro,
            tiles,
            guide,
            dcc.Graph(figure=coverage(df)),
            html.H5("Derived tables", className="mt-3"),
            dbc.Table.from_dataframe(tables(), size="sm", hover=True, className="w-auto"),
        ]
    )
