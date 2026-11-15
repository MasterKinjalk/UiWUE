"""Irrigation advisor: which plants need water, why, and how far to trust the call."""

from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from components import INK, SERIES, STATUS, header, label_of, mark_stress_period, not_available
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

from uiwue.core import load_traits, read_table
from uiwue.diurnal import RULES, SCHEDULES, status

dash.register_page(
    __name__,
    path="/irrigation",
    name="Irrigation advisor",
    order=60,
    description="which plants need water on a given day, under five alert rules, with their track record",
)

BEDS = {"W1": "W1 · watered control", "W2": "W2 · water deficit", "W3": "W3 · water deficit"}
ROWS = "ABCDEFGHIJKLMNOPQR"
DEFAULT_RULE, DEFAULT_DAY, DEFAULT_PLANT = "local_line", 4, "W2_H5"


def recommended(rule):
    sch = read_table("diurnal", "schedules")
    return sch[(sch.rule == rule) & sch.recommended].schedule.iloc[0]


@lru_cache
def scans(rule, schedule):
    """Every run of every plant with its alert status under (rule, schedule), plus the threshold."""
    sch = read_table("diurnal", "schedules")
    k = sch[(sch.rule == rule) & (sch.schedule == schedule)].threshold.iloc[0]
    df = load_traits()
    col = RULES[rule][0]
    return df.assign(status=status(df, col, SCHEDULES[schedule], k), value=df[col], pid=df.bed + "_" + df.plant), k


def layout(**_):
    df, sch = load_traits(), read_table("diurnal", "schedules")
    head = header("Irrigation advisor", "Which plants need water today, why, and how far can the call be trusted?")
    if df is None or sch is None or "diurnal_score_ref" not in df:
        return html.Div([head, not_available("The daily-rhythm scores")])
    days = sorted(int(d) for d in df.days_since_cutoff.unique())
    controls = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Alert rule"),
                    dcc.Dropdown(
                        [{"label": v[1], "value": k} for k, v in RULES.items()],
                        DEFAULT_RULE,
                        id="ir-rule",
                        clearable=False,
                    ),
                ],
                md=4,
            ),
            dbc.Col(
                [
                    dbc.Label("Flyovers used"),
                    dcc.Dropdown(list(SCHEDULES), recommended(DEFAULT_RULE), id="ir-schedule", clearable=False),
                ],
                md=2,
            ),
            dbc.Col(
                [
                    dbc.Label("Day (days since irrigation was turned off)"),
                    dcc.Slider(
                        min(days),
                        max(days),
                        1,
                        value=DEFAULT_DAY,
                        id="ir-day",
                        marks={d: str(d) for d in days if d % 5 == 0},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                md=6,
            ),
        ],
        className="g-3 mb-2",
    )
    plants = sorted(df.bed.str.cat(df.plant, sep="_").unique())
    return html.Div(
        [
            head,
            controls,
            dbc.Row([dbc.Col(dcc.Graph(id="ir-map"), lg=8), dbc.Col(html.Div(id="ir-card"), lg=4)], className="g-3"),
            html.H5("One plant over time", className="mt-3"),
            dcc.Dropdown(
                plants,
                DEFAULT_PLANT if DEFAULT_PLANT in plants else plants[0],
                id="ir-plant",
                clearable=False,
                style={"maxWidth": "240px"},
            ),
            dcc.Graph(id="ir-plant-fig"),
        ]
    )


@callback(Output("ir-schedule", "value"), Input("ir-rule", "value"))
def pick_schedule(rule):
    return recommended(rule)


@callback(
    Output("ir-map", "figure"),
    Output("ir-card", "children"),
    Input("ir-rule", "value"),
    Input("ir-schedule", "value"),
    Input("ir-day", "value"),
)
def update_map(rule, schedule, day):
    s, k = scans(rule, schedule)
    today = s[(s.days_since_cutoff == day) & s.status.notna()].sort_values("hour").groupby("pid").tail(1)
    m = s[["pid", "bed", "plant"]].drop_duplicates().merge(today, on=["pid", "bed", "plant"], how="left")
    m["status"] = m.status.fillna("no reading")
    date = s.loc[s.days_since_cutoff == day, "date"].iloc[0]

    fig = make_subplots(1, 3, subplot_titles=list(BEDS.values()), shared_yaxes=True, horizontal_spacing=0.03)
    shown = set()
    for i, bed in enumerate(BEDS, 1):
        for name, (color, symbol) in STATUS.items():
            g = m[(m.bed == bed) & (m.status == name)]
            hover = [
                f"<b>{r.pid}</b> · {name}<br>{label_of(RULES[rule][0])}: {r.value:.2f} (threshold {k:.2f})"
                f"<br>scan at {r.hour:.0f} h, {r.env_sky} · demand since watering {r.env_vpd_since_water:.0f} kPa·h"
                if name != "no reading"
                else f"<b>{r.pid}</b> · no usable scan this day"
                for r in g.itertuples()
            ]
            fig.add_trace(
                go.Scatter(
                    x=g.plant.str[1:].astype(int),
                    y=g.plant.str[0].map(ROWS.index),
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=name not in shown and len(g) > 0,
                    hovertext=hover,
                    hoverinfo="text",
                    marker=dict(color=color, symbol=symbol, size=15, line=dict(width=2, color=INK["surface"])),
                ),
                row=1,
                col=i,
            )
            shown |= {name} if len(g) else set()
    fig.update_xaxes(range=[0.3, 8.7], tickvals=list(range(1, 9)), showgrid=False)
    fig.update_yaxes(range=[17.6, -0.6], tickvals=list(range(18)), ticktext=list(ROWS), showgrid=False)
    fig.update_layout(
        height=560, title=f"{date:%a %d %b} (day {day}): status at the last usable scan", margin=dict(t=80)
    )

    sch = read_table("diurnal", "schedules")
    row = {r.set: r for r in sch[(sch.rule == rule) & (sch.schedule == schedule)].itertuples()}
    w2, w3 = row["W2 tuning"], row["W3 check"]
    when = lambda v: "never" if np.isnan(v) else f"day {v:g}"
    need = sorted(m.loc[m.status == "irrigate", "pid"])
    sky = s[(s.days_since_cutoff == day) & s.hour.isin(SCHEDULES[schedule])].groupby("hour").env_sky.first()
    counts = m.status.value_counts()
    card = dbc.Card(
        dbc.CardBody(
            [
                html.H6("Water now", className="mb-1"),
                html.P(", ".join(need) if need else "No plant is flagged.", className="mb-2"),
                html.P(" · ".join(f"{n}: {counts.get(n, 0)}" for n in STATUS), className="text-secondary small"),
                html.P(
                    "Sky at the flyovers (DWD Belm): " + " · ".join(f"{h:02d} h {v}" for h, v in sky.dropna().items()),
                    className="text-secondary small",
                ),
                html.Hr(),
                html.H6("Track record of this rule", className="mb-1"),
                html.P(
                    f"Needs {RULES[rule][2]}. Alerts when the value is above {k:.2f} on two usable scans in a row.",
                    className="small",
                ),
                html.Ul(
                    [
                        html.Li(
                            f"W2 (used to set the threshold): {w2.n_detected}/{w2.n_plants} plants flagged, "
                            f"median {when(w2.median_alert_day)}."
                        ),
                        html.Li(
                            f"W3 (held out): {w3.n_detected}/{w3.n_plants} flagged, median {when(w3.median_alert_day)}."
                        ),
                        html.Li(
                            f"False alerts on held-out watered plants: {w3.false_alerts_controls}/{w3.n_controls}; "
                            f"before the cutoff: {w3.false_alerts_placebo}/{w3.n_placebo_plants}."
                        ),
                    ],
                    className="small",
                ),
                html.P(
                    "Weak-evidence scans (missing data, or a dull day at 09–15 h) are skipped, not counted.",
                    className="text-secondary small mb-0",
                ),
            ]
        )
    )
    return fig, card


@callback(
    Output("ir-plant-fig", "figure"),
    Input("ir-rule", "value"),
    Input("ir-schedule", "value"),
    Input("ir-plant", "value"),
)
def update_plant(rule, schedule, pid):
    s, k = scans(rule, schedule)
    p = s[(s.pid == pid) & s.hour.isin(SCHEDULES[schedule])].sort_values(["date", "hour"])
    x = p.days_since_cutoff + p.hour / 24
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=p.value,
            mode="lines",
            line=dict(width=2, color=SERIES[0]),
            name=label_of(RULES[rule][0]),
            hoverinfo="skip",
        )
    )
    for name, (color, symbol) in STATUS.items():
        g = p[p.status.fillna("no reading") == name]
        fig.add_trace(
            go.Scatter(
                x=g.days_since_cutoff + g.hour / 24,
                y=g.value,
                mode="markers",
                name=name if name != "no reading" else "skipped scan",
                marker=dict(color=color, symbol=symbol, size=9, line=dict(width=1, color=INK["surface"])),
                customdata=g[["hour"]].assign(d=g.date.dt.strftime("%d %b")),
                hovertemplate="%{customdata[1]} %{customdata[0]:.0f} h: %{y:.2f}<extra>" + name + "</extra>",
            )
        )
    fig.add_hline(
        y=k,
        line_width=1,
        line_dash="dot",
        line_color=INK["secondary"],
        annotation_text=f"threshold {k:.2f}",
        annotation_position="bottom right",
    )
    first = p[(p.status == "irrigate") & (p.days_since_cutoff >= 0)]
    title = (
        f"{pid}: first 'irrigate' on day {first.days_since_cutoff.iloc[0]}" if len(first) else f"{pid}: never flagged"
    )
    fig.update_layout(
        height=380,
        title=title,
        xaxis_title="days since irrigation was turned off",
        yaxis_title=label_of(RULES[rule][0]),
    )
    return mark_stress_period(fig)
