"""Daily rhythm: how the daily pattern changes under stress, and when in the day to image."""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components import DIVERGING, GROUP, header, label_of, mark_stress_period, not_available, trait_options
from dash import Input, Output, callback, dcc, html

from uiwue.core import add_design, load_traits, read_table
from uiwue.diurnal import PHASE_BINS, RULES, SCHEDULES, phase_bin, rhythm

dash.register_page(
    __name__,
    path="/daily-rhythm",
    name="Daily rhythm",
    order=50,
    description="how each trait moves through the day under stress, and which time of day to image",
)

DEFAULT_TRAIT = "thermal_leaf_ref_dt"
METRICS = {
    "afternoon_minus_morning": "Afternoon minus morning (15 h − 09 h)",
    "daily_range": "Daily range (highest − lowest run)",
    "overnight_change": "Overnight change (next 06 h − 18 h)",
}
SCORES = {"diurnal_score_ref": "Thermal score vs control bed", "diurnal_score_local": "Thermal score, no control bed"}


def schedule_table(sch, rule):
    t = sch[sch.rule == rule]
    w2, w3 = (t[t.set == s].set_index("schedule").reindex(list(SCHEDULES)) for s in ("W2 tuning", "W3 check"))
    frac = lambda a, b: a.astype("Int64").astype(str) + " / " + b.astype("Int64").astype(str)
    day = lambda s: s.map(lambda v: "never" if pd.isna(v) else f"day {v:g}")
    out = pd.DataFrame(
        {
            "Flyovers": [f"{s} ★ recommended" if r else s for s, r in zip(w2.index, w2.recommended)],
            "Per day": w2.scans_per_day.values,
            "Threshold": w2.threshold.round(2).values,
            "Alert, W2 (tuning)": day(w2.median_alert_day).values,
            "Alert, W3 (check)": day(w3.median_alert_day).values,
            "W3 plants flagged": frac(w3.n_detected, w3.n_plants).values,
            "False alerts: W3-check controls": frac(w3.false_alerts_controls, w3.n_controls).values,
            "False alerts: before cutoff": frac(w3.false_alerts_placebo, w3.n_placebo_plants).values,
        }
    )
    return dbc.Table.from_dataframe(out, size="sm", hover=True, striped=False)


def layout(**_):
    df = load_traits()
    head = header(
        "Daily rhythm",
        "How does a stressed plant's day differ from a watered one's, "
        "and at which time of day should you image to see it first?",
    )
    if df is None or "thermal_leaf_t" not in df:
        return html.Div([head, not_available("The thermal table")])
    sep, sch = read_table("diurnal", "separability"), read_table("diurnal", "schedules")
    options = trait_options(df.columns)
    controls = dbc.Row(
        [
            dbc.Col([dbc.Label("Trait"), dcc.Dropdown(options, DEFAULT_TRAIT, id="dr-trait", clearable=False)], md=5),
            dbc.Col(
                [
                    dbc.Label("Compare"),
                    dbc.RadioItems(
                        options=[
                            {"label": "control vs deficit", "value": "treatment"},
                            {"label": "by bed", "value": "bed"},
                        ],
                        value="treatment",
                        id="dr-group",
                        inline=True,
                    ),
                ],
                md=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Daily metric"),
                    dcc.Dropdown(
                        [{"label": v, "value": k} for k, v in METRICS.items()],
                        "afternoon_minus_morning",
                        id="dr-metric",
                        clearable=False,
                    ),
                ],
                md=4,
            ),
        ],
        className="g-3 mb-2",
    )
    parts = [
        head,
        controls,
        html.H5("1 · The day, phase by phase", className="mt-3"),
        html.P("Mean over plants at each scheduled run (bars: standard error).", className="text-secondary small"),
        dcc.Graph(id="dr-profile"),
        html.H5("2 · One number per day", className="mt-3"),
        dcc.Graph(id="dr-rhythm"),
    ]
    if sep is not None and sch is not None:
        parts += [
            html.H5("3 · When does each time of day separate stressed from watered plants?", className="mt-3"),
            html.P(
                "Area under the ROC curve, deficit against control plants, per day and scheduled run. "
                "0.5 means no difference; 1 means every stressed plant scores above every watered one. "
                "Before the cutoff every plant was watered, so cells left of day 0 should sit near 0.5: "
                "the 06 h and 09 h runs show a bed difference even then, so read early-morning cells with care.",
                className="text-secondary small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            [{"label": v, "value": k} for k, v in SCORES.items()],
                            "diurnal_score_ref",
                            id="dr-score",
                            clearable=False,
                        ),
                        md=4,
                    ),
                    dbc.Col(dcc.Dropdown(["W2 + W3", "W2", "W3"], "W2 + W3", id="dr-sep-group", clearable=False), md=2),
                ]
            ),
            dcc.Graph(id="dr-separability"),
            html.H5("4 · What does imaging less often cost?", className="mt-3"),
            html.P(
                "Alert rule: score above the threshold on two usable scans in a row. Each threshold is the lowest "
                "that raises no false alert on W2 and its four tuning controls; W3 and the other controls are "
                "never used for tuning, so their columns are the honest check.",
                className="text-secondary small",
            ),
            dcc.Dropdown(
                [{"label": v[1], "value": k} for k, v in RULES.items()],
                "control_ref",
                id="dr-rule",
                clearable=False,
                className="mb-2",
                style={"maxWidth": "420px"},
            ),
            html.Div(id="dr-schedules"),
        ]
    return html.Div(parts)


@callback(
    Output("dr-profile", "figure"),
    Output("dr-rhythm", "figure"),
    Input("dr-trait", "value"),
    Input("dr-group", "value"),
    Input("dr-metric", "value"),
)
def update_rhythm(trait, group, metric):
    df = load_traits().dropna(subset=[trait])
    d = df.assign(phase_bin=phase_bin(df.days_since_cutoff))
    d = d[d.phase_bin != "nan"]
    p = d.groupby(["phase_bin", group, "hour"])[trait].agg(["mean", "sem"]).reset_index()
    prof = px.line(
        p,
        x="hour",
        y="mean",
        error_y="sem",
        color=group,
        facet_col="phase_bin",
        markers=True,
        category_orders={"phase_bin": PHASE_BINS[1]},
        color_discrete_map=GROUP,
        labels={"hour": "time of day (h)", "mean": label_of(trait)},
    )
    prof.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    prof.update_traces(line_width=2, marker_size=8)
    prof.update_xaxes(tickvals=[6, 12, 18])
    prof.update_layout(height=380, margin=dict(t=40), legend_title_text="")

    r = add_design(rhythm(df, trait)).dropna(subset=[metric])
    m = r.groupby(["days_since_cutoff", group])[metric].agg(["mean", "sem"]).reset_index()
    rhy = px.line(
        m,
        x="days_since_cutoff",
        y="mean",
        error_y="sem",
        color=group,
        markers=True,
        color_discrete_map=GROUP,
        labels={"days_since_cutoff": "days since irrigation was turned off", "mean": f"{METRICS[metric]}"},
    )
    rhy.update_traces(line_width=2, marker_size=8)
    rhy.update_layout(height=380, title=f"{METRICS[metric]} of {label_of(trait)}", legend_title_text="")
    return prof, mark_stress_period(rhy)


@callback(Output("dr-separability", "figure"), Input("dr-score", "value"), Input("dr-sep-group", "value"))
def update_separability(score, group):
    s = read_table("diurnal", "separability")
    s = s[(s.score == score) & (s.group == group)].pivot(index="hour", columns="day", values="auc")
    fig = go.Figure(
        go.Heatmap(
            z=s.values,
            x=s.columns,
            y=[f"{h:02d} h" for h in s.index],
            xgap=2,
            ygap=2,
            colorscale=DIVERGING,
            zmin=0,
            zmax=1,
            colorbar=dict(title="AUC"),
            hovertemplate="day %{x}, %{y} run<br>AUC %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=320, xaxis_title="days since irrigation was turned off")
    return mark_stress_period(fig)


@callback(Output("dr-schedules", "children"), Input("dr-rule", "value"))
def update_schedules(rule):
    return schedule_table(read_table("diurnal", "schedules"), rule)
