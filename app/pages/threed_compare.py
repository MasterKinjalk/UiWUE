"""3D Compare: one trait over time by treatment, bed or chosen plants, with the experiment results.

Plan section 8.2. Trajectories, the diurnal heatmap and the trait-vs-turgor plot are computed from
the trait table (and ``data/leaf_readings.parquet``); the result card, the E4 depression panel
and the E5 correlation table read ``data/derived/threed/results/*.csv`` and say so when a file
is missing.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import threed_data as td
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
from threed_data import schema

dash.register_page(
    __name__,
    path="/threed-compare",
    name="3D Compare",
    title="3D Compare · UiWUE",
    description="A trait over time by treatment, bed or plant, with onset, noise and turgor results.",
    order=31,
)

GRAPH_CONFIG = {"displaylogo": False}
DEFAULT_TRAIT = "height_p75"
DEFAULT_PLANTS = 4  # plants pre-selected for the hand-picked grouping
GROUP_LABELS = {"treatment": "Treatment", "bed": "Bed", "plants": "Plants"}


# ---------------------------------------------------------------- figures (pure functions)
def trajectory_figure(
    per_plant: pd.DataFrame, summary: pd.DataFrame, trait: str, normalise: bool
) -> go.Figure:
    """Mean line with a ±1 SD band per group, thin lines per plant, calendar marked."""
    if summary.empty:
        return td.message_figure("No runs match these settings.", 480)
    fmt = ".1f" if normalise else td.value_format(trait)
    u = "%" if normalise else td.unit(trait)
    fig = go.Figure()
    for i, g in enumerate(sorted(summary["group"].unique())):
        c = td.group_colour(g, i)
        s = summary[summary["group"] == g].sort_values("day")
        pp = per_plant[per_plant["group"] == g]
        n_plants = pp["plant_id"].nunique()
        if (s["n"] > 1).any():
            fig.add_trace(
                go.Scatter(
                    x=list(s["day"]) + list(s["day"][::-1]),
                    y=list(s["mean"] + s["sd"]) + list((s["mean"] - s["sd"])[::-1]),
                    fill="toself",
                    fillcolor=td.rgba(c, 0.18),
                    line={"width": 0},
                    hoverinfo="skip",
                    legendgroup=g,
                    showlegend=False,
                    name=f"{g} ±1 SD",
                )
            )
        for pid, part in pp.groupby("plant_id"):
            fig.add_trace(
                go.Scatter(
                    x=part["day"],
                    y=part["value"],
                    mode="lines",
                    line={"width": 1, "color": c},
                    opacity=0.35,
                    legendgroup=g,
                    showlegend=False,
                    name=pid,
                    hovertemplate=f"{pid}<br>%{{x|%Y-%m-%d}}: %{{y:{fmt}}} {u}<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=s["day"],
                y=s["mean"],
                mode="lines+markers",
                name=f"{g} (n = {n_plants})",
                line={"width": 3, "color": c},
                marker={"size": 5},
                legendgroup=g,
                customdata=np.c_[s["sd"], s["n"]],
                hovertemplate=(
                    f"{g}<br>%{{x|%Y-%m-%d}}<br>mean %{{y:{fmt}}} ± %{{customdata[0]:{fmt}}} {u}"
                    "<br>n = %{customdata[1]}<extra></extra>"
                ),
            )
        )
    td.add_calendar(fig)
    if normalise:
        fig.add_hline(y=0, line={"color": "#ADB5BD", "width": 1})
    ytitle = f"{trait}: change from baseline (%)" if normalise else td.trait_label(trait)
    fig.update_layout(
        template="plotly_white",
        height=480,
        margin={"l": 70, "r": 20, "t": 40, "b": 40},
        xaxis={"title": {"text": "date"}},
        yaxis={"title": {"text": ytitle}},
        legend={"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"},
        hovermode="closest",
    )
    return fig


def diurnal_figure(mats: dict[str, pd.DataFrame], trait: str, deviation: bool) -> go.Figure:
    """Day x hour heatmap per group, one shared colour scale."""
    if not mats:
        return td.message_figure("No runs match these settings.", 360)
    groups = list(mats)
    fig = make_subplots(
        rows=len(groups),
        cols=1,
        shared_xaxes=True,
        subplot_titles=groups,
        vertical_spacing=min(0.08, 0.3 / max(len(groups) - 1, 1)),
    )
    fmt = td.value_format(trait)
    for i, g in enumerate(groups, start=1):
        m = mats[g]
        fig.add_trace(
            go.Heatmap(
                x=m.index,
                y=[f"{int(h):02d} h" for h in m.columns],
                z=m.T.to_numpy(),
                coloraxis="coloraxis",
                hovertemplate=f"{g}<br>%{{x|%Y-%m-%d}} %{{y}}: %{{z:{fmt}}}<extra></extra>",
            ),
            row=i,
            col=1,
        )
    td.add_calendar(fig, annotate=False)
    u = td.unit(trait)
    if deviation:
        coloraxis = {
            "colorscale": "RdBu_r",
            "cmid": 0,
            "colorbar": {"title": {"text": f"Δ from daily mean ({u})"}},
        }
    else:
        coloraxis = {
            "colorscale": "Viridis",
            "colorbar": {"title": {"text": td.trait_label(trait)}},
        }
    fig.update_layout(
        template="plotly_white",
        height=140 + 170 * len(groups),
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        coloraxis=coloraxis,
    )
    return fig


def depression_figure(effects: pd.DataFrame, grouping: str, trait: str, start=None, end=None):
    """Mean ± SD per group of midday depression and overnight recovery (E4 columns)."""
    e = effects.copy()
    if e.empty:
        return td.message_figure("No midday or overnight pairs in this range.", 320)
    e["plant"] = e["plant"].astype(str)
    e["bed"] = e["bed"].astype(str)
    e["plant_id"] = np.where(e["plant"].str.contains("_"), e["plant"], e["bed"] + "_" + e["plant"])
    e["treatment"] = e["bed"].map(schema.BED_TREATMENT).fillna("unknown")
    e["day"] = pd.to_datetime(e["date"])
    if (s := td.parse_date(start)) is not None:
        e = e[e["day"] >= pd.Timestamp(s)]
    if (t := td.parse_date(end)) is not None:
        e = e[e["day"] <= pd.Timestamp(t)]
    e["group"] = e[td.GROUPINGS[grouping]].astype(str)
    metrics = [
        ("midday_depression", "midday depression (15 h − 9 h)"),
        ("overnight_recovery", "overnight recovery (6 h next day − 18 h)"),
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[m[1] for m in metrics])
    u = td.unit(trait)
    for col, (metric, _) in enumerate(metrics, start=1):
        daily = e.dropna(subset=[metric]).groupby(["group", "day"])[metric].agg(["mean", "std"])
        for i, g in enumerate(sorted(e["group"].unique())):
            if g not in daily.index.get_level_values(0):
                continue
            s = daily.loc[g].reset_index().fillna({"std": 0.0})
            c = td.group_colour(g, i)
            fig.add_trace(
                go.Scatter(
                    x=list(s["day"]) + list(s["day"][::-1]),
                    y=list(s["mean"] + s["std"]) + list((s["mean"] - s["std"])[::-1]),
                    fill="toself",
                    fillcolor=td.rgba(c, 0.15),
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=g,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=s["day"],
                    y=s["mean"],
                    mode="lines+markers",
                    name=g,
                    legendgroup=g,
                    line={"color": c, "width": 2},
                    marker={"size": 4},
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=0, line={"color": "#ADB5BD", "width": 1}, row=1, col=col)
    td.add_calendar(fig, annotate=False)
    fig.update_yaxes(title_text=f"{trait} ({u})", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=340,
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def turgor_figure(runs: pd.DataFrame, turgor: pd.DataFrame | None, trait: str, plant_id: str):
    """Trait per run (left axis) and the hourly leaf-clip reading (right axis), local time."""
    if runs.empty:
        return td.message_figure(f"No runs for {plant_id}.", 380)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fmt = td.value_format(trait)
    local = runs["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    for h in sorted(runs["slot"].unique()):
        m = (runs["slot"] == h).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=local[m],
                y=runs.loc[m, trait],
                mode="markers",
                name=f"{trait} {int(h):02d} h",
                marker={"size": 6, "color": td.hour_colour(h)},
                hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}: %{{y:{fmt}}}<extra></extra>",
            ),
            secondary_y=False,
        )
    if turgor is not None and not turgor.empty:
        t_local = turgor["time_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
        fig.add_trace(
            go.Scatter(
                x=t_local,
                y=turgor["turgor"],
                mode="lines",
                name="leaf clip (hourly median)",
                line={"color": "#495057", "width": 1},
                hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y:.1f}<extra>clip</extra>",
            ),
            secondary_y=True,
        )
    td.add_calendar(fig)
    fig.update_yaxes(title_text=td.trait_label(trait), secondary_y=False)
    fig.update_yaxes(
        title_text="clip reading (higher = lower turgor)",
        secondary_y=True,
        showgrid=False,
        tickformat=".0f",
    )
    fig.update_layout(
        template="plotly_white",
        height=440,
        margin={"l": 70, "r": 70, "t": 90, "b": 40},
        xaxis={"title": {"text": "local time (Europe/Berlin)"}},
        legend={"orientation": "h", "y": 1.08, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
    )
    return fig


# ---------------------------------------------------------------- result card
def _num(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _p(v) -> str:
    p = _num(v)
    if p is None:
        return "p not given"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _onset_text(v) -> str:
    n = _num(v)
    if n is not None:
        d = schema.CUTOFF + pd.Timedelta(days=int(n))
        return f"day {int(n):+d} from the cutoff ({d:%Y-%m-%d})"
    if isinstance(v, str) and v.strip() and v.strip().lower() not in {"nan", "none"}:
        return v.strip()
    return "no onset detected"


def _warning_text(v) -> str | None:
    """E2 ``warning`` may be a flag or a message; returns the text to show, or None."""
    if isinstance(v, str):
        s = v.strip()
        return None if s.lower() in {"", "false", "0", "no", "nan", "none"} else s
    if isinstance(v, (bool, np.bool_)):
        return "the beds differed before the cutoff" if v else None
    n = _num(v)
    return "the beds differed before the cutoff" if n else None


def result_card(trait: str) -> list:
    """E1 reliability, E3 slope / onset / lead time and the E2 warning for one trait."""
    u = td.unit(trait)
    row1, e1 = td.result_row("e1_noise.csv", trait)
    row3, e3 = td.result_row("e3_onset.csv", trait)
    row2, e2 = td.result_row("e2_bed_baseline.csv", trait)
    out: list = [html.H6(td.trait_label(trait), className="mb-2")]
    if e1.synthetic or e3.synthetic:
        out.append(
            dbc.Badge(
                "synthetic placeholder, not a result",
                color="warning",
                text_color="dark",
                class_name="mb-2",
            )
        )

    out.append(html.Div("Noise (E1)", className="fw-semibold small mt-1"))
    if row1:
        reliable = td._truthy(row1.get("reliable"))
        out.append(
            dbc.Badge(
                "reliable" if reliable else "not reliable at this sensor resolution",
                color="success" if reliable else "danger",
                class_name="mb-1",
            )
        )
        snr = _num(row1.get("snr"))
        out.append(
            html.Div(
                f"noise floor {td.fmt_value(trait, row1.get('noise'))} {u} · signal "
                f"{td.fmt_value(trait, row1.get('signal'))} {u} · SNR "
                f"{'–' if snr is None else format(snr, '.1f')}",
                className="small",
            )
        )
    elif e1.ok:
        out.append(html.Div(f"No row for {trait} in e1_noise.csv.", className="text-muted small"))
    else:
        out.append(html.Div(e1.note, className="text-muted small"))

    out.append(
        html.Div(
            "Stress onset (E3, 15 h, % change from baseline)", className="fw-semibold small mt-3"
        )
    )
    if row3:
        slope, lo, hi = (_num(row3.get(k)) for k in ("slope_interaction", "ci_low", "ci_high"))
        slope_unit = str(row3.get("unit") or f"{u}/day")
        if slope is not None:
            ci = "" if lo is None or hi is None else f" [{lo:+.3g}, {hi:+.3g}]"
            out.append(
                html.Div(
                    f"deficit − control slope: {slope:+.3g} {slope_unit}{ci}, "
                    f"{_p(row3.get('p_value'))}",
                    className="small",
                )
            )
        out.append(html.Div(f"onset: {_onset_text(row3.get('onset_day'))}", className="small"))
        if "onset_day_unadjusted" in row3:
            out.append(
                html.Div(
                    f"onset without Holm correction: {_onset_text(row3.get('onset_day_unadjusted'))}",
                    className="small text-muted",
                )
            )
        lead = _num(row3.get("lead_vs_symptom_days"))
        out.append(
            html.Div(
                "lead vs first visible symptom: "
                + ("not available" if lead is None else f"{lead:+.0f} days (negative = earlier)"),
                className="small",
            )
        )
    elif e3.ok:
        out.append(html.Div(f"No row for {trait} in e3_onset.csv.", className="text-muted small"))
    else:
        out.append(html.Div(e3.note, className="text-muted small"))

    if row2 and (w := _warning_text(row2.get("warning"))):
        out.append(
            dbc.Alert(
                f"E2 warning: {w}. Read E3 for this trait with care.",
                color="warning",
                class_name="small py-1 px-2 mt-3 mb-0",
            )
        )
    notes = {r.note for r in (e1, e3) if r.ok} | ({e2.note} if row2 else set())
    if notes:
        out.append(
            html.Div(
                ["Source: ", *[html.Div(n) for n in sorted(notes)]],
                className="text-muted mt-3",
                style={"fontSize": "0.75rem"},
            )
        )
    return out


def rho_table(trait: str, plant_id: str):
    """E5 rows for this plant and trait (Spearman rho per lag), or the reason they are absent."""
    res = td.load_result("e5_turgor.csv")
    if not res.ok:
        return html.Div(f"Trait–turgor correlation (E5): {res.note}.", className="text-muted small")
    df = res.df.copy()
    df["plant"] = df["plant"].astype(str)
    code = plant_id.split("_", 1)[-1]
    bed = plant_id.split("_", 1)[0]
    of_trait = df[df["trait"].astype(str) == trait]
    rows = of_trait[of_trait["plant"] == plant_id]
    if rows.empty:
        rows = of_trait[of_trait["plant"] == code]
    pooled = of_trait[of_trait["plant"] == f"pooled_{bed}"]
    if rows.empty and pooled.empty:
        return html.Div(f"E5 has no row for {plant_id} and {trait}.", className="text-muted small")
    rows = pd.concat([rows.assign(scope=plant_id), pooled.assign(scope=f"{bed} pooled")])
    rows = rows.sort_values(["scope", "lag_days"])

    def cell(v, spec):
        n = _num(v)
        return "–" if n is None else format(n, spec)

    body = [
        html.Tr(
            [
                html.Td(r.scope),
                html.Td(cell(r.lag_days, "+.0f")),
                html.Td(cell(r.rho, "+.2f")),
                html.Td(_p(r.p_value).removeprefix("p ").removeprefix("= ")),
                html.Td(cell(r.n, ".0f")),
            ]
        )
        for r in rows.itertuples()
    ]
    return html.Div(
        [
            html.Div(f"Spearman correlation with turgor (E5) · {res.note}", className="small mb-1"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("plant"),
                                html.Th("lag (days)"),
                                html.Th("rho"),
                                html.Th("p"),
                                html.Th("n"),
                            ]
                        )
                    ),
                    html.Tbody(body),
                ],
                size="sm",
                bordered=False,
                class_name="small w-auto",
            ),
        ]
    )


# ---------------------------------------------------------------- layout
def trait_options(unreliable: set[str]) -> list[dict]:
    return [
        {
            "label": td.trait_label(t) + ("  · not reliable (E1)" if t in unreliable else ""),
            "value": t,
        }
        for t in schema.TRAITS
    ]


def _label(text: str) -> dbc.Label:
    return dbc.Label(text, class_name="small fw-semibold mb-1")


def layout(**_):
    df = td.load_runs()
    banner = td.source_banner()
    if df.empty:
        return dbc.Container([banner, dbc.Alert("The trait table has no runs.", color="danger")])
    pids = td.plant_ids(df)
    clip = td.turgor_plants(df)
    first, last = df["date"].min(), df["date"].max()
    picked = [p for p in pids if p.endswith("_J5")][:2] or pids[:DEFAULT_PLANTS]
    controls = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _label("Trait"),
                                dcc.Dropdown(
                                    id="tc-trait",
                                    options=trait_options(td.unreliable_traits()),
                                    value=DEFAULT_TRAIT,
                                    clearable=False,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                _label("Group by"),
                                dbc.RadioItems(
                                    id="tc-group",
                                    options=[
                                        {"label": v, "value": k} for k, v in GROUP_LABELS.items()
                                    ],
                                    value="treatment",
                                    inline=True,
                                    class_name="small",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                _label("Plants (for 'Plants')"),
                                dcc.Dropdown(
                                    id="tc-plants",
                                    options=pids,
                                    value=picked,
                                    multi=True,
                                    disabled=True,
                                    placeholder="all plants",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                _label("Hour slot"),
                                dcc.Dropdown(
                                    id="tc-slot",
                                    options=[
                                        {"label": f"{s:02d} h", "value": s} for s in schema.SLOTS
                                    ],
                                    value=15,
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                    ],
                    class_name="g-3 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _label("Date range"),
                                dcc.DatePickerRange(
                                    id="tc-dates",
                                    start_date=first.isoformat(),
                                    end_date=last.isoformat(),
                                    min_date_allowed=first.isoformat(),
                                    max_date_allowed=last.isoformat(),
                                    display_format="YYYY-MM-DD",
                                    minimum_nights=0,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Switch(
                                    id="tc-normalise",
                                    value=False,
                                    class_name="small",
                                    label="Baseline-normalise (% change from each "
                                    "plant's 2025-09-01/02 mean)",
                                )
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Switch(
                                    id="tc-failed",
                                    value=False,
                                    class_name="small",
                                    label="Include QC-failed runs",
                                )
                            ],
                            md=2,
                        ),
                        dbc.Col(html.Div(id="tc-count", className="text-muted small"), md=2),
                    ],
                    class_name="g-3 align-items-end",
                ),
            ]
        ),
        class_name="mb-3",
    )
    diurnal_tab = [
        dbc.RadioItems(
            id="tc-deviation",
            inline=True,
            class_name="small mt-2",
            options=[
                {"label": "Deviation from the plant's daily mean", "value": "dev"},
                {"label": "Absolute value", "value": "abs"},
            ],
            value="dev",
        ),
        dcc.Graph(id="tc-heat", config=GRAPH_CONFIG, responsive=True),
        html.Div(id="tc-depr-note", className="text-muted small"),
        dcc.Graph(id="tc-depr", config=GRAPH_CONFIG, responsive=True),
    ]
    turgor_tab = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        _label("Clip plant"),
                        dcc.Dropdown(
                            id="tc-clip",
                            options=clip,
                            value=clip[0] if clip else None,
                            clearable=False,
                            placeholder="no clip plant in the table",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    html.Div(
                        "Leaf-clip readings from data/leaf_readings.parquet (from 2025-09-02); "
                        "higher readings mean lower turgor.",
                        className="text-muted small",
                    ),
                    md=9,
                    class_name="align-self-end",
                ),
            ],
            class_name="g-3 mt-1",
        ),
        dcc.Graph(id="tc-turgor", config=GRAPH_CONFIG, responsive=True),
        html.Div(id="tc-rho"),
    ]
    tabs = dbc.Tabs(
        [
            dbc.Tab(
                dcc.Graph(id="tc-traj", config=GRAPH_CONFIG, responsive=True),
                label="Trajectories",
                tab_id="traj",
            ),
            dbc.Tab(diurnal_tab, label="Diurnal rhythm", tab_id="diurnal"),
            dbc.Tab(turgor_tab, label="Trait vs turgor", tab_id="turgor"),
        ],
        id="tc-tabs",
        active_tab="traj",
    )
    return dbc.Container(
        [
            banner,
            controls,
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(tabs, class_name="p-2")), lg=9),
                    dbc.Col(dbc.Card(dbc.CardBody(id="tc-card", class_name="p-3")), lg=3),
                ],
                class_name="g-3 mb-4",
            ),
        ],
        fluid=True,
        class_name="py-3",
    )


# ---------------------------------------------------------------- callbacks
@callback(Output("tc-plants", "disabled"), Input("tc-group", "value"))
def toggle_plants(grouping):
    return grouping != "plants"


@callback(
    Output("tc-traj", "figure"),
    Output("tc-count", "children"),
    Input("tc-trait", "value"),
    Input("tc-group", "value"),
    Input("tc-plants", "value"),
    Input("tc-slot", "value"),
    Input("tc-dates", "start_date"),
    Input("tc-dates", "end_date"),
    Input("tc-normalise", "value"),
    Input("tc-failed", "value"),
)
def update_trajectories(trait, grouping, plants, slot, start, end, normalise, include_failed):
    per_plant, summary = td.trajectories(
        td.load_runs(),
        trait,
        grouping,
        plants or None,
        int(slot),
        start,
        end,
        bool(normalise),
        bool(include_failed),
    )
    fig = trajectory_figure(per_plant, summary, trait, bool(normalise))
    n = per_plant["plant_id"].nunique()
    days = per_plant["day"].nunique()
    return fig, f"{n} plants · {days} days · {len(per_plant):,} plant-days"


@callback(Output("tc-card", "children"), Input("tc-trait", "value"))
def update_card(trait):
    return result_card(trait)


@callback(
    Output("tc-heat", "figure"),
    Output("tc-depr", "figure"),
    Output("tc-depr-note", "children"),
    Input("tc-trait", "value"),
    Input("tc-group", "value"),
    Input("tc-plants", "value"),
    Input("tc-dates", "start_date"),
    Input("tc-dates", "end_date"),
    Input("tc-failed", "value"),
    Input("tc-deviation", "value"),
)
def update_diurnal(trait, grouping, plants, start, end, include_failed, deviation):
    df = td.load_runs()
    picked = (plants or None) if grouping == "plants" else None
    mats = td.diurnal_matrix(
        df, trait, grouping, start, end, bool(include_failed), deviation == "dev", picked
    )
    heat = diurnal_figure(mats, trait, deviation == "dev")
    res = td.load_result("e4_diurnal.csv")
    if res.ok:
        effects = res.df[res.df["trait"].astype(str) == trait]
        note = f"Midday depression and overnight recovery from {res.note}."
    else:
        effects = td.diurnal_effects(df, trait, bool(include_failed))
        note = f"Midday depression and overnight recovery computed from the table (E4: {res.note})."
    if picked:
        ids = effects["bed"].astype(str) + "_" + effects["plant"].astype(str)
        effects = effects[ids.isin(picked) | effects["plant"].astype(str).isin(picked)]
    return heat, depression_figure(effects, grouping, trait, start, end), note


@callback(
    Output("tc-turgor", "figure"),
    Output("tc-rho", "children"),
    Input("tc-trait", "value"),
    Input("tc-clip", "value"),
    Input("tc-failed", "value"),
    Input("tc-dates", "start_date"),
    Input("tc-dates", "end_date"),
)
def update_turgor(trait, clip, include_failed, start, end):
    if not clip:
        msg = "No plant with a leaf clip is in the trait table."
        return td.message_figure(msg, 380), html.Div()
    df = td.load_runs()
    runs = td.filter_runs(
        df, plant_ids=[clip], start=start, end=end, qc_pass_only=not include_failed
    )
    turgor = td.load_turgor(clip)
    if turgor is not None:
        lo = pd.Timestamp(td.parse_date(start) or schema.BASELINE_DATES[0], tz="UTC")
        hi = pd.Timestamp(td.parse_date(end) or df["date"].max(), tz="UTC") + pd.Timedelta(days=1)
        turgor = turgor[(turgor["time_utc"] >= lo) & (turgor["time_utc"] < hi)]
    return turgor_figure(runs, turgor, trait, clip), rho_table(trait, clip)
