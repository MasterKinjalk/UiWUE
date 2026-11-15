"""Daily rhythm: how the daily pattern changes under stress, and when in the day to image."""

import dash
import pandas as pd
import plotly.express as px
from components import (
    GROUP,
    RULE_SHORT,
    C,
    chip,
    day_label,
    graph,
    header,
    mark_stress_period,
    not_available,
    panel,
    readout,
    segmented,
    select,
    signed,
    trait_options,
    unit_of,
)
from dash import Input, Output, callback, html

from uiwue.core import SLOTS, add_design, load_traits, read_table
from uiwue.diurnal import PHASE_BINS, SCHEDULES, phase_bin, rhythm

dash.register_page(
    __name__,
    path="/daily-rhythm",
    name="Daily rhythm",
    order=50,
    description="how each trait moves through the day under stress, and which time of day to image",
)

DEFAULT_TRAIT = "thermal_leaf_ref_dt"
SCORES = {
    "diurnal_score_ref": "vs control bed",
    "diurnal_score_local": "no control bed",
    "diurnal_score_idso": "published line",
}
SCORE_HOVER = {
    "diurnal_score_ref": "Leaf minus control-bed leaf, z against the plant's own baseline at the same hour",
    "diurnal_score_local": "Leaf minus air above a healthy-plant VPD line, z against the plant's own baseline",
    "diurnal_score_idso": "Leaf minus air above Idso's (1982) sugar beet line, 2.50 − 1.92 × VPD",
}
PLAN_RULES = ("control_ref", "local_line", "idso", "leaf_air")  # "demand" knows irrigation stopped: not a detector
METRICS = {"afternoon_minus_morning": "15 h − 09 h", "daily_range": "daily range", "overnight_change": "overnight"}
METRIC_HOVER = {
    "afternoon_minus_morning": "afternoon minus morning run, per plant and day",
    "daily_range": "highest minus lowest run of the day (at least 4 runs)",
    "overnight_change": "next morning's 06 h minus this evening's 18 h",
}
# AUC bins on the diverging value ramp, 0.5 = gray (DESIGN 2.2)
AUC_BINS = [
    (0.30, C["teal-600"], "< 0.3"),
    (0.45, C["teal-300"], "0.3–0.45"),
    (0.55, C["gray"], "0.45–0.55"),
    (0.70, C["amber-200"], "0.55–0.7"),
    (0.90, C["amber-700"], "0.7–0.9"),
    (9.99, C["amber-900"], "≥ 0.9"),
]


def auc_fill(v):
    return next(c for hi, c, _ in AUC_BINS if v < hi)


def first_separation(row):
    """First day ≥ 0 where AUC ≥ 0.9 on that day and the next, else None."""
    for d in sorted(x for x in row.index if x >= 0):
        if row.get(d, 0) >= 0.9 and row.get(d + 1, 0) >= 0.9:
            return d
    return None


def heatmap(sep):
    days = list(range(int(sep.columns.min()), int(sep.columns.max()) + 1))
    head = html.Div(
        [
            html.Div("hour", className="uw-heat-head", style={"flex": "0 0 56px", "textAlign": "left"}),
            *[
                html.Div(
                    f"{d}".replace("-", "−"), className="uw-heat-head", style={"fontWeight": 700 if d == 0 else 400}
                )
                for d in days
            ],
            html.Div("separates", className="uw-heat-head", style={"flex": "0 0 76px", "textAlign": "right"}),
        ],
        className="uw-heat-row",
    )
    rows = [head]
    for h in SLOTS:
        r = sep.loc[h].dropna() if h in sep.index else pd.Series(dtype=float)
        first = first_separation(r)
        cells = []
        for d in days:
            v = r.get(d)
            day, when = day_label(d)
            if v is None:
                cells.append(
                    html.Div(className="uw-heat-cell uw-missing", title=f"{h:02d} h · {day} · {when}: no scans")
                )
                continue
            ring = " uw-heat-first" if first is not None and d in (first, first + 1) else ""
            cells.append(
                html.Div(
                    className="uw-heat-cell" + ring,
                    style={"background": auc_fill(v)},
                    title=f"{h:02d} h · {day} · {when}: AUC {v:.2f}",
                )
            )
        rows.append(
            html.Div(
                [
                    html.Div(f"{h:02d}", className="uw-heat-key"),
                    *cells,
                    html.Div(
                        "never" if first is None else f"day {first}",
                        className="uw-heat-end",
                        style={"color": C["ink-muted"]} if first is None else None,
                    ),
                ],
                className="uw-heat-row",
            )
        )
    band = {d: C["mint"] if -4 <= d <= -1 else C["sand"] if 0 <= d <= 12 else "transparent" for d in days}
    rows.append(
        html.Div(
            [
                html.Div(style={"flex": "0 0 56px"}),
                *[html.Div(className="uw-heat-band", style={"background": band[d]}) for d in days],
                html.Div(style={"flex": "0 0 76px"}),
            ],
            className="uw-heat-row",
            title="baseline 30 Aug – 2 Sep (mint) · no water 3 – 15 Sep (sand)",
        )
    )
    ring = {"outline": f"2px solid {C['navy']}", "outlineOffset": "-2px"}
    legend = html.Div(
        [
            html.Span("AUC"),
            *[html.Span([html.Span(className="uw-swatch", style={"background": c}), t]) for _, c, t in AUC_BINS],
            html.Span([html.Span(className="uw-swatch", style=ring), "≥ 0.9 two days"]),
            html.Span([html.Span(className="uw-swatch", style={"background": C["mint"]}), "baseline"]),
            html.Span([html.Span(className="uw-swatch", style={"background": C["sand"]}), "no water"]),
        ],
        className="uw-legend",
    )
    return [html.Div(html.Div(rows, className="uw-heat"), className="uw-scroll"), legend]


def bed_effect(sep):
    """Caveat chip when the placebo window (before irrigation off) already separates the beds at some hour."""
    pre = sep[[c for c in sep.columns if c < 0]].mean(axis=1)
    off = pre[(pre - 0.5).abs() >= 0.1]
    if off.empty:
        return []
    hours = " and ".join(f"{h:02d} h" for h in off.index)
    detail = ", ".join(f"{h:02d} h AUC {v:.2f}" for h, v in off.items())
    hover = (
        f"Before irrigation off the beds already differ ({detail}); part of the signal at these hours may be the bed"
    )
    return [chip(f"bed effect at {hours}", hover)]


def plan_cards(sch):
    w3 = sch[(sch.set == "W3 check") & sch.recommended & sch.rule.isin(PLAN_RULES)].set_index("rule")
    ok = w3[w3.false_alerts_controls == 0]
    best = ok.sort_values(["median_alert_day", "scans_per_day"]).index[0] if len(ok) else None
    cards = []
    for rule in PLAN_RULES:
        if rule not in w3.index:
            continue
        r = w3.loc[rule]
        tint = "sky" if rule == best else "mint" if r.false_alerts_controls == 0 else "gray"
        title = [html.B(RULE_SHORT[rule])] + (
            [html.Span("best", className="uw-chip uw-chip-best")] if rule == best else []
        )
        false = html.Div(
            [
                html.Div("false", className="uw-readout-label"),
                html.Div(
                    f"{r.false_alerts_controls} / {r.n_controls}",
                    className="uw-section",
                    style={"color": C["amber-900"]} if r.false_alerts_controls else None,
                ),
            ]
        )
        cards.append(
            html.Div(
                [
                    html.Div(title, style={"display": "flex", "gap": "8px", "alignItems": "center"}),
                    html.Div(
                        [
                            readout("flyover", r.schedule.replace("all five", "all 5"), size="section"),
                            readout(
                                "W3 alert",
                                "–" if pd.isna(r.median_alert_day) else f"day {r.median_alert_day:g}",
                                size="section",
                            ),
                            false,
                        ],
                        className="uw-readouts uw-readouts-3",
                    ),
                ],
                className=f"uw-card uw-{tint}",
                title=(
                    f"Threshold {r.threshold:.2f}, set on W2 with 4 controls; W3 and the other 9 controls held out. "
                    f"false: held-out watered plants flagged. Before irrigation off: "
                    f"{r.false_alerts_placebo} / {r.n_placebo_plants} plants flagged."
                ),
            )
        )
    return cards


def sky_card():
    sky = read_table("diurnal", "sky_effect")
    if sky is None:
        return []
    sky = sky.dropna(subset=["sunny_extra"])
    cells = [
        html.Div(
            [
                html.Div(f"{r.hour:02d} h", className="uw-readout-label"),
                html.Div(signed(r.sunny_extra, "°C"), className="uw-section"),
            ],
            title=(
                f"Stressed minus watered leaf, stress days 2–12: {r.gap_sunny:.1f} °C on {r.n_sunny} sunny runs, "
                f"{r.gap_cloudy:.1f} °C on {r.n_cloudy} cloudy runs; shown: the sunny extra on the same day. "
                "Sky from DWD station Belm (11 km). Few runs: a direction, not a precise size."
            ),
        )
        for r in sky.itertuples()
    ]
    return [
        html.Div(
            [
                html.Div("Sun vs cloud", className="uw-kicker", style={"color": C["slate-600"]}),
                html.Div(
                    cells,
                    className="uw-readouts",
                    style={"gridTemplateColumns": "repeat(2, minmax(0, 1fr))"},
                ),
            ],
            className="uw-card uw-gray",
        )
    ]


def schedule_table(sch, rule):
    t = sch[sch.rule == rule]
    w2, w3 = (t[t.set == s].set_index("schedule").reindex(list(SCHEDULES)) for s in ("W2 tuning", "W3 check"))
    day = lambda v: "–" if pd.isna(v) else f"{v:g}"
    frac = lambda a, b: f"{int(a)} / {int(b)}"
    cols = [
        ("Flyovers", "", "flyovers used per day"),
        ("Per day", "num", ""),
        ("Threshold", "num", "set on W2 with 4 controls"),
        ("W2 day", "num", "median alert day, W2 (tuning)"),
        ("W3 day", "num", "median alert day, W3 (held out)"),
        ("W3 flagged", "num", "W3 plants flagged"),
        ("vs clip", "num", "days the alert comes before the leaf clip's turgor shift (W3 median)"),
        ("False · ctrl", "num", "held-out watered plants flagged"),
        ("False · pre", "num", "plants flagged before irrigation off"),
    ]
    head = html.Tr([html.Th(n, className=c, title=h) for n, c, h in cols])
    body = []
    for name in SCHEDULES:
        a, b = w2.loc[name], w3.loc[name]
        body.append(
            html.Tr(
                [
                    html.Td(name),
                    html.Td(f"{int(a.scans_per_day)}", className="num"),
                    html.Td(f"{a.threshold:.2f}", className="num"),
                    html.Td(day(a.median_alert_day), className="num"),
                    html.Td(day(b.median_alert_day), className="num"),
                    html.Td(frac(b.n_detected, b.n_plants), className="num"),
                    html.Td(signed(b.median_lead_vs_clip, "d", 0), className="num"),
                    html.Td(frac(b.false_alerts_controls, b.n_controls), className="num"),
                    html.Td(frac(b.false_alerts_placebo, b.n_placebo_plants), className="num"),
                ],
                className="uw-on" if a.recommended else None,
                title="recommended for this rule" if a.recommended else None,
            )
        )
    return html.Div(html.Table([html.Thead(head), html.Tbody(body)], className="uw-table"), className="uw-scroll")


def layout(**_):
    df = load_traits()
    head = header(
        "Daily rhythm",
        "How does a stressed plant's day differ from a watered one's, and when should you image to see it first?",
        segmented("dr-score", SCORES, "diurnal_score_ref", "Score"),
    )
    if df is None or "thermal_leaf_t" not in df:
        return html.Div([head, not_available("Thermal")])
    sep, sch = read_table("diurnal", "separability"), read_table("diurnal", "schedules")
    if sep is None or sch is None:
        return html.Div([head, not_available("Daily rhythm")])
    traits = {o["value"]: o["label"] for o in trait_options(df.columns)}
    top = html.Div(
        [
            html.Main(
                panel(
                    "Separability",
                    html.Div(id="dr-heat"),
                    hover="AUC of the score, deficit against control plants (W2 + W3 vs W1), per day and flyover: "
                    "0.5 = no difference, 1 = every stressed plant above every watered one",
                    extra=[html.Div(id="dr-chips", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"})],
                ),
                className="uw-main",
            ),
            html.Aside(
                [
                    *plan_cards(sch),
                    *sky_card(),
                    html.A("Irrigation advisor", href="/irrigation", className="uw-btn uw-btn-primary"),
                ],
                className="uw-side",
                **{"aria-label": "Flyover plans"},
            ),
        ],
        className="uw-row",
    )
    profile = panel(
        "Day profile",
        graph("dr-profile", "Mean of the chosen trait at each flyover hour, by group and stress phase", 360),
        graph("dr-rhythm", "The chosen daily metric of the trait over days since irrigation off, by group", 320),
        hover="Mean over plants at each scheduled run; bars: standard error",
        extra=[
            html.Div(className="uw-spacer"),
            select("dr-trait", traits, DEFAULT_TRAIT, "Trait"),
            segmented("dr-group", {"treatment": "Treatment", "bed": "Bed"}, "treatment", "Group by"),
            select("dr-metric", METRICS, "afternoon_minus_morning", "Daily"),
        ],
    )
    schedules = panel(
        "Flyover schedules",
        html.Div(id="dr-schedules"),
        hover="Alert: score above the threshold on two usable scans in a row. Thresholds are set on W2 and its 4 "
        "tuning controls; W3 and the other 9 controls are the honest check.",
        extra=[html.Div(className="uw-spacer"), select("dr-rule", RULE_SHORT, "control_ref", "Rule")],
    )
    return html.Div([head, top, html.Div([profile, schedules], className="uw-main", style={"paddingTop": "24px"})])


@callback(Output("dr-heat", "children"), Output("dr-chips", "children"), Input("dr-score", "value"))
def update_separability(score):
    s = read_table("diurnal", "separability")
    s = s[(s.score == score) & (s.group == "W2 + W3")].pivot(index="hour", columns="day", values="auc")
    return heatmap(s), [*bed_effect(s), chip(SCORES[score], SCORE_HOVER[score], "rule")]


@callback(
    Output("dr-profile", "figure"),
    Output("dr-rhythm", "figure"),
    Input("dr-trait", "value"),
    Input("dr-group", "value"),
    Input("dr-metric", "value"),
)
def update_rhythm(trait, group, metric):
    df = load_traits().dropna(subset=[trait])
    unit = unit_of(trait)
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
        labels={"hour": "h", "mean": unit},
    )
    prof.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    prof.update_traces(line_width=2.5, marker_size=8, hovertemplate=f"%{{x}} h<br>%{{y:.2f}} {unit}")
    prof.update_xaxes(tickvals=[6, 12, 18])
    prof.update_layout(height=360, margin=dict(t=80), legend=dict(y=1.14, title_text=""))

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
        labels={"days_since_cutoff": "day", "mean": unit},
    )
    rhy.update_traces(
        line_width=2.5, marker_size=7, hovertemplate=f"day %{{x}}<br>%{{y:.2f}} {unit}<br>{METRIC_HOVER[metric]}"
    )
    rhy.update_layout(height=320, legend_title_text="")
    return prof, mark_stress_period(rhy)


@callback(Output("dr-schedules", "children"), Input("dr-rule", "value"))
def update_schedules(rule):
    return schedule_table(read_table("diurnal", "schedules"), rule)
