"""Irrigation advisor: which plants need water, why, and how far to trust the call."""

from functools import lru_cache

import dash
import numpy as np
import pandas as pd
from components import (
    RULE_SHORT,
    C,
    chip,
    day_label,
    header,
    mark,
    not_available,
    readout,
    select,
    signed,
    size_chip,
    unit_of,
)
from dash import Input, Output, State, callback, ctx, dcc, html

from uiwue.core import load_traits, read_table
from uiwue.diurnal import PLACEBO_DAYS, RULES, SCHEDULES, status

dash.register_page(
    __name__,
    path="/irrigation",
    name="Irrigation advisor",
    order=60,
    description="which plants need water on a given day, under five alert rules, with their track record",
)

DEFAULT_RULE, DEFAULT_DAY = "control_ref", 4
TILES = (("irrigate", "sand"), ("watch", "sand"), ("fine", "sky"), ("no scan", "gray"))


def recommended(rule):
    sch = read_table("diurnal", "schedules")
    return sch[(sch.rule == rule) & sch.recommended].schedule.iloc[0]


@lru_cache
def scans(rule, schedule):
    """Every run of every thermal plant with its alert status under (rule, schedule), plus the threshold."""
    sch = read_table("diurnal", "schedules")
    k = sch[(sch.rule == rule) & (sch.schedule == schedule)].threshold.iloc[0]
    df = load_traits()
    df = df[df.thermal_leaf_t.notna()]
    col = RULES[rule][0]
    return df.assign(status=status(df, col, SCHEDULES[schedule], k), value=df[col], pid=df.bed + "_" + df.plant), k


def day_plants(s, day):
    """One row per plant: its status at the last usable scan of the day, or 'no scan'."""
    today = s[(s.days_since_cutoff == day) & s.status.notna()].sort_values("hour").groupby("pid").tail(1)
    plants = s[["pid", "bed", "plant"]].drop_duplicates()
    m = plants.merge(today[["pid", "hour", "status", "value", "env_sky"]], on="pid", how="left")
    m["status"] = m.status.fillna("no scan")
    return m.sort_values(["value", "pid"], ascending=[False, True], na_position="last")


def bed_effect(rule, schedule):
    """Caveat chip when this rule's score already separated the beds before irrigation off at a flyover it uses."""
    sep = read_table("diurnal", "separability")
    col = RULES[rule][0]
    if sep is None or col not in set(sep.score):
        return []
    placebo = sep.day.between(*PLACEBO_DAYS)  # 24-29 Aug: every plant watered
    pre = sep[(sep.score == col) & (sep.group == "W2 + W3") & placebo].groupby("hour").auc.mean()
    off = pre[pre.index.isin(SCHEDULES[schedule]) & ((pre - 0.5).abs() >= 0.1)]
    if off.empty:
        return []
    detail = ", ".join(f"{h:02d} h AUC {v:.2f}" for h, v in off.items())
    hover = f"In the placebo window (24–29 Aug, all watered) the beds already differ ({detail}); part of the signal may be the bed"
    return [chip("bed effect at " + " and ".join(f"{h:02d} h" for h in off.index), hover)]


def layout(**_):
    df, sch = load_traits(), read_table("diurnal", "schedules")
    rec = sch[sch.recommended & (sch.set == "W2 tuning")] if sch is not None else None
    rules = (
        {r.rule: f"{RULE_SHORT[r.rule]} · {r.schedule.replace('all five', 'all 5')}" for r in rec.itertuples()}
        if rec is not None
        else {}
    )
    stepper = html.Div(
        [
            html.Button("‹", id="ir-prev", className="uw-icon-btn", **{"aria-label": "Previous day"}),
            html.Span(id="ir-day-label", className="uw-day", **{"aria-live": "polite"}),
            html.Button("›", id="ir-next", className="uw-icon-btn", **{"aria-label": "Next day"}),
        ],
        className="uw-stepper",
    )
    head = header(
        "Irrigation advisor",
        "Which plants need water today, why, and how far can the call be trusted?",
        select("ir-rule", rules, DEFAULT_RULE, "Rule") if rules else None,
        stepper,
    )
    if df is None or not rules or "diurnal_score_ref" not in df:
        return html.Div([head, not_available("Daily rhythm")])
    days = sorted(int(d) for d in df.loc[df.thermal_leaf_t.notna(), "days_since_cutoff"].unique())
    start = min(days, key=lambda d: abs(d - DEFAULT_DAY))
    return html.Div(
        [
            head,
            dcc.Store(id="ir-day", data=start),
            dcc.Store(id="ir-days", data=days),
            html.Div(
                [
                    html.Main(
                        [
                            html.Div(id="ir-tiles", className="uw-tiles"),
                            html.Div(id="ir-plants", className="uw-plants"),
                            html.Div(id="ir-legend", className="uw-legend"),
                        ],
                        className="uw-main",
                    ),
                    html.Aside(id="ir-side", className="uw-side", **{"aria-label": "Track record"}),
                ],
                className="uw-row",
            ),
        ]
    )


@callback(
    Output("ir-day", "data"),
    Input("ir-prev", "n_clicks"),
    Input("ir-next", "n_clicks"),
    State("ir-day", "data"),
    State("ir-days", "data"),
    prevent_initial_call=True,
)
def step(_prev, _next, day, days):
    """The previous or next day that has data, relative to the shown day; stays put at either end."""
    if ctx.triggered_id == "ir-prev":
        return max((d for d in days if d < day), default=day)
    return min((d for d in days if d > day), default=day)


@callback(
    Output("ir-day-label", "children"),
    Output("ir-tiles", "children"),
    Output("ir-plants", "children"),
    Output("ir-legend", "children"),
    Output("ir-side", "children"),
    Input("ir-rule", "value"),
    Input("ir-day", "data"),
)
def update(rule, day):
    schedule = recommended(rule)
    s, k = scans(rule, schedule)
    col = RULES[rule][0]
    unit = unit_of(col)
    m = day_plants(s, day)
    d, when = day_label(day)
    counts = m.status.value_counts()

    tiles = [
        html.Div(
            [
                mark(name),
                html.Div(
                    [
                        html.Div(f"{counts.get(name, 0)}", className="uw-stat"),
                        html.Div(name, className="uw-readout-label"),
                    ]
                ),
            ],
            className=f"uw-tile uw-{tint}",
            style={"display": "flex", "alignItems": "center", "gap": "14px"},
        )
        for name, tint in TILES
    ]

    vals = m.value.dropna()
    lo, hi = min(vals.min(), k, 0) if len(vals) else 0, max(vals.max(), k) if len(vals) else 1
    pos = lambda x: (
        f"{np.clip((x - lo) / (hi - lo) * 100, 0, 98):.1f}%" if hi > lo else "0%"
    )  # 98: marker stays in view
    fill = {"irrigate": C["amber-900"], "watch": C["amber-700"], "fine": C["teal-700"]}
    plants = []
    for r in m.itertuples():
        scanned = r.status != "no scan"
        hover = (
            f"{r.bed} {r.plant} · {d} · {when}: {RULE_SHORT[rule]} {signed(r.value, unit, 2)} at {r.hour:02.0f} h "
            f"({r.env_sky if isinstance(r.env_sky, str) else 'no sky record'}); threshold {k:.2f}; {r.status}"
            if scanned
            else f"{r.bed} {r.plant} · {d} · {when}: no usable scan (no flyover at {schedule}, or weak evidence)"
        )
        plants.append(
            html.A(
                [
                    mark(r.status),
                    html.Span(f"{r.bed} {r.plant}", className="uw-plant-id"),
                    html.Span(
                        [
                            *(
                                [html.Span(style={"width": pos(r.value), "background": fill[r.status]})]
                                if scanned
                                else []
                            ),
                            html.I(style={"left": pos(k)}),
                        ],
                        className="uw-bar",
                    ),
                    html.Span(signed(r.value, "", 1) if scanned else "–", className="uw-num"),
                ],
                href=f"/plant-profile?bed={r.bed}&plant={r.plant}",
                title=hover,
                className="uw-plant",
            )
        )
    legend = [
        html.Span(f"{unit} at {schedule}" if unit else schedule),
        html.Span([html.Span(style={"width": "2px", "height": "12px", "background": C["navy"]}), f"threshold {k:.2f}"]),
    ]

    sch = read_table("diurnal", "schedules")
    row = {r.set: r for r in sch[(sch.rule == rule) & (sch.schedule == schedule)].itertuples()}
    w2, w3 = row["W2 tuning"], row["W3 check"]
    day_of = lambda v: "–" if pd.isna(v) else f"{v:g}"
    sky = (
        s[(s.days_since_cutoff == day) & s.hour.isin(SCHEDULES[schedule])]
        .groupby("hour")[["env_sky", "env_sunshine"]]
        .first()
    )
    side = [
        html.Div(
            [
                html.Div("W3 check · not tuned", className="uw-kicker", style={"color": C["forest-700"]}),
                html.Div(
                    [
                        readout("flagged", f"{w3.n_detected} / {w3.n_plants}", size="readout"),
                        readout("median day", day_of(w3.median_alert_day), size="readout"),
                        readout("false · controls", f"{w3.false_alerts_controls} / {w3.n_controls}", size="readout"),
                        readout(
                            "false · before cutoff",
                            f"{w3.false_alerts_placebo} / {w3.n_placebo_plants}",
                            size="readout",
                        ),
                        html.Div(
                            readout("vs leaf clip", signed(w3.median_lead_vs_clip, "d", 0), size="readout"),
                            title="median days the alert comes before the leaf clip's turgor shift (W3 clip plants)",
                        ),
                    ],
                    className="uw-readouts",
                ),
            ],
            className="uw-card uw-mint",
            title="W3 and 9 of the 13 control plants were never used to set the threshold",
        ),
        html.Div(
            [
                html.Div("W2 tuning", className="uw-kicker", style={"color": C["slate-600"]}),
                html.Div(
                    [
                        readout("flagged", f"{w2.n_detected} / {w2.n_plants}", size="readout"),
                        readout(
                            "median · latest",
                            f"{day_of(w2.median_alert_day)} · {day_of(w2.latest_alert_day)}",
                            size="readout",
                        ),
                    ],
                    className="uw-readouts",
                ),
            ],
            className="uw-card uw-gray",
            title=f"threshold {k:.2f}: the lowest with no alert on W2's 4 tuning controls or before irrigation off",
        ),
        html.Div(
            [
                *[
                    chip(
                        f"{h:02d} h {v.env_sky}",
                        f"Sky at the {h:02d} h flyover, DWD station Belm (11 km): sun {v.env_sunshine:.0%} of the hour before",
                        "context",
                    )
                    for h, v in sky.dropna().iterrows()
                ],
                *bed_effect(rule, schedule),
                size_chip(),
                *(
                    [
                        chip(
                            "knows watering stopped",
                            "Demand since the last watering fires on day 1 because the schedule stopped; a reference, not a stress detector",
                        )
                    ]
                    if rule == "microclimate"
                    else []
                ),
                chip(
                    "rule",
                    f"Needs {RULES[rule][2]}. Irrigate: above {k:.2f} on two usable scans in a row; watch: once.",
                    "rule",
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
        ),
        html.A("Bed map", href="/bed-map", className="uw-btn uw-btn-primary"),
    ]
    return [html.B(d), html.Span(when)], tiles, plants, legend, side
