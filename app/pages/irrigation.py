"""Irrigation advisor: which plants need water today, why, and how far to trust the call."""

from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components import (
    RULE_SHORT,
    TREATMENT,
    C,
    chip,
    day_label,
    graph,
    header,
    mark,
    mark_stress_period,
    not_available,
    panel,
    readout,
    select,
    signed,
    size_chip,
    unit_of,
)
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from uiwue.core import load_traits, read_table
from uiwue.diurnal import (
    PLACEBO_DAYS,
    PRE_DAYS,
    PRE_MAX_DAY,
    PRE_MAX_RATE,
    REWATER_DAY,
    RULES,
    SCHEDULES,
    best_rule,
    daily_status,
    pre_check,
    status,
)

dash.register_page(
    __name__,
    path="/irrigation",
    name="Irrigation advisor",
    order=60,
    description="Which plants need water today, why, and how far can the call be trusted?",
)

FALLBACK_RULE = "control_ref"
ORDER = {"irrigate": 0, "watch": 1, "fine": 2, "no scan": 3}
TILES = (("irrigate", "sand"), ("watch", "sand"), ("fine", "sky"), ("no scan", "gray"))
FILL = {"irrigate": C["amber-900"], "watch": C["amber-700"], "fine": C["teal-700"]}
SYMBOL = {  # Plotly marks, DESIGN 3
    "fine": dict(symbol="circle", size=10, color=C["teal-200"], line=dict(color=C["teal-700"], width=2)),
    "watch": dict(symbol="triangle-up", size=12, color=C["amber-700"]),
    "irrigate": dict(symbol="diamond", size=12, color=C["amber-900"]),
}
OUTLINED = {"border": f"1px solid {C['teal-700']}", "color": C["teal-700"], "background": C["surface"]}


def schedules_table():
    return read_table("diurnal", "schedules")


def recommended(rule):
    sch = schedules_table()
    return sch[(sch.rule == rule) & sch.recommended].schedule.iloc[0]


@lru_cache
def scans(rule, schedule):
    """Every run of every thermal plant with its scan status under (rule, schedule), plus the threshold."""
    sch = schedules_table()
    k = float(sch[(sch.rule == rule) & (sch.schedule == schedule)].threshold.iloc[0])
    df = load_traits()
    df = df[df.thermal_leaf_t.notna()].reset_index(drop=True)
    col = RULES[rule][0]
    st = status(df, col, SCHEDULES[schedule], k)
    return df.assign(status=st, value=df[col], pid=df.bed + "_" + df.plant), k


@lru_cache
def days_of(rule, schedule):
    """Latched advice per plant and day, with the latest usable scan up to that day."""
    s, _ = scans(rule, schedule)
    ds = daily_status(s, s.status)
    ds["pid"] = ds.bed + "_" + ds.plant
    used = s[s.status.notna()].sort_values(["days_since_cutoff", "hour"])
    last = used[["pid", "days_since_cutoff", "hour", "value", "thermal_leaf_ref_dt", "env_sky"]].rename(
        columns={"days_since_cutoff": "scan_day", "hour": "scan_hour"}
    )
    ds = pd.merge_asof(
        ds.sort_values("day"), last.sort_values("scan_day"), left_on="day", right_on="scan_day", by="pid"
    )
    return ds.sort_values(["pid", "day"], ignore_index=True)


@lru_cache
def track(rule, schedule):
    """Pooled record over all 39 plants: first-alert days, control and watered-day false alerts."""
    s, k = scans(rule, schedule)
    ds = days_of(rule, schedule)
    stress = ds[ds.day.between(0, REWATER_DAY) & (ds.raw == "irrigate")]
    first = stress.groupby("pid").day.min()
    deficit = first[~first.index.str.startswith("W1")]
    n_deficit = ds[ds.bed != "W1"].pid.nunique()
    pre = ds[ds.day.between(*PRE_DAYS) & (ds.raw == "irrigate")].pid.nunique()
    return dict(
        median_first=deficit.median() if len(deficit) else np.nan,
        n_alerted=len(deficit),
        n_deficit=n_deficit,
        by_day={d: int((deficit <= d).sum()) for d in (3, 5, 7)},
        false_ctrl=int(first.index.str.startswith("W1").sum()),
        n_ctrl=ds[ds.bed == "W1"].pid.nunique(),
        false_pre=pre,
        n_plants=ds.pid.nunique(),
        check=pre_check(s, RULES[rule][0], SCHEDULES[schedule], k),
    )


def day_plants(rule, schedule, day):
    """One row per plant on `day`: latched status, raw status, since, latest usable scan."""
    ds = days_of(rule, schedule)
    m = ds[ds.day == day].copy()
    m["order"] = m.status.map(ORDER)
    return m.sort_values(["order", "value", "pid"], ascending=[True, False, True], na_position="last")


def water_now(m):
    """The plants to water today, as a table: plant, bed, days flagged, latest leaf − control bed."""
    w = m[m.status == "irrigate"].copy()
    w["days_flagged"] = (w.day - w.since + 1).astype(int)
    return w.sort_values(["days_flagged", "value"], ascending=False)


def latest_day(rule, schedule):
    ds = days_of(rule, schedule)
    return int(ds[ds.scanned].day.max())


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


def rule_options():
    sch = schedules_table()
    rec = sch[sch.recommended & (sch.set == "W2 tuning")]
    return {r.rule: f"{RULE_SHORT[r.rule]} · {r.schedule.replace('all five', 'all 5')}" for r in rec.itertuples()}


def layout(**_):
    df, sch = load_traits(), schedules_table()
    stepper = html.Div(
        [
            html.Button("‹", id="ir-prev", className="uw-icon-btn", **{"aria-label": "Previous day"}),
            html.Span(id="ir-day-label", className="uw-day", **{"aria-live": "polite"}),
            html.Button("›", id="ir-next", className="uw-icon-btn", **{"aria-label": "Next day"}),
        ],
        className="uw-stepper",
    )
    need = {"pre_pass", "recommended"}
    if df is None or sch is None or not need <= set(sch.columns) or "diurnal_score_ref" not in df:
        return html.Div([header("Irrigation advisor", "Which plants need water today?"), not_available("Daily rhythm")])
    rules = rule_options()
    rule = best_rule(sch) or FALLBACK_RULE
    days = sorted(int(d) for d in df.loc[df.thermal_leaf_t.notna(), "days_since_cutoff"].unique())
    head = header(
        "Irrigation advisor",
        "Which plants need water today, why, and how far can the call be trusted?",
        select("ir-rule", rules, rule, "Rule"),
        stepper,
    )
    return html.Div(
        [
            head,
            dcc.Store(id="ir-day", data=latest_day(rule, recommended(rule))),
            dcc.Store(id="ir-days", data=days),
            dcc.Store(id="ir-sel"),
            dcc.Download(id="ir-download"),
            html.Div(
                [
                    html.Main(
                        [
                            html.Div(id="ir-tiles", className="uw-tiles"),
                            html.Div(id="ir-plants", className="uw-plants"),
                            html.Div(id="ir-legend", className="uw-legend"),
                            panel(
                                "Plant",
                                graph("ir-trace", "Alert score of the selected plant by day since irrigation off", 300),
                                hover="Score at each usable flyover of the schedule; mark = that scan's status; "
                                "dashed line = threshold",
                                extra=[html.Div(id="ir-trace-head", style={"display": "contents"})],
                            ),
                        ],
                        className="uw-main",
                    ),
                    html.Aside(id="ir-side", className="uw-side", **{"aria-label": "Water now and track record"}),
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
    Output("ir-sel", "data"),
    Input({"type": "ir-plant", "pid": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def pick(clicks):
    """The plant whose card was clicked (re-rendered cards come back with no clicks: ignored)."""
    if not ctx.triggered_id or not any(clicks or []):
        raise PreventUpdate
    trig = ctx.triggered[0]["value"] if ctx.triggered else None
    if not trig:
        raise PreventUpdate
    return ctx.triggered_id["pid"]


def plant_card(r, k, unit, schedule, d, when, selected, pos):
    """One plant: status mark, ID with "since day N", its latest score against the threshold."""
    scanned_today = bool(r.scanned)
    has_value = not pd.isna(r.value)
    since = f"since day {int(r.since)}" if r.status == "irrigate" and not pd.isna(r.since) else ""
    if has_value:
        seen = "today" if int(r.scan_day) == r.day else f"day {int(r.scan_day)}"
        scan = f"score {signed(r.value, unit, 2)} at {int(r.scan_hour):02d} h {seen}"
    else:
        scan = "no usable scan yet"
    why = {
        "irrigate": f"above {k:.2f} {unit} on two usable scans in a row; stays until re-watering (day {REWATER_DAY})",
        "watch": f"above {k:.2f} {unit} once",
        "fine": f"at or below {k:.2f} {unit}",
        "no scan": f"no usable scan today (no flyover at {schedule}, or weak evidence)",
    }[r.status]
    extra = "" if scanned_today or r.status == "no scan" else " · no scan today"
    hover = f"{r.bed} {r.plant} · {d} · {when}: {r.status} {since}{extra} · {why} · {scan}".replace("  ", " ")
    return html.Button(
        [
            mark(r.status),
            html.Span(
                [
                    html.Span(f"{r.bed} {r.plant}", className="uw-plant-id", style={"width": "auto"}),
                    html.Small(since, style={"margin": 0}) if since else None,
                ],
                style={"display": "flex", "flexDirection": "column", "minWidth": "72px", "lineHeight": "16px"},
            ),
            html.Span(
                [
                    *(
                        [html.Span(style={"width": pos(r.value), "background": FILL.get(r.status, C["gray-500"])})]
                        if has_value
                        else []
                    ),
                    html.I(style={"left": pos(k)}),
                ],
                className="uw-bar",
            ),
            html.Span(signed(r.value, "", 1) if has_value else "–", className="uw-num"),
        ],
        id={"type": "ir-plant", "pid": r.pid},
        title=hover,
        className="uw-plant",
        n_clicks=0,
        **{"aria-pressed": "true" if selected else "false"},
        style={
            "font": "inherit",
            "textAlign": "left",
            "width": "100%",
            "cursor": "pointer",
            "background": C["sky"] if selected else C["surface"],
            **({"outline": f"2px solid {C['navy']}", "outlineOffset": "-2px"} if selected else {}),
        },
    )


def trace(rule, schedule, pid, day, k, unit):
    s, _ = scans(rule, schedule)
    p = s[(s.pid == pid) & s.hour.isin(SCHEDULES[schedule])].sort_values(["days_since_cutoff", "hour"])
    x = p.days_since_cutoff + (p.hour - 12) / 24  # flyovers spread within their day
    fig = go.Figure()
    treat = "control" if pid.startswith("W1") else "deficit"
    used = p.status.notna()
    fig.add_trace(
        go.Scatter(
            x=x[used],
            y=p.value[used],
            mode="lines",
            line=dict(color=TREATMENT[treat], width=2.5),
            name=treat,
            hoverinfo="skip",
        )
    )
    for name, sty in SYMBOL.items():
        q = p[p.status == name]
        if q.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=q.days_since_cutoff + (q.hour - 12) / 24,
                y=q.value,
                mode="markers",
                marker=sty,
                name=name,
                customdata=np.c_[q.days_since_cutoff, q.hour, q.date.dt.strftime("%d %b")],
                hovertemplate=f"day %{{customdata[0]}} · %{{customdata[2]}} · %{{customdata[1]}} h<br>"
                f"%{{y:.2f}} {unit}<br>{name}<extra></extra>",
            )
        )
    q = p[~used & p.value.notna()]
    if len(q):
        fig.add_trace(
            go.Scatter(
                x=q.days_since_cutoff + (q.hour - 12) / 24,
                y=q.value,
                mode="markers",
                name="weak",
                marker=dict(symbol="x", size=8, color=C["gray-500"]),
                hovertemplate=f"day %{{x:.0f}}<br>%{{y:.2f}} {unit}<br>weak evidence, not used<extra></extra>",
            )
        )
    fig.add_hline(y=k, line_dash="dash", line_width=1, line_color=C["ink-muted"])
    fig.add_vline(x=day, line_width=2, line_color=C["navy"])
    mark_stress_period(fig)
    fig.update_layout(height=300, xaxis_title="day", yaxis_title=unit, hovermode="closest")
    return fig


def readouts(t, unit):
    c = t["check"]
    rate = "–" if pd.isna(c["pre_above_rate"]) else f"{100 * c['pre_above_rate']:.1f}"
    by = " · ".join(f"day {d}: {n}" for d, n in t["by_day"].items())
    cells = [
        (
            readout("first alert", "–" if pd.isna(t["median_first"]) else f"day {t['median_first']:g}", " median"),
            (
                f"Median first alert day of the {t['n_alerted']} / {t['n_deficit']} deficit plants that alert "
                f"(days 0–{REWATER_DAY}); plants alerted by {by}. All plants, tuning set included."
            ),
        ),
        (
            readout("false · controls", f"{t['false_ctrl']} / {t['n_ctrl']}", " plants"),
            f"Watered W1 plants that reach irrigate during days 0–{REWATER_DAY}",
        ),
        (
            readout("false · before cutoff", f"{t['false_pre']} / {t['n_plants']}", " plants"),
            (
                f"Plants that reach irrigate on a watered day ({PRE_DAYS[0]} to {PRE_DAYS[1]}); depends on the "
                "weather sequence, so see the watered-scan rate too"
            ),
        ),
        (
            readout("above · watered", rate, " %"),
            (
                f"Watered-day check: {c['pre_above']} of {c['pre_scans']} usable scans before the cutoff above the "
                f"threshold; worst day {c['pre_worst_day']}: {100 * c['pre_worst_day_frac']:.0f} % of plants. "
                f"Passes at ≤ {100 * PRE_MAX_RATE:.0f} % and ≤ {100 * PRE_MAX_DAY:.0f} % on any day."
            ),
        ),
    ]
    return html.Div(
        [html.Div(cell, title=hover) for cell, hover in cells],
        className="uw-readouts",
        style={"gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))"},
    )


def water_table(w, unit):
    head = html.Tr(
        [
            html.Th("Plant"),
            html.Th("Bed"),
            html.Th("Days", className="num", title="days flagged, counting today"),
            html.Th("Leaf − ctrl", className="num", title="leaf minus control-bed leaf at the latest usable scan"),
        ]
    )
    rows = [
        html.Tr(
            [
                html.Td(r.plant),
                html.Td(r.bed),
                html.Td(f"{r.days_flagged} d", className="num", title=f"since day {int(r.since)}"),
                html.Td(
                    signed(r.thermal_leaf_ref_dt, "°C", 1),
                    className="num",
                    title=f"score {signed(r.value, unit, 2)} at {int(r.scan_hour):02d} h, day {int(r.scan_day)}"
                    if not pd.isna(r.value)
                    else "no usable scan",
                ),
            ]
        )
        for r in w.itertuples()
    ]
    return html.Div(html.Table([html.Thead(head), html.Tbody(rows)], className="uw-table"), className="uw-scroll")


@callback(
    Output("ir-day-label", "children"),
    Output("ir-tiles", "children"),
    Output("ir-plants", "children"),
    Output("ir-legend", "children"),
    Output("ir-side", "children"),
    Output("ir-trace", "figure"),
    Output("ir-trace-head", "children"),
    Input("ir-rule", "value"),
    Input("ir-day", "data"),
    Input("ir-sel", "data"),
)
def update(rule, day, sel):
    schedule = recommended(rule)
    _, k = scans(rule, schedule)
    unit = unit_of(RULES[rule][0])
    m = day_plants(rule, schedule, day)
    d, when = day_label(day)
    counts = m.status.value_counts()
    if sel not in set(m.pid):
        sel = m.pid.iloc[0]

    tiles = [
        html.Div(
            [
                mark(name),
                html.Div(
                    [
                        html.Div([f"{counts.get(name, 0)}", html.Small("plants")], className="uw-stat"),
                        html.Div(name, className="uw-readout-label"),
                    ]
                ),
            ],
            className=f"uw-tile uw-{tint}",
            style={"display": "flex", "alignItems": "center", "gap": "14px"},
            title={
                "irrigate": f"above threshold on two usable scans in a row, held until re-watering (day {REWATER_DAY})",
                "watch": "above threshold on the last usable scan, once",
                "fine": "at or below threshold today",
                "no scan": "no usable scan today and no standing alert",
            }[name],
        )
        for name, tint in TILES
    ]

    vals = m.value.dropna()
    lo = min(vals.min(), k, 0) if len(vals) else 0
    hi = max(vals.max(), k) if len(vals) else 1
    pos = lambda x: f"{np.clip((x - lo) / (hi - lo) * 100, 0, 98):.1f}%" if hi > lo else "0%"
    cards = [plant_card(r, k, unit, schedule, d, when, r.pid == sel, pos) for r in m.itertuples()]
    legend = [
        *[html.Span([mark(n), n]) for n, _ in TILES],
        html.Span(f"{unit} at {schedule}" if unit else schedule),
        html.Span(
            [html.Span(style={"width": "2px", "height": "12px", "background": C["navy"]}), f"threshold {k:.2f} {unit}"]
        ),
    ]

    sch = schedules_table()
    row = {r.set: r for r in sch[(sch.rule == rule) & (sch.schedule == schedule)].itertuples()}
    w2, w3 = row["W2 tuning"], row["W3 check"]
    day_of = lambda v: "–" if pd.isna(v) else f"{v:g}"
    w = water_now(m)
    r = m[m.pid == sel].iloc[0]
    since = f"since day {int(r.since)}" if r.status == "irrigate" and not pd.isna(r.since) else r.status
    selected = html.Div(
        [
            html.Div(
                [mark(r.status), html.Span(f"{r.bed} {r.plant}", className="uw-section"), html.Small(since)],
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
            ),
            html.A(
                "Profile",
                href=f"/plant-profile?bed={r.bed}&plant={r.plant}",
                className="uw-btn uw-btn-primary",
                title=f"Every sensor for {r.bed} {r.plant}",
            ),
        ],
        className="uw-card uw-sky" if r.bed == "W1" else "uw-card uw-sand",
        **{"aria-label": "Selected plant"},
    )
    t = track(rule, schedule)
    water = panel(
        "Water now",
        water_table(w, unit)
        if len(w)
        else html.Div("–", title="no plant is flagged on this day", className="uw-section"),
        hover=f"Plants flagged irrigate on {d} ({when}), latched until re-watering",
        extra=[
            html.Span(f"{len(w)} plants", className="uw-readout-label"),
            html.Div(className="uw-spacer"),
            html.Button(
                "CSV",
                id="ir-csv",
                className="uw-btn",
                style=OUTLINED,
                disabled=not len(w),
                title="Download the water-now list",
                **{"aria-label": "Download water-now list as CSV"},
            ),
        ],
    )
    check_chip = (
        []
        if w2.pre_pass
        else [
            chip(
                "fails watered days",
                f"Before the cutoff, with every plant watered, {100 * w2.pre_above_rate:.0f} % of scans are above the "
                f"threshold and {100 * w2.pre_worst_day_frac:.0f} % of plants on day {int(w2.pre_worst_day)}",
            )
        ]
    )
    side = [
        selected,
        water,
        html.Div(
            [
                html.Div("W3 check · not tuned", className="uw-kicker", style={"color": C["forest-700"]}),
                html.Div(
                    [
                        readout("flagged", f"{w3.n_detected} / {w3.n_plants}", " plants"),
                        readout(
                            "median · latest", f"day {day_of(w3.median_alert_day)} · {day_of(w3.latest_alert_day)}"
                        ),
                        readout("false · controls", f"{w3.false_alerts_controls} / {w3.n_controls}", " plants"),
                        html.Div(
                            readout("vs leaf clip", signed(w3.median_lead_vs_clip, "d", 0)),
                            title="median days the alert comes before the leaf clip's turgor shift (W3 clip plants)",
                        ),
                    ],
                    className="uw-readouts",
                ),
            ],
            className="uw-card uw-mint",
            title=f"W3 and 9 of the 13 control plants were never used to set the threshold. W2 tuning: "
            f"{w2.n_detected} / {w2.n_plants} flagged, median day {day_of(w2.median_alert_day)}, latest "
            f"{day_of(w2.latest_alert_day)}",
        ),
        html.Div(
            [
                *bed_effect(rule, schedule),
                *check_chip,
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
                    f"Needs {RULES[rule][2]}. Threshold {k:.2f} {unit}: the lowest with no alert on W2's 4 tuning "
                    f"controls or before irrigation off. Irrigate: above it on two usable scans in a row, held until "
                    f"re-watering (day {REWATER_DAY}); watch: once.",
                    "rule",
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
        ),
    ]
    tiles_row = [*tiles, html.Div(readouts(t, unit), style={"gridColumn": "1 / -1"})]
    trace_head = [html.Span(f"{r.bed} {r.plant}", className="uw-readout-label")]
    return (
        [html.B(d), html.Span(when)],
        tiles_row,
        cards,
        legend,
        side,
        trace(rule, schedule, sel, day, k, unit),
        trace_head,
    )


@callback(
    Output("ir-download", "data"),
    Input("ir-csv", "n_clicks"),
    State("ir-rule", "value"),
    State("ir-day", "data"),
    prevent_initial_call=True,
)
def download(n, rule, day):
    if not n:
        raise PreventUpdate
    schedule = recommended(rule)
    w = water_now(day_plants(rule, schedule, day))
    unit = unit_of(RULES[rule][0])
    out = pd.DataFrame(
        {
            "bed": w.bed,
            "plant": w.plant,
            "day_since_cutoff": day,
            "flagged_since_day": w.since.astype(int),
            "days_flagged": w.days_flagged,
            "latest_scan_day": w.scan_day,
            "latest_scan_hour": w.scan_hour,
            "leaf_minus_control_bed_C": w.thermal_leaf_ref_dt.round(2),
            f"score_{unit or 'value'}": w.value.round(2),
            "rule": rule,
            "schedule": schedule,
        }
    )
    return dcc.send_data_frame(out.to_csv, f"water_now_day{day}_{rule}.csv", index=False)
