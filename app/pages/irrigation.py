"""Irrigation advisor: which plants need water on the chosen day, and how far to trust the trigger.

The answer comes first, in tiles: how many plants to water, then how far to trust the trigger (first alert, false
alarms, the held-out bed). Then every plant against the threshold, the plants to water first, and the selected
plant's scans. It opens on the last day without water (day 12, the
re-watering), or on ``?day=``; ``?bed=&plant=`` selects a plant, and the page writes its selection back into the
address bar and carries it on to the next pages.
"""

from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components import (
    BASELINE_DAYS,
    RULE_SHORT,
    TREATMENT,
    C,
    chip,
    day_label,
    graph,
    header,
    mark,
    mark_stress_period,
    next_steps,
    not_available,
    panel,
    plant_link,
    select,
    signed,
    size_chip,
    stat,
    unit_of,
)
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html
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

QUESTION = "Which plants need water today, and how far can the trigger be trusted?"
dash.register_page(
    __name__,
    path="/irrigation",
    name="Irrigation advisor",
    order=60,
    description=QUESTION,
)

FALLBACK_RULE = "control_ref"
ORDER = {"irrigate": 0, "watch": 1, "fine": 2, "no scan": 3}
STATUS_HOVER = {
    "irrigate": "above threshold on two usable scans in a row, held until re-watering (day {day})",
    "watch": "above threshold on the last usable scan, once",
    "fine": "at or below threshold",
    "no scan": "no usable scan this day and no standing alert",
}
NEXT = (  # the page's next steps, at its foot: (text, path, hover); they carry the selection
    ("Export", "/export", "the tables behind the advice, as CSV"),
    ("Methods & limits", "/methods", "how far the trigger can be trusted, and what it cannot tell you"),
)
FILL = {"irrigate": C["amber-900"], "watch": C["amber-700"], "fine": C["teal-700"]}
SYMBOL = {  # Plotly marks, DESIGN 3
    "fine": dict(symbol="circle", size=10, color=C["teal-200"], line=dict(color=C["teal-700"], width=2)),
    "watch": dict(symbol="triangle-up", size=12, color=C["amber-700"]),
    "irrigate": dict(symbol="diamond", size=12, color=C["amber-900"]),
}


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
    detail = ", ".join(f"{h:02d} h score {v:.2f}" for h, v in off.items())
    hover = f"The beds already differ on watered days 24–29 Aug ({detail}); part of the signal may be the bed"
    return [chip("bed effect at " + " and ".join(f"{h:02d} h" for h in off.index), hover, anchor="bed-effect")]


def rule_options():
    sch = schedules_table()
    rec = sch[sch.recommended & (sch.set == "W2 tuning")]
    return {r.rule: f"{RULE_SHORT[r.rule]} · {r.schedule.replace('all five', 'all 5')}" for r in rec.itertuples()}


def default_day(rule, schedule):
    """The day the advisor opens on: the last day without water (re-watering, day 12) when it has a usable scan,
    the decision a grower faces; else the latest day with one."""
    ds = days_of(rule, schedule)
    scanned = {int(d) for d in ds[ds.scanned].day}
    return REWATER_DAY if REWATER_DAY in scanned else latest_day(rule, schedule)


def asked_day(day, days):
    """The day of a link (``?day=``) when the advisor has data for it, else None."""
    try:
        d = int(float(day))
    except (TypeError, ValueError):
        return None
    return d if d in days else None


def next_of():
    """The page's next steps at its foot, (text, href, hover). Export and Methods take no plant, so no selection
    rides along (Export's ``bed`` would quietly narrow the download)."""
    return list(NEXT)


def layout(bed=None, plant=None, day=None, **_):
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
        return html.Div([header("Irrigation advisor", QUESTION), not_available("Daily rhythm")])
    rules = rule_options()
    rule = best_rule(sch) or FALLBACK_RULE
    days = sorted(int(d) for d in df.loc[df.thermal_leaf_t.notna(), "days_since_cutoff"].unique())
    start = asked_day(day, days)
    head = header(
        "Irrigation advisor",
        QUESTION,
        html.Span(id="ir-chips", className="ir-chips"),
        html.Div(select("ir-rule", rules, rule, "Rule"), id="ir-rule-box"),
        stepper,
    )
    return html.Div(
        [
            head,
            dcc.Store(id="ir-day", data=start if start is not None else default_day(rule, recommended(rule))),
            dcc.Store(id="ir-days", data=days),
            dcc.Store(id="ir-sel", data=f"{bed}_{plant}" if bed and plant else None),
            dcc.Store(id="ir-url"),
            dcc.Store(id="ir-url-done"),
            dcc.Download(id="ir-download"),
            html.Div(id="ir-tiles", className="uw-tiles uw-lead", **{"aria-label": "Water now and track record"}),
            html.Div(
                html.Div(id="ir-strip", className="ir-strip", role="group", **{"aria-label": "Day"}),
                className="uw-scroll ir-strip-box",
            ),
            html.Div(
                [
                    html.Main(
                        panel(
                            "All plants",
                            html.Div(id="ir-plants", className="uw-plants ir-grid"),
                            hover="Each plant's latest usable score against the threshold (the navy tick); the mark is its "
                            "advice for the day, the plants to water first. Click a plant to see its scans.",
                            extra=[
                                html.Div(id="ir-legend", className="uw-legend ir-counts"),
                                html.Button(
                                    "CSV",
                                    id="ir-csv",
                                    className="uw-btn uw-btn-outline",
                                    title="Download the plants to water on this day",
                                    **{"aria-label": "Download the water-now list as CSV"},
                                ),
                            ],
                        ),
                        className="uw-main ir-main",
                    ),
                    html.Aside(
                        panel(
                            "Plant",
                            graph("ir-trace", "Alert score of the selected plant by day since irrigation off", 300),
                            hover="Score at each usable flyover of the schedule; mark = that scan's status; "
                            "dashed line = threshold",
                            extra=[html.Div(id="ir-trace-head", className="ir-trace-head")],
                        ),
                        className="uw-side ir-side",
                    ),
                ],
                className="uw-row ir-row",
            ),
            next_steps(*next_of(), id="ir-more"),
        ]
    )


@callback(
    Output("ir-day", "data"),
    Input("ir-prev", "n_clicks"),
    Input("ir-next", "n_clicks"),
    Input({"type": "ir-daycell", "day": ALL}, "n_clicks"),
    State("ir-day", "data"),
    State("ir-days", "data"),
    prevent_initial_call=True,
)
def step(_prev, _next, _cells, day, days):
    """The day picked on the strip, or the previous or next day that has data; stays put at either end."""
    if isinstance(ctx.triggered_id, dict):  # a cell of the day strip (a re-drawn strip comes back with no clicks)
        if not ctx.triggered[0]["value"]:
            raise PreventUpdate
        return ctx.triggered_id["day"]
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


def day_strip(rule, schedule, day, days):
    """Day strip (DESIGN 5): one cell per day with data, baseline mint, no-water sand, the shown day navy; over
    each day a bar for the plants flagged irrigate that day, so the strip shows when the alerts arrive."""
    ds = days_of(rule, schedule)
    n = ds[ds.status == "irrigate"].groupby("day").size()
    top = max(int(n.max()) if len(n) else 0, 1)
    cells = []
    for d in days:
        name, when = day_label(d)
        k = int(n.get(d, 0))
        band = "uw-mint" if BASELINE_DAYS[0] < d < BASELINE_DAYS[1] else "uw-sand" if 0 <= d <= REWATER_DAY else ""
        on = d == day
        cells.append(
            html.Button(
                [
                    html.Span(
                        html.Span(style={"height": f"{100 * k / top:.0f}%"}) if k else None,
                        className="ir-cell-bar",
                        **{"aria-hidden": "true"},
                    ),
                    html.B(f"{d}".replace("-", "−")),
                    html.Span(when, className="ir-cell-date"),
                ],
                id={"type": "ir-daycell", "day": d},
                n_clicks=0,
                title=f"{name} · {when}: {k} plants to water",
                className=f"ir-cell {band} {'ir-cell-on' if on else ''}".strip(),
                **{"aria-pressed": "true" if on else "false", "aria-label": f"{name}, {when}: {k} plants to water"},
            )
        )
    return cells


@callback(
    Output("ir-strip", "children"),
    Input("ir-rule", "value"),
    Input("ir-day", "data"),
    State("ir-days", "data"),
)
def update_strip(rule, day, days):
    return day_strip(rule, recommended(rule), day, days)


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
    vals = p.value.dropna()
    if len(vals):  # the threshold line always inside the frame, never on its edge
        lo, hi = min(vals.min(), k), max(vals.max(), k)
        pad = 0.08 * (hi - lo or 1.0)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def tiles(w, d, when, t, w3, unit):
    """The answer first, in four tiles: how many plants to water on the day, then how far to trust the trigger
    (first alert with the size-only baseline beside it, false alarms on the controls, the held-out bed)."""
    n = len(w)
    c = t["check"]
    rate = "–" if pd.isna(c["pre_above_rate"]) else f"{100 * c['pre_above_rate']:.1f} %"
    by = " · ".join(f"day {d}: {k}" for d, k in t["by_day"].items())
    day_of = lambda v: "–" if pd.isna(v) else f"{v:g}"
    since = f"flagged since day {int(w.since.min())}–{int(w.since.max())}" if n else "no plant flagged"
    return [
        stat(
            "Water now",
            str(n),
            "plants",
            "sand" if n else "mint",
            f"{n} plants flagged irrigate on {d.lower()} ({when}): above the threshold on two usable scans in a row, held "
            f"until re-watering (day {REWATER_DAY}). They lead the list below.",
            sub=since,
            chips=[mark("irrigate" if n else "fine")],
        ),
        stat(
            "First alert",
            "–" if pd.isna(t["median_first"]) else f"day {t['median_first']:g}",
            "median",
            "gray",
            f"Median first alert of the {t['n_alerted']} / {t['n_deficit']} deficit plants that alert (days "
            f"0–{REWATER_DAY}); alerted by {by}.",
            sub=f"{t['n_alerted']} / {t['n_deficit']} deficit plants",
            chips=[size_chip()],
        ),
        stat(
            "False alarms",
            f"{t['false_ctrl']} / {t['n_ctrl']}",
            "controls",
            "sky",
            f"Watered W1 plants that reach irrigate on days 0–{REWATER_DAY}; before irrigation off, {t['false_pre']} "
            f"of {t['n_plants']} plants reach it. Before the cutoff, with every plant watered, {c['pre_above']} of "
            f"{c['pre_scans']} usable scans are above the threshold; worst day "
            f"{str(c['pre_worst_day']).replace('-', '−')}: {100 * c['pre_worst_day_frac']:.0f} % of plants. Passes at "
            f"≤ {100 * PRE_MAX_RATE:.0f} % and ≤ {100 * PRE_MAX_DAY:.0f} % on any day.",
            sub=f"watered scans {rate} above",
        ),
        stat(
            "Held-out W3",
            f"{w3.n_detected} / {w3.n_plants}",
            "flagged",
            "gray",
            f"W3 and 9 of the 13 control plants were never used to set the threshold: median alert day "
            f"{day_of(w3.median_alert_day)}, latest {day_of(w3.latest_alert_day)}, {w3.false_alerts_controls} / "
            f"{w3.n_controls} controls flagged, {signed(w3.median_lead_vs_clip, 'days', 0)} against the leaf clip.",
            sub=f"median day {day_of(w3.median_alert_day)}",
        ),
    ]


def rule_hover(rule, schedule, k, unit):
    """The rule's definition, on the hover of its picker."""
    return (
        f"Needs {RULES[rule][2]}. Threshold {k:.2f} {unit}: the lowest with no alert on W2's 4 tuning controls or "
        f"before irrigation off. Irrigate: above it on two usable scans in a row at {schedule}, held until "
        f"re-watering (day {REWATER_DAY}); watch: once."
    )


def header_chips(rule, schedule, w2):
    """The rule's caveats, in the header: a bed effect at a flyover it uses, a failed watered-day check, a rule
    that knows when watering stopped."""
    return [
        *bed_effect(rule, schedule),
        *(
            []
            if w2.pre_pass
            else [
                chip(
                    "fails watered days",
                    f"Before the cutoff, with every plant watered, {100 * w2.pre_above_rate:.0f} % of scans are above "
                    f"the threshold and {100 * w2.pre_worst_day_frac:.0f} % of plants on day {int(w2.pre_worst_day)}",
                    anchor="watered-check",
                )
            ]
        ),
        *(
            [
                chip(
                    "knows watering stopped",
                    "Demand since the last watering fires on day 1: a reference, not a detector",
                )
            ]
            if rule == "microclimate"
            else []
        ),
    ]


@callback(
    Output("ir-day-label", "children"),
    Output("ir-tiles", "children"),
    Output("ir-chips", "children"),
    Output("ir-rule-box", "title"),
    Output("ir-plants", "children"),
    Output("ir-legend", "children"),
    Output("ir-trace", "figure"),
    Output("ir-trace-head", "children"),
    Output("ir-csv", "disabled"),
    Output("ir-url", "data"),
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

    vals = m.value.dropna()
    lo = min(vals.min(), k, 0) if len(vals) else 0
    hi = max(vals.max(), k) if len(vals) else 1
    pos = lambda x: f"{np.clip((x - lo) / (hi - lo) * 100, 0, 98):.1f}%" if hi > lo else "0%"
    cards = [plant_card(r, k, unit, schedule, d, when, r.pid == sel, pos) for r in m.itertuples()]
    legend = [
        *[
            html.Span([mark(n), html.B(f"{counts.get(n, 0)}"), n], title=STATUS_HOVER[n].format(day=REWATER_DAY))
            for n in ORDER
        ],
        html.Span(
            [html.Span(className="ir-tick", **{"aria-hidden": "true"}), f"threshold {k:.2f} {unit}"],
            title=f"The navy tick in each bar: the score ({unit}) above which a scan at {schedule} counts",
        ),
    ]

    sch = schedules_table()
    row = {r.set: r for r in sch[(sch.rule == rule) & (sch.schedule == schedule)].itertuples()}
    w2, w3 = row["W2 tuning"], row["W3 check"]
    w = water_now(m)
    r = m[m.pid == sel].iloc[0]
    since = f"since day {int(r.since)}" if r.status == "irrigate" and not pd.isna(r.since) else r.status
    trace_head = [
        mark(r.status),
        plant_link(r.bed, r.plant, day, "uw-section", f"{r.bed} {r.plant}: every sensor, on its profile"),
        html.Small(since),
    ]
    t = track(rule, schedule)
    return (
        [html.B(d), html.Span(when)],
        tiles(w, d, when, t, w3, unit),
        header_chips(rule, schedule, w2),
        rule_hover(rule, schedule, k, unit),
        cards,
        legend,
        trace(rule, schedule, sel, day, k, unit),
        trace_head,
        not len(w),
        dict(bed=r.bed, plant=r.plant, day=int(day)),
    )


# the address bar follows the day and plant shown (replaceState: no reload, no new history entry)
clientside_callback(
    """
    function(s) {
        if (!s || !s.bed || !s.plant || window.location.pathname.slice(-11) !== '/irrigation') {
            return window.dash_clientside.no_update;
        }
        var q = '?bed=' + encodeURIComponent(s.bed) + '&plant=' + encodeURIComponent(s.plant) + '&day=' + s.day;
        if (window.location.search !== q) {
            window.history.replaceState(window.history.state, '', window.location.pathname + q);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("ir-url-done", "data"),
    Input("ir-url", "data"),
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
