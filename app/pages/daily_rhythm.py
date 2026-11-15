"""Daily rhythm: how the signals move through the day, whether stress changes that rhythm, and when to image.

Finding first: three tiles computed live from the tables (how much hotter the deficit leaves run than the control
bed at midday and at dawn, the 3D team's midday-dip test, the best flyover). The day profile is the page's figure;
the flyover table and the hour × day separation grid support it. A link's ``?bed=&plant=&day=`` is carried on to
the next pages.
"""

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import threed_canopy as tc
from components import (
    GROUP,
    RULE_SHORT,
    C,
    chip,
    day_label,
    graph,
    header,
    next_steps,
    not_available,
    page_href,
    panel,
    placebo_chip,
    segmented,
    select,
    signed,
    size_chip,
    stat,
    trait_options,
    unit_of,
)
from dash import Input, Output, callback, html

from uiwue.core import SLOTS, load_traits, read_table
from uiwue.diurnal import PHASE_BINS, PLACEBO_DAYS, PRE_MAX_DAY, PRE_MAX_RATE, SCHEDULES, best_rule, phase_bin
from uiwue.stats._design import qc_pass

QUESTION = "How do the signals move through the day, does stress change that rhythm, and when should you image?"
dash.register_page(__name__, path="/daily-rhythm", name="Daily rhythm", order=50, description=QUESTION)

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
KEPT_3D = {t.column for t in tc.TRAITS.values()}  # the app's only 3D traits; the noisy depth traits were dropped
PROFILE_HOVER = (
    "Mean over plants at each flyover hour, QC-pass runs only; bars: standard error. Raw values, not change from "
    "baseline: the shape of the day is the point."
)
NOT_POOLED_3D = (
    "3D by treatment: deficit is W2 only, as on 3D canopy. W3 is not pooled (leaves cut for water content); "
    "group by bed to see it."
)
HEAT_DAYS = (8, 12)  # the last days without water: the profile's "day 8–12" panel
PROFILE_PX, PROFILE_MIN_PX = 380, 560  # height; the five panels need this width to keep level hour labels
NEXT = (  # the page's next steps, at its foot: (text, path, hover); a link's selection is carried on
    ("Irrigation advisor", "/irrigation", "which plants need water, imaged at the best flyover"),
    ("Early warning", "/early-warning", "which sensor warns first, and by how many days"),
)
# Panel titles of the day profile, short enough not to run into each other at phone width; the span goes on hover
PHASE_TITLES = dict(
    zip(
        PHASE_BINS[1],
        (
            ("before", "days −10 to −1: all plants watered"),
            ("day 0–3", "days 0 to 3 without water"),
            ("day 4–7", "days 4 to 7 without water"),
            ("day 8–12", "days 8 to 12 without water (re-watered on day 12 at 18:35)"),
            ("after", "days 13 to 19, after re-watering"),
        ),
        strict=True,
    )
)


def facet_title(a):
    """'phase_bin=days 0-3' -> 'day 0–3', with the span of days as the annotation's hover text."""
    name = a.text.split("=")[-1]
    short, hover = PHASE_TITLES.get(name, (name, name))
    a.update(text=short, hovertext=hover)


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


def best_hours(sch):
    """The flyover hours of the best rule's recommended schedule (uiwue.diurnal), () without one."""
    rule = best_rule(sch) if sch is not None and "recommended" in sch else None
    rec = sch[sch.recommended & (sch.rule == rule)] if rule else None
    return tuple(SCHEDULES.get(rec.schedule.iloc[0], ())) if rec is not None and len(rec) else ()


def band_row(days):
    """The time bands over the grid, named once each: the baseline (mint) and the no-water window (sand)."""
    kind = {d: "mint" if -4 <= d <= -1 else "sand" if 0 <= d <= 12 else None for d in days}
    name = {"mint": "baseline", "sand": "no water"}
    cells, last = [], None
    for d in days:
        k = kind[d]
        cells.append(
            html.Div(
                html.Span(name[k], className="rx-band-name") if k and k != last else None,
                className=f"uw-heat-band rx-band uw-{k}" if k else "uw-heat-band rx-band",
            )
        )
        last = k
    return html.Div(
        [html.Div(style={"flex": "0 0 56px"}), *cells, html.Div(style={"flex": "0 0 76px"})],
        className="uw-heat-row",
        title="baseline 30 Aug – 2 Sep (mint) · no water 3 – 15 Sep (sand)",
    )


def heatmap(sep, best=()):
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
    rows = [head, band_row(days)]
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
                    title=f"{h:02d} h · {day} · {when}: score {v:.2f}",
                )
            )
        rows.append(
            html.Div(
                [
                    html.Div(
                        html.Span(f"{h:02d}", className="rx-best" if h in best else None),
                        className="uw-heat-key",
                        title="a flyover of the recommended schedule" if h in best else None,
                    ),
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
    ring = {"outline": f"2px solid {C['navy']}", "outlineOffset": "-2px"}
    # one ramp, named at its ends (each bin on hover), and the outline; the band row explains itself on hover
    ramp = html.Span(
        [
            html.Span(className="uw-swatch rx-ramp-cell", style={"background": c}, title=f"score {t}")
            for _, c, t in AUC_BINS
        ],
        className="rx-ramp",
    )
    legend = html.Div(
        [
            html.Span(
                ["score < 0.3", ramp, "≥ 0.9"],
                title="Detection score (AUC) of deficit against control plants: 0.5 (grey) alike; amber, the deficit plants score "
                "higher (more stressed); teal, lower",
            ),
            html.Span([html.Span(className="uw-swatch", style=ring), "first 2 days"], title="first two days ≥ 0.9"),
            *(
                [
                    html.Span(
                        [html.Span("12", className="rx-best rx-best-key", **{"aria-hidden": "true"}), "best flyover"],
                        title="The flyovers of the recommended schedule: "
                        + " + ".join(f"{h:02d}" for h in best)
                        + " h",
                    )
                ]
                if best
                else []
            ),
        ],
        className="uw-legend",
    )
    return [html.Div(html.Div(rows, className="uw-heat"), className="uw-scroll"), legend]


def bed_effect(sep):
    """Caveat chip when the placebo window (24-29 Aug, days -10 to -5) already separates the beds at some hour."""
    pre = sep[[c for c in sep.columns if PLACEBO_DAYS[0] <= c <= PLACEBO_DAYS[1]]].mean(axis=1)
    off = pre[(pre - 0.5).abs() >= 0.1]
    if off.empty:
        return []
    hours = " and ".join(f"{h:02d} h" for h in off.index)
    detail = ", ".join(f"{h:02d} h score {v:.2f}" for h, v in off.items())
    hover = f"The beds already differ on watered days 24–29 Aug ({detail}; 0.5 = no difference)"
    return [chip(f"bed effect at {hours}", hover, anchor="bed-effect")]


def verdict(a):
    """The one mark of a schedule row: 'best' (uiwue.diurnal's recommendation for this rule), a caveat
    chip when it fails the watered-day check, else nothing."""
    if a.recommended:
        return html.Span("best", className="uw-chip uw-chip-best", title="recommended for this rule (W2 alone)")
    if a.pre_pass:
        return ""
    return chip(
        "bed effect" if a.pre_worst_day_frac > PRE_MAX_DAY else "noisy",
        f"Watered days: {100 * a.pre_above_rate:.0f} % of scans above threshold, "
        f"{100 * a.pre_worst_day_frac:.0f} % of plants on day {str(int(a.pre_worst_day)).replace('-', '−')} "
        f"(passes at ≤ {100 * PRE_MAX_RATE:.0f} % and ≤ {100 * PRE_MAX_DAY:.0f} %)",
        anchor="bed-effect",
    )


def when_table(sch, rule):
    """When to image: per flyover schedule, alert day, control false alerts, watered-day check, one mark."""
    t = sch[sch.rule == rule]
    w2, w3 = (t[t.set == s].set_index("schedule").reindex(list(SCHEDULES)) for s in ("W2 tuning", "W3 check"))
    day = lambda v: "–" if pd.isna(v) else f"day {v:g}"
    cols = [
        ("Flyover", "", "flyovers per day"),
        ("Alert day", "num", "median alert day on the held-out W3 bed; W2 on hover"),
        ("False alarms", "num", "held-out watered W1 plants flagged, days 0–12"),
        ("Noise", "num", "scans above threshold before the cutoff, with every plant watered (W2 set)"),
        ("", "", ""),
    ]
    head = html.Tr([html.Th(n, className=c, title=h) for n, c, h in cols])
    body = []
    for name in SCHEDULES:
        a, b = w2.loc[name], w3.loc[name]
        if pd.isna(a.scans_per_day) or pd.isna(b.n_plants):
            continue
        body.append(
            html.Tr(
                [
                    html.Td(name.replace("all five", "all 5"), title=f"threshold {a.threshold:.2f}"),
                    html.Td(
                        day(b.median_alert_day),
                        className="num",
                        title=f"W3 {int(b.n_detected)} / {int(b.n_plants)} flagged; "
                        f"W2 {day(a.median_alert_day)}, {int(a.n_detected)} / {int(a.n_plants)}",
                    ),
                    html.Td(f"{int(b.false_alerts_controls)} / {int(b.n_controls)}", className="num"),
                    html.Td(
                        f"{100 * a.pre_above_rate:.1f} %",
                        className="num",
                        title=f"{int(a.pre_above)} of {int(a.pre_scans)} scans; W3 set {100 * b.pre_above_rate:.1f} %",
                    ),
                    html.Td(verdict(a)),
                ],
                className="uw-on" if a.recommended else None,
            )
        )
    return html.Div(html.Table([html.Thead(head), html.Tbody(body)], className="uw-table"), className="uw-scroll")


def midday_heat():
    """The day profile's last no-water panel in numbers: deficit leaves (W2 + W3) minus the control bed's leaves
    at each flyover hour over days 8–12 (QC-pass runs), and the same for the control plants. None without data."""
    df = rhythm_runs(DEFAULT_TRAIT, "treatment")
    d = df[df.days_since_cutoff.between(*HEAT_DAYS)]
    if d.empty:
        return None
    deficit = d[d.bed != "W1"].groupby("hour")[DEFAULT_TRAIT].mean()
    control = d[d.bed == "W1"].groupby("hour")[DEFAULT_TRAIT].mean()
    return deficit, control


def tiles(sch):
    """The page's answer: midday heat (thermal), the 3D midday dip, the best flyover."""
    out = []
    heat = midday_heat()
    if heat is not None and {6, 12} <= set(heat[0].index):
        deficit, control = heat
        sky = read_table("diurnal", "sky_effect")
        sky = None if sky is None else sky.dropna(subset=["sunny_extra"])
        sun = (
            f" Sunny runs add up to {signed(sky.sunny_extra.max(), '°C')} "
            f"({int(sky.loc[sky.sunny_extra.idxmax(), 'hour']):02d} h; few runs)."
            if sky is not None and len(sky)
            else ""
        )
        out.append(
            stat(
                f"Midday heat · day {HEAT_DAYS[0]}–{HEAT_DAYS[1]}",
                signed(deficit[12], ""),
                "°C",
                "sand",
                f"Deficit leaves (W2 + W3) minus the control bed's leaves, days {HEAT_DAYS[0]}–{HEAT_DAYS[1]}, mean of "
                f"QC-pass runs: {signed(deficit[12], '°C')} at 12 h, {signed(deficit[6], '°C')} at 06 h; control "
                f"plants {signed(control.get(12), '°C')}.{sun}",
                sub=f"06 h {signed(deficit[6], '°C')}",
            )
        )
    else:
        out.append(not_available("Thermal"))
    out.append(dip_tile())
    out.append(flyover_tile(sch))
    return out


def dip_tile():
    """The 3D team's test (``threed_canopy.midday_dip``): does drought deepen the midday dip in height?"""
    m = tc.midday_dip()
    if m is None or "height_p75" not in m:
        return stat("Midday dip · 3D", "–", "", "gray", "no 3D daily-rhythm table (derived/threed/results/e4)")
    h = m["height_p75"]
    unit = h["unit"].replace(" per day", "/day")
    none = h["ci_low"] <= 0 <= h["ci_high"]
    leaf = m.get("leaf_area_est")
    extra = (
        f" Leaf area does dip more at midday once dark runs are left out: {signed(leaf['nodark_slope'], '', 1)} "
        f"{leaf['unit'].replace('cm2', 'cm²')} (p = {leaf['nodark_p']:.3f})."
        if leaf and np.isfinite(leaf["nodark_ci_high"]) and leaf["nodark_ci_high"] < 0
        else ""
    )
    return stat(
        "Midday dip · 3D",
        "none" if none else signed(h["slope"], "", 2),
        "" if none else unit,
        "gray",
        f"Height, 15 h minus 09 h: drought changes it by {signed(h['slope'], '', 2)} {h['unit']} without water "
        f"(95 % CI {signed(h['ci_low'], '', 2)} to {signed(h['ci_high'], '', 2)}, p = {h['p']:.2f}; mixed model with "
        f"VPD, {h['n_plants']} plants, derived/threed/results/e4_models).{extra}",
        sub=f"height {signed(h['slope'], unit, 2)}",
    )


def flyover_tile(sch):
    """The best rule's recommended flyover (chosen on W2) and how it did on the held-out W3 bed."""
    rule = best_rule(sch)
    rec = sch[sch.recommended & (sch.rule == rule) & (sch.set == "W3 check")]
    if rule is None or rec.empty:
        return stat("Best flyover", "–", "", "mint", "no recommended schedule in derived/diurnal/schedules")
    r = rec.iloc[0]
    day = "–" if pd.isna(r.median_alert_day) else f"day {r.median_alert_day:g}"
    return stat(
        "Best flyover",
        r.schedule.replace("all five", "all 5"),
        "",
        "mint",
        f"Rule {RULE_SHORT[rule]}, threshold set on W2. On the held-out W3 bed: {r.n_detected} / {r.n_plants} plants "
        f"flagged, median {day}; {r.false_alerts_controls} / {r.n_controls} watered controls flagged.",
        sub=f"alert {day} · {r.false_alerts_controls} / {r.n_controls} false",
        chips=[size_chip()],
    )


def layout(bed=None, plant=None, day=None, **_):
    df = load_traits()
    head = header("Daily rhythm", QUESTION)
    if df is None or "thermal_leaf_t" not in df:
        return html.Div([head, not_available("Thermal")])
    sep, sch = read_table("diurnal", "separability"), read_table("diurnal", "schedules")
    if sep is None or sch is None or "pre_pass" not in sch:
        return html.Div([head, not_available("Daily rhythm")])
    shown = [c for c in df.columns if not c.startswith("depth_") or c in KEPT_3D]
    traits = {o["value"]: o["label"] for o in trait_options(shown)}
    day_profile = graph("dr-profile", "Mean of the chosen trait at each flyover hour, by group and stress phase")
    day_profile.children.style = {"height": f"{PROFILE_PX}px", "minWidth": f"{PROFILE_MIN_PX}px"}
    profile = panel(
        "Day profile",
        html.Div(day_profile, className="uw-scroll"),  # five panels: narrower, it scrolls in its own box
        hover=PROFILE_HOVER,
        extra=[
            html.Div(id="dr-trait-chips", className="rx-chips"),
            html.Div(className="uw-spacer"),
            select("dr-trait", traits, DEFAULT_TRAIT, "Trait"),
            segmented("dr-group", {"treatment": "Treatment", "bed": "Bed"}, "treatment", "Group by"),
        ],
    )
    schedules = panel(
        "When to image",
        html.Div(id="dr-schedules"),
        hover="Per flyover: median alert day on the held-out W3 bed, watered controls flagged, and watered-day scans "
        "above the threshold. Alert: above threshold on two usable scans in a row; threshold set on W2.",
        extra=[html.Div(className="uw-spacer"), select("dr-rule", RULE_SHORT, best_rule(sch) or "control_ref", "Rule")],
    )
    heat = panel(
        "Separation by hour",
        html.Div(id="dr-heat"),
        hover="Detection score (AUC) of deficit (W2 + W3) against control (W1) plants, per flyover hour and day; 0.5 = no difference. "
        "Outlined: the first two days in a row at ≥ 0.9.",
        extra=[
            html.Div(id="dr-chips", className="rx-chips"),
            html.Div(className="uw-spacer"),
            segmented("dr-score", SCORES, "diurnal_score_ref", "Score"),
        ],
    )
    return html.Div(
        [
            head,
            html.Div(tiles(sch), className="uw-tiles uw-lead"),
            html.Div(
                [html.Main(profile, className="uw-main"), html.Aside(schedules, className="uw-side rx-side")],
                className="uw-row",
            ),
            html.Div(heat, className="rx-below"),
            next_steps(*((t, page_href(path, bed, plant, day), hover) for t, path, hover in NEXT)),
        ]
    )


@callback(Output("dr-heat", "children"), Output("dr-chips", "children"), Input("dr-score", "value"))
def update_separability(score):
    s = read_table("diurnal", "separability")
    s = s[(s.score == score) & (s.group == "W2 + W3")].pivot(index="hour", columns="day", values="auc")
    if s.empty:
        return not_available("Separability for this score"), []
    first = {h: first_separation(s.loc[h].dropna()) for h in s.index}
    label = (
        f"AUC of {SCORES[score]} by flyover hour (06–18 h) and day since irrigation off "
        f"({s.columns.min()} to {s.columns.max()}); stressed and watered plants separate "
        + ", ".join(f"at {h:02d} h from day {d}" if d is not None else f"never at {h:02d} h" for h, d in first.items())
    )
    best = best_hours(read_table("diurnal", "schedules"))
    grid = html.Div(heatmap(s, best), role="img", title=SCORE_HOVER[score], **{"aria-label": label})
    return grid, bed_effect(s)


def rhythm_runs(trait, group):
    """The runs behind the day profile: those with a value that passed their module's QC (the tables keep the
    values of failed runs). A 3D trait grouped by treatment compares W1 with W2 only, as the 3D canopy page does:
    W3 is never pooled into its deficit numbers (leaves cut for water content)."""
    df = load_traits()
    df = df[df[trait].notna() & qc_pass(df, trait)]
    if trait in KEPT_3D and group == "treatment":
        df = df[df.bed.isin(tc.BEDS)]
    return df


def trait_chips(trait, group):
    """The bed-effect chip of the trait, and for a 3D trait grouped by treatment the W1 + W2 caveat."""
    out = list(placebo_chip(trait))
    if trait in KEPT_3D and group == "treatment":
        out.append(chip("W1 + W2 only", NOT_POOLED_3D, anchor="w3-sampled"))
    return out


@callback(
    Output("dr-profile", "figure"),
    Output("dr-trait-chips", "children"),
    Input("dr-trait", "value"),
    Input("dr-group", "value"),
)
def update_rhythm(trait, group):
    df = rhythm_runs(trait, group)
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
    prof.for_each_annotation(facet_title)
    prof.update_traces(line_width=2.5, marker_size=8, hovertemplate=f"%{{x}} h<br>%{{y:.2f}} {unit}")
    # level hour labels (they rotate on narrow panels otherwise; the chart scrolls below PROFILE_MIN_PX) and one
    # shared unit title under the middle panel
    prof.update_xaxes(tickvals=[6, 12, 18], tickangle=0, title_text=None)
    prof.update_xaxes(title_text="h", col=3)
    prof.update_layout(height=PROFILE_PX, margin=dict(t=72), legend=dict(y=1.16, title_text=""))
    return prof, trait_chips(trait, group)


@callback(Output("dr-schedules", "children"), Input("dr-rule", "value"))
def update_schedules(rule):
    sch = read_table("diurnal", "schedules")
    if sch is None or "pre_pass" not in sch:
        return not_available("Daily rhythm")
    return when_table(sch, rule)
