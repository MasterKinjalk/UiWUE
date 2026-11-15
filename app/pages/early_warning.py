"""Early warning: which sensor warns of water deficit first, by how many days, against the leaf clip. (Jack)

Deep link: ``/early-warning?group=W2`` (``W2``, ``W3`` or ``all`` for W2 + W3). The default is W2 against the
control bed W1, the clean comparison used across the app: W3 plants had leaves cut for water content, which shrinks
their 3D canopy too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import components
import numpy as np
import pandas as pd
from _jack_common import (
    ONE_CONTROL_BED,
    ONSET_HOVER,
    SCORE_HOVER,
    C,
    bed_effect_chip,
    chip,
    chip_link,
    day_text,
    details,
    graph,
    header,
    load_result,
    min_width,
    missing_tile,
    next_steps,
    not_available,
    panel,
    register_page,
    run_alone,
    run_info,
    run_time,
    segmented,
    select,
    short,
    source_tag,
    stat,
    table,
)
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from uiwue.detect import figures

QUESTION = "Which sensor warns of water deficit first, by how many days, and against which ground truth?"
register_page(__name__, path="/early-warning", name="Early warning", order=30, description=QUESTION)

GROUPS = {"W2": "W2", "W3": "W3", "all": "W2 + W3"}  # W2 first: the clean comparison, the default
DEFAULT_GROUP = "W2"
RECOVERY = {  # 3D rows: the 3D page's leaf area and height; projected area is only the size baseline
    "thermal_leaf_air_dt": "function",
    "rgb_ngrdi": "function",
    "depth_leaf_area_est": "structure",
    "depth_height_p75": "structure",
}
SENSORS = (("Thermal", "Thermal"), ("Depth", "3D"), ("RGB", "RGB"))  # sensor in the tables -> tile label
W3_SAMPLED = (
    "Most W3 plants had a leaf cut for water content on one of days 1–12, so their 3D canopy loss mixes drought "
    "and sampling. W2 against W1 is the clean comparison."
)


def _rob(row):
    return row.get("robust_onset_day", row.get("onset_day", np.nan))


def _bed(group) -> str:
    return "W3" if group == "W3" else "W2"


def signal_href(signal, sensor, group) -> str:
    """The page that shows a signal on a plant: thermal and RGB on Plant profile, 3D on 3D canopy."""
    bed = _bed(group)
    if sensor == "Depth":
        return f"/threed?bed={bed}"
    return f"/plant-profile?bed={bed}&trait={signal}"


def w3_chip(onsets, group) -> list:
    """One caveat chip in the header while the group holds W3 plants; the hover gives W2's 3D leaf-area onset."""
    if group == "W2":
        return []
    w2 = onsets[(onsets["signal"] == "depth_leaf_area_est") & (onsets["group"] == "W2")]
    rob = _rob(w2.iloc[0]) if len(w2) else np.nan
    alone = f" W2 alone: 3D leaf area {day_text(rob)}." if pd.notna(rob) else ""
    return [chip_link("W3 sampled", W3_SAMPLED + alone, "w3-sampled")]


def _date(day) -> str:
    return components.day_label(day)[1] if pd.notna(day) else ""


def _n(row) -> str:
    return f"{int(row['n_deficit'])} deficit · {int(row['n_control'])} control plants"


def display_names(o: pd.DataFrame) -> np.ndarray:
    """Signal names as the ranking chart writes them: the size-only baseline as 'Size only' (DESIGN 7.6)."""
    size = o["is_size_baseline"].fillna(False).astype(bool).to_numpy()
    return np.where(size, components.SIZE_ONLY, o["label"].astype(str))


def _clip(onsets, group):
    """(median clip-turgor onset day, clip plants) of the group's imaged deficit plants; (nan, 0) if none."""
    o = figures.ranked(onsets, group)
    if o.empty or "clip_day" not in o or not int(o["n_clip"].fillna(0).max()):
        return np.nan, 0
    r = o.dropna(subset=["clip_day"])
    return (float(r["clip_day"].iloc[0]), int(r["n_clip"].iloc[0])) if len(r) else (np.nan, 0)


def ranking(onsets, group):
    """figures.onset_ranking (the onset only; the contract rule on hover) with room for the names, each signal's
    plant counts on hover, and the leaf clip's onset as the ground-truth line."""
    fig = figures.onset_ranking(onsets, group, rule=False)
    fig.update_layout(showlegend=False)  # the panel's legend names the sensors
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(title="day since irrigation off")
    o = figures.ranked(onsets, group)
    counts = dict(zip(display_names(o), zip(o["n_deficit"], o["n_control"], strict=True), strict=True))
    for tr in fig.data:
        if tr.y is None or "<extra>" not in (tr.hovertemplate or ""):
            continue
        cd = np.empty((len(tr.y), 0)) if tr.customdata is None else np.asarray(tr.customdata, dtype=object)
        k = cd.shape[1]
        n = np.array([[int(v) for v in counts[y]] if y in counts else ["–", "–"] for y in tr.y], dtype=object)
        tr.customdata = np.c_[cd, n]
        tr.hovertemplate = tr.hovertemplate.replace(
            "<extra>", f"<br>%{{customdata[{k}]}} deficit · %{{customdata[{k + 1}]}} control plants<extra>", 1
        )
    day, n_clip = _clip(onsets, group)
    if pd.notna(day):
        fig.add_vline(x=day, line_color=C["slate-600"], line_dash="dash", line_width=1.5)
        fig.add_annotation(
            x=day,
            y=1,
            yref="paper",
            yanchor="bottom",
            text="leaf clip",
            showarrow=False,
            font=dict(size=12, color=C["slate-600"]),
            hovertext=f"Leaf clip (ground truth): median turgor onset of {n_clip} deficit plants, {day_text(day)}",
        )
    return fig


def _best(o, sensor):
    """The sensor's earliest clean signal: not flagged, not plant size, with an onset; None if it has none."""
    clean = o[~figures.flagged(o) & ~o["is_size_baseline"].astype(bool) & (o["sensor"] == sensor)]
    clean = clean.dropna(subset=["robust_onset_day"])
    return clean.iloc[0] if len(clean) else None


def _sensor_tile(o, sensor, name, group):
    r = _best(o, sensor)
    if r is None:
        why = (
            f"No {name} plants in {GROUPS[group]}"
            if o[o["sensor"] == sensor].empty
            else f"No unflagged {name} signal clears the bar in {GROUPS[group]}"
        )
        return missing_tile(name, why)
    tech = components.tech(r["signal"])
    hover = (
        f"{tech}: onset {day_text(_rob(r))}, the first day it beats every watered-day score "
        f"(contract rule: {day_text(r['onset_day'])}). {_n(r)}."
    )
    return stat(
        name,
        day_text(_rob(r)),
        _date(_rob(r)),
        "gray",
        hover,
        sub=short(r["signal"], r["label"]),
        href=signal_href(r["signal"], sensor, group),
    )


def tiles(onsets, group):
    """The answer, one tile per sensor (its earliest clean signal), then plant size alone (the floor) and the
    leaf clip (the ground truth)."""
    o = figures.ranked(onsets, group)
    if o.empty:
        return [missing_tile("Onset", "No detection results for this group")]
    out = [_sensor_tile(o, sensor, name, group) for sensor, name in SENSORS]
    size = o[o["is_size_baseline"].astype(bool)]
    if len(size):
        s = size.iloc[0]
        out.append(
            stat(
                components.SIZE_ONLY,
                day_text(_rob(s)),
                _date(_rob(s)),
                "gray",
                hover=f"Plant size alone ({s['label'].lower()}): the floor every signal must beat. {_n(s)}.",
                sub=short(s["signal"], s["label"]),
                href=components.methods_href("size-baseline"),
            )
        )
    day, n_clip = _clip(onsets, group)
    thermal = _best(o, "Thermal")
    if pd.notna(day):
        lead = f" Thermal warns {int(day - _rob(thermal))} days earlier." if thermal is not None else ""
        out.append(
            stat(
                "Leaf clip",
                day_text(day),
                _date(day),
                "gray",
                hover=f"Ground truth: median day the leaf clip showed turgor loss in {n_clip} deficit plants.{lead}",
                sub=f"ground truth · {n_clip} plants",
                href=components.methods_href("lead-time"),
            )
        )
    else:
        out.append(missing_tile("Leaf clip", "No clip-turgor onset for the imaged plants of this group"))
    return out


RANKING_MIN_PX = 420  # narrower, the 14 day columns get closer than two markers: scroll in the panel instead


def evidence_legend():
    sw = lambda bg: html.Span(className="uw-swatch", style={"background": bg}, **{"aria-hidden": "true"})
    items = [
        html.Span([sw(figures.SENSOR_COLOURS[k]), lbl])
        for k, lbl in (("Thermal", "Thermal"), ("Depth", "3D"), ("RGB", "RGB"), ("size", components.SIZE_ONLY))
    ]
    items += [
        html.Span([html.Span("◇", **{"aria-hidden": "true"}), "flagged"], title="beds differed on watered days"),
        html.Span([html.Span("┆", **{"aria-hidden": "true"}), "leaf clip"], title="ground-truth onset"),
    ]
    return html.Div(items, className="uw-legend")


def curves_legend():
    sw = lambda bg: html.Span(className="uw-swatch", style={"background": bg}, **{"aria-hidden": "true"})
    return html.Div(
        [
            html.Span([html.Span("━", style={"color": C["teal-700"]}, **{"aria-hidden": "true"}), "signal · 95 % CI"]),
            html.Span([html.Span("┅", style={"color": C["gray-500"]}, **{"aria-hidden": "true"}), "Size only"]),
            html.Span([sw(C["gray"]), "watered-day range"], title="the onset must clear the largest watered-day score"),
            html.Span([sw(C["sand"]), "no water"], title="day 0 (3 Sep) to re-watering (15 Sep)"),
        ],
        className="uw-legend",
    )


def layout(group=None, bed=None, plant=None, day=None, **_):
    """``bed``, ``plant`` and ``day`` are not used here: the next steps carry them on, so the plant chosen on an
    earlier page is still chosen on the next one."""
    group = group if group in GROUPS else DEFAULT_GROUP
    auc, onsets = load_result("auc_by_day"), load_result("onsets")
    head = header(
        "Early warning",
        QUESTION,
        chip_link("1 control bed", ONE_CONTROL_BED, "one-control-bed"),
        html.Span(id="ew-w3"),
        segmented("ew-group", GROUPS, group, "Deficit bed, against control bed W1"),
    )
    if onsets is None or auc is None or onsets.empty:
        return html.Div([head, not_available("Detection onsets")])
    options = signal_options(onsets, group)
    best = figures.headline(onsets, group)
    first = best["signal"] if best is not None else next(iter(options), None)
    when = run_time(run_info())

    return html.Div(
        [
            head,
            dcc.Store(id="ew-url"),
            html.Div(id="ew-tiles", className="uw-tiles uw-lead"),
            html.Div(
                [
                    html.Div(
                        [
                            panel(
                                "Onset by signal",
                                html.Div(
                                    min_width(
                                        graph("ew-ranking", "Onset day per signal and the leaf clip, days 0 to 12"),
                                        RANKING_MIN_PX,
                                    ),
                                    className="uw-scroll",
                                ),
                                evidence_legend(),
                                hover=f"{ONSET_HOVER} Dot: the day; open diamond: flagged, not ranked. Click a dot "
                                "for its daily score."
                                + (f" Plant counts from the detect run of {when}." if when else ""),
                                extra=[html.Span(id="ew-flags")],
                            ),
                            panel(
                                "Daily score",
                                graph(
                                    "ew-curves",
                                    "Daily detection score of the chosen signal and plant size, day −10 to 18",
                                ),
                                curves_legend(),
                                hover=f"{SCORE_HOVER} Band: 95 % CI. Grey band: the largest score on watered days "
                                "(mint); the onset must clear it. Dashed grey: plant size alone.",
                                extra=[
                                    select("ew-signal", options, first, "Signal"),
                                    html.Span(id="ew-signal-chips", className="dt-chips"),
                                ],
                            ),
                        ],
                        className="uw-main",
                    ),
                    html.Aside(
                        panel(
                            "Loss regained",
                            html.Div(id="ew-recovery", className="dt-recovery"),
                            hover="Share of each signal's change that came back by 20–21 Sep (3D: 18–19 Sep), "
                            "against the plant's own baseline; 100 % = all of it. 3D canopy reads the same days "
                            "against each plant's peak.",
                        ),
                        className="uw-side",
                        **{"aria-label": "Loss regained"},
                    ),
                ],
                className="uw-row",
            ),
            details("All signals", html.Div(id="ew-table"), id="ew-all"),
            next_steps(
                (
                    "Sensor comparison",
                    components.page_href("/sensor-comparison", bed, plant, day),
                    "which sensor or combination is worth buying",
                ),
                (
                    "Daily rhythm",
                    components.page_href("/daily-rhythm", bed, plant, day),
                    "the best hour of the day to image",
                ),
            ),
        ]
    )


def recovery_rows(group):
    rec = load_result("recovery")
    s = figures.recovery_summary(rec, list(RECOVERY), group)
    if s.empty:
        return [not_available("Recovery")]
    s = s.set_index("signal").reindex([k for k in RECOVERY if k in set(s["signal"])]).reset_index()
    out = []
    for kind in ("function", "structure"):
        out.append(
            html.Span(
                kind,
                className="uw-kicker",
                style={"color": C["slate-600"]},
                title="leaf temperature and colour: how the plant works"
                if kind == "function"
                else "leaf area and height from the 3D camera: how big the plant is. Height comes back past its "
                "baseline, leaf area does not",
            )
        )
        for r in s[s["signal"].map(RECOVERY) == kind].itertuples():
            pct = float(np.clip(r.recovery, 0, 1)) * 100
            past = "; over 100 %: past its own baseline, the plants kept growing" if r.recovery > 1 else ""
            out.append(
                html.Div(
                    [
                        html.Span(short(r.signal, r.label), className="dt-rec-name"),
                        html.Span(
                            html.Span(style={"width": f"{pct:.0f}%", "background": C["teal-700"]}),
                            className="uw-bar",
                        ),
                        html.Span(f"{r.recovery * 100:.0f} %", className="uw-num"),
                    ],
                    className="dt-rec-row",
                    title=f"{short(r.signal, r.label)}: {r.recovery * 100:.0f} % of its change regained, median over "
                    f"{r.n} deficit plants{past}",
                )
            )
    return out


TABLE_COLUMNS = {
    "name": "Signal",
    "sensor": "Sensor",
    "robust_onset_day": "Onset (day)",
    "onset_day": "Contract rule (day)",
    "null_auc_max": "Watered-day max score",
    "early_auc": "Score days 1–4",
    "placebo_auc": "Score 24–29 Aug",
    "bed_effect": "Bed effect",
    "fires_before_cutoff": "Fires before day 0",
    "lead_vs_clip": "Lead vs clip (d)",
    "n_deficit": "Deficit plants",
    "n_control": "Control plants",
}


@callback(
    Output("ew-tiles", "children"),
    Output("ew-ranking", "figure"),
    Output("ew-flags", "children"),
    Output("ew-recovery", "children"),
    Output("ew-table", "children"),
    Output("ew-w3", "children"),
    Input("ew-group", "value"),
)
def _update(group):
    onsets = load_result("onsets")
    o = figures.ranked(onsets, group)
    flagged = o[figures.flagged(o)]
    flags = (
        [
            chip_link(
                f"{len(flagged)} flagged",
                "The beds already differed on watered days, so these are not ranked: "
                + "; ".join(flagged["label"].astype(str)),
                "bed-effect",
            )
        ]
        if len(flagged)
        else []
    )
    if o.empty:
        tbl = not_available("Onsets for this group")
    else:
        tbl = table(o.assign(name=display_names(o)), TABLE_COLUMNS, id="ew-datatable")
    return (
        tiles(onsets, group),
        ranking(onsets, group),
        flags,
        recovery_rows(group),
        tbl,
        w3_chip(onsets, group),
    )


def _hour_chip(slots, signal, group):
    """The onset from each flyover hour alone, in one chip (the hour itself is Daily rhythm's question)."""
    s = slots[(slots["signal"] == signal) & (slots["group"] == group)] if slots is not None else pd.DataFrame()
    if s.empty:
        return []
    parts = []
    for h in (6, 9, 12, 15, 18):
        r = s[s["hour"] == h]
        if r.empty:
            parts.append(f"{h:02d} h –")
            continue
        r = r.iloc[0]
        fire = " (fires on a watered day)" if bool(r["fires_before_cutoff"]) else ""
        parts.append(f"{h:02d} h {day_text(r['robust_onset_day'])}{fire}")
    return [chip("by hour", "Onset from each flyover hour alone: " + " · ".join(parts) + f". {_n(s.iloc[0])}.", "rule")]


@callback(
    Output("ew-curves", "figure"),
    Output("ew-signal-chips", "children"),
    Input("ew-group", "value"),
    Input("ew-signal", "value"),
)
def _signal(group, signal):
    auc, onsets, slots = load_result("auc_by_day"), load_result("onsets"), load_result("onsets_by_slot")
    o = onsets[(onsets["group"] == group) & (onsets["signal"] == signal)]
    if o.empty or int(o["n_deficit"].max()) == 0:
        return figures.empty_figure("No plants of this group have this signal"), []
    row = o.iloc[0]
    size = onsets.loc[onsets["is_size_baseline"].astype(bool) & (onsets["group"] == group), "signal"]
    fig = figures.auc_curves(auc, onsets, [signal, *[s for s in size if s != signal]], group)
    fig.update_layout(height=300, showlegend=False)  # the panel's legend names the lines
    chips = [source_tag(row["sensor"]), *bed_effect_chip(row), *_hour_chip(slots, signal, group)]
    return fig, chips


def signal_options(onsets, group) -> dict:
    o = figures.ranked(onsets, group)
    return dict(zip(o["signal"], display_names(o), strict=True))


@callback(
    Output("ew-signal", "options"),
    Output("ew-signal", "value"),
    Input("ew-group", "value"),
    Input("ew-ranking", "clickData"),
    State("ew-signal", "value"),
    prevent_initial_call=True,
)
def _pick(group, click, current):
    """The group's signals in the picker (the current one kept when the group has it, else the first signal); a
    click on a signal's dot in the ranking shows its daily score."""
    onsets = load_result("onsets")
    if onsets is None:
        return no_update, no_update
    opts = signal_options(onsets, group)
    options = [{"label": n, "value": s} for s, n in opts.items()]
    if ctx.triggered_id == "ew-ranking":
        y = ((click or {}).get("points") or [{}])[0].get("y")
        hit = [s for s, n in opts.items() if n == y]
        return options, hit[0] if hit else no_update
    if current in opts:
        return options, current
    best = figures.headline(onsets, group)
    return options, best["signal"] if best is not None else next(iter(opts), None)


# the address bar follows the group (replaceState: no reload, no new history entry)
clientside_callback(
    """
    function(group) {
        var u = new URL(window.location.href);
        if (group === 'W2') { u.searchParams.delete('group'); } else { u.searchParams.set('group', group); }
        window.history.replaceState(window.history.state, '', u.pathname + u.search + u.hash);
        return window.dash_clientside.no_update;
    }
    """,
    Output("ew-url", "data"),
    Input("ew-group", "value"),
)


if __name__ == "__main__":
    run_alone(layout)
