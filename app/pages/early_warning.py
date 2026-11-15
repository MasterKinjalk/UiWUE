"""Early warning: which sensor warns first, by how many days, against which ground truth. (Jack)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from _jack_common import (
    GROUPS,
    ONE_CONTROL_BED,
    C,
    bed_effect_chip,
    chip,
    day_text,
    fmt,
    graph,
    header,
    load_result,
    min_width,
    missing_tile,
    not_available,
    panel,
    readout,
    register_page,
    run_alone,
    run_info,
    run_time,
    segmented,
    select,
    signed_days,
    source_tag,
    table,
    tile,
)
from dash import Input, Output, callback, html

from uiwue.detect import figures

QUESTION = "Which sensor warns of water deficit first, by how many days, and against which ground truth?"
register_page(__name__, path="/early-warning", name="Early warning", order=30, description=QUESTION)

AUC_RULE = (
    "Daily value: median of the 12 h and 15 h runs passing QC, minus the plant's own 30 Aug – 2 Sep median. "
    "AUC: deficit vs control plants that day, 95 % CI from 2,000 plant resamples. "
    "Rule onset: first day the CI lower bound is above 0.70 on that day and the next. "
    "Robust onset: the same, with the bar raised to the largest AUC seen on any watered day (24 Aug – day 0)."
)
RECOVERY = {  # 3D rows: the 3D page's leaf area and height; projected area is only the size baseline
    "thermal_leaf_air_dt": "function",
    "rgb_ngrdi": "function",
    "depth_leaf_area_est": "structure",
    "depth_height_p75": "structure",
}


def _rob(row):
    return row.get("robust_onset_day", row.get("onset_day", np.nan))


W3_SAMPLED = (
    "W3 plants each had a leaf cut for relative water content on one of days 1–12, so their 3D canopy loss mixes "
    "drought and sampling; 3D canopy never pools W3 and compares W1 with W2 only."
)


def w3_chip(onsets, signal, sensor, group) -> list:
    """Caveat chip for a 3D result whose group holds W3 plants (W2 + W3, or W3): the leaf cut for water content
    shrinks their canopy too. The hover gives the same signal's W2-only onset, the number 3D canopy shows."""
    if sensor != "Depth" or group == "W2":
        return []
    w2 = onsets[(onsets["signal"] == signal) & (onsets["group"] == "W2")]
    rob = _rob(w2.iloc[0]) if len(w2) else np.nan
    alone = f" W2 alone: robust onset {day_text(rob)}." if pd.notna(rob) else ""
    return [chip("W3 sampled", W3_SAMPLED + alone)]


def _n(row) -> str:
    return f"{int(row['n_deficit'])} deficit · {int(row['n_control'])} control plants"


def display_names(o: pd.DataFrame) -> np.ndarray:
    """Signal names as the ranking chart writes them: the size-only baseline tagged 'size' (DESIGN 7.6)."""
    size = o["is_size_baseline"].fillna(False).astype(bool).to_numpy()
    return np.where(size, "size · " + o["label"].astype(str), o["label"].astype(str))


def ranking(onsets, group):
    """figures.onset_ranking with room for the signal names (automargin) and each signal's plant counts on
    hover, so a reader can see which plants an onset rests on."""
    fig = figures.onset_ranking(onsets, group)
    # The panel's legend explains rule / robust / flagged; Plotly's own would colour 'robust' after
    # whichever sensor happens to come first.
    fig.update_layout(showlegend=False)
    fig.update_yaxes(automargin=True)
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
    return fig


def _range(slots, onsets, row, group):
    """Robust onsets of the same signal over the bed groups and the 12 h / 15 h slots."""
    vals = [_rob(row)]
    if group == "all":
        o = onsets[(onsets["signal"] == row["signal"]) & onsets["group"].isin(["W2", "W3"])]
        vals += list(o["robust_onset_day"]) if "robust_onset_day" in o else []
    if slots is not None and not slots.empty:
        s = slots[(slots["signal"] == row["signal"]) & (slots["group"] == group) & slots["hour"].isin([12, 15])]
        vals += list(s["robust_onset_day"])
    vals = [v for v in vals if pd.notna(v)]
    return (min(vals), max(vals)) if vals else (np.nan, np.nan)


def tiles(onsets, slots, group):
    o = figures.ranked(onsets, group)
    best = figures.headline(onsets, group)
    out = []
    if best is None:
        return [missing_tile("Robust onset", "No unflagged signal clears the robust bar in this group")]
    lo, hi = _range(slots, onsets, best, group)
    rng = f"{int(lo)}–{int(hi)}" if pd.notna(lo) and lo != hi else ""
    out.append(
        tile(
            f"First · {best['sensor']}",
            day_text(_rob(best)),
            rng,
            "gray",
            hover=(
                f"{best['label']}: robust onset {day_text(_rob(best))} (bar {fmt(best.get('null_auc_max'))}, the "
                f"largest watered-day AUC); contract rule {day_text(best['onset_day'])}. Range {rng or 'none'}: "
                f"W2 and W3 apart and the 12 h and 15 h runs alone. {int(best['n_deficit'])} deficit · "
                f"{int(best['n_control'])} control plants."
            ),
        )
    )
    # next sensor: earliest unflagged signal of another sensor
    rest = o[~figures.flagged(o) & ~o["is_size_baseline"].astype(bool) & (o["sensor"] != best["sensor"])].dropna(
        subset=["robust_onset_day"]
    )
    if len(rest):
        r = rest.iloc[0]
        w3 = onsets[(onsets["signal"] == r["signal"]) & (onsets["group"] == "W3")]
        only = group == "all" and (w3.empty or int(w3["n_deficit"].max()) == 0)
        out.append(
            tile(
                f"Next · {r['sensor']}",
                day_text(_rob(r)),
                "W2 only" if only else "",
                "gray",
                hover=f"{r['label']}: robust onset {day_text(_rob(r))}, rule {day_text(r['onset_day'])}; {_n(r)}"
                + ("; no W3 plants in the detect results for this sensor" if only else ""),
                extra=w3_chip(onsets, r["signal"], r["sensor"], group),
            )
        )
    size = o[o["is_size_baseline"].astype(bool)]
    if len(size):
        s = size.iloc[0]
        out.append(
            tile(
                "Size only",
                day_text(_rob(s)),
                f"AUC {fmt(s['early_auc'])}",
                "gray",
                hover=f"{s['label']} alone (DESIGN 7.6): robust onset {day_text(_rob(s))}; "
                f"early AUC (days 1–4) {fmt(s['early_auc'])}, 0.5 = chance; {_n(s)}",
                extra=w3_chip(onsets, s["signal"], s["sensor"], group),
            )
        )
    n_clip = int(best.get("n_clip", 0) or 0)
    if n_clip:
        out.append(
            tile(
                "Lead vs leaf clip",
                signed_days(best.get("lead_vs_clip")),
                f"n {n_clip}",
                "gray",
                hover=(
                    f"Median clip-turgor onset of the {n_clip} imaged deficit clip plants: "
                    f"{day_text(best['clip_day'])} (clip shift > 5 units two days in a row; truth/plants), "
                    f"minus the robust onset {day_text(_rob(best))}. Positive = the camera warns first."
                ),
            )
        )
    else:
        out.append(missing_tile("Lead vs leaf clip", "No clip-turgor onset for the imaged plants of this group"))
    n_sym, n_img = int(best.get("n_symptom", 0) or 0), int(best.get("n_imaged", 0) or 0)
    if n_sym:
        first, med = best.get("symptom_first_day"), best.get("symptom_day")
        val = f"day {int(first)}–{int(med)}" if pd.notna(first) and first != med else day_text(med)
        out.append(
            tile(
                "Written notes",
                val,
                f"{n_sym} / {n_img} plants",
                "sand",
                hover=(
                    f"Wilting or yellowing written for {n_sym} of {n_img} imaged deficit plants; first {day_text(first)}, "
                    f"median {day_text(med)}. Notes were taken on 2, 6, 8, 10, 12 and 15 Sep only, so each date is an "
                    f"upper bound (median interval day {fmt(best.get('symptom_day_low'), 0)} to {fmt(med, 0)}). "
                    "Too few plants for a lead time."
                ),
            )
        )
    else:
        out.append(missing_tile("Written notes", "No imaged deficit plant of this group has a written symptom"))
    return out


RANKING_MIN_PX = 420  # narrower, the 14 day columns get closer than two markers: scroll in the panel instead


def evidence_legend():
    sw = lambda bg, edge=None: html.Span(
        className="uw-swatch", style={"background": bg, **({"border": edge} if edge else {})}, **{"aria-hidden": "true"}
    )
    items = [html.Span("sensor", style={"color": C["ink-muted"]})]
    items += [
        html.Span([sw(figures.SENSOR_COLOURS[k]), lbl])
        for k, lbl in (("Thermal", "IR"), ("Depth", "3D"), ("RGB", "RGB"), ("size", "size"))
    ]
    items += [
        html.Span([html.Span("○", **{"aria-hidden": "true"}), "rule"]),
        html.Span([html.Span("●", **{"aria-hidden": "true"}), "robust"]),
        html.Span([html.Span("◇", **{"aria-hidden": "true"}), "flagged"]),
    ]
    return html.Div(items, className="uw-legend")


def layout(**_):
    auc, onsets = load_result("auc_by_day"), load_result("onsets")
    head = header(
        "Early warning",
        QUESTION,
        chip("AUC rule", AUC_RULE, "rule"),
        chip("1 control bed", ONE_CONTROL_BED),
        segmented("ew-group", GROUPS, "all", "Deficit beds"),
    )
    if onsets is None or auc is None or onsets.empty:
        return html.Div([head, not_available("Detection onsets")])
    traits = figures.ranked(onsets, "all")
    options = {
        r["signal"]: ("size · " if r["is_size_baseline"] else "") + figures.signal_name(r) for _, r in traits.iterrows()
    }
    best = figures.headline(onsets, "all")
    first = best["signal"] if best is not None else next(iter(options))
    when = run_time(run_info())

    return html.Div(
        [
            head,
            html.Div(id="ew-tiles", className="uw-tiles", style={"paddingBottom": "16px"}),
            html.Div(
                [
                    html.Div(
                        [
                            panel(
                                "Onset by signal",
                                html.Div(
                                    min_width(
                                        graph("ew-ranking", "Onset day per signal, rule and robust, days 0 to 12"),
                                        RANKING_MIN_PX,
                                    ),
                                    className="uw-scroll",
                                ),
                                evidence_legend(),
                                hover="Open circle: contract rule (bar 0.70). Filled: robust onset (bar = largest "
                                "watered-day AUC). Diamond: flagged by the placebo or the pre-cutoff check. Plant "
                                "counts on hover: the plants with usable values when the onsets were computed"
                                + (f" (detect run {when})." if when else "."),
                                extra=[html.Span(id="ew-flags")],
                            ),
                            panel(
                                "Daily AUC",
                                html.Div(id="ew-signal-chips", style={"display": "flex", "gap": "8px"}),
                                graph(
                                    "ew-curves", "Daily AUC of the chosen signal and the size baseline, day −10 to 18"
                                ),
                                html.Div(
                                    id="ew-slots",
                                    className="uw-readouts",
                                    style={"gridTemplateColumns": "repeat(5, minmax(0, 1fr))"},
                                ),
                                hover="Line and band: AUC and 95 % CI per day. Grey band: largest separation on "
                                "watered days (mint). Dashed grey: plant size alone. Navy line: robust onset.",
                                extra=[select("ew-signal", options, first, "Signal")],
                            ),
                        ],
                        className="uw-main",
                    ),
                    html.Aside(
                        panel(
                            "Recovery",
                            html.Div(
                                id="ew-recovery", style={"display": "flex", "flexDirection": "column", "gap": "10px"}
                            ),
                            hover="Share of the stress change undone after re-watering (15 Sep): end of stress "
                            "14–15 Sep, after 20–21 Sep (depth 18–19 Sep, no depth later). Median over deficit "
                            "plants whose change exceeds twice the control spread.",
                        ),
                        className="uw-side",
                        **{"aria-label": "Recovery"},
                    ),
                ],
                className="uw-row",
            ),
            html.Div(panel("All signals", html.Div(id="ew-table")), style={"paddingTop": "16px"}),
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
        caveat = [chip("W3 sampled", W3_SAMPLED)] if kind == "structure" and group != "W2" else []
        out.append(
            html.Div(
                [html.Span(kind, className="uw-kicker", style={"color": C["slate-600"]}), *caveat],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
            )
        )
        for r in s[s["signal"].map(RECOVERY) == kind].itertuples():
            pct = float(np.clip(r.recovery, 0, 1)) * 100
            past = (
                "; over 100 %: past its own pre-cutoff value, the plants kept growing after re-watering"
                if r.recovery > 1
                else ""
            )
            out.append(
                html.Div(
                    [
                        html.Span(r.label, style={"flex": "0 0 140px"}),
                        html.Span(
                            html.Span(style={"width": f"{pct:.0f}%", "background": C["teal-700"]}),
                            className="uw-bar",
                        ),
                        html.Span(f"{r.recovery * 100:.0f} %", className="uw-num"),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                    title=f"{r.label}: median recovery {r.recovery * 100:.0f} % over {r.n} deficit plants{past}",
                )
            )
    return out


@callback(
    Output("ew-tiles", "children"),
    Output("ew-ranking", "figure"),
    Output("ew-flags", "children"),
    Output("ew-recovery", "children"),
    Output("ew-table", "children"),
    Input("ew-group", "value"),
)
def _update(group):
    onsets, slots = load_result("onsets"), load_result("onsets_by_slot")
    o = figures.ranked(onsets, group)
    flagged = o[figures.flagged(o)]
    flags = (
        [
            chip(
                f"{len(flagged)} flagged",
                "Bed or day effect: " + "; ".join(flagged["label"].astype(str)),
            )
        ]
        if len(flagged)
        else []
    )
    if o.empty:
        tbl = not_available("Onsets for this group")
    else:
        tbl = table(
            o.assign(name=display_names(o)),
            {
                "name": "Signal",
                "sensor": "Sensor",
                "onset_day": "Rule onset (day)",
                "robust_onset_day": "Robust onset (day)",
                "null_auc_max": "Watered max AUC",
                "fires_before_cutoff": "Fires before day 0",
                "early_auc": "AUC days 1–4",
                "placebo_auc": "Placebo AUC",
                "bed_effect": "Bed effect",
                "lead_vs_clip": "Lead vs clip (d)",
                "n_deficit": "Deficit plants",
                "n_control": "Control plants",
            },
            id="ew-datatable",
        )
    return tiles(onsets, slots, group), ranking(onsets, group), flags, recovery_rows(group), tbl


@callback(
    Output("ew-curves", "figure"),
    Output("ew-signal-chips", "children"),
    Output("ew-slots", "children"),
    Input("ew-group", "value"),
    Input("ew-signal", "value"),
)
def _signal(group, signal):
    auc, onsets, slots = load_result("auc_by_day"), load_result("onsets"), load_result("onsets_by_slot")
    o = onsets[(onsets["group"] == group) & (onsets["signal"] == signal)]
    if o.empty or int(o["n_deficit"].max()) == 0:
        return figures.empty_figure("No plants of this group have this signal"), [], []
    row = o.iloc[0]
    size = onsets.loc[onsets["is_size_baseline"].astype(bool) & (onsets["group"] == group), "signal"]
    fig = figures.auc_curves(auc, onsets, [signal, *[s for s in size if s != signal]], group)
    chips = [source_tag(row["sensor"]), *bed_effect_chip(row), *w3_chip(onsets, signal, row["sensor"], group)]
    cells = []
    s = slots[(slots["signal"] == signal) & (slots["group"] == group)] if slots is not None else pd.DataFrame()
    for h in (6, 9, 12, 15, 18):
        r = s[s["hour"] == h] if len(s) else s
        if r.empty:
            cells.append(html.Div(readout(f"{h:02d} h", "–"), title=f"{h:02d} h: no runs for this signal"))
            continue
        r = r.iloc[0]
        fire = bool(r["fires_before_cutoff"])
        cells.append(
            html.Div(
                readout(f"{h:02d} h" + (" · day effect" if fire else ""), day_text(r["robust_onset_day"])),
                title=(
                    f"{h:02d} h run alone: robust onset {day_text(r['robust_onset_day'])}, rule "
                    f"{day_text(r['onset_day'])}, watered max AUC {fmt(r['null_auc_max'])}; {_n(r)}"
                    + (f"; rule fires on a watered day ({day_text(r['precutoff_fire_day'])})" if fire else "")
                ),
            )
        )
    return fig, chips, cells


if __name__ == "__main__":
    run_alone(layout)
