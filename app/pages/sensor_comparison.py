"""Sensor comparison: which sensor or combination is worth buying, scored on the same plants. (Jack)

Deep link: ``/sensor-comparison?view=fusion&set=own&group=W3&window=full_auc``; every key is optional and the
defaults (sensors, same plants, W2, days 1–4) show the answer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import components
import pandas as pd
from _jack_common import (
    ONE_CONTROL_BED,
    SCORE_HOVER,
    C,
    bed_effect_chip,
    chip,
    chip_link,
    day_text,
    details,
    fmt,
    graph,
    header,
    load_result,
    missing_tile,
    next_steps,
    not_available,
    panel,
    register_page,
    run_alone,
    segmented,
    stat,
    table,
)
from dash import Input, Output, callback, clientside_callback, dcc, html

from uiwue.detect import figures
from uiwue.detect.benchmark import cheapest_close
from uiwue.detect.data import load_benchmark

QUESTION = "Which sensor, or combination, detects water deficit well enough to be worth buying?"
register_page(__name__, path="/sensor-comparison", name="Sensor comparison", order=40, description=QUESTION)

VIEWS = {"sensors": "Sensors", "fusion": "Fusion"}
PLANT_SETS = {"common": "Same plants", "own": "All plants"}
GROUPS = {"W2": "W2", "W3": "W3", "all": "W2 + W3"}
METRICS = {"early_auc": "Days 1–4", "full_auc": "Days 1–12"}
DEFAULTS = {"view": "sensors", "set": "common", "group": "W2", "window": "early_auc"}
FUSION = "fusion"  # the prefix of every fusion model name
MODEL = "logistic"  # the simpler of the two benchmark models: the one the tiles and the chart show
FUSION_MODELS = ("fusion", "fusion+lora_thermal", "fusion_v2")
LOPO = (
    "Each setup is scored on plants it never saw (leave one plant out), from each plant's change against its own "
    "30 Aug – 2 Sep baseline; never date, bed or position."
)
SETS_HOVER = {
    "common": "Every setup on the same plant-days: those with thermal, 3D and RGB (W1 and W2; W3 has no RGB).",
    "own": "Each setup on the plant-days that have all its sensors, so plant counts differ between setups.",
}
FUSION_HOVER = (
    "Fusion models, scored out of fold on whole plants. Frozen and thermal LoRA: image encoders on the 39 plants with "
    "all four sensors. v2: hand-made thermal features over a 6 h window, on 171 plants. Every fusion setup also tells "
    "the beds apart on 24–29 Aug, when all were watered."
)
FUSION_MODEL_HOVER = {
    "fusion": "frozen image encoders (DOFA, ImageBind), 39 plants",
    "fusion+lora_thermal": "image encoders with the thermal one fine-tuned (LoRA), 39 plants",
    "fusion_v2": "the chosen design: hand-made thermal features over 6 h, 171 plants",
}
SETUPS_MIN_PX = 0  # the setups now run down the page, so the chart fits a phone without its own scroll


def _is_fusion(bench):
    return bench["model"].astype(str).str.startswith(FUSION)


def _view(bench, plant_set):
    """Rows of one plant set, or of the fusion view ('fusion'): the fusion models with the feature benchmark's
    size-only rows beside them (DESIGN 7.6); the two never mix."""
    fus = _is_fusion(bench)
    sets = bench["plant_set"] if "plant_set" in bench.columns else None
    if plant_set == FUSION:
        size = bench["is_size_baseline"].astype(bool) & ((sets == "own") if sets is not None else True)
        return bench[fus | size]
    bench = bench[~fus]
    return bench if sets is None else bench[bench["plant_set"] == plant_set]


def _charted(b):
    """Full-resolution setups, plus the two low-resolution thermal cameras alone (a cheaper camera is a buying
    question); low-resolution combinations stay in the table."""
    low = b["sensors"].str.contains("_r", regex=False)
    return b[~low | (b["n_sensors"] == 1)]


def _row(b, sensors, model=MODEL):
    r = b[(b["sensors"] == sensors) & (b["model"] == model)]
    return r.iloc[0] if len(r) else None


def _onset(r) -> str:
    """The onset (robust bar) of a benchmark row, 'no onset' when it never clears it; the fusion tables have the
    contract rule only, said as such."""
    if str(r.get("model", "")).startswith(FUSION):
        return f"rule {day_text(r.get('onset_day'))}"
    rob = r.get("robust_onset_day")
    return day_text(rob) if pd.notna(rob) else "no onset"


def _hover(r, b=None) -> str:
    other = None
    if b is not None:
        o = b[(b["sensors"] == r["sensors"]) & (b["model"] != r["model"])]
        other = o.iloc[0] if len(o) else None
    onset = _onset(r)
    return (
        f"{figures.setup_label(r['setup'])}: score {fmt(r['early_auc'])} on days 1–4 "
        f"(95 % CI {fmt(r['early_ci_low'])}–{fmt(r['early_ci_high'])}), "
        + (onset if onset == "no onset" else f"onset {onset}")
        + (f"; boosted trees {fmt(other['early_auc'])}" if other is not None else "")
        + f". {int(r['n_deficit'])} deficit · {int(r['n_control'])} control plants."
    )


def _href(sensors, group):
    bed = "W3" if group == "W3" else "W2"
    return {
        "thermal": f"/plant-profile?bed={bed}&trait=thermal_leaf_air_dt",
        "rgb": f"/plant-profile?bed={bed}&trait=rgb_ngrdi",
        "depth": f"/threed?bed={bed}",
    }.get(sensors)


def _tile(b, sensors, label, metric, group, best=None):
    r = _row(b, sensors)
    if r is None:
        return missing_tile(label, f"No {label.lower()} setup for this group and plant set")
    is_best = best is not None and best["sensors"] == sensors
    return stat(
        label,
        fmt(r[metric]),
        "",
        "gray",
        _hover(r, b),
        sub=(f"onset {_onset(r)}" if pd.notna(r.get("robust_onset_day")) else "no onset")
        + f" · {int(r['n_deficit'])} + {int(r['n_control'])} plants",
        chips=[
            *(
                [chip("best buy", "Cheapest setup within 0.05 of the best score, no bed effect", "best")]
                if is_best
                else []
            ),
            *bed_effect_chip(r),
        ],
        href=_href(sensors, group),
        cls="uw-stat-best" if is_best else "",
    )


def tiles(bench, group, plant_set, metric="early_auc"):
    if plant_set == FUSION:
        return fusion_tiles(bench, group, metric)
    b = _view(bench, plant_set)
    b = b[b["group"] == group]
    if b.empty:
        return [missing_tile("Thermal alone", "No setup has plants in this group and plant set")]
    best = cheapest_close(bench[~_is_fusion(bench)], group, plant_set=plant_set) if "plant_set" in bench else None
    out = [
        _tile(b, "thermal", "Thermal alone", metric, group, best),
        _tile(b, "depth+thermal", "Thermal + 3D", metric, group, best),
        _tile(b, "rgb", "RGB alone", metric, group, best),
        _tile(b, "depth", "3D alone", metric, group, best),
    ]
    size = b[b["is_size_baseline"].astype(bool) & (b["model"] == MODEL)]
    if len(size):
        r = size.iloc[0]
        out.append(size_tile(r, metric))
    return out


def size_tile(r, metric):
    """Plant size alone, the floor every setup must beat (DESIGN 7.6): the last tile, as on every page."""
    return stat(
        components.SIZE_ONLY,
        fmt(r[metric]),
        "",
        "gray",
        f"Plant size alone (projected leaf area): the floor every setup must beat. Score {fmt(r[metric])}, "
        f"{int(r['n_deficit'])} deficit · {int(r['n_control'])} control plants.",
        sub="the floor",
        href=components.methods_href("size-baseline"),
    )


def fusion_tiles(bench, group, metric):
    b = _view(bench, FUSION)
    b = b[b["group"] == group]
    out = []
    v2 = _row(b, "thermal", "fusion_v2")
    img = b[b["model"].isin(["fusion", "fusion+lora_thermal"]) & ~b["sensors"].str.contains("_r", regex=False)]
    best_img = img.sort_values(metric, ascending=False).iloc[0] if len(img) else None
    shown = [r for r in (v2, best_img) if r is not None]
    # DESIGN 7.4: when every fusion setup is flagged, the chart's one 'all flagged' chip says it once
    per_tile = not all(bed_effect_chip(r) for r in shown)
    if v2 is not None:
        out.append(
            stat(
                "Fusion v2",
                fmt(v2[metric]),
                "",
                "gray",
                _hover(v2),
                sub=_onset(v2),
                chips=bed_effect_chip(v2) if per_tile else [],
            )
        )
    if best_img is not None:
        out.append(
            stat(
                "Image models",
                fmt(best_img[metric]),
                "",
                "gray",
                "Best image-encoder setup. " + _hover(best_img),
                sub=figures.setup_label(best_img["setup"]),
                chips=bed_effect_chip(best_img) if per_tile else [],
            )
        )
    lb = load_result("fusion_lockbox_benchmark")
    if lb is not None and not lb.empty:
        sel = lb[(lb["group"] == "all") & (lb["model"] == "fusion_v2") & (lb["sensors"] == "thermal")]
        if "plants" in lb.columns:
            sel = sel[sel["plants"] == "all"]
        if len(sel):
            r = sel.iloc[0]
            out.append(
                stat(
                    "Held-out plants",
                    day_text(r["onset_day"]),
                    "",
                    "sand",
                    f"Fusion v2 on 45 plants set aside before any experiment: score {fmt(r['early_auc'])}, but it "
                    f"passes the rule on {day_text(r['onset_day'])}, before any deficit: a bed difference, not "
                    "detection.",
                    sub=f"score {fmt(r['early_auc'])} · a bed difference",
                    href=components.methods_href("fusion-lockbox"),
                )
            )
    size = b[b["is_size_baseline"].astype(bool) & (b["model"] == MODEL)]
    if len(size):
        out.append(size_tile(size.iloc[0], metric))
    return out or [missing_tile("Fusion v2", "No fusion results for this group")]


def _legend(fusion):
    items = [
        html.Span([html.Span("●", **{"aria-hidden": "true"}), "score · 95 % CI"]),
        html.Span(
            [html.Span("○", **{"aria-hidden": "true"}), "flagged"],
            title="the setup already separates the beds on watered days",
        ),
        html.Span([html.Span("┆", **{"aria-hidden": "true"}), "Size only"], title="plant size alone: the floor"),
    ]
    if fusion:  # one colour per model (the chart's own legend stacks over the plot at phone width)
        sw = lambda bg: html.Span(className="uw-swatch", style={"background": bg}, **{"aria-hidden": "true"})
        items = [
            html.Span([sw(figures.MODEL_COLOUR[m]), figures.MODEL_LABEL[m]], title=FUSION_MODEL_HOVER[m])
            for m in FUSION_MODELS
        ] + items
    else:
        items.insert(
            0, html.Span([html.Span("●", style={"color": C["teal-700"]}, **{"aria-hidden": "true"}), "best buy"])
        )
    return html.Div(items, className="uw-legend")


def layout(view=None, set=None, group=None, window=None, bed=None, plant=None, day=None, **_):  # URL key: set
    """``bed``, ``plant`` and ``day`` are not used here: the next steps carry them on (the plant chosen earlier)."""
    view = view if view in VIEWS else DEFAULTS["view"]
    plant_set = set if set in PLANT_SETS else DEFAULTS["set"]
    group = group if group in GROUPS else DEFAULTS["group"]
    metric = window if window in METRICS else DEFAULTS["window"]
    bench = load_benchmark()
    head = header(
        "Sensor comparison",
        QUESTION,
        chip_link("1 control bed", ONE_CONTROL_BED, "one-control-bed"),
        segmented("sc-view", VIEWS, view, "View"),
    )
    if bench is None or bench.empty:
        return html.Div([head, not_available("Sensor benchmark")])
    options = details(
        "Options",
        html.Div(
            [
                html.Div(
                    [html.Span("Plants", className="uw-label"), segmented("sc-set", PLANT_SETS, plant_set, "Plants")],
                    title=" ".join(SETS_HOVER.values()),
                    className="dt-chips",
                ),
                html.Div(
                    [
                        html.Span("Deficit bed", className="uw-label"),
                        segmented("sc-group", GROUPS, group, "Deficit bed"),
                    ],
                    title="Deficit bed against the control bed W1. W3 has no RGB.",
                    className="dt-chips",
                ),
                html.Div(
                    [html.Span("Days", className="uw-label"), segmented("sc-metric", METRICS, metric, "Score window")],
                    title="Score on each plant's mean over days 1–4 (early) or days 1–12",
                    className="dt-chips",
                ),
            ],
            className="dt-options",
        ),
        id="sc-options",
    )
    return html.Div(
        [
            head,
            dcc.Store(id="sc-url"),
            html.Div(id="sc-tiles", className="uw-tiles uw-lead"),
            panel(
                "Detection score",
                graph("sc-figure", "Detection score per sensor setup, best at the top, with plant size alone"),
                html.Div(id="sc-legend"),
                hover=f"{SCORE_HOVER} {LOPO} Simple model (logistic) shown; boosted trees on hover and in the table.",
                extra=[html.Span(id="sc-flags", className="dt-chips")],
            ),
            options,
            details("All setups", html.Div(id="sc-table"), id="sc-all"),
            next_steps(
                (
                    "Daily rhythm",
                    components.page_href("/daily-rhythm", bed, plant, day),
                    "the best hour of the day to image",
                ),
                (
                    "Irrigation advisor",
                    components.page_href("/irrigation", bed, plant, day),
                    "which plants need water today",
                ),
            ),
        ]
    )


TABLE_COLUMNS = {
    "setup": "Sensors",
    "model": "Model",
    "early_auc": "Score days 1–4",
    "early_ci_low": "CI low",
    "early_ci_high": "CI high",
    "full_auc": "Score days 1–12",
    "robust_onset_day": "Onset (day)",
    "onset_day": "Contract rule (day)",
    "placebo_auc": "Score 24–29 Aug",
    "placebo_auc_main": "Own score 24–29 Aug",
    "bed_effect": "Bed effect",
    "fires_before_cutoff": "Fires before day 0",
    "n_deficit": "Deficit plants",
    "n_control": "Control plants",
    "n_features": "Features",
}


@callback(
    Output("sc-tiles", "children"),
    Output("sc-figure", "figure"),
    Output("sc-flags", "children"),
    Output("sc-table", "children"),
    Output("sc-legend", "children"),
    Output("sc-set", "options"),
    Input("sc-view", "value"),
    Input("sc-set", "value"),
    Input("sc-group", "value"),
    Input("sc-metric", "value"),
)
def _update(view, plant_set, group, metric):
    fusion = view == "fusion"
    ps = FUSION if fusion else plant_set
    bench = load_benchmark()
    b = _view(bench, ps)
    g = b[b["group"] == group]
    if fusion:
        rows = b[~b["sensors"].str.contains("_r", regex=False)]
        fig = figures.setup_scores(rows, group, metric, models=FUSION_MODELS)
        fig.update_layout(showlegend=False)  # the panel's legend names the models
    else:
        best = cheapest_close(bench[~_is_fusion(bench)], group, plant_set=ps) if "plant_set" in bench else None
        fig = figures.setup_scores(_charted(b), group, metric, highlight=None if best is None else best["setup"])
    shown = g[~g["sensors"].str.contains("_r", regex=False)] if fusion else _charted(g)
    fl = shown[
        figures.flagged(shown)
        & ~shown["is_size_baseline"].astype(bool)
        & shown["model"].isin(FUSION_MODELS if fusion else [MODEL])
    ]
    flags = []
    if fusion and len(fl):
        flags = [chip_link("all flagged", FUSION_HOVER, "fusion-lockbox")]
    elif len(fl):
        flags = [
            chip_link(
                f"{fl['setup'].nunique()} flagged",
                "These already separate the beds on watered days, before any deficit: "
                + "; ".join(sorted({figures.setup_label(s) for s in fl["setup"]})),
                "bed-effect",
            )
        ]
    set_options = [{"label": t, "value": v, "disabled": fusion} for v, t in PLANT_SETS.items()]
    if g.empty:
        tbl = not_available("Setups for this group")
    else:
        cols = {k: v for k, v in TABLE_COLUMNS.items() if k in g.columns and g[k].notna().any()}  # no empty column
        tbl = table(g.sort_values(metric, ascending=False), cols, id="sc-datatable")
    return tiles(bench, group, ps, metric), fig, flags, tbl, _legend(fusion), set_options


# the address bar follows the view and the options (replaceState: no reload, no new history entry)
clientside_callback(
    """
    function(view, set, group, win) {
        var u = new URL(window.location.href);
        var vals = {view: [view, 'sensors'], set: [set, 'common'], group: [group, 'W2'], window: [win, 'early_auc']};
        Object.keys(vals).forEach(function(k) {
            if (vals[k][0] === vals[k][1] || vals[k][0] == null) { u.searchParams.delete(k); }
            else { u.searchParams.set(k, vals[k][0]); }
        });
        window.history.replaceState(window.history.state, '', u.pathname + u.search + u.hash);
        return window.dash_clientside.no_update;
    }
    """,
    Output("sc-url", "data"),
    Input("sc-view", "value"),
    Input("sc-set", "value"),
    Input("sc-group", "value"),
    Input("sc-metric", "value"),
)


if __name__ == "__main__":
    run_alone(layout)
