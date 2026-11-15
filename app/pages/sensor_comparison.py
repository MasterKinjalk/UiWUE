"""Sensor comparison: does one thermal camera need the other sensors, scored on the same plants. (Jack)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _jack_common import (
    GROUPS,
    ONE_CONTROL_BED,
    bed_effect_chip,
    chip,
    day_text,
    fmt,
    graph,
    header,
    missing_tile,
    not_available,
    panel,
    register_page,
    run_alone,
    segmented,
    table,
    tile,
)
from dash import Input, Output, callback, html

from uiwue.detect import figures
from uiwue.detect.benchmark import cheapest_close
from uiwue.detect.data import load_benchmark

QUESTION = "Does adding depth or RGB to one thermal camera detect deficit better, on the same plants?"
register_page(__name__, path="/sensor-comparison", name="Sensor comparison", order=40, description=QUESTION)

PLANT_SETS = {"common": "Same plants", "own": "All plants"}
METRICS = {"early_auc": "Days 1–4", "full_auc": "Days 1–12"}
LOPO = (
    "Each setup is scored on plants it never saw: leave-one-plant-out, logistic and boosted trees, inputs are "
    "each plant's change from its own 30 Aug – 2 Sep baseline, never date, bed or position. Early AUC: each "
    "plant's mean probability over days 1–4, deficit vs control. Cost rises left to right (rank, not price)."
)
SETS_HOVER = {
    "common": "Every setup on the same plant-days: those with thermal, depth and RGB (W1 and W2 only; W3 has "
    "thermal only).",
    "own": "Each setup on the plant-days that have all its sensors, so plant counts differ between setups.",
}


def _view(bench, plant_set):
    if "plant_set" not in bench.columns:
        return bench
    return bench[(bench["plant_set"] == plant_set) | (bench["plant_set"].isna() & (plant_set == "own"))]


def _n(r) -> str:
    return f"{int(r['n_deficit'])} · {int(r['n_control'])} plants"


def _hover(r, what) -> str:
    return (
        f"{what}: {r['setup']} ({r['model']}), early AUC {fmt(r['early_auc'])} "
        f"(95 % CI {fmt(r['early_ci_low'])}–{fmt(r['early_ci_high'])}); robust onset "
        f"{day_text(r.get('robust_onset_day'))}; placebo AUC {fmt(r['placebo_auc'])}; "
        f"{int(r['n_deficit'])} deficit · {int(r['n_control'])} control plants"
    )


def tiles(bench, group, plant_set):
    b = _view(bench, plant_set)
    b = b[(b["group"] == group) & ~b["sensors"].str.contains("_r", regex=False)]  # full-resolution thermal
    if b.empty:
        return [missing_tile("Thermal only", "No setup has plants in this group and plant set")]
    out = []
    th = b[b["sensors"] == "thermal"].sort_values("early_auc", ascending=False)
    if len(th):
        r = th.iloc[0]
        out.append(
            tile(
                "Thermal only", fmt(r["early_auc"]), _n(r), "gray", _hover(r, "One thermal camera"), bed_effect_chip(r)
            )
        )
    else:
        out.append(missing_tile("Thermal only", "No thermal rows"))
    multi = b[(b["n_sensors"] >= 2) & b["sensors"].str.contains("thermal")].sort_values("early_auc", ascending=False)
    if len(multi):
        r = multi.iloc[0]
        out.append(
            tile(
                "Best combination",
                fmt(r["early_auc"]),
                r["setup"],
                "gray",
                _hover(r, "Best multi-sensor setup"),
                bed_effect_chip(r),
            )
        )
    else:
        out.append(missing_tile("Best combination", "No multi-sensor setup for this group: W3 has thermal only"))
    size = b[b["is_size_baseline"].astype(bool)].sort_values("early_auc", ascending=False)
    if len(size):
        r = size.iloc[0]
        out.append(tile("Size only", fmt(r["early_auc"]), _n(r), "gray", _hover(r, "Plant size alone")))
    pick = cheapest_close(bench, group, plant_set=plant_set) if "plant_set" in bench.columns else None
    if pick is not None:
        out.append(
            tile(
                "Cheapest close",
                pick["setup"],
                pick["model"],
                "mint",
                _hover(pick, "Cheapest setup within 0.05 early AUC of the best, placebo clean"),
            )
        )
    return out


def layout(**_):
    bench = load_benchmark()
    head = header(
        "Sensor comparison",
        QUESTION,
        chip("LOPO", LOPO, "rule"),
        chip("1 control bed", ONE_CONTROL_BED),
        html.Span(id="sc-set-chip"),
        segmented("sc-set", PLANT_SETS, "common", "Plant set"),
        segmented("sc-group", GROUPS, "all", "Deficit beds"),
    )
    if bench is None or bench.empty:
        return html.Div([head, not_available("Sensor benchmark")])
    return html.Div(
        [
            head,
            html.Div(id="sc-tiles", className="uw-tiles", style={"paddingBottom": "16px"}),
            panel(
                "Setups",
                graph("sc-figure", "AUC per sensor setup and model, cheapest setup on the left"),
                hover="One point per setup and model; whiskers are 95 % CIs from resampling plants. Tick label: "
                "deficit · control plants. Open marker: placebo bed effect or the model already separates the "
                "beds on watered days. Dashed grey: plant size alone.",
                extra=[segmented("sc-metric", METRICS, "early_auc", "AUC window"), html.Span(id="sc-flags")],
            ),
            html.Div(panel("All setups", html.Div(id="sc-table")), style={"paddingTop": "16px"}),
        ]
    )


@callback(
    Output("sc-tiles", "children"),
    Output("sc-figure", "figure"),
    Output("sc-flags", "children"),
    Output("sc-table", "children"),
    Output("sc-set-chip", "children"),
    Input("sc-set", "value"),
    Input("sc-group", "value"),
    Input("sc-metric", "value"),
)
def _update(plant_set, group, metric):
    bench = load_benchmark()
    b = _view(bench, plant_set)
    g = b[b["group"] == group]
    fig = figures.cost_performance(b, group, metric)
    fl = g[figures.flagged(g) & ~g["is_size_baseline"].astype(bool)]
    flags = (
        [chip(f"{fl['setup'].nunique()} bed effect", "Flagged: " + "; ".join(sorted(set(fl["setup"]))))]
        if len(fl)
        else []
    )
    set_chip = chip(PLANT_SETS[plant_set].lower(), SETS_HOVER[plant_set], "context")
    if g.empty:
        return tiles(bench, group, plant_set), fig, flags, not_available("Setups for this group"), set_chip
    cols = {
        "setup": "Sensors",
        "model": "Model",
        "early_auc": "AUC days 1–4",
        "early_ci_low": "CI low",
        "early_ci_high": "CI high",
        "full_auc": "AUC days 1–12",
        "onset_day": "Rule onset (day)",
        "robust_onset_day": "Robust onset (day)",
        "placebo_auc": "Placebo AUC",
        "bed_effect": "Bed effect",
        "fires_before_cutoff": "Fires before day 0",
        "n_deficit": "Deficit plants",
        "n_control": "Control plants",
        "n_features": "Features",
    }
    cols = {k: v for k, v in cols.items() if k in g.columns}
    tbl = table(g.sort_values(metric, ascending=False), cols, id="sc-datatable")
    return tiles(bench, group, plant_set), fig, flags, tbl, set_chip


if __name__ == "__main__":
    run_alone(layout)
