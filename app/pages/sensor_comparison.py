"""Sensor comparison: what combining sensors adds, and the cheapest setup that comes close. (Jack)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _jack_common import header, load_result, not_available, register_page, run_alone, table
from dash import Input, Output, callback, dcc, html

from uiwue.detect import figures
from uiwue.detect.benchmark import cheapest_close

register_page(__name__, path="/sensor-comparison", name="Sensor comparison", order=40)


def _headline(bench, group):
    pick = cheapest_close(bench, group)
    if pick is None:
        return "No sensor setup has a result for this group yet."
    b = bench[(bench["group"] == group) & ~bench["is_size_baseline"] & ~bench["bed_effect"]]
    best = b.sort_values("early_auc", ascending=False).iloc[0]
    size = bench[(bench["group"] == group) & bench["is_size_baseline"]]["early_auc"].max()
    onset = (
        f", onset day {int(pick['onset_day'])}" if pd.notna(pick["onset_day"]) else ", no onset in the stress window"
    )
    return (
        f"Best setup: {best['setup']} ({best['model']}), early AUC {best['early_auc']:.2f}. "
        f"Cheapest within 0.05 of it: {pick['setup']} ({pick['model']}), "
        f"early AUC {pick['early_auc']:.2f}{onset}. "
        + (f"Plant size alone: {size:.2f}." if pd.notna(size) else "No plant-size baseline for this group.")
    )


def layout(**_):
    bench = load_result("benchmark")
    head = [
        header(
            "Sensor comparison",
            "Does combining sensors beat the best single sensor, and what is the cheapest setup that comes close?",
        )
    ]
    if bench is None or bench.empty:
        return html.Div(head + [not_available("The sensor benchmark")])
    return html.Div(
        head
        + [
            html.Div(id="sc-headline", style={"fontSize": "1.05rem", "marginBottom": "1rem"}),
            html.Div(
                [
                    dcc.RadioItems(
                        [
                            {"label": "All deficit plants", "value": "all"},
                            {"label": "W2", "value": "W2"},
                            {"label": "W3", "value": "W3"},
                        ],
                        "all",
                        id="sc-group",
                        inline=True,
                    ),
                    dcc.RadioItems(
                        [
                            {"label": "Early AUC (days 1-4)", "value": "early_auc"},
                            {"label": "AUC days 1-12", "value": "full_auc"},
                        ],
                        "early_auc",
                        id="sc-metric",
                        inline=True,
                    ),
                ],
                style={"display": "flex", "gap": "2rem", "flexWrap": "wrap"},
            ),
            dcc.Graph(id="sc-figure"),
            html.P(
                "Each point is one sensor setup scored on plants it never saw "
                "(leave-one-plant-out), from each plant's change against its own pre-cutoff "
                "baseline. Setups run left to right from cheapest (webcam) to dearest "
                "(hyperspectral). Open markers separated the beds before any plant was stressed "
                "and are excluded from the recommendation. Cost is a rank, not a price.",
                style={"color": "#555", "fontSize": "0.9rem"},
            ),
            html.Div(id="sc-table"),
        ]
    )


@callback(
    Output("sc-figure", "figure"),
    Output("sc-table", "children"),
    Output("sc-headline", "children"),
    Input("sc-group", "value"),
    Input("sc-metric", "value"),
)
def _update(group, metric):
    bench = load_result("benchmark")
    fig = figures.cost_performance(bench, group, metric)
    b = bench[bench["group"] == group].sort_values(metric, ascending=False)
    tbl = table(
        b,
        {
            "setup": "Sensors",
            "model": "Model",
            "early_auc": "AUC days 1-4",
            "early_ci_low": "CI low",
            "early_ci_high": "CI high",
            "full_auc": "AUC days 1-12",
            "onset_day": "Onset day",
            "placebo_auc": "Placebo AUC",
            "bed_effect": "Bed effect",
            "n_deficit": "Deficit plants",
            "n_control": "Control plants",
            "n_features": "Features",
        },
    )
    return fig, tbl, _headline(bench, group)


if __name__ == "__main__":
    run_alone(layout)
