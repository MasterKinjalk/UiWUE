"""Early warning: which signal detects water deficit first, and how long before symptoms. (Jack)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _jack_common import answer, load_result, not_available, register, run_alone, table
from dash import Input, Output, callback, dcc, html

from uiwue.detect import figures

register(__name__, path="/early-warning", name="Early warning", order=4)

GROUP_OPTIONS = [
    {"label": "All deficit plants (W2 + W3)", "value": "all"},
    {"label": "W2 only", "value": "W2"},
    {"label": "W3 only (replication)", "value": "W3"},
]


def _headline(onsets, group="all") -> str:
    o = onsets[(onsets["group"] == group) & ~onsets["bed_effect"] & ~onsets["is_size_baseline"]]
    o = o.dropna(subset=["onset_day"]).sort_values(["onset_day", "early_auc"],
                                                   ascending=[True, False])
    if o.empty:
        return "No signal has passed the onset bar yet for this group."
    best = o.iloc[0]
    lead = best.get("lead_vs_symptom")
    tail = f", {lead:.0f} days before the first visible symptom" if pd.notna(lead) else ""
    return (f"{best['label']} ({best['sensor']}) is the first signal to separate stressed "
            f"from watered plants: day {int(best['onset_day'])} after irrigation stopped{tail}.")


def layout(**_):
    auc, onsets = load_result("auc_by_day"), load_result("onsets")
    head = [html.H2("Early warning"),
            answer("Which signal detects water deficit first, and how many days before the "
                   "plant visibly wilts?")]
    if onsets is None or auc is None or onsets.empty:
        return html.Div(head + [not_available("No onset results in derived/detect/.")])
    traits = onsets[onsets["kind"] == "trait"].drop_duplicates("signal")
    options = [{"label": figures.signal_name(r), "value": r["signal"]} for _, r in traits.iterrows()]
    return html.Div(head + [
        html.Div(id="ew-headline", style={"fontSize": "1.05rem", "marginBottom": "1rem"}),
        html.Div([
            html.Div([html.Label("Deficit plants"),
                      dcc.RadioItems(GROUP_OPTIONS, "all", id="ew-group", inline=True)],
                     style={"flex": "1"}),
            html.Div([html.Label("Signals"),
                      dcc.Dropdown(options, figures.default_signals(onsets), id="ew-signals",
                                   multi=True)],
                     style={"flex": "2"}),
        ], style={"display": "flex", "gap": "2rem", "flexWrap": "wrap"}),
        dcc.Graph(id="ew-curves"),
        html.P("Each line is the AUC between deficit and watered plants on that day "
               "(1 = perfectly separated, 0.5 = no better than chance), on each plant's change "
               "from its own 30 Aug - 2 Sep baseline. Bands are 95% intervals from resampling "
               "plants. A star marks onset: the first day the band's lower edge clears 0.70 on two "
               "scored days in a row. The dashed grey line is plant size alone; a signal only "
               "counts if it beats it. Dotted lines separated the beds before any plant was "
               "stressed (bed effect) and are not ranked.",
               style={"color": "#555", "fontSize": "0.9rem"}),
        html.H4("Onset day per signal"),
        dcc.Graph(id="ew-ranking"),
        html.Div(id="ew-table"),
    ])


@callback(
    Output("ew-curves", "figure"), Output("ew-ranking", "figure"),
    Output("ew-table", "children"), Output("ew-headline", "children"),
    Input("ew-group", "value"), Input("ew-signals", "value"),
)
def _update(group, signals):
    auc, onsets = load_result("auc_by_day"), load_result("onsets")
    curves = figures.auc_curves(auc, onsets, signals or [], group)
    ranking = figures.onset_ranking(onsets, group)
    o = onsets[(onsets["group"] == group) & (onsets["kind"] == "trait")]
    o = o.sort_values(["onset_day", "early_auc"], ascending=[True, False], na_position="last")
    tbl = table(o, {
        "label": "Signal", "sensor": "Sensor", "onset_day": "Onset day",
        "early_auc": "AUC days 1-4", "lead_vs_symptom": "Days before symptom",
        "lead_vs_visible": "Days before RGB", "placebo_auc": "Placebo AUC",
        "bed_effect": "Bed effect", "n_deficit": "Deficit plants", "n_control": "Control plants",
    }, id="ew-datatable")
    return curves, ranking, tbl, _headline(onsets, group)


if __name__ == "__main__":
    run_alone(layout)
