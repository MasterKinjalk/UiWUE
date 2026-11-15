"""Methods & limits: how detection results were checked, and what they cannot tell you. (Jack)"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _jack_common import (
    GROUPS,
    ONE_CONTROL_BED,
    C,
    chip,
    fmt,
    graph,
    header,
    load_result,
    not_available,
    panel,
    readout,
    register_page,
    run_alone,
    segmented,
)
from dash import Input, Output, callback, html

from uiwue import core
from uiwue.detect import figures
from uiwue.stats._design import derived_dir

QUESTION = "How were the detection results checked, and what can they not tell you?"
register_page(__name__, path="/methods", name="Methods & limits", order=90, description=QUESTION)

RULES = [
    ("unit = plant", "Models are trained and tested leave-one-plant-out; intervals resample plants, never runs."),
    ("no date · bed · position", "Models never see date, bed, hour or grid position."),
    ("own baseline", "Every trait is the change from the plant's own median at the same slot over 30 Aug – 2 Sep."),
    ("size baseline", "Every result sits beside projected leaf area alone; stressed plants stop growing."),
    ("placebo first", "The same AUC over 24–29 Aug, all plants watered. Separation there = bed effect, not ranked."),
    (
        "pre-cutoff null",
        (
            "The daily AUC on every watered day (24 Aug – day 0). The robust onset must clear the largest of them, "
            "and a rule that fires on a watered day is flagged 'day effect'."
        ),
    ),
    ("fit inside fold", "Imputation, scaling and fitting use training plants only; a missing sensor is never imputed."),
    ("same plants", "Sensor setups are compared on the plant-days that have all sensors (W1 + W2, 18 plants)."),
]
DEFINITIONS = [
    ("daily value", "Median of the 12 h and 15 h runs that pass the module's QC."),
    ("AUC", "P(a deficit plant is further in the stress direction than a control plant). 0.5 chance, 1 perfect."),
    ("rule onset", "First day (day 0 = 3 Sep 2025) whose 95 % lower bound is above 0.70, that day and the next."),
    ("robust onset", "Rule onset with the bar raised to the largest watered-day AUC (0.5 + its largest distance)."),
    ("early AUC", "AUC on each plant's mean over days 1–4."),
    ("lead time", "Median clip-turgor onset of the imaged deficit clip plants minus the robust onset."),
    ("bed effect", "Placebo AUC whose interval excludes 0.5 and which is at least 0.70 (or at most 0.30)."),
    ("recovery", "(after − end) / (before − end): 1 = back to its own pre-cutoff value, 0 = none."),
]


def _scope():
    df = core.load_traits()
    if df is None:
        return not_available("Trait tables")
    n = {}
    for p, lbl in (("thermal", "thermal"), ("depth", "3D"), ("rgb", "RGB"), ("hsi", "HSI")):
        cols = [c for c in df.columns if c.startswith(p + "_") and pd.api.types.is_numeric_dtype(df[c])]
        sub = df.loc[df[cols].notna().any(axis=1), ["bed", "plant"]].drop_duplicates() if cols else df.iloc[0:0]
        n[lbl] = sub
    cells = []
    for lbl, sub in n.items():
        beds = " · ".join(f"{b} {k}" for b, k in sub.groupby("bed").size().items())
        cells.append(
            html.Div(
                readout(f"plants · {lbl}", str(len(sub)) if len(sub) else "–"),
                title=f"{lbl}: {beds}" if len(sub) else f"{lbl}: no table in derived/ yet",
            )
        )
    ctl = n["thermal"]
    cells += [
        html.Div(readout("controls", str(int((ctl["bed"] == "W1").sum()))), title="W1, the one control bed"),
        html.Div(readout("dates", "24 Aug–22 Sep", size="section"), title="Placebo 24–29 Aug, re-watered 15 Sep 18:35"),
        html.Div(readout("baseline", "30 Aug–2 Sep", size="section"), title="Each plant's own median, same slot"),
    ]
    return html.Div(cells, className="uw-readouts uw-readouts-3")


def _limits():
    onsets, truth = load_result("onsets"), load_result("truth_link")
    th = None
    if onsets is not None and "null_auc_max" in onsets:
        t = onsets[(onsets["group"] == "all") & (onsets["signal"] == "thermal_leaf_t")]
        th = t.iloc[0] if len(t) else None
    rho = None
    if truth is not None and "rho_within_day_deficit" in truth:
        r = truth[truth["signal"] == "thermal_leaf_air_dt"]
        rho = r.iloc[0] if len(r) else None
    sym = onsets[onsets["group"] == "all"].iloc[0] if onsets is not None and "n_symptom" in onsets else None
    items = [
        ("One control bed", "W1", ONE_CONTROL_BED, True),
        (
            "Beds differ on watered days",
            f"AUC {fmt(th['null_auc_max'])}" if th is not None else "–",
            (
                "Leaf temperature already separates the beds on 2–3 Sep, before any plant was short of water "
                "(sun, doors, pots). Hence the robust onset; the contract rule's day 2 is not used."
            ),
            True,
        ),
        ("W3", "thermal only", "W3 has no RGB or depth table: RGB and 3D results are W2 only, not replicated.", True),
        ("Hyperspectral", "no table", "No hyperspectral trait table in derived/ yet; nothing is claimed for it.", True),
        (
            "Symptom dates",
            f"{int(sym['n_symptom'])} / {int(sym['n_imaged'])} plants" if sym is not None else "upper bound",
            (
                "Notes on 2, 6, 8, 10, 12, 15 Sep only, so a date is an upper bound; most imaged deficit plants "
                "have no note. Lead times use the leaf clip instead."
            ),
            True,
        ),
        (
            "Clip vs thermal",
            f"ρ {fmt(rho['rho_within_day_deficit'])}" if rho is not None else "–",
            "Within-day Spearman of leaf − air with the clip shift, deficit plants only"
            + (
                f" ({int(rho['n_plants_deficit'])} plants); {fmt(rho['rho_within_day'])} with the 3 clip controls."
                if rho is not None
                else "."
            ),
            False,
        ),
        (
            "Air sensors at 15 h",
            "+2.9 °C",
            "W3-end sensor reads warmer; the W1-end sensor is used for every bed.",
            False,
        ),
        ("Doors closed", "~2 Sep", "Inside the baseline window; changed the microclimate for every bed.", False),
        ("Aphids", "W2 A6 · A8", "Black bean aphid noted with wilting.", False),
        ("Missing runs", "~6 %", "5 Sep 06 h and 22 Sep 12–18 h for all cameras; depth from 19 Sep 15 h.", False),
    ]
    rows = []
    for key, val, tip, caveat in items:
        rows.append(
            html.Div(
                [
                    html.Span(key, style={"flex": "1 1 auto", "fontWeight": 700}),
                    html.Span(
                        val,
                        style={
                            "fontWeight": 700,
                            "fontVariantNumeric": "tabular-nums",
                            "color": C["amber-900"] if caveat else C["ink-display"],
                        },
                    ),
                ],
                title=tip,
                className="uw-sand" if caveat else "uw-gray",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "minHeight": "36px",
                    "padding": "0 12px",
                    "borderRadius": "8px",
                },
            )
        )
    return html.Div(rows, style={"display": "flex", "flexDirection": "column", "gap": "6px"})


def _split():
    card = lambda kicker, bed, ctl, tint: html.Div(
        [
            html.Span(kicker, className="uw-kicker", style={"color": C["slate-600"]}),
            html.Div(bed, className="uw-readout"),
            html.Span(f"controls {ctl}", style={"fontSize": "13px", "color": C["slate-600"]}),
        ],
        className=f"uw-tile uw-{tint}",
        style={"flex": "1 1 0", "display": "flex", "flexDirection": "column", "gap": "4px"},
    )
    return html.Div(
        [
            html.Div(
                [
                    card("tune", "W2", "A2 · J5 · R7 · B3", "gray"),
                    card("check, unchanged", "W3", "A8 · R1 · Q6", "mint"),
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                title="Thresholds and settings are picked on W2 and checked unchanged on W3; onsets are reported "
                "for W2 and W3 apart.",
            ),
            html.Div(
                [chip(k, v, "rule") for k, v in RULES], style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "12px"},
    )


def _run_info():
    f = derived_dir() / "detect" / "run_info.json"
    if not f.exists():
        return []
    i = json.loads(f.read_text())
    return [
        chip(
            "run " + str(i.get("finished", "?"))[:10],
            f"Computed {i.get('finished', '?')} on {i.get('plants', '?')} plants and {len(i.get('traits', []))} "
            f"traits with {i.get('n_boot', '?')} bootstrap resamples (python -m uiwue.detect.run).",
            "context",
        )
    ]


def layout(**_):
    pb = load_result("placebo")
    head = header("Methods & limits", QUESTION, *_run_info())
    placebo = (
        not_available("Placebo")
        if pb is None or pb.empty
        else graph("ml-placebo", "Placebo AUC per trait, 24 to 29 Aug, all plants watered")
    )
    grid = html.Div(
        [
            panel(
                "Placebo",
                placebo,
                hover="AUC before any plant was stressed. Sand = beds differ (≤ 0.30 or ≥ 0.70). Circle fine, "
                "triangle day effect (rule fires on a watered day), diamond bed effect.",
                extra=[segmented("ml-group", GROUPS, "all", "Deficit beds")] if pb is not None else [],
            ),
            panel("Split", _split()),
            panel("Scope", _scope()),
            panel("Known limits", _limits()),
            panel(
                "Definitions",
                html.Div(
                    [chip(k, v, "rule") for k, v in DEFINITIONS],
                    style={"display": "flex", "flexWrap": "wrap", "gap": "6px"},
                ),
            ),
            panel(
                "Data",
                html.Div(
                    [
                        html.A(
                            "ADA 2025 · osnaData",
                            href="https://doi.org/10.26249/FK2/DMDLTE",
                            title="Powell, Franz, Smith & Ihorst (2026), doi:10.26249/FK2/DMDLTE, CC BY 4.0",
                        ),
                        html.A(
                            "Companion code",
                            href="https://github.com/pippowell/ada2025",
                            title="github.com/pippowell/ada2025 (MIT)",
                        ),
                    ],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(min(400px, 100%), 1fr))",
            "gap": "16px",
        },
    )
    return html.Div([head, grid])


@callback(Output("ml-placebo", "figure"), Input("ml-group", "value"))
def _placebo(group):
    return figures.placebo_figure(load_result("placebo"), group)


if __name__ == "__main__":
    run_alone(layout)
