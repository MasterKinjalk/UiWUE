"""Methods & limits: how results were validated and what they cannot tell you. (Jack)"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _jack_common import header, load_result, register_page, run_alone, table
from dash import Input, Output, callback, dcc, html

from uiwue.detect import figures
from uiwue.stats._design import derived_dir

register_page(__name__, path="/methods", name="Methods & limits", order=90)

RULES = [
    (
        "The plant is the unit.",
        (
            "Models are trained and tested leave-one-plant-out; intervals "
            "come from resampling plants, never images or runs."
        ),
    ),
    (
        "No shortcuts.",
        (
            "Models never see date, bed, hour or grid position. Every deficit plant was "
            "stressed on the same schedule, so the date alone would give the answer."
        ),
    ),
    (
        "Own baseline.",
        (
            "Every trait is the change from that plant's median at the same time of day "
            "over 30 Aug - 2 Sep, the four days before irrigation stopped."
        ),
    ),
    (
        "Beat plant size.",
        ("Stressed plants stop growing. Every result is shown beside a model that uses projected plant area alone."),
    ),
    (
        "Placebo first.",
        (
            "The same analysis runs on 24-29 Aug, before any plant was stressed. A "
            "trait or model that separates the beds then is flagged 'bed effect' and is not ranked."
        ),
    ),
    (
        "Tune on W2, confirm on W3.",
        (
            "Thresholds and chosen settings are picked on W2 with controls "
            "A2 J5 R7 B3, then checked unchanged on W3 with controls A8 R1 Q6. Onset is reported for W2 "
            "and W3 separately."
        ),
    ),
    ("Fit inside the fold.", "Imputation, scaling and model fitting use training plants only."),
]

DEFINITIONS = [
    ("Daily value", "median of the 12 h and 15 h runs that pass the module's QC."),
    (
        "AUC",
        (
            "probability that a random deficit plant's daily value is further in the stress "
            "direction than a random control plant's. 0.5 = chance, 1 = perfect separation."
        ),
    ),
    (
        "Onset day",
        (
            "first day after the cutoff (3 Sep 2025 = day 0) whose 95% lower bound is above "
            "0.70, on that day and the next scored day."
        ),
    ),
    ("Early AUC", "AUC on each plant's mean over days 1-4."),
    (
        "Lead time",
        (
            "first written symptom (median over the group's plants), or the onset day of RGB "
            "greenness, minus the signal's onset day."
        ),
    ),
    ("Bed effect", ("placebo AUC whose interval excludes 0.5 and which is at least 0.70 (or at most 0.30).")),
]

LIMITS = [
    "Few plants: 7 watered controls in the core set, so intervals are wide. They are shown as they are.",
    (
        "One stress episode in one greenhouse and one variety (CALLEDIA KWS, BBCH ~14 at cutoff). Lead "
        "times are for this drying rate and will differ in the field."
    ),
    (
        "The greenhouse doors were closed on 2 Sep, the last baseline day, which changed the microclimate "
        "for every bed. Leaf - air temperature uses the W1-end air sensor for all beds; the W3-end sensor "
        "reads about 2.9 C warmer at 15 h."
    ),
    (
        "W2 has no air sensor. The W3 leaf clips show no midday dip even before the cutoff, which "
        "looks like a hardware artefact."
    ),
    (
        "Hyperspectral is not co-registered with the other cameras, and the 06 h and 18 h RGB and "
        "hyperspectral runs are often dark, so colour and spectral traits use 09-15 h."
    ),
    (
        "Hyperspectral and some modalities cover fewer plants than thermal; sensor setups are compared "
        "on the plants each one has, listed with every result."
    ),
    (
        "About 6% of runs are missing (all cameras on 5 Sep 06 h and 22 Sep 12-18 h; depth from "
        "19 Sep 15 h). Re-runs within 1 h stand in for the scheduled run."
    ),
    (
        "Neighbouring plants are in view and leaves overlap from about BBCH 12-14. RGB traits use a "
        "centre crop and an excess-green mask, not a per-plant segmentation."
    ),
    (
        "The fine-tuned image-encoder track (DOFA, ImageBind) in the original plan is deferred: it "
        "needs raw cubes and a GPU that the deadline did not allow."
    ),
]


def _run_info() -> html.Div:
    f = derived_dir() / "detect" / "run_info.json"
    if not f.exists():
        return html.Div()
    i = json.loads(f.read_text())
    return html.P(
        f"Results computed {i.get('finished', '?')} on {i.get('plants', '?')} plants and "
        f"{len(i.get('traits', []))} traits, {i.get('n_boot', '?')} bootstrap resamples.",
        style={"color": "#555"},
    )


def layout(**_):
    pb = load_result("placebo")
    bench = load_result("benchmark")
    parts = [
        header("Methods & limits", "How were the results checked, and what can they not tell you?"),
        _run_info(),
        html.H4("Rules every result follows"),
        html.Ol([html.Li([html.Strong(t + " "), d]) for t, d in RULES]),
        html.H4("Placebo: did the beds differ before any plant was stressed?"),
    ]
    if pb is None or pb.empty:
        parts.append(html.P("Placebo results are not available yet.", style={"color": "#777"}))
    else:
        parts += [
            dcc.RadioItems(
                [{"label": g, "value": g} for g in ("all", "W2", "W3") if g in set(pb["group"])],
                "all",
                id="ml-group",
                inline=True,
            ),
            dcc.Graph(id="ml-placebo"),
            html.P(
                "Red points and points outside the dotted lines at 0.30 and 0.70 separated the "
                "beds before the cutoff; their later results are not ranked.",
                style={"color": "#555", "fontSize": "0.9rem"},
            ),
        ]
    if bench is not None and not bench.empty:
        b = bench[bench["group"] == "all"]
        parts += [
            html.H5("Placebo for the sensor models"),
            table(
                b,
                {
                    "setup": "Sensors",
                    "model": "Model",
                    "placebo_auc": "Placebo AUC",
                    "placebo_ci_low": "CI low",
                    "placebo_ci_high": "CI high",
                    "bed_effect": "Bed effect",
                },
                page_size=10,
            ),
        ]
    parts += [
        html.H4("Definitions"),
        html.Ul([html.Li([html.Strong(t + ": "), d]) for t, d in DEFINITIONS]),
        html.H4("Known limits"),
        html.Ul([html.Li(x) for x in LIMITS]),
        html.H4("Data"),
        html.P(
            [
                "ADA 2025: Powell, Franz, Smith & Ihorst (2026), osnaData, ",
                html.A("doi:10.26249/FK2/DMDLTE", href="https://doi.org/10.26249/FK2/DMDLTE"),
                ", CC BY 4.0. Companion code: ",
                html.A("github.com/pippowell/ada2025", href="https://github.com/pippowell/ada2025"),
                " (MIT).",
            ]
        ),
    ]
    return html.Div(parts)


@callback(Output("ml-placebo", "figure"), Input("ml-group", "value"))
def _placebo(group):
    return figures.placebo_figure(load_result("placebo"), group)


if __name__ == "__main__":
    run_alone(layout)
