"""Methods & limits: how detection results were checked, and what they cannot tell you. (Jack)

The one page where sentences are on screen: every limit is one short entry, its consequence in plain words, under an
anchor (``/methods#<anchor>``, see ``_jack_common.METHODS_ANCHORS``) that chips and hovers on other pages link to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import components
import pandas as pd
from _jack_common import (
    C,
    chip,
    day_text,
    fmt,
    graph,
    header,
    kept_signal,
    load_result,
    next_steps,
    not_available,
    panel,
    readout,
    register_page,
    run_alone,
    run_info,
    run_time,
    segmented,
    stat,
)
from dash import Input, Output, callback, clientside_callback, dcc, html

from uiwue import core
from uiwue.detect import figures

QUESTION = "How were the detection results checked, and what can they not tell you?"
register_page(__name__, path="/methods", name="Methods & limits", order=90, description=QUESTION)

GROUPS = {"W2": "W2", "W3": "W3", "all": "W2 + W3"}
KEY_LIMITS = 4  # the first entries of ``_limits``: one control bed, the bed effect, W3's cut leaves, few RGB plants

# How results were checked: (anchor or None, title, what it means)
RULES = [
    ("plant-unit", "One plant, one vote", "Models are tested on plants they never saw; intervals resample plants."),
    ("blind-models", "No date, bed or position", "Models never see the date, bed, hour or grid position."),
    ("own-baseline", "Own baseline", "Every value is the change from the plant's own 30 Aug – 2 Sep median."),
    ("size-baseline", "Size beside every result", "Plant size alone is shown next to each result: the floor."),
    (
        "watered-check",
        "Watered-day check first",
        "The same score on 24–29 Aug, all watered; separation there is flagged.",
    ),
    ("same-plants", "Same plants", "Sensor setups are compared on the 18 plants that have all three sensors."),
    (
        "fit-in-fold",
        "Fit inside the fold",
        "Scaling and fitting use training plants only; a missing sensor is never filled.",
    ),
]
# Words used on the detect pages, in plain terms; the technical name in brackets
WORDS = [
    (
        "detection-score",
        "Detection score (AUC)",
        "How well a signal separates stressed from watered plants: 0.5 = chance, 1 = perfect.",
    ),
    ("onset", "Onset", "First day the score beats the largest score of any watered day, that day and the next."),
    ("contract-rule", "Contract rule", "The plan's simpler onset: the 95 % lower bound above 0.70 two days running."),
    ("early-score", "Early score", "The score on each plant's mean over days 1–4."),
    ("lead-time", "Lead time", "Days between a signal's onset and the leaf clip's turgor onset (the ground truth)."),
    (
        "recovery",
        "Loss regained",
        (
            "Share of the change undone after re-watering (Early warning): 100 % = back to the plant's own "
            "baseline. 3D canopy reads the same days against each plant's peak."
        ),
    ),
    ("placebo", "Placebo", "The detection score on watered days (24–29 Aug), where it should be 0.5."),
    ("lockbox", "Lock-box", "Plants set aside before any experiment, scored once at the end."),
    ("ngrdi", "Green–red index (NGRDI)", "(green − red) / (green + red) from the RGB camera: falls as leaves pale."),
    ("p75", "Height (p75)", "The height below which 75 % of the plant's 3D points lie."),
]
SENSOR_SHORT = {"Thermal": "thermal", "Depth": "3D", "RGB": "RGB", "Hyperspectral": "HSI"}  # sensor -> scope label
PREFIX = {"thermal": "thermal", "3D": "depth", "RGB": "rgb", "HSI": "hsi"}


def _table_plants() -> dict | None:
    """bed, plant of every plant in each sensor's trait table in derived/ now, by scope label."""
    df = core.load_traits()
    if df is None:
        return None
    n = {}
    for lbl, p in PREFIX.items():
        cols = [c for c in df.columns if c.startswith(p + "_") and pd.api.types.is_numeric_dtype(df[c])]
        n[lbl] = df.loc[df[cols].notna().any(axis=1), ["bed", "plant"]].drop_duplicates() if cols else df.iloc[0:0]
    return n


def _scope(n):
    if n is None:
        return not_available("Trait tables")
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
        html.Div(_dates("dates", "24 Aug", "22 Sep"), title="Placebo 24–29 Aug, re-watered 15 Sep 18:35"),
        html.Div(_dates("baseline", "30 Aug", "2 Sep"), title="Each plant's own median, same slot"),
    ]
    return html.Div(cells, className="uw-readouts uw-readouts-3")


def _dates(label, start, end):
    """A date-range readout that may wrap after the dash, never inside a date: the range is wider than a
    narrow column (phone width)."""
    r = readout(label, f"{start}–{end}".replace(" ", " "), size="section")
    r.children[1].style = {"whiteSpace": "normal"}
    return r


def _w3(onsets, tables) -> tuple[str, str, bool]:
    """The W3 limit from the detect results, not a fixed sentence: which sensors' results include W3
    deficit plants, and why the others do not (no W3 table, or a W3 table newer than the detect run).
    Returns (value, hover, caveat)."""
    if onsets is None or "n_deficit" not in onsets:
        return "–", "No detection results yet.", True
    sensors = [SENSOR_SHORT[s] for s in SENSOR_SHORT if s in set(onsets["sensor"])]
    w3 = {SENSOR_SHORT.get(s, s) for s in onsets.loc[(onsets["group"] == "W3") & (onsets["n_deficit"] > 0), "sensor"]}
    have, rest = [s for s in sensors if s in w3], [s for s in sensors if s not in w3]
    # most W3 plants lost a leaf to the water-content sampling: their 3D canopy shrinks for that reason too
    cut = (
        " W3's 3D results mix drought and sampling: most W3 plants had a leaf cut for relative water content on one "
        "of days 1–12 (3D canopy compares W1 with W2 only)."
        if "3D" in have
        else ""
    )
    if not rest:
        return " · ".join(have), "W3 deficit plants are in every sensor's detection results." + cut, bool(cut)
    run = f"the detect run of {when}" if (when := run_time(run_info())) else "the detect run"
    why = []
    for s in rest:
        t = tables.get(s) if tables else None
        k = int((t["bed"] == "W3").sum()) if t is not None else 0
        if k:
            why.append(f"{s}: {k} W3 plants in its table, none in {run} (rerun python -m uiwue.detect.run)")
        else:
            why.append(f"{s}: no W3 table")
    tip = f"W3 results: {', '.join(have) or 'none'}. " + "; ".join(why) + ". Those results are W2 only, not replicated."
    return (" · ".join(have) + " only") if have else "none", tip + cut, True


def plants(bed, *ids):
    """'W2 A6 · A8': the bed once, then each plant's ID as a link to its profile (components.plant_link)."""
    out = [f"{bed} "]
    for i, p in enumerate(ids):
        out += [" · "] if i else []
        out.append(components.plant_link(bed, p, text=p))
    return out


def entry(anchor, title, value, text, caveat=False, hover=None, extra=()):
    """One limit or term: its title and value, then what it means in one short sentence. The title links to the
    entry itself, so its address can be copied."""
    head = [html.B(html.A(title, href=f"#{anchor}", className="dt-self") if anchor else title)]
    if value:
        head.append(html.Span(value, className="dt-entry-value"))
    return html.Li(
        [html.Div(head, className="dt-entry-head"), html.Span(text, className="dt-entry-text"), *extra],
        id=anchor,
        title=hover,
        className="dt-entry uw-sand" if caveat else "dt-entry",
    )


def _limits(tables=None):
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
    rgb = None
    if onsets is not None:
        r = onsets[(onsets["group"] == "W2") & (onsets["sensor"] == "RGB") & (onsets["n_deficit"] > 0)]
        rgb = r.iloc[0] if len(r) else None
    w3, w3_tip, _ = _w3(onsets, tables)
    items = [
        entry(
            "one-control-bed",
            "One control bed",
            "W1",
            "Bed and treatment cannot be separated: a score of 1.00 means the beds differ that day, not certainty.",
            True,
            "Intervals resample plants, but plants in a bed share the day's light and air, so the intervals are too "
            "narrow. The watered-day checks test the beds before the cut.",
        ),
        entry(
            "bed-effect",
            "Beds differ on watered days",
            f"score {fmt(th['null_auc_max'])}" if th is not None else "–",
            "Leaf temperature already separates the beds on 2–3 Sep. Onsets must beat that, and signals that "
            "separate the beds when all were watered are flagged and not ranked.",
            True,
            "Sun, doors and pots differ by bed. 'Bed effect': the placebo score's interval excludes 0.5 and is at "
            "least 0.70 (or at most 0.30). 'Day effect': the contract rule fires on a watered day.",
        ),
        entry(
            "w3-sampled",
            "W3 leaves sampled",
            w3,
            "Most W3 plants lost a leaf for water content, so their 3D canopy loss mixes drought and sampling. Pages "
            "default to W2 against W1.",
            True,
            w3_tip,
        ),
        entry(
            "small-rgb",
            "Few RGB plants",
            f"{int(rgb['n_deficit'])} · {int(rgb['n_control'])} plants" if rgb is not None else "–",
            "RGB and the same-plants comparison rest on 18 plants: wide intervals, one plant can move a score.",
            True,
            "RGB was imaged on W1 and W2 only; W3 has no RGB.",
        ),
        entry(
            "symptom-notes",
            "Symptom notes",
            f"{int(sym['n_symptom'])} / {int(sym['n_imaged'])} plants" if sym is not None else "–",
            "Written on six dates only, and most plants have none: too coarse for a lead time, so the leaf clip is "
            "the ground truth.",
            True,
            "Notes on 2, 6, 8, 10, 12 and 15 Sep: each date is an upper bound.",
        ),
        entry(
            "hyperspectral",
            "Hyperspectral",
            "no table",
            "No hyperspectral trait table yet, so nothing is claimed for it outside the fusion models.",
            True,
        ),
        entry(
            "clip-vs-thermal",
            "Leaf clip vs thermal",
            f"ρ {fmt(rho['rho_within_day_deficit'])}" if rho is not None else "–",
            "Within a day the two rank deficit plants only weakly alike: the lead time compares onset days, not values.",
            False,
            "Within-day Spearman of leaf − air with the clip shift, deficit plants only"
            + (
                f" ({int(rho['n_plants_deficit'])} plants); {fmt(rho['rho_within_day'])} with the 3 clip controls."
                if rho is not None
                else "."
            ),
        ),
        entry(
            "missing-runs",
            "Missing runs",
            "~6 %",
            "5 Sep 06 h and 22 Sep 12–18 h are missing for all cameras; 3D ends 19 Sep 15 h, so 3D recovery is read "
            "on 18–19 Sep.",
        ),
        entry(
            "microclimate",
            "Microclimate",
            "+2.9 °C",
            "The W3-end air sensor reads warmer at 15 h, so the W1-end sensor serves every bed. Doors were closed "
            "around 2 Sep, inside the baseline.",
        ),
        entry("aphids", "Aphids", plants("W2", "A6", "A8"), "Black bean aphid noted with wilting on two W2 plants."),
    ]
    # the four limits that bound every claim lead the page as cards; the rest follow as a list
    return html.Ul(items[:KEY_LIMITS], className="dt-entries dt-key"), html.Ul(
        items[KEY_LIMITS:], className="dt-entries"
    )


LOCKBOX_MODELS = (  # model in the table -> row label
    ("fusion_v2", "Fusion v2"),
    ("fusion_v1dev", "Fusion v1"),
    ("logistic", "Benchmark"),
    ("size_only", "Size only"),
)


def _lockbox():
    """The fusion lock-box: each model's early score and rule onset on the held-out plants, from the table."""
    lb = load_result("fusion_lockbox_benchmark")
    if lb is None or lb.empty:
        return _lockbox_entry(None, not_available("Fusion lock-box"))
    lb = lb[(lb["group"] == "all") & (lb["plants"] == "all")] if "plants" in lb.columns else lb[lb["group"] == "all"]
    rows, v2_day = [], None
    for model, label in LOCKBOX_MODELS:
        r = lb[(lb["model"] == model) & lb["sensors"].isin(["thermal", "size"])]
        if r.empty:
            continue
        r = r.iloc[0]
        day0 = pd.notna(r["onset_day"]) and r["onset_day"] <= 0
        if model == "fusion_v2":
            v2_day = r["onset_day"]
        own = r.get("placebo_auc_main")
        tip = (
            f"{label}: early score {fmt(r['early_auc'])} (95 % CI {fmt(r['early_ci_low'])}–{fmt(r['early_ci_high'])}), "
            f"rule onset {day_text(r['onset_day'])}"
            + (f", own score on 24–29 Aug {fmt(own)}" if pd.notna(own) else "")
            + f"; {int(r['n_deficit'])} deficit · {int(r['n_control'])} control plants."
        )
        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(fmt(r["early_auc"]), className="num"),
                    html.Td(
                        day_text(r["onset_day"]),
                        className="num",
                        style={"color": C["amber-900"], "fontWeight": 700} if day0 else None,
                    ),
                ],
                title=tip,
            )
        )
    if not rows:
        return _lockbox_entry(None, not_available("Fusion lock-box"))
    tbl = html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th("Model"), html.Th("Score", className="num"), html.Th("Rule onset", className="num")]
                    )
                ),
                html.Tbody(rows),
            ],
            className="uw-table",
        ),
        className="uw-scroll",
        style={"paddingTop": "8px"},
    )
    return _lockbox_entry(v2_day, tbl)


def _lockbox_entry(v2_day, body):
    """The lock-box limit; its anchor stays when the table is missing (the empty state in its place)."""
    return entry(
        "fusion-lockbox",
        "Fusion on held-out plants",
        day_text(v2_day) if v2_day is not None else "–",
        "On 45 plants kept out of every experiment, fusion v2 keeps its score but passes the rule on day 0, before "
        "any deficit: a bed difference, not detection.",
        True,
        "45 plants (15 per bed) set aside before any fusion v2 experiment and scored once after the design was frozen. "
        "One thermal camera; early score over days 1–4; onset is the contract rule.",
        extra=[body],
    )


def _split():
    """Where thresholds were set and where they were checked: two stat tiles, the control plants as links."""
    return html.Div(
        [
            stat("Tune", "W2", sub=["controls ", *plants("W1", "A2", "J5", "R7", "B3")], tint="gray"),
            stat("Check, unchanged", "W3", sub=["controls ", *plants("W1", "A8", "R1", "Q6")], tint="mint"),
        ],
        className="uw-tiles",
        style={"paddingTop": "4px"},
        title="Thresholds and settings are picked on W2 and checked unchanged on W3; onsets are reported for W2 and "
        "W3 apart.",
    )


def _run_info():
    i = run_info()
    if not i:
        return []
    traits = i.get("traits", [])
    left_out = sum(not kept_signal(t) for t in traits)
    return [
        chip(
            "run " + str(i.get("finished", "?"))[:10],
            f"Computed {run_time(i) or '?'} on {i.get('plants', '?')} plants and {len(traits)} traits"
            + (f" ({left_out} noisy 3D traits not shown, {len(traits) - left_out} shown)" if left_out else "")
            + f" with {i.get('n_boot', '?')} bootstrap resamples (python -m uiwue.detect.run).",
            "context",
        )
    ]


JUMP = (("limits", "Limits"), ("checks", "Checks"), ("words", "Words"), ("scope", "Scope"))


def layout(**_):
    pb, tables = load_result("placebo"), _table_plants()
    head = header("Methods & limits", QUESTION, *_run_info())
    placebo = (
        not_available("Placebo")
        if pb is None or pb.empty
        else graph("ml-placebo", "Detection score per trait on 24 to 29 Aug, all plants watered")
    )
    checks = html.Section(
        [
            html.Div(html.H2("Checks"), className="uw-panel-head"),
            html.Ol([entry(a, t, "", x) for a, t, x in RULES], className="dt-entries dt-steps"),
            _split(),
        ],
        id="checks",
        className="uw-panel",
    )
    words = html.Section(
        [
            html.Div(html.H2("Words"), className="uw-panel-head"),
            html.Ul([entry(a, t, "", x) for a, t, x in WORDS], className="dt-entries dt-words"),
        ],
        id="words",
        className="uw-panel",
    )
    key, rest = _limits(tables)
    lead = html.Section(key, id="limits", className="dt-lead", **{"aria-label": "Limits"})
    limits = html.Section(
        [
            html.Div(html.H2("More limits"), className="uw-panel-head"),
            rest,
            html.Ul(_lockbox(), className="dt-entries", style={"paddingTop": "8px"}),
        ],
        className="uw-panel",
    )
    watered = panel(
        "Watered-day check",
        placebo,
        hover="Each trait's detection score on 24–29 Aug, before any plant was short of water: it should sit near "
        "0.5. Sand: the beds already differ (≤ 0.30 or ≥ 0.70). Circle fine, triangle day effect, diamond bed effect.",
        extra=[segmented("ml-group", GROUPS, "W2", "Deficit bed")] if pb is not None else [],
    )
    scope = html.Section(
        [
            html.Div(html.H2("Scope"), className="uw-panel-head"),
            _scope(tables),
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
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "paddingTop": "16px"},
            ),
        ],
        id="scope",
        className="uw-panel dt-scope",
    )
    jump = html.Nav(
        [html.A(t, href=f"#{a}", className="uw-btn uw-btn-outline") for a, t in JUMP],
        className="dt-jump",
        **{"aria-label": "On this page"},
    )
    return html.Div(
        [
            head,
            dcc.Store(id="ml-hash", data=1),
            jump,
            lead,
            html.Div([limits, html.Div([watered, checks], className="dt-col")], className="dt-groups"),
            html.Div([words, scope], className="dt-wide"),
            next_steps(
                ("Home", "/", "the experiment and its findings"),
                ("Early warning", "/early-warning", "which sensor warns first, and by how many days"),
            ),
        ]
    )


@callback(Output("ml-placebo", "figure"), Input("ml-group", "value"))
def _placebo(group):
    """Placebo score per kept trait (load_result drops the noisy 3D traits), the size-only baseline named
    'Size only' (DESIGN 7.6); automargin keeps the names whole."""
    pb, onsets = load_result("placebo"), load_result("onsets")
    if pb is None:
        return figures.empty_figure("No placebo results")
    if onsets is not None and "is_size_baseline" in onsets:
        size = set(onsets.loc[onsets["is_size_baseline"].fillna(False).astype(bool), "signal"])
        pb = pb.assign(label=pb["label"].where(~pb["signal"].isin(size), components.SIZE_ONLY))
    fig = figures.placebo_figure(pb, group)
    fig.update_yaxes(automargin=True)
    return fig


# an entry opened from another page's chip (/methods#bed-effect) scrolls into view and is outlined once rendered
clientside_callback(
    """
    function(_) {
        var mark = function(n) {
            var h = decodeURIComponent((window.location.hash || '').slice(1));
            if (!h) { return; }
            var el = document.getElementById(h);
            if (!el) { if (n > 0) { setTimeout(function() { mark(n - 1); }, 200); } return; }
            document.querySelectorAll('.dt-hit').forEach(function(e) { e.classList.remove('dt-hit'); });
            if (el.classList.contains('dt-entry')) { el.classList.add('dt-hit'); }
            el.scrollIntoView({block: 'start'});
        };
        if (!window._dtHash) {
            window._dtHash = true;
            window.addEventListener('hashchange', function() { mark(0); });
        }
        setTimeout(function() { mark(20); }, 300);
        return window.dash_clientside.no_update;
    }
    """,
    Output("ml-hash", "data"),
    Input("ml-hash", "data"),
)


if __name__ == "__main__":
    run_alone(layout)
