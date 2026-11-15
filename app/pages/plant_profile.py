"""Plant profile: one plant's story across sensors. Is it stressed, since when, how badly, did it recover?

Top to bottom: four stat tiles (peak stress, first sign, leaf area at the trough, after re-watering), one focal
timeline (any trait, the plant against its bed and the control bed), then one small chart per sensor; a sensor's
title puts it in the timeline.

Deep link: ``/plant-profile?bed=W2&plant=R7`` (optional ``&trait=<column>`` for the timeline and ``&day=<n>``, carried
on to the 3D canopy and Bed map links). Bed and plant use the identifiers of the tables (bed W1/W2/W3, plant = grid
position such as R7). An unknown bed or plant falls back to the default plant.
"""

import dash
import pandas as pd
import profile_views as pv
from components import C, chip, header, next_links, next_steps, select, size_chip, stat
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from uiwue.core import TRUTH_CLIP_ONSET, TRUTH_FIRST_SYMPTOM, read_table

QUESTION = "Is this plant stressed, since when, how badly, and did it recover?"
dash.register_page(__name__, path="/plant-profile", name="Plant profile", order=20, description=QUESTION)

DEFAULT_BED, DEFAULT_PLANT, DEFAULT_TRAIT = "W2", "R7", "thermal_leaf_ref_dt"
# min(…, 100%): with less than 320 px of room the one column shrinks to fit instead of pushing the page sideways
GRID = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(min(320px, 100%), 1fr))", "gap": "12px"}
LEAF_AREA = "depth_leaf_area_est"  # the 3D trait that leads on the 3D canopy page
FUSION_V2 = "detect_fusion_v2_prob"
TROUGH_TEXT, REWATERED_TEXT = pv.span_text(*pv.TROUGH), pv.span_text(*pv.REWATERED)


def resolve(df, bed, plant):
    """The (bed, plant) a link asks for when it exists in the tables, else the default plant."""
    if bed in pv.BEDS and plant in {o["value"] for o in pv.plant_options(df, bed)}:
        return bed, plant
    if (DEFAULT_BED, DEFAULT_PLANT) in set(zip(df.bed, df.plant)):
        return DEFAULT_BED, DEFAULT_PLANT
    return pv.imaged_plants(df)[0]


# ------------------------------------------------------------------ pieces
def treatment_chip(bed):
    if bed == "W1":
        return html.Span(
            "control",
            title="W1: watered throughout",
            className="uw-chip uw-chip-context",
            style={"background": C["sky"], "color": C["slate-600"]},
        )
    return html.Span(
        "water deficit",
        title=f"{bed}: no water from day 0 (3 Sep 11:00) to re-watering on day 12 (15 Sep 18:35)",
        className="uw-chip uw-chip-context",
        style={"background": C["sand"], "color": C["amber-900"]},
    )


def fusion_chip(column):
    """'bed effect' chip on the fusion v2 score, from derived/detect/fusion_benchmark (thermal, all deficit plants):
    its placebo bed model tells the beds apart before irrigation off."""
    if column != FUSION_V2:
        return []
    b = read_table("detect", "fusion_benchmark")
    if b is None:
        return []
    r = b[(b.model == "fusion_v2") & (b.sensors == "thermal") & (b.group == "all") & b.bed_effect.astype(bool)]
    if r.empty:
        return []
    r = r.iloc[0]
    return [
        chip(
            "bed effect",
            f"A model on the same thermal inputs tells the beds apart on watered days 24–29 Aug (score "
            f"{r.placebo_auc:.2f}). Read the level after day 2, not the first rise.",
            anchor="bed-effect",
        )
    ]


def chips_of(column):
    return [*pv.bed_effect_chip(column), *fusion_chip(column)]


def sensor_card(df, bed, plant, tag, column, label, day=None):
    """One sensor's small chart with its trough number (days 10–12); the title puts the trait in the timeline."""
    state, why = pv.sensor_state(df, bed, plant, column)
    if state != "ok" and column == FUSION_V2 and column in df:
        state = "notable"
        why = f"Fusion v2: no out-of-fold score for {bed} {plant} (a held-out lock-box plant, or no thermal runs)"
    if state != "ok":
        return missing(tag, label, state, why)
    fig = pv.small_multiple(df, bed, plant, column, day=day)
    if fig is None:  # runs, but no daily change to draw: never an empty chart
        return missing(tag, label, "noscan", pv.no_chart_reason(column))
    value, n = pv.window_mean(df, bed, plant, column, pv.TROUGH)
    shown, unit = pv.value_text(column, value)
    when = f"Mean of days 10–12, the last dry days ({n} with a value)" if n else "No value on days 10–12"
    return html.Section(
        [
            html.Div(
                [
                    html.Span(tag, className="uw-kicker pl-tag"),
                    html.Button(
                        label,
                        id={"type": "pp-pick", "col": column},
                        className="pl-pick",
                        title=pv.method_hover(column),
                    ),
                    *chips_of(column),
                    html.Span([shown, html.Small(unit)], className="uw-readout pl-value", title=when),
                ],
                className="pl-card-head",
            ),
            pv.graph_box(fig, f"{label}, {bed} {plant}, change from baseline by day, against the control bed", 150),
        ],
        className="uw-panel pl-card",
        **{"aria-label": label},
    )


def missing(tag, label, kind, hover):
    """A sensor without a chart for this plant: its mark and kind of nothing, one line (DESIGN 3)."""
    return html.Span(
        [pv.empty_mark(kind), html.B(label), html.Span(pv.EMPTY_TEXT[kind])],
        className="pl-missing-item",
        title=f"{tag}: {hover}" if tag else hover,
        **{"aria-label": f"{label}: {pv.EMPTY_TEXT[kind]}"},
    )


def multiples(df, bed, plant, day=None):
    panels = [sensor_card(df, bed, plant, tag, col, label, day) for tag, col, label in pv.HEADLINES]
    if not any(c.startswith("hsi_") for c in df.columns):
        panels.append(missing("HSI", "Spectrum", "notable", "hyperspectral camera: no table in derived/spectral/ yet"))
    ok = [p for p in panels if isinstance(p, html.Section)]
    empty = [p for p in panels if not isinstance(p, html.Section)]
    return [html.Div(ok, style=GRID), html.Div(empty, className="pl-missing")]


def when(d):
    """Days since irrigation off for a date in the tables, or None when missing."""
    if d is None or (not isinstance(d, str) and pd.isna(d)):
        return None
    return (pd.Timestamp(d) - pd.Timestamp(pv.CUTOFF)).days


def signs(row):
    """[(day, short label, hover)] of the plant's own ground-truth onsets: leaf-clip shift and first written symptom."""
    if row is None:
        return []
    out = []
    clip = when(row[TRUTH_CLIP_ONSET]) if bool(row.truth_has_clip) else None
    if clip is not None:
        out.append((clip, "leaf clip", "Leaf-clip turgor shift onset (derived/truth/plants)"))
    symptom = when(row[TRUTH_FIRST_SYMPTOM])
    if symptom is not None:
        note = row.truth_first_symptom_note if isinstance(row.truth_first_symptom_note, str) else ""
        out.append((symptom, "symptom", f"First written symptom{': ' + note if note else ''} (derived/truth/plants)"))
    return sorted(out)


def leaf_area_hover(bed, plant, n_trough, n_after):
    """Method of the leaf-area numbers, and the −20 % rule's caveat for W1 and W2 plants (it also flags controls)."""
    how = (
        f"3D leaf area against the plant's own baseline: mean of days 10–12 ({n_trough} usable), as on 3D canopy; "
        f"re-watered: days 15–16 ({n_after} usable)."
    )
    s = pv.structure_summary(bed, plant)
    if s is None or not s["has_baseline"]:
        return how
    bar, first = pv.fmt(s["threshold"], "%", 0), s["first_below"]
    rule = f"Below {bar} two days running from day {first}" if first is not None else f"Never below {bar}"
    alarms, n_control = s["alarms"]
    if alarms:
        rule += f"; that rule also flags {alarms} of {n_control} controls, so not an alert"
    return f"{how} {rule}."


def tiles(df, bed, plant):
    """The answer in four numbers: how high stress got, the first sign, leaf area at the trough, after re-watering."""
    tint = "sky" if bed == "W1" else "sand"
    s = pv.stress_summary(df, bed, plant)
    if s is None or s["peak"][0] is None:
        peak = stat("Peak stress", "–", tint=tint, hover="detection model: no score for this plant")
    else:
        (pk, pk_day), _ = s["peak"], s["now"]
        peak = stat(
            "Peak stress",
            f"{pk * 100:.0f}",
            "%",
            tint,
            "Highest daily stress probability while dry (days 0–12), detection model. Source: detect/scores_runs",
            sub=pv.day_text(pk_day).replace("Day", "day"),
            chips=[size_chip()],
        )

    row = pv.plant_row(bed, plant)
    found = signs(row)
    if found:
        day, label, _ = found[0]
        first = stat(
            "First sign",
            f"day {day}".replace("-", "−"),
            pv.day_text(day).split(" · ")[1],
            tint,
            " · ".join(f"{h}: {pv.day_text(d)}" for d, _, h in found),
            sub=label,
        )
    else:
        clip = row is not None and bool(row.truth_has_clip)
        first = stat(
            "First sign",
            "–",
            tint=tint,
            hover=("leaf clip: no turgor shift" if clip else "no leaf clip on this plant")
            + "; no written symptom (derived/truth/plants)",
        )

    state, why = pv.sensor_state(df, bed, plant, LEAF_AREA)
    trough, n_t = pv.window_mean(df, bed, plant, LEAF_AREA, pv.TROUGH) if state == "ok" else (None, 0)
    after, n_a = pv.window_mean(df, bed, plant, LEAF_AREA, pv.REWATERED) if state == "ok" else (None, 0)
    hover = leaf_area_hover(bed, plant, n_t, n_a) if state == "ok" else why
    if state == "ok" and trough is None:
        hover = f"No leaf-area change on {TROUGH_TEXT}: no baseline, or no QC-pass daytime run. {hover}"
    leaf = stat(
        "Leaf area, trough",
        *pv.value_text(LEAF_AREA, trough),
        tint,
        hover,
        sub=f"{TROUGH_TEXT.split(' · ')[0].lower()} · vs baseline",
        chips=pv.bed_effect_chip(LEAF_AREA),
    )

    stress_after, _ = pv.window_mean(df, bed, plant, pv.STRESS, pv.REWATERED) if pv.STRESS in df else (None, 0)
    p_val, p_unit = pv.value_text(pv.STRESS, stress_after)
    l_val, l_unit = pv.value_text(LEAF_AREA, after)
    rewatered = stat(
        "After re-watering",
        None,
        tint=tint,
        hover="Mean of days 15–16, after re-watering: stress probability, and leaf area against the plant's own "
        "baseline (3D canopy reads the bed against each plant's peak instead)",
        sub=f"{REWATERED_TEXT.split(' · ')[0].lower()} · vs baseline",
        pairs=[(p_val, f"{p_unit} stress".strip()), (l_val, f"{l_unit} leaf area".strip())],
    )
    out = [peak, first, leaf, rewatered]
    rwc, rwc_hover = pv.rwc_state(bed, plant)
    if rwc != "–":  # the RWC leaves come from plants that were not imaged, so this is rare
        out.append(stat("Leaf water", rwc.removesuffix(" %"), "%", tint, rwc_hover, sub="RWC"))
    return out


def actions(df, bed, plant, day=None):
    """The next steps, at the page's foot: this plant in 3D (when it has a 3D table), then which sensor warns first;
    without 3D, this plant in its bed first."""
    early = (
        "Early warning",
        pv.page_href("/early-warning", bed, plant, day),
        "which sensor warns first, and by how many days",
    )
    if pv.has_depth(df, bed, plant):
        return next_links(
            (
                "3D canopy",
                pv.threed_href(bed, plant, day),
                "this plant's point clouds, leaf area, height and volume against its own baseline",
            ),
            early,
        )
    return next_links(("Bed map", pv.bedmap_href(bed, plant, day), "this plant in its bed"), early)


# ------------------------------------------------------------------ page
def layout(bed=None, plant=None, trait=None, day=None, **_):
    df = pv.traits()
    if df is None:
        return html.Div(
            [
                header("Plant profile", QUESTION),
                pv.empty_panel("", "Run traits", "notable", "no traits_runs table in derived/ yet"),
            ]
        )
    bed, plant = resolve(df, bed, plant)
    day = pv.day_arg(day)
    options = pv.trait_choices(df)
    values = {o["value"] for o in options}
    trait = trait if trait in values else DEFAULT_TRAIT if DEFAULT_TRAIT in values else options[0]["value"]
    prev_, next_ = pv.neighbours(df, bed, plant)
    nav = lambda target, sym, name: html.A(  # a real link with an aria-label (dcc.Link takes none)
        sym,
        id=f"pp-{name}",
        href=pv.profile_href(*target, day),
        className="uw-icon-btn",
        title=f"{name} plant",
        **{"aria-label": f"{name} plant"},
    )
    head = header(
        "Plant profile",
        QUESTION,
        html.Div(id="pp-chips", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        select("pp-bed", {b: b for b in pv.BEDS}, bed, "Bed"),
        select("pp-plant", {o["value"]: o["label"] for o in pv.plant_options(df, bed)}, plant, "Plant"),
        html.Div([nav(prev_, "‹", "previous"), nav(next_, "›", "next")], className="uw-stepper"),
    )
    focal = html.Section(
        [
            html.Div(
                [
                    html.Div(
                        dcc.Dropdown(options, trait, id="pp-trait", clearable=False, searchable=True),
                        className="pl-trait",
                        title="trait in the timeline",
                    ),
                    html.Div(id="pp-detail-chips", className="pl-chips"),
                    html.Div(className="uw-spacer"),
                    html.Button(
                        "CSV",
                        id="pp-csv",
                        className="uw-btn uw-btn-outline",
                        title="Download every run row of this plant, all columns with a value",
                    ),
                ],
                className="uw-panel-head",
            ),
            html.Div(id="pp-detail"),
        ],
        className="uw-panel",
        **{"aria-label": "Timeline"},
    )
    by_sensor = html.Section(
        [html.Div(html.H2("By sensor"), className="uw-panel-head pl-section-head"), html.Div(id="pp-multiples")],
        **{"aria-label": "By sensor"},
    )
    return html.Div(
        [
            head,
            dcc.Store(id="pp-day", data=day),
            dcc.Download(id="pp-download"),
            html.Div(
                [html.Div(id="pp-tiles", className="uw-tiles pl-tiles"), focal, by_sensor],
                className="pl-stack",
            ),
            next_steps(id="pp-actions"),
        ]
    )


@callback(
    Output("pp-plant", "options"),
    Output("pp-plant", "value"),
    Input("pp-bed", "value"),
    State("pp-plant", "value"),
)
def pick_bed(bed, plant):
    """Plants of the chosen bed; keeps the current plant when the bed has it (so a deep link survives the load)."""
    options = pv.plant_options(pv.traits(), bed)
    values = [o["value"] for o in options]
    if plant in values:
        return options, plant
    clips = [o["value"] for o in options if o["label"].endswith("★")]
    return options, (clips or values or [None])[0]


@callback(
    Output("pp-chips", "children"),
    Output("pp-tiles", "children"),
    Output("pp-actions", "children"),
    Output("pp-multiples", "children"),
    Output("pp-previous", "href"),
    Output("pp-next", "href"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
    State("pp-day", "data"),
)
def show_plant(bed, plant, day=None):
    df = pv.traits()
    if df is None or plant is None or plant not in {o["value"] for o in pv.plant_options(df, bed)}:
        return (no_update,) * 6
    row = pv.plant_row(bed, plant)
    chips = [treatment_chip(bed)]
    if row is not None and bool(row.truth_has_clip):
        chips.append(chip("leaf clip", "this plant carries a leaf-clip turgor sensor (★ in the picker)", "rule"))
    prev_, next_ = pv.neighbours(df, bed, plant)
    return (
        chips,
        tiles(df, bed, plant),
        actions(df, bed, plant, day),
        multiples(df, bed, plant, day),
        pv.profile_href(*prev_, day),
        pv.profile_href(*next_, day),
    )


@callback(
    Output("pp-trait", "value"),
    Input({"type": "pp-pick", "col": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def pick_trait(clicks):
    """A sensor's title puts its trait in the timeline (a re-drawn grid fires this with no click: ignored)."""
    hit = ctx.triggered_id
    if not hit or not any(clicks or []) or not (ctx.triggered and ctx.triggered[0]["value"]):
        return no_update
    return hit["col"]


@callback(
    Output("pp-detail", "children"),
    Output("pp-detail-chips", "children"),
    Input("pp-bed", "value"),
    Input("pp-plant", "value"),
    Input("pp-trait", "value"),
    State("pp-day", "data"),
)
def show_detail(bed, plant, trait, day=None):
    df = pv.traits()
    if df is None or plant is None or trait is None:
        return no_update, no_update
    state, why = pv.sensor_state(df, bed, plant, trait)
    chips = [*chips_of(trait), chip("method", pv.method_hover(trait), "rule", "own-baseline")]
    if state != "ok":
        return pv.empty_panel(pv.source_of(trait)[0], pv.label_text(trait), state, why), chips
    events = [(d, label) for d, label, _ in signs(pv.plant_row(bed, plant))]
    fig = pv.detail_figure(df, bed, plant, trait, events=events)
    if fig is None:
        return pv.empty_panel(pv.source_of(trait)[0], pv.label_text(trait), "noscan", pv.no_chart_reason(trait)), chips
    pv.mark_day(fig, day)
    aria = f"{pv.label_text(trait)}, {bed} {plant} against the control bed and its own bed, day −10 to 19"
    return pv.graph_box(fig, aria, 340), chips


@callback(
    Output("pp-download", "data"),
    Input("pp-csv", "n_clicks"),
    State("pp-bed", "value"),
    State("pp-plant", "value"),
    prevent_initial_call=True,
)
def download(_, bed, plant):
    df = pv.traits()
    if df is None or plant is None:
        return no_update
    rows = pv.plant_csv_frame(df, bed, plant)
    return dcc.send_data_frame(rows.to_csv, f"uiwue_{bed}_{plant}.csv", index=False)
