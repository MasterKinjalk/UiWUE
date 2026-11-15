"""3D canopy: how much canopy the water-deficit plants lost, what came back after re-watering, plant by plant in 3D.

One selection drives the page: the plant (navy everywhere) and the day drawn as NOW. Deep link
``/threed?bed=W2&plant=O8&day=12`` (day = days since irrigation off; ``date=2025-09-15`` works too); the page
writes the selection back into the address bar. The old ``/threed-browser``, ``/threed-compare`` and
``/threed-export`` links land here with their query (one page route with a path template, no nav entry).
Without a link it opens on ``threed_canopy.default_plant()`` (a rule, not a fixed plant).

Numbers are computed live from the committed 3D table (``threed_canopy``): bed tiles from ``headline()`` (W1
against W2; W3 plants are shown one by one, never pooled), onsets from derived/detect (marked when older than
the table), the plant card from the run the NOW panel draws, as change from the plant's own baseline. Clouds
are the local cache; a missing cloud shows the empty state and the trait charts still render. Figures are in
``threed_figures``.
"""

from __future__ import annotations

from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import threed_canopy as tc
import threed_figures as tf
from components import (
    C,
    bed_effect_chip,
    chip,
    day_label,
    header,
    mark,
    next_links,
    next_steps,
    not_available,
    page_href,
    plant_link,
    segmented,
    select,
    size_chip,
    stat,
)
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from uiwue.core import all_plants

QUESTION = (
    "How much canopy did the water-deficit plants lose, what came back after re-watering, "
    "and what does that look like for one plant?"
)
dash.register_page(
    __name__,
    path="/threed",
    path_template="/threed<legacy>",  # also /threed-browser, -compare, -export: old links keep working
    name="3D canopy",
    order=25,
    description=QUESTION,
)

METHOD = (
    "W1 control against W2 deficit; W3 plants are shown one by one, never pooled. Day value: mean of the plant's "
    "QC-pass runs at 9–18 h, as % change from its median on days −4 to −1. Peak days 1–3, trough 10–12, re-watered "
    "15–16. Intervals: 2,000 plant resamples. The clouds are one QC-pass run each: NOW the day's most typical run, "
    "THEN the baseline run closest to the baseline."
)
ONE_BED = "One control bed and one deficit bed: treatment and bed cannot be separated."
QC_ONLY = "QC-pass runs only: W2's failed runs are its smaller plants (days 6–11), so the loss is understated."
TINT = {
    "W1": ("sky", "CONTROL", "slate-600"),
    "W2": ("sand", "WATER DEFICIT", "amber-900"),
    "W3": ("sand", "WATER DEFICIT", "amber-900"),
}
STATE_TEXT = {
    "ok": "",
    "no file": "QC-pass run, but its cloud is not on this machine (data/derived/threed/clouds is local)",
    "sparse": "QC-pass run, but its cloud is too sparse to draw (under 100 plant points or under half the run's)",
    "no scan": "no daytime run passed QC",
}
SLOPE_PX, GAP_PX = 270, 200  # the collapse chart and the gap chart under it
NEXT = (  # the page's next steps, at its foot: (text, path, hover); they carry the selection
    ("Early warning", "/early-warning", "which sensor warns first, and by how many days"),
    ("Plant profile", "/plant-profile", "every sensor for this plant"),
)


# ---------------------------------------------------------------- selection
@lru_cache(maxsize=4)
def _plants(key) -> tuple:
    r = tc.runs(tc.SHOWN_BEDS)
    have = set(zip(r["bed"], r["plant"], strict=True))
    return tuple((b, p) for b in tc.SHOWN_BEDS for p in all_plants() if (b, p) in have)


def plants() -> tuple:
    """(bed, plant) of every plant in the 3D table (W1, W2 and W3), bed by bed in grid order."""
    key = tc.table_key()
    return () if key is None else _plants(key)


def beds() -> dict:
    """{bed: bed} for the bed picker: the beds the 3D table has, W1, W2, W3 in that order."""
    have = {b for b, _ in plants()}
    return {b: b for b in tc.SHOWN_BEDS if b in have}


def plant_options(bed: str) -> dict:
    return {p: p for b, p in plants() if b == bed}


def resolve(bed, plant) -> tuple[str, str] | None:
    """The (bed, plant) asked for when the table has it, else the page default (a W2 responder with clouds)."""
    every = plants()
    if (bed, plant) in every:
        return bed, plant
    best = tc.default_plant()
    if best in every:
        return best
    w = tc.windows()
    if w is not None:  # no clouds on this machine: the W2 plant that lost most height
        h = w[(w["trait"] == "height_p75") & (w["bed"] == tc.DEFICIT_BED)].sort_values("peak_to_trough")
        if len(h):
            return tc.DEFICIT_BED, str(h["plant"].iloc[0])
    return every[0] if every else None


def asked_day(day, date) -> int | None:
    """The day of a link: ``day`` (days since irrigation off) or ``date`` (YYYY-MM-DD, old browser links)."""
    try:
        if day not in (None, ""):
            return int(float(day))
        if date not in (None, ""):
            return (pd.Timestamp(date) - pd.Timestamp(tc.core.CUTOFF)).days
    except (TypeError, ValueError):
        return None
    return None


def state_for(bed: str, plant: str, want: int | None = None) -> dict:
    """Page state of a plant: its THEN run (``threed_canopy.then_run``: the baseline-day run closest to its
    baseline) and the NOW day (asked for and snapped to a drawable day, else the plant's default)."""
    now = tf.snap_day(tf.strip_days(bed, plant), want) if want is not None else tc.now_day(bed, plant)
    then = tc.then_run(bed, plant)
    return dict(
        bed=bed,
        plant=plant,
        day=now,
        then=None if then is None else then["day"],
        then_slot=None if then is None else then["slot"],
    )


# ---------------------------------------------------------------- pieces
def tiles() -> list:
    """The page's answer, computed live: height and leaf area lost (peak → trough), what came back, onset."""
    h = tc.headline()
    if h is None:
        return [not_available("3D traits")]
    out = []
    effects = tc.bed_effects()
    for trait, label in (("height_p75", "Height lost"), ("leaf_area_est", "Leaf area lost")):
        d, c = h[trait]["peak_to_trough"]["deficit"], h[trait]["peak_to_trough"]["control"]
        out.append(
            stat(
                label,
                tf.pct(d["mean"], 0, ""),
                "%",
                "sand",
                f"{tc.TRAITS[trait].label}, days 10–12 against each plant's own days 1–3: W2 mean {tf.pct(d['mean'])} "
                f"(95 % CI {tf.pct(d['ci_low'], 0, '')} to {tf.pct(d['ci_high'])}, {d['n']} plants, from "
                f"{tf.pct(d['min'], 0, '')} to {tf.pct(d['max'])}); W1 control {tf.pct(c['mean'])}. {QC_ONLY}",
                sub=["control ", html.B(tf.pct(c["mean"]))],
                chips=trait_chips(trait, effects),
            )
        )
    hr = h["height_p75"]["peak_to_rewatered"]["deficit"]
    ar = h["leaf_area_est"]["peak_to_rewatered"]["deficit"]
    out.append(
        stat(
            "After re-watering",
            None,
            tint="sand",
            hover=(
                f"Days 15–16 against each plant's own peak (days 1–3), W2: height comes back ({tf.pct(hr['mean'])}, "
                f"95 % CI {tf.pct(hr['ci_low'], 0, '')} to {tf.pct(hr['ci_high'])}), leaf area does not "
                f"({tf.pct(ar['mean'])}, {tf.pct(ar['ci_low'], 0, '')} to {tf.pct(ar['ci_high'])}). Early warning "
                "reads the same event as the share of the loss regained."
            ),
            sub="vs each plant's peak",
            pairs=[(tf.pct(hr["mean"], 0, ""), "% height"), (tf.pct(ar["mean"], 0, ""), "% leaf area")],
        )
    )
    out.append(onset_tile())
    return out


def _after(day: float, ref: float, what: str) -> str:
    """'1 day after thermal', '2 days before thermal', 'same day as thermal'."""
    gap = round(float(day) - float(ref))
    if gap == 0:
        return f"same day as {what}"
    n = abs(gap)
    return f"{n} day{'s' if n != 1 else ''} {'after' if gap > 0 else 'before'} {what}"


def onset_tile():
    """First seen: the robust onset of 3D leaf area against thermal, with the size-only baseline (DESIGN 7.6);
    every onset of the strip that used to sit under the charts is on hover (Early warning has the full table)."""
    o = tc.onsets()
    if o is None or o.empty:
        return not_available("Detection onsets")
    leaf = o[o["trait"] == "leaf_area_est"]
    if leaf.empty or not np.isfinite(leaf["robust_onset_day"].iloc[0]):
        return stat("First seen", "–", "", "gray", "no robust onset for leaf area in derived/detect/onsets")
    r = leaf.iloc[0]
    thermal = o[(o["sensor"] == "Thermal") & ~o["is_size_baseline"]]
    t = thermal.iloc[0] if len(thermal) and np.isfinite(thermal["robust_onset_day"].iloc[0]) else None
    sub = _after(r["robust_onset_day"], t["robust_onset_day"], "thermal") if t is not None else None
    every = "; ".join(
        f"{tc.TRAITS[x.trait].label if isinstance(x.trait, str) else tf.short_label(str(x.label))} "
        f"({'size' if x.is_size_baseline else x.sensor}) day {x.robust_onset_day:g}"
        for x in o.itertuples()
        if np.isfinite(x.robust_onset_day)
    )
    return stat(
        "First seen",
        f"day {r['robust_onset_day']:g}",
        day_label(r["robust_onset_day"])[1],
        "gray",
        f"Leaf area first separates W2 from W1 on day {r['robust_onset_day']:g} (the Early warning onset rule, "
        f"{int(r['n_deficit'])} deficit and {int(r['n_control'])} control plants; single runs day {r['slot_min']:g}–{r['slot_max']:g}). "
        f"All: {every}.",
        sub=sub,
        chips=[*onset_chips(), size_chip()],
        href=page_href("/early-warning"),
    )


def onset_chips() -> list:
    """'older table' when derived/detect was computed before the 3D table on this page was re-run (its n on hover)."""
    info = tc.onsets_info()
    if info is None or not info["stale"]:
        return []
    return [
        chip(
            "older table",
            f"The onsets come from derived/detect, written {info['written']:%d %b %H:%M} from an earlier 3D table: "
            f"n = {info['n_deficit']} deficit and {info['n_control']} control plants, not the {info['table_deficit']} "
            f"W2 and {info['table_control']} W1 plants behind the tiles and charts. Refresh: python -m uiwue.detect.run",
        )
    ]


def bed_effect_text(trait: str, e: dict) -> str:
    """Why the bed checks flag a trait, in one line: the placebo, the E2 growth check or both."""
    parts = []
    if e["placebo"]:
        parts.append(f"score {e['placebo_auc']:.2f} on watered days 24–29 Aug")
    if e["e2"]:
        p = "p < 0.001" if e["e2_p"] < 0.001 else f"p = {e['e2_p']:.3f}"
        unit = e["e2_unit"].replace("cm2", "cm²").replace("cm3", "cm³")
        parts.append(f"W2 grew {abs(e['e2_coef']):.1f} {unit} {'slower' if e['e2_coef'] < 0 else 'faster'} ({p})")
    return f"{tc.TRAITS[trait].label}: " + ", ".join(parts)


def trait_bed_chip(trait: str, effects: dict | None = None):
    e = (effects or tc.bed_effects())[trait]
    return bed_effect_chip(bed_effect_text(trait, e) + ".")


def every_trait_flagged(effects: dict) -> bool:
    return all(effects[k]["chip"] for k in tc.TRAITS)


def trait_chips(trait: str, effects: dict) -> list:
    """The ``bed effect`` chip of one trait (DESIGN 7.4), unless every trait on the page is flagged: then the one
    chip in the header covers them all (``header_chips``), as a chip on every tile and chart would say nothing."""
    return [trait_bed_chip(trait, effects)] if effects[trait]["chip"] and not every_trait_flagged(effects) else []


def header_chips(effects: dict) -> list:
    """At most two: ``bed effect`` (one control bed; the beds differed before day 0, per flagged trait, when every
    trait is flagged) and ``method`` (how the numbers are made, and the QC caveat)."""
    flagged = [bed_effect_text(k, effects[k]) for k in tc.TRAITS if effects[k]["chip"]]
    if every_trait_flagged(effects):
        caveat = bed_effect_chip(
            f"{ONE_BED} So every value is change from the plant's own baseline. " + "; ".join(flagged) + "."
        )
    else:
        caveat = chip("1 control bed", ONE_BED, anchor="one-control-bed")
    return [caveat, chip("method", f"{METHOD} {QC_ONLY}", "rule", "own-baseline")]


def context_chips(bed: str, plant: str) -> list:
    """For a W3 plant the chip that says it is left out of every bed number (its treatment is in the card)."""
    if bed != tc.SAMPLED_BED:
        return []
    cut = tc.sampled_leaf(bed, plant)
    mine = f"This plant: a leaf cut on {tf.day_text(cut)}." if cut is not None else "This plant: no leaf cut."
    return [
        chip(
            "not pooled",
            "W3 is left out of every bed number, band and onset on this page: most of its plants had a leaf cut for "
            f"water content on days 1–12, so its loss mixes drought and sampling. {mine}",
            anchor="w3-sampled",
        )
    ]


def phase_of(day: int) -> str:
    if tc.BASELINE_DAYS[0] <= day <= tc.BASELINE_DAYS[1]:
        return "base"
    if tc.STRESS_DAYS[0] <= day <= tc.STRESS_DAYS[1]:
        return "dry"
    return "out"


WHY_RUN = {
    "then": "of the plant's QC-pass runs on the baseline days (−4 to −1), the one closest to its baseline",
    "now": "the day's most typical QC-pass run (closest to the plant's median over that day and the days either side)",
}


def view_head(which: str, day, cloud: dict | None, state: str = "ok"):
    """THEN / NOW head: kicker, day in display type, date, and the run drawn (hover: its numbers)."""
    if day is None:
        return [html.Span(which, className="uw-kicker td-which"), html.Span("–", className="td-day-num")]
    d, date = day_label(day)
    run = (
        html.Span(f"{int(cloud['slot']):02d} h", className="td-run", title=run_hover(which, cloud))
        if cloud is not None
        else html.Span(state, className="td-run")
    )
    return [
        html.Span(which, className="uw-kicker td-which"),
        html.Span(d, className="td-day-num"),
        html.Span(date, className="td-date"),
        html.Span(className="uw-spacer"),
        run,
    ]


def value_text(k: str, value, change) -> str:
    """'leaf area 81 cm² (−67 %)': a run or day value with its change from the plant's baseline."""
    t = tc.TRAITS[k]
    if value is None or not np.isfinite(value):
        return f"{t.label.lower()} –"
    return f"{t.label.lower()} {value:,.0f} {t.unit} ({tf.pct(change)})"


def run_hover(which: str, cloud: dict) -> str:
    """The run drawn: why this one, its own values against the baseline, its points, and the points not drawn."""
    values = "; ".join(value_text(k, cloud.get(f"{k}_run"), cloud.get(f"{k}_run_pct")) for k in tc.TRAITS)
    hidden = (
        f"; {cloud['n_stray']:,} detached edge points not drawn (a neighbour's leaf or debris cut by the crop)"
        if cloud.get("n_stray")
        else ""
    )
    when = f"{int(cloud['slot']):02d} h slot" + (
        f", taken at {int(cloud['capture_hour']):02d} h" if int(cloud["capture_hour"]) != int(cloud["slot"]) else ""
    )
    return (
        f"QC-pass run, {when}: {WHY_RUN.get(which, '')}. {values} vs the plant's baseline. "
        f"{int(cloud['n']):,} plant points{hidden}."
    )


def nothing(kind: str, text: str, hover: str):
    """Empty state of a stage panel: the dashed box with the mark of its kind of nothing (DESIGN 3)."""
    return html.Div(
        [html.Span(className=f"td-nothing td-nothing-{kind}", **{"aria-hidden": "true"}), html.Span(text)],
        className="uw-empty td-empty",
        title=hover,
    )


def strip(state: dict, days: pd.DataFrame):
    """Day strip (DESIGN 5): one cell per day, baseline mint, no water sand, NOW navy; a day without a QC-pass
    daytime run is a dashed 'no scan' cell and cannot be picked."""
    cells = []
    for r in days.itertuples():
        day = int(r.day)
        d, date = day_label(day)
        on = day == state.get("day")
        scan = r.state != "no scan"
        hover = f"{d} · {date}"
        if r.state != "ok":
            hover += f": {STATE_TEXT[r.state]}"
        elif np.isfinite(r.leaf_area_est_pct) and np.isfinite(r.height_p75_pct):
            hover += (
                f": leaf area {tf.pct(r.leaf_area_est_pct)}, height {tf.pct(r.height_p75_pct)} vs baseline "
                f"(day mean of {int(r.n_pass)} QC-pass run{'s' if int(r.n_pass) != 1 else ''})"
            )
        if day == state.get("then"):
            hover += " · then"
        cells.append(
            html.Button(
                [html.B(str(day).replace("-", "−")), html.Span(date)],
                id={"type": "td-cell", "day": day},
                n_clicks=0,
                disabled=not scan,
                className=f"td-cell td-{phase_of(day)}" + (" td-on" if on else "") + ("" if scan else " td-noscan"),
                title=hover,
                **{"aria-pressed": "true" if on else "false", "aria-label": hover},
            )
        )
    return cells


def legend(colour_top: float, days: pd.DataFrame | None = None):
    """Key of the strip and the clouds: baseline and no-water days, 'no scan' only when the strip has one, and
    the height ramp of the points."""
    swatch = lambda cls: html.Span(className=f"uw-swatch {cls}", **{"aria-hidden": "true"})
    noscan = days is None or (days["state"] == "no scan").any()
    return [
        html.Span([swatch("td-sw-base"), "baseline"], title="30 Aug – 2 Sep (days −4 to −1): the plant's own baseline"),
        html.Span(
            [swatch("td-sw-dry"), "no water"],
            title="W2 without water: day 0 (3 Sep 11:00) to re-watering on day 12 (15 Sep 18:35); W1 watered throughout",
        ),
        *([html.Span([mark("no scan"), "no scan"], title="no daytime run passed QC that day")] if noscan else []),
        html.Span(
            [
                "height 0",
                html.Span(
                    className="td-ramp",
                    style={"background": f"linear-gradient(90deg, {C['leaf-300']}, {C['leaf-700']}, {C['leaf-900']})"},
                    **{"aria-hidden": "true"},
                ),
                f"{colour_top:.0f} mm",
            ],
            title="point colour: height above the soil, one range per plant",
            className="td-ramp-key",
        ),
    ]


def next_of(bed: str | None = None, plant: str | None = None, day=None) -> list:
    """The page's next steps at its foot, (text, href, hover); each carries the selection (?bed=&plant=&day=)."""
    return [(t, page_href(path, bed, plant, day), hover) for t, path, hover in NEXT]


def readout_row(label, value, hover, chips=()):
    return html.Div(
        [
            html.Span(label, className="td-trait"),
            *chips,
            html.Span(className="uw-spacer"),
            html.Span([value, html.Small("%")] if value != "–" else value, className="uw-readout"),
        ],
        className="td-readout",
        title=hover,
    )


def _finite(v) -> bool:
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def plant_card(state: dict, days: pd.DataFrame):
    """The selected plant on the NOW day: its three traits as change from its own baseline, read from the run the
    NOW panel draws, so the numbers describe the picture beside them (the day's mean over its QC-pass runs is on
    hover; without a drawable cloud the card shows that mean and says so). No status mark: the leaf-area rule
    (−20 % on two days) also flags watered controls, so it is not shown as a status."""
    bed, plant, day = state["bed"], state["plant"], state.get("day")
    tint, kicker, ink = TINT.get(bed, TINT["W2"])
    effects = tc.bed_effects()
    row = days[days["day"] == day].iloc[0] if day is not None and (days["day"] == day).any() else None
    drawn = row is not None and row["state"] == "ok" and _finite(row.get("slot"))
    rows = []
    for k, t in tc.TRAITS.items():
        value = None if row is None else row.get(f"{k}_run" if drawn else k)
        change = None if row is None else row.get(f"{k}_run_pct" if drawn else f"{k}_pct")
        if not _finite(change):
            why = (
                "no QC-pass daytime run on this day"
                if not _finite(value)
                else "no baseline: no QC-pass daytime run on days −4 to −1"
            )
            rows.append(readout_row(t.label, "–", f"{t.label}: {why}"))
            continue
        source = f"the {int(row['slot']):02d} h run drawn" if drawn else "the day's mean of its QC-pass runs"
        hover = (
            f"{t.label} on {tf.day_text(day)}, {source}: {value:,.0f} {t.unit} against a baseline of "
            f"{row[f'{k}_base']:,.0f} {t.unit} (median of the plant's day means on days −4 to −1)."
        )
        if drawn and _finite(row.get(f"{k}_pct")):
            slots = str(row["slots"]).replace(",", ", ")
            hover += f" Day mean of the QC-pass {slots} h runs: {tf.pct(row[f'{k}_pct'])}."
        rows.append(readout_row(t.label, tf.pct(change, 0, ""), hover, trait_chips(k, effects)))
    head = [
        plant_link(bed, plant, day, "uw-section td-plant-id", f"{bed} {plant}: every sensor, on its profile"),
        html.Span(kicker, className="uw-kicker", style={"color": C[ink]}),
    ]
    when = tf.day_text(day) if day is not None else "–"
    what = f"the {int(row['slot']):02d} h run drawn as NOW" if drawn else "the day mean (no cloud to draw)"
    return html.Div(
        [
            html.Div(head, className="td-card-head"),
            html.Div(
                "vs own baseline",
                className="uw-readout-label td-when",
                title=f"{when}: {what}, as change from the plant's own baseline; each readout's hover has the "
                "day mean over its QC-pass runs",
            ),
            html.Div(rows, className="td-readouts"),
        ],
        className=f"uw-card uw-{tint} td-card",
        **{"aria-label": f"{bed} {plant}"},
    )


def graph(id, label, cls="", height=None, config=None, figure=None):
    style = {"height": "100%"} if height is None else {"height": f"{height}px"}
    extra = {} if figure is None else {"figure": figure}
    return html.Div(
        dcc.Graph(id=id, config=config or tf.GRAPH_CONFIG, style=style, responsive=True, **extra),
        role="img",
        className=cls,
        **{"aria-label": label},
    )


def stage_view(which: str, label: str):
    return html.Section(
        [
            html.Div(id=f"td-{which}-head", className="td-view-head"),
            html.Div(
                [
                    dcc.Loading(
                        graph(
                            f"td-{which}",
                            f"{label}: plant points of one QC-pass run, coloured by height above the soil",
                            "td-graph",
                            config=tf.STAGE_CONFIG,
                        ),
                        type="circle",
                        color=C["teal-700"],
                        delay_show=150,
                        overlay_style={"visibility": "visible"},
                        parent_className="td-loading",
                    ),
                    html.Div(id=f"td-{which}-empty", className="td-empty-wrap"),
                ],
                className="td-canvas",
            ),
        ],
        className=f"td-view td-view-{which}",
        **{"aria-label": label},
    )


def chart_head(title, *extra, hover=None):
    return html.Div([html.H3(title, title=hover), *extra], className="td-chart-head")


def trait_column(key: str, label: str, chips: list):
    """One trait, named once: every plant at peak, trough and re-watered, then the bed gap day by day."""
    return html.Div(
        [
            html.Div(
                [
                    chart_head(label, *chips),
                    graph(
                        f"td-slope-{key}",
                        f"{label} of every plant: peak, trough, re-watered, % of its own peak",
                        height=SLOPE_PX,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        "by day",
                        className="uw-kicker td-sub",
                        title="Deficit minus control (W2 − W1), mean change from each plant's own baseline, in "
                        "percentage points; band: 95 % interval; grey: its range on watered days; dashed: onset",
                    ),
                    graph(f"td-gap-{key}", f"{label}, deficit minus control by day, day −10 to 16", height=GAP_PX),
                ]
            ),
        ],
        className="td-trait-col",
    )


# ---------------------------------------------------------------- page
def layout(bed=None, plant=None, day=None, date=None, legacy="", **_):
    sel = resolve(bed, plant) if tc.runs() is not None else None
    if sel is None:
        return html.Div([header("3D canopy", QUESTION), not_available("3D traits")])
    state = state_for(*sel, asked_day(day, date))
    bed, plant = sel
    effects = tc.bed_effects()
    head_chips = header_chips(effects)
    stepper = html.Div(
        [
            html.Button(
                "‹", id="td-prev", className="uw-icon-btn", title="previous plant", **{"aria-label": "Previous plant"}
            ),
            html.Button("›", id="td-next", className="uw-icon-btn", title="next plant", **{"aria-label": "Next plant"}),
        ],
        className="uw-stepper",
    )
    head = header(
        "3D canopy",
        QUESTION,
        html.Span(context_chips(bed, plant), id="td-context", className="td-context"),
        *head_chips,
        html.Div(
            [
                select("td-bed", beds(), bed, "Bed"),
                select("td-plant", plant_options(bed), plant, "Plant"),
                stepper,
            ],
            className="td-picker",
        ),
    )
    # DESIGN 7.4: a flagged trait carries its chip on every chart that uses it (or the header carries one for all)
    leaf_chip, height_chip = trait_chips("leaf_area_est", effects), trait_chips("height_p75", effects)
    return html.Div(
        [
            head,
            dcc.Store(id="td-state", data=state),
            dcc.Store(id="td-sel", data=dict(bed=bed, plant=plant)),
            dcc.Store(id="td-view"),
            dcc.Store(id="td-link"),
            dcc.Store(id="td-url"),
            dcc.Store(id="td-scroll"),
            html.Div(tiles(), className="uw-tiles uw-lead td-tiles"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                segmented("td-which", {"then": "Then", "now": "Now"}, "now", "Cloud shown"),
                                className="td-toggle",
                            ),
                            html.Div(
                                [
                                    stage_view("then", "Then: the plant on its baseline day"),
                                    stage_view("now", "Now: the chosen day"),
                                ],
                                id="td-views",
                                className="td-views td-show-now",
                            ),
                        ],
                        className="td-stage",
                    ),
                    html.Aside(
                        [
                            html.Div(id="td-card"),
                            html.Section(
                                [
                                    chart_head(
                                        "Side view",
                                        html.Span(id="td-side-key", className="td-side-key"),
                                        hover="Seen from the side: the highest plant point in each 5 mm of x. "
                                        "Then dotted, now filled, on the stage's height range.",
                                    ),
                                    graph(
                                        "td-side",
                                        "Side view: height profile of the plant then and now",
                                        "td-side-graph",
                                    ),
                                ],
                                className="uw-panel td-side",
                            ),
                        ],
                        className="td-rail",
                        **{"aria-label": "Selected plant"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                "‹", id="td-day-prev", className="uw-icon-btn", **{"aria-label": "Previous day"}
                            ),
                            html.Div(
                                id="td-strip", className="td-strip", role="group", **{"aria-label": "Day shown as now"}
                            ),
                            html.Button("›", id="td-day-next", className="uw-icon-btn", **{"aria-label": "Next day"}),
                        ],
                        className="td-days",
                    ),
                    html.Div(id="td-legend", className="uw-legend td-legend"),
                ],
                className="td-hero",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2(
                                "Collapse and recovery",
                                title="Top: every plant against its own peak (days 1–3) at the trough (days 10–12) and "
                                "after re-watering (days 15–16); line the bed's median plant, band its middle half, a dot "
                                "per plant (click one to show it). Below: deficit minus control by day, in percentage "
                                "points of each plant's baseline, with its 95 % band. W1 against W2.",
                            ),
                            html.Span(
                                [
                                    html.Span(
                                        [html.Span(className="td-key td-key-band td-key-dry"), "deficit"],
                                        className="td-key-item",
                                        title="W2: median plant and middle half of the plants; below, W2 − W1",
                                    ),
                                    html.Span(
                                        [html.Span(className="td-key td-key-band td-key-ctrl"), "control"],
                                        className="td-key-item",
                                        title="W1: median plant and middle half of the plants",
                                    ),
                                    html.Span(
                                        [html.Span(className="td-key td-key-sel"), html.Span(id="td-sel-name")],
                                        className="td-key-item",
                                        title="the plant shown above; below, it minus the W1 mean",
                                    ),
                                ],
                                className="td-keys",
                            ),
                        ],
                        className="uw-panel-head",
                    ),
                    html.Div(
                        [
                            trait_column("leaf", "Leaf area", leaf_chip),
                            trait_column("height", "Height", height_chip),
                        ],
                        className="td-pair-charts",
                    ),
                ],
                className="uw-panel td-collapse",
            ),
            next_steps(*next_of(bed, plant, state["day"]), id="td-steps"),
        ],
        className="td-page",
    )


# ---------------------------------------------------------------- callbacks
@callback(
    Output("td-state", "data"),
    Output("td-sel", "data"),
    Output("td-bed", "value"),
    Output("td-plant", "options"),
    Output("td-plant", "value"),
    Input("td-bed", "value"),
    Input("td-plant", "value"),
    Input("td-prev", "n_clicks"),
    Input("td-next", "n_clicks"),
    Input("td-slope-leaf", "clickData"),
    Input("td-slope-height", "clickData"),
    Input({"type": "td-cell", "day": ALL}, "n_clicks"),
    Input("td-day-prev", "n_clicks"),
    Input("td-day-next", "n_clicks"),
    State("td-state", "data"),
    prevent_initial_call=True,
)
def navigate(bed, plant, _prev, _next, click_leaf, click_height, _cells, _dprev, _dnext, state):
    """Every way of changing the selection: bed and plant pickers, plant stepper, a line in the collapse chart,
    a day cell and the day stepper. A new plant opens on its own default NOW day."""
    trig = ctx.triggered_id
    if trig is None or not state:
        return (no_update,) * 5
    every = list(plants())
    cur = (state["bed"], state["plant"])
    target = None
    if trig == "td-bed":
        options = plant_options(bed)
        target = (bed, plant if plant in options else next(iter(options), None))
    elif trig == "td-plant":
        target = (state["bed"] if bed is None else bed, plant)
    elif trig in ("td-prev", "td-next") and cur in every:
        i = every.index(cur)
        target = every[(i + (1 if trig == "td-next" else -1)) % len(every)]
    elif trig in ("td-slope-leaf", "td-slope-height"):
        data = click_leaf if trig == "td-slope-leaf" else click_height
        pt = (data or {}).get("points", [{}])[0]
        custom = pt.get("customdata")
        if custom and len(custom) >= 2 and (custom[0], custom[1]) in every:
            target = (custom[0], custom[1])
    if target is not None:
        if target[1] is None or target == cur:
            return (no_update,) * 5
        new = state_for(*target)
        return new, dict(bed=target[0], plant=target[1]), target[0], plant_options(target[0]), target[1]
    # a day: a cell, or the stepper through the days with a cloud (else with a QC-pass run)
    days = tf.strip_days(*cur)
    if isinstance(trig, dict) and trig.get("type") == "td-cell":
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 5
        day = int(trig["day"])
    elif trig in ("td-day-prev", "td-day-next"):
        ok = days.loc[days["state"] == "ok", "day"]
        pool = sorted(int(d) for d in (ok if len(ok) else days.loc[days["state"] != "no scan", "day"]))
        now = state.get("day")
        if not pool or now is None:
            return (no_update,) * 5
        step = [d for d in pool if (d > now if trig == "td-day-next" else d < now)]
        if not step:
            return (no_update,) * 5
        day = step[0] if trig == "td-day-next" else step[-1]
    else:
        return (no_update,) * 5
    if day == state.get("day"):
        return (no_update,) * 5
    return {**state, "day": day}, no_update, no_update, no_update, no_update


@callback(
    Output("td-then", "figure"),
    Output("td-now", "figure"),
    Output("td-then-head", "children"),
    Output("td-now-head", "children"),
    Output("td-then-head", "className"),
    Output("td-now-head", "className"),
    Output("td-then-empty", "children"),
    Output("td-now-empty", "children"),
    Output("td-legend", "children"),
    Input("td-state", "data"),
    State("td-view", "data"),
)
def show_stage(state, view):
    """THEN and NOW on one scale, one camera and one colour range; the camera keeps the reader's last turn."""
    bed, plant, then, now = state["bed"], state["plant"], state.get("then"), state.get("day")
    days = tf.strip_days(bed, plant)
    a, b = tf.stage_cloud(bed, plant, then, state.get("then_slot")), tf.stage_cloud(bed, plant, now)
    ranges = tc.axis_ranges(bed, plant)
    colour_top = ranges["z"][1] if ranges else tf.frame_height(a, b)
    direction = tf.view_of(view) if view else tf.VIEW
    revision = f"{bed}-{plant}-{then}-{now}"
    figs, empties = [], []
    for cloud, day in ((a, then), (b, now)):
        if cloud is None:
            figs.append(blank_figure())
            empties.append(empty_for(days, day))
        else:
            figs.append(tf.stage_figure(cloud, (a, b), colour_top, direction, revision))
            empties.append(None)
    heads = [view_head("then", then, a, state_of(days, then)), view_head("now", now, b, state_of(days, now))]
    classes = [f"td-view-head td-{phase_of(d)}" if d is not None else "td-view-head" for d in (then, now)]
    return (*figs, *heads, *classes, *empties, legend(colour_top, days))


def state_of(days: pd.DataFrame, day) -> str:
    hit = days[days["day"] == day] if day is not None else days.iloc[0:0]
    return str(hit["state"].iloc[0]) if len(hit) else "no scan"


def empty_for(days: pd.DataFrame, day):
    """A stage panel without a cloud: the shared empty state when the cloud file or the whole local cache is
    missing, the 'no scan' mark when no QC-pass run of the day can be drawn. Never a stand-in cloud."""
    state = state_of(days, day)
    if not tc.clouds_dir().is_dir() or (day is not None and state == "no file"):
        box = not_available("Point clouds")
        box.title = (
            "no cloud cache on this machine: data/derived/threed/clouds is local, not in git"
            if not tc.clouds_dir().is_dir()
            else STATE_TEXT["no file"]
        )
        return box
    if state == "sparse":
        return nothing("scan", "too sparse", STATE_TEXT["sparse"])
    return nothing("scan", "no scan", "no QC-pass daytime run with a cloud on a baseline day")


def blank_figure():
    """An empty canvas under a stage panel's empty state."""
    return dict(
        data=[], layout=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0))
    )


@callback(
    Output("td-strip", "children"),
    Output("td-card", "children"),
    Output("td-side", "figure"),
    Output("td-side-key", "children"),
    Input("td-state", "data"),
)
def show_plant(state):
    bed, plant = state["bed"], state["plant"]
    days = tf.strip_days(bed, plant)
    a = tf.stage_cloud(bed, plant, state.get("then"), state.get("then_slot"))
    b = tf.stage_cloud(bed, plant, state.get("day"))
    frame = tf.frame_height(a, b)
    labels = tuple(
        f"{which} · day {state[key]}".replace("-", "−") if state.get(key) is not None else which
        for which, key in (("then", "then"), ("now", "day"))
    )
    key = [
        html.Span([html.Span(className="td-key td-key-then", **{"aria-hidden": "true"}), labels[0]]),
        html.Span(
            [
                html.Span(
                    className="td-key td-key-now", style={"background": C["leaf-300"]}, **{"aria-hidden": "true"}
                ),
                labels[1],
            ]
        ),
    ]
    return strip(state, days), plant_card(state, days), tf.side_figure(a, b, frame, labels), key


@callback(Output("td-steps", "children"), Input("td-state", "data"))
def show_next(state):
    """The foot links follow the selection, so the next page opens on the same plant and day."""
    return next_links(*next_of(state["bed"], state["plant"], state.get("day")))


@callback(
    Output("td-slope-leaf", "figure"),
    Output("td-slope-height", "figure"),
    Output("td-gap-leaf", "figure"),
    Output("td-gap-height", "figure"),
    Output("td-context", "children"),
    Output("td-sel-name", "children"),
    Input("td-sel", "data"),
)
def show_evidence(sel):
    bed, plant = sel["bed"], sel["plant"]
    o = tc.onsets()
    onset = {}
    if o is not None:
        for k in tc.TRAITS:
            r = o[o["trait"] == k]
            onset[k] = float(r["robust_onset_day"].iloc[0]) if len(r) else None
    return (
        tf.slope_figure("leaf_area_est", bed, plant, height=SLOPE_PX),
        tf.slope_figure("height_p75", bed, plant, height=SLOPE_PX),
        tf.gap_figure("leaf_area_est", bed, plant, onset.get("leaf_area_est"), height=GAP_PX),
        tf.gap_figure("height_p75", bed, plant, onset.get("height_p75"), height=GAP_PX),
        context_chips(bed, plant),
        f"{bed} {plant}",
    )


# the address bar follows the selection (replaceState: no reload, no new history entry); an old /threed-* link
# is renamed to /threed and the shell told once, so the nav marks this page
clientside_callback(
    """
    function(state) {
        if (!state || !state.bed || !state.plant) { return window.dash_clientside.no_update; }
        var q = '?bed=' + encodeURIComponent(state.bed) + '&plant=' + encodeURIComponent(state.plant);
        if (state.day !== null && state.day !== undefined) { q += '&day=' + state.day; }
        var path = window.location.pathname;
        var m = path.match(/^(.*\\/)threed(-browser|-compare|-export)?\\/?$/);
        if (!m) { return window.dash_clientside.no_update; }
        var target = m[1] + 'threed' + q;
        if (window.location.pathname + window.location.search !== target) {
            window.history.replaceState(window.history.state, '', target);
            if (m[2]) { window.dispatchEvent(new CustomEvent('_dashprivate_pushstate')); }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("td-url", "data"),
    Input("td-state", "data"),
)

# THEN and NOW turn together: while one is dragged the other follows (plotly_relayouting), and the one being
# dragged leads, so the follower never pulls back
clientside_callback(
    """
    function(f1, f2) {
        var attempt = function(n) {
            var g = function(id) { var el = document.getElementById(id); return el && el.querySelector('.js-plotly-plot'); };
            var a = g('td-then'), b = g('td-now');
            if (!a || !b || !a.on || !b.on || !a._fullLayout || !b._fullLayout) {
                if (n > 0) { setTimeout(function() { attempt(n - 1); }, 150); }
                return;
            }
            [[a, b], [b, a]].forEach(function(pair) {
                var src = pair[0];
                src.__tdPeer = pair[1];
                if (src.__tdLinked) { return; }
                src.__tdLinked = true;
                var same = function(p, q) {
                    return p && q && ['x', 'y', 'z'].every(function(k) { return Math.abs((p[k] || 0) - (q[k] || 0)) < 1e-6; });
                };
                var copy = function(ev, live) {
                    var lead = window.__tdLead;
                    if (lead && lead !== src) { return; }
                    if (live) {
                        window.__tdLead = src;
                        clearTimeout(window.__tdLeadT);
                        window.__tdLeadT = setTimeout(function() { window.__tdLead = null; }, 400);
                    }
                    var cam = ev && ev['scene.camera'];
                    var peer = src.__tdPeer;
                    if (!cam || !peer || !peer._fullLayout || !peer._fullLayout.scene) { return; }
                    var cur = peer._fullLayout.scene.camera;
                    if (cur && same(cur.eye, cam.eye) && same(cur.center, cam.center) && same(cur.up, cam.up)) { return; }
                    window.Plotly.relayout(peer, {'scene.camera': cam});
                };
                src.on('plotly_relayouting', function(ev) { copy(ev, true); });
                src.on('plotly_relayout', function(ev) { copy(ev, false); });
            });
        };
        attempt(30);
        return window.dash_clientside.no_update;
    }
    """,
    Output("td-link", "data"),
    Input("td-then", "figure"),
    Input("td-now", "figure"),
)

# the last turn is kept, so the next day or plant opens from the same side
clientside_callback(
    """
    function(r1, r2) {
        var trig = (window.dash_clientside.callback_context.triggered || [])[0];
        var r = trig && trig.prop_id.indexOf('td-now') === 0 ? r2 : r1;
        var cam = r && r['scene.camera'];
        if (!cam || !cam.eye) { return window.dash_clientside.no_update; }
        return {eye: cam.eye, center: cam.center || {x: 0, y: 0, z: 0}};
    }
    """,
    Output("td-view", "data"),
    Input("td-then", "relayoutData"),
    Input("td-now", "relayoutData"),
    prevent_initial_call=True,
)

# the NOW cell stays in view when the strip scrolls (tablet and phone)
clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var strip = document.getElementById('td-strip');
            var cell = strip && strip.querySelector('.td-cell.td-on');
            if (!cell || strip.scrollWidth <= strip.clientWidth) { return; }
            strip.scrollLeft = cell.offsetLeft - strip.offsetLeft - (strip.clientWidth - cell.clientWidth) / 2;
        }, 0);
        return window.dash_clientside.no_update;
    }
    """,
    Output("td-scroll", "data"),
    Input("td-strip", "children"),
)

# phone: one cloud at a time, THEN or NOW
clientside_callback(
    """
    function(which) { return 'td-views td-show-' + (which || 'now'); }
    """,
    Output("td-views", "className"),
    Input("td-which", "value"),
)
