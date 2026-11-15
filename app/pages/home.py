"""Home: the experiment in one line, its finding in numbers read live from the tables, and the way to every page."""

from datetime import date

import dash
import numpy as np
import pandas as pd
import threed_canopy as tc
from components import BLURB, NAV, SIZE_ONLY, C, bed_effect_chip, day_label, details, header, not_available, plain, stat
from dash import dcc, html

from uiwue.core import BASELINE, CUTOFF, DERIVED, REWATER, load_traits

QUESTION = "What was the experiment, what did it find, and where do I start?"
dash.register_page(__name__, path="/", name="Home", order=0, description=QUESTION)

SPAN = (date(2025, 8, 12), date(2025, 9, 22))  # the dataset's imaging period
DRY = (REWATER - CUTOFF).days  # day 12: re-watered
MODULES = {  # folder in derived/ -> (name, owner); the main table is its largest *_runs table
    "threed": ("3D structure", "Kinjalk"),
    "thermal": ("Thermal", "Yuming"),
    "env": ("Environment", "Yuming"),
    "diurnal": ("Daily rhythm", "Yuming"),
    "truth": ("Leaf clip", "Sam"),
    "rgb": ("RGB", "Jack"),
    "spectral": ("Hyperspectral", "Sam"),
    "detect": ("Detection", "Jack"),
}
GROUPS = {  # page path -> (tile tint, disc colour)
    "/early-warning": ("sky", "teal-700"),
    "/sensor-comparison": ("sky", "teal-700"),
    "/daily-rhythm": ("mint", "forest-700"),
    "/irrigation": ("mint", "forest-700"),
    "/bed-map": ("mint", "forest-700"),
    "/plant-profile": ("lilac", "plum-700"),
    "/export": ("gray", "slate-600"),
    "/threed": ("mint", "forest-700"),  # DESIGN 2.1: forest-700 is the 3D tile colour
}
SENSOR_TAG = {"Thermal": "Thermal", "Depth": "3D", "RGB": "RGB"}
SENSOR_PAGE = {"Thermal": "/early-warning", "Depth": "/threed", "RGB": "/early-warning"}
BEDS = (  # bed, treatment word, tint, text colour, hover
    ("W1", "control", "sky", "ink-display", "Watered throughout"),
    ("W2", "deficit", "sand", "amber-900", f"No water from day 0 (3 Sep) to re-watering on day {DRY} (15 Sep)"),
    (
        "W3",
        "deficit, leaves cut",
        "sand",
        "amber-900",
        (
            f"No water from day 0 to day {DRY}, as W2; most plants also had a leaf cut for water content, "
            "so the 3D numbers never pool W3"
        ),
    ),
)


def pct(d):
    return f"{(pd.Timestamp(d) - pd.Timestamp(SPAN[0])).days / ((SPAN[1] - SPAN[0]).days + 1) * 100:.2f}%"


def width(a, b):
    return f"{((pd.Timestamp(b) - pd.Timestamp(a)).days + 1) / ((SPAN[1] - SPAN[0]).days + 1) * 100:.2f}%"


def signed_pct(x):
    """'−61' with a real minus sign, '–' when missing."""
    return "–" if x is None or not np.isfinite(x) else f"{x:+.0f}".replace("-", "−")


# ---------------------------------------------------------------- the experiment, in one row
def facts(df):
    """W1 / W2 / W3 with their plant counts (from the trait table) and the no-water window."""
    n = df.groupby("bed").plant.nunique().to_dict() if df is not None and {"bed", "plant"} <= set(df) else {}
    out = [
        html.Span(
            [html.B(bed), f"{word} · {n[bed]} plants" if n.get(bed) else f"{word} · –"],
            title=hover,
            className=f"uw-fact uw-{tint}",
            style={"color": C[ink]},
        )
        for bed, word, tint, ink, hover in BEDS
    ]
    out.append(
        html.Span(
            [html.B("No water"), f"day 0–{DRY} · {CUTOFF.day}–{REWATER.day} Sep"],
            title=f"Irrigation off on 3 Sep 2025 (day 0), re-watered in the evening of 15 Sep (day {DRY}); "
            "baseline 30 Aug – 2 Sep",
            className="uw-fact",
        )
    )
    return html.Div(out, className="uw-facts", **{"aria-label": "The experiment"})


# ---------------------------------------------------------------- the finding, in numbers
def onset_tiles():
    """First day each sensor separates W2 deficit from W1 control (robust onset, detect/onsets via the 3D page's
    selection), earliest first, then the size-only baseline (DESIGN 7.6). Each tile links to its page."""
    o = tc.onsets()
    if o is None or o.empty:
        return [not_available("Detection onsets")]
    o = o[o["robust_onset_day"].notna()]
    out = []
    size = o[o["is_size_baseline"]]
    picks = o[~o["is_size_baseline"] & ~o["bed_effect"] & ~o["fires_before_cutoff"]].drop_duplicates("sensor")
    for r in [*picks.itertuples(), *size.head(1).itertuples()]:
        day = int(r.robust_onset_day)
        is_size = bool(r.is_size_baseline)
        label = SIZE_ONLY if is_size else f"{SENSOR_TAG.get(r.sensor, r.sensor)} sees stress"
        out.append(
            stat(
                label,
                f"day {day}",
                day_label(day)[1],
                "gray",
                hover=(
                    f"{r.label}: W2 deficit plants separate from W1 control from day {day} "
                    f"({int(r.n_deficit)} deficit, {int(r.n_control)} control plants; robust onset, detect/onsets)."
                    + (" The size-only baseline: what plant size alone tells apart." if is_size else "")
                ),
                sub=plain(r.signal, r.label),
                href="/early-warning" if is_size else SENSOR_PAGE.get(r.sensor, "/early-warning"),
            )
        )
    return out


def leaf_tile():
    """Leaf area lost by the deficit plants, peak to trough, against the controls' change (the 3D page's tile)."""
    h = tc.headline()
    if h is None or "leaf_area_est" not in h:
        return not_available("3D traits")
    d, c = (h["leaf_area_est"]["peak_to_trough"][g] for g in ("deficit", "control"))
    e = tc.bed_effects().get("leaf_area_est", {})
    chips = []
    if e.get("chip"):
        why = []
        if e.get("placebo"):
            why.append(f"score {e['placebo_auc']:.2f} on watered days 24–29 Aug")
        if e.get("e2"):
            unit = str(e["e2_unit"]).replace("cm2", "cm²").replace("cm3", "cm³")
            pace = "slower" if e["e2_coef"] < 0 else "faster"
            why.append(f"W2 grew {abs(e['e2_coef']):.1f} {unit} {pace} than W1 before day 0")
        chips = [bed_effect_chip("Leaf area: " + "; ".join(why) + ".")]
    return stat(
        "Leaf area lost",
        signed_pct(d["mean"]),
        "%",
        "sand",
        hover=(
            f"W2 deficit plants, days 10–12 against each plant's own days 1–3: mean {signed_pct(d['mean'])} % over "
            f"{d['n']} plants; W1 control {signed_pct(c['mean'])} % over {c['n']}."
        ),
        sub=f"control {signed_pct(c['mean'])} %",
        chips=chips,
        href="/threed",
    )


# ---------------------------------------------------------------- where to go: the nav's path as tiles
def page_links():
    """One tile per page, in the nav's order: its tag, its name and what the reader learns there."""
    pages = {p["path"]: p for p in dash.page_registry.values()}
    order = [path for _, _, items in NAV for path, _ in items]
    order += [p for p in sorted(pages, key=lambda k: pages[k].get("order") or 0) if p not in order and p != "/"]
    out = []
    for path in order:
        p = pages.get(path)
        if p is None:
            continue
        tint, disc = GROUPS.get(path, ("gray", "slate-600"))
        words = [w for w in p["name"].split() if w[0].isalnum()]
        tag = "3D" if p["name"].startswith("3D") else "".join(w[0] for w in words[:2]).upper()
        out.append(
            dcc.Link(
                [
                    html.Span(tag, className="uw-disc", style={"background": C[disc]}, **{"aria-hidden": "true"}),
                    html.Span([html.B(p["name"]), html.Span(BLURB.get(path, ""), className="uw-pagetile-blurb")]),
                ],
                href=p["relative_path"],
                title=p.get("description") or p["name"],
                className=f"uw-pagetile uw-{tint}",
            )
        )
    return out


def page_tiles():
    """The tiles in the nav's groups (See, Detect, Act, Data), one column each."""
    tiles = {t.href: t for t in page_links()}
    cols = []
    for group, question, items in NAV:
        found = [tiles.pop(path) for path, _ in items if path in tiles]
        if found:
            cols.append(
                html.Div(
                    [html.Span(group, title=question, className="uw-kicker uw-path-head"), *found], className="uw-path"
                )
            )
    if tiles:
        cols.append(
            html.Div([html.Span("More", className="uw-kicker uw-path-head"), *tiles.values()], className="uw-path")
        )
    return html.Nav(cols, className="uw-paths", **{"aria-label": "Pages"})


# ---------------------------------------------------------------- what each table covers
def main_table(folder):
    runs = sorted(folder.glob("*_runs.parquet"), key=lambda p: p.stat().st_size, reverse=True)
    return pd.read_parquet(runs[0]) if runs else None


def coverage():
    """One row per module: the dates its main table covers, rows and plants; a dashed track if it has none."""
    folders = {p.name: p for p in DERIVED.iterdir() if p.is_dir()} if DERIVED.exists() else {}
    names = MODULES | {k: (k, "") for k in folders if k not in MODULES}
    axis = [
        (SPAN[0], f"{SPAN[0].day} {SPAN[0]:%b}"),
        (date(2025, 8, 24), "24 Aug"),
        (CUTOFF, "3 Sep"),
        (REWATER, "15 Sep"),
        (SPAN[1], "22 Sep"),
    ]
    rows = [
        html.Div(
            [
                html.Div(className="uw-cover-name"),
                html.Div([html.Span(t, style={"left": pct(d)}) for d, t in axis], className="uw-cover-axis"),
                html.Div(className="uw-cover-meta"),
            ],
            className="uw-cover-row",
            style={"minHeight": "18px"},
        )
    ]
    for key, (name, owner) in names.items():
        t = main_table(folders[key]) if key in folders else None
        has = t is not None and "date" in t
        track = [
            html.Span(style={"left": pct(BASELINE[0]), "width": width(*BASELINE), "background": C["mint"]}),
            html.Span(style={"left": pct(CUTOFF), "width": width(CUTOFF, REWATER), "background": C["sand"]}),
        ]
        by = f" · {owner}" if owner else ""
        if has:
            d = pd.to_datetime(t.date)
            track.append(
                html.Span(className="uw-cover-bar", style={"left": pct(d.min()), "width": width(d.min(), d.max())})
            )
            pairs = t[["bed", "plant"]].drop_duplicates() if {"bed", "plant"} <= set(t) else None
            n_plants = 0 if pairs is None else len(pairs)
            beds = (
                ""
                if pairs is None
                else " (" + ", ".join(f"{b} {n}" for b, n in pairs.bed.value_counts().sort_index().items()) + ")"
            )
            meta = [html.B(f"{len(t):,}"), html.Span(f"{n_plants} plants")]
            hover = f"{name}{by}: {len(t):,} rows, {n_plants} plants{beds}, {d.min():%d %b} – {d.max():%d %b}"
        else:
            meta = [html.B("–"), html.Span("no table")]
            hover = f"{name}{by}: no table in derived/{key}/ yet"
        rows.append(
            html.Div(
                [
                    html.Div(
                        [html.Span(className="uw-dot" if has else "uw-dot uw-dot-none"), html.B(name)],
                        className="uw-cover-name",
                    ),
                    html.Div(track, className="uw-cover-track" if has else "uw-cover-track uw-missing"),
                    html.Div(meta, className="uw-cover-meta"),
                ],
                className="uw-cover-row",
                title=hover,
            )
        )
    return details(
        html.Span("Data coverage", title="The dates, rows and plants of each module's main table in derived/"),
        html.Section(rows, className="uw-cover", **{"aria-label": "Tables"}),
        id="home-coverage",
    )


def layout(**_):
    df = load_traits()
    head = header("Sugar beet water deficit · 2025", QUESTION)
    finding = html.Div([*onset_tiles(), leaf_tile()], className="uw-tiles uw-home-tiles", **{"aria-label": "Finding"})
    if df is None:
        return html.Div([head, facts(None), finding, not_available("Trait table"), page_tiles()])
    return html.Div([head, facts(df), finding, page_tiles(), coverage()], className="uw-home")
