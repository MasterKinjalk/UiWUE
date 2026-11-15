"""Home: the experiment at a glance, what each module has extracted, and the way to every page."""

from datetime import date

import dash
import pandas as pd
from components import C, header, not_available
from dash import html

from uiwue.core import BASELINE, CUTOFF, DERIVED, REWATER, catalogue, load_traits

dash.register_page(
    __name__,
    path="/",
    name="Home",
    order=0,
    description="what the pipeline extracted from each sensor, for which plants and dates",
)

SPAN = (date(2025, 8, 12), date(2025, 9, 22))  # the dataset's imaging period
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
    "/threed-browser": ("lilac", "plum-700"),
    "/threed-compare": ("lilac", "plum-700"),
}
CAMERAS = ("thermal_", "depth_", "rgb_", "hsi_")


def pct(d):
    return f"{(pd.Timestamp(d) - pd.Timestamp(SPAN[0])).days / ((SPAN[1] - SPAN[0]).days + 1) * 100:.2f}%"


def width(a, b):
    return f"{((pd.Timestamp(b) - pd.Timestamp(a)).days + 1) / ((SPAN[1] - SPAN[0]).days + 1) * 100:.2f}%"


SENSORS = (  # source tag, name, columns that count as "imaged by this sensor"
    ("IR", "Thermal", "thermal_"),
    ("3D", "Depth", "depth_"),
    ("RGB", "RGB", "rgb_"),
    ("HSI", "Hyperspectral", "hsi_"),
    ("CLIP", "Leaf clip", "truth_turgor_"),
)
WORKFLOWS = (  # path, title, tile tint, disc colour, tag
    ("/irrigation", "Water today", "mint", "forest-700", "IA"),
    ("/early-warning", "Earliest warning", "sky", "teal-700", "EW"),
    ("/plant-profile", "One plant", "lilac", "plum-700", "PP"),
)


def coverage_counts(df, prefix):
    """{bed: plants with a value} for the catalogue columns of one sensor, counted from the run table."""
    cat = set(catalogue().column)
    cols = [c for c in df.columns if c.startswith(prefix) and c in cat and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return None
    have = df[df[cols].notna().any(axis=1)]
    return have.groupby("bed").plant.nunique().to_dict()


def sensor_tiles(df):
    out = []
    for tag, name, prefix in SENSORS:
        n = coverage_counts(df, prefix)
        if n is None:
            out.append(
                html.Div(
                    [
                        html.Div(f"{tag} · {name}", className="uw-readout-label"),
                        html.Div("–", className="uw-stat"),
                        html.Div("no table", className="uw-readout-label"),
                    ],
                    className="uw-tile uw-missing",
                    title=f"{name}: no table in derived/ yet",
                )
            )
            continue
        beds = " · ".join(f"{b} {n.get(b, 0) or '–'}" for b in ("W1", "W2", "W3"))
        out.append(
            html.Div(
                [
                    html.Div(f"{tag} · {name}", className="uw-readout-label"),
                    html.Div([f"{sum(n.values())}", html.Small("plants")], className="uw-stat"),
                    html.Div(beds, className="uw-readout-label"),
                ],
                className="uw-tile uw-gray",
                title=f"{name}: plants with at least one value in the run table, by bed",
            )
        )
    return html.Div(out, className="uw-tiles", **{"aria-label": "Sensor coverage"})


def workflows():
    """The three jobs the app is for, as large links."""
    pages = {p["path"]: p for p in dash.page_registry.values()}
    out = []
    for path, title, tint, disc, tag in WORKFLOWS:
        page = pages.get(path, {})
        out.append(
            html.A(
                [html.Span(tag, className="uw-disc", style={"background": C[disc]}, **{"aria-hidden": "true"}), title],
                href=path,
                title=page.get("description") or page.get("name") or title,
                className=f"uw-pagetile uw-{tint}",
                style={"minHeight": "88px", "fontSize": "21px", "fontFamily": "var(--uw-font-display)"},
            )
        )
    return html.Nav(out, className="uw-pages", style={"paddingTop": 0}, **{"aria-label": "Workflows"})


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
        if has:
            d = pd.to_datetime(t.date)
            track.append(
                html.Span(className="uw-cover-bar", style={"left": pct(d.min()), "width": width(d.min(), d.max())})
            )
            n_plants = t[["bed", "plant"]].drop_duplicates().shape[0] if {"bed", "plant"} <= set(t) else 0
            meta = [html.B(f"{len(t):,}"), html.Span(f"{n_plants} plants")]
            hover = f"{name}: {len(t):,} rows, {d.min():%d %b} – {d.max():%d %b}"
        else:
            meta = [html.B("–"), html.Span("no table")]
            hover = f"{name}: no table in derived/{key}/ yet"
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(className="uw-dot" if has else "uw-dot uw-dot-none"),
                            html.B(name),
                            html.Span(owner),
                        ],
                        className="uw-cover-name",
                    ),
                    html.Div(track, className="uw-cover-track" if has else "uw-cover-track uw-missing", title=hover),
                    html.Div(meta, className="uw-cover-meta"),
                ],
                className="uw-cover-row",
            )
        )
    return html.Section(rows, className="uw-panel uw-cover", **{"aria-label": "Tables"})


def page_tiles():
    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order") or 0)
    out = []
    for p in pages:
        if p["path"] == "/":
            continue
        tint, disc = GROUPS.get(p["path"], ("gray", "slate-600"))
        words = [w for w in p["name"].split() if w[0].isalnum()]
        tag = "3D" if p["name"].startswith("3D") else "".join(w[0] for w in words[:2]).upper()
        out.append(
            html.A(
                [
                    html.Span(tag, className="uw-disc", style={"background": C[disc]}, **{"aria-hidden": "true"}),
                    p["name"],
                ],
                href=p["relative_path"],
                title=p.get("description") or p["name"],
                className=f"uw-pagetile uw-{tint}",
            )
        )
    return html.Nav(out, className="uw-pages", **{"aria-label": "Pages"})


def layout(**_):
    df = load_traits()
    head = header(
        "Sugar beet water deficit · 2025",
        "What did the pipeline extract from each sensor, for which plants and dates?",
    )
    if df is None:
        return html.Div([head, workflows(), not_available("Every module"), page_tiles()])
    gap = {"paddingTop": "20px"}
    return html.Div(
        [head, workflows(), html.Div(sensor_tiles(df), style=gap), html.Div(coverage(), style=gap), page_tiles()]
    )
