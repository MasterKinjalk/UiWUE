"""Home: the experiment at a glance, what each module has extracted, and the way to every page."""

from datetime import date

import dash
import pandas as pd
from components import C, header, not_available
from dash import html

from uiwue.core import BASELINE, CUTOFF, DERIVED, REWATER, SLOTS, load_traits

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
    "/threed-browser": ("lilac", "plum-700"),
    "/threed-compare": ("lilac", "plum-700"),
}
CAMERAS = ("thermal_", "depth_", "rgb_", "hsi_")


def pct(d):
    return f"{(pd.Timestamp(d) - pd.Timestamp(SPAN[0])).days / ((SPAN[1] - SPAN[0]).days + 1) * 100:.2f}%"


def width(a, b):
    return f"{((pd.Timestamp(b) - pd.Timestamp(a)).days + 1) / ((SPAN[1] - SPAN[0]).days + 1) * 100:.2f}%"


def tiles(df):
    cams = [c for c in df.columns if c.startswith(CAMERAS) and pd.api.types.is_float_dtype(df[c])]
    imaged = df[df[cams].notna().any(axis=1)].groupby("bed").plant.nunique()

    def tile(tint, label, value, unit, hover):
        return html.Div(
            [
                html.Div(label, className="uw-readout-label"),
                html.Div(value, className="uw-stat"),
                html.Div(unit, className="uw-readout-label"),
            ],
            className=f"uw-tile uw-{tint}",
            title=hover,
        )

    beds = [("W1", "control", "sky"), ("W2", "deficit", "sand"), ("W3", "deficit", "sand")]
    return html.Div(
        [
            *[
                tile(
                    t,
                    f"{b} · {r}",
                    f"{imaged.get(b, 0)}",
                    "plants imaged",
                    "plants with any camera reading in the tables",
                )
                for b, r, t in beds
            ],
            tile(
                "mint",
                "runs per day",
                f"{len(SLOTS)}",
                " · ".join(f"{h:02d}" for h in SLOTS) + " h",
                "scheduled flyovers",
            ),
            tile(
                "gray",
                "irrigation off",
                f"{CUTOFF.day} {CUTOFF:%b}",
                f"re-watered {REWATER.day} {REWATER:%b}",
                "W2 and W3: last watering 3 Sep 11:00, re-watered 15 Sep 18:35",
            ),
        ],
        className="uw-tiles",
    )


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
        return html.Div([head, not_available("Every module"), page_tiles()])
    return html.Div([head, tiles(df), html.Div(coverage(), style={"paddingTop": "20px"}), page_tiles()])
