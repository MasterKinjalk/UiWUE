"""3D Browser: one plant's cached point cloud per run, with its traits, QC flags and RGB frame.

Plan section 8.1. Data come from ``threed_data`` (trait table, cached clouds, RGB frame 7, E1 for
the reliability mark). Deep link: ``/threed-browser?bed=W2&plant=J5&date=2025-09-08&hour=15``.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import threed_data as td
from dash import Input, Output, State, callback, ctx, dcc, html, no_update
from PIL import Image
from threed_data import schema

dash.register_page(
    __name__,
    path="/threed-browser",
    name="3D Browser",
    title="3D Browser · UiWUE",
    description="One plant in 3D over time: point cloud, traits, QC flags and the RGB frame.",
    order=30,
)

TRAIT_SERIES = "height_p75"  # the trait plotted under the 3D view
DEFAULT_DATE = schema.TREATMENT_WINDOW[1]  # last day of the drought: the clearest effect
GRAPH_CONFIG = {"displaylogo": False, "scrollZoom": True}
PLANT_GREEN, SOIL_BROWN, PLANE_GREY = "#2E9E44", "#A08C78", "#8A9BB0"
NO_RGB_GREY = "#8a959c"  # design token gray-500: points the RGB camera did not see (rgb = 0)
# The RGB camera is under-exposed (coloured points average about 48 of 255), so colours are
# scaled by one gain per cloud that puts the 99th percentile at 255, as in the pipeline's
# diagnostics. The cap keeps dark (night) frames dark instead of amplifying their noise.
RGB_PERCENTILE, MAX_RGB_GAIN = 99, 4.0
# Points more than 15 cm below the pot soil are strays (the floor between pots lies at about
# −60 to −120 mm). With aspectmode="data" a handful of them stretched the height axis.
Z_FLOOR_MM = -150
COLOURS = [
    ("rgb", "RGB"),
    ("height", "Height"),
    ("plant", "Plant / background"),
    ("temp", "Temperature"),
]


# ---------------------------------------------------------------- figures (pure functions)
def rgb_colour_index(rgb: np.ndarray) -> tuple[np.ndarray, list, int]:
    """Map N x 3 colours to an adaptive palette of at most 256 colours.

    Returns (index per point, Plotly colour scale, cmax). Sending one byte per point plus a
    palette is about 10x smaller than one colour string per point."""
    if len(rgb) == 0:
        return np.zeros(0, np.uint8), [[0, "rgb(0,0,0)"], [1, "rgb(0,0,0)"]], 1
    img = Image.fromarray(np.ascontiguousarray(rgb.reshape(-1, 1, 3), dtype=np.uint8), "RGB")
    q = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(q, dtype=np.uint8).ravel()
    pal = np.asarray(q.getpalette(), dtype=np.uint8).reshape(-1, 3)
    k = int(idx.max()) + 1
    colours = [f"rgb({r},{g},{b})" for r, g, b in pal[:k]]
    if k == 1:
        return idx, [[0, colours[0]], [1, colours[0]]], 1
    return idx, [[i / (k - 1), c] for i, c in enumerate(colours)], k - 1


def rgb_gain(rgb: np.ndarray) -> float:
    """One brightness gain for a cloud: 255 over the 99th percentile of its sampled colour
    values, between 1 and ``MAX_RGB_GAIN``. Points without an RGB sample (0, 0, 0) are ignored."""
    seen = rgb[rgb.any(axis=1)]
    if len(seen) == 0:
        return 1.0
    hi = float(np.percentile(seen, RGB_PERCENTILE))
    return float(np.clip(255.0 / max(hi, 1.0), 1.0, MAX_RGB_GAIN))


def cloud_figure(
    cloud: dict | None,
    colour: str = "rgb",
    plant_only: bool = False,
    title: str = "",
    revision: str | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Scatter3d of a cloud in mm, marker size 2, true aspect ratio, reference plane at z = 0.

    ``zmax`` fixes the top of the height colour scale (mm), so dates and views compare."""
    if cloud is None or len(cloud["xyz"]) == 0:
        return td.message_figure("No cached cloud for this run yet.", 560)
    # whole mm as int16: half the bytes of float32, finer than the depth noise (6-12 mm)
    full = np.clip(np.round(cloud["xyz"] * 1000.0), -32768, 32767).astype(np.int16)
    keep = cloud["plant"] if plant_only else np.ones(len(full), bool)
    keep = keep & (full[:, 2] >= Z_FLOOR_MM)
    xyz = full[keep]
    if len(xyz) == 0:
        return td.message_figure("This cloud has no plant points.", 560)

    fig = go.Figure()
    (x0, y0), (x1, y1) = full[:, :2].min(0).tolist(), full[:, :2].max(0).tolist()
    fig.add_trace(
        go.Mesh3d(
            x=[x0, x1, x1, x0],
            y=[y0, y0, y1, y1],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=PLANE_GREY,
            opacity=0.18,
            hoverinfo="skip",
            name="reference plane (z = 0)",
        )
    )
    marker: dict = {"size": 2}
    hover = "x %{x:.0f} mm, y %{y:.0f} mm<br>height %{z:.0f} mm<extra></extra>"
    temp = cloud.get("temp")
    if colour == "temp" and temp is not None:
        t = temp[keep]
        ok = np.isfinite(t)
        if (~ok).any():  # points without a thermal sample, drawn grey
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[~ok, 0],
                    y=xyz[~ok, 1],
                    z=xyz[~ok, 2],
                    mode="markers",
                    marker={"size": 2, "color": "#C8CDD2"},
                    hoverinfo="skip",
                    name="no temperature",
                )
            )
        xyz, t = xyz[ok], t[ok]
        marker.update(
            color=t,
            colorscale="Inferno",
            colorbar={"title": {"text": "leaf temp. (°C)"}, "thickness": 12},
        )
    elif colour == "height":
        z = xyz[:, 2]
        top = zmax if zmax else (float(np.percentile(z, 99)) if len(z) else 1.0)
        marker.update(
            color=z,
            colorscale="Viridis",
            cmin=0,
            cmax=max(top, 1.0),
            colorbar={"title": {"text": "height (mm)"}, "thickness": 12},
        )
    elif colour == "plant":
        marker.update(
            color=cloud["plant"][keep].astype(np.uint8),
            cmin=0,
            cmax=1,
            showscale=False,
            colorscale=[[0, SOIL_BROWN], [0.5, SOIL_BROWN], [0.5, PLANT_GREEN], [1, PLANT_GREEN]],
        )
    else:
        rgb = cloud["rgb"][keep]
        seen = rgb.any(axis=1)
        if (~seen).any():  # points without an RGB sample, drawn grey rather than black
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[~seen, 0],
                    y=xyz[~seen, 1],
                    z=xyz[~seen, 2],
                    mode="markers",
                    marker={"size": 2, "color": NO_RGB_GREY},
                    hoverinfo="skip",
                    name="no colour",
                )
            )
        gain = rgb_gain(cloud["rgb"])
        xyz = xyz[seen]
        bright = np.clip(np.round(rgb[seen].astype(np.float32) * gain), 0, 255).astype(np.uint8)
        idx, scale, cmax = rgb_colour_index(bright)
        marker.update(color=idx, colorscale=scale, cmin=0, cmax=cmax, showscale=False)
        hover = hover.replace("<extra>", f"<br>colour ×{gain:.1f}<extra>")
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=marker,
            name="points",
            hovertemplate=hover,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        title={"text": title, "x": 0.01, "font": {"size": 13}},
        margin={"l": 0, "r": 0, "t": 34, "b": 0},
        showlegend=False,
        uirevision=revision,
        scene={
            "aspectmode": "data",  # true proportions: x, y and height share one scale
            "xaxis": {"title": {"text": "x (mm)"}},
            "yaxis": {"title": {"text": "y (mm)"}},
            "zaxis": {"title": {"text": "height (mm)"}},
            "camera": {"eye": {"x": 1.25, "y": -1.75, "z": 1.35}},
        },
    )
    return fig


def cloud_title(bed: str, plant: str, d, hour, cloud: dict | None, note: str = "") -> str:
    parts = [f"{bed}_{plant} · {td.parse_date(d)} · {int(hour):02d} h"]
    if cloud is not None:
        n = len(cloud["xyz"])
        total = cloud.get("n_total", n)
        parts.append(f"{n:,} points" + (f" of {total:,}" if total > n else ""))
        if cloud.get("source") == "synthetic":
            parts.append("<b>synthetic cloud</b>")
    if note:
        parts.append(note)
    return " · ".join(parts)


def series_figure(runs, current, hour, second=None) -> go.Figure:
    """``height_p75`` of one plant over time, one trace per hour, the shown date(s) marked.

    Points carry ``customdata = [iso date, hour]`` so a click can move the 3D view."""
    if runs.empty:
        return td.message_figure("No runs for this plant.", 260)
    fig = go.Figure()
    fmt = td.value_format(TRAIT_SERIES)
    for h in sorted(runs["hour"].unique()):
        part = runs[(runs["hour"] == h) & runs["qc_pass"]]
        selected = int(h) == int(hour or -1)
        fig.add_trace(
            go.Scatter(
                x=part["day"],
                y=part[TRAIT_SERIES],
                name=f"{int(h):02d} h",
                mode="lines+markers" if selected else "markers",
                marker={"size": 8 if selected else 5, "color": td.hour_colour(h)},
                line={"width": 2, "color": td.hour_colour(h)},
                opacity=1.0 if selected else 0.45,
                customdata=np.c_[[d.isoformat() for d in part["date"]], part["hour"]],
                hovertemplate=f"%{{x|%Y-%m-%d}} {int(h):02d} h: %{{y:{fmt}}} mm<extra></extra>",
            )
        )
    failed = runs[~runs["qc_pass"]]
    if not failed.empty:
        fig.add_trace(
            go.Scatter(
                x=failed["day"],
                y=failed[TRAIT_SERIES],
                name="QC failed",
                mode="markers",
                marker={"symbol": "x", "size": 7, "color": "#868E96"},
                customdata=np.c_[[d.isoformat() for d in failed["date"]], failed["hour"]],
                hovertemplate="%{x|%Y-%m-%d} %{customdata[1]} h: %{y:.1f} mm (QC failed)<extra></extra>",
            )
        )
    td.add_calendar(fig)
    if current is not None:
        fig.add_vline(x=td.parse_date(current).isoformat(), line={"color": "#212529", "width": 2})
    if second is not None:
        fig.add_vline(
            x=td.parse_date(second).isoformat(),
            line={"color": "#212529", "width": 2, "dash": "dash"},
        )
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        xaxis={"title": {"text": "date"}},
        yaxis={"title": {"text": td.trait_label(TRAIT_SERIES)}},
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        hovermode="closest",
        clickmode="event",
    )
    return fig


def side_panel(row, bed: str, plant: str, d, hour, unreliable: set[str]) -> list:
    """Run header, QC badges, trait table and the RGB frame of one run."""
    if row is None:
        return [html.P("No run for this selection.", className="text-muted")]
    d = td.parse_date(d)
    blocking = [f for f, b in schema.QC_FLAGS.items() if b and row[f]]
    soft = [f for f, b in schema.QC_FLAGS.items() if not b and row[f]]
    badges = [
        dbc.Badge(
            "QC pass" if row["qc_pass"] else "QC fail",
            color="success" if row["qc_pass"] else "danger",
            class_name="me-1",
        ),
        *[dbc.Badge(f.removeprefix("qc_").replace("_", " "), color="danger", class_name="me-1") for f in blocking],
        *[
            dbc.Badge(
                f.removeprefix("qc_").replace("_", " "),
                color="warning",
                text_color="dark",
                class_name="me-1",
            )
            for f in soft
        ],
    ]
    local = row["timestamp_utc"].tz_convert("Europe/Berlin").strftime("%H:%M")
    meta = (
        f"{row['treatment']} · slot {int(row['slot'])} h · day {int(row['days_since_cutoff']):+d} "
        f"from cutoff · {int(row['n_frames_ok'])} frames ok · median frame {local} local"
    )
    rows = []
    for t in schema.TRAITS:
        name = [t]
        if t in unreliable:
            name.append(html.Span(" not reliable (E1)", className="text-danger small"))
        mad = td.fmt_value(t, row.get(f"{t}_mad"))
        rows.append(
            html.Tr(
                [
                    html.Td(name),
                    html.Td(
                        [td.fmt_value(t, row[t]), html.Span(f" ± {mad}", className="text-muted")],
                        className="text-end text-nowrap",
                    ),
                    html.Td(td.unit(t), className="text-muted"),
                ]
            )
        )
    uri = td.rgb_thumbnail(bed, plant, d, hour)
    rel = td.rgb_frame_path(bed, plant, d, hour).relative_to(td.repo_root()).as_posix()
    image = (
        html.Figure(
            [
                html.Img(src=uri, className="img-fluid rounded border", alt="RGB frame 7"),
                html.Figcaption(f"RGB frame 7 · {rel}", className="text-muted small mt-1"),
            ],
            className="mb-0",
        )
        if uri
        else html.P(f"No RGB frame on disk ({rel}).", className="text-muted small")
    )
    return [
        html.H6(f"{bed}_{plant} · {d.isoformat()} · {int(hour):02d} h", className="mb-1"),
        html.Div(meta, className="text-muted small mb-2"),
        html.Div(badges, className="mb-2"),
        dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("trait"),
                            html.Th("median ± MAD", className="text-end"),
                            html.Th("unit"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            size="sm",
            borderless=True,
            hover=True,
            class_name="small mb-2",
        ),
        image,
    ]


# ---------------------------------------------------------------- layout
def date_marks(dates: list) -> dict:
    """Slider marks: every 7th run date, plus the cutoff and re-watering dates highlighted."""
    marks = {}
    special = {schema.CUTOFF: "off", schema.REWATER: "on"}
    special_idx = [i for i, d in enumerate(dates) if d in special]
    for i, d in enumerate(dates):
        if d in special:
            marks[i] = {"label": f"{d:%m-%d} {special[d]}", "style": {"color": "#D55E00"}}
        elif i % 7 == 0 and all(abs(i - j) > 2 for j in special_idx):
            marks[i] = {"label": f"{d:%m-%d}"}
    return marks


def colour_options(has_temp: bool) -> list[dict]:
    return [{"label": label, "value": v, "disabled": v == "temp" and not has_temp} for v, label in COLOURS]


def hour_options(hours: list[int]) -> list[dict]:
    return [{"label": f"{h:02d}", "value": h} for h in hours]


def _date_at(dates: list[str] | None, idx) -> str | None:
    if not dates or idx is None:
        return None
    idx = int(idx)
    return dates[idx] if 0 <= idx < len(dates) else None


def _label(text: str, **kw) -> dbc.Label:
    return dbc.Label(text, class_name="small fw-semibold mb-1", **kw)


def layout(bed=None, plant=None, date=None, hour=None, **_):
    df = td.load_runs()
    banner = td.source_banner()
    if df.empty:
        return dbc.Container([banner, dbc.Alert("The trait table has no runs.", color="danger")])
    beds = td.beds(df)
    bed = bed if bed in beds else ("W2" if "W2" in beds else beds[0])
    plants = td.plants_in_bed(df, bed)
    plant = plant if plant in plants else ("J5" if "J5" in plants else plants[0])
    dates = td.run_dates(df, bed, plant)
    d = td.nearest(dates, td.parse_date(date) or DEFAULT_DATE)
    hours = td.run_hours(df, bed, plant, d)
    h = td.nearest(hours, int(hour) if hour else 15)
    d2 = td.nearest(dates, schema.CUTOFF)
    iso = [x.isoformat() for x in dates]

    controls = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _label("Bed"),
                                dcc.Dropdown(id="tb-bed", options=beds, value=bed, clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                _label("Plant"),
                                dcc.Dropdown(id="tb-plant", options=plants, value=plant, clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                _label("Hour"),
                                html.Div(
                                    dbc.RadioItems(
                                        id="tb-hour",
                                        options=hour_options(hours),
                                        value=h,
                                        class_name="btn-group",
                                        input_class_name="btn-check",
                                        label_class_name="btn btn-outline-primary btn-sm",
                                        label_checked_class_name="active",
                                    ),
                                    className="radio-group",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                _label("Colour by"),
                                dbc.RadioItems(
                                    id="tb-colour",
                                    options=colour_options(False),
                                    value="rgb",
                                    inline=True,
                                    class_name="small",
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Switch(
                                    id="tb-plant-only",
                                    label="Plant only",
                                    value=False,
                                    class_name="small mb-1",
                                ),
                                dbc.Switch(
                                    id="tb-second",
                                    label="Second view",
                                    value=False,
                                    class_name="small mb-0",
                                ),
                            ],
                            md=2,
                        ),
                    ],
                    class_name="g-3 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                _label("Date", id="tb-date-label"),
                                dcc.Slider(
                                    id="tb-date",
                                    min=0,
                                    max=max(len(dates) - 1, 0),
                                    step=1,
                                    value=iso.index(d.isoformat()),
                                    marks=date_marks(dates),
                                    updatemode="mouseup",
                                    allow_direct_input=False,
                                ),
                            ],
                            md=9,
                        ),
                        dbc.Col(
                            [
                                _label("Second view date"),
                                dcc.Dropdown(
                                    id="tb-date2",
                                    options=iso,
                                    value=d2.isoformat(),
                                    clearable=False,
                                    disabled=True,
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    class_name="g-3",
                ),
            ]
        ),
        class_name="mb-3",
    )
    graph = {"config": GRAPH_CONFIG, "responsive": True}
    views = dbc.Row(
        [
            dbc.Col(dcc.Loading(dcc.Graph(id="tb-cloud", **graph), delay_show=300), id="tb-col1", md=12),
            dbc.Col(
                dcc.Loading(dcc.Graph(id="tb-cloud2", **graph), delay_show=300),
                id="tb-col2",
                md=6,
                style={"display": "none"},
            ),
        ],
        class_name="g-2",
    )
    return dbc.Container(
        [
            banner,
            controls,
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(views, class_name="p-2")), lg=8),
                    dbc.Col(dbc.Card(dbc.CardBody(id="tb-side", class_name="p-3")), lg=4),
                ],
                class_name="g-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Click a point to show that run in 3D.", className="text-muted small"),
                        dcc.Graph(id="tb-series", config=GRAPH_CONFIG, responsive=True),
                    ],
                    class_name="p-2",
                ),
                class_name="mt-3 mb-4",
            ),
            dcc.Store(id="tb-dates", data=iso),
        ],
        fluid=True,
        class_name="py-3",
    )


# ---------------------------------------------------------------- callbacks
@callback(
    Output("tb-plant", "options"),
    Output("tb-plant", "value"),
    Input("tb-bed", "value"),
    State("tb-plant", "value"),
)
def update_plants(bed, plant):
    plants = td.plants_in_bed(td.load_runs(), bed)
    return plants, (plant if plant in plants else (plants[0] if plants else None))


@callback(
    Output("tb-date", "max"),
    Output("tb-date", "marks"),
    Output("tb-date", "value"),
    Output("tb-dates", "data"),
    Output("tb-date2", "options"),
    Output("tb-date2", "value"),
    Input("tb-bed", "value"),
    Input("tb-plant", "value"),
    Input("tb-series", "clickData"),
    State("tb-date", "value"),
    State("tb-dates", "data"),
    State("tb-date2", "value"),
)
def update_dates(bed, plant, click, idx, old_dates, date2):
    """Run dates of the plant; keeps the shown date (or the nearest), or jumps to a clicked one."""
    dates = td.run_dates(td.load_runs(), bed, plant)
    if not dates:
        return 0, {}, 0, [], [], None
    iso = [d.isoformat() for d in dates]
    target = td.parse_date(_date_at(old_dates, idx)) or DEFAULT_DATE
    if ctx.triggered_id == "tb-series" and click and click.get("points"):
        point = click["points"][0]
        target = td.parse_date((point.get("customdata") or [point.get("x")])[0])
    d = td.nearest(dates, target)
    d2 = td.nearest(dates, td.parse_date(date2) or schema.CUTOFF)
    return len(dates) - 1, date_marks(dates), iso.index(d.isoformat()), iso, iso, d2.isoformat()


@callback(
    Output("tb-hour", "options"),
    Output("tb-hour", "value"),
    Input("tb-date", "value"),
    Input("tb-dates", "data"),
    Input("tb-series", "clickData"),
    State("tb-bed", "value"),
    State("tb-plant", "value"),
    State("tb-hour", "value"),
)
def update_hours(idx, dates, click, bed, plant, hour):
    """Hours with a run on the shown date; keeps the hour (or the nearest), or the clicked one."""
    d = _date_at(dates, idx)
    hours = td.run_hours(td.load_runs(), bed, plant, d) if d else []
    want = hour if hour is not None else 15
    if "tb-series.clickData" in ctx.triggered_prop_ids and click and click.get("points"):
        custom = click["points"][0].get("customdata")
        if custom and len(custom) > 1:
            want = int(custom[1])
    return hour_options(hours), td.nearest(hours, int(want))


def colour_top(bed: str, plant: str) -> float | None:
    """Top of the height colour scale for a plant: its tallest run (height_p95) plus 15 %."""
    df = td.load_runs()
    h = df.loc[(df["bed"] == bed) & (df["plant"] == plant), "height_p95"].max()
    return float(h) * 1.15 if np.isfinite(h) and h > 0 else None


@callback(
    Output("tb-cloud", "figure"),
    Output("tb-colour", "options"),
    Input("tb-bed", "value"),
    Input("tb-plant", "value"),
    Input("tb-date", "value"),
    Input("tb-dates", "data"),
    Input("tb-hour", "value"),
    Input("tb-colour", "value"),
    Input("tb-plant-only", "value"),
)
def render_cloud(bed, plant, idx, dates, hour, colour, plant_only):
    d = _date_at(dates, idx)
    if d is None or hour is None:
        return td.message_figure("No run for this selection.", 560), colour_options(False)
    cloud = td.load_cloud(bed, plant, d, hour)
    has_temp = cloud is not None and cloud.get("temp") is not None
    note = "no temperature in this cloud, coloured by height" if colour == "temp" and not has_temp else ""
    shown = "height" if note else colour
    title = cloud_title(bed, plant, d, hour, cloud, note)
    top = colour_top(bed, plant)
    fig = cloud_figure(cloud, shown, bool(plant_only), title, f"{bed}_{plant}", top)
    return fig, colour_options(has_temp)


@callback(
    Output("tb-cloud2", "figure"),
    Output("tb-col1", "md"),
    Output("tb-col2", "style"),
    Output("tb-date2", "disabled"),
    Input("tb-second", "value"),
    Input("tb-date2", "value"),
    Input("tb-bed", "value"),
    Input("tb-plant", "value"),
    Input("tb-hour", "value"),
    Input("tb-colour", "value"),
    Input("tb-plant-only", "value"),
)
def render_second(on, d2, bed, plant, hour, colour, plant_only):
    """The same plant on a second date, at the same hour or the nearest one that ran."""
    if not on:
        return no_update, 12, {"display": "none"}, True
    hours = td.run_hours(td.load_runs(), bed, plant, d2) if d2 else []
    h2 = td.nearest(hours, int(hour) if hour is not None else 15)
    if h2 is None:
        return td.message_figure("No run on the second date.", 560), 6, {}, False
    cloud = td.load_cloud(bed, plant, d2, h2)
    has_temp = cloud is not None and cloud.get("temp") is not None
    shown = "height" if colour == "temp" and not has_temp else colour
    title = cloud_title(bed, plant, d2, h2, cloud)
    fig = cloud_figure(cloud, shown, bool(plant_only), title, f"{bed}_{plant}_2", colour_top(bed, plant))
    return fig, 6, {}, False


@callback(
    Output("tb-side", "children"),
    Output("tb-series", "figure"),
    Output("tb-date-label", "children"),
    Input("tb-bed", "value"),
    Input("tb-plant", "value"),
    Input("tb-date", "value"),
    Input("tb-dates", "data"),
    Input("tb-hour", "value"),
    Input("tb-second", "value"),
    Input("tb-date2", "value"),
)
def render_side(bed, plant, idx, dates, hour, second_on, d2):
    df = td.load_runs()
    d = _date_at(dates, idx)
    runs = df[(df["bed"] == bed) & (df["plant"] == plant)]
    fig = series_figure(runs, d, hour, d2 if second_on else None)
    if d is None:
        return side_panel(None, bed, plant, None, hour, set()), fig, "Date"
    row = td.get_run(df, bed, plant, d, hour)
    panel = side_panel(row, bed, plant, d, hour, td.unreliable_traits())
    days = td.schema.days_since_cutoff(td.parse_date(d))
    label = f"Date: {d} (day {days:+d} from the irrigation cutoff)"
    return panel, fig, label
