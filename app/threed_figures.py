"""Figures for the 3D canopy page (``/threed``): the then/now stage, the side view and the trait charts.

Numbers come from ``threed_canopy`` (one table, three traits, real clouds only); this module only picks
which real run to draw, frames it and colours it. Colours are design tokens (``components.C``); 2D charts
use the ``uiwue`` template and the band conventions of DESIGN.md section 6.

Stage, as built after comparing renders on screenshots (points, a Delaunay canopy surface, a height map):
plant points coloured by height on the leaf ramp, soil as one flat disc with the 85 mm crop ring, a 50 mm
scale bar instead of axes, an orthographic three-quarter camera fitted to the pair so the plant fills its
panel, and THEN and NOW on one scale, one camera and one colour range.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import threed_canopy as tc
from components import BAND, C, day_label, mark_stress_period

# ---------------------------------------------------------------- constants
# cloud height ramp: the leaf ramp of DESIGN 2.1 from its middle stop up, so no point is pale on white
LEAF_SCALE = [[0.0, C["leaf-300"]], [0.5, C["leaf-700"]], [1.0, C["leaf-900"]]]
DISC_SEGMENTS = 96
SCALE_BAR_MM = 50.0
VIEW = (0.75, -0.95, 0.32)  # default eye direction from the plant: three-quarter view, 15° up
STAGE_ASPECT = 1.04  # width / height of a stage panel (CSS aspect-ratio of .td-canvas)
STAGE_PX = 500  # panel height at 1440 px, for the marker size only (the frame scales with the panel)
MARGIN = 0.03  # share of the panel height kept free above the plant and below the scale bar
FRONT_MM = 22.0  # room in front of the disc for the scale bar (its label sits beside it)
PAD_MM = 34.0  # x and y range beyond the crop cylinder (the scale bar stands there)
TIP = 1.15  # the frame reaches the highest point, or 1.15 x the 99th percentile when one stray point is higher
X_DAYS = (-10.5, 16.5)  # day axis of the trait charts: the 3D table runs from day -10 to 16
GRAPH_CONFIG = {"displayModeBar": False}
STAGE_CONFIG = {"displayModeBar": False, "scrollZoom": False}  # rotate only: the scale stays locked


def _minus(text: str) -> str:
    return text.replace("-", "−")


def pct(x, digits: int = 0, unit: str = " %") -> str:
    """'−46 %' with a real minus sign and a sign on gains; '–' when missing."""
    if x is None or not np.isfinite(x):
        return "–"
    return _minus(f"{x:+.{digits}f}") + unit


def day_text(day) -> str:
    """'Day 4 · 7 Sep'."""
    d, date = day_label(int(day))
    return f"{d} · {date}"


# ---------------------------------------------------------------- which run and which days to draw
def stage_cloud(bed: str, plant: str, day: int | None, slot: int | None = None) -> dict | None:
    """The cloud drawn for a plant and day: the run ``threed_canopy.cloud_days`` picks (the day's most typical
    QC-pass run with a drawable cloud), or the given slot (THEN: ``threed_canopy.then_run``). None when there is
    none: never a failed scan, never a made-up cloud."""
    if day is None:
        return None
    return tc.plant_cloud(bed, plant, int(day), None if slot is None else int(slot))


def strip_days(bed: str, plant: str) -> pd.DataFrame:
    """One row per day of the plant in the 3D table: day, date, state ('ok' cloud to draw, 'no file', 'sparse',
    'no scan' = no QC-pass daytime run), the run drawn (slot, its values ``<trait>_run`` and change from baseline
    ``<trait>_run_pct``, ``n_stray``) and the plant-day means with their change from baseline (``<trait>``,
    ``<trait>_pct``, n_pass, slots). Works for a W3 plant too."""
    d = tc.cloud_days(bed, plant)
    p = tc.plant_days(tc.SHOWN_BEDS)
    if p is None or d.empty:
        return pd.DataFrame(columns=["day", "date", "state"])
    mine = p[(p["bed"] == bed) & (p["plant"] == plant)].drop(columns=["date", "bed", "plant", "treatment", "n_pass"])
    out = d.rename(columns={"reason": "state"}).merge(mine, on="day", how="left")
    return out.sort_values("day", ignore_index=True)


def snap_day(days: pd.DataFrame, want: int | None) -> int | None:
    """The asked-for day when it can be shown, else the nearest drawable day (ties: the earlier one), so a
    link never opens on a failed scan."""
    if want is None or days.empty:
        return None
    ok = days.loc[days["state"] == "ok", "day"]
    pool = ok if len(ok) else days.loc[days["state"] != "no scan", "day"]
    if pool.empty:
        return None
    pool = pool.astype(int).to_numpy()
    return int(pool[np.lexsort((pool, np.abs(pool - int(want))))][0])


def _kept(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y, z of the points a picture keeps (detached edge points hidden, ``threed_canopy.stray_mask``)."""
    keep = ~np.asarray(c.get("stray", np.zeros(len(c["x"]), bool)), bool)
    return tuple(np.asarray(c[k], float)[keep] for k in "xyz")


def frame_height(*clouds) -> float:
    """Height (mm, rounded up to 10) the stage frames: the highest point the pictures keep, or 1.15 x their
    99th percentile when a lone point stands far above the plant."""
    tops = []
    for c in clouds:
        if c is None or not len(c["z"]):
            continue
        z = _kept(c)[2]
        if len(z):
            tops.append(min(float(z.max()), TIP * float(np.percentile(z, 99))))
    return float(math.ceil(max(tops, default=100.0) / 10.0) * 10.0)


# ---------------------------------------------------------------- stage geometry
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return np.asarray(VIEW, float) / np.linalg.norm(VIEW) if n < 1e-9 else v / n


def view_of(camera: dict | None) -> tuple[float, float, float]:
    """Eye direction (eye minus centre, unit length) of a Plotly scene camera, else the default view."""
    try:
        eye, centre = camera["eye"], camera.get("center") or {"x": 0, "y": 0, "z": 0}
        d = _unit([eye["x"] - centre["x"], eye["y"] - centre["y"], eye["z"] - centre["z"]])
    except (KeyError, TypeError):
        d = _unit(VIEW)
    if abs(d[2]) > 0.995:  # straight down: keep a sliver of azimuth so the scale bar has a direction
        d = _unit([0.05, -0.05, np.sign(d[2])])
    return float(d[0]), float(d[1]), float(d[2])


def _elevation(view) -> tuple[float, float]:
    d = _unit(view)
    el = math.asin(max(-1.0, min(1.0, d[2])))
    return abs(math.sin(el)), max(math.cos(el), 0.05)


def reach(clouds, view=VIEW) -> float:
    """How high the clouds reach on screen (mm, before scaling), whichever way the plant is turned about its
    stem: the largest z · cos(el) + ρ · sin(el) over the points (ρ = distance from the stem axis), so turning
    it never lifts a leaf out of the frame. A stray point far above the rest does not set it (at most 1.15 x
    the 99th percentile)."""
    sin_e, cos_e = _elevation(view)
    tops = []
    for c in clouds:
        if c is None or not c["n"]:
            continue
        x, y, z = _kept(c)
        if not len(z):
            continue
        v = z * cos_e + np.hypot(x, y) * sin_e
        tops.append(min(float(v.max()), TIP * float(np.percentile(v, 99))))
    return max([*tops, tc.CYLINDER_MM * sin_e + 40.0 * cos_e])


def framing(clouds, top: float, view=VIEW, aspect: float = STAGE_ASPECT) -> dict:
    """Camera and zoom so the clouds, the soil disc and the scale bar fill a panel of width / height
    ``aspect``, seen from ``view``; THEN and NOW pass both clouds, so they share it.

    Plotly's orthographic scene: one scene unit is L / k mm (L the x range, k the aspect-ratio zoom), the
    screen scale is k · H / (2 L) px per mm for a panel H px high, and moving the camera centre by c units
    moves the picture by c · H / 2 · cos(elevation) px (both measured on screenshots). The content spans
    from the scale bar in front of the disc, −(r + front) · sin(el), up to ``reach``, and 2 r across;
    solving for k and c centres it."""
    r = tc.CYLINDER_MM
    span = 2 * (r + PAD_MM)
    sin_e, cos_e = _elevation(view)
    high, low = reach(clouds, view), -(r + FRONT_MM) * sin_e
    k = 2 * span * (1 - 2 * MARGIN) * min(1.0 / (high - low), aspect / (2 * r + 12.0))
    c = ((high + low) / 2 / cos_e - top / 2) * k / span  # the camera looks at the middle of the content
    eye = _unit(view) * 2.0
    return dict(
        aspectratio=dict(x=k, y=k, z=k * top / span),
        camera=dict(
            eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2] + c)),
            center=dict(x=0.0, y=0.0, z=float(c)),
            up=dict(x=0.0, y=0.0, z=1.0),
            projection=dict(type="orthographic"),
        ),
        scale=k * STAGE_PX / (2 * span),  # px per mm in a STAGE_PX high panel
    )


def _ground(r: float) -> list:
    """Soil as one flat disc, with the 85 mm ring the pipeline crops the plant to."""
    n = DISC_SEGMENTS
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    disc = go.Mesh3d(
        x=np.r_[0.0, r * np.cos(a)],
        y=np.r_[0.0, r * np.sin(a)],
        z=np.zeros(n + 1),
        i=np.zeros(n, np.int32),
        j=np.arange(1, n + 1, dtype=np.int32),
        k=np.r_[np.arange(2, n + 1), 1].astype(np.int32),
        color=C["soil"],
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        hoverinfo="skip",
        showscale=False,
        name="soil",
    )
    b = np.linspace(0, 2 * np.pi, n + 1)
    ring = go.Scatter3d(
        x=r * np.cos(b),
        y=r * np.sin(b),
        z=np.zeros(n + 1),
        mode="lines",
        line=dict(color=C["gray-500"], width=2),
        hoverinfo="skip",
        name="crop ring",
    )
    return [disc, ring]


def _scale_bar(r: float, view) -> list:
    """A 50 mm bar on the ground in front of the disc, square to the view: in an orthographic view it is
    exactly 50 mm long at the plant's scale (until the scene is turned)."""
    d = _unit(view)
    toward = np.asarray(d[:2]) / max(float(np.hypot(*d[:2])), 1e-9)  # on the ground, towards the camera
    across = np.array([-toward[1], toward[0]])
    mid = toward * (r + 14.0)
    a, b = mid - across * SCALE_BAR_MM / 2, mid + across * SCALE_BAR_MM / 2
    label = b + across * 3.0
    return [
        go.Scatter3d(
            x=[a[0], b[0]],
            y=[a[1], b[1]],
            z=[0.0, 0.0],
            mode="lines",
            line=dict(color=C["ink"], width=5),
            hoverinfo="skip",
            name="scale bar",
        ),
        go.Scatter3d(
            x=[label[0]],
            y=[label[1]],
            z=[0.0],
            mode="text",
            text=[f"{SCALE_BAR_MM:.0f} mm"],
            textposition="middle right",
            textfont=dict(size=13, color=C["ink"]),
            hoverinfo="skip",
            name="scale label",
        ),
    ]


def stage_figure(
    cloud: dict | None,
    frame_clouds,
    colour_top: float,
    view=VIEW,
    revision: str | None = None,
) -> go.Figure:
    """One plant cloud on the stage: points coloured by height on the leaf ramp (0 to ``colour_top`` mm, one
    range per plant), the soil disc and crop ring, a 50 mm bar, no axes. ``frame_clouds`` (THEN and NOW) set
    the frame, so both panels share one camera and one scale. Detached edge points (``cloud["stray"]``) are not
    drawn; the panel head says how many."""
    r = tc.CYLINDER_MM
    highest = [float(z.max()) for c in frame_clouds if c is not None and c["n"] for z in [_kept(c)[2]] if len(z)]
    top = float(math.ceil(max([colour_top, *highest]) / 10.0) * 10.0)
    view = tuple(_unit(view or VIEW))
    f = framing(frame_clouds, top, view)
    traces = _ground(r) + _scale_bar(r, view)
    if cloud is not None:
        x, y, z = _kept(cloud)
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=float(np.clip(1.6 * f["scale"], 2.5, 4.5)),
                    color=z,
                    colorscale=LEAF_SCALE,
                    cmin=0.0,
                    cmax=float(colour_top),
                    line=dict(width=0),
                ),
                hovertemplate="%{z:.0f} mm<extra></extra>",
                name="plant",
            )
        )
    axis = dict(visible=False, showspikes=False)
    fig = go.Figure(traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision=revision,
        paper_bgcolor=C["surface"],
        hoverlabel=dict(bgcolor=C["surface"], font_color=C["ink"]),
        scene=dict(
            xaxis=axis | dict(range=[-r - PAD_MM, r + PAD_MM]),
            yaxis=axis | dict(range=[-r - PAD_MM, r + PAD_MM]),
            zaxis=axis | dict(range=[0.0, top]),
            aspectmode="manual",
            aspectratio=f["aspectratio"],
            camera=f["camera"],
            bgcolor=C["surface"],
            dragmode="turntable",
        ),
    )
    return fig


# ---------------------------------------------------------------- side view
def side_figure(then: dict | None, now: dict | None, frame_mm: float, labels=("then", "now")) -> go.Figure:
    """The plant seen from the side (along y): the highest point per 5 mm of x. THEN as a dotted outline,
    NOW filled, on the stage's height range, so wilting reads as the fill dropping below the outline."""
    a, b = tc.side_profile(then), tc.side_profile(now)
    fig = go.Figure()
    fig.add_scatter(
        x=b["x"],
        y=b["z"],
        mode="lines",
        line=dict(color=C["forest-700"], width=2, shape="hvh"),
        fill="tozeroy",
        fillcolor=C["leaf-300"],
        connectgaps=False,
        name=labels[1],
        hovertemplate=f"{labels[1]} · %{{y:.0f}} mm at %{{x:.0f}} mm<extra></extra>",
    )
    fig.add_scatter(
        x=a["x"],
        y=a["z"],
        mode="lines",
        line=dict(color=C["ink"], width=2, dash="dot", shape="hvh"),
        connectgaps=False,
        name=labels[0],
        hovertemplate=f"{labels[0]} · %{{y:.0f}} mm at %{{x:.0f}} mm<extra></extra>",
    )
    fig.update_layout(
        margin=dict(l=44, r=8, t=8, b=30),
        hovermode="closest",
        showlegend=False,  # the panel head carries the key
        xaxis=dict(range=[-tc.CYLINDER_MM, tc.CYLINDER_MM], title=None, dtick=40, showgrid=False),
        yaxis=dict(range=[0, frame_mm * 1.04], title="mm", rangemode="tozero"),
    )
    return fig


# ---------------------------------------------------------------- deficit minus control
def plant_gap(trait: str, bed: str, plant: str) -> pd.DataFrame:
    """The plant's own change minus the control bed's mean change, day by day (percentage points); W3 too."""
    p = tc.plant_days(tc.SHOWN_BEDS)
    g = tc.gap(trait)
    if p is None or g is None:
        return pd.DataFrame(columns=["day", "value"])
    mine = p[(p["bed"] == bed) & (p["plant"] == plant)][["day", f"{trait}_pct"]]
    ctrl = g["by_day"][["day", "control"]]
    out = mine.merge(ctrl, on="day", how="left")
    out["value"] = out[f"{trait}_pct"] - out["control"]
    return out[["day", "value"]].dropna()


def gap_figure(trait: str, bed: str, plant: str, onset=None, height: int = 250) -> go.Figure | None:
    """Deficit minus control in % of each plant's own baseline: the bed gap (amber) with its 95 % plant
    bootstrap band, the watered-day spread as a grey band, the onset, and the selected plant (navy)."""
    g = tc.gap(trait)
    if g is None:
        return None
    by = g["by_day"]
    lo, hi = g["null_range"]
    fig = go.Figure()
    mark_stress_period(fig)
    if np.isfinite(lo) and np.isfinite(hi):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=C["gray"], line_width=0, layer="below")
    band = by.dropna(subset=["ci_low", "ci_high"])
    fig.add_scatter(
        x=list(band["day"]) + list(band["day"][::-1]),
        y=list(band["ci_high"]) + list(band["ci_low"][::-1]),
        mode="lines",
        fill="toself",
        fillcolor=BAND["deficit"],
        line=dict(width=0),
        hoverinfo="skip",
        name="95 % CI",
        showlegend=False,
    )
    custom = np.c_[
        [day_text(d) for d in by["day"]],
        [pct(v, 0, " pp") for v in by["gap"]],
        [
            f"{pct(a, 0, '')} to {pct(b, 0, ' pp')}" if np.isfinite(a) else "–"
            for a, b in zip(by["ci_low"], by["ci_high"])
        ],
        by["n_deficit"],
        by["n_control"],
    ]
    mine = plant_gap(trait, bed, plant)
    if len(mine):  # under the bed line: the plant is the second read, the bed gap the first
        fig.add_scatter(
            x=mine["day"],
            y=mine["value"],
            mode="lines+markers",
            line=dict(color=C["navy"], width=1.25),
            marker=dict(size=4, color=C["navy"]),
            customdata=np.c_[[day_text(d) for d in mine["day"]], [pct(v, 0, " pp") for v in mine["value"]]],
            name=f"{bed} {plant}",
            hovertemplate=f"{bed} {plant} − W1 mean · %{{customdata[0]}} · %{{customdata[1]}}<extra></extra>",
        )
    fig.add_scatter(
        x=by["day"],
        y=by["gap"],
        mode="lines",
        line=dict(color=C["amber-700"], width=3.5),
        customdata=custom,
        name="W2 − W1",
        hovertemplate=(
            "W2 − W1 · %{customdata[0]} · %{customdata[1]}<br>95 % CI %{customdata[2]} · "
            "%{customdata[3]} deficit, %{customdata[4]} control plants<extra></extra>"
        ),
    )
    if onset is not None and np.isfinite(onset):
        fig.add_vline(x=float(onset), line_width=1.5, line_dash="dash", line_color=C["navy"])
        fig.add_annotation(
            x=float(onset),
            y=1,
            yref="paper",
            text="onset",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=12, color=C["navy"]),
        )
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=12, t=26, b=30),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(range=list(X_DAYS), dtick=5, title=None, tickformat="d"),
        yaxis=dict(title="pp", zeroline=True, zerolinecolor=C["gray-500"], zerolinewidth=1),
    )
    return fig


# ---------------------------------------------------------------- collapse and recovery
STEPS = ("peak", "trough", "rewatered")
STEP_TEXT = {"peak": "peak", "trough": "trough", "rewatered": "re-watered"}


GROUP_STYLE = (  # (group, bed, line colour, band colour, dot offset on the x axis)
    ("control", tc.CONTROL_BED, C["teal-700"], BAND["control"], 0.09),
    ("deficit", tc.DEFICIT_BED, C["amber-700"], BAND["deficit"], -0.09),
)
JITTER = 0.05  # half-width of a group's dot column, in steps


def _jitter(plants) -> np.ndarray:
    """A fixed horizontal spread per plant (from its name), so the dots do not move between renders."""
    return np.array([((sum(map(ord, str(p))) * 37) % 101 / 100.0 - 0.5) * 2 * JITTER for p in plants])


def slope_figure(trait: str, bed: str, plant: str, height: int = 300) -> go.Figure | None:
    """Every plant against its own peak (days 1–3): the trough (days 10–12) and after re-watering (days 15–16).

    Per group (control teal, deficit amber, W1 and W2 only) the median plant as a line and the middle half of the
    plants (interquartile range) as a band; each plant as a dot at the trough and after re-watering (hover names it,
    a click shows it); the selected plant in navy on top, also when it is a W3 plant (never pooled)."""
    w = tc.windows(tc.SHOWN_BEDS)
    if w is None:
        return None
    w = w[w["trait"] == trait]
    x = [0, 1, 2]
    fig = go.Figure()
    for group, b, colour, band, offset in GROUP_STYLE:
        g = w[(w["bed"] == b)]
        steps = [pd.Series([0.0] * len(g), index=g.index), g["peak_to_trough"], g["peak_to_rewatered"]]
        lo = [float(s.quantile(0.25)) if s.notna().any() else np.nan for s in steps]
        hi = [float(s.quantile(0.75)) if s.notna().any() else np.nan for s in steps]
        med = [float(s.median()) if s.notna().any() else np.nan for s in steps]
        n = [int(s.notna().sum()) for s in steps]
        fig.add_scatter(
            x=x + x[::-1],
            y=hi + lo[::-1],
            mode="lines",  # a band only: no vertex markers
            fill="toself",
            fillcolor=band,
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name=f"{group} middle half",
        )
        for i in (1, 2):
            s = steps[i].dropna()
            fig.add_scatter(
                x=i + offset + _jitter(g.loc[s.index, "plant"]),
                y=s,
                mode="markers",
                marker=dict(size=5, color=colour, line=dict(width=0)),
                customdata=[[b, p, pct(v, 0)] for p, v in zip(g.loc[s.index, "plant"], s, strict=True)],
                hovertemplate="%{customdata[0]} %{customdata[1]} · %{customdata[2]} vs own peak<extra></extra>",
                name=f"{group} plants",
                showlegend=False,
            )
        fig.add_scatter(
            x=x,
            y=med,
            mode="lines+markers",
            line=dict(color=colour, width=3),
            marker=dict(size=8, color=colour),
            name=group,
            customdata=[
                [b, "", pct(m, 0), f"{pct(a, 0, '')} to {pct(c, 0)}" if np.isfinite(a) else "–", k]
                for m, a, c, k in zip(med, lo, hi, n, strict=True)
            ],
            hovertemplate=f"{group} median ({b}, %{{customdata[4]}} plants) · %{{customdata[2]}} vs own peak · "
            "middle half %{customdata[3]}<extra></extra>",
        )
    sel = w[(w["bed"] == bed) & (w["plant"] == plant)]
    if len(sel):
        r = sel.iloc[0]
        y = [0.0, r["peak_to_trough"], r["peak_to_rewatered"]]
        note = " · W3: not in the bands" if bed == tc.SAMPLED_BED else ""
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=C["navy"], width=2.5),
            marker=dict(size=9, color=C["navy"], line=dict(width=2, color=C["surface"])),
            connectgaps=False,
            name=f"{bed} {plant}",
            customdata=[[bed, plant, pct(v, 0)] for v in y],
            hovertemplate=f"{bed} {plant} · %{{customdata[2]}} vs own peak{note}<extra></extra>",
        )
    fig.add_hline(y=0, line_width=1, line_color=C["gray-500"], layer="below")
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=12, t=12, b=30),
        hovermode="closest",
        clickmode="event",
        xaxis=dict(
            range=[-0.3, 2.3],
            tickmode="array",
            tickvals=x,
            ticktext=[STEP_TEXT[s] for s in STEPS],
            showgrid=False,
            fixedrange=True,
        ),
        yaxis=dict(title="%", fixedrange=True, ticksuffix=""),
        showlegend=False,  # the panel head carries the key: deficit, control, the selected plant
    )
    return fig


# ---------------------------------------------------------------- labels
def short_label(label: str) -> str:
    """'Leaf minus air temperature' -> 'Leaf − air temp.', 'Projected leaf area' -> 'Projected area'."""
    return (
        label.replace(" minus ", " − ")
        .replace(" temperature", " temp.")
        .replace("Projected leaf area", "Projected area")
    )
