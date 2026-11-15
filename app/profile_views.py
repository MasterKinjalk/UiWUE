"""Data and figures for the Plant profile and Bed map pages (kept apart from the pages so they can be tested alone)."""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components import BED, INK, SERIES, label_of, mark_stress_period
from plotly.subplots import make_subplots

from uiwue.core import (
    CUTOFF,
    DERIVED,
    TRUTH_CLIP_ONSET,
    TRUTH_FIRST_SYMPTOM,
    all_plants,
    catalogue,
    read_table,
)

WATER_BAND, RED_EDGE = (950, 990), (690, 750)  # nm; the 970 nm water dip and the red edge
PROB_COLUMN = re.compile(r"^detect_.*prob")
BEDS = ("W1", "W2", "W3")
SCALES = {1: "YlOrRd", -1: "YlOrRd_r", 0: "Viridis"}  # stress traits: warm = more stress; other traits: neutral


# ------------------------------------------------------------------ plant profile
def day_of(dates):
    """Days since irrigation was switched off (day 0)."""
    return (pd.to_datetime(dates) - pd.Timestamp(CUTOFF)).dt.days


def daily(df, column):
    """One value per plant and day: the median of the 12 h and 15 h runs (Master-Plan 6.5)."""
    d = df[df.hour.isin((12, 15))].dropna(subset=[column])
    d = d.groupby(["bed", "plant", "date"], as_index=False)[column].median()
    return d.assign(day=day_of(d.date))


def load_spectra():
    """(spectra, wavelengths) from derived/spectral, or None until the spectral module has pushed them.

    spectra_runs holds one row per run with `hsi_spectrum`, the plant's mean reflectance as a list of floats
    in the order of the `wavelength_nm` column of derived/spectral/wavelengths.csv.
    """
    spec, wl = read_table("spectral", "spectra_runs"), DERIVED / "spectral" / "wavelengths.csv"
    if spec is None or "hsi_spectrum" not in spec or not wl.exists():
        return None
    return spec, pd.read_csv(wl).wavelength_nm.to_numpy()


def plant_row(bed, plant):
    plants = read_table("truth", "plants")
    if plants is None:
        return None
    r = plants[(plants.bed == bed) & (plants.plant == plant)]
    return r.iloc[0] if len(r) else None


def plant_options(df, bed):
    """Plants of a bed that have runs; a star marks the ones with a leaf clip."""
    plants = read_table("truth", "plants")
    clips = set(plants[plants.truth_has_clip & (plants.bed == bed)].plant) if plants is not None else set()
    names = sorted(df[df.bed == bed].plant.unique())
    return [{"label": f"{p} ★" if p in clips else p, "value": p} for p in names]


def fmt_date(d):
    return "none" if d is None or pd.isna(d) else f"{pd.Timestamp(d):%d %b}"


# ------------------------------------------------------------------ figures
def trace_figure(df, bed, plant, trait):
    """The plant's daily value against the control bed's median (and its own bed's, for a deficit plant)."""
    d = daily(df, trait)
    mine = d[(d.bed == bed) & (d.plant == plant)].sort_values("day")
    if mine.empty:
        return None
    fig = go.Figure()
    beds = ["W1"] if bed == "W1" else ["W1", bed]
    for b in beds:
        m = d[d.bed == b].groupby("day")[trait].median()
        fig.add_scatter(
            x=m.index, y=m.values, mode="lines", line=dict(color=BED[b], width=2, dash="dot"), name=f"{b} median"
        )
    fig.add_scatter(
        x=mine.day,
        y=mine[trait],
        mode="lines+markers",
        line=dict(color=INK["primary"], width=2.5),
        marker=dict(size=7),
        name=f"{bed} {plant}",
    )
    fig.update_layout(
        title=f"{label_of(trait)}: {bed} {plant} against the beds' median",
        xaxis_title="days since irrigation stopped",
        yaxis_title=label_of(trait),
        height=360,
    )
    return mark_stress_period(fig)


def turgor_figure(turgor, bed, plant, row):
    """Leaf-clip turgor shift through the stress, with the clip onset and the first written symptom."""
    t = turgor[(turgor.bed == bed) & (turgor.plant == plant)].sort_values(["date", "hour"])
    if t.empty:
        return None
    x = day_of(t.date) + t.hour / 24
    fig = go.Figure(
        go.Scatter(x=x, y=t.truth_turgor_shift, mode="lines", line=dict(color=BED[bed], width=1.5), name="clip shift")
    )
    if row is not None:
        for col, text in ((TRUTH_CLIP_ONSET, "clip shift begins"), (TRUTH_FIRST_SYMPTOM, "first symptom noted")):
            if pd.notna(row[col]):
                fig.add_vline(
                    x=(pd.Timestamp(row[col]) - pd.Timestamp(CUTOFF)).days,
                    line_dash="dash",
                    line_color=INK["secondary"],
                    annotation_text=text,
                    annotation_position="bottom right",
                    annotation_font=dict(size=11, color=INK["secondary"]),
                )
    fig.update_layout(
        title="Leaf-clip turgor (higher = less turgor)",
        xaxis_title="days since irrigation stopped",
        yaxis_title="clip shift (raw units)",
        showlegend=False,
        height=330,
    )
    return mark_stress_period(fig)


def rwc_figure(rwc, bed, plant):
    """Every sampled W3 leaf's relative water content, with this plant's leaves marked."""
    if bed != "W3" or rwc is None:
        return None
    mine = rwc[rwc.plant == plant]
    if mine.empty:
        return None
    fig = go.Figure(
        go.Scatter(
            x=day_of(rwc.date),
            y=rwc.truth_rwc,
            mode="markers",
            marker=dict(color=INK["muted"], size=6, opacity=0.6),
            name="other W3 leaves",
        )
    )
    for col, name, symbol in (
        ("truth_rwc", "this plant, first leaf", "diamond"),
        ("truth_rwc_leaf2", "second leaf", "square"),
    ):
        m = mine.dropna(subset=[col])
        fig.add_scatter(
            x=day_of(m.date),
            y=m[col],
            mode="markers",
            marker=dict(color=SERIES[1], size=13, symbol=symbol, line=dict(width=1.5, color=INK["primary"])),
            name=name,
        )
    fig.update_layout(
        title=f"Relative water content of the sampled leaves; {plant} marked",
        xaxis_title="days since irrigation stopped",
        yaxis_title="relative water content (%)",
        height=330,
    )
    return mark_stress_period(fig)


def spectrum_figure(spec, wl, bed, plant, dates):
    """The plant's mean reflectance spectrum on the chosen dates (15 h run, else the latest slot that day)."""
    s = spec[(spec.bed == bed) & (spec.plant == plant)]
    fig = go.Figure()
    for i, d in enumerate(sorted(dates)):
        day = s[s.date == pd.Timestamp(d)].sort_values("hour")
        if day.empty:
            continue
        run = day[day.hour == 15].iloc[0] if (day.hour == 15).any() else day.iloc[-1]
        fig.add_scatter(
            x=wl,
            y=np.asarray(run.hsi_spectrum, dtype=float),
            mode="lines",
            line=dict(color=SERIES[i % len(SERIES)], width=2),
            name=f"{pd.Timestamp(d):%d %b} (day {(pd.Timestamp(d) - pd.Timestamp(CUTOFF)).days})",
        )
    if not fig.data:
        return None
    for (lo, hi), text in ((RED_EDGE, "red edge"), (WATER_BAND, "water band")):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=INK["grid"], opacity=0.5, line_width=0, annotation_text=text)
    fig.update_layout(
        title=f"Reflectance spectrum, {bed} {plant}",
        xaxis_title="wavelength (nm)",
        yaxis_title="reflectance",
        height=360,
    )
    return fig


# ------------------------------------------------------------------ bed map
def position(plant):
    """Grid position of a pot: row letter down the bed (A at the top), column digit across (1-8)."""
    return int(plant[1:]), "ABCDEFGHIJKLMNOPQR".index(plant[0])


def dates_of(df):
    return sorted(df.date.unique())


def color_range(df, trait):
    """One colour range per trait over every run, so the beds and days stay comparable."""
    v = df[trait].dropna()
    return (float(v.quantile(0.02)), float(v.quantile(0.98))) if len(v) else (0.0, 1.0)


def bed_map_figure(df, trait, date, hour, clips=frozenset()):
    """The three beds side by side, one dot per pot; pots without a reading are grey rings."""
    direction = catalogue().set_index("column").direction.get(trait, 0)
    lo, hi = color_range(df, trait)
    run = df[(df.date == pd.Timestamp(date)) & (df.hour == hour)].dropna(subset=[trait])
    fig = make_subplots(rows=1, cols=3, subplot_titles=BEDS, horizontal_spacing=0.04)
    for c, bed in enumerate(BEDS, start=1):
        pots = pd.DataFrame({"plant": all_plants()})
        pots[["x", "y"]] = [position(p) for p in pots.plant]
        pots = pots.merge(run[run.bed == bed][["plant", trait]], on="plant", how="left")
        have = pots[trait].notna()
        fig.add_scatter(
            x=pots.x[~have],
            y=pots.y[~have],
            mode="markers",
            marker=dict(size=15, symbol="circle-open", color=INK["muted"], line=dict(width=1.5)),
            customdata=pots.plant[~have],
            hovertemplate=f"{bed} %{{customdata}}: no reading<extra></extra>",
            showlegend=False,
            row=1,
            col=c,
        )
        outline = [INK["primary"] if (bed, p) in clips else "rgba(0,0,0,0)" for p in pots.plant[have]]
        fig.add_scatter(
            x=pots.x[have],
            y=pots.y[have],
            mode="markers",
            marker=dict(
                size=16,
                color=pots[trait][have],
                coloraxis="coloraxis",
                line=dict(width=2, color=outline),
            ),
            customdata=np.c_[pots.plant[have], pots[trait][have].round(2)],
            hovertemplate=f"{bed} %{{customdata[0]}}: %{{customdata[1]}}<extra></extra>",
            showlegend=False,
            row=1,
            col=c,
        )
        fig.update_xaxes(range=[0.2, 8.8], dtick=1, showgrid=False, row=1, col=c)
        fig.update_yaxes(
            range=[17.6, -0.6],
            tickmode="array",
            tickvals=list(range(18)),
            ticktext=list("ABCDEFGHIJKLMNOPQR") if c == 1 else [""] * 18,
            showgrid=False,
            row=1,
            col=c,
        )
    day = (pd.Timestamp(date) - pd.Timestamp(CUTOFF)).days
    fig.update_layout(
        title=f"{label_of(trait)}, {pd.Timestamp(date):%d %b} {hour} h (day {day} since irrigation stopped)",
        coloraxis=dict(colorscale=SCALES[direction], cmin=lo, cmax=hi, colorbar=dict(title=label_of(trait), len=0.8)),
        height=620,
        margin=dict(l=40, r=20, t=90, b=30),
    )
    return fig
