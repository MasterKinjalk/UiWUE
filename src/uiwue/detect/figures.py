"""Plotly figures and small summaries for the Early warning, Sensor comparison and
Methods pages.

Plain functions of the derived/detect tables, so the same figure works in the
app, a notebook, or an exported PNG for the video. Colours are the design tokens
(design/tokens.json); the x axis is days since irrigation off (day 0 = 3 Sep 2025).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from uiwue.stats import ONSET_THRESHOLD

# Token values (design/tokens.json); read from the file when the repo is present.
_FALLBACK = {
    "surface": "#ffffff",
    "surface-2": "#f6f8f9",
    "mint": "#eef4ee",
    "sand": "#fbf1de",
    "gray": "#eef1f2",
    "ink": "#1e2a24",
    "ink-muted": "#5e6b64",
    "navy": "#0b2e3f",
    "rule": "#cfe0d4",
    "gray-500": "#8a959c",
    "teal-700": "#1c7293",
    "teal-600": "#3f7f93",
    "teal-200": "#bfe0e8",
    "amber-700": "#c98300",
    "amber-900": "#5a3e00",
    "forest-700": "#2e6b4f",
    "plum-700": "#6b3f7a",
}


def _tokens() -> dict:
    f = Path(__file__).resolve().parents[3] / "design" / "tokens.json"
    try:
        raw = {t["name"]: t["value"] for t in json.loads(f.read_text(encoding="utf-8"))["color"]["tokens"]}
    except (OSError, KeyError, ValueError):
        return dict(_FALLBACK)
    return {k: raw[v[1:-1]] if v.startswith("{") else v for k, v in raw.items()}


C = _tokens()

# Sensor identity, as in the Sensors mockup (IR teal, 3D forest); size is the grey baseline.
SENSOR_COLOURS = {
    "Thermal": C["teal-700"],
    "Depth": C["forest-700"],
    "RGB": C["plum-700"],
    "Hyperspectral": C["navy"],
    "size": C["gray-500"],
}
SENSOR_TAG = {"Thermal": "IR", "Depth": "3D", "RGB": "RGB", "Hyperspectral": "HSI"}
MODEL_SYMBOL = {
    "logistic": "circle",
    "boosted": "diamond",
    "fusion": "star",  # main track (uiwue.fusion): frozen encoders
    "fusion+lora_hsi": "hexagram",
    "fusion+lora_thermal": "star-diamond",
}
TEMPLATE = "plotly_white"
STRESS_END_DAY = 12  # rewatered 15 Sep 18:35
REWATER_DAY = 12.5
EVIDENCE = ((0.9, C["navy"]), (0.7, C["teal-600"]), (0.5, C["teal-200"]), (-1, C["gray"]))  # DESIGN 2.2
LOW_RES = ("_r160", "_r80")


def evidence_colour(ci_low: float) -> str:
    """Evidence ramp on the AUC lower bound: gray < 0.5, teal-200, teal-600, navy >= 0.9."""
    if ci_low is None or not np.isfinite(ci_low):
        return "transparent"
    return next(c for bar, c in EVIDENCE if ci_low >= bar)


def _theme(fig: go.Figure, height: int = 420) -> go.Figure:
    # app/components.py registers the shared "uiwue" template; use it when loaded.
    template = "uiwue" if "uiwue" in pio.templates else TEMPLATE
    fig.update_layout(template=template, height=height, margin=dict(l=56, r=16, t=24, b=48))
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font=dict(size=14, color=C["ink-muted"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _theme(fig, 240)


def signal_name(row) -> str:
    """'Leaf temperature (Thermal)'. Bed effects are shown as a chip, not in the name."""
    return f"{row['label']} ({row['sensor']})"


def is_low_res(signal: str) -> bool:
    return str(signal).endswith(LOW_RES)


def _flag(o: pd.DataFrame, col: str) -> pd.Series:
    return o[col].fillna(False).astype(bool) if col in o.columns else pd.Series(False, index=o.index)


def flagged(o: pd.DataFrame) -> pd.Series:
    """Placebo bed effect or a rule that fires before the cutoff."""
    return _flag(o, "bed_effect") | _flag(o, "fires_before_cutoff")


def ranked(onsets: pd.DataFrame, group: str = "all") -> pd.DataFrame:
    """Trait rows of a group with data, low-res thermal copies dropped, earliest robust onset first."""
    o = onsets[(onsets["group"] == group) & (onsets["kind"] == "trait") & (onsets["n_deficit"] > 0)]
    o = o[~o["signal"].map(is_low_res)].copy()
    rob = o["robust_onset_day"] if "robust_onset_day" in o.columns else o["onset_day"]
    o["_rob"] = rob.fillna(99)
    o["_on"] = o["onset_day"].fillna(99)
    return o.sort_values(["_rob", "_on", "early_auc"], ascending=[True, True, False]).drop(columns=["_rob", "_on"])


def default_signals(onsets: pd.DataFrame, group: str = "all") -> list[str]:
    """Earliest robust-onset trait per sensor that no check flags, plus the size baseline."""
    o = ranked(onsets, group)
    picks = list(o[~flagged(o) & ~o["is_size_baseline"].astype(bool)].groupby("sensor").head(1)["signal"])
    picks += list(o.loc[o["is_size_baseline"].astype(bool), "signal"])
    return list(dict.fromkeys(picks))


def headline(onsets: pd.DataFrame, group: str = "all") -> pd.Series | None:
    """The earliest robust onset among unflagged, non-size traits of the group."""
    o = ranked(onsets, group)
    o = o[~flagged(o) & ~o["is_size_baseline"].astype(bool)]
    rob = "robust_onset_day" if "robust_onset_day" in o.columns else "onset_day"
    o = o.dropna(subset=[rob])
    return None if o.empty else o.iloc[0]


# --------------------------------------------------------------------------- #
# Early warning
# --------------------------------------------------------------------------- #
def auc_curves(
    auc: pd.DataFrame, onsets: pd.DataFrame, signals, group: str = "all", *, envelope: bool = True
) -> go.Figure:
    """Daily AUC with its 95 % band for one highlighted signal (the first) and greyed
    others, the pre-cutoff envelope as a grey band and the robust onset as a navy line."""
    signals = [signals] if isinstance(signals, str) else list(signals or [])
    a = auc[(auc["group"] == group) & auc["signal"].isin(signals)]
    if a.empty:
        return empty_figure("No AUC curve for this selection")
    meta = onsets[onsets["group"] == group].drop_duplicates("signal").set_index("signal")
    fig = go.Figure()
    fig.add_vrect(x0=-10.5, x1=0, fillcolor=C["mint"], line_width=0, layer="below")
    fig.add_vrect(x0=0, x1=REWATER_DAY, fillcolor=C["sand"], line_width=0, layer="below")
    for x in (0, REWATER_DAY):
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=C["ink-muted"])
    fig.add_hline(y=ONSET_THRESHOLD, line_dash="dot", line_color=C["gray-500"], line_width=1)
    fig.add_hline(y=0.5, line_color=C["rule"], line_width=1)
    first = signals[0]
    if envelope and first in meta.index and np.isfinite(meta.loc[first].get("null_auc_max", np.nan)):
        env = float(meta.loc[first, "null_auc_max"])
        fig.add_hrect(y0=1 - env, y1=env, fillcolor=C["gray"], opacity=0.8, line_width=0, layer="below")
    for sig in reversed(signals):
        s = a[a["signal"] == sig].sort_values("day")
        if s.empty or sig not in meta.index:
            continue
        m = meta.loc[sig]
        main = sig == first
        colour = C["teal-700"] if main else C["gray-500"]
        name = "size" if m["is_size_baseline"] else str(m["label"])
        if main:
            fig.add_trace(
                go.Scatter(
                    x=np.r_[s["day"], s["day"][::-1]],
                    y=np.r_[s["ci_high"], s["ci_low"][::-1]],
                    fill="toself",
                    fillcolor="rgba(28,114,147,0.14)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        dates = [f"{d.day} {d:%b}" for d in pd.to_datetime(s["date"])]
        fig.add_trace(
            go.Scatter(
                x=s["day"],
                y=s["auc"],
                mode="lines+markers" if main else "lines",
                name=name,
                line=dict(color=colour, width=2.5 if main else 1.25, dash="dash" if m["is_size_baseline"] else "solid"),
                marker=dict(size=5),
                customdata=np.c_[s["ci_low"], s["ci_high"], s["n_deficit"], s["n_control"], dates],
                hovertemplate=(
                    "day %{x} · %{customdata[4]}<br>AUC %{y:.2f} (95 % CI %{customdata[0]:.2f}–%{customdata[1]:.2f})"
                    "<br>%{customdata[2]} deficit · %{customdata[3]} control plants"
                    f"<extra>{name}</extra>"
                ),
            )
        )
    m = meta.loc[first] if first in meta.index else None
    rob = m.get("robust_onset_day", np.nan) if m is not None else np.nan
    if m is not None and np.isfinite(rob):
        fig.add_vline(x=float(rob), line_color=C["navy"], line_dash="dash", line_width=1.5)
        fig.add_annotation(
            x=float(rob), y=1.04, yref="y", text="onset", showarrow=False, font=dict(size=12, color=C["navy"])
        )
    fig.update_xaxes(title="day", range=[-10.5, 18.5], zeroline=False, dtick=2)
    fig.update_yaxes(title="AUC", range=[0, 1.08])
    return _theme(fig, 380)


def onset_ranking(onsets: pd.DataFrame, group: str = "all", include_bed_effect: bool = True) -> go.Figure:
    """Dot plot: robust onset (filled) and the contract-rule onset (open) per signal,
    coloured by sensor; flagged signals drawn as open diamonds."""
    o = ranked(onsets, group)
    if not include_bed_effect:
        o = o[~flagged(o)]
    if o.empty:
        return empty_figure("No onset results")
    o = o.iloc[::-1]
    names = [("size · " if s else "") + str(lbl) for lbl, s in zip(o["label"], o["is_size_baseline"], strict=True)]
    colours = [
        SENSOR_COLOURS["size"] if s else SENSOR_COLOURS.get(x, C["gray-500"])
        for s, x in zip(o["is_size_baseline"], o["sensor"], strict=True)
    ]
    rob = o["robust_onset_day"] if "robust_onset_day" in o.columns else o["onset_day"]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=REWATER_DAY, fillcolor=C["sand"], line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=o["onset_day"],
            y=names,
            mode="markers",
            name="rule",
            marker=dict(symbol="circle-open", size=11, color=colours, line=dict(width=2)),
            hovertemplate="%{y}<br>contract rule: day %{x}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rob,
            y=names,
            mode="markers",
            name="robust",
            marker=dict(
                symbol=np.where(flagged(o), "diamond-open", "circle"), size=13, color=colours, line=dict(width=2)
            ),
            customdata=np.c_[o["null_auc_max"] if "null_auc_max" in o else np.full(len(o), np.nan), o["placebo_auc"]],
            hovertemplate=(
                "%{y}<br>robust onset: day %{x}<br>bar: largest watered-day AUC %{customdata[0]:.2f}"
                "<br>placebo AUC %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="onset day", range=[-0.5, 13], dtick=1)
    fig.update_yaxes(title=None)
    fig = _theme(fig, max(260, 26 * len(o) + 90))
    return fig


def recovery_summary(rec: pd.DataFrame | None, signals=None, group: str = "all") -> pd.DataFrame:
    """Median recovery ratio per trait over measurable deficit plants of the group."""
    if rec is None or rec.empty:
        return pd.DataFrame(columns=["signal", "label", "recovery", "n"])
    beds = {"all": ("W2", "W3"), "W2": ("W2",), "W3": ("W3",)}[group]
    r = rec[rec["plant_id"].str[:2].isin(beds) & rec["measurable"].astype(bool)]
    if signals is not None:
        r = r[r["signal"].isin(signals)]
    out = r.groupby(["signal", "label"], as_index=False).agg(
        recovery=("recovery_ratio", "median"), n=("recovery_ratio", "size")
    )
    return out


# --------------------------------------------------------------------------- #
# Sensor comparison
# --------------------------------------------------------------------------- #
def cost_performance(bench: pd.DataFrame, group: str = "all", metric: str = "early_auc") -> go.Figure:
    """One point per setup and model, cheapest setup on the left, n plants in the tick label.
    Open markers: the placebo or the pre-cutoff check flagged the setup."""
    b = bench[bench["group"] == group]
    if b.empty:
        return empty_figure("No benchmark rows for this selection")
    fig = go.Figure()
    for _, r in b[b["is_size_baseline"].astype(bool)].iterrows():
        fig.add_hline(y=r[metric], line_dash="dash", line_color=SENSOR_COLOURS["size"], line_width=1)
    fig.add_hline(y=0.5, line_color=C["rule"], line_width=1)
    rest = b[~b["is_size_baseline"].astype(bool)].copy()
    rest["tick"] = (
        rest["setup"]
        + "<br>"
        + rest["n_deficit"].astype(int).astype(str)
        + " · "
        + rest["n_control"].astype(int).astype(str)
    )
    order = rest.drop_duplicates("setup").sort_values("cost_rank")
    ticks = dict(zip(order["setup"], order["tick"], strict=True))
    colour = {"logistic": C["teal-700"], "boosted": C["navy"]}
    for model, sub in rest.groupby("model"):
        sub = sub.set_index("setup").reindex(order["setup"]).dropna(subset=["model"]).reset_index()
        flag = flagged(sub).to_numpy()
        err = metric == "early_auc"
        sym = MODEL_SYMBOL.get(model, "circle")
        fig.add_trace(
            go.Scatter(
                x=[ticks[s] for s in sub["setup"]],
                y=sub[metric],
                mode="markers",
                name=model,
                marker=dict(
                    size=12,
                    color=colour.get(model, C["plum-700"]),
                    symbol=[sym + ("-open" if f else "") for f in flag],
                    line=dict(width=2, color=colour.get(model, C["plum-700"])),
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["early_ci_high"] - sub[metric]) if err else None,
                    arrayminus=(sub[metric] - sub["early_ci_low"]) if err else None,
                    visible=err,
                    thickness=1,
                    color=C["gray-500"],
                ),
                customdata=np.c_[
                    sub["robust_onset_day"] if "robust_onset_day" in sub else sub["onset_day"],
                    sub["placebo_auc"],
                    np.where(flag, "flagged", "clean"),
                ],
                hovertemplate=(
                    "%{x}<br>AUC %{y:.2f}<br>robust onset day %{customdata[0]}"
                    "<br>placebo AUC %{customdata[1]:.2f} · %{customdata[2]}<extra>" + model + "</extra>"
                ),
            )
        )
    fig.update_xaxes(title=None, categoryorder="array", categoryarray=list(ticks.values()), tickangle=0)
    fig.update_yaxes(title="AUC", range=[0.3, 1.04])
    fig = _theme(fig, 420)
    fig.update_layout(margin=dict(l=56, r=16, t=24, b=96))
    return fig


# --------------------------------------------------------------------------- #
# Methods & limits
# --------------------------------------------------------------------------- #
def placebo_figure(placebo: pd.DataFrame, group: str = "all") -> go.Figure:
    """Placebo AUC per trait: circle = fine, triangle = rule fires on watered days,
    diamond = placebo bed effect."""
    p = placebo[(placebo["group"] == group) & ~placebo["signal"].map(is_low_res)].dropna(subset=["placebo_auc"])
    if p.empty:
        return empty_figure("No placebo results")
    p = p.assign(name=p["label"].astype(str) + " · " + p["sensor"].map(SENSOR_TAG).fillna(p["sensor"]))
    p = p.sort_values("placebo_auc")
    bed = _flag(p, "bed_effect").to_numpy()
    day = _flag(p, "fires_before_cutoff").to_numpy()
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=1 - ONSET_THRESHOLD, fillcolor=C["sand"], line_width=0, layer="below")
    fig.add_vrect(x0=ONSET_THRESHOLD, x1=1, fillcolor=C["sand"], line_width=0, layer="below")
    fig.add_vline(x=0.5, line_color=C["gray-500"], line_width=1)
    for label, mask, sym, col in (
        ("fine", ~bed & ~day, "circle", C["teal-700"]),
        ("day effect", ~bed & day, "triangle-up", C["amber-700"]),
        ("bed effect", bed, "diamond", C["amber-900"]),
    ):
        q = p[mask]
        if q.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=q["placebo_auc"],
                y=q["name"],
                mode="markers",
                name=label,
                marker=dict(size=11, symbol=sym, color=col),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=q["placebo_ci_high"] - q["placebo_auc"],
                    arrayminus=q["placebo_auc"] - q["placebo_ci_low"],
                    thickness=1,
                    color=C["gray-500"],
                ),
                customdata=np.c_[q["placebo_ci_low"], q["placebo_ci_high"]],
                hovertemplate="%{y}<br>placebo AUC %{x:.2f} (95 % CI %{customdata[0]:.2f}–%{customdata[1]:.2f})"
                + f"<extra>{label}</extra>",
            )
        )
    fig.update_xaxes(title="AUC, 24–29 Aug", range=[0, 1])
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=list(p["name"]))
    return _theme(fig, max(280, 24 * len(p) + 90))
