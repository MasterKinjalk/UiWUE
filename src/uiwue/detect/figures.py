"""Plotly figures for the Early warning, Sensor comparison and Methods pages.

Plain functions of the derived/detect tables, so the same figure works in the
app, a notebook, or an exported PNG for the video.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from uiwue.stats import ONSET_THRESHOLD

SENSOR_COLOURS = {
    "Thermal": "#d6604d",
    "Hyperspectral": "#7b3294",
    "RGB": "#1b7837",
    "Depth": "#2166ac",
    "size": "#636363",
}
MODEL_SYMBOL = {
    "logistic": "circle",
    "boosted": "diamond",
    "fusion": "star",  # main track (uiwue.fusion): frozen encoders
    "fusion+lora_hsi": "hexagram",
    "fusion+lora_thermal": "star-diamond",
}
TEMPLATE = "plotly_white"
STRESS_END_DAY = 12  # rewatered 15 Sep 18:35


def _theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    # app/components.py registers the shared "uiwue" template; use it when loaded.
    template = "uiwue" if "uiwue" in pio.templates else TEMPLATE
    fig.update_layout(
        template=template,
        margin=dict(l=60, r=20, t=50 if title else 20, b=50),
        title=title,
        legend=dict(orientation="h", y=-0.2),
        height=460,
    )
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color="#666")
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _theme(fig)


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def signal_name(row) -> str:
    name = f"{row['label']} ({row['sensor']})"
    return name + " - bed effect" if bool(row.get("bed_effect", False)) else name


def default_signals(onsets: pd.DataFrame, group: str = "all") -> list[str]:
    """Earliest-onset (then highest early AUC) trait per sensor, plus the size baseline."""
    o = onsets[(onsets["group"] == group) & (onsets["kind"] == "trait")]
    o = o.assign(_on=o["onset_day"].fillna(99)).sort_values(["_on", "early_auc"], ascending=[True, False])
    picks = list(o[~o["bed_effect"]].groupby("sensor").head(1)["signal"])
    picks += list(o.loc[o["is_size_baseline"], "signal"])
    return list(dict.fromkeys(picks))


# --------------------------------------------------------------------------- #
# Early warning
# --------------------------------------------------------------------------- #
def auc_curves(auc: pd.DataFrame, onsets: pd.DataFrame, signals: list[str], group: str = "all") -> go.Figure:
    a = auc[(auc["group"] == group) & auc["signal"].isin(signals)]
    if a.empty:
        return empty_figure("No AUC curves for this selection yet.")
    meta = onsets[onsets["group"] == group].set_index("signal")
    fig = go.Figure()
    fig.add_vrect(
        x0=0,
        x1=STRESS_END_DAY,
        fillcolor="#fdf1e3",
        opacity=0.6,
        line_width=0,
        layer="below",
        annotation_text="no irrigation",
        annotation_position="top left",
    )
    fig.add_hline(
        y=ONSET_THRESHOLD,
        line_dash="dot",
        line_color="#444",
        annotation_text=f"onset bar: CI low > {ONSET_THRESHOLD}",
        annotation_position="bottom right",
    )
    fig.add_hline(y=0.5, line_color="#bbb", line_width=1)
    for sig in signals:
        s = a[a["signal"] == sig].sort_values("day")
        if s.empty or sig not in meta.index:
            continue
        m = meta.loc[sig]
        colour = SENSOR_COLOURS["size"] if m["is_size_baseline"] else SENSOR_COLOURS.get(m["sensor"], "#333")
        name = ("Size only: " if m["is_size_baseline"] else "") + signal_name(m)
        dash = "dash" if m["is_size_baseline"] else ("dot" if m["bed_effect"] else "solid")
        fig.add_trace(
            go.Scatter(
                x=np.r_[s["day"], s["day"][::-1]],
                y=np.r_[s["ci_high"], s["ci_low"][::-1]],
                fill="toself",
                fillcolor=_rgba(colour, 0.12),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=sig,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=s["day"],
                y=s["auc"],
                mode="lines+markers",
                name=name,
                legendgroup=sig,
                line=dict(color=colour, dash=dash, width=2.5),
                marker=dict(size=5),
                customdata=np.c_[s["ci_low"], s["ci_high"], s["n_deficit"], s["n_control"]],
                hovertemplate=(
                    "day %{x}<br>AUC %{y:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                    "<br>%{customdata[2]} deficit vs %{customdata[3]} control plants"
                    f"<extra>{name}</extra>"
                ),
            )
        )
        if np.isfinite(m["onset_day"]):
            y = s.loc[s["day"] == m["onset_day"], "auc"]
            fig.add_trace(
                go.Scatter(
                    x=[m["onset_day"]],
                    y=[y.iloc[0] if len(y) else ONSET_THRESHOLD],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color=colour, line=dict(color="white", width=1)),
                    showlegend=False,
                    legendgroup=sig,
                    hovertemplate=f"onset: day {int(m['onset_day'])}<extra>{name}</extra>",
                )
            )
    sym = meta["symptom_day"].dropna()
    if len(sym):
        fig.add_vline(
            x=float(sym.iloc[0]),
            line_color="#8c510a",
            line_dash="dash",
            annotation_text="first visible symptom (median)",
            annotation_position="top right",
        )
    fig.update_xaxes(title="days since irrigation stopped (3 Sep 2025)", zeroline=False)
    fig.update_yaxes(title="AUC, deficit vs control plants", range=[0, 1.02])
    return _theme(fig)


def onset_ranking(onsets: pd.DataFrame, group: str = "all", include_bed_effect=False) -> go.Figure:
    o = onsets[(onsets["group"] == group) & (onsets["kind"] == "trait")].copy()
    if not include_bed_effect:
        o = o[~o["bed_effect"]]
    if o.empty:
        return empty_figure("No onset results yet.")
    o["name"] = o.apply(signal_name, axis=1)
    o["_on"] = o["onset_day"].fillna(STRESS_END_DAY + 1)
    o = o.sort_values(["_on", "early_auc"], ascending=[False, True])
    detected = o["onset_day"].notna()
    colours = [
        SENSOR_COLOURS["size"] if s else SENSOR_COLOURS.get(x, "#333")
        for s, x in zip(o["is_size_baseline"], o["sensor"], strict=False)
    ]
    fig = go.Figure(
        go.Bar(
            x=o["_on"],
            y=o["name"],
            orientation="h",
            marker=dict(color=colours, opacity=np.where(detected, 1.0, 0.25)),
            text=np.where(detected, "day " + o["onset_day"].fillna(0).astype(int).astype(str), "not detected"),
            textposition="outside",
            customdata=np.c_[o["early_auc"], o["lead_vs_symptom"], o["placebo_auc"]],
            hovertemplate=(
                "%{y}<br>early AUC (days 1-4) %{customdata[0]:.2f}"
                "<br>lead vs symptom %{customdata[1]} days"
                "<br>placebo AUC %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    sym = o["symptom_day"].dropna()
    if len(sym):
        fig.add_vline(
            x=float(sym.iloc[0]),
            line_color="#8c510a",
            line_dash="dash",
            annotation_text="visible symptoms",
            annotation_position="top",
        )
    fig.update_xaxes(title="onset day (earlier is better)", range=[0, STRESS_END_DAY + 2.5])
    fig.update_yaxes(title=None)
    fig = _theme(fig)
    fig.update_layout(height=max(300, 28 * len(o) + 120), showlegend=False)
    return fig


# --------------------------------------------------------------------------- #
# Sensor comparison
# --------------------------------------------------------------------------- #
def cost_performance(bench: pd.DataFrame, group: str = "all", metric: str = "early_auc") -> go.Figure:
    b = bench[bench["group"] == group]
    if b.empty:
        return empty_figure("No benchmark results yet.")
    fig = go.Figure()
    size = b[b["is_size_baseline"]]
    for _, r in size.iterrows():
        fig.add_hline(
            y=r[metric],
            line_dash="dash",
            line_color=SENSOR_COLOURS["size"],
            annotation_text=f"size only ({r['model']})",
            annotation_position="bottom right",
        )
    fig.add_hline(y=0.5, line_color="#bbb", line_width=1)
    rest = b[~b["is_size_baseline"]]
    order = rest.drop_duplicates("setup").sort_values("cost_rank")["setup"].tolist()
    for model, sub in rest.groupby("model"):
        sub = sub.set_index("setup").reindex(order).dropna(subset=["model"]).reset_index()
        err = metric == "early_auc"
        fig.add_trace(
            go.Scatter(
                x=sub["setup"],
                y=sub[metric],
                mode="markers",
                name=model,
                marker=dict(
                    size=12,
                    symbol=[MODEL_SYMBOL.get(model, "circle") + ("-open" if be else "") for be in sub["bed_effect"]],
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["early_ci_high"] - sub[metric]) if err else None,
                    arrayminus=(sub[metric] - sub["early_ci_low"]) if err else None,
                    visible=err,
                    thickness=1,
                ),
                customdata=np.c_[
                    sub["onset_day"], sub["placebo_auc"], sub["n_deficit"], sub["n_control"], sub["bed_effect"]
                ],
                hovertemplate=(
                    "%{x}<br>AUC %{y:.2f}<br>onset day %{customdata[0]}"
                    "<br>placebo AUC %{customdata[1]:.2f}"
                    "<br>%{customdata[2]} deficit vs %{customdata[3]} control plants"
                    "<br>bed effect: %{customdata[4]}<extra>" + model + "</extra>"
                ),
            )
        )
    label = {"early_auc": "early AUC (days 1-4)", "full_auc": "AUC (days 1-12)"}[metric]
    fig.update_xaxes(
        title="sensor setup, cheapest on the left", categoryorder="array", categoryarray=order, tickangle=-35
    )
    fig.update_yaxes(title=label, range=[0.3, 1.02])
    return _theme(fig)


# --------------------------------------------------------------------------- #
# Methods & limits
# --------------------------------------------------------------------------- #
def placebo_figure(placebo: pd.DataFrame, group: str = "all") -> go.Figure:
    p = placebo[placebo["group"] == group].copy()
    if p.empty:
        return empty_figure("No placebo results yet.")
    p["name"] = p["label"].astype(str) + " (" + p["sensor"].astype(str) + ")"
    p = p.sort_values("placebo_auc")
    fig = go.Figure(
        go.Scatter(
            x=p["placebo_auc"],
            y=p["name"],
            mode="markers",
            marker=dict(size=10, color=np.where(p["bed_effect"], "#b2182b", "#4d4d4d")),
            error_x=dict(
                type="data",
                symmetric=False,
                array=p["placebo_ci_high"] - p["placebo_auc"],
                arrayminus=p["placebo_auc"] - p["placebo_ci_low"],
                thickness=1,
            ),
            hovertemplate="%{y}<br>placebo AUC %{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.5, line_color="#bbb")
    for x in (1 - ONSET_THRESHOLD, ONSET_THRESHOLD):
        fig.add_vline(x=x, line_dash="dot", line_color="#b2182b")
    fig.update_xaxes(title="AUC before any plant was stressed (24-29 Aug); 0.5 = beds alike", range=[0, 1])
    fig = _theme(fig)
    fig.update_layout(height=max(300, 26 * len(p) + 120), showlegend=False)
    return fig
