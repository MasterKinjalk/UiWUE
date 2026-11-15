"""Data and figures for Plant profile, Bed map and Export (kept apart from the pages so they can be tested alone).

Values are shown as change from the plant's own baseline (DESIGN 7.2): the median over the baseline
window (30 Aug – 2 Sep), as a percentage for size traits and as a difference for the rest. Scores and
probabilities (detect_, diurnal_), the leaf-clip shift and the environment are shown as they are.
Only runs that pass their module's QC are used (Master-Plan 6.5); a QC-failed run is "no scan".
The 3D numbers (the three 3D small multiples and detail chart, the leaf-area readout, the 3D export
table) come from ``threed_canopy``, the data layer of the 3D canopy page, so one plant shows one number
on both pages: the plant-day mean of its QC-pass 9–18 h runs, as % of its own baseline. The Bed map
keeps one run per square, against the same slot's baseline, and says so on hover.
"""

import time
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import threed_canopy as tc
from components import BAND, GRAPH_CONFIG, C, key_traits, mark_stress_period, page_href, placebo_chip, plain, tech
from components import bed_effect_chip as _bed_effect_chip
from dash import dcc, html

from uiwue.core import (
    BASELINE,
    CUTOFF,
    DERIVED,
    REWATER,
    all_plants,
    catalogue,
    load_traits,
    normalize_keys,
    read_table,
)
from uiwue.stats._design import qc_pass

BEDS = ("W1", "W2", "W3")
ROWS = "ABCDEFGHIJKLMNOPQR"
BASE_DAYS = ((BASELINE[0] - CUTOFF).days, (BASELINE[1] - CUTOFF).days)  # −4 .. −1
END_DAY = (REWATER - CUTOFF).days  # day 12: the last day without water
X_RANGE = (-10.5, 19.5)
SOURCE = {  # column prefix -> (source tag, module folder, sensor name)
    "thermal_": ("IR", "thermal", "thermal camera"),
    "depth_": ("3D", "threed", "depth camera"),
    "rgb_": ("RGB", "rgb", "RGB camera"),
    "hsi_": ("HSI", "spectral", "hyperspectral camera"),
    "truth_": ("CLIP", "truth", "leaf clip"),
    "detect_": ("MODEL", "detect", "detection model"),
    "diurnal_": ("IR", "diurnal", "thermal score"),
    "env_": ("ENV", "env", "environment"),
}
# The app's only 3D traits: the three kept traits of the 3D canopy page (leaf area, height p75, hull volume), with
# its labels. The size-only baseline (projected area, DESIGN 7.6) is not one of them: it is the `size` chip beside
# the detection result (stress probability).
THREED = tuple(("3D", t.column, t.label) for t in tc.TRAITS.values())
THREED_COLUMNS = frozenset(c for _, c, _ in THREED)
THREED_KEY = {t.column: k for k, t in tc.TRAITS.items()}  # contract column -> threed_canopy's short key
THREED_BEDS = tc.BEDS  # the 3D canopy page's bed numbers: W1 (control) against W2 (deficit)
THREED_LINK_BEDS = tc.SHOWN_BEDS  # the plants the 3D canopy page shows on their own: W1, W2 and W3
# Headline traits per sensor for the small multiples (tag, column, short label)
HEADLINES = (
    ("IR", "thermal_leaf_ref_dt", "Leaf − control leaf"),
    ("IR", "diurnal_score_ref", "Thermal score"),
    *THREED,
    ("RGB", "rgb_ngrdi", "Green–red index"),  # RGB's one clean signal (greenness is flagged: a bed effect)
    ("CLIP", "truth_turgor_shift", "Leaf-clip shift"),
    ("MODEL", "detect_stress_prob", "Stress probability"),
    ("MODEL", "detect_fusion_v2_prob", "Fusion probability"),  # out of fold on 171 plants, 5-fold mean on 45 held-out
)
SIZE_UNITS = {"mm", "cm²", "cm³", "count"}
RAW_PREFIXES = ("detect_", "diurnal_", "env_")
RAW_COLUMNS = {"truth_turgor_shift", "truth_turgor_raw"}
STRESS = "detect_stress_prob"


# ------------------------------------------------------------------ catalogue helpers
def retry(fn, *args, tries=4, wait=0.4):
    """Call fn, retrying a few times: another process may be rewriting a derived table right now."""
    for i in range(tries):
        try:
            return fn(*args)
        except (OSError, ValueError, KeyError):  # pyarrow raises ArrowInvalid (a ValueError) mid-write
            if i == tries - 1:
                raise
            time.sleep(wait)
    return None


def traits():
    return retry(load_traits)


@lru_cache(maxsize=1)
def _cat_rows(_id):
    return {r.column: r for r in catalogue().drop_duplicates("column").itertuples(index=False)}


def cat_row(column):
    """The catalogue row of a column (label, unit, direction, description, module), or None."""
    return _cat_rows(id(catalogue())).get(column)


def unit_text(column):
    """The catalogue unit, or '' for a dimensionless index (blank unit) or an unknown column."""
    r = cat_row(column)
    return r.unit if r is not None and isinstance(r.unit, str) and r.unit.strip() else ""


def label_text(column):
    """The signal's plain name, the one every page uses (``components.plain``)."""
    r = cat_row(column)
    return plain(column, r.label if r is not None and isinstance(r.label, str) else column)


def label_unit(column):
    """'Leaf temperature (°C)', or 'Green–red index' when the unit is blank (never '(nan)')."""
    u = unit_text(column)
    return f"{label_text(column)} ({u})" if u else label_text(column)


def direction_of(column):
    r = cat_row(column)
    return int(r.direction) if r is not None and pd.notna(r.direction) else 0


def source_of(column):
    return next((v for k, v in SOURCE.items() if column.startswith(k)), ("", "", ""))


def mode_of(column):
    """'raw' (shown as is), 'pct' (size: % of baseline) or 'diff' (difference from baseline)."""
    if column.startswith(RAW_PREFIXES) or column in RAW_COLUMNS:
        return "raw"
    return "pct" if column.startswith("depth_") and unit_text(column) in SIZE_UNITS else "diff"


def change_unit(column):
    return "%" if mode_of(column) == "pct" else unit_text(column)


def trait_choices(df, stress_only=False, rows=None):
    """Dropdown options for catalogue traits with numeric values in df (or in `rows`, one plant's runs); of the 3D
    traits only the three the app shows (THREED), not the dropped noisy ones."""
    cat = catalogue().drop_duplicates("column")
    cat = cat[cat.column.isin(key_traits(cat.column))]  # the app's short list: no flagged, late or unscored signal
    have = rows if rows is not None else df
    out = []
    for r in cat.itertuples():
        if r.column not in have or r.unit == "date" or not pd.api.types.is_numeric_dtype(have[r.column]):
            continue
        if r.column.startswith("depth_") and r.column not in THREED_COLUMNS:
            continue
        if stress_only and r.direction == 0:
            continue
        if have[r.column].notna().any():
            out.append({"label": f"{source_of(r.column)[0]} · {label_unit(r.column)}", "value": r.column})
    return out


def fmt(x, unit="", digits=1, sign=True):
    """'+4.7 °C' with a real minus sign, '–' when missing; no unit bracket for a blank unit."""
    if x is None or pd.isna(x):
        return "–"
    if round(float(x), digits) == 0:
        x, sign = 0.0, False  # never "−0.0" or "+0": no change is unsigned
    s = (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−")
    return f"{s} {unit}".strip() if unit != "%" else f"{s} %"


def digits_for(values):
    v = pd.Series(values).abs().dropna()
    m = v.median() if len(v) else 1
    return 0 if m >= 20 else 1 if m >= 1 else 2


def day_text(day):
    d = CUTOFF + timedelta(days=int(day))
    return f"Day {int(day)} · {d.day} {d:%b}".replace("-", "−")


def fmt_date(d):
    return "–" if d is None or pd.isna(d) else f"{pd.Timestamp(d).day} {pd.Timestamp(d):%b}"


def day_of(dates):
    """Days since irrigation was switched off (day 0)."""
    return (pd.to_datetime(dates) - pd.Timestamp(CUTOFF)).dt.days


# ------------------------------------------------------------------ plant data
def usable(df, column):
    """True for the runs whose value of `column` counts: it has one and the run passed its module's QC
    (``<prefix>qc_pass``; a module without that flag passes). The tables keep the values of QC-failed runs."""
    return df[column].notna() & qc_pass(df, column)


def daily(df, column):
    """One value per plant and day: the median of the 12 h and 15 h runs that pass QC (Master-Plan 6.5)."""
    d = df[df.hour.isin((12, 15)) & usable(df, column)]
    d = d.groupby(["bed", "plant", "date"], as_index=False)[column].median()
    return d.assign(day=day_of(d.date))


def threed_series(column):
    """A kept 3D trait as the 3D canopy page computes it (``threed_canopy.plant_days``, every bed, each plant on its
    own baseline): one row per plant and day with a value, the plant-day mean of its QC-pass 9–18 h runs under the
    column's name and its % change from the plant's baseline (median of its days −4 to −1) as `chg`. None without
    the 3D table."""
    p = tc.plant_days(beds=tc.SHOWN_BEDS)
    if p is None:
        return None
    k = THREED_KEY[column]
    d = p.loc[p[k].notna(), ["bed", "plant", "treatment", "date", "day", k, f"{k}_pct"]]
    return d.rename(columns={k: column, f"{k}_pct": "chg"}).reset_index(drop=True)


def with_change(d, column, by=("bed", "plant")):
    """Adds `chg`: change from each plant's own baseline (median over the baseline days), or the raw value."""
    mode = mode_of(column)
    if mode == "raw":
        return d.assign(chg=d[column])
    in_base = d.day.between(*BASE_DAYS)
    base = d[in_base].groupby(list(by))[column].median().rename("_base")
    d = d.merge(base, left_on=list(by), right_index=True, how="left")
    chg = (d[column] / d._base - 1) * 100 if mode == "pct" else d[column] - d._base
    return d.drop(columns="_base").assign(chg=chg.where(np.isfinite(chg)))


_SERIES = {}


def series(df, column):
    """Daily change-from-baseline series for every plant (cached per table and column). The kept 3D traits come
    from the 3D canopy page's plant-day table instead (``threed_series``), so both pages show one number."""
    if column in THREED_KEY:
        return threed_series(column)  # threed_canopy caches it per table file
    key = (id(df), column)
    if key not in _SERIES:
        _SERIES[key] = with_change(daily(df, column), column) if column in df else None
    return _SERIES[key]


def plant_rows(df, bed, plant):
    return df[(df.bed == bed) & (df.plant == plant)]


def no_chart_reason(column):
    """Hover of a sensor panel whose plant has runs but no daily change to draw (no QC-pass run, or no baseline)."""
    runs = "9–18 h runs" if column in THREED_KEY else "12 h and 15 h runs"
    return f"no daily value with a baseline: none of its {runs} passed QC, or none did on 30 Aug – 2 Sep"


def sensor_state(df, bed, plant, column):
    """('ok' | 'noscan' | 'notable', hover). noscan: the plant is in the module's table but no run has a
    usable value; notable: the module has no rows for this plant (or no table at all)."""
    _, module, sensor = source_of(column)
    if column not in df:
        return "notable", f"{sensor}: no table in derived/{module}/ yet"
    rows = plant_rows(df, bed, plant)
    if usable(rows, column).any():
        return "ok", ""
    if column.startswith("truth_turgor"):
        return "notable", "no leaf clip on this plant"
    prefix = column.split("_")[0] + "_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(rows) and rows[cols].notna().any().any():
        return "noscan", f"{sensor}: {bed} {plant} is in derived/{module}/ but no run passed QC"
    return "notable", f"{sensor}: no rows for {bed} {plant} in derived/{module}/ (not imaged by this sensor)"


def plant_options(df, bed):
    """Plants of a bed that have runs, in grid order; a star marks the ones with a leaf clip."""
    plants = read_table("truth", "plants")
    clips = set(plants[plants.truth_has_clip & (plants.bed == bed)].plant) if plants is not None else set()
    have = set(df[df.bed == bed].plant)
    return [{"label": f"{p} ★" if p in clips else p, "value": p} for p in all_plants() if p in have]


def imaged_plants(df):
    """(bed, plant) of every imaged plant, bed by bed in grid order."""
    have = set(zip(df.bed, df.plant))
    return [(b, p) for b in BEDS for p in all_plants() if (b, p) in have]


def neighbours(df, bed, plant):
    """(previous, next) imaged plant, wrapping round."""
    order = imaged_plants(df)
    if (bed, plant) not in order:
        return None, None
    i = order.index((bed, plant))
    return order[i - 1], order[(i + 1) % len(order)]


def plant_row(bed, plant):
    plants = read_table("truth", "plants")
    if plants is None:
        return None
    r = plants[(plants.bed == bed) & (plants.plant == plant)]
    return r.iloc[0] if len(r) else None


def _href(path, bed, plant, day=None):
    """A page link carrying the selection (``?bed=&plant=&day=``); the day only when one is chosen."""
    return page_href(path, bed, plant, day)


def profile_href(bed, plant, day=None):
    return _href("/plant-profile", bed, plant, day)


def threed_href(bed, plant, day=None):
    """The 3D canopy page with this plant (and day) selected."""
    return _href("/threed", bed, plant, day)


def bedmap_href(bed, plant, day=None):
    return _href("/bed-map", bed, plant, day)


def day_arg(day):
    """A ``?day=`` value as an int in the app's day range (−10 to 19), else None."""
    try:
        d = int(day)
    except (TypeError, ValueError):
        return None
    return d if X_RANGE[0] < d < X_RANGE[1] else None


def has_depth(df, bed, plant):
    """True when a "3D canopy" link fits this plant: it has a value in a kept 3D trait and the 3D canopy page shows
    its bed (``threed_canopy.SHOWN_BEDS``: W1, W2, and W3 plant by plant), so the link lands on this plant."""
    cols = [c for _, c, _ in THREED if c in df]
    return bed in THREED_LINK_BEDS and bool(cols) and bool(plant_rows(df, bed, plant)[cols].notna().any().any())


TROUGH, REWATERED = tc.TROUGH_DAYS, tc.REWATERED_DAYS  # days 10–12 and 15–16, the 3D canopy page's windows


def window_mean(df, bed, plant, column, days):
    """(mean change, usable days) of the plant over a window of days, as the 3D canopy page's ``windows`` makes its
    trough and re-watered numbers (mean of the daily change over the days with one); (None, 0) without a value."""
    s = series(df, column)
    if s is None:
        return None, 0
    m = s.loc[(s.bed == bed) & (s.plant == plant) & s.day.between(*days), "chg"].dropna()
    return (float(m.mean()) if len(m) else None), len(m)


def stress_summary(df, bed, plant):
    """Stress probability on the daily midday median: the peak while unwatered (value, day) and the latest (value, day)."""
    if STRESS not in df:
        return None
    s = series(df, STRESS)
    m = s[(s.bed == bed) & (s.plant == plant)].dropna(subset=["chg"]).sort_values("day")
    if m.empty:
        return None
    dry = m[m.day.between(0, END_DAY)]
    peak = dry.loc[dry.chg.idxmax()] if len(dry) else None
    last = m.iloc[-1]
    return {
        "peak": (float(peak.chg), int(peak.day)) if peak is not None else (None, None),
        "now": (float(last.chg), int(last.day)),
    }


def span_text(lo, hi):
    """'Day 10–12 · 13–15 Sep': a window of days since irrigation off, then its dates."""
    a, b = CUTOFF + timedelta(days=int(lo)), CUTOFF + timedelta(days=int(hi))
    dates = f"{a.day}–{b.day} {b:%b}" if a.month == b.month else f"{a.day} {a:%b} – {b.day} {b:%b}"
    return f"Day {int(lo)}–{int(hi)} · {dates}".replace("-", "−")


def structure_summary(bed, plant):
    """3D leaf area of one plant against its own baseline, for a plain readout on Plant profile.

    From ``threed_canopy.structure_status`` (plant-day mean of the QC-pass 9–18 h runs, % of the plant's
    median on days −4 to −1): ``end`` and ``after``, the mean change over days 10–12 (end of the no-water
    window) and days 15–16 (after re-watering), each {"value" (% or None), "n" (usable days), "days"};
    ``first_below``, the first day of two usable days in a row below ``threshold`` % while unwatered
    (``structure_alerts``), or None; ``alarms``, (control plants that rule also flags, control plants). The
    rule flags control plants too, so the profile shows the values, not an alert. None when the plant is
    not in the 3D table (or there is no table).
    """
    st, alerts = tc.structure_status(), tc.structure_alerts()
    if st is None or alerts is None:
        return None
    mine = st[(st.bed == bed) & (st.plant == plant)]
    if mine.empty:
        return None
    a = alerts[(alerts.bed == bed) & (alerts.plant == plant)].iloc[0]
    control = alerts[alerts.treatment == "control"]

    def window(days):
        v = mine.loc[mine.day.between(*days), "leaf_area_pct"].dropna()
        return {"value": float(v.mean()) if len(v) else None, "n": len(v), "days": tuple(days)}

    return {
        "has_baseline": bool(a.has_baseline),
        "end": window(tc.TROUGH_DAYS),
        "after": window(tc.REWATERED_DAYS),
        "first_below": None if pd.isna(a.first_irrigate) else int(a.first_irrigate),
        "alarms": (int(control.first_irrigate.notna().sum()), len(control)),
        "threshold": tc.STRUCTURE_THRESHOLD,
    }


def rwc_state(bed, plant):
    """(value text, hover) for relative water content: these leaves were sampled on W3 plants that were not imaged."""
    rwc = read_table("truth", "rwc_leaves")
    if rwc is None:
        return "–", "RWC: no table in derived/truth/ yet"
    mine = rwc[(rwc.bed == bed) & (rwc.plant == plant)].dropna(subset=["truth_rwc"])
    if mine.empty:
        n = rwc.plant.nunique()
        return (
            "–",
            f"RWC leaves were sampled on other plants ({len(rwc)} leaves from {n} W3 plants that were not imaged)",
        )
    r = mine.sort_values("date").iloc[-1]
    return (
        f"{r.truth_rwc:.0f} %",
        f"Relative water content of the sampled leaf, {day_text(day_of(pd.Series([r.date])).iloc[0])}",
    )


def plant_csv_frame(df, bed, plant):
    """This plant's run rows, with every column that has a value for it."""
    rows = plant_rows(df, bed, plant)
    return rows.loc[:, rows.notna().any()]


# ------------------------------------------------------------------ figures
def is_prob(column):
    return column.endswith("_prob")


def plot_series(df, column):
    """`series` as the charts draw it: probabilities in %, as the readouts show them."""
    s = series(df, column)
    return s.assign(chg=s.chg * 100) if s is not None and is_prob(column) else s


def _base_layout(fig, column, height, aria=None):
    unit = "%" if is_prob(column) else change_unit(column)
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=12, t=8, b=32),
        showlegend=False,
        xaxis=dict(range=list(X_RANGE), title=None, dtick=5),
        yaxis=dict(
            title=unit or None,
            zeroline=mode_of(column) != "raw",
            zerolinecolor=C["gray-500"],
            zerolinewidth=1,
        ),
        hovermode="closest",
    )
    return mark_stress_period(fig)


def _band(fig, s, bed_set, color_line, color_band, name):
    g = s[s.bed.isin(bed_set)].groupby("day").chg
    m, sd = g.mean(), g.std().fillna(0)
    if m.empty:
        return
    fig.add_scatter(
        x=list(m.index) + list(m.index[::-1]),
        y=list(m + sd) + list((m - sd)[::-1]),
        fill="toself",
        fillcolor=color_band,
        line=dict(width=0),
        hoverinfo="skip",
        name=f"{name} ±1 SD",
    )
    fig.add_scatter(
        x=m.index,
        y=m.values,
        mode="lines",
        line=dict(color=color_line, width=1.5, dash="dot"),
        name=name,
        hovertemplate=f"{name} mean · day %{{x}} · %{{y:.2f}}<extra></extra>",
    )


def _plant_line(fig, mine, bed, plant, column):
    tag = source_of(column)[0]
    shown = [f"{v:.0f} %" if is_prob(column) else " ".join(value_text(column, v)) for v in mine.chg]
    custom = np.c_[[day_text(d) for d in mine.day], shown]
    fig.add_scatter(
        x=mine.day,
        y=mine.chg,
        mode="lines+markers",
        line=dict(color=C["navy"], width=2.5),
        marker=dict(size=6, color=C["navy"]),
        customdata=custom,
        name=f"{bed} {plant}",
        hovertemplate=f"{bed} {plant} · %{{customdata[0]}} · %{{customdata[1]}} · {tag}<extra></extra>",
    )


def small_multiple(df, bed, plant, column, height=150, day=None):
    """The plant's change from baseline (navy) over the control bed's mean ± 1 SD (teal), or None; `day` marks the
    selected day."""
    s = plot_series(df, column)
    if s is None:
        return None
    mine = s[(s.bed == bed) & (s.plant == plant)].dropna(subset=["chg"]).sort_values("day")
    if mine.empty:
        return None
    fig = go.Figure()
    _band(fig, s[~((s.bed == bed) & (s.plant == plant))], ["W1"], C["teal-700"], BAND["control"], "control")
    _plant_line(fig, mine, bed, plant, column)
    fig = _base_layout(fig, column, height)
    fig.update_layout(yaxis_title=None, margin_l=36)  # the unit sits beside the card's number
    return mark_day(fig, day)


def mark_day(fig, day):
    """The selected day (navy, DESIGN 2.2) as a dashed hairline, when a link carried one in."""
    if day is not None:
        fig.add_vline(x=day, line_width=1.5, line_dash="dash", line_color=C["navy"])
    return fig


def detail_figure(df, bed, plant, column, height=340, events=()):
    """One trait in full: control bed band, the plant's own deficit bed band (deficit plants), and the plant.
    `events`: (day, label) onsets of this plant (leaf clip, first symptom), drawn as navy dashed lines (DESIGN 6)."""
    s = plot_series(df, column)
    if s is None:
        return None
    mine = s[(s.bed == bed) & (s.plant == plant)].dropna(subset=["chg"]).sort_values("day")
    if mine.empty:
        return None
    others = s[~((s.bed == bed) & (s.plant == plant))]
    fig = go.Figure()
    _band(fig, others, ["W1"], C["teal-700"], BAND["control"], "control")
    if bed != "W1":
        _band(fig, others, [bed], C["amber-700"], BAND["deficit"], f"{bed} deficit")
    _plant_line(fig, mine, bed, plant, column)
    fig = _base_layout(fig, column, height)
    fig.update_layout(showlegend=True, margin=dict(l=48, r=12, t=32, b=32))
    for t in fig.data:
        t.showlegend = not t.name.endswith("SD")
    for i, (day, label) in enumerate(events):  # a second label sits one line lower, so close days stay apart
        fig.add_vline(x=day, line_width=1.5, line_dash="dash", line_color=C["navy"])
        fig.add_annotation(
            x=day,
            y=1,
            yref="paper",
            xanchor="left",
            yanchor="top",
            xshift=4,
            yshift=-16 * (i % 2),
            text=label,
            showarrow=False,
            font=dict(size=12, color=C["navy"]),
        )
    return fig


CHANGE_TEXT = {
    "pct": "% change from the plant's own baseline (median of 30 Aug – 2 Sep)",
    "diff": "change from the plant's own baseline (median of 30 Aug – 2 Sep)",
    "raw": "value as the module computes it (no baseline)",
}
THREED_DAY_TEXT = (
    "% change from the plant's own baseline (median of days −4 to −1); daily value = mean of its QC-pass runs at "
    "9, 12, 15 and 18 h, as on 3D canopy"
)


def method_hover(column):
    """How Plant profile's daily value and its change are made, for the hover of a chart."""
    module = source_of(column)[1]
    if column in THREED_KEY:
        how = THREED_DAY_TEXT
    else:
        how = f"{CHANGE_TEXT[mode_of(column)]}; daily median of the 12 h and 15 h runs that pass QC"
    name = tech(column)
    also = f" ({name})" if name != label_text(column) else ""
    return f"{label_unit(column)}{also}: {how}. Band: control bed ± 1 SD. Source: {module}/"


def run_hover(column, hour):
    """How the Bed map's squares are made: one run, not a daily value, against the same slot's baseline."""
    module = source_of(column)[1]
    how = CHANGE_TEXT[mode_of(column)]
    if mode_of(column) != "raw":
        how = how.replace(
            "(median of 30 Aug – 2 Sep)", f"at {hour:02d} h (median of its {hour:02d} h runs 30 Aug – 2 Sep)"
        )
    out = f"{label_unit(column)}: {how}. Each square is one {hour:02d} h run; a run that failed QC is no scan."
    if column in THREED_KEY:
        out += " Plant profile and 3D canopy show the daily mean of the QC-pass 9–18 h runs instead."
    return out + f" Source: {module}/"


# ------------------------------------------------------------------ marks (HTML)
EMPTY_TEXT = {"noscan": "no scan", "notable": "no table", "notimaged": "not imaged"}


def empty_mark(kind):
    """The three kinds of nothing (DESIGN 3): dashed circle, dashed square, dot."""
    base = {"display": "inline-block", "boxSizing": "border-box", "flex": "none"}
    style = {
        "noscan": {"width": "18px", "height": "18px", "borderRadius": "50%", "border": f"2px dashed {C['gray-500']}"},
        "notable": {"width": "18px", "height": "18px", "borderRadius": "2px", "border": f"2px dashed {C['gray-500']}"},
        "notimaged": {
            "width": "4px",
            "height": "4px",
            "borderRadius": "50%",
            "background": C["rule"],
            "margin": "0 7px",
        },
    }[kind]
    return html.Span(style=base | style, **{"aria-hidden": "true"})


def empty_panel(tag, label, kind, hover):
    """A sensor without data for this plant: dashed box, its mark and the kind of nothing."""
    return html.Section(
        [
            html.Span(tag, className="uw-kicker", style={"color": C["slate-600"]}),
            html.B(label),
            empty_mark(kind),
            html.Span(EMPTY_TEXT[kind]),
        ],
        className="uw-empty",
        title=hover,
        style={"alignItems": "center", "minHeight": "64px"},
        **{"aria-label": f"{label}: {EMPTY_TEXT[kind]}"},
    )


def legend_ramp(lo_text, hi_text, flip=False):
    """Value-ramp legend (the one allowed gradient): teal = healthier, amber = more stressed."""
    ramp = ", ".join(_ramp_colors()[::-1] if flip else _ramp_colors())
    return html.Span(
        [
            html.Span(lo_text),
            html.Span(
                style={
                    "width": "120px",
                    "height": "12px",
                    "borderRadius": "4px",
                    "background": f"linear-gradient(90deg, {ramp})",
                },
                **{"aria-hidden": "true"},
            ),
            html.Span(hi_text),
        ],
        style={"display": "inline-flex", "alignItems": "center", "gap": "6px"},
    )


def _ramp_colors():
    return [C[k] for k in ("teal-600", "teal-300", "gray", "amber-200", "amber-700", "amber-900")]


def value_text(column, value):
    """(number, unit) for a readout: probabilities as %, size traits as % of baseline, the rest signed in their
    unit; ('–', '') when missing."""
    if value is None or pd.isna(value):
        return "–", ""
    if column.endswith("_prob"):
        return f"{value * 100:.0f}", "%"
    unit = change_unit(column)
    digits = 0 if unit == "%" else 1 if abs(value) >= 1 else 2
    return fmt(value, "", digits, mode_of(column) != "raw"), unit


def bed_effect_chip(column):
    """The `bed effect` chip of a trait. The kept 3D traits use the 3D canopy page's own check (W2 against W1,
    placebo or E2), so both pages flag the same traits; the rest use the app-wide placebo test (W2 + W3)."""
    if column not in THREED_KEY:
        return placebo_chip(column)
    b = tc.bed_effects().get(THREED_KEY[column])
    if not b or not b["chip"]:
        return []
    why = []
    if b["placebo"]:
        why.append(f"score {b['placebo_auc']:.2f} on watered days 24–29 Aug")
    if b["e2"]:
        why.append("W2 grew apart from W1 before day 0")
    return [_bed_effect_chip(f"{label_text(column)}: {'; '.join(why)}, as on 3D canopy.")]


def graph_box(fig, label, height):
    return html.Div(
        dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": f"{height}px"}),
        role="img",
        **{"aria-label": label},
    )


# ------------------------------------------------------------------ bed map
def position(plant):
    """Grid position of a pot: column digit across (1-8), row letter down the bed (A at the top)."""
    return int(plant[1:]), ROWS.index(plant[0])


_RUN_CHANGE = {}


def run_change(df, column, hour):
    """Run-level change from each plant's own baseline at the same slot (median of the baseline days), QC-pass runs
    only: a run that failed its module's QC is no value here (and no part of a baseline)."""
    key = (id(df), column, hour)
    if key not in _RUN_CHANGE:
        d = df[(df.hour == hour) & usable(df, column)][["bed", "plant", "date", column]]
        d = d.assign(day=day_of(d.date))
        _RUN_CHANGE[key] = with_change(d, column)
    return _RUN_CHANGE[key]


def ramp_scale(column):
    """Plotly colourscale on the value ramp, amber = more stressed (follows the catalogue direction)."""
    cols = _ramp_colors()
    if direction_of(column) < 0 and mode_of(column) != "raw":
        cols = cols[::-1]
    n = len(cols) - 1
    return [[i / n, c] for i, c in enumerate(cols)]


def ramp_limits(df, column, hour):
    v = run_change(df, column, hour).chg.dropna()
    if mode_of(column) == "raw":
        return (float(v.quantile(0.02)), float(v.quantile(0.98))) if len(v) else (0.0, 1.0)
    m = float(v.abs().quantile(0.96)) if len(v) else 1.0
    return -(m or 1.0), m or 1.0


def pot_states(df, column, day, hour, bed):
    """One row per pot of a bed: state (value / noscan / notable / notimaged), change and raw value. A run with a
    value but no change (the plant has no baseline at this slot) is "no scan" too: there is nothing to colour."""
    pots = pd.DataFrame({"plant": all_plants()})
    pots[["x", "y"]] = [position(p) for p in pots.plant]
    imaged = set(df[df.bed == bed].plant)
    if column in df:
        r = run_change(df, column, hour)
        r = r[(r.bed == bed) & (r.day == day)][["plant", column, "chg"]]
        pots = pots.merge(r, on="plant", how="left")
        seen = set(df[(df.bed == bed) & df[column].notna()].plant)
    else:
        pots[column], pots["chg"], seen = np.nan, np.nan, set()
    pots["state"] = np.select(
        [pots.chg.notna(), pots.plant.isin(seen), pots.plant.isin(imaged)],
        ["value", "noscan", "notable"],
        "notimaged",
    )
    return pots


NO_BASE = "no baseline at this slot (no QC-pass run on 30 Aug – 2 Sep)"


def bed_figure(df, column, day, hour, bed, selected=None, imaged_only=False, height=560):
    """One bed as a grid of pots: value squares on the ramp, dashed marks for the kinds of nothing."""
    pots = pot_states(df, column, day, hour, bed)
    lo, hi = ramp_limits(df, column, hour)
    unit, raw_unit = change_unit(column), unit_text(column)
    tag = source_of(column)[0]
    fig = go.Figure()
    styles = {
        "notimaged": dict(symbol="circle", size=4, color=C["rule"]),
        "notable": dict(symbol="square-open", size=13, color=C["gray-500"], line=dict(width=2, color=C["gray-500"])),
        "noscan": dict(symbol="circle-open-dot", size=13, color=C["gray-500"], line=dict(width=2, color=C["gray-500"])),
    }
    why = {"notimaged": "not imaged", "notable": "no table for this sensor", "noscan": "no usable scan this run"}
    for state, style in styles.items():
        p = pots[pots.state == state]
        if imaged_only and state == "notimaged":
            continue
        fig.add_scatter(
            x=p.x,
            y=p.y,
            mode="markers",
            marker=style,
            customdata=np.c_[p.plant, [bed] * len(p), [NO_BASE if pd.notna(x) else why[state] for x in p[column]]],
            hovertemplate=f"{bed} %{{customdata[0]}} · %{{customdata[2]}}<extra></extra>",
            name=EMPTY_TEXT[state],
        )
    v = pots[pots.state == "value"]
    sel = [selected == (bed, p) for p in v.plant]
    signed = mode_of(column) != "raw"
    fig.add_scatter(
        x=v.x,
        y=v.y,
        mode="markers",
        marker=dict(
            symbol="square",
            size=22 if imaged_only else 20,
            color=v.chg,
            colorscale=ramp_scale(column),
            cmin=lo,
            cmax=hi,
            line=dict(width=[3 if s else 1 for s in sel], color=[C["navy"] if s else C["gray-500"] for s in sel]),
        ),
        customdata=np.c_[
            v.plant,
            [bed] * len(v),
            [fmt(c, unit, 2, signed) for c in v.chg],
            [fmt(x, raw_unit, 2, False) for x in v[column]],
        ],
        hovertemplate=(
            f"{bed} %{{customdata[0]}} · {day_text(day)} · {hour:02d} h<br>"
            f"%{{customdata[2]}} vs baseline · %{{customdata[3]}} · {tag}<extra></extra>"
        ),
        name="value",
    )
    others = pots[pots.state != "value"]
    hit = others[[selected == (bed, p) for p in others.plant]]
    if len(hit):
        fig.add_scatter(
            x=hit.x,
            y=hit.y,
            mode="markers",
            marker=dict(symbol="square-open", size=26, color=C["navy"], line=dict(width=3, color=C["navy"])),
            hoverinfo="skip",
            name="selected",
        )
    fig.update_layout(
        height=height,
        showlegend=False,
        margin=dict(l=24, r=4, t=20, b=4),
        dragmode=False,
        xaxis=dict(
            range=[0.4, 8.6],
            tickmode="array",
            tickvals=list(range(1, 9)),
            side="top",
            showgrid=False,
            showline=False,
            fixedrange=True,
            tickfont=dict(size=12, color=C["ink-muted"]),
        ),
        yaxis=dict(
            range=[17.6, -0.6],
            tickmode="array",
            tickvals=list(range(18)),
            ticktext=list(ROWS),
            showgrid=False,
            showline=False,
            fixedrange=True,
            tickfont=dict(size=12, color=C["ink-muted"]),
        ),
    )
    return fig


def bed_counts(df, column, day, hour, bed):
    s = pot_states(df, column, day, hour, bed).state.value_counts()
    return {k: int(s.get(k, 0)) for k in ("value", "noscan", "notable", "notimaged")}


# ------------------------------------------------------------------ export
THREED_DAYS = "threed/plant_days"  # the 3D canopy page's plant-day table


def export_tables():
    """{key: (label, path or None)}: the merged run table first, the 3D canopy plant-day table when the 3D
    trait table exists, then every result table in detect/ and diurnal/."""
    out = {"traits": ("Run traits (merged)", None)}
    if (DERIVED / "threed" / "traits_runs.parquet").exists():
        out[THREED_DAYS] = ("3D canopy · plant-days", None)
    for module in ("detect", "diurnal"):
        for p in sorted((DERIVED / module).glob("*.parquet")):
            out[f"{module}/{p.stem}"] = (f"{module} · {p.stem.replace('_', ' ')}", p)
    return out


def read_export(key):
    """The chosen table as a frame (keys normalised, treatment and day added where the keys allow), or None."""
    tables = export_tables()
    if key not in tables:
        return None
    if key == "traits":
        df = traits()
        return None if df is None else df.copy()
    if key == THREED_DAYS:
        return threed_days()
    path = tables[key][1]
    t = retry(lambda: normalize_keys(pd.read_parquet(path)))
    if "bed" in t and "treatment" not in t:
        t["treatment"] = np.where(t.bed == "W1", "control", "deficit")
    if "date" in t and "days_since_cutoff" not in t and "day" not in t:
        t["days_since_cutoff"] = day_of(t.date)
    return t


def threed_days():
    """The 3D canopy page's plant-day table under contract column names, or None without the 3D table.

    One row per W1 or W2 plant and day: the three kept 3D traits (leaf area, height p75, hull volume) as the
    mean of the QC-pass 9–18 h runs, each with the plant's baseline (``<column>_base``) and the % change from
    it (``<column>_pct``), and the runs behind the day (n_runs, n_pass, slots). ``threed_canopy.plant_days``.
    """
    p = tc.plant_days()
    if p is None:
        return None
    names = {"day": "days_since_cutoff"}
    for k, t in tc.TRAITS.items():
        names |= {k: t.column, f"{k}_base": f"{t.column}_base", f"{k}_pct": f"{t.column}_pct"}
    return p.rename(columns=names)


def day_column(t):
    return next((c for c in ("days_since_cutoff", "day") if c in t), None)


def filter_export(t, beds=None, treatment="all", prefixes=None, days=None):
    """Rows of the chosen beds / treatment / day range, and only the sensor columns of the chosen prefixes."""
    if t is None:
        return None
    if beds is not None and "bed" in t:
        t = t[t.bed.isin(beds)]
    if treatment in ("control", "deficit") and "treatment" in t:
        t = t[t.treatment == treatment]
    dc = day_column(t)
    if days and dc:
        t = t[pd.to_numeric(t[dc], errors="coerce").between(*days)]
    if prefixes is not None:
        known = tuple(SOURCE)
        keep = [c for c in t.columns if not c.startswith(known) or c.startswith(tuple(prefixes))]
        t = t[keep]
    return t


def export_counts(t):
    rows = len(t)
    plants = t[["bed", "plant"]].drop_duplicates().shape[0] if {"bed", "plant"} <= set(t) else None
    dc = day_column(t)
    days = t[dc].nunique() if dc else None
    return rows, plants, days


def data_dictionary(columns, table_key="traits"):
    """column, label, unit, direction, source, description for every exported column."""
    cat = catalogue().drop_duplicates("column").set_index("column")
    design = {
        "bed": ("Bed (W1 control, W2/W3 deficit)", "", "key"),
        "plant": ("Pot position (row letter, column digit)", "", "key"),
        "date": ("Run date", "date", "key"),
        "hour": ("Scheduled slot", "h", "key"),
        "treatment": ("control or deficit, from the bed", "", "design"),
        "days_since_cutoff": ("Days since irrigation off (day 0 = 3 Sep 2025)", "d", "design"),
        "phase": ("before cutoff / stress / re-watered", "", "design"),
    }
    plant_day = plant_day_terms(cat) if table_key == THREED_DAYS else {}
    rows = []
    for c in columns:
        if c in plant_day:
            rows.append(dict(column=c, **plant_day[c]))
        elif c in cat.index:
            r = cat.loc[c]
            name = label_text(c)  # the plain name the pages use; the catalogue's own name leads the description
            what = r.description if isinstance(r.description, str) else ""
            if name != r.label:
                what = f"{r.label}: {what}" if what else r.label
            rows.append(
                dict(
                    column=c,
                    label=name,
                    unit=unit_text(c),
                    direction=int(r.direction) if pd.notna(r.direction) else "",
                    source=f"derived/{r.module}/",
                    description=what,
                )
            )
        elif c in design:
            label, unit, kind = design[c]
            rows.append(dict(column=c, label=label, unit=unit, direction="", source=kind, description=""))
        else:
            module = source_of(c)[1] or (table_key.split("/")[0] if "/" in table_key else "")
            rows.append(
                dict(
                    column=c,
                    label="",
                    unit="",
                    direction="",
                    source=f"derived/{module}/" if module else "",
                    description="not in the catalogue (QC flag, count or helper column)",
                )
            )
    return pd.DataFrame(rows, columns=["column", "label", "unit", "direction", "source", "description"])


def plant_day_terms(cat):
    """Dictionary rows of the 3D plant-day columns (``threed_days``): each trait, its baseline and change,
    and the run counts; the trait's label, unit, direction and definition come from the catalogue."""
    src = "derived/threed/"
    day = "mean of the plant's QC-pass runs at 9, 12, 15 and 18 h that day"
    out = {
        "date": dict(label="Date of the plant-day", unit="date", direction="", source="key", description=""),
        "n_runs": dict(
            label="Runs that day",
            unit="count",
            direction="",
            source=src,
            description="runs of the plant in the 3D table at 9, 12, 15 and 18 h that day",
        ),
        "n_pass": dict(
            label="QC-pass runs",
            unit="count",
            direction="",
            source=src,
            description="runs of n_runs that passed QC (depth_qc_pass); 0 = no usable scan that day",
        ),
        "slots": dict(
            label="Slots averaged",
            unit="h",
            direction="",
            source=src,
            description="slots of the QC-pass runs averaged into the daily values",
        ),
    }
    for t in tc.TRAITS.values():
        r = cat.loc[t.column] if t.column in cat.index else None
        label = label_text(t.column)
        unit = unit_text(t.column) or t.unit
        direction = int(r.direction) if r is not None and pd.notna(r.direction) else ""
        what = r.description if r is not None and isinstance(r.description, str) else ""
        out[t.column] = dict(
            label=label, unit=unit, direction=direction, source=src, description=f"{what}; {day}" if what else day
        )
        out[f"{t.column}_base"] = dict(
            label=f"{label}, baseline",
            unit=unit,
            direction="",
            source=src,
            description="the plant's baseline: median of its daily values on days −4 to −1 (30 Aug – 2 Sep); "
            "empty when it has no QC-pass daytime run then",
        )
        out[f"{t.column}_pct"] = dict(
            label=f"{label}, change",
            unit="%",
            direction=direction,
            source=src,
            description="% change from the plant's own baseline: (daily value / baseline − 1) × 100",
        )
    return out


def preview_table(t, n=12):
    """First rows as an HTML table: navy header, numbers right-aligned and tabular, '–' for missing."""
    head = t.head(n)
    num = [pd.api.types.is_numeric_dtype(head[c]) for c in head.columns]
    # a date stays on one line (it broke at its hyphens in the narrow first columns)
    date = [pd.api.types.is_datetime64_any_dtype(head[c]) for c in head.columns]

    def cell(v, is_num):
        if v is None or (not isinstance(v, (list, np.ndarray)) and pd.isna(v)):
            return "–"
        if isinstance(v, pd.Timestamp):
            return f"{v:%Y-%m-%d}"
        if is_num and isinstance(v, float):
            return f"{v:.3g}".replace("-", "−")
        return str(v)

    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th(c, scope="col", className="num" if k else None) for c, k in zip(head.columns, num)]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    cell(v, k),
                                    className="num" if k else None,
                                    style={"whiteSpace": "nowrap"} if d else None,
                                )
                                for v, k, d in zip(row, num, date)
                            ]
                        )
                        for row in head.itertuples(index=False)
                    ]
                ),
            ],
            className="uw-table",
        ),
        className="uw-scroll",
        style={"border": f"1px solid {C['rule']}", "borderRadius": "12px"},
    )
