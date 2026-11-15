"""Daily rhythm, plant stress scores and irrigation alerts.

    python -m uiwue.diurnal

Reads the merged table (thermal + environment) and writes to derived/diurnal/:
- scores_runs: three stress scores and an evidence flag per plant and run
- separability: how well each score separates deficit from control plants, by day and time of day
- schedules: alert day and false alerts for each rule and flyover schedule
- sky_effect: how much stronger the stress signal is on sunny than on cloudy flyovers (DWD sunshine)

Rule: a plant is flagged "irrigate" when its score is above the threshold on
two usable scans in a row. Thresholds and the recommended schedule are picked
on W2 with the development controls, then checked unchanged on W3.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uiwue.core import (
    BASELINE,
    CUTOFF,
    DEV_CONTROLS,
    KEYS,
    PLACEBO,
    SLOTS,
    TRUTH_CLIP_ONSET,
    in_window,
    load_traits,
    read_table,
    write_table,
)

STRESS_DAYS = (0, 12)
PLACEBO_DAYS = tuple((pd.Timestamp(d) - pd.Timestamp(CUTOFF)).days for d in PLACEBO)  # (-10, -5)
DULL = 0.5  # a daytime scan is "weak evidence" when light is below half that slot's usual level
# Published non-water-stressed baseline for sugar beet, leaf − air (°C) = A + B·VPD (kPa): Idso (1982), Agric.
# Meteorol. 27:59-70, Table I (sunlit canopy, Mesa, Arizona; N = 47, r = 0.898). Shaded canopies sit ~3.8 °C lower.
IDSO_A, IDSO_B = 2.50, -1.92

RULES = {  # name -> (column, label, what the grower needs to run it)
    "control_ref": ("diurnal_score_ref", "Thermal score vs control bed", "a well-watered reference bed in view"),
    "local_line": (
        "diurnal_score_local",
        "Thermal score vs healthy-plant line",
        "a thermal camera plus air temperature and humidity",
    ),
    "idso": (
        "diurnal_score_idso",
        "Leaf minus air above the published sugar beet line",
        "a thermal camera plus air temperature and humidity",
    ),
    "leaf_air": ("thermal_leaf_air_dt", "Leaf minus air (raw)", "a thermal camera plus air temperature"),
    "microclimate": (
        "env_vpd_since_water",
        "Demand since last watering",
        "watering records and an air sensor; no plant sensing",
    ),
}
SCHEDULES = {
    "06 h": (6,),
    "09 h": (9,),
    "12 h": (12,),
    "15 h": (15,),
    "18 h": (18,),
    "12 + 15 h": (12, 15),
    "all five": SLOTS,
}
PHASE_BINS = ([-11, -1, 3, 7, 12, 19], ["before cutoff", "days 0-3", "days 4-7", "days 8-12", "re-watered"])

CATALOGUE = [
    (
        "diurnal_score_ref",
        "Thermal stress score (vs control bed)",
        "z",
        1,
        "leaf minus control-bed leaf temperature, as a z-score against the plant's own 30 Aug–2 Sep values at the same time of day",
    ),
    (
        "diurnal_score_local",
        "Thermal stress score (no control bed)",
        "z",
        1,
        "leaf minus air temperature above a line in VPD fitted on healthy plants (24 Aug–2 Sep), as a z-score against the plant's own baseline",
    ),
    (
        "diurnal_score_idso",
        "Leaf − air above the published sugar beet line",
        "°C",
        1,
        "leaf minus air temperature minus Idso's (1982) non-water-stressed sugar beet line, 2.50 − 1.92 × VPD (kPa)",
    ),
]


def own_z(df, x):
    """z-score of column x against each plant's own baseline mean at the same slot.

    The spread is the within-plant baseline variance pooled over plants, per slot.
    """
    base = df[in_window(df.date, BASELINE)]
    st = base.groupby(["bed", "plant", "hour"])[x].agg(mu="mean", var="var", n="count")
    st = st[st.n >= 2]
    sd = np.sqrt(st.groupby("hour")["var"].mean()).rename("sd")
    d = df[["bed", "plant", "hour", x]].join(st.mu, on=["bed", "plant", "hour"]).join(sd, on="hour")
    return (d[x] - d.mu) / d.sd


def healthy_line(df):
    """Per slot: (slope, intercept) of leaf − air on VPD, fitted on all plants before the cutoff."""
    h = df[(df.days_since_cutoff < 0)].dropna(subset=["thermal_leaf_air_dt", "env_vpd"])
    return {s: np.polyfit(g.env_vpd, g.thermal_leaf_air_dt, 1) for s, g in h.groupby("hour")}


def evidence(df):
    """'weak' when the thermal reading or air data is missing, or a daytime scan is unusually dull."""
    typical = df.groupby("hour").env_par_mv.transform("median")
    dull = df.hour.isin([9, 12, 15]) & (df.env_par_mv < DULL * typical)
    missing = df.thermal_leaf_t.isna() | df.env_vpd.isna() | ~df.thermal_qc_pass.fillna(False).astype(bool)
    return pd.Series(np.where(missing | dull, "weak", "ok"), index=df.index)


def scores(df):
    line = healthy_line(df)
    expected = [np.polyval(line[h], v) if h in line else np.nan for h, v in zip(df.hour, df.env_vpd)]
    out = df[KEYS].copy()
    out["diurnal_score_ref"] = own_z(df, "thermal_leaf_ref_dt")
    out["diurnal_score_local"] = own_z(df.assign(resid=df.thermal_leaf_air_dt - expected), "resid")
    out["diurnal_score_idso"] = df.thermal_leaf_air_dt - (IDSO_A + IDSO_B * df.env_vpd)
    out["diurnal_evidence"] = evidence(df)
    return out


def status(df, col, slots, k):
    """Per scan: 'watch' if col > k, 'irrigate' if also above k on the plant's previous usable scan, else 'fine'.

    Only scans in `slots` with ok evidence and a value are used; other rows get NaN.
    """
    use = df.hour.isin(slots) & df[col].notna() & (df.diurnal_evidence == "ok")
    d = df[use].sort_values(KEYS)
    above = d[col] > k
    before = above.groupby([d.bed, d.plant]).shift(1, fill_value=False)
    st = np.where(above & before, "irrigate", np.where(above, "watch", "fine"))
    return pd.Series(st, index=d.index).reindex(df.index)


def alert_days(df, st, window=STRESS_DAYS):
    """Per plant id: first day since cutoff within `window` with an 'irrigate' scan (NaN if none)."""
    d = df.assign(st=st, pid=df.bed + "_" + df.plant)
    d = d[d.days_since_cutoff.between(*window)]
    first = d[d.st == "irrigate"].groupby("pid").days_since_cutoff.min()
    return first.reindex(sorted(d.pid.unique()))


def tune(df, col, slots, bed, controls):
    """Lowest threshold with no false alert in the tuning set (one deficit bed plus `controls`):
    no control alerts during the stress window, and no plant alerts in the placebo window."""
    pid = df.bed + "_" + df.plant
    t = df[(df.bed == bed) | pid.isin(controls)]
    is_ctrl = (t.bed + "_" + t.plant).isin(controls)
    healthy = (is_ctrl & t.days_since_cutoff.between(*STRESS_DAYS)) | t.days_since_cutoff.between(*PLACEBO_DAYS)
    vals = t.loc[healthy & t.hour.isin(slots), col].dropna()
    for k in np.unique(np.quantile(vals, np.linspace(0.5, 1, 101))):
        st = status(t, col, slots, k)
        if alert_days(t[is_ctrl], st[is_ctrl]).isna().all() and alert_days(t, st, PLACEBO_DAYS).isna().all():
            return float(k)
    return float(vals.max())


def evaluate(df, col, slots, k, bed, controls, clip_onset=None):
    """Alert results for one deficit bed against a set of control plants."""
    pid = df.bed + "_" + df.plant
    d = df[(df.bed == bed) | pid.isin(controls)]
    st = status(d, col, slots, k)
    days = alert_days(d, st)
    deficit, ctrl = days[days.index.str.startswith(bed)], days[days.index.isin(controls)]
    placebo = alert_days(d, st, window=PLACEBO_DAYS)
    lead = (clip_onset.reindex(deficit.index) - deficit).dropna() if clip_onset is not None else pd.Series(dtype=float)
    return dict(
        n_plants=len(deficit),
        n_detected=int(deficit.notna().sum()),
        median_alert_day=deficit.median(),
        latest_alert_day=deficit.max(),
        n_controls=len(ctrl),
        false_alerts_controls=int(ctrl.notna().sum()),
        n_placebo_plants=len(placebo),
        false_alerts_placebo=int(placebo.notna().sum()),
        median_lead_vs_clip=lead.median() if len(lead) else np.nan,
        n_clip=len(lead),
    )


def clip_onset_days():
    """Day since cutoff each clip's turgor shift began, from Sam's truth/plants table, if it exists."""
    t = read_table("truth", "plants")
    if t is None or TRUTH_CLIP_ONSET not in t:
        return None
    t = t.dropna(subset=[TRUTH_CLIP_ONSET])
    days = (pd.to_datetime(t[TRUTH_CLIP_ONSET]) - pd.Timestamp(CUTOFF)).dt.days
    return pd.Series(days.values, index=t.bed + "_" + t.plant)


def schedules(df):
    """Every rule x schedule: threshold tuned on W2, results on W2 (tuning) and W3 (check)."""
    w1 = sorted((df.loc[df.bed == "W1", "bed"] + "_" + df.loc[df.bed == "W1", "plant"]).unique())
    check_controls = [p for p in w1 if p not in DEV_CONTROLS]
    onset, rows = clip_onset_days(), []
    for rule, (col, *_) in RULES.items():
        for name, slots in SCHEDULES.items():
            k = tune(df, col, slots, "W2", DEV_CONTROLS)
            for role, bed, controls in (("W2 tuning", "W2", DEV_CONTROLS), ("W3 check", "W3", check_controls)):
                rows.append(
                    dict(rule=rule, schedule=name, scans_per_day=len(slots), threshold=k, set=role)
                    | evaluate(df, col, slots, k, bed, controls, onset)
                )
    out = pd.DataFrame(rows)
    tuning = out[out.set == "W2 tuning"].assign(day=lambda t: t.median_alert_day.fillna(99))
    best = (
        tuning.sort_values(["n_detected", "day", "scans_per_day"], ascending=[False, True, True])
        .groupby("rule")
        .head(1)
    )
    picked = set(zip(best.rule, best.schedule))
    out["recommended"] = [(r, s) in picked for r, s in zip(out.rule, out.schedule)]
    return out


def separability(df):
    """AUC of each score, deficit against control plants, per day since cutoff and slot."""
    rows = []
    for col in ("diurnal_score_ref", "diurnal_score_local", "diurnal_score_idso"):
        for group, beds in (("W2 + W3", ("W2", "W3")), ("W2", ("W2",)), ("W3", ("W3",))):
            d = df[df.bed.isin(("W1",) + beds) & df[col].notna()]
            for (day, hour), g in d.groupby(["days_since_cutoff", "hour"]):
                y = g.bed != "W1"
                if y.sum() >= 3 and (~y).sum() >= 3:
                    rows.append(
                        dict(
                            score=col,
                            group=group,
                            day=day,
                            hour=hour,
                            auc=roc_auc_score(y, g[col]),
                            n_deficit=int(y.sum()),
                            n_control=int((~y).sum()),
                        )
                    )
    return pd.DataFrame(rows)


def sky_effect(df, days=(2, 12)):
    """Per slot: deficit-minus-control leaf temperature gap on sunny and cloudy flyovers (DWD sky label).

    One gap per run (stress days only); `sunny_extra` is the sunny-minus-cloudy difference with the day
    since cutoff held fixed (least squares on day and sunny), since stress grows day by day.
    """
    d = df[df.days_since_cutoff.between(*days) & df.env_sky.isin(["sunny", "cloudy"])]
    d = d.assign(deficit=d.bed != "W1")
    runs = d.groupby(["date", "hour", "days_since_cutoff", "env_sky", "deficit"]).thermal_leaf_ref_dt.mean()
    runs = (runs.xs(True, level="deficit") - runs.xs(False, level="deficit")).dropna().rename("gap").reset_index()
    rows = []
    for hour, g in runs.groupby("hour"):
        sunny = (g.env_sky == "sunny").astype(float)
        x = np.column_stack([np.ones(len(g)), g.days_since_cutoff, sunny])
        extra = np.linalg.lstsq(x, g.gap, rcond=None)[0][2] if 0 < sunny.sum() < len(g) and len(g) >= 4 else np.nan
        rows.append(
            dict(
                hour=hour,
                n_sunny=int(sunny.sum()),
                n_cloudy=int(len(g) - sunny.sum()),
                gap_sunny=g.gap[sunny == 1].mean(),
                gap_cloudy=g.gap[sunny == 0].mean(),
                sunny_extra=extra,
            )
        )
    return pd.DataFrame(rows)


def phase_bin(days):
    return pd.cut(days, PHASE_BINS[0], labels=PHASE_BINS[1]).astype(str)


def rhythm(df, col):
    """Per plant and day: afternoon minus morning (15 h − 09 h), daily range, and overnight change (next 06 h − 18 h)."""
    w = df.pivot_table(index=["bed", "plant", "date"], columns="hour", values=col).reindex(columns=list(SLOTS))
    out = pd.DataFrame(index=w.index)
    out["afternoon_minus_morning"] = w[15] - w[9]
    out["daily_range"] = (w.max(axis=1) - w.min(axis=1)).where(w.notna().sum(axis=1) >= 4)
    nxt = w.reset_index().groupby(["bed", "plant"])[[6, "date"]].shift(-1)
    consecutive = (nxt["date"] - w.reset_index()["date"]).dt.days.eq(1).values
    out["overnight_change"] = np.where(consecutive, nxt[6].values - w[18].values, np.nan)
    return out.reset_index()


def main():
    df = load_traits()
    if df is None or "thermal_leaf_t" not in df:
        raise SystemExit("needs derived/thermal/traits_runs.parquet; run `python -m uiwue.thermal` first")
    df = df.drop(columns=[c for c in df.columns if c.startswith("diurnal_")])
    df = df[df.thermal_leaf_t.notna()].reset_index(drop=True)
    sc = scores(df)
    write_table(sc, "diurnal", "scores_runs", CATALOGUE)
    df = df.merge(sc, on=KEYS)
    sep, sch, sky = separability(df), schedules(df), sky_effect(df)
    write_table(sep, "diurnal", "separability")
    write_table(sch, "diurnal", "schedules")
    write_table(sky, "diurnal", "sky_effect")

    pd.set_option("display.width", 200)
    print(f"{len(sc)} scored runs; evidence weak: {(sc.diurnal_evidence == 'weak').mean():.0%}")
    pre = sep[(sep.day < 0) & (sep.group == "W2 + W3")].pivot_table(index="score", columns="hour", values="auc")
    print("placebo check, mean AUC before the cutoff by time of day (want ~0.5):")
    print(pre.round(2).to_string())
    w = df[(df.hour == 15) & df.days_since_cutoff.between(0, 11)]
    gap = (
        w[w.bed == "W2"].groupby("days_since_cutoff").thermal_leaf_air_dt.mean()
        - w[w.bed == "W1"].groupby("days_since_cutoff").thermal_leaf_air_dt.mean()
    )
    print(
        f"paper check, W2 minus W1 leaf-air trend at 15 h: {np.polyfit(gap.index, gap.values, 1)[0]:+.2f} °C/day (paper +0.60)"
    )
    ctrl = df[(df.bed == "W1") & df.hour.isin([12, 15])]
    print(
        "published sugar beet line: watered plants sit "
        f"{ctrl.diurnal_score_idso.median():+.2f} °C from it at 12 and 15 h (0 = on the line)"
    )
    w1 = sorted(set(ctrl.bed + "_" + ctrl.plant))
    for name, slots in (("12 + 15 h", (12, 15)), ("15 h", (15,))):
        r = evaluate(df, "diurnal_score_idso", slots, 0.0, "W3", w1)
        print(
            f"  untuned 'above the line' rule, {name}: W3 {r['n_detected']}/{r['n_plants']} by day "
            f"{r['median_alert_day']:g}; false alerts {r['false_alerts_controls']}/{r['n_controls']} controls, "
            f"{r['false_alerts_placebo']}/{r['n_placebo_plants']} plants before the cutoff"
        )
    print("sunny against cloudy flyovers (DWD Belm), stress days 2-12, deficit minus control leaf gap (°C):")
    print(sky.round(2).to_string(index=False))
    cols = [
        "rule",
        "schedule",
        "set",
        "threshold",
        "n_detected",
        "n_plants",
        "median_alert_day",
        "false_alerts_controls",
        "n_controls",
        "false_alerts_placebo",
        "recommended",
    ]
    print(sch[cols].round(2).to_string(index=False))
