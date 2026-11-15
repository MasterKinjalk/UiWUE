"""Experiments E0a-E5 of plan/Kinjalk-Work-Plan.md, section 7, on the 3D trait table.

    python -m uiwue.threed.experiments --exp E0a E0b E1 E2 E3 E4 E5
    python -m uiwue.threed.experiments --exp E0a E1 E3 --runs data/derived/threed/traits_runs.parquet
    python -m uiwue.threed.experiments --synthetic          # demonstration on synthetic tables
    python -m uiwue.threed.experiments --runs derived/threed/traits_runs.parquet --out derived/threed/results
                                                            # from the committed team table (fresh clone)

Run from the repository root. Writes the CSVs of ``schema.RESULT_FILES`` (plus a few extra
tables), PNG figures and ``summary.md`` into ``schema.RESULTS_DIR``. ``--synthetic`` writes into
``RESULTS_DIR/synthetic`` instead, so synthetic numbers never sit where real results belong.
Every CSV has a ``source`` column ("real" or "synthetic").

All statistics use ``qc_pass`` runs only, except E0a (the paper-parity table has no QC flags;
the paper's own filter of at least 1000 vegetation points is applied when the column exists)
and E0b (which compares the flags themselves, so it uses all runs).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # a pipeline batch shares this machine

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from uiwue.threed import envdata, schema, stats
from uiwue.threed.contract import from_team_table

# ---------------------------------------------------------------- constants
TIER1 = ("A2", "A8", "J5", "R1", "R7")  # the paper's plants, in W1 and W2
PAPER = {  # ADA README / paper, W2 against W1, 2025-09-03 to 2025-09-14
    "control_slope_mm_per_day": 4.30,
    "interaction_slope_mm_per_day": -13.08,
}
PARITY_TOL = 0.20
PARITY_WINDOW = (date(2025, 9, 3), date(2025, 9, 14))  # run_plant_height_pipeline.py defaults
PARITY_MIN_VEG_POINTS = 1000
VEG_POINT_COLUMNS = ("num_points_vegetation", "n_points_veg", "n_veg_points", "n_veg")
SNR_MIN = 3.0
SIGNAL_DATE = date(2025, 9, 14)
E2_WINDOW = (date(2025, 8, 20), date(2025, 9, 2))
E3_WINDOW = schema.BASELINE_DATES[0], schema.TREATMENT_WINDOW[1]  # 2025-09-01 .. 2025-09-14
E3_PLOT_END = date(2025, 9, 18)  # last 15 h run with depth
E4_WINDOW = (date(2025, 8, 28), date(2025, 9, 18))
E5_WINDOW = (date(2025, 9, 2), date(2025, 9, 19))
ALPHA = 0.05

COLORS = {"control": "#2a78d6", "deficit": "#eb6834"}  # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#f0efec", "#d03b3b"]).with_extremes(
    bad="#a9a8a3"  # no data: mid grey, distinct from the near-white zero
)
CYL_TRAITS = (  # pipeline extras: all vegetation in the plant cylinder, no clustering
    "cyl_height_p50",
    "cyl_height_p75",
    "cyl_height_p95",
    "cyl_projected_area",
    "cyl_n_points",
)
DAYTIME_SLOTS = (9, 12, 15, 18)  # 06 h is dark (height-only) throughout the treatment window
HEATMAP_TRAITS = ("height_p75", "droop_index", "projected_area", "height_p95", "inclination_mean")


def _p(p: float) -> str:
    """'= 0.0123' or '< 1e-16' (a p-value of exactly 0 is a floating-point underflow)."""
    return "< 1e-16" if p < 1e-16 else f"= {p:.2g}"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- inputs
def prepare_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Contract checks plus the derived columns the experiments need."""
    if "depth_capture_hour" in df:  # the committed team table (derived/threed/), not the pipeline's own
        df = from_team_table(df)
    schema.validate_runs(df)
    df = df.copy()
    df["date"] = stats.ensure_dates(df["date"])
    df["bed"] = df["bed"].astype(str)
    df["plant"] = df["plant"].astype(str)
    df["treatment"] = df["treatment"].astype(str)
    df["plant_id"] = df["bed"] + "_" + df["plant"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    for c in list(schema.QC_FLAGS) + ["qc_pass"]:
        df[c] = df[c].fillna(False).astype(bool)
    return df


def passing(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[runs["qc_pass"]]


def _in(df: pd.DataFrame, window: tuple[date, date]) -> pd.DataFrame:
    return df[(df["date"] >= window[0]) & (df["date"] <= window[1])]


def load_frames(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    f = pd.read_parquet(path)
    missing = [c for c in schema.FRAME_COLUMNS if c not in f.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    f["date"] = stats.ensure_dates(f["date"])
    f["bed"] = f["bed"].astype(str)
    f["plant"] = f["plant"].astype(str)
    return f


# ---------------------------------------------------------------- E0a paper parity
def paper_lmm_input(parity: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """The model input of the ADA plant-height model (``step_04_fit_lmm.py``): W1/W2 Tier-1
    plants, folder hour 15, 2025-09-03 to 2025-09-14, at least 1000 vegetation points, height in
    metres, day counted from the first retained date."""
    p = parity.copy()
    p["date"] = stats.ensure_dates(p["date"])
    p["bed"] = p["bed"].astype(str)
    p["plant"] = p["plant"].astype(str)
    sel = (
        p["bed"].isin(["W1", "W2"])
        & p["plant"].isin(TIER1)
        & (p["hour"].astype(int) == 15)
        & (p["date"] >= PARITY_WINDOW[0])
        & (p["date"] <= PARITY_WINDOW[1])
        & p["height_parity_mm"].notna()
    )
    note = f"no vegetation-point column found ({', '.join(VEG_POINT_COLUMNS)}); filter not applied"
    for c in VEG_POINT_COLUMNS:
        if c in p.columns:
            sel &= p[c] >= PARITY_MIN_VEG_POINTS
            note = f"{c} >= {PARITY_MIN_VEG_POINTS}"
            break
    f = p[sel].copy()
    mi = pd.DataFrame(
        {
            "plant_id": f["bed"] + "_" + f["plant"],
            "date": pd.to_datetime(f["date"]),
            "treatment": pd.Categorical(f["bed"], categories=["W1", "W2"], ordered=True),
            "height_estimate": f["height_parity_mm"].astype(float) / 1000.0,
        }
    )
    mi["day"] = (mi["date"] - mi["date"].min()).dt.days.astype(float)
    return mi.reset_index(drop=True), note


def _paper_model_rows(mi: pd.DataFrame, label: str, scored: bool) -> list[dict]:
    """Fit ``height_estimate ~ day * treatment + (1 | plant_id)`` exactly as the ADA code and
    return the two slopes in mm/day against the paper."""
    res = None
    if mi["treatment"].nunique() == 2 and mi["plant_id"].nunique() >= 3:
        res = stats.fit_lmm(mi, "height_estimate ~ day * treatment", group="plant_id")
    rows = []
    for q, term in (
        ("control_slope_mm_per_day", "day"),
        ("interaction_slope_mm_per_day", "day:treatment[T.W2]"),
    ):
        t = stats.lmm_term(res, term)
        ours = 1000 * t["coef"]
        paper = PAPER[q]
        rel = (ours - paper) / abs(paper)
        rows.append(
            {
                "quantity": q if scored else f"{q} [{label}]",
                "ours": ours,
                "paper": paper,
                "rel_diff": rel,
                "pass": bool(abs(rel) <= PARITY_TOL) if scored and np.isfinite(rel) else pd.NA,
                "ci_low": 1000 * t["ci_low"],
                "ci_high": 1000 * t["ci_high"],
                "p_value": t["p_value"],
                "n_rows": len(mi),
                "n_plants": int(mi["plant_id"].nunique()),
                "height": label,
            }
        )
    return rows


def e0a(runs: pd.DataFrame | None, parity: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paper parity. Scored rows use the parity table (0.60 m - 25th percentile depth of ExG > 5
    points, frame 7). An informational row repeats the model on our soil-relative
    ``height_p75`` (qc_pass runs). Returns (table, model input of the scored rows)."""
    rows, mi = [], pd.DataFrame()
    if parity is not None:
        mi, note = paper_lmm_input(parity)
        _log(f"E0a: parity model input {len(mi)} rows, {mi['plant_id'].nunique()} plants; {note}")
        rows += _paper_model_rows(mi, "parity: 0.60 m - z_p25, frame 7", scored=True)
    if runs is not None:
        p = passing(runs)
        own = p[["bed", "plant", "date", "hour"]].assign(height_parity_mm=p["height_p75"])
        own = own.assign(num_points_vegetation=PARITY_MIN_VEG_POINTS)  # qc_pass replaces it
        mi_own, _ = paper_lmm_input(own)
        if len(mi_own):
            rows += _paper_model_rows(mi_own, "ours: height_p75 above soil, qc_pass", scored=False)
    return pd.DataFrame(rows), mi


# ---------------------------------------------------------------- E0b QC agreement
QC_RULES = {  # rule name -> (our flag, ADA label, function of the ADA run table)
    "dark: >= 8 of 15 RGB frames Dark": ("qc_dark", "Dark", lambda a: a["n_rgb_dark"] >= 8),
    "dark: any RGB frame Dark": ("qc_dark", "Dark", lambda a: a["n_rgb_dark"] >= 1),
    "dark: >= 8 RGB frames Dark or Medium Dark": (
        "qc_dark",
        "Dark|Medium Dark",
        lambda a: (a["n_rgb_dark"] + a["n_rgb_medium_dark"]) >= 8,
    ),
    "not centred: any depth/intensity/RGB/3D frame": (
        "qc_not_centred",
        "Not Centered",
        lambda a: a["n_not_centred"] >= 1,
    ),
    "not centred: any depth frame": (
        "qc_not_centred",
        "Not Centered",
        lambda a: a["n_depth_not_centred"] >= 1,
    ),
    "not centred: >= 3 frames": (
        "qc_not_centred",
        "Not Centered",
        lambda a: a["n_not_centred"] >= 3,
    ),
    "no plant: any frame": ("qc_no_plant", "no plant", lambda a: a["n_no_plant"] >= 1),
}
QC_MAIN_RULES = (
    "not centred: any depth/intensity/RGB/3D frame",
    "dark: >= 8 of 15 RGB frames Dark",
    "no plant: any frame",
)


def _agreement(pred: pd.Series, truth: pd.Series) -> dict:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec else np.nan
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_labelled": int(truth.sum()),
        "n_flagged": int(pred.sum()),
    }


def e0b(runs: pd.DataFrame, ada: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Our run flags against the ADA hand labels on the runs both have. Returns (main table
    with one row per flag, all rule variants)."""
    a = ada.copy()
    a["bed"] = a["bed"].astype(str)
    a["plant"] = a["plant"].astype(str)
    j = runs.merge(a, on=schema.KEYS, how="inner")
    rows = []
    for rule, (flag, label, fn) in QC_RULES.items():
        truth = fn(j).astype(bool) if len(j) else pd.Series(dtype=bool)
        pred = j[flag].astype(bool) if len(j) else pd.Series(dtype=bool)
        rows.append(
            {
                "flag": flag,
                "ada_label": label,
                **_agreement(pred, truth),
                "n_runs": len(j),
                "n_plants": int(j["plant_id"].nunique()) if len(j) else 0,
                "rule": rule,
            }
        )
    var = pd.DataFrame(rows)
    main = var[var["rule"].isin(QC_MAIN_RULES)].reset_index(drop=True)
    main["meets_target_recall"] = main["recall"] >= 0.7
    return main, var


# ---------------------------------------------------------------- E1 noise floor
def unit_of(trait: str) -> str:
    """Unit of a contract trait or of a pipeline extra such as ``cyl_height_p75``."""
    return schema.TRAITS.get(trait, schema.TRAITS.get(trait.removeprefix("cyl_"), ""))


def e1(runs: pd.DataFrame, frames: pd.DataFrame | None, traits: list[str] | None = None) -> pd.DataFrame:
    """Noise (median frame-to-frame MAD over qc_pass runs), signal (|control - deficit| of the
    baseline change on 2025-09-14, 15 h) and their ratio per trait. ``traits`` defaults to the
    contract traits; the pipeline's ``cyl_*`` extras can be passed for the secondary analysis."""
    traits = list(schema.TRAITS) if traits is None else list(traits)
    traits = [t for t in traits if t in runs.columns and (frames is None or t in frames.columns)]
    p = passing(runs)
    if frames is not None:
        f = frames.merge(p[schema.KEYS], on=schema.KEYS, how="inner")
        # the pipeline records the frames it used for the run median ("used") and whether a
        # frame gave traits ("ok"); the noise is measured on exactly those frames
        flags = [c for c in ("used", "ok") if c in f.columns]
        f = f.assign(_use=np.logical_and.reduce([f[c].astype(bool) for c in flags]))
        noise = stats.noise_floor(f, traits, used_col="_use")
        noise["noise_method"] = f"frames table, frames with {' and '.join(flags)} True"
    else:
        noise = stats.noise_from_run_mad(p, traits)
        noise["noise_method"] = "runs table <trait>_mad (no frame table)"
    p15 = _in(p[p["slot"] == 15], (schema.BASELINE_DATES[0], SIGNAL_DATE))
    norm = stats.baseline_normalise(p15, traits, mode="diff")
    sig_rows = []
    for t in traits:
        d = norm[[t, "date", "treatment", "plant_id"]].dropna()
        per = d.groupby(["date", "treatment", "plant_id"])[t].mean().reset_index()
        ok_days = [
            day
            for day, g in per.groupby("date")
            if (g["treatment"] == "control").sum() >= 2 and (g["treatment"] == "deficit").sum() >= 2
        ]
        day = max(ok_days) if ok_days else None
        sig = raw = np.nan
        if day is not None:
            g = per[per["date"] == day].groupby("treatment")[t].mean()
            sig = abs(g["control"] - g["deficit"])
            r = p15[p15["date"] == day].groupby("treatment")[t].mean()
            raw = abs(r.get("control", np.nan) - r.get("deficit", np.nan))
        sig_rows.append({"trait": t, "signal": sig, "signal_raw": raw, "signal_date": day})
    out = noise.merge(pd.DataFrame(sig_rows), on="trait")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["snr"] = out["signal"] / out["noise"].where(out["noise"] > 0)
        out["snr_within_column"] = out["signal"] / out["noise_within_column"].where(out["noise_within_column"] > 0)
    out["reliable"] = (out["snr"] >= SNR_MIN).fillna(False)
    out["unit"] = out["trait"].map(unit_of)
    cols = [
        "trait",
        "noise",
        "signal",
        "snr",
        "reliable",
        "unit",
        "noise_within_column",
        "snr_within_column",
        "signal_raw",
        "signal_date",
        "n_runs",
        "frames_per_run",
        "noise_method",
    ]
    return out[cols]


# ---------------------------------------------------------------- E2 beds before the cutoff
def e2(runs: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    """``trait ~ day * bed + (1 | plant)`` on 15 h runs, 2025-08-20 to 2025-09-02, W1 vs W2."""
    p = _in(passing(runs), E2_WINDOW)
    p = p[(p["slot"] == 15) & p["bed"].isin(["W1", "W2"])].copy()
    if p.empty or p["bed"].nunique() < 2:
        return pd.DataFrame()
    p["day"] = [(d - E2_WINDOW[0]).days for d in p["date"]]
    rows = []
    for t in traits:
        r = stats.trend_interaction(p, t, day_col="day", group_col="bed", treated="W2")
        rows.append(
            {
                "trait": t,
                "day_bed_coef": r["coef"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p_value": r["p_value"],
                "w1_slope": r["ref_slope"],
                "unit": f"{schema.TRAITS[t]}/day",
                "n_rows": r["n_rows"],
                "n_plants": r["n_plants"],
                "converged": r["converged"],
                "first_date": p["date"].min(),
                "last_date": p["date"].max(),
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = stats.holm(out["p_value"].to_numpy())
    out["warning"] = (out["p_value"] < ALPHA).fillna(False)
    return out


# ---------------------------------------------------------------- E3 stress onset
def symptom_reference(deficit_plants: pd.DataFrame, symptoms: pd.DataFrame | None) -> tuple:
    """Earliest first written drought symptom among ``deficit_plants`` (bed, plant); else the
    bed-wide protocol note. Returns (date or None, source text)."""
    if symptoms is not None and len(deficit_plants):
        s = symptoms.copy()
        s["bed"] = s["bed"].astype(str)
        s["plant"] = s["plant"].astype(str)
        hit = deficit_plants.merge(s, on=["bed", "plant"], how="inner")
        if len(hit):
            first = hit.sort_values("first_symptom_date").iloc[0]
            return first["first_symptom_date"], (
                f"BBCH sheet: {first['bed']}_{first['plant']} '{first['note']}' "
                f"({len(hit)} of {len(deficit_plants)} deficit plants have a written symptom)"
            )
    beds = sorted(set(deficit_plants["bed"])) if len(deficit_plants) else []
    dates = [envdata.PROTOCOL_BED_SYMPTOM[b] for b in beds if b in envdata.PROTOCOL_BED_SYMPTOM]
    if dates:
        return min(dates), "Raw Protocol 8.9.25: many plants in beds 2 and 3 wilting"
    return None, "none"


def _n_plants(df: pd.DataFrame, trait: str, treatment: str) -> int:
    """Plants of one group with at least one value of ``trait``."""
    d = df[(df["treatment"] == treatment) & df[trait].notna()]
    return int(d["plant_id"].nunique())


def e3(
    runs: pd.DataFrame,
    traits: list[str],
    symptoms: pd.DataFrame | None,
    beds: tuple[str, ...] = ("W1", "W2"),
    slots: tuple[int, ...] = (15,),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trend (LMM on % change from baseline), daily onset (Mann-Whitney, Holm over days, two
    consecutive days) and lead time against the first written symptom.

    ``slots`` = (15,) is the plan. A sensitivity run with the daytime slots (9, 12, 15, 18)
    normalises every run by the plant's baseline over the same slots and tests the plant's daily
    mean; it keeps plants whose 15 h baseline runs failed QC.
    Returns (onset table, daily tests, normalised table for the figures)."""
    p = passing(runs)
    p = _in(p[p["slot"].isin(slots) & p["bed"].isin(beds)], (E3_WINDOW[0], E3_PLOT_END))
    norm = stats.baseline_normalise(p, traits, mode="pct")
    norm["day"] = norm["days_since_cutoff"].astype(float)
    fit_rows = _in(norm, E3_WINDOW)
    test_rows = _in(norm, schema.TREATMENT_WINDOW)
    analysed = norm[norm["baseline_n"] > 0]  # plants without a baseline drop out
    deficit_plants = analysed.loc[analysed["treatment"] == "deficit", ["bed", "plant"]].drop_duplicates()
    sym_date, sym_src = symptom_reference(deficit_plants, symptoms)
    rows, daily_all = [], []
    for t in traits:
        tr = stats.trend_interaction(fit_rows, t, day_col="day")
        daily = stats.daily_tests(test_rows, t)
        on = stats.onset_day(daily, "p_holm", ALPHA)
        on_raw = stats.onset_day(daily, "p_raw", ALPHA)
        m = len(daily.dropna(subset=["p_raw"])) if len(daily) else 0
        min_p = float(daily["min_p"].min()) if len(daily) else np.nan
        rows.append(
            {
                "trait": t,
                "slope_interaction": tr["coef"],
                "ci_low": tr["ci_low"],
                "ci_high": tr["ci_high"],
                "p_value": tr["p_value"],
                "onset_day": schema.days_since_cutoff(on) if on else np.nan,
                "lead_vs_symptom_days": (on - sym_date).days if on and sym_date else np.nan,
                "onset_date": on,
                "onset_day_unadjusted": schema.days_since_cutoff(on_raw) if on_raw else np.nan,
                "onset_date_unadjusted": on_raw,
                "lead_unadjusted_days": (on_raw - sym_date).days if on_raw and sym_date else np.nan,
                "symptom_date": sym_date,
                "symptom_source": sym_src,
                "control_slope": tr["ref_slope"],
                "unit": "% of baseline per day",
                "n_control": _n_plants(fit_rows, t, "control"),
                "n_deficit": _n_plants(fit_rows, t, "deficit"),
                "n_days_tested": m,
                "min_attainable_p": min_p,
                "holm_reachable": bool(m and min_p < ALPHA / m),
                "converged": tr["converged"],
            }
        )
        if len(daily):
            daily = daily.assign(
                trait=t,
                significant=(daily["p_holm"] < ALPHA).fillna(False),
                significant_unadjusted=(daily["p_raw"] < ALPHA).fillna(False),
            )
            daily_all.append(daily)
    out = pd.DataFrame(rows)
    out = out.sort_values(["onset_day", "onset_day_unadjusted", "p_value"], na_position="last")
    out.insert(0, "rank", range(1, len(out) + 1))
    daily_cols = [
        "trait",
        "date",
        "p_raw",
        "p_holm",
        "significant",
        "significant_unadjusted",
        "n_treated",
        "n_reference",
        "effect",
        "median_diff",
        "min_p",
    ]
    daily_df = pd.concat(daily_all, ignore_index=True)[daily_cols] if daily_all else pd.DataFrame(columns=daily_cols)
    return out.reset_index(drop=True), daily_df, norm


# ---------------------------------------------------------------- E4 daily rhythm
def daily_covariates(days: list[date], air: pd.DataFrame | None, par: pd.DataFrame | None, bed: str) -> pd.DataFrame:
    """Per-day mean VPD (air sensor used for ``bed``) and PAR between 09:00 and 15:00 local."""
    out = pd.DataFrame({"date": days})
    out["vpd_kpa"] = (
        envdata.daily_window_mean(envdata.air_for_bed(air, bed), "vpd_kpa", days).to_numpy()
        if air is not None
        else np.nan
    )
    out["par_mv"] = envdata.daily_window_mean(par, "par_mv", days).to_numpy() if par is not None else np.nan
    return out


def e4(
    runs: pd.DataFrame,
    traits: list[str],
    air: pd.DataFrame | None,
    par: pd.DataFrame | None,
    beds: tuple[str, ...] = ("W1", "W2"),
    exclude_dark: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Midday depression (15 h - 09 h) and overnight recovery (06 h next day - 18 h) per plant
    per day, and ``depression ~ days_since_cutoff * treatment + VPD + (1 | plant)``.

    Dark runs (``qc_dark``; every 06 h run in the treatment window) are segmented by height
    alone, so the overnight recovery compares two vegetation rules. ``exclude_dark`` drops dark
    runs (and with them the recovery) as a sensitivity check of the midday depression.
    Returns (diurnal table, model table); both empty if the runs lack 09 h and 15 h."""
    p = passing(runs)
    p = p[p["bed"].isin(beds)]
    if exclude_dark:
        p = p[~p["qc_dark"]]
    p = _in(p, (E4_WINDOW[0], E4_WINDOW[1] + timedelta(days=1)))
    slots = set(p["slot"].unique())
    if not {9, 15} <= slots:
        return pd.DataFrame(), pd.DataFrame()
    idx = ["bed", "plant", "plant_id", "treatment", "date"]
    v = p.groupby(idx + ["slot"], as_index=False)[traits].mean()
    days = sorted(set(v["date"]))
    cov = pd.concat([daily_covariates(days, air, par, b).assign(bed=b) for b in beds], ignore_index=True)
    parts = []
    for t in traits:
        w = v.pivot_table(index=idx, columns="slot", values=t).reset_index()
        w.columns = [c if isinstance(c, str) else int(c) for c in w.columns]

        def col(s, frame=w):
            return frame[s] if s in frame.columns else pd.Series(np.nan, index=frame.index)

        dep = col(15) - col(9)
        nxt = w[["plant_id", "date"]].assign(v06=col(6))
        nxt["date"] = [d - timedelta(days=1) for d in nxt["date"]]
        w = w.merge(nxt, on=["plant_id", "date"], how="left")
        part = w[idx].assign(
            trait=t,
            midday_depression=dep.to_numpy(),
            overnight_recovery=(w["v06"] - col(18, w)).to_numpy(),
            midday_depression_pct=(100 * dep / ((col(9) + col(15)) / 2)).to_numpy(),
        )
        parts.append(part)
    diurnal = pd.concat(parts, ignore_index=True)
    diurnal = _in(diurnal, E4_WINDOW)
    diurnal["days_since_cutoff"] = [schema.days_since_cutoff(d) for d in diurnal["date"]]
    diurnal = diurnal.merge(cov, on=["bed", "date"], how="left")
    model_rows = []
    for t in traits:
        d = diurnal[diurnal["trait"] == t].rename(
            columns={"days_since_cutoff": "dsc", "vpd_kpa": "vpd", "par_mv": "par"}
        )
        d = d.assign(trt=(d["treatment"] == "deficit").astype(float))
        for resp in ("midday_depression", "overnight_recovery"):
            dd = d.dropna(subset=[resp, "vpd"]).rename(columns={resp: "y"})
            ok = dd["trt"].nunique() == 2 and dd["plant_id"].nunique() >= 3
            res = stats.fit_lmm(dd, "y ~ dsc * trt + vpd") if ok else None
            dpar = dd.dropna(subset=["par"])
            res_par = stats.fit_lmm(dpar, "y ~ dsc * trt + vpd + par") if ok and len(dpar) > 10 else None
            inter, vpdt = stats.lmm_term(res, "dsc:trt"), stats.lmm_term(res, "vpd")
            daily = stats.daily_tests(_in(d.rename(columns={resp: "val"}), schema.TREATMENT_WINDOW), "val")
            on = stats.onset_day(daily, "p_holm")
            on_raw = stats.onset_day(daily, "p_raw")
            model_rows.append(
                {
                    "trait": t,
                    "response": resp,
                    "slope_interaction": inter["coef"],
                    "ci_low": inter["ci_low"],
                    "ci_high": inter["ci_high"],
                    "p_value": inter["p_value"],
                    "vpd_coef": vpdt["coef"],
                    "vpd_p": vpdt["p_value"],
                    "slope_interaction_par_adjusted": stats.lmm_term(res_par, "dsc:trt")["coef"],
                    "onset_day": schema.days_since_cutoff(on) if on else np.nan,
                    "onset_date": on,
                    "onset_day_unadjusted": schema.days_since_cutoff(on_raw) if on_raw else np.nan,
                    "unit": f"{unit_of(t)} per day",
                    "n_rows": len(dd),
                    "n_plants": int(dd["plant_id"].nunique()),
                    "slots_present": ",".join(str(s) for s in sorted(slots)),
                }
            )
    return diurnal, pd.DataFrame(model_rows)


# ---------------------------------------------------------------- E5 turgor
def e5(
    runs: pd.DataFrame,
    traits: list[str],
    turgor: pd.DataFrame,
    slots: tuple[int, ...] | None = None,
    exclude_dark: bool = False,
    detrend: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Per-plant Spearman and pooled W2 repeated-measures correlation of each trait with leaf
    turgor at lags -2..+2 days. Returns (table, best structural proxy).

    Defaults follow the plan (every qc_pass run). Sensitivity options: ``slots`` keeps only
    those slots, ``exclude_dark`` drops height-only (dark) runs, ``detrend`` removes each
    plant's linear time trend from trait and turgor first, so only deviations from the shared
    drought trend are correlated."""
    p = _in(passing(runs), E5_WINDOW)
    p = p[p["plant_id"].isin(set(turgor["plant_id"]))]
    if slots is not None:
        p = p[p["slot"].isin(slots)]
    if exclude_dark:
        p = p[~p["qc_dark"]]
    if p.empty:
        return pd.DataFrame(), {}
    res = stats.turgor_corr(p, turgor, traits, detrend=detrend)
    res["slots"] = ",".join(str(s) for s in sorted(p["slot"].unique()))
    pooled = res[(res["method"] == "rmcorr") & res["rho"].notna()]
    basis = "pooled repeated-measures r"
    if pooled.empty:  # fewer than two clip plants in every bed: fall back to single plants
        # only deficit plants can show turgor loss; a control clip barely moves, and its rho
        # with a growing trait is a shared time trend, not water status
        deficit_beds = [b for b, t in schema.BED_TREATMENT.items() if t == "deficit"]
        pooled = res[(res["method"] == "spearman") & res["rho"].notna() & res["bed"].isin(deficit_beds)]
        basis = "single-plant Spearman rho (no bed has two clip plants)"
    best = {}
    if len(pooled):
        sig = pooled[pooled["p_value"] < ALPHA]
        pick = sig if len(sig) else pooled
        r = pick.loc[pick["rho"].abs().idxmax()]
        per = res[
            (res["method"] == "spearman")
            & (res["trait"] == r["trait"])
            & (res["lag_days"] == r["lag_days"])
            & (res["bed"] == r["bed"])
        ]
        best = {
            "trait": r["trait"],
            "lag_days": int(r["lag_days"]),
            "r": float(r["rho"]),
            "p_value": float(r["p_value"]),
            "n": int(r["n"]),
            "significant": bool(len(sig)),
            "pooled": r["plant"],
            "basis": basis,
            "per_plant_median_rho": float(per["rho"].median()),
            "n_plants": int(per["rho"].notna().sum()),
        }
    return res, best


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=8)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))


def _events(ax, label=False):
    for d, name in ((schema.CUTOFF, "cutoff"), (schema.REWATER, "re-water")):
        ax.axvline(pd.Timestamp(d), color=MUTED, linestyle="--", linewidth=0.8)
        if label:
            ax.text(pd.Timestamp(d), ax.get_ylim()[1], f" {name}", color=MUTED, fontsize=7, va="top")


def _save(fig, path: Path, tag: str):
    if tag:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=8, color="#d03b3b")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def fig_e0a(mi: pd.DataFrame, path: Path, tag: str):
    if mi.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for _pid, g in mi.groupby("plant_id"):
        trt = "control" if str(g["treatment"].iloc[0]) == "W1" else "deficit"
        ax.plot(g["date"], 1000 * g["height_estimate"], color=COLORS[trt], lw=0.8, alpha=0.35)
    for bed, trt in (("W1", "control"), ("W2", "deficit")):
        m = mi[mi["treatment"] == bed].groupby("date")["height_estimate"].mean()
        ax.plot(
            m.index,
            1000 * m.values,
            color=COLORS[trt],
            lw=2,
            marker="o",
            ms=4,
            label=f"{bed} ({trt})",
        )
    ax.set_ylabel("paper height, 0.60 m - z_p25 (mm)")
    ax.set_title("E0a paper parity: frame 7, 15 h, Tier-1 plants", fontsize=10, loc="left")
    _style(ax)
    _date_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path, tag)


def fig_e1(t: pd.DataFrame, path: Path, tag: str):
    d = t.sort_values("snr", na_position="first")
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(d) + 1.2))
    colors = [COLORS["control"] if r else "#c3c2b7" for r in d["reliable"]]
    snr = d["snr"].clip(lower=1e-2).fillna(1e-2)
    ax.barh(d["trait"], snr, color=colors, height=0.6)
    ax.axvline(SNR_MIN, color=INK, lw=0.8, linestyle="--")
    ax.text(SNR_MIN, len(d) - 0.4, f" SNR = {SNR_MIN:g}", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("signal-to-noise ratio (2025-09-14 treatment difference / frame MAD)")
    ax.set_title("E1 noise floor: blue = reliable, grey = left out", fontsize=10, loc="left")
    _style(ax)
    _save(fig, path, tag)


def fig_e3_trajectories(norm: pd.DataFrame, onset: pd.DataFrame, path: Path, tag: str):
    traits = list(onset["trait"])
    if not traits:
        return
    n = len(traits)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False, sharex=True)
    for ax, t in zip(axes.flat, traits, strict=False):
        d = norm[["date", "treatment", "plant_id", t]].dropna()
        for pid, g in d.groupby("plant_id"):
            g = g.groupby("date")[t].mean()
            trt = d.loc[d["plant_id"] == pid, "treatment"].iloc[0]
            ax.plot(pd.to_datetime(list(g.index)), g.values, color=COLORS[trt], lw=0.5, alpha=0.3)
        for trt in ("control", "deficit"):
            g = d[d["treatment"] == trt].groupby(["date", "plant_id"])[t].mean().groupby("date")
            m, s = g.mean(), g.std()
            x = pd.to_datetime(list(m.index))
            ax.fill_between(x, m - s, m + s, color=COLORS[trt], alpha=0.18, lw=0)
            ax.plot(x, m.values, color=COLORS[trt], lw=2, label=trt)
        row = onset[onset["trait"] == t].iloc[0]
        if pd.notna(row["onset_date"]):
            ax.axvline(pd.Timestamp(row["onset_date"]), color=COLORS["deficit"], lw=1.2, linestyle=":")
        ax.axhline(0, color=MUTED, lw=0.6)
        _style(ax)
        _events(ax)
        _date_axis(ax)
        ax.set_title(f"{t} ({schema.TRAITS[t]})", fontsize=9, loc="left")
        ax.set_ylabel("% change from baseline", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "E3 trajectories, 15 h runs: mean +- 1 SD per group, thin lines = plants; "
        "dashed = cutoff / re-water, dotted = onset (Holm)",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path, tag)


def fig_e3_onset(onset: pd.DataFrame, path: Path, tag: str):
    if onset.empty:
        return
    d = onset.iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 0.35 * len(d) + 1.4), sharey=True)
    y = np.arange(len(d))
    a1.scatter(d["onset_day"], y, color=COLORS["deficit"], s=40, label="onset, Holm over days", zorder=3)
    a1.scatter(
        d["onset_day_unadjusted"],
        y,
        facecolors="none",
        edgecolors=COLORS["deficit"],
        s=40,
        label="onset, unadjusted",
        zorder=3,
    )
    sym = d["symptom_date"].dropna()
    if len(sym):
        a1.axvline(schema.days_since_cutoff(sym.iloc[0]), color=INK, lw=0.8, linestyle="--")
        a1.text(
            schema.days_since_cutoff(sym.iloc[0]),
            len(d) - 0.5,
            " first written symptom",
            fontsize=7,
        )
    a1.set_xlim(-0.5, 12.5)
    a1.set_yticks(y, d["trait"])
    a1.set_xlabel("onset (days since cutoff 2025-09-03)")
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    err = np.vstack([d["slope_interaction"] - d["ci_low"], d["ci_high"] - d["slope_interaction"]])
    a2.errorbar(d["slope_interaction"], y, xerr=err, fmt="o", color=COLORS["control"], ms=5, lw=1.2)
    a2.axvline(0, color=MUTED, lw=0.8)
    a2.set_xlabel("deficit - control slope (% of baseline / day, 95 % CI)")
    for ax in (a1, a2):
        _style(ax)
    fig.suptitle("E3 ranked by onset day", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path, tag)


def fig_e4_heatmap(runs: pd.DataFrame, trait: str, path: Path, tag: str, beds=("W1", "W2")):
    p = _in(passing(runs), E4_WINDOW)
    p = p[p["bed"].isin(beds)]
    if p.empty:
        return
    v = p.groupby(["bed", "plant_id", "date", "slot"])[trait].mean().reset_index()
    dm = v.groupby(["plant_id", "date"])[trait].transform("mean")
    ns = v.groupby(["plant_id", "date"])[trait].transform("size")
    v["dev"] = 100 * (v[trait] / dm - 1)
    v = v[ns >= 3]
    if v.empty:
        return
    days = sorted(set(v["date"]))
    slots = sorted(set(v["slot"]))
    mats = {}
    for b in beds:
        m = v[v["bed"] == b].groupby(["slot", "date"])["dev"].mean().unstack("date")
        mats[b] = m.reindex(index=slots, columns=days)
    lim = np.nanpercentile(np.abs(np.concatenate([m.to_numpy().ravel() for m in mats.values()])), 98)
    lim = float(lim) if np.isfinite(lim) and lim > 0 else 1.0
    fig, axes = plt.subplots(len(beds), 1, figsize=(10, 1.7 * len(beds) + 0.8), sharex=True, squeeze=False)
    for ax, b in zip(axes[:, 0], beds, strict=True):
        im = ax.imshow(
            mats[b].to_numpy(),
            aspect="auto",
            cmap=DIVERGING,
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(slots)), [f"{s:02d} h" for s in slots], fontsize=8)
        ax.set_title(f"{b} ({schema.BED_TREATMENT[b]})", fontsize=9, loc="left")
        for d, ls in ((schema.CUTOFF, "--"), (schema.REWATER, "--")):
            if d in days:
                ax.axvline(days.index(d) - 0.5, color=INK, lw=0.8, linestyle=ls)
    ticks = list(range(0, len(days), 3))
    axes[-1, 0].set_xticks(ticks, [days[i].strftime("%d.%m") for i in ticks], fontsize=8)
    cb = fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.9, pad=0.01)
    cb.set_label("% of the plant's daily mean", fontsize=8)
    fig.suptitle(
        f"E4 daily rhythm of {trait}: bed mean deviation from each plant's daily mean (dashed = cutoff / re-water)",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    _save(fig, path, tag)


def fig_e5(runs: pd.DataFrame, turgor: pd.DataFrame, best: dict, path: Path, tag: str):
    if not best:
        return
    t, lag = best["trait"], best["lag_days"]
    p = _in(passing(runs), E5_WINDOW)
    p = p[p["plant_id"].isin(set(turgor["plant_id"]))].copy()
    p["turgor"] = stats.turgor_per_run(p, turgor, lag)
    plants = sorted(p["plant_id"].unique(), key=lambda s: (s[:2] != "W2", s))
    n = len(plants)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), squeeze=False, sharex=True, sharey=True)

    def z(s):
        return (s - s.mean()) / s.std() if s.std() > 0 else s * 0

    for ax, pid in zip(axes.flat, plants, strict=False):
        g = p[p["plant_id"] == pid].sort_values("timestamp_utc").dropna(subset=[t, "turgor"])
        x = g["timestamp_utc"].dt.tz_convert(None)
        ax.plot(x, z(g[t]), color="#2a78d6", lw=1.2, marker="o", ms=2.5, label=t)
        ax.plot(x, -z(g["turgor"]), color="#eb6834", lw=1.2, label="turgor (sign flipped)")
        _style(ax)
        _events(ax)
        _date_axis(ax)
        ax.set_title(pid, fontsize=8, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(
        f"E5 best structural proxy: {t} against leaf turgor, lag {lag:+d} d; both z-scored "
        f"per plant; {best['basis']} = {best['r']:.2f} (p {_p(best['p_value'])})",
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    _save(fig, path, tag)


def fig_e5_lags(res: pd.DataFrame, path: Path, tag: str):
    pooled = res[res["method"] == "rmcorr"]
    if pooled.empty:
        return
    m = pooled.pivot_table(index="trait", columns="lag_days", values="rho")
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(m) + 1.3))
    ax.imshow(m.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1]), [f"{c:+d}" for c in m.columns], fontsize=8)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xlabel("lag (days; + = turgor reading before the trait)")
    ax.set_title(
        "E5 pooled W2 repeated-measures r, trait vs turgor value\n"
        "(value rises as turgor falls: r < 0 = trait falls with turgor)",
        fontsize=9,
        loc="left",
    )
    _save(fig, path, tag)


# ---------------------------------------------------------------- synthetic inputs
def synthetic_turgor(runs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Leaf-clip series for every plant in ``runs``: rises (turgor falls) with the deficit
    stress, falls after re-watering, plus a midday peak and noise. Development only."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-09-02 08:00", "2025-09-20 00:00", freq="10min", tz="UTC")
    ds = (t - pd.Timestamp(schema.CUTOFF.isoformat(), tz="UTC")).total_seconds() / 86400
    hours = t.hour + t.minute / 60
    rows = []
    for (bed, plant), _ in runs.groupby(["bed", "plant"]):
        deficit = schema.BED_TREATMENT[str(bed)] == "deficit"
        stress = np.clip(ds, 0, 12) * deficit
        val = 1830 + 3.0 * stress + 2.0 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 1.0, len(t))
        rows.append(
            pd.DataFrame(
                {
                    "utc_timestamp": t,
                    "sensor_number": -1,
                    "bed": str(bed),
                    "plant": str(plant),
                    "plant_id": f"{bed}_{plant}",
                    "turgor_value": val,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def synthetic_inputs(seed: int = 0, plants_per_bed: int = 10, ada: pd.DataFrame | None = None) -> dict:
    """Synthetic runs (``schema.synthetic_runs``), a parity table built from them, QC flags
    copied from the real ADA labels with 15 % random flips, and synthetic turgor."""
    rng = np.random.default_rng(seed)
    runs = prepare_runs(schema.synthetic_runs(plants_per_bed=plants_per_bed, seed=seed))
    par = runs[runs["hour"] == 15][["bed", "plant", "date", "hour"]].copy()
    par["height_parity_mm"] = runs.loc[par.index, "height_p75"] + rng.normal(0, 2, len(par))
    par["num_points_vegetation"] = 5000
    if ada is not None:
        a = ada.copy()
        a["bed"] = a["bed"].astype(str)
        a["plant"] = a["plant"].astype(str)
        j = runs[schema.KEYS].reset_index().merge(a, on=schema.KEYS, how="inner").set_index("index")
        for flag, col in (
            ("qc_dark", "ada_dark"),
            ("qc_not_centred", "ada_not_centred"),
            ("qc_no_plant", "ada_no_plant"),
        ):
            flip = rng.random(len(j)) < 0.15
            runs.loc[j.index, flag] = j[col].to_numpy() ^ flip
    return {"runs": runs, "parity": par, "frames": None, "turgor": synthetic_turgor(runs, seed)}


# ---------------------------------------------------------------- summary
def _md_table(df: pd.DataFrame, cols: list[str], fmt: int = 3) -> str:
    if df is None or df.empty:
        return "_no rows_\n"
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}g}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    body = "".join(
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |\n" for r in d.itertuples(index=False)
    )
    return head + body


def write_summary(out: Path, source: str, sections: list[tuple[str, str]]):
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# 3D experiments summary ({source} data)\n",
        f"Generated {stamp} by `python -m uiwue.threed.experiments`.\n",
    ]
    if source == "synthetic":
        lines.append(
            "**SYNTHETIC DEMONSTRATION. These numbers come from `schema.synthetic_runs()` and are not results.**\n"
        )
    for title, body in sections:
        lines.append(f"\n## {title}\n\n{body}")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- CLI
EXPERIMENTS = ("E0a", "E0b", "E1", "E2", "E3", "E4", "E5")


def _write(df: pd.DataFrame, out: Path, name: str, source: str):
    need = schema.RESULT_FILES.get(name, [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing contract columns {missing}")
    df = df.assign(source=source)
    df.to_csv(out / name, index=False, lineterminator="\n")
    _log(f"  wrote {out / name} ({len(df)} rows)")


def _write_extra(df: pd.DataFrame, out: Path, name: str, source: str):
    df.assign(source=source).to_csv(out / name, index=False, lineterminator="\n")
    _log(f"  wrote {out / name} ({len(df)} rows)")


def _traits_for_e3_e5(out: Path, e1_table: pd.DataFrame | None, all_traits: bool) -> tuple[list[str], str]:
    if all_traits:
        return list(schema.TRAITS), "all traits (--all-traits)"
    t = e1_table
    if t is None and (out / "e1_noise.csv").exists():
        t = pd.read_csv(out / "e1_noise.csv")
    if t is None:
        return list(schema.TRAITS), "all traits (E1 not run)"
    rel = [x for x in t.loc[t["reliable"].astype(bool), "trait"] if x in schema.TRAITS]
    if not rel:
        return list(schema.TRAITS), "all traits (E1 found none reliable; shown for inspection)"
    return rel, f"E1-reliable traits ({len(rel)})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    ap.add_argument("--runs", type=Path, default=schema.TRAITS_RUNS)
    ap.add_argument("--frames", type=Path, default=schema.TRAITS_FRAMES)
    ap.add_argument("--parity", type=Path, default=schema.TRAITS_PARITY)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: RESULTS_DIR (synthetic: RESULTS_DIR/synthetic)",
    )
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic tables (demonstration)")
    ap.add_argument("--all-traits", action="store_true", help="E2-E5 on all traits, not only E1-reliable ones")
    ap.add_argument("--beds", nargs="+", default=["W1", "W2"], help="beds for E3/E4 (default W1 W2)")
    a = ap.parse_args(argv)

    source = "synthetic" if a.synthetic else "real"
    out = a.out or (schema.RESULTS_DIR / "synthetic" if a.synthetic else schema.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC DEMONSTRATION - not results" if a.synthetic else ""
    ada = envdata.load_ada_qc() if envdata.ADA_QC_JSON.exists() else None

    if a.synthetic:
        syn = synthetic_inputs(ada=ada)
        runs, frames, parity, turgor = syn["runs"], syn["frames"], syn["parity"], syn["turgor"]
    else:
        if not a.runs.exists():
            _log(f"error: {a.runs} not found. Use --runs PATH, or --synthetic for a demonstration.")
            return 2
        runs = prepare_runs(pd.read_parquet(a.runs))
        frames = load_frames(a.frames)
        parity = pd.read_parquet(a.parity) if a.parity.exists() else None
        turgor = None
    hours = sorted(int(s) for s in runs["slot"].unique())
    _log(
        f"{source} runs: {len(runs)} rows, {runs['plant_id'].nunique()} plants, "
        f"{runs['date'].min()} .. {runs['date'].max()}, slots {hours}, qc_pass {runs['qc_pass'].mean():.0%}"
    )
    sections: list[tuple[str, str]] = [
        (
            "Input",
            (
                f"- runs: {len(runs)} rows, {runs['plant_id'].nunique()} plants "
                f"({', '.join(sorted(runs['plant_id'].unique()))})\n"
                f"- dates {runs['date'].min()} .. {runs['date'].max()}; slots {hours}; "
                f"qc_pass {int(runs['qc_pass'].sum())} of {len(runs)}\n"
                f"- frame table: {'yes' if frames is not None else 'no'}; parity table: "
                f"{'yes' if parity is not None else 'no'}\n"
            ),
        )
    ]
    e1_table = None

    if "E0a" in a.exp:
        _log("E0a paper parity")
        t, mi = e0a(runs, parity)
        if t.empty:
            _log("  skipped: no parity table")
            sections.append(("E0a paper parity", "_skipped: no parity table_\n"))
        else:
            _write(t, out, "e0a_parity.csv", source)
            fig_e0a(mi, out / "e0a_parity.png", tag)
            sections.append(
                (
                    "E0a paper parity",
                    _md_table(
                        t,
                        [
                            "quantity",
                            "ours",
                            "paper",
                            "rel_diff",
                            "pass",
                            "ci_low",
                            "ci_high",
                            "p_value",
                            "n_rows",
                            "n_plants",
                        ],
                    ),
                )
            )
    if "E0b" in a.exp:
        _log("E0b QC agreement")
        if ada is None:
            _log("  skipped: ADA labels not found")
        else:
            main_t, var = e0b(runs, ada)
            if main_t["n_runs"].max() == 0:
                _log("  skipped: no run of the trait table has an ADA label (Tier-2 plants missing)")
                sections.append(("E0b QC agreement", "_skipped: no labelled runs in the table_\n"))
            else:
                _write(main_t, out, "e0b_qc_agreement.csv", source)
                var.assign(source=source).to_csv(
                    out / "e0b_qc_agreement_variants.csv", index=False, lineterminator="\n"
                )
                sections.append(
                    (
                        "E0b QC agreement",
                        _md_table(
                            var,
                            [
                                "flag",
                                "rule",
                                "precision",
                                "recall",
                                "f1",
                                "n_labelled",
                                "n_flagged",
                                "n_runs",
                                "n_plants",
                            ],
                        ),
                    )
                )
    if "E1" in a.exp:
        _log("E1 noise floor")
        e1_table = e1(runs, frames)
        _write(e1_table, out, "e1_noise.csv", source)
        fig_e1(e1_table, out / "e1_snr.png", tag)
        sections.append(
            (
                "E1 noise floor",
                _md_table(
                    e1_table,
                    [
                        "trait",
                        "unit",
                        "noise",
                        "noise_within_column",
                        "signal",
                        "snr",
                        "reliable",
                        "signal_date",
                        "n_runs",
                        "frames_per_run",
                    ],
                )
                + f"\nNoise source: {e1_table['noise_method'].iloc[0]}.\n",
            )
        )
        if all(c in runs.columns for c in CYL_TRAITS):
            e1_cyl = e1(runs, frames, list(CYL_TRAITS))
            _write_extra(e1_cyl, out, "e1_noise_cyl.csv", source)
            sections.append(
                (
                    "E1 secondary: cyl_* traits (all vegetation in the cylinder, no clustering)",
                    _md_table(
                        e1_cyl,
                        [
                            "trait",
                            "unit",
                            "noise",
                            "noise_within_column",
                            "signal",
                            "snr",
                            "reliable",
                            "n_runs",
                        ],
                    ),
                )
            )
    traits, trait_note = _traits_for_e3_e5(out, e1_table, a.all_traits)
    _log(f"traits for E2-E5: {trait_note}: {traits}")

    if "E2" in a.exp:
        _log("E2 beds before the cutoff")
        t = e2(runs, traits)
        if t.empty:
            _log(f"  skipped: no 15 h qc_pass runs of W1 and W2 in {E2_WINDOW[0]} .. {E2_WINDOW[1]}")
            sections.append(("E2 bed baseline", "_skipped: no pre-cutoff 15 h runs_\n"))
        else:
            _write(t, out, "e2_bed_baseline.csv", source)
            sections.append(
                (
                    "E2 bed baseline",
                    f"Traits: {trait_note}.\n\n"
                    + _md_table(
                        t,
                        [
                            "trait",
                            "day_bed_coef",
                            "ci_low",
                            "ci_high",
                            "p_value",
                            "p_holm",
                            "warning",
                            "w1_slope",
                            "unit",
                            "n_plants",
                            "first_date",
                            "last_date",
                        ],
                    ),
                )
            )
    if "E3" in a.exp:
        _log("E3 stress onset")
        symptoms = envdata.first_symptoms() if envdata.METADATA_XLSX.exists() else None
        onset, daily, norm = e3(runs, traits, symptoms, beds=tuple(a.beds))
        _write(onset, out, "e3_onset.csv", source)
        _write(daily, out, "e3_daily_tests.csv", source)
        fig_e3_trajectories(norm, onset, out / "e3_trajectories.png", tag)
        fig_e3_onset(onset, out / "e3_onset.png", tag)
        sections.append(
            (
                "E3 stress onset (ranked)",
                f"Traits: {trait_note}.\n\n"
                + _md_table(
                    onset,
                    [
                        "rank",
                        "trait",
                        "onset_day",
                        "onset_date",
                        "onset_day_unadjusted",
                        "lead_vs_symptom_days",
                        "lead_unadjusted_days",
                        "slope_interaction",
                        "ci_low",
                        "ci_high",
                        "p_value",
                        "n_control",
                        "n_deficit",
                        "holm_reachable",
                    ],
                )
                + f"\nSymptom reference: {onset['symptom_date'].iloc[0]} ({onset['symptom_source'].iloc[0]}).\n",
            )
        )
        onset_cols = [
            "rank",
            "trait",
            "onset_day",
            "onset_day_unadjusted",
            "slope_interaction",
            "ci_low",
            "ci_high",
            "p_value",
            "n_control",
            "n_deficit",
        ]
        if set(DAYTIME_SLOTS) <= set(hours):
            od, dd, nd = e3(runs, traits, symptoms, beds=tuple(a.beds), slots=DAYTIME_SLOTS)
            _write_extra(od, out, "e3_onset_daytime.csv", source)
            _write_extra(dd, out, "e3_daily_tests_daytime.csv", source)
            fig_e3_trajectories(nd, od, out / "e3_trajectories_daytime.png", tag)
            sections.append(
                (
                    "E3 sensitivity: daytime slots 09-18 h, plant-day means",
                    _md_table(od, onset_cols),
                )
            )
        if all(c in runs.columns for c in CYL_TRAITS):
            oc, dc, _ = e3(runs, list(CYL_TRAITS), symptoms, beds=tuple(a.beds))
            _write_extra(oc, out, "e3_onset_cyl.csv", source)
            _write_extra(dc, out, "e3_daily_tests_cyl.csv", source)
            sections.append(("E3 secondary: cyl_* traits, 15 h", _md_table(oc, onset_cols)))
    if "E4" in a.exp:
        _log("E4 daily rhythm")
        air = envdata.load_air() if envdata.AIR_READINGS.exists() else None
        par = envdata.load_par() if envdata.METADATA_XLSX.exists() else None
        diurnal, models = e4(runs, traits, air, par, beds=tuple(a.beds))
        if diurnal.empty:
            _log(f"  skipped: E4 needs the 09 h and 15 h slots; table has {hours}")
            sections.append(("E4 daily rhythm", f"_skipped: E4 needs all hours; slots present {hours}_\n"))
        else:
            _write(diurnal, out, "e4_diurnal.csv", source)
            models.assign(source=source).to_csv(out / "e4_models.csv", index=False, lineterminator="\n")
            heat = [t for t in HEATMAP_TRAITS if t in traits][:3] or traits[:1]
            for t in heat:
                fig_e4_heatmap(runs, t, out / f"e4_heatmap_{t}.png", tag, beds=tuple(a.beds))
            sections.append(
                (
                    "E4 daily rhythm",
                    _md_table(
                        models,
                        [
                            "trait",
                            "response",
                            "slope_interaction",
                            "ci_low",
                            "ci_high",
                            "p_value",
                            "vpd_coef",
                            "vpd_p",
                            "slope_interaction_par_adjusted",
                            "onset_day",
                            "onset_day_unadjusted",
                            "unit",
                            "slots_present",
                        ],
                    ),
                )
            )
            _, models_nd = e4(runs, traits, air, par, beds=tuple(a.beds), exclude_dark=True)
            _write_extra(models_nd, out, "e4_models_nodark.csv", source)
            sections.append(
                (
                    "E4 sensitivity: dark runs excluded (midday depression only)",
                    _md_table(
                        models_nd[models_nd["response"] == "midday_depression"],
                        [
                            "trait",
                            "slope_interaction",
                            "ci_low",
                            "ci_high",
                            "p_value",
                            "onset_day",
                            "onset_day_unadjusted",
                            "unit",
                            "n_plants",
                        ],
                    ),
                )
            )
    if "E5" in a.exp:
        _log("E5 turgor")
        if turgor is None:
            turgor = envdata.load_turgor() if envdata.LEAF_READINGS.exists() else None
        res, best = (pd.DataFrame(), {}) if turgor is None else e5(runs, traits, turgor)
        if res.empty:
            _log("  skipped: no qc_pass run of a leaf-clip plant in the table")
            sections.append(("E5 turgor", "_skipped: no clip plant in the table_\n"))
        else:
            _write(res, out, "e5_turgor.csv", source)
            fig_e5(runs, turgor, best, out / "e5_turgor_best.png", tag)
            fig_e5_lags(res, out / "e5_turgor_lags.png", tag)
            pooled = res[res["method"] == "rmcorr"]
            if pooled.empty:
                pooled = res
            pooled = pooled.sort_values(["trait", "lag_days", "plant"])
            best_txt = (
                (
                    f"Best structural proxy: **{best['trait']}** at lag {best['lag_days']:+d} d, {best['basis']} = "
                    f"{best['r']:.3f}, p {_p(best['p_value'])}, n = {best['n']}; per-plant median rho = "
                    f"{best['per_plant_median_rho']:.3f} over {best['n_plants']} plants"
                    f"{'' if best['significant'] else ' (NOT significant)'}.\n\n"
                )
                if best
                else "No structural proxy named: no deficit-bed clip plant (or pair of plants) "
                "in the table. The rows below are descriptive only.\n\n"
            )
            sections.append(
                (
                    "E5 turgor",
                    best_txt + _md_table(pooled, ["plant", "trait", "lag_days", "rho", "p_value", "n"]),
                )
            )
            variants = {
                "all qc_pass runs (plan)": {},
                "dark runs excluded": {"exclude_dark": True},
                "15 h only": {"slots": (15,)},
                "dark runs excluded, detrended": {"exclude_dark": True, "detrend": True},
            }
            vrows = []
            for name, kw in variants.items():
                rv = res if not kw else e5(runs, traits, turgor, **kw)[0]
                if len(rv):
                    vrows.append(rv.assign(variant=name))
            var = pd.concat(vrows, ignore_index=True)
            _write_extra(var, out, "e5_turgor_variants.csv", source)
            vp = var[var["method"] == "rmcorr"]
            if len(vp):
                piv = vp[vp["lag_days"].isin([0, 1])].pivot_table(
                    index=["trait", "lag_days"], columns="variant", values="rho"
                )
                piv = piv[[v for v in variants if v in piv.columns]].reset_index()
                sections.append(
                    (
                        "E5 sensitivity: pooled W2 r by variant (lags 0 and +1)",
                        _md_table(piv, list(piv.columns)),
                    )
                )
    write_summary(out, source, sections)
    _log(f"summary: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
