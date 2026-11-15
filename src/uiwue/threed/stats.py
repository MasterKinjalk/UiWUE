"""Statistics for the 3D trait table (plan/Kinjalk-Work-Plan.md, section 7).

Pure functions on pandas tables that follow ``uiwue.threed.schema``. ``experiments`` wires them
to files. Conventions:

- ``plant_id`` = ``<bed>_<plant>``; the grid position alone is not unique across beds.
- Treatment groups are the strings ``"control"`` and ``"deficit"`` (``schema.BED_TREATMENT``).
- ``date`` holds ``datetime.date`` objects (``ensure_dates`` converts other forms).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as sps

from uiwue.threed import schema


# ---------------------------------------------------------------- small helpers
def vpd(temp_c, rh_pct):
    """Vapour pressure deficit in kPa from air temperature (degC) and relative humidity (%).
    Saturation pressure by the Tetens formula: es = 0.6108 exp(17.27 T / (T + 237.3))."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    out = es * (1.0 - rh / 100.0)
    if isinstance(temp_c, pd.Series):
        return pd.Series(out, index=temp_c.index, name="vpd_kpa")
    return out


def mad(x) -> float:
    """Median absolute deviation from the median, unscaled (same as the pipeline's ``_mad``)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))))


def ensure_dates(s: pd.Series) -> pd.Series:
    """Return ``s`` as ``datetime.date`` objects (accepts dates, datetimes and ISO strings)."""
    if len(s) and all(type(v) is date for v in s.head(50)):
        return s
    return pd.to_datetime(s).dt.date


def add_plant_id(df: pd.DataFrame) -> pd.DataFrame:
    if "plant_id" not in df.columns:
        df = df.assign(plant_id=df["bed"].astype(str) + "_" + df["plant"].astype(str))
    return df


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values. NaN entries stay NaN and do not count as tests."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def min_mw_p(n1: int, n2: int) -> float:
    """Smallest two-sided p-value an exact Mann-Whitney test can give for group sizes n1, n2
    (complete separation, no ties): 2 / C(n1 + n2, n1)."""
    if n1 < 1 or n2 < 1:
        return float("nan")
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


# ---------------------------------------------------------------- baseline
def baseline_normalise(
    df: pd.DataFrame,
    traits: Iterable[str],
    mode: str = "pct",
    baseline_dates: Sequence[date] = schema.BASELINE_DATES,
    by: Sequence[str] = ("bed", "plant"),
    baseline_slots: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Express each trait relative to the plant's own baseline (mean over ``baseline_dates``).

    ``mode``: ``"pct"`` = 100 (value / baseline - 1); ``"diff"`` = value - baseline (trait
    units); ``"ratio"`` = value / baseline. ``by`` sets what one baseline belongs to; add
    ``"slot"`` for a baseline per time of day. ``baseline_slots`` limits the baseline to those
    slots. Rows whose plant has no baseline get NaN. Returns a copy with the trait columns
    replaced and a ``baseline_n`` column (baseline runs used)."""
    traits = list(traits)
    by = list(by)
    if mode not in ("pct", "diff", "ratio"):
        raise ValueError(f"unknown mode {mode!r}")
    out = df.copy()
    base_rows = out[out["date"].isin(list(baseline_dates))]
    if baseline_slots is not None:
        base_rows = base_rows[base_rows["slot"].isin(list(baseline_slots))]
    base = base_rows.groupby(by)[traits].mean()
    base["baseline_n"] = base_rows.groupby(by).size()
    joined = out[by].join(base, on=by)
    for t in traits:
        b = joined[t].astype(float)
        v = out[t].astype(float)
        if mode == "diff":
            out[t] = v - b
        else:
            b = b.where(b.abs() > 1e-9)
            out[t] = 100.0 * (v / b - 1.0) if mode == "pct" else v / b
    out["baseline_n"] = joined["baseline_n"].fillna(0).astype(int)
    return out


# ---------------------------------------------------------------- mixed models
def fit_lmm(data: pd.DataFrame, formula: str, group: str = "plant_id", reml: bool = False):
    """Random-intercept mixed model ``formula + (1 | group)``, fitted as in the ADA code
    (``analysis/plant_height/step_04_fit_lmm.py``): ML, L-BFGS with 300 iterations, falling back
    to the statsmodels default optimiser. Returns the statsmodels result, or None on failure."""
    import statsmodels.formula.api as smf

    data = data.reset_index(drop=True)
    model = smf.mixedlm(formula, data=data, groups=data[group])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.fit(reml=reml, method="lbfgs", maxiter=300)
        except Exception:
            try:
                return model.fit(reml=reml)
            except Exception:
                return None


def lmm_term(result, term: str, alpha: float = 0.05) -> dict:
    """Estimate, standard error, CI and p-value of one fixed-effect term."""
    nan = float("nan")
    if result is None or term not in result.params.index:
        return {"coef": nan, "se": nan, "ci_low": nan, "ci_high": nan, "p_value": nan}
    ci = result.conf_int(alpha=alpha).loc[term]
    return {
        "coef": float(result.params[term]),
        "se": float(result.bse[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(result.pvalues[term]),
    }


def trend_interaction(
    df: pd.DataFrame,
    value_col: str,
    day_col: str = "day",
    group_col: str = "treatment",
    treated: str = "deficit",
    plant_col: str = "plant_id",
) -> dict:
    """Fit ``value ~ day * treated + (1 | plant)`` and return the ``day:treated`` slope (the
    difference in daily slope, treated minus reference) plus the reference slope."""
    d = df[[value_col, day_col, group_col, plant_col]].dropna()
    d = d.rename(columns={value_col: "y", day_col: "day"})
    d["trt"] = (d[group_col] == treated).astype(float)
    if d["trt"].nunique() < 2 or d[plant_col].nunique() < 3:
        res = None
    else:
        res = fit_lmm(d, "y ~ day * trt", group=plant_col)
    inter = lmm_term(res, "day:trt")
    ref = lmm_term(res, "day")
    return {
        **inter,
        "ref_slope": ref["coef"],
        "n_rows": len(d),
        "n_plants": int(d[plant_col].nunique()),
        "converged": bool(getattr(res, "converged", False)) if res is not None else False,
    }


# ---------------------------------------------------------------- onset
def daily_tests(
    df: pd.DataFrame,
    value_col: str,
    day_col: str = "date",
    group_col: str = "treatment",
    treated: str = "deficit",
    reference: str = "control",
    plant_col: str = "plant_id",
    min_n: int = 2,
) -> pd.DataFrame:
    """Two-sided Mann-Whitney test of treated against reference on each day, Holm-corrected over
    the tested days. One value per plant per day (the mean if a plant has several runs).

    Columns: date, n_treated, n_reference, u, p_raw, p_holm, effect (rank-biserial correlation,
    positive = treated higher), median_diff (treated minus reference), min_p (smallest p the
    exact test can reach with these group sizes)."""
    d = df[[day_col, group_col, plant_col, value_col]].dropna()
    d = d.groupby([day_col, group_col, plant_col], as_index=False)[value_col].mean()
    rows = []
    for day, g in d.groupby(day_col, sort=True):
        a = g.loc[g[group_col] == treated, value_col].to_numpy(float)
        b = g.loc[g[group_col] == reference, value_col].to_numpy(float)
        row = {
            "date": day,
            "n_treated": len(a),
            "n_reference": len(b),
            "u": np.nan,
            "p_raw": np.nan,
            "effect": np.nan,
            "median_diff": np.nan,
            "min_p": min_mw_p(len(a), len(b)),
        }
        if len(a) >= min_n and len(b) >= min_n:
            r = sps.mannwhitneyu(a, b, alternative="two-sided")
            row.update(
                u=float(r.statistic),
                p_raw=float(r.pvalue),
                effect=2.0 * float(r.statistic) / (len(a) * len(b)) - 1.0,
                median_diff=float(np.median(a) - np.median(b)),
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(p_holm=pd.Series(dtype=float))
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    return out


def onset_day(
    daily: pd.DataFrame, p_col: str = "p_holm", alpha: float = 0.05, consecutive: int = 2
) -> date | None:
    """First day of the first run of ``consecutive`` significant days (adjacent in the sequence
    of tested days, so a day without data does not break a run). None if there is no such run."""
    if daily.empty:
        return None
    d = daily.sort_values("date").reset_index(drop=True)
    sig = (d[p_col] < alpha).fillna(False).to_numpy()
    for i in range(len(d) - consecutive + 1):
        if sig[i : i + consecutive].all():
            return d.loc[i, "date"]
    return None


# ---------------------------------------------------------------- noise floor
def noise_floor(
    frames: pd.DataFrame,
    traits: Iterable[str],
    run_keys: Sequence[str] = tuple(schema.KEYS),
    used_col: str | None = "ok",
    min_frames: int = 3,
) -> pd.DataFrame:
    """Frame-to-frame noise per trait (plan E1).

    For each run: the MAD of the trait over the frames the pipeline used (``used_col`` True).
    Noise = median of that MAD over runs. The 15 frames are a 3 x 5 grid of robot positions.
    With the corrected robot-x mapping (docs/threed/PIPELINE.md) frames from different grid
    columns agree as well as frames of one column, so this spread is close to the real
    frame-to-frame noise (ToF noise, small viewpoint effects and leaf motion between the
    sweeps about 40 s apart). ``noise_within_column`` repeats the calculation within each grid
    column (``frame % 3``); the two should then be similar.

    Columns: trait, noise, noise_within_column, n_runs, frames_per_run."""
    traits = list(traits)
    keys = list(run_keys)
    f = frames
    if used_col is not None and used_col in f.columns:
        f = f[f[used_col].astype(bool)]
    f = f.assign(_col=f["frame"].astype(int) % 3)

    def per_group_mad(g_keys):
        g = f.groupby(g_keys)
        n = g.size()
        med = g[traits].transform("median")
        dev = (f[traits] - med).abs()
        m = dev.groupby([f[k] for k in g_keys]).median()
        return m[n.reindex(m.index) >= min_frames], n

    run_mad, n_run = per_group_mad(keys)
    col_mad, _ = per_group_mad(keys + ["_col"])
    rows = []
    for t in traits:
        rows.append(
            {
                "trait": t,
                "noise": float(run_mad[t].median()) if len(run_mad) else np.nan,
                "noise_within_column": float(col_mad[t].median()) if len(col_mad) else np.nan,
                "n_runs": int(run_mad[t].notna().sum()),
                "frames_per_run": float(n_run[n_run >= min_frames].median())
                if (n_run >= min_frames).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def noise_from_run_mad(runs: pd.DataFrame, traits: Iterable[str]) -> pd.DataFrame:
    """Fallback for E1 when no frame table exists: median over runs of ``<trait>_mad``."""
    rows = []
    for t in traits:
        col = f"{t}_mad"
        v = runs[col].astype(float) if col in runs.columns else pd.Series(dtype=float)
        rows.append(
            {
                "trait": t,
                "noise": float(v.median()) if v.notna().any() else np.nan,
                "noise_within_column": np.nan,
                "n_runs": int(v.notna().sum()),
                "frames_per_run": float(runs["n_frames_ok"].median())
                if "n_frames_ok" in runs.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- turgor
def rmcorr(df: pd.DataFrame, x: str, y: str, subject: str) -> dict:
    """Repeated-measures correlation (Bakdash & Marusich 2017): the common within-subject
    Pearson correlation, from x and y centred on each subject's mean. dof = N - k - 1."""
    d = df[[x, y, subject]].dropna()
    d = d[d.groupby(subject)[x].transform("size") >= 2]
    k = d[subject].nunique()
    n = len(d)
    dof = n - k - 1
    nan = {"r": np.nan, "p_value": np.nan, "dof": dof, "n": n, "n_subjects": k}
    if dof < 2:
        return nan
    xc = d[x] - d.groupby(subject)[x].transform("mean")
    yc = d[y] - d.groupby(subject)[y].transform("mean")
    den = math.sqrt(float((xc**2).sum()) * float((yc**2).sum()))
    if den == 0:
        return nan
    r = float((xc * yc).sum()) / den
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt(dof / (1 - r * r))
        p = float(2 * sps.t.sf(abs(t), dof))
    return {"r": r, "p_value": p, "dof": dof, "n": n, "n_subjects": k}


def value_near(
    series_times: pd.Series,
    series_values: pd.Series,
    query_times: pd.Series,
    window_min: float = 15.0,
) -> np.ndarray:
    """Median of ``series_values`` within +-``window_min`` minutes of each query time."""

    def ns(s) -> np.ndarray:  # UTC nanoseconds since the epoch, whatever the stored unit
        s = pd.Series(pd.to_datetime(s, utc=True))
        return s.dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)

    t = ns(series_times)
    order = np.argsort(t, kind="stable")
    t = t[order]
    v = np.asarray(series_values, dtype=float)[order]
    q = ns(query_times)
    w = int(window_min * 60e9)
    lo = np.searchsorted(t, q - w, side="left")
    hi = np.searchsorted(t, q + w, side="right")
    out = np.full(len(q), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        if b > a:
            out[i] = np.nanmedian(v[a:b])
    return out


def detrend_by(df: pd.DataFrame, cols: Sequence[str], group: str, time_col: str) -> pd.DataFrame:
    """Residuals of each column after a least-squares line against time, fitted per group."""
    out = df.copy()
    t = pd.to_datetime(out[time_col], utc=True)
    days = (t - t.min()).dt.total_seconds().to_numpy() / 86400.0
    for _, idx in out.groupby(group).groups.items():
        pos = out.index.get_indexer(idx)
        x = days[pos]
        for c in cols:
            y = out[c].to_numpy(dtype=float)[pos]
            ok = np.isfinite(y)
            if ok.sum() >= 3 and np.ptp(x[ok]) > 0:
                k, b = np.polyfit(x[ok], y[ok], 1)
                out.loc[idx, c] = y - (k * x + b)
    return out


def turgor_per_run(
    runs: pd.DataFrame, turgor: pd.DataFrame, lag_days: int = 0, window_min: float = 15.0
) -> pd.Series:
    """Turgor value for each run: median of the plant's leaf-clip readings within
    +-``window_min`` of ``timestamp_utc - lag_days``. With a positive lag the turgor reading
    comes before the trait (turgor leads). NaN for plants without a clip."""
    runs = add_plant_id(runs)
    out = pd.Series(np.nan, index=runs.index, name="turgor_value")
    for pid, g in runs.groupby("plant_id"):
        s = turgor[turgor["plant_id"] == pid]
        if s.empty:
            continue
        q = pd.to_datetime(g["timestamp_utc"], utc=True) - pd.Timedelta(days=lag_days)
        out.loc[g.index] = value_near(s["utc_timestamp"], s["turgor_value"], q, window_min)
    return out


def turgor_corr(
    runs: pd.DataFrame,
    turgor: pd.DataFrame,
    traits: Iterable[str],
    lags: Sequence[int] = (-2, -1, 0, 1, 2),
    window_min: float = 15.0,
    pooled_beds: Sequence[str] = ("W2",),
    min_n: int = 5,
    detrend: bool = False,
) -> pd.DataFrame:
    """Trait against leaf turgor (plan E5).

    Per plant with a clip: Spearman correlation of each trait with the turgor reading at each
    lag (``turgor_per_run``). Pooled over the plants of each bed in ``pooled_beds``: the
    repeated-measures correlation (``rmcorr``) on the same pairs. The turgor value rises when
    turgor falls, so a negative rho means the trait falls as the leaf loses turgor. With
    ``detrend`` each plant's linear time trend is removed from the trait and the turgor value
    before correlating (``detrend_by``).

    Columns: plant (``<bed>_<plant>`` or ``pooled_<bed>``), bed, trait, lag_days, rho, p_value,
    n, method."""
    traits = list(traits)
    runs = add_plant_id(runs)
    rows = []
    for lag in lags:
        tv = turgor_per_run(runs, turgor, lag, window_min)
        d = runs.assign(_turgor=tv)
        d = d[d["_turgor"].notna()]
        if detrend:
            d = detrend_by(d, traits + ["_turgor"], "plant_id", "timestamp_utc")
        for pid, g in d.groupby("plant_id"):
            for t in traits:
                gg = g[[t, "_turgor"]].dropna()
                rho = p = np.nan
                if len(gg) >= min_n and gg[t].nunique() > 1 and gg["_turgor"].nunique() > 1:
                    r = sps.spearmanr(gg[t], gg["_turgor"])
                    rho, p = float(r.statistic), float(r.pvalue)
                rows.append(
                    {
                        "plant": pid,
                        "bed": str(g["bed"].iloc[0]),
                        "trait": t,
                        "lag_days": lag,
                        "rho": rho,
                        "p_value": p,
                        "n": len(gg),
                        "method": "spearman",
                    }
                )
        for bed in pooled_beds:
            db = d[d["bed"] == bed]
            if db["plant_id"].nunique() < 2:
                continue
            for t in traits:
                r = rmcorr(db, t, "_turgor", "plant_id")
                rows.append(
                    {
                        "plant": f"pooled_{bed}",
                        "bed": bed,
                        "trait": t,
                        "lag_days": lag,
                        "rho": r["r"],
                        "p_value": r["p_value"],
                        "n": r["n"],
                        "method": "rmcorr",
                    }
                )
    cols = ["plant", "bed", "trait", "lag_days", "rho", "p_value", "n", "method"]
    return pd.DataFrame(rows, columns=cols)
