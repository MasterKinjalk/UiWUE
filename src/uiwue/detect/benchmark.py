"""Sensor benchmark: which sensors, alone or combined, detect deficit earliest.

Rules (Master Plan 6.7): leave-one-plant-out; no date/bed/hour/position inputs;
features are baseline deltas; all fitting inside the fold; a size-only model and
a placebo run beside every result.

Rows are plant-days (daily value = median of the 12 and 15 h runs). Training
rows are days 0-12; label = plant is in a deficit bed. The held-out plant's
out-of-fold probability is then scored per day with the shared AUC.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uiwue.detect.data import SENSORS, SETUP_LABEL, cost_rank, setup_of, size_baseline_column, trait_columns
from uiwue.stats import GROUPS, auc_ci, baseline_delta, daily_values, onset_day
from uiwue.stats._design import CONTROL_BEDS, DEFICIT_BEDS, PLACEBO, prefix_of
from uiwue.stats.core import EARLY_DAYS, SEED

TRAIN_DAYS = (0, 12)
FULL_DAYS = (1, 12)
MODELS = ("logistic", "boosted")


def make_model(kind: str):
    if kind == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        )
    if kind == "boosted":
        return HistGradientBoostingClassifier(
            max_depth=2,
            max_iter=100,
            learning_rate=0.1,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED,
        )
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def daily_features(df: pd.DataFrame, columns: list[str], hours=(12, 15)) -> pd.DataFrame:
    """Wide plant-day table of baseline-delta daily values: bed, plant, date, day, <cols>."""
    out = None
    for c in columns:
        d = daily_values(df, c, hours=hours).rename(columns={"value": c})
        d = d.drop(columns="treatment")
        out = d if out is None else out.merge(d, on=["bed", "plant", "date", "days_since_cutoff"], how="outer")
    if out is None:
        return pd.DataFrame(columns=["bed", "plant", "date", "days_since_cutoff"])
    out["y"] = out["bed"].isin(DEFICIT_BEDS).astype(int)
    out["pid"] = out["bed"] + "_" + out["plant"]
    return out.sort_values(["pid", "date"]).reset_index(drop=True)


LOW_RES_THERMAL = ("thermal_r160", "thermal_r80")


def sensor_sets(columns: list[str]) -> dict[tuple[str, ...], list[str]]:
    """Every non-empty combination of the sensors present -> its feature columns.

    Each combination with thermal is repeated with a reduced-resolution thermal
    camera in its place (the cheaper-setup question), when those columns exist.
    """
    by: dict[str, list[str]] = {}
    for c in columns:
        by.setdefault(setup_of(c), []).append(c)
    present = [s for s in SENSORS if by.get(s)]
    out = {}
    for k in range(1, len(present) + 1):
        for combo in combinations(present, k):
            out[combo] = [c for s in combo for c in by[s]]
            if "thermal" in combo:
                for low in LOW_RES_THERMAL:
                    if by.get(low):
                        alt = tuple(low if s == "thermal" else s for s in combo)
                        out[alt] = [c for s in alt for c in by[s]]
    return out


def full_resolution(columns: list[str]) -> list[str]:
    return [c for c in columns if setup_of(c) not in LOW_RES_THERMAL]


# --------------------------------------------------------------------------- #
# leave-one-plant-out
# --------------------------------------------------------------------------- #
def lopo_predict(train: pd.DataFrame, feats: list[str], kind: str, predict: pd.DataFrame | None = None) -> np.ndarray:
    """Out-of-fold P(deficit) for each row of `predict` (default: `train`).

    The model for a plant never saw any row of that plant.
    """
    predict = train if predict is None else predict
    prob = np.full(len(predict), np.nan)
    has = train[feats].notna().any(axis=1)
    tr = train[has]
    if tr["y"].nunique() < 2:
        return prob
    for pid in predict["pid"].unique():
        fit = tr[tr["pid"] != pid]
        if fit["y"].nunique() < 2:
            continue
        rows = (predict["pid"] == pid).to_numpy() & predict[feats].notna().any(axis=1).to_numpy()
        if not rows.any():
            continue
        m = make_model(kind).fit(fit[feats].to_numpy(float), fit["y"].to_numpy())
        prob[rows] = m.predict_proba(predict.loc[rows, feats].to_numpy(float))[:, 1]
    return prob


def _per_plant_auc(rows: pd.DataFrame, group: str, n_boot: int, days=None):
    r = rows if days is None else rows[rows["days_since_cutoff"].between(*days)]
    pp = r.dropna(subset=["prob"]).groupby(["bed", "pid"], as_index=False)["prob"].mean()
    pos = pp.loc[pp["bed"].isin(GROUPS[group]), "prob"].to_numpy()
    neg = pp.loc[pp["bed"].isin(CONTROL_BEDS), "prob"].to_numpy()
    return auc_ci(pos, neg, n_boot=n_boot, rng=np.random.default_rng(SEED)), len(pos), len(neg)


def _curve(rows: pd.DataFrame, group: str, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for (date, day), sub in rows.dropna(subset=["prob"]).groupby(["date", "days_since_cutoff"]):
        pos = sub.loc[sub["bed"].isin(GROUPS[group]), "prob"].to_numpy()
        neg = sub.loc[sub["bed"].isin(CONTROL_BEDS), "prob"].to_numpy()
        auc, lo, hi = auc_ci(pos, neg, n_boot=n_boot, rng=rng)
        out.append(
            dict(date=date, day=int(day), auc=auc, ci_low=lo, ci_high=hi, n_deficit=len(pos), n_control=len(neg))
        )
    return pd.DataFrame(out)


def evaluate(daily: pd.DataFrame, feats: list[str], kind: str, groups, n_boot: int):
    """Detection results for one feature set and model, plus its placebo."""
    stress = daily[daily["days_since_cutoff"].between(*TRAIN_DAYS)].copy()
    stress["prob"] = lopo_predict(stress, feats, kind)
    pre = daily[daily["date"].between(*PLACEBO)].copy()
    pre["prob"] = lopo_predict(pre, feats, kind)

    summary, curves = [], []
    for g in groups:
        if not daily["bed"].isin(GROUPS[g]).any():
            continue
        curve = _curve(stress, g, n_boot)
        (e, elo, ehi), npos, nneg = _per_plant_auc(stress, g, n_boot, EARLY_DAYS)
        (f, _, _), _, _ = _per_plant_auc(stress, g, 0, FULL_DAYS)
        (p, plo, phi), _, _ = _per_plant_auc(pre, g, n_boot)
        summary.append(
            dict(
                group=g,
                early_auc=e,
                early_ci_low=elo,
                early_ci_high=ehi,
                full_auc=f,
                onset_day=onset_day(curve) if len(curve) else np.nan,
                placebo_auc=p,
                placebo_ci_low=plo,
                placebo_ci_high=phi,
                bed_effect=bool(np.isfinite(plo) and (plo > 0.5 or phi < 0.5) and max(p, 1 - p) >= 0.7),
                n_deficit=npos,
                n_control=nneg,
            )
        )
        curves.append(curve.assign(group=g))
    return summary, curves


def run_benchmark(
    df: pd.DataFrame, models=MODELS, groups=("all", "W2", "W3"), n_boot: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every sensor combination x model, plus the size-only baseline.

    Returns (benchmark, curves). `curves` uses the auc_by_day layout with
    kind="model" and signal="model:<sensors>:<model>".
    """
    cols = trait_columns(df)
    size_col = size_baseline_column(df)
    feat_cols = sorted(set(cols) | ({size_col} if size_col else set()))
    daily = daily_features(df, feat_cols)
    sets = {("size",): [size_col]} if size_col else {}
    sets.update(sensor_sets(cols))

    rows, curves = [], []
    for combo, feats in sets.items():
        name = "+".join(combo)
        for kind in models:
            summary, cs = evaluate(daily, feats, kind, groups, n_boot)
            for s in summary:
                rows.append(
                    dict(
                        sensors=name,
                        setup=("Plant size only" if combo == ("size",) else " + ".join(SETUP_LABEL[s] for s in combo)),
                        model=kind,
                        n_sensors=0 if combo == ("size",) else len(combo),
                        n_features=len(feats),
                        cost_rank=0 if combo == ("size",) else cost_rank(combo),
                        is_size_baseline=combo == ("size",),
                        size_column=size_col,
                        **s,
                    )
                )
            for c in cs:
                curves.append(c.assign(signal=f"model:{name}:{kind}", kind="model", column=f"model:{name}:{kind}"))
    bench = pd.DataFrame(rows)
    if not bench.empty:
        bench = bench.sort_values(["group", "cost_rank", "model"]).reset_index(drop=True)
    return bench, (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def cheapest_close(bench: pd.DataFrame, group: str = "all", margin: float = 0.05) -> pd.Series | None:
    """Cheapest setup whose early AUC is within `margin` of the best one.

    Setups whose placebo separated the beds are not eligible (rule 6.7.4).
    """
    b = bench[(bench["group"] == group) & ~bench["is_size_baseline"] & ~bench["bed_effect"]]
    b = b.dropna(subset=["early_auc"])
    if b.empty:
        return None
    best = b["early_auc"].max()
    ok = b[b["early_auc"] >= best - margin]
    return ok.sort_values(["cost_rank", "early_auc"], ascending=[True, False]).iloc[0]


# --------------------------------------------------------------------------- #
# stress probability per run (handoff to Sam's Plant profile)
# --------------------------------------------------------------------------- #
def stress_scores(df: pd.DataFrame, kind: str = "logistic") -> pd.DataFrame:
    """detect_stress_prob for every run, out of fold, from all sensors present
    (full resolution only).

    The model is trained on plant-day rows (days 0-12) of the other plants and
    applied to each run of the held-out plant, on run-level baseline deltas.
    """
    cols = full_resolution(trait_columns(df))
    if not cols:
        return pd.DataFrame()
    daily = daily_features(df, cols)
    train = daily[daily["days_since_cutoff"].between(*TRAIN_DAYS)]
    runs = df[["bed", "plant", "date", "hour"]].copy()
    for c in cols:
        runs[c] = baseline_delta(df, c).to_numpy()
    runs["pid"] = runs["bed"] + "_" + runs["plant"]
    runs["detect_stress_prob"] = lopo_predict(train, cols, kind, predict=runs)
    out = runs[["bed", "plant", "date", "hour", "detect_stress_prob"]].copy()
    out["detect_model_sensors"] = "+".join(s for s in SENSORS if any(prefix_of(c) == s for c in cols))
    out["date"] = out["date"].dt.date
    return out
