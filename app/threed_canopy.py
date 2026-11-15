"""Data and geometry for the 3D canopy page (``/threed``): one table, three traits, real clouds only.

Source: the committed 3D trait table ``derived/threed/traits_runs.parquet``, read through
``uiwue.core.read_table`` like every other page. The analysis is W1 (control) against W2 (deficit) only
(``BEDS``): every bed-level number, band and count uses those two beds. W3 (a second deficit bed whose
plants each lost a leaf to the relative-water-content sampling) is read too, so a W3 plant can be shown on
its own, but it is never pooled: ``runs``, ``plant_days`` and ``windows`` return W1 and W2 unless asked for
``SHOWN_BEDS``. Onsets and the placebo come from ``derived/detect/`` (no second onset rule), the E2 bed
check from ``derived/threed/results/e2_bed_baseline.csv``. Clouds are the local cache
``data/derived/threed/clouds/<bed>_<plant>/<YYYY-MM-DD>_<HH>.npz`` (not in git): when the folder or a
file is missing the answer is None, never a made-up cloud.

Pure data: pandas and numpy in, plain frames, arrays and dicts out; no Dash, no colours. Results are
cached by file and modification time (``table_key``), so a page callback costs a lookup once a file has
been read, and a rewritten table is picked up on the next call. Returned frames are copy-on-write views of
the cache; returned arrays are read-only.

Conventions (plan/Master-Plan.md section 6, design/DESIGN.md section 7):

- ``day`` = days since irrigation off (day 0 = 3 Sep 2025); re-watered on day 12 at 18:35.
- plant-day value = mean of the plant's QC-pass runs at the 9, 12, 15 and 18 h slots.
- baseline = median of the plant's plant-day values on days −4 to −1 (30 Aug – 2 Sep,
  ``uiwue.core.BASELINE``): the app's convention (DESIGN 7.2, ``profile_views.with_change``). Change is
  % of that baseline. A plant without a baseline keeps its rows, with NaN change.
- intervals: 2,000 resamples of plants (``uiwue.stats`` seed), so every call gives the same band.
- the run drawn for a day (``cloud_days``): of the day's QC-pass daytime runs with a drawable cloud, the one
  closest to the plant's own level around that day (per-trait median of its QC-pass runs on the day and the
  days either side), so one odd run (a neighbour's leaf in the crop) is not the picture. The page shows that
  run's own values beside it; THEN (``then_run``) is the baseline-window run closest to the baseline.
"""

from __future__ import annotations

import base64
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

try:
    from uiwue import core
except ModuleNotFoundError:  # a fresh clone without `pip install -e .`: use the package from src/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from uiwue import core
from uiwue.stats import NULL_DAYS
from uiwue.stats.core import N_BOOT, SEED
from uiwue.threed.schema import QC_FLAGS

__all__ = [
    "TRAITS",
    "Trait",
    "axis_ranges",
    "bed_effects",
    "clear_caches",
    "cloud_days",
    "default_candidates",
    "default_pair",
    "default_plant",
    "gap",
    "headline",
    "now_day",
    "onsets",
    "onsets_info",
    "plant_cloud",
    "plant_days",
    "rgb_frame",
    "run_values",
    "runs",
    "sampled_leaf",
    "side_profile",
    "stray_mask",
    "structure_alerts",
    "structure_status",
    "table_key",
    "then_run",
    "windows",
]


# ---------------------------------------------------------------- traits and constants
@dataclass(frozen=True)
class Trait:
    """One trait of the page: its short key, the contract column, the display label and the raw unit."""

    key: str
    column: str
    label: str
    unit: str


TRAITS = {  # the only three traits on the page (brief C3); leaf area leads, height carries the bed check
    "leaf_area_est": Trait("leaf_area_est", "depth_leaf_area_est", "Leaf area", "cm²"),
    "height_p75": Trait("height_p75", "depth_height_p75", "Height", "mm"),
    "hull_volume": Trait("hull_volume", "depth_hull_volume", "Volume", "cm³"),
}
CONTROL_BED, DEFICIT_BED, SAMPLED_BED = "W1", "W2", "W3"
BEDS = (CONTROL_BED, DEFICIT_BED)  # the analysis: every bed-level number compares these two
SHOWN_BEDS = (CONTROL_BED, DEFICIT_BED, SAMPLED_BED)  # the picker: W3 plants one by one, never pooled
GROUPS = {"control": CONTROL_BED, "deficit": DEFICIT_BED}
DAY_SLOTS = (9, 12, 15, 18)  # daytime slots averaged into one plant-day value (06 h runs are dark)
CLOUD_SLOTS = (12, 15, 9, 18)  # tie-break between equally typical runs: the contract's daily slots first (6.5)
BASELINE_DAYS = ((core.BASELINE[0] - core.CUTOFF).days, (core.BASELINE[1] - core.CUTOFF).days)  # (−4, −1)
STRESS_DAYS = (0, (core.REWATER - core.CUTOFF).days)  # (0, 12): no water
PEAK_DAYS, TROUGH_DAYS, REWATERED_DAYS = (1, 3), (10, 12), (15, 16)
WINDOWS = {"peak": PEAK_DAYS, "trough": TROUGH_DAYS, "rewatered": REWATERED_DAYS}
STRUCTURE_THRESHOLD = -20.0  # % leaf area vs own baseline, two usable days in a row (brief E)
MIN_PLANT_POINTS = 100  # a cloud with fewer plant points is not drawn (broken segmentation)
MIN_POINT_SHARE = 0.5  # ... nor one with under half the plant points of its run (median over frames)
CYLINDER_MM = 85.0  # radius of the plant cylinder the pipeline crops to
PROFILE_BIN_MM = 5.0  # x-bin width of the side elevation
MAX_POINTS = 20_000  # plant clouds above this are subsampled (seeded); real plant clouds stay below 10k
THUMB_WIDTH = 480  # px, width of the RGB frame data URI
FLAG_COLUMNS = tuple(QC_FLAGS)  # qc_no_plant, qc_few_points, qc_high_spread, qc_overlap, ...
RUN_COLUMNS = ["bed", "plant", "treatment", "date", "day", "slot", "capture_hour", "qc_pass", *FLAG_COLUMNS]
SENSOR_ORDER = ("Thermal", "RGB", "Depth")
# detached edge points (display only, ``stray_mask``): checked by image on tall controls and sparse wilted plants
STRAY_LINK_MM = 12.0  # plant points closer than this are one cluster (single linkage; about 4 point spacings)
STRAY_WALL_MM = 3.0  # a cluster enters through the crop wall when a point lies this close to the 85 mm cylinder
STRAY_MAX_SHARE = 0.10  # ... holds at most this share of the plant points (a real top leaf is larger)
STRAY_GAP_MM = 15.0  # ... floats above every point kept, and stays at least this far from all of them
STRAY_SPECK_SHARE = 0.01  # a floating cluster this small is a speck, hidden wherever it floats
NEAR_DAYS = 1  # the drawn run is the one closest to the plant's median over the day and this many days either side


def clouds_dir() -> Path:
    """``data/derived/threed/clouds``, read from ``uiwue.core.DATA`` at call time (UIWUE_DATA moves it)."""
    return Path(core.DATA) / "derived" / "threed" / "clouds"


def rgb_dir() -> Path:
    return Path(core.DATA) / "rgb"


def clear_caches() -> None:
    """Forget every cached table, cloud and figure input (tests, or after replacing files in place)."""
    for fn in (
        _table,
        _plant_days,
        _gap,
        _windows,
        _headline,
        _structure,
        _alerts,
        _onsets,
        _bed_effects,
        _read_parquet,
        _read_csv,
        _cloud_file,
        _cloud_stats,
        _cloud_days,
        _then_run,
        _default_candidates,
        _sampled,
        _thumbnail,
    ):
        fn.cache_clear()


def _stamp(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def table_key() -> tuple[str, int] | None:
    """(path, mtime) of the committed 3D table, or None without it: the cache key of everything derived from it.
    Callers that cache their own results on the table (the page's plant list) key them on this."""
    path = core.table_path("threed", "traits_runs")
    stamp = _stamp(path)
    return None if stamp is None else (str(path), stamp)


_key = table_key  # the private name the module's own caches use


def _view(df: pd.DataFrame | None) -> pd.DataFrame | None:
    return None if df is None else df.copy(deep=False)


def _in_beds(df: pd.DataFrame | None, beds) -> pd.DataFrame | None:
    """Rows of the asked beds (a new frame: changing it leaves the cache alone)."""
    return None if df is None else df[df["bed"].isin(tuple(beds))].reset_index(drop=True)


def _frozen(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


# ---------------------------------------------------------------- 1. runs
def runs(beds=BEDS) -> pd.DataFrame | None:
    """Runs of the committed 3D table, one row per bed, plant, date and slot: W1 and W2 unless ``beds`` asks for W3.

    Columns: bed, plant, treatment ('control' / 'deficit'), date (datetime64, midnight), day (days since
    3 Sep 2025), slot (scheduled slot 6/9/12/15/18 h: the table's ``hour`` key), capture_hour (folder hour
    of the run; it names the cloud and RGB files), qc_pass, the eight QC flags (``FLAG_COLUMNS``) and the
    three traits under their short keys: leaf_area_est (cm²), height_p75 (mm), hull_volume (cm³). QC-failed
    runs keep their trait values (the table has them); every aggregate below uses QC-pass runs only.
    None when the table is missing.
    """
    key = _key()
    t = None if key is None else _table(*key)
    return None if t is None else _in_beds(t[RUN_COLUMNS + list(TRAITS)], beds)


@lru_cache(maxsize=2)
def _table(path: str, stamp: int) -> pd.DataFrame | None:
    try:
        df = core.read_table("threed", "traits_runs")
    except Exception:  # noqa: BLE001 - a half-written table must not take the page down; next stamp retries
        return None
    if df is None or df.empty or "depth_qc_pass" not in df:
        return None
    df = core.add_design(df[df["bed"].isin(SHOWN_BEDS)])
    out = pd.DataFrame(
        {
            "bed": df["bed"].astype(str),
            "plant": df["plant"].astype(str),
            "treatment": df["treatment"].astype(str),
            "date": df["date"],
            "day": df["days_since_cutoff"].astype("int64"),
            "slot": df["hour"].astype("int64"),
            "capture_hour": df.get("depth_capture_hour", df["hour"]).fillna(df["hour"]).astype("int64"),
            "qc_pass": df["depth_qc_pass"].fillna(False).astype(bool),
        }
    )
    for f in FLAG_COLUMNS:
        out[f] = df[f"depth_{f}"].fillna(False).astype(bool) if f"depth_{f}" in df else False
    for k, t in TRAITS.items():
        out[k] = pd.to_numeric(df[t.column], errors="coerce") if t.column in df else np.nan
    out["n_points"] = pd.to_numeric(df.get("depth_n_points", np.nan), errors="coerce")  # cloud check only
    return out.sort_values(["bed", "plant", "date", "slot"], ignore_index=True)


# ---------------------------------------------------------------- 2. plant-day table
def plant_days(beds=BEDS) -> pd.DataFrame | None:
    """One row per plant and day: the mean of its QC-pass daytime runs and each trait's change from baseline.

    Columns: bed, plant, treatment, date, day, n_runs (runs in the table at 9–18 h), n_pass (of which passed
    QC), slots (the slots averaged, e.g. '9,12,15'), and for each trait key t: t (plant-day mean, trait unit),
    t_base (the plant's baseline, trait unit), t_pct (% change from baseline). A day whose daytime runs all
    failed QC keeps its row with n_pass 0 and NaN values (a "no usable scan" day); a plant without a QC-pass
    daytime run on days −4 to −1 keeps its rows with NaN t_base and t_pct. W1 and W2 unless ``beds`` asks for
    W3 (each plant has its own baseline, so a W3 row never mixes with the others). None without the table.
    """
    key = _key()
    return None if key is None else _in_beds(_plant_days(*key), beds)


@lru_cache(maxsize=2)
def _plant_days(path: str, stamp: int) -> pd.DataFrame | None:
    t = _table(path, stamp)
    if t is None:
        return None
    keys = ["bed", "plant", "treatment", "date", "day"]
    d = t[t["slot"].isin(DAY_SLOTS)]
    ok = d[d["qc_pass"]]
    out = d.groupby(keys).size().rename("n_runs").to_frame()
    out = out.join(ok.groupby(keys).size().rename("n_pass"))
    slots = ok.sort_values("slot").groupby(keys)["slot"].agg(lambda s: ",".join(str(int(x)) for x in s))
    out = out.join(slots.rename("slots")).join(ok.groupby(keys)[list(TRAITS)].mean()).reset_index()
    out["n_pass"] = out["n_pass"].fillna(0).astype("int64")
    out["slots"] = out["slots"].fillna("")
    in_base = out["day"].between(*BASELINE_DAYS)
    base = out[in_base].groupby(["bed", "plant"])[list(TRAITS)].median()
    out = out.join(base.add_suffix("_base"), on=["bed", "plant"])
    for k in TRAITS:
        b = out[f"{k}_base"].where(out[f"{k}_base"] > 0)
        out[f"{k}_pct"] = (out[k] / b - 1.0) * 100.0
    cols = keys + ["n_runs", "n_pass", "slots"] + [c for k in TRAITS for c in (k, f"{k}_base", f"{k}_pct")]
    return out[cols].sort_values(["bed", "plant", "day"], ignore_index=True)


# ---------------------------------------------------------------- 3. gap series
def _ci_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95 % percentile interval of the mean over plants, resampling plants (NaN below two plants)."""
    if len(values) < 2:
        return np.nan, np.nan
    m = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def _ci_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95 % interval of mean(a) − mean(b), resampling plants within each group (NaN below two per group)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return float(lo), float(hi)


def _rng(*parts: int) -> np.random.Generator:
    """A generator seeded from the shared seed and the item (day, trait index ...), so each band is fixed."""
    return np.random.default_rng([SEED, *(int(p) + 1000 for p in parts)])


def gap(trait: str) -> dict | None:
    """Deficit minus control in % of each plant's own baseline, day by day, with a plant-bootstrap 95 % CI.

    Returns {"trait", "by_day", "null_range", "null_days", "n_boot", "seed"}:
    - by_day: one row per day: day, date, gap (percentage points: mean deficit % − mean control %), ci_low,
      ci_high (2,000 resamples of plants within each group, seeded per day; NaN when a group has fewer than
      two plants that day), deficit and control (group means, %), n_deficit and n_control (plants with a
      value that day).
    - null_range: (lowest, highest) daily gap over the watered days −10 to 0 (``uiwue.stats.NULL_DAYS``):
      how far apart the beds sat before any plant went without water. Draw it as a horizontal band.
    None when the table is missing; KeyError for a trait outside ``TRAITS``.
    """
    if trait not in TRAITS:
        raise KeyError(f"{trait!r} is not one of {list(TRAITS)}")
    key = _key()
    g = None if key is None else _gap(*key, trait)
    return None if g is None else {**g, "by_day": g["by_day"].copy(deep=False)}


@lru_cache(maxsize=8)
def _gap(path: str, stamp: int, trait: str) -> dict | None:
    pdays = _plant_days(path, stamp)
    if pdays is None:
        return None
    col, rows = f"{trait}_pct", []
    for (day, date), g in pdays.groupby(["day", "date"], sort=True):
        a = g.loc[g["bed"] == DEFICIT_BED, col].dropna().to_numpy(float)
        b = g.loc[g["bed"] == CONTROL_BED, col].dropna().to_numpy(float)
        ma = float(a.mean()) if len(a) else np.nan
        mb = float(b.mean()) if len(b) else np.nan
        lo, hi = _ci_diff(a, b, _rng(list(TRAITS).index(trait), day))
        rows.append(
            dict(
                day=int(day),
                date=date,
                gap=ma - mb,
                ci_low=lo,
                ci_high=hi,
                deficit=ma,
                control=mb,
                n_deficit=len(a),
                n_control=len(b),
            )
        )
    by_day = pd.DataFrame(
        rows, columns=["day", "date", "gap", "ci_low", "ci_high", "deficit", "control", "n_deficit", "n_control"]
    )
    pre = by_day.loc[by_day["day"].between(*NULL_DAYS), "gap"].dropna()
    null = (float(pre.min()), float(pre.max())) if len(pre) else (np.nan, np.nan)
    return dict(trait=trait, by_day=by_day, null_range=null, null_days=tuple(NULL_DAYS), n_boot=N_BOOT, seed=SEED)


# ---------------------------------------------------------------- 4. collapse and recovery
def windows(beds=BEDS) -> pd.DataFrame | None:
    """Collapse and recovery per plant and trait: one row per plant × trait (W1 and W2 unless ``beds`` asks for W3).

    Columns: bed, plant, treatment, trait; peak, trough, rewatered (mean % change from baseline over days
    1–3, 10–12 and 15–16; NaN without a baseline or without a usable day in the window); n_peak, n_trough,
    n_rewatered (usable days averaged); peak_to_trough and peak_to_rewatered (% change of the trough and of
    the re-watered window against the plant's own peak window: the paired loss and what came back. Ratios
    of the raw window means, so they need no baseline and every plant with both windows counts).
    """
    key = _key()
    return None if key is None else _in_beds(_windows(*key), beds)


@lru_cache(maxsize=2)
def _windows(path: str, stamp: int) -> pd.DataFrame | None:
    pdays = _plant_days(path, stamp)
    if pdays is None:
        return None
    plants = pdays[["bed", "plant", "treatment"]].drop_duplicates().set_index(["bed", "plant"])
    parts = []
    for k in TRAITS:
        w = plants.copy()
        for name, (lo, hi) in WINDOWS.items():
            sel = pdays[pdays["day"].between(lo, hi)].dropna(subset=[k])
            g = sel.groupby(["bed", "plant"])
            w[f"_{name}_raw"] = g[k].mean()
            w[name] = g[f"{k}_pct"].mean()
            w[f"n_{name}"] = g[k].size()
        peak = w["_peak_raw"].where(w["_peak_raw"] > 0)
        w["peak_to_trough"] = (w["_trough_raw"] / peak - 1.0) * 100.0
        w["peak_to_rewatered"] = (w["_rewatered_raw"] / peak - 1.0) * 100.0
        w = w.drop(columns=[c for c in w.columns if c.startswith("_")]).assign(trait=k)
        parts.append(w.reset_index())
    out = pd.concat(parts, ignore_index=True)
    for name in WINDOWS:
        out[f"n_{name}"] = out[f"n_{name}"].fillna(0).astype("int64")
    cols = ["bed", "plant", "treatment", "trait", "peak", "trough", "rewatered", "n_peak", "n_trough", "n_rewatered"]
    return out[cols + ["peak_to_trough", "peak_to_rewatered"]]


def headline() -> dict | None:
    """Bed-level numbers for the stat tiles: per trait, the paired loss and the recovery, per group.

    {trait: {"peak_to_trough": {"deficit": s, "control": s}, "peak_to_rewatered": {"deficit": s, "control": s}}}
    with s = {"mean", "ci_low", "ci_high" (% change vs the plant's own peak window; mean over plants, 95 %
    CI from 2,000 plant resamples, seeded), "n" (plants), "min", "max" (plant range, %)}. Height lost
    peak→trough is ``headline()["height_p75"]["peak_to_trough"]["deficit"]["mean"]``. None without the table.
    """
    key = _key()
    h = None if key is None else _headline(*key)
    return (
        None if h is None else {t: {m: {g: dict(s) for g, s in v.items()} for m, v in d.items()} for t, d in h.items()}
    )


@lru_cache(maxsize=2)
def _headline(path: str, stamp: int) -> dict | None:
    w = _windows(path, stamp)
    if w is None:
        return None
    out = {}
    for i, k in enumerate(TRAITS):
        out[k] = {}
        for j, metric in enumerate(("peak_to_trough", "peak_to_rewatered")):
            out[k][metric] = {}
            for n, (group, bed) in enumerate(GROUPS.items()):
                v = w.loc[(w["trait"] == k) & (w["bed"] == bed), metric].dropna().to_numpy(float)
                lo, hi = _ci_mean(v, _rng(i, j, n))
                out[k][metric][group] = dict(
                    mean=float(v.mean()) if len(v) else np.nan,
                    ci_low=lo,
                    ci_high=hi,
                    n=len(v),
                    min=float(v.min()) if len(v) else np.nan,
                    max=float(v.max()) if len(v) else np.nan,
                )
    return out


# ---------------------------------------------------------------- 5. onsets and bed checks
def _file_key(path: Path) -> tuple[str, int] | None:
    stamp = _stamp(path)
    return None if stamp is None else (str(path), stamp)


def _detect_key(name: str) -> tuple[str, int] | None:
    return _file_key(Path(core.DERIVED) / "detect" / f"{name}.parquet")


@lru_cache(maxsize=8)
def _read_parquet(path: str, stamp: int) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a half-written file must not take the page down
        return None


@lru_cache(maxsize=4)
def _read_csv(path: str, stamp: int) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def onsets() -> pd.DataFrame | None:
    """Onset days for the onset strip: the three 3D traits beside the first thermal and RGB signal, W2 vs W1.

    Read from ``derived/detect/onsets.parquet`` and ``onsets_by_slot.parquet`` (group 'W2', the deficit bed
    of this page: the detect group 'all' pools W3, whose 3D plants lost a leaf to sampling); no new rule. One
    row per signal, earliest robust onset first:
    signal, label, sensor ('Thermal' / 'RGB' / 'Depth'), trait (short key for the three 3D traits, else
    None), is_size_baseline (projected area, the size-only reference DESIGN 7.6 asks for beside detection
    results), onset_day (contract rule: CI lower bound > 0.70 two scored days in a row), robust_onset_day
    (same rule, bar raised to the largest watered-day AUC: the number to show), slot_min and slot_max (range
    of the robust onset when each daytime slot 9–18 h is scored alone, for a whisker), onset_09, onset_12,
    onset_15, onset_18 (those per-slot robust onsets), bed_effect (placebo), fires_before_cutoff,
    null_auc_max, n_deficit, n_control. Thermal and RGB are the earliest robust onset per sensor that neither
    check flags, picked as Early warning picks them (``uiwue.detect.figures``). None without onsets.parquet.
    """
    key = _detect_key("onsets")
    return None if key is None else _view(_onsets(key, _detect_key("onsets_by_slot")))


@lru_cache(maxsize=2)
def _onsets(onsets_key: tuple[str, int], slots_key: tuple[str, int] | None) -> pd.DataFrame | None:
    o = _read_parquet(*onsets_key)
    if o is None or o.empty:
        return None
    from uiwue.detect.figures import flagged, ranked  # the Early warning selection, not a new rule

    group = DEFICIT_BED
    r = ranked(o, group)
    keep = {t.column: k for k, t in TRAITS.items()}
    picks = [r[r["signal"].isin(keep)]]
    other = r[r["sensor"].isin(["Thermal", "RGB"]) & ~flagged(r) & ~r["is_size_baseline"].astype(bool)]
    picks.append(other.dropna(subset=["robust_onset_day"]).groupby("sensor").head(1))
    picks.append(r[r["is_size_baseline"].astype(bool) & ~r["signal"].isin(keep)])
    out = pd.concat(picks, ignore_index=True).drop_duplicates("signal")
    out["trait"] = pd.Series([keep.get(s) for s in out["signal"]], index=out.index, dtype=object)
    slots = None if slots_key is None else _read_parquet(*slots_key)
    for h in DAY_SLOTS:
        out[f"onset_{h:02d}"] = np.nan
    if slots is not None and not slots.empty:
        s = slots[(slots["group"] == group) & slots["hour"].isin(DAY_SLOTS)]
        wide = s.pivot_table(index="signal", columns="hour", values="robust_onset_day", aggfunc="first")
        for h in DAY_SLOTS:
            if h in wide.columns:
                out[f"onset_{h:02d}"] = out["signal"].map(wide[h])
    per_slot = out[[f"onset_{h:02d}" for h in DAY_SLOTS]]
    out["slot_min"], out["slot_max"] = per_slot.min(axis=1), per_slot.max(axis=1)
    for c in ("bed_effect", "fires_before_cutoff", "is_size_baseline"):
        out[c] = out[c].fillna(False).astype(bool) if c in out else False
    cols = [
        "signal",
        "label",
        "sensor",
        "trait",
        "is_size_baseline",
        "onset_day",
        "robust_onset_day",
        "slot_min",
        "slot_max",
        *[f"onset_{h:02d}" for h in DAY_SLOTS],
        "bed_effect",
        "fires_before_cutoff",
        "null_auc_max",
        "n_deficit",
        "n_control",
    ]
    out = out.reindex(columns=cols)
    out["_rob"] = out["robust_onset_day"].fillna(99)
    out["_s"] = out["sensor"].map({s: i for i, s in enumerate(SENSOR_ORDER)}).fillna(9)
    return out.sort_values(["_rob", "_s", "is_size_baseline"], ignore_index=True).drop(columns=["_rob", "_s"])


def bed_effects() -> dict:
    """Bed checks per trait key, for the ``bed effect`` chip; the numbers are for its hover.

    {trait: {"chip": placebo or e2,
             "placebo", "placebo_auc", "placebo_ci_low", "placebo_ci_high": detect/placebo.parquet, group 'W2'
                 (this page's W2 against W1, as ``onsets``; the app-wide ``components.placebo_chip`` reads the
                 same test for W2 + W3): mean change from baseline over 24–29 Aug, future deficit vs control
                 plants, flagged when the CI excludes 0.5 and AUC ≥ 0.70 either way;
             "e2", "e2_coef", "e2_ci_low", "e2_ci_high", "e2_p", "e2_unit": E2 in
                 derived/threed/results/e2_bed_baseline.csv: day × bed slope difference (W2 − W1, trait unit
                 per day) on 24 Aug – 2 Sep, ``warning`` = p < 0.05}}
    A missing table gives False and NaN for its part.
    """
    e2_key = _file_key(Path(core.DERIVED) / "threed" / "results" / "e2_bed_baseline.csv")
    return {k: dict(v) for k, v in _bed_effects(_detect_key("placebo"), e2_key).items()}


@lru_cache(maxsize=2)
def _bed_effects(placebo_key: tuple[str, int] | None, e2_key: tuple[str, int] | None) -> dict:
    p = None if placebo_key is None else _read_parquet(*placebo_key)
    e2 = None if e2_key is None else _read_csv(*e2_key)
    out = {}
    for k, t in TRAITS.items():
        d = dict(placebo=False, placebo_auc=np.nan, placebo_ci_low=np.nan, placebo_ci_high=np.nan)
        if p is not None and {"signal", "group", "bed_effect"} <= set(p.columns):
            r = p[(p["signal"] == t.column) & (p["group"] == DEFICIT_BED)]
            if len(r):
                r = r.iloc[0]
                d.update(
                    placebo=bool(r["bed_effect"]),
                    placebo_auc=float(r["placebo_auc"]),
                    placebo_ci_low=float(r["placebo_ci_low"]),
                    placebo_ci_high=float(r["placebo_ci_high"]),
                )
        d.update(e2=False, e2_coef=np.nan, e2_ci_low=np.nan, e2_ci_high=np.nan, e2_p=np.nan, e2_unit="")
        if e2 is not None and {"trait", "warning"} <= set(e2.columns):
            r = e2[e2["trait"].astype(str) == k]
            if len(r):
                r = r.iloc[0]
                d.update(
                    e2=str(r["warning"]).strip().lower() in {"true", "1", "yes"},
                    e2_coef=float(r.get("day_bed_coef", np.nan)),
                    e2_ci_low=float(r.get("ci_low", np.nan)),
                    e2_ci_high=float(r.get("ci_high", np.nan)),
                    e2_p=float(r.get("p_value", np.nan)),
                    e2_unit=str(r.get("unit", "")),
                )
        d["chip"] = bool(d["placebo"] or d["e2"])
        out[k] = d
    return out


def midday_dip() -> dict | None:
    """The 3D team's daily-rhythm test (E4), for Daily rhythm: does drought deepen the midday dip (15 h minus 09 h)?

    Read from ``derived/threed/results/e4_models.csv`` (all QC-pass runs) and ``e4_models_nodark.csv`` (dark runs
    left out): the deficit − control slope per day of a mixed model ``response ~ day × treatment + VPD + (1 | plant)``
    on 144 plants. {trait key: {"slope", "ci_low", "ci_high", "p", "unit", "n_plants", "nodark_slope",
    "nodark_ci_low", "nodark_ci_high", "nodark_p"}} for the three kept traits; None without the table."""
    folder = Path(core.DERIVED) / "threed" / "results"
    return _midday(_file_key(folder / "e4_models.csv"), _file_key(folder / "e4_models_nodark.csv"))


@lru_cache(maxsize=2)
def _midday(key: tuple[str, int] | None, nodark_key: tuple[str, int] | None) -> dict | None:
    e4 = None if key is None else _read_csv(*key)
    if e4 is None or not {"trait", "response", "slope_interaction"} <= set(e4.columns):
        return None
    nodark = None if nodark_key is None else _read_csv(*nodark_key)
    out = {}
    for k in TRAITS:
        r = e4[(e4["trait"] == k) & (e4["response"] == "midday_depression")]
        if r.empty:
            continue
        r = r.iloc[0]
        d = dict(
            slope=float(r["slope_interaction"]),
            ci_low=float(r["ci_low"]),
            ci_high=float(r["ci_high"]),
            p=float(r["p_value"]),
            unit=str(r["unit"]),
            n_plants=int(r["n_plants"]),
            nodark_slope=np.nan,
            nodark_ci_low=np.nan,
            nodark_ci_high=np.nan,
            nodark_p=np.nan,
        )
        if nodark is not None and "trait" in nodark:
            n = nodark[(nodark["trait"] == k) & (nodark["response"] == "midday_depression")]
            if len(n):
                n = n.iloc[0]
                d.update(
                    nodark_slope=float(n["slope_interaction"]),
                    nodark_ci_low=float(n["ci_low"]),
                    nodark_ci_high=float(n["ci_high"]),
                    nodark_p=float(n["p_value"]),
                )
        out[k] = d
    return out or None


STALE_AFTER = pd.Timedelta(minutes=10)  # a table written this much later than the onsets is newer than them


def onsets_info() -> dict | None:
    """What the onsets on screen rest on, so the page never shows them as if computed from the table beside them.

    {"n_deficit", "n_control": plants behind the 3D leaf-area onset (derived/detect/onsets, W2 against W1);
     "table_deficit", "table_control": W2 and W1 plants in the 3D table now; "written": when onsets.parquet was
     written; "stale": True when the 3D table was written more than 10 minutes after it (``STALE_AFTER``): the
     detection tables were computed from an earlier 3D table and need ``python -m uiwue.detect.run``}. A fresh
     checkout writes both files within seconds, so it reads as current. None without onsets.parquet.
    """
    o = onsets()
    key, okey = _key(), _detect_key("onsets")
    if o is None or okey is None:
        return None
    leaf = o[o["trait"] == "leaf_area_est"]
    p = plant_days()
    plants = p[["bed", "plant"]].drop_duplicates() if p is not None else pd.DataFrame(columns=["bed", "plant"])
    written = pd.Timestamp(datetime.fromtimestamp(okey[1] / 1e9))  # local time, as a reader's clock shows it
    table = None if key is None else pd.Timestamp(datetime.fromtimestamp(key[1] / 1e9))
    return dict(
        n_deficit=int(leaf["n_deficit"].iloc[0]) if len(leaf) and pd.notna(leaf["n_deficit"].iloc[0]) else 0,
        n_control=int(leaf["n_control"].iloc[0]) if len(leaf) and pd.notna(leaf["n_control"].iloc[0]) else 0,
        table_deficit=int((plants["bed"] == DEFICIT_BED).sum()),
        table_control=int((plants["bed"] == CONTROL_BED).sum()),
        written=written,
        stale=bool(table is not None and table - written > STALE_AFTER),
    )


def sampled_leaf(bed: str, plant: str) -> int | None:
    """The day (days since irrigation off) a leaf was cut from this plant for relative water content, or None.

    From ``derived/truth/rwc_leaves`` (W3 only: one leaf per sampled plant, eight plants a day on days 1–12). That
    cut is why W3 is shown plant by plant and never pooled with W2."""
    key = _file_key(core.table_path("truth", "rwc_leaves"))
    hit = None if key is None else _sampled(*key).get((str(bed), str(plant)))
    return None if hit is None else int(hit)


@lru_cache(maxsize=2)
def _sampled(path: str, stamp: int) -> dict:
    t = _read_parquet(path, stamp)
    if t is None or not {"bed", "plant", "date"} <= set(t.columns):
        return {}
    days = (pd.to_datetime(t["date"]) - pd.Timestamp(core.CUTOFF)).dt.days
    first = pd.DataFrame({"bed": t["bed"].astype(str), "plant": t["plant"].astype(str), "day": days})
    first = first.groupby(["bed", "plant"])["day"].min()
    return {k: int(v) for k, v in first.items()}


# ---------------------------------------------------------------- 6. structure status (Plant profile)
def structure_status(threshold: float = STRUCTURE_THRESHOLD) -> pd.DataFrame | None:
    """Per-plant structure status from leaf area, day by day from day 0, for the status marks (DESIGN 3).

    Rule (brief E, the shape of ``uiwue.diurnal.status``): a usable day is 'watch' when leaf area is below
    ``threshold`` % of the plant's own baseline, 'irrigate' when it is also below on the plant's previous
    usable day (a day without a usable scan does not break the pair), else 'fine'. From day 0 to day 12 (no
    water) an 'irrigate' stays latched, as the Irrigation advisor's ``daily_status`` does; after re-watering
    each day stands alone. Days before 0 are not scored (plants were still growing into their baseline).
    Columns: bed, plant, treatment, day, leaf_area_pct, status ('fine', 'watch', 'irrigate', 'no scan',
    'no baseline'), raw (the day's own status, before latching), since (first day of the current irrigate
    run, NaN otherwise). None without the table.
    """
    key = _key()
    return None if key is None else _view(_structure(*key, float(threshold)))


@lru_cache(maxsize=4)
def _structure(path: str, stamp: int, threshold: float) -> pd.DataFrame | None:
    pdays = _plant_days(path, stamp)
    if pdays is None:
        return None
    d = pdays[(pdays["day"] >= STRESS_DAYS[0]) & pdays["bed"].isin(BEDS)]
    days = sorted(int(x) for x in d["day"].unique())
    rows = []
    for (bed, plant, treatment), g in d.groupby(["bed", "plant", "treatment"], sort=True):
        pct = dict(zip(g["day"].astype(int), g["leaf_area_est_pct"], strict=True))
        has_base = g["leaf_area_est_base"].notna().any()
        prev_below, since = False, np.nan
        for day in days:
            v = pct.get(day, np.nan)
            if not has_base:
                raw = "no baseline"
            elif pd.isna(v):
                raw = "no scan"
            else:
                below = bool(v < threshold)
                raw = "irrigate" if below and prev_below else "watch" if below else "fine"
                prev_below = below
            window = STRESS_DAYS[0] <= day <= STRESS_DAYS[1]
            if not window:
                since = day if raw == "irrigate" else np.nan
                status = raw
            else:
                if raw == "irrigate" and np.isnan(since):
                    since = day
                status = "irrigate" if not np.isnan(since) else raw
            rows.append(
                dict(
                    bed=bed,
                    plant=plant,
                    treatment=treatment,
                    day=day,
                    leaf_area_pct=v,
                    status=status,
                    raw=raw,
                    since=since,
                )
            )
    return pd.DataFrame(rows, columns=["bed", "plant", "treatment", "day", "leaf_area_pct", "status", "raw", "since"])


def structure_alerts(threshold: float = STRUCTURE_THRESHOLD) -> pd.DataFrame | None:
    """One row per plant: when the leaf-area rule of ``structure_status`` first warned while unwatered.

    Columns: bed, plant, treatment, has_baseline, first_watch, first_irrigate (first day 0–12 whose own status
    was that, NaN if never), status_day12 (status on day 12, the last day without water), worst_pct (lowest
    leaf area change on days 0–12, %). The rule's hit rate is the share of deficit plants with a
    first_irrigate; a control plant with one is a false alarm.
    """
    key = _key()
    return None if key is None else _view(_alerts(*key, float(threshold)))


@lru_cache(maxsize=4)
def _alerts(path: str, stamp: int, threshold: float) -> pd.DataFrame | None:
    st = _structure(path, stamp, threshold)
    if st is None:
        return None
    rows = []
    for (bed, plant, treatment), g in st.groupby(["bed", "plant", "treatment"], sort=True):
        w = g[g["day"].between(*STRESS_DAYS)]
        watch = w.loc[w["raw"].isin(["watch", "irrigate"]), "day"]
        irr = w.loc[w["raw"] == "irrigate", "day"]
        d12 = w.loc[w["day"] == STRESS_DAYS[1], "status"]
        rows.append(
            dict(
                bed=bed,
                plant=plant,
                treatment=treatment,
                has_baseline=bool((g["raw"] != "no baseline").any()),
                first_watch=float(watch.min()) if len(watch) else np.nan,
                first_irrigate=float(irr.min()) if len(irr) else np.nan,
                status_day12=d12.iloc[0] if len(d12) else "no scan",
                worst_pct=float(w["leaf_area_pct"].min()) if w["leaf_area_pct"].notna().any() else np.nan,
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7. clouds
def _cloud_path(bed: str, plant: str, date, capture_hour: int) -> Path:
    return clouds_dir() / f"{bed}_{plant}" / f"{pd.Timestamp(date):%Y-%m-%d}_{int(capture_hour):02d}.npz"


def stray_mask(x, y, z) -> np.ndarray:
    """Detached edge points of one plant cloud (mm), for the picture only: the trait table keeps every point.

    Plant points closer than ``STRAY_LINK_MM`` form one cluster (single linkage). A cluster is hidden when it
    floats over the plant (its lowest point is higher than the highest point kept, and it stays at least
    ``STRAY_GAP_MM`` from every point kept) and is either a piece that enters through the crop wall (a point
    within ``STRAY_WALL_MM`` of the 85 mm cylinder, at most ``STRAY_MAX_SHARE`` of the points) or a speck (under
    ``STRAY_SPECK_SHARE`` of the points): a neighbour's leaf or debris the crop cut, not a leaf of this plant,
    which joins it from below. The test repeats until nothing changes, so strays side by side do not shield each
    other. Checked by image on tall controls, whose top leaves stay, and on sparse wilted plants. Returns a
    boolean array, True = hidden.
    """
    p = np.c_[np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)]
    n = len(p)
    if n < MIN_PLANT_POINTS:
        return np.zeros(n, bool)
    pairs = cKDTree(p).query_pairs(STRAY_LINK_MM, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, label = connected_components(graph, directed=False)
    size = np.bincount(label, minlength=k)
    at_wall = np.zeros(k, bool)
    np.logical_or.at(at_wall, label, np.hypot(p[:, 0], p[:, 1]) >= CYLINDER_MM - STRAY_WALL_MM)
    lowest = np.full(k, np.inf)
    np.minimum.at(lowest, label, p[:, 2])
    hidden = (at_wall & (size <= STRAY_MAX_SHARE * n)) | (size < STRAY_SPECK_SHARE * n)
    while hidden.any():
        kept = ~hidden[label]
        if not kept.any():
            return np.zeros(n, bool)
        new = hidden & (lowest > p[kept, 2].max())
        if new.any():
            out = new[label]
            near = cKDTree(p[~out]).query(p[out], k=1)[0] < STRAY_GAP_MM
            close = np.zeros(k, bool)
            np.logical_or.at(close, label[out], near)
            new &= ~close
        if (new == hidden).all():
            break
        hidden = new
    return hidden[label]


@lru_cache(maxsize=256)
def _cloud_file(path: str, stamp: int) -> dict | None:
    """Plant points of a cloud file in mm with their ``stray_mask``, or None when the file cannot be read.

    {"x", "y", "z", "stray": read-only arrays (clouds above ``MAX_POINTS`` plant points are subsampled, seeded),
    "frame" (RGB frame the cloud was built from), "n" (plant points in the file), "n_stray", "z_max" (highest
    point kept, mm)}."""
    try:
        with np.load(path, allow_pickle=False) as z:
            xyz = np.asarray(z["xyz"], np.float32).reshape(-1, 3)
            plant = np.asarray(z["plant"], bool) if "plant" in z.files else None
            frame = int(z["frame"]) if "frame" in z.files else 7
    except Exception:  # noqa: BLE001 - an unreadable file is "no file", not a crash
        return None
    if plant is None or len(plant) != len(xyz):
        return None
    p = xyz[plant] * 1000.0
    stray = stray_mask(p[:, 0], p[:, 1], p[:, 2])
    n, n_stray = len(p), int(stray.sum())
    z_max = float(p[~stray, 2].max()) if (~stray).any() else np.nan
    if n > MAX_POINTS:
        keep = np.sort(np.random.default_rng(SEED).choice(n, MAX_POINTS, replace=False))
        p, stray = p[keep], stray[keep]
    return dict(
        x=_frozen(p[:, 0].copy()),
        y=_frozen(p[:, 1].copy()),
        z=_frozen(p[:, 2].copy()),
        stray=_frozen(stray),
        frame=frame,
        n=n,
        n_stray=n_stray,
        z_max=z_max,
    )


@lru_cache(maxsize=4096)
def _cloud_stats(path: str, stamp: int) -> tuple[int, float, int, int] | None:
    """(plant points, highest plant point kept in mm, RGB frame, detached edge points) of a cloud file."""
    f = _cloud_file(path, stamp)
    return None if f is None else (f["n"], f["z_max"], f["frame"], f["n_stray"])


def _distance(values: pd.DataFrame, centre) -> pd.Series:
    """How far each run sits from ``centre`` over the three traits: summed |log ratio| (9 for a missing value)."""
    total = pd.Series(0.0, index=values.index)
    for k in TRAITS:
        c = float(centre[k]) if k in centre and pd.notna(centre[k]) else np.nan
        v = pd.to_numeric(values[k], errors="coerce").where(lambda s: s > 0)
        d = np.log(v / c).abs() if np.isfinite(c) and c > 0 else pd.Series(np.nan, index=values.index)
        total = total + d.fillna(9.0)
    return total


def _typical_first(ok: pd.DataFrame) -> dict[int, list]:
    """{day: [run, ...]}: each day's QC-pass daytime runs, the most typical first. Typical = closest (``_distance``)
    to the per-trait median of the plant's QC-pass daytime runs on that day and ``NEAR_DAYS`` days either side, so
    one odd run is not the picture of its day; ties go to the slot order ``CLOUD_SLOTS``. A run is (date, slot,
    capture_hour, n_points)."""
    order = {s: i for i, s in enumerate(CLOUD_SLOTS)}
    out: dict[int, list] = {}
    for day, g in ok.groupby("day", sort=True):
        near = ok[ok["day"].between(day - NEAR_DAYS, day + NEAR_DAYS)]
        g = g.assign(_d=_distance(g, near[list(TRAITS)].median()).round(9), _o=g["slot"].map(order))
        g = g.sort_values(["_d", "_o"])
        out[int(day)] = list(zip(g["date"], g["slot"].astype(int), g["capture_hour"].astype(int), g["n_points"]))
    return out


def cloud_days(bed: str, plant: str) -> pd.DataFrame:
    """Every day the plant has a daytime run, and the run to draw for it (``ok``) or why there is none.

    One row per day, sorted: day, date; slot and capture_hour (folder hour; names the file) of the run drawn:
    of the day's QC-pass daytime runs with a drawable cloud, the most typical (closest to the plant's per-trait
    median over that day and the day either side; ties: 12, 15, 9, 18 h); when none is drawable, the most
    typical QC-pass run, to name in the hover. Then n_plant (plant points in that cloud), share (n_plant / the
    run's median plant points over its frames), z_max (highest plant point the picture keeps, mm), frame (RGB
    frame of the cloud), n_stray (detached edge points the picture hides, ``stray_mask``), n_pass (QC-pass
    daytime runs that day), the run's own values ``<trait>_run`` (trait unit) and their change from the plant's
    baseline ``<trait>_run_pct`` (%), ok, and reason: 'ok'; 'no scan' (no daytime run passed QC); 'no file'
    (cloud folder or file missing or unreadable); 'sparse' (under ``MIN_PLANT_POINTS`` plant points, or under
    ``MIN_POINT_SHARE`` of the run's points: a frame that lost part of the plant). Available days are
    ``cloud_days(...).query("ok")``. Works for W3 plants too. Empty frame when the plant is not in the table.
    """
    key = _key()
    if key is None:
        return _empty_days()
    folder = clouds_dir() / f"{bed}_{plant}"
    return _cloud_days(*key, str(bed), str(plant), str(folder), _stamp(folder)).copy(deep=False)


RUN_VALUES = [c for k in TRAITS for c in (f"{k}_run", f"{k}_run_pct")]
DAY_COLUMNS = ["day", "date", "slot", "capture_hour", "n_plant", "share", "z_max", "frame", "n_stray", "n_pass"]


def _empty_days() -> pd.DataFrame:
    return pd.DataFrame(columns=[*DAY_COLUMNS, *RUN_VALUES, "ok", "reason"])


def _judge(bed: str, plant: str, date, capture_hour: int, n_points: float) -> dict:
    """What the cloud of one QC-pass run holds: reason 'ok', 'no file' or 'sparse', and its numbers."""
    f = _cloud_path(bed, plant, date, capture_hour)
    stamp = _stamp(f)
    stats = None if stamp is None else _cloud_stats(str(f), stamp)
    if stats is None:
        return dict(reason="no file", path=f)
    n, z_max, frame, n_stray = stats
    share = n / n_points if pd.notna(n_points) and n_points > 0 else np.nan
    ok = n >= MIN_PLANT_POINTS and not share < MIN_POINT_SHARE
    return dict(
        reason="ok" if ok else "sparse", path=f, n_plant=n, share=share, z_max=z_max, frame=frame, n_stray=n_stray
    )


def _ok(j: dict | None) -> bool:
    return j is not None and j["reason"] == "ok"


def _judge_many(bed: str, plant: str, items: list) -> list[dict]:
    """``_judge`` for several runs; files are opened in parallel, since a first open costs ~10 ms here."""
    work = [(bed, plant, date, cap, n_points) for date, _, cap, n_points in items]
    if len(work) < 4:
        return [_judge(*w) for w in work]
    with ThreadPoolExecutor(max_workers=min(8, len(work))) as pool:
        return list(pool.map(lambda w: _judge(*w), work))


def _baseline_of(path: str, stamp: int, bed: str, plant: str) -> dict:
    """{trait: the plant's baseline (trait unit, NaN without one)} from the plant-day table."""
    p = _plant_days(path, stamp)
    mine = p[(p["bed"] == bed) & (p["plant"] == plant)] if p is not None else None
    if mine is None or mine.empty:
        return {k: np.nan for k in TRAITS}
    return {k: float(mine[f"{k}_base"].iloc[0]) for k in TRAITS}


def _values(row, base: dict) -> dict:
    """A run's three values and their change from the plant's baseline, as ``<trait>_run`` / ``<trait>_run_pct``."""
    out = {}
    for k in TRAITS:
        v = float(row[k]) if pd.notna(row[k]) else np.nan
        b = base.get(k, np.nan)
        out[f"{k}_run"] = v
        out[f"{k}_run_pct"] = (v / b - 1.0) * 100.0 if np.isfinite(b) and b > 0 else np.nan
    return out


@lru_cache(maxsize=256)
def _cloud_days(path: str, stamp: int, bed: str, plant: str, folder: str, folder_stamp: int | None) -> pd.DataFrame:
    """Cached per plant; the folder's mtime is in the key, so an added or removed cloud file is noticed."""
    t = _table(path, stamp)
    if t is None:
        return _empty_days()
    mine = t[(t["bed"] == bed) & (t["plant"] == plant) & t["slot"].isin(DAY_SLOTS)]
    if mine.empty:
        return _empty_days()
    ok = mine[mine["qc_pass"]]
    queue = _typical_first(ok)  # day -> its QC-pass runs, the most typical first
    chosen: dict[int, dict] = {}
    rank = 0
    while True:  # round k judges the k-th choice of every day still without a drawable cloud
        todo = [(day, runs_[rank]) for day, runs_ in queue.items() if rank < len(runs_) and not _ok(chosen.get(day))]
        if not todo:
            break
        for (day, (_, slot, cap, _)), j in zip(todo, _judge_many(bed, plant, [r for _, r in todo]), strict=True):
            j.pop("path")
            j.update(slot=slot, capture_hour=cap)
            if day not in chosen or j["reason"] == "ok":
                chosen[day] = j  # the most typical QC-pass run names the problem when no run is drawable
        rank += 1
    base = _baseline_of(path, stamp, bed, plant)
    by_run = ok.set_index(["day", "slot"])
    passed = ok.groupby("day").size()
    rows = []
    for day, date in mine[["day", "date"]].drop_duplicates().sort_values("day").itertuples(index=False):
        day = int(day)
        row = dict(day=day, date=date, slot=pd.NA, capture_hour=pd.NA, n_plant=pd.NA, share=np.nan, z_max=np.nan)
        row.update(frame=pd.NA, n_stray=pd.NA, n_pass=int(passed.get(day, 0)), ok=False, reason="no scan")
        row.update({c: np.nan for c in RUN_VALUES})
        if day in chosen:
            j = chosen[day]
            row.update(j, ok=j["reason"] == "ok")
            row.update(_values(by_run.loc[(day, j["slot"])], base))
        rows.append(row)
    out = pd.DataFrame(rows)[list(_empty_days().columns)]
    for c in ("slot", "capture_hour", "frame", "n_plant", "n_stray", "n_pass"):
        out[c] = out[c].astype("Int64")
    for c in ("share", "z_max", *RUN_VALUES):
        out[c] = out[c].astype(float)
    out["ok"] = out["ok"].astype(bool)
    return out


def then_run(bed: str, plant: str) -> dict | None:
    """THEN, the plant before the water was cut: of its QC-pass daytime runs on the baseline days (−4 to −1) with a
    drawable cloud, the run whose three values sit closest to the plant's own baseline (summed |log ratio|; ties: the
    later day, then 12, 15, 9, 18 h), so a broken run is never the reference. Without a baseline or such a run: the
    run drawn on the latest drawable day before day 0. Returns {"day", "date", "slot", "capture_hour", "in_baseline",
    "n_stray", and the run's values as in ``cloud_days``}, or None (no drawable cloud before day 0)."""
    key = _key()
    if key is None:
        return None
    folder = clouds_dir() / f"{bed}_{plant}"
    out = _then_run(*key, str(bed), str(plant), str(folder), _stamp(folder))
    return None if out is None else dict(out)


@lru_cache(maxsize=256)
def _then_run(path: str, stamp: int, bed: str, plant: str, folder: str, folder_stamp: int | None) -> dict | None:
    t = _table(path, stamp)
    if t is None:
        return None
    mine = t[(t["bed"] == bed) & (t["plant"] == plant) & t["slot"].isin(DAY_SLOTS) & t["qc_pass"]]
    base = _baseline_of(path, stamp, bed, plant)
    cand = mine[mine["day"].between(*BASELINE_DAYS)]
    if len(cand) and any(np.isfinite(v) for v in base.values()):
        order = {s: i for i, s in enumerate(CLOUD_SLOTS)}
        cand = cand.assign(_d=_distance(cand, base).round(9), _o=cand["slot"].map(order))
        for r in cand.sort_values(["_d", "day", "_o"], ascending=[True, False, True]).itertuples(index=False):
            j = _judge(bed, plant, r.date, r.capture_hour, r.n_points)
            if _ok(j):
                row = mine[(mine["day"] == r.day) & (mine["slot"] == r.slot)].iloc[0]
                return dict(
                    day=int(r.day),
                    date=r.date,
                    slot=int(r.slot),
                    capture_hour=int(r.capture_hour),
                    in_baseline=True,
                    n_stray=int(j["n_stray"]),
                    **_values(row, base),
                )
    d = _cloud_days(path, stamp, bed, plant, folder, folder_stamp)
    before = d[d["ok"] & (d["day"] < 0)]
    if before.empty:
        return None
    r = before.iloc[-1]
    return dict(
        day=int(r["day"]),
        date=r["date"],
        slot=int(r["slot"]),
        capture_hour=int(r["capture_hour"]),
        in_baseline=bool(BASELINE_DAYS[0] <= int(r["day"]) <= BASELINE_DAYS[1]),
        n_stray=int(r["n_stray"]),
        **{c: float(r[c]) for c in RUN_VALUES},
    )


def run_values(bed: str, plant: str, day: int, slot: int) -> dict | None:
    """The three values of one QC-pass daytime run and their change from the plant's baseline, as in
    ``cloud_days`` (``<trait>_run``, ``<trait>_run_pct``); None when it is not a QC-pass daytime run."""
    key = _key()
    t = None if key is None else _table(*key)
    if t is None:
        return None
    m = t[(t["bed"] == bed) & (t["plant"] == plant) & (t["day"] == int(day)) & (t["slot"] == int(slot))]
    if m.empty or not bool(m["qc_pass"].iloc[0]) or int(slot) not in DAY_SLOTS:
        return None
    return _values(m.iloc[0], _baseline_of(*key, bed, plant))


def now_day(bed: str, plant: str) -> int | None:
    """The default NOW of a plant: its latest drawable day without water (day 0–12; the trough when it has one),
    else its latest drawable day, else its latest day with a QC-pass run (no cloud: the readouts still show)."""
    d = cloud_days(bed, plant)
    if d.empty:
        return None
    ok = d[d["ok"]]
    for pick in (ok[ok["day"].between(*STRESS_DAYS)], ok, d[d["reason"] != "no scan"]):
        if len(pick):
            return int(pick["day"].max())
    return None


def default_pair(bed: str, plant: str) -> dict | None:
    """The default THEN and NOW of a plant, both drawable QC-pass clouds (never a failed scan): THEN ``then_run``,
    NOW the latest drawable day of the no-water window 0 to 12 (the trough, days 10 to 12, when it has one).
    {"then", "then_slot", "now", "now_slot", "then_in_baseline", "now_in_trough"}, or None when either is missing."""
    then = then_run(bed, plant)
    d = cloud_days(bed, plant)
    dry = d[d["ok"] & d["day"].between(*STRESS_DAYS)] if len(d) else d
    if then is None or dry.empty:
        return None
    now = dry.iloc[-1]
    return dict(
        then=then["day"],
        then_slot=then["slot"],
        now=int(now["day"]),
        now_slot=int(now["slot"]),
        then_in_baseline=then["in_baseline"],
        now_in_trough=bool(TROUGH_DAYS[0] <= int(now["day"]) <= TROUGH_DAYS[1]),
    )


def default_candidates() -> pd.DataFrame | None:
    """Every W2 plant ranked for the page's default, from the table alone (no cloud is opened here).

    A candidate lost leaf area and height from its peak (days 1–3) to its trough (days 10–12) and has QC-pass
    daytime runs on a baseline day and on a trough day. Columns: plant; then_day, then_slot (its baseline-window
    run closest to its baseline, as ``then_run`` picks it when drawable); now_day, now_slot (the most typical run
    of its latest trough day, as ``cloud_days`` picks it); <trait>_pct (that NOW run against the baseline, %);
    drop (NOW run height / THEN run height); agrees (the NOW run is below the baseline on all three traits, as
    the bed tiles say the deficit bed is); typical (the plant's peak→trough height and leaf area losses both sit
    inside the deficit bed's interquartile range). Sorted best first: typical and agreeing plants, then the
    largest visible drop. None without the table."""
    key = _key()
    return None if key is None else _view(_default_candidates(*key))


@lru_cache(maxsize=2)
def _default_candidates(path: str, stamp: int) -> pd.DataFrame | None:
    t, w = _table(path, stamp), _windows(path, stamp)
    if t is None or w is None:
        return None
    w = w[w["bed"] == DEFICIT_BED].pivot(index="plant", columns="trait", values="peak_to_trough")
    if w.empty:
        return None
    iqr = {k: w[k].quantile([0.25, 0.75]).to_numpy() for k in ("height_p75", "leaf_area_est")}
    ok = t[(t["bed"] == DEFICIT_BED) & t["qc_pass"] & t["slot"].isin(DAY_SLOTS)]
    order = {s: i for i, s in enumerate(CLOUD_SLOTS)}
    rows = []
    for plant, g in ok.groupby("plant", sort=True):
        if plant not in w.index or not (w.loc[plant, "height_p75"] < 0 and w.loc[plant, "leaf_area_est"] < 0):
            continue
        base = _baseline_of(path, stamp, DEFICIT_BED, plant)
        then = g[g["day"].between(*BASELINE_DAYS)]
        trough = g[g["day"].between(*TROUGH_DAYS)]
        if then.empty or trough.empty or not all(np.isfinite(v) for v in base.values()):
            continue
        then = then.assign(_d=_distance(then, base), _o=then["slot"].map(order))
        then = then.sort_values(["_d", "day", "_o"], ascending=[True, False, True]).iloc[0]
        now_day_ = int(trough["day"].max())
        slot = _typical_first(g[g["day"].between(now_day_ - NEAR_DAYS, now_day_ + NEAR_DAYS)])[now_day_][0][1]
        now = g[(g["day"] == now_day_) & (g["slot"] == slot)].iloc[0]
        values = _values(now, base)
        rows.append(
            dict(
                plant=plant,
                then_day=int(then["day"]),
                then_slot=int(then["slot"]),
                now_day=now_day_,
                now_slot=int(slot),
                **{f"{k}_pct": values[f"{k}_run_pct"] for k in TRAITS},
                drop=float(now["height_p75"] / then["height_p75"]) if then["height_p75"] > 0 else np.nan,
                agrees=all(values[f"{k}_run_pct"] < 0 for k in TRAITS),
                typical=all(iqr[k][0] <= w.loc[plant, k] <= iqr[k][1] for k in iqr),
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["typical", "agrees", "drop"], ascending=[False, False, True], ignore_index=True)
    return out


def default_plant() -> tuple[str, str] | None:
    """The page's default plant, by rule: the first of ``default_candidates`` (a typical W2 responder whose NOW run
    on a trough day is below its baseline on all three traits, the largest visible height drop first) whose THEN
    and NOW clouds are drawable on a baseline and a trough day. Clouds are opened candidate by candidate, so this
    costs one or two plants' files. Else the first W2, then W1 plant with any drawable pair; None without the
    table or without clouds."""
    c = default_candidates()
    if c is None:
        return None

    def full(b, p):
        pair = default_pair(b, p)
        if pair is None or not (pair["then_in_baseline"] and pair["now_in_trough"]):
            return False
        now = cloud_days(b, p).set_index("day").loc[pair["now"]]  # the run the card will show
        return all(now[f"{k}_run_pct"] < 0 for k in TRAITS)

    for p in c.loc[c["agrees"], "plant"]:
        if full(DEFICIT_BED, p):
            return DEFICIT_BED, str(p)
    w = windows()
    for b in (DEFICIT_BED, CONTROL_BED):
        for p in sorted(set(w.loc[w["bed"] == b, "plant"])) if w is not None else []:
            if default_pair(b, p) is not None:
                return b, str(p)
    return None


def plant_cloud(bed: str, plant: str, day: int, slot: int | None = None) -> dict | None:
    """Plant-only points of one run in mm, or None: no QC-pass daytime run that day, cloud folder or file
    missing or unreadable, or too sparse to draw (``cloud_days`` reason). Never a synthetic cloud.

    ``slot`` None draws the run ``cloud_days`` picked for that day; a given slot must be a QC-pass run whose
    cloud is drawable. Returns {"x", "y", "z": read-only float32 arrays (mm; x, y on the soil plane with the
    85 mm plant cylinder centred at (0, 0), z = height above soil), "stray" (read-only boolean array: the
    detached edge points ``stray_mask`` hides from the picture), "n" (points), "n_stray", "bed", "plant", "day",
    "date", "slot", "capture_hour", "frame" (RGB frame of the cloud), "path", and the run's own values
    ``<trait>_run`` / ``<trait>_run_pct`` (change from the plant's baseline, %)}.
    """
    run = _run(bed, plant, day, slot)
    if run is None:
        return None
    j = _judge(bed, plant, run["date"], run["capture_hour"], run["n_points"])
    if j["reason"] != "ok":
        return None
    pts = _cloud_file(str(j["path"]), _stamp(j["path"]))
    if pts is None:
        return None
    return dict(
        x=pts["x"],
        y=pts["y"],
        z=pts["z"],
        stray=pts["stray"],
        n=len(pts["x"]),
        n_stray=int(pts["n_stray"]),
        bed=bed,
        plant=plant,
        day=int(day),
        date=run["date"],
        slot=run["slot"],
        capture_hour=run["capture_hour"],
        frame=pts["frame"],
        path=str(j["path"]),
        **run["values"],
    )


def _run(bed: str, plant: str, day: int, slot: int | None) -> dict | None:
    """The QC-pass daytime run to draw for a plant and day: the given slot, or the one ``cloud_days`` picked."""
    key = _key()
    t = None if key is None else _table(*key)
    if t is None:
        return None
    if slot is None:
        d = cloud_days(bed, plant)
        hit = d[(d["day"] == int(day)) & d["ok"]] if len(d) else d
        if hit.empty:
            return None
        slot = int(hit["slot"].iloc[0])
    m = t[(t["bed"] == bed) & (t["plant"] == plant) & (t["day"] == int(day)) & (t["slot"] == int(slot))]
    if m.empty or not bool(m["qc_pass"].iloc[0]) or int(slot) not in DAY_SLOTS:
        return None
    r = m.iloc[0]
    return dict(
        date=r["date"],
        slot=int(r["slot"]),
        capture_hour=int(r["capture_hour"]),
        n_points=r["n_points"],
        values=_values(r, _baseline_of(*key, bed, plant)),
    )


def side_profile(cloud: dict | None, bin_mm: float = PROFILE_BIN_MM) -> pd.DataFrame:
    """Side elevation of a plant cloud (seen along y): the highest plant point in each x-bin.

    Columns: x (bin centre, mm), z (highest point, mm; NaN where the bin is empty), n (points in the bin).
    The bins cover the plant cylinder, −85 to +85 mm, the same for every cloud, so a THEN outline and a NOW
    fill line up bin by bin. Detached edge points (``cloud["stray"]``) are left out, as in the picture.
    ``cloud`` None gives the bins with NaN heights.
    """
    edges = np.arange(-CYLINDER_MM, CYLINDER_MM + bin_mm / 2, bin_mm)
    centre = (edges[:-1] + edges[1:]) / 2
    z = np.full(len(centre), np.nan)
    n = np.zeros(len(centre), dtype="int64")
    if cloud is not None and len(cloud["x"]):
        keep = ~np.asarray(cloud.get("stray", np.zeros(len(cloud["x"]), bool)), bool)
        x, h = np.asarray(cloud["x"], float)[keep], np.asarray(cloud["z"], float)[keep]
        i = np.clip(np.digitize(x, edges) - 1, 0, len(centre) - 1)
        np.fmax.at(z, i, h)
        np.add.at(n, i, 1)
    return pd.DataFrame({"x": centre, "z": z, "n": n})


def axis_ranges(bed: str, plant: str) -> dict | None:
    """Axis ranges in mm shared by every cloud and side profile of a plant, so THEN and NOW use one scale.

    {"x": (−85, 85), "y": (−85, 85), "z": (0, top)}: x and y span the plant cylinder; top is the highest
    plant point the pictures keep over all the plant's drawable clouds, rounded up to 10 mm. None when it has none.
    """
    d = cloud_days(bed, plant)
    z = d.loc[d["ok"], "z_max"].dropna() if len(d) else pd.Series(dtype=float)
    then = then_run(bed, plant)
    tops = [float(z.max())] if len(z) else []
    if then is not None:
        stats = _then_stats(bed, plant, then)
        if stats is not None:
            tops.append(stats)
    if not tops:
        return None
    top = float(np.ceil(max(tops) / 10.0) * 10.0)
    return {"x": (-CYLINDER_MM, CYLINDER_MM), "y": (-CYLINDER_MM, CYLINDER_MM), "z": (0.0, top)}


def _then_stats(bed: str, plant: str, then: dict) -> float | None:
    """Highest kept point of the THEN run's cloud (mm), which need not be the run drawn on its day."""
    f = _cloud_path(bed, plant, then["date"], then["capture_hour"])
    stamp = _stamp(f)
    stats = None if stamp is None else _cloud_stats(str(f), stamp)
    return None if stats is None or not np.isfinite(stats[1]) else float(stats[1])


# ---------------------------------------------------------------- 8. RGB frame
def rgb_frame(bed: str, plant: str, day: int, slot: int | None = None, width: int = THUMB_WIDTH) -> str | None:
    """The RGB frame of a run as a downscaled JPEG data URI (``width`` px wide), or None when it is not on disk.

    ``slot`` None takes the run drawn for that day (``cloud_days``; None when the day has no drawable cloud)
    and the frame its cloud was built from; a given slot takes that QC-pass daytime run. The file is
    ``data/rgb/<bed>_<plant>/<YYYY_MM_DD>/<HH>/rgb_<frame>.jpg`` with HH the folder (capture) hour, so a
    re-run at 13 h is found in ``13/``."""
    run = _run(bed, plant, day, slot)
    if run is None:
        return None
    j = _judge(bed, plant, run["date"], run["capture_hour"], run["n_points"])
    frame = int(j.get("frame", 7))
    folder = rgb_dir() / f"{bed}_{plant}" / f"{pd.Timestamp(run['date']):%Y_%m_%d}" / f"{run['capture_hour']:02d}"
    path = folder / f"rgb_{frame:03d}.jpg"
    stamp = _stamp(path)
    return None if stamp is None else _thumbnail(str(path), stamp, int(width))


@lru_cache(maxsize=128)
def _thumbnail(path: str, stamp: int, width: int) -> str | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.draft("RGB", (width, width))  # JPEG: decode at reduced scale, much faster
            im = im.convert("RGB")
            im.thumbnail((width, width * 4))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
    except OSError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
