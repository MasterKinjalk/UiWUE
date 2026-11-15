"""Data access for the 3D phenotyping pages (Challenge 3, ``app/pages/threed_*.py``).

Every file name and column comes from the shared contract ``uiwue.threed.schema``. Loaders cache
by path and modification time, so a page callback costs a dictionary lookup once a file is read,
and a file rewritten by the pipeline is picked up on the next callback.

The trait table is the pipeline's ``data/derived/threed/traits_runs.parquet`` when this machine has
it, else the team table committed in git (``derived/threed/traits_runs.parquet``). When neither is
usable, or when the environment variable ``UIWUE_SYNTHETIC=1`` is set, the loaders fall back to the
contract's synthetic data and ``runs_source().synthetic`` is True. Pages show ``source_banner()`` so a reader always knows.

Environment:
    UIWUE_SYNTHETIC=1   always use synthetic data, even when the real files exist
    UIWUE_ROOT=<dir>    repository root that holds ``data/`` (default: the parent of ``app/``)
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import zlib
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from uiwue.threed import schema
    from uiwue.threed.contract import from_team_table
except ModuleNotFoundError:  # package not installed in this environment: use the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from uiwue.threed import schema
    from uiwue.threed.contract import from_team_table

MAX_POINTS = 40_000  # clouds larger than this are randomly downsampled for the browser
THUMB_WIDTH = 480  # px, width of the RGB frame sent to the side panel
RGB_FRAME = "rgb_007.jpg"  # the frame the cached cloud comes from
LEAF_READINGS = schema.DATA_DIR / "leaf_readings.parquet"
SENSOR_SHEET = schema.DATA_DIR / "metadata_rwc_par_en.xlsx"
PIPELINE_DOC = Path("docs") / "threed" / "PIPELINE.md"
TEAM_RUNS = Path("derived") / "threed" / "traits_runs.parquet"  # the committed team table (Master-Plan 6.4)
INTERNAL_COLUMNS = ("plant_id", "day")  # added by the loader, never exported
SYNTHETIC_RESULTS = ("e1_noise.csv", "e3_onset.csv")  # placeholders made in synthetic mode
GROUPINGS = {"treatment": "treatment", "bed": "bed", "plants": "plant_id"}
UNIT_LABELS = {"cm2": "cm²", "cm3": "cm³", "deg": "°", "ratio": "ratio", "count": "count"}
POINT_KEYS = ("xyz", "rgb", "height", "plant", "temp")


# ---------------------------------------------------------------- environment
def repo_root() -> Path:
    env = os.environ.get("UIWUE_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def synthetic_forced() -> bool:
    return os.environ.get("UIWUE_SYNTHETIC", "").strip().lower() in {"1", "true", "yes", "on"}


def _stamp(path: Path) -> int | None:
    """Modification time in ns, or None when the file does not exist."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _fmt_stamp(stamp: int | None) -> str:
    if stamp is None:
        return "unknown"
    return datetime.fromtimestamp(stamp / 1e9).strftime("%Y-%m-%d %H:%M")


def clear_caches() -> None:
    """Forget every cached file (tests, or after replacing files with the same mtime)."""
    for fn in (
        _read_runs,
        _synthetic_runs,
        _read_cloud,
        _synthetic_cloud,
        _thumbnail,
        _read_csv,
        _synthetic_result,
        _sensor_map,
        _leaf_readings,
        _turgor_hourly,
    ):
        fn.cache_clear()


# ---------------------------------------------------------------- labels and parsing
def unit(trait: str) -> str:
    u = schema.TRAITS.get(trait, "")
    return UNIT_LABELS.get(u, u)


def trait_label(trait: str) -> str:
    """'projected_area' -> 'projected_area (cm²)', for axis titles."""
    u = unit(trait)
    return f"{trait} ({u})" if u else trait


def parse_date(value) -> date | None:
    """ISO string, date, datetime or Timestamp -> date; None stays None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(str(value)[:10]).date()


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return bool(value)


# ---------------------------------------------------------------- trait table
@dataclass(frozen=True)
class Source:
    """Where the trait table comes from."""

    synthetic: bool
    reason: str  # why it is synthetic, or the file used
    path: Path | None = None
    stamp: int | None = None

    @property
    def updated(self) -> str:
        return _fmt_stamp(self.stamp)


def runs_source() -> Source:
    """The pipeline's working table if this machine has it, else the team table committed in git."""
    if synthetic_forced():
        return Source(True, "UIWUE_SYNTHETIC=1 is set")
    reasons = []
    for rel_path in (schema.TRAITS_RUNS, TEAM_RUNS):
        rel, path = rel_path.as_posix(), repo_root() / rel_path
        stamp = _stamp(path)
        if stamp is None:
            reasons.append(f"{rel} does not exist yet")
            continue
        _, error = _read_runs(str(path), stamp)
        if error:
            reasons.append(f"{rel} could not be used: {error}")
            continue
        return Source(False, rel, path, stamp)
    return Source(True, "; ".join(reasons))


def load_runs() -> pd.DataFrame:
    """The run table, prepared (see ``_prepare``). Treat it as read-only: it is cached."""
    src = runs_source()
    if src.synthetic:
        return _synthetic_runs()
    df, _ = _read_runs(str(src.path), src.stamp)
    return df


@lru_cache(maxsize=2)
def _read_runs(path: str, stamp: int) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_parquet(path)
        if "depth_capture_hour" in df:  # the team table (derived/threed/), not the pipeline's own
            df = from_team_table(df)
        schema.validate_runs(df)
        return _prepare(df), None
    except Exception as exc:  # noqa: BLE001 - a half-written or wrong file must not take the app down
        return None, f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def _synthetic_runs() -> pd.DataFrame:
    return _prepare(schema.synthetic_runs())


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes, add ``plant_id`` ('W2_J5') and ``day`` (datetime64), sort by the keys."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("bed", "plant", "treatment"):
        df[c] = df[c].astype(str)
    for c in ("hour", "slot", "days_since_cutoff", "n_frames_ok"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype("int64")
    for c in [*schema.QC_FLAGS, "qc_pass"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["plant_id"] = df["bed"] + "_" + df["plant"]
    df["day"] = pd.to_datetime(df["date"])
    return df.sort_values(schema.KEYS).reset_index(drop=True)


def table_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "0 runs"
    return (
        f"{len(df):,} runs · {df['plant_id'].nunique()} plants · "
        f"{df['date'].min().isoformat()} to {df['date'].max().isoformat()}"
    )


# ---------------------------------------------------------------- options and lookups
def beds(df: pd.DataFrame) -> list[str]:
    return sorted(df["bed"].unique())


def plants_in_bed(df: pd.DataFrame, bed: str) -> list[str]:
    return sorted(df.loc[df["bed"] == bed, "plant"].unique())


def plant_ids(df: pd.DataFrame, beds_: list[str] | None = None) -> list[str]:
    d = df if beds_ is None else df[df["bed"].isin(beds_)]
    return sorted(d["plant_id"].unique())


def run_dates(df: pd.DataFrame, bed: str, plant: str) -> list[date]:
    m = (df["bed"] == bed) & (df["plant"] == plant)
    return sorted(df.loc[m, "date"].unique())


def run_hours(df: pd.DataFrame, bed: str, plant: str, d) -> list[int]:
    d = parse_date(d)
    m = (df["bed"] == bed) & (df["plant"] == plant) & (df["date"] == d)
    return sorted(int(h) for h in df.loc[m, "hour"].unique())


def nearest(values: list, target):
    """The element of ``values`` closest to ``target`` (dates or numbers); None if empty."""
    if not values:
        return None
    if target in values:
        return target
    if isinstance(target, date):
        return min(values, key=lambda v: abs((v - target).days))
    return min(values, key=lambda v: abs(v - target))


def get_run(df: pd.DataFrame, bed: str, plant: str, d, hour) -> pd.Series | None:
    d = parse_date(d)
    if d is None or hour is None:
        return None
    m = (df["bed"] == bed) & (df["plant"] == plant) & (df["date"] == d) & (df["hour"] == int(hour))
    rows = df[m]
    return None if rows.empty else rows.iloc[0]


def filter_runs(
    df: pd.DataFrame,
    beds: list[str] | None = None,
    plant_ids: list[str] | None = None,
    start=None,
    end=None,
    hours: list[int] | None = None,
    slots: list[int] | None = None,
    qc_pass_only: bool = False,
) -> pd.DataFrame:
    """Rows matching every given filter. ``None`` means no filter; an empty list matches nothing."""
    m = pd.Series(True, index=df.index)
    if beds is not None:
        m &= df["bed"].isin(beds)
    if plant_ids is not None:
        m &= df["plant_id"].isin(plant_ids)
    if (s := parse_date(start)) is not None:
        m &= df["day"] >= pd.Timestamp(s)
    if (e := parse_date(end)) is not None:
        m &= df["day"] <= pd.Timestamp(e)
    if hours is not None:
        m &= df["hour"].isin([int(h) for h in hours])
    if slots is not None:
        m &= df["slot"].isin([int(h) for h in slots])
    if qc_pass_only:
        m &= df["qc_pass"]
    return df[m]


# ---------------------------------------------------------------- clouds
def cloud_file(bed: str, plant: str, d, hour) -> Path:
    return repo_root() / schema.cloud_path(bed, plant, parse_date(d), int(hour))


def load_cloud(bed: str, plant: str, d, hour) -> dict | None:
    """The cached cloud of one run (CLOUD format, at most ``MAX_POINTS`` points) or None.

    ``cloud["source"]`` is "file" or "synthetic". A synthetic cloud is used when synthetic data is
    forced, when the trait table is synthetic, or when no cloud has been cached at all yet. When
    real clouds exist but not this one, the answer is None, never a made-up cloud.
    """
    d = parse_date(d)
    if d is None or hour is None:
        return None
    if not synthetic_forced():
        path = cloud_file(bed, plant, d, hour)
        stamp = _stamp(path)
        if stamp is not None:
            return _read_cloud(str(path), stamp)
    clouds_exist = (repo_root() / schema.CLOUDS_DIR).is_dir()
    if synthetic_forced() or runs_source().synthetic or not clouds_exist:
        return _synthetic_cloud(bed, plant, d.isoformat(), int(hour))
    return None


def downsample(cloud: dict, max_points: int = MAX_POINTS, seed: int = 0) -> dict:
    n = len(cloud["xyz"])
    if n <= max_points:
        return cloud
    idx = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
    out = dict(cloud)
    for k in POINT_KEYS:
        if out.get(k) is not None:
            out[k] = out[k][idx]
    return out


@lru_cache(maxsize=48)
def _read_cloud(path: str, stamp: int) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            xyz = np.asarray(z["xyz"], dtype=np.float32).reshape(-1, 3)
            n = len(xyz)
            rgb = np.asarray(z["rgb"], np.uint8) if "rgb" in files else np.zeros((n, 3), np.uint8)
            height = np.asarray(z["height"], np.float32) if "height" in files else xyz[:, 2].copy()
            plant = np.asarray(z["plant"], bool) if "plant" in files else np.ones(n, bool)
            temp = np.asarray(z["temp"], np.float32) if "temp" in files else None
            frame = int(z["frame"]) if "frame" in files else None
    except Exception:  # noqa: BLE001 - optional file; the page degrades
        return None
    if temp is not None and not np.isfinite(temp).any():
        temp = None
    cloud = {"xyz": xyz, "rgb": rgb, "height": height, "plant": plant, "temp": temp}
    cloud = downsample(cloud)
    cloud.update(frame=frame, source="file", n_total=n, path=path)
    return cloud


@lru_cache(maxsize=48)
def _synthetic_cloud(bed: str, plant: str, iso: str, hour: int) -> dict:
    row = get_run(load_runs(), bed, plant, iso, hour)
    h = float(row["height_p95"]) if row is not None and np.isfinite(row["height_p95"]) else 120.0
    seed = zlib.crc32(f"{bed}_{plant}_{iso}_{hour}".encode())
    cloud = schema.synthetic_cloud(n=20_000, height_mm=max(h, 5.0), seed=seed)
    cloud.setdefault("temp", None)
    cloud.update(source="synthetic", n_total=len(cloud["xyz"]), path=None)
    return cloud


# ---------------------------------------------------------------- RGB frame
def rgb_frame_path(bed: str, plant: str, d, hour) -> Path:
    d = parse_date(d)
    return (
        repo_root()
        / schema.DATA_DIR
        / "rgb"
        / f"{bed}_{plant}"
        / d.strftime("%Y_%m_%d")
        / f"{int(hour):02d}"
        / RGB_FRAME
    )


def rgb_thumbnail(bed: str, plant: str, d, hour, width: int = THUMB_WIDTH) -> str | None:
    """Frame 7 of the run as a downscaled JPEG data URI, or None when it is not on disk."""
    if parse_date(d) is None or hour is None:
        return None
    path = rgb_frame_path(bed, plant, d, hour)
    stamp = _stamp(path)
    return None if stamp is None else _thumbnail(str(path), stamp, int(width))


@lru_cache(maxsize=256)
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


# ---------------------------------------------------------------- experiment results
@dataclass(frozen=True)
class Result:
    """One experiment table, or the reason it is not available."""

    df: pd.DataFrame | None
    note: str
    synthetic: bool = False

    @property
    def ok(self) -> bool:
        return self.df is not None


def results_dir() -> Path:
    return repo_root() / schema.RESULTS_DIR


def load_result(name: str) -> Result:
    """``name`` is a key of ``schema.RESULT_FILES``, for example 'e3_onset.csv'."""
    if name not in schema.RESULT_FILES:
        raise KeyError(f"{name} is not in schema.RESULT_FILES")
    rel = (schema.RESULTS_DIR / name).as_posix()
    if runs_source().synthetic:
        if name in SYNTHETIC_RESULTS:
            note = "synthetic placeholder computed from the synthetic table, not a result"
            return Result(_synthetic_result(name), note, True)
        return Result(None, f"{name} is not available for synthetic data", True)
    path = results_dir() / name
    stamp = _stamp(path)
    if stamp is None:
        return Result(None, f"not available yet: {rel} does not exist")
    df, error = _read_csv(str(path), stamp)
    if error:
        return Result(None, f"{rel} could not be read: {error}")
    missing = [c for c in schema.RESULT_FILES[name] if c not in df.columns]
    if missing:
        return Result(None, f"{rel} lacks the columns {missing}")
    return Result(df, f"{rel}, updated {_fmt_stamp(stamp)}")


def result_row(name: str, trait: str) -> tuple[dict | None, Result]:
    """The row of a per-trait result table for ``trait`` (first match), plus the table."""
    res = load_result(name)
    if not res.ok or "trait" not in res.df.columns:
        return None, res
    rows = res.df[res.df["trait"].astype(str) == trait]
    return (None if rows.empty else rows.iloc[0].to_dict()), res


def unreliable_traits() -> set[str]:
    """Traits E1 calls not reliable at this sensor resolution (empty when E1 is missing)."""
    res = load_result("e1_noise.csv")
    if not res.ok:
        return set()
    return {str(r["trait"]) for _, r in res.df.iterrows() if not _truthy(r["reliable"])}


@lru_cache(maxsize=16)
def _read_csv(path: str, stamp: int) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # noqa: BLE001 - optional file; the page degrades
        return None, f"{type(exc).__name__}: {exc}"


@lru_cache(maxsize=4)
def _synthetic_result(name: str) -> pd.DataFrame:
    """Rough stand-ins for E1 and E3 so the result card can be seen before the real run.

    E1: noise = median run MAD, signal = |control - deficit| on 2025-09-14 (15 h), reliable when
    the ratio is at least 3. E3: least-squares slope of the daily deficit-minus-control
    difference of % change from baseline; onset = first of two days where that difference
    exceeds twice the pooled SD. No p-values or lead times: those need the real statistics.
    """
    df = _synthetic_runs()
    d15 = df[(df["slot"] == 15) & df["qc_pass"]]
    rows = []
    if name == "e1_noise.csv":
        day = d15[d15["date"] == schema.TREATMENT_WINDOW[1]]
        for t in schema.TRAITS:
            noise = float(d15[f"{t}_mad"].median())
            m = day.groupby("treatment")[t].mean()
            signal = abs(float(m.get("control", np.nan)) - float(m.get("deficit", np.nan)))
            snr = signal / noise if noise > 0 else np.nan
            rows.append({"trait": t, "noise": noise, "signal": signal, "snr": snr, "reliable": snr >= 3})
        return pd.DataFrame(rows)
    if name == "e3_onset.csv":
        for t in schema.TRAITS:
            _per_plant, summary = trajectories(
                df,
                t,
                "treatment",
                slot=15,
                start=schema.CUTOFF,
                end=schema.TREATMENT_WINDOW[1],
                normalise=True,
            )
            w = summary.pivot(index="day", columns="group", values=["mean", "sd"]).dropna()
            row = {
                "trait": t,
                "slope_interaction": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "onset_day": np.nan,
                "lead_vs_symptom_days": np.nan,
                "unit": "% of baseline per day",
            }
            if len(w) >= 3 and {"control", "deficit"} <= set(w["mean"].columns):
                diff = (w["mean"]["deficit"] - w["mean"]["control"]).to_numpy()
                pooled = np.sqrt((w["sd"]["deficit"] ** 2 + w["sd"]["control"] ** 2) / 2)
                days = np.array([(x.date() - schema.CUTOFF).days for x in w.index], float)
                (slope, _), cov = np.polyfit(days, diff, 1, cov=True)
                half = 1.96 * float(np.sqrt(cov[0, 0]))
                big = np.abs(diff) > 2 * pooled.to_numpy()
                onset = next((days[i] for i in range(len(big) - 1) if big[i] and big[i + 1]), None)
                row.update(
                    slope_interaction=slope,
                    ci_low=slope - half,
                    ci_high=slope + half,
                    onset_day=np.nan if onset is None else int(onset),
                )
            rows.append(row)
        return pd.DataFrame(rows)
    raise KeyError(name)


# ---------------------------------------------------------------- turgor (leaf clip) readings
_LOCATION = re.compile(r"^(W[1-3])\s+([A-R]\d+)$")


def sensor_map() -> dict[str, str]:
    """Leaf sensor number ('10') -> plant id ('W2_R7'), from the 'Sensor Guide' sheet."""
    path = repo_root() / SENSOR_SHEET
    stamp = _stamp(path)
    return {} if stamp is None else dict(_sensor_map(str(path), stamp))


@lru_cache(maxsize=1)
def _sensor_map(path: str, stamp: int) -> tuple[tuple[str, str], ...]:
    try:
        sheet = pd.read_excel(path, sheet_name="Sensor Guide", header=None)
    except Exception:  # noqa: BLE001 - optional file; the page degrades
        return ()
    pairs, in_leaf = [], False
    for _, row in sheet.iterrows():
        first = str(row[0]).strip()
        if first == "Leaf Sensors":
            in_leaf = True
            continue
        if in_leaf and first in ("nan", ""):
            break
        if in_leaf and (m := _LOCATION.match(str(row[2]).strip())):
            try:
                pairs.append((str(int(float(first))), f"{m.group(1)}_{m.group(2)}"))
            except ValueError:
                continue
    return tuple(pairs)


def turgor_plants(df: pd.DataFrame) -> list[str]:
    """Plants in the table that carry a leaf clip."""
    return sorted(set(sensor_map().values()) & set(df["plant_id"]))


def load_turgor(plant_id: str) -> pd.DataFrame | None:
    """Hourly median clip reading for one plant: columns ``time_utc``, ``turgor``. None if absent.

    Higher readings mean lower turgor (leaf_readings.parquet, ``turgic_value``)."""
    sensors = [s for s, p in sensor_map().items() if p == plant_id]
    path = repo_root() / LEAF_READINGS
    stamp = _stamp(path)
    if not sensors or stamp is None:
        return None
    out = _turgor_hourly(str(path), stamp, sensors[0])
    return None if out is None or out.empty else out


@lru_cache(maxsize=1)
def _leaf_readings(path: str, stamp: int) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path, columns=["utc_timestamp", "sensor_number", "turgic_value"])
    except Exception:  # noqa: BLE001 - optional file; the page degrades
        return None
    df["sensor_number"] = df["sensor_number"].astype(str)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    return df


@lru_cache(maxsize=32)
def _turgor_hourly(path: str, stamp: int, sensor: str) -> pd.DataFrame | None:
    df = _leaf_readings(path, stamp)
    if df is None:
        return None
    s = df.loc[df["sensor_number"] == sensor].set_index("utc_timestamp")["turgic_value"]
    hourly = s.resample("1h").median().dropna()
    return pd.DataFrame({"time_utc": hourly.index, "turgor": hourly.to_numpy()})


# ---------------------------------------------------------------- analysis for the Compare page
def trajectories(
    df: pd.DataFrame,
    trait: str,
    grouping: str = "treatment",
    plant_ids: list[str] | None = None,
    slot: int = 15,
    start=None,
    end=None,
    normalise: bool = False,
    include_failed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily values of one trait at one slot, per plant and per group.

    Returns ``per_plant`` (group, plant_id, day, value) and ``summary`` (group, day, mean, sd, n).
    ``normalise`` gives the % change from each plant's mean over ``schema.BASELINE_DATES`` (same
    slot), as E3 does (``stats.baseline_normalise(mode="pct")``); plants without a baseline run
    are dropped. QC-failed runs are left out unless ``include_failed``.
    """
    d = df[df["slot"] == int(slot)]
    if not include_failed:
        d = d[d["qc_pass"]]
    if grouping == "plants" and plant_ids:
        d = d[d["plant_id"].isin(plant_ids)]
    d = d[["plant_id", "bed", "treatment", "date", "day", trait]].dropna(subset=[trait])
    if normalise:
        base = d[d["date"].isin(schema.BASELINE_DATES)].groupby("plant_id")[trait].mean()
        b = d["plant_id"].map(base).where(lambda x: x != 0)
        d = d.assign(value=100.0 * (d[trait] / b - 1.0)).dropna(subset=["value"])
    else:
        d = d.assign(value=d[trait])
    d = filter_runs(d, start=start, end=end)
    d = d.assign(group=d[GROUPINGS[grouping]].astype(str))
    per_plant = d.groupby(["group", "plant_id", "day"], as_index=False)["value"].mean()
    summary = per_plant.groupby(["group", "day"])["value"].agg(mean="mean", sd="std", n="count").reset_index()
    summary["sd"] = summary["sd"].fillna(0.0)
    return per_plant, summary


def diurnal_matrix(
    df: pd.DataFrame,
    trait: str,
    grouping: str = "treatment",
    start=None,
    end=None,
    include_failed: bool = False,
    deviation: bool = True,
    plant_ids: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Group -> table of mean trait value, index = day, columns = slot hour.

    With ``deviation`` each run is taken relative to its plant's mean of that day, so the table
    shows the daily rhythm rather than growth."""
    d = filter_runs(df, start=start, end=end, qc_pass_only=not include_failed)
    if grouping == "plants" and plant_ids:
        d = d[d["plant_id"].isin(plant_ids)]
    d = d.dropna(subset=[trait])
    if d.empty:
        return {}
    value = d[trait]
    if deviation:
        value = value - d.groupby(["plant_id", "date"])[trait].transform("mean")
    d = d.assign(value=value, group=d[GROUPINGS[grouping]].astype(str))
    per_plant = d.groupby(["group", "plant_id", "day", "slot"], as_index=False)["value"].mean()
    out = {}
    for g, part in per_plant.groupby("group"):
        out[str(g)] = part.pivot_table(index="day", columns="slot", values="value", aggfunc="mean")
    return out


def diurnal_effects(df: pd.DataFrame, trait: str, include_failed: bool = False) -> pd.DataFrame:
    """Per plant and day: midday depression (15 h - 9 h) and overnight recovery (6 h next day -
    18 h), with the columns of ``e4_diurnal.csv``. Computed from the table, as a fallback."""
    cols = schema.RESULT_FILES["e4_diurnal.csv"]
    d = filter_runs(df, qc_pass_only=not include_failed).dropna(subset=[trait])
    if d.empty:
        return pd.DataFrame(columns=cols)
    w = d.pivot_table(
        index=["bed", "plant", "plant_id", "day"], columns="slot", values=trait, aggfunc="mean"
    ).reset_index()
    w.columns = [c if isinstance(c, str) else int(c) for c in w.columns]
    for s in schema.SLOTS:
        if s not in w.columns:
            w[s] = np.nan
    nxt = w[["plant_id", "day", 6]].rename(columns={6: "next6"})
    nxt["day"] = nxt["day"] - pd.Timedelta(days=1)
    w = w.merge(nxt, on=["plant_id", "day"], how="left")
    out = pd.DataFrame(
        {
            "bed": w["bed"],
            "plant": w["plant"],
            "date": w["day"].dt.date,
            "trait": trait,
            "midday_depression": w[15] - w[9],
            "overnight_recovery": w["next6"] - w[18],
        }
    )
    return out[cols]


# ---------------------------------------------------------------- export
def export_columns(traits: list[str] | None = None, include_mad: bool = True) -> list[str]:
    traits = [t for t in (traits or list(schema.TRAITS)) if t in schema.TRAITS]
    cols = list(schema.RUN_META)
    for t in traits:
        cols += [t, f"{t}_mad"] if include_mad else [t]
    return cols + list(schema.QC_FLAGS) + ["qc_pass"]


def export_frame(df: pd.DataFrame, traits: list[str] | None = None, include_mad: bool = True) -> pd.DataFrame:
    """Contract columns only (no loader helpers), dates as ISO text, ready for CSV."""
    out = df[export_columns(traits, include_mad)].copy()
    out["date"] = [d.isoformat() for d in out["date"]]
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def _md_table(df: pd.DataFrame) -> str:
    def cell(v) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _pipeline_parameters(text: str) -> str:
    """The section of PIPELINE.md whose heading mentions parameters, else the whole file."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)", line)
        if m and "param" in m.group(2).lower():
            level = len(m.group(1))
            end = next(
                (
                    j
                    for j in range(i + 1, len(lines))
                    if (n := re.match(r"^(#+)\s", lines[j])) and len(n.group(1)) <= level
                ),
                len(lines),
            )
            return "\n".join(lines[i:end]).strip()
    return text.strip()


def method_summary(selection: pd.DataFrame, filters: dict, traits: list[str] | None = None) -> str:
    """A Markdown method summary for a download: data source, selection, calendar, noise floor
    per trait (E1), QC flags, pipeline parameters (docs/threed/PIPELINE.md) and E0a parity."""
    src = runs_source()
    traits = [t for t in (traits or list(schema.TRAITS)) if t in schema.TRAITS]
    out = ["# 3D phenotyping (Challenge 3): method summary", ""]
    out.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} by the UiWUE app (3D Export page).")
    out.append("")
    if src.synthetic:
        out += [f"**SYNTHETIC DATA** ({src.reason}). These numbers are not measurements.", ""]
    else:
        out += [f"Data: `{src.reason}`, updated {src.updated}.", ""]
    out += ["## Selection", ""]
    for k, v in filters.items():
        out.append(f"- {k}: {v}")
    out += [f"- rows: {len(selection)} ({table_summary(selection)})", ""]
    out += [
        "## Calendar",
        "",
        f"- irrigation cutoff in deficit beds: {schema.CUTOFF.isoformat()}",
        f"- re-watering: {schema.REWATER.isoformat()}",
        f"- baseline: each plant's mean over {', '.join(d.isoformat() for d in schema.BASELINE_DATES)}",
        f"- treatment window: {schema.TREATMENT_WINDOW[0]} to {schema.TREATMENT_WINDOW[1]}",
        f"- beds: {', '.join(f'{b} {t}' for b, t in schema.BED_TREATMENT.items())}",
        "",
        "## Traits and noise floor (E1)",
        "",
    ]
    e1 = load_result("e1_noise.csv")
    table = pd.DataFrame({"trait": traits, "unit": [schema.TRAITS[t] for t in traits]})
    if e1.ok:
        cols = ["trait", "noise", "signal", "snr", "reliable"]
        table = table.merge(e1.df[cols].astype({"trait": str}), on="trait", how="left")
        out += [f"Source: {e1.note}. A trait with SNR below 3 is not reliable.", ""]
    else:
        out += [f"Noise floor: {e1.note}.", ""]
    out += [_md_table(table), ""]
    out += ["## QC flags", "", "A blocking flag sets `qc_pass = False`.", ""]
    qc = pd.DataFrame({"flag": list(schema.QC_FLAGS), "blocking": list(schema.QC_FLAGS.values())})
    out += [_md_table(qc), ""]
    out += ["## Pipeline parameters", ""]
    doc = repo_root() / PIPELINE_DOC
    if doc.is_file():
        out += [
            f"From `{PIPELINE_DOC.as_posix()}`:",
            "",
            _pipeline_parameters(doc.read_text("utf-8")),
        ]
    else:
        out.append(f"`{PIPELINE_DOC.as_posix()}` is not written yet.")
    out.append("")
    e0a = load_result("e0a_parity.csv")
    out += ["## Paper parity (E0a)", ""]
    out += [_md_table(e0a.df), "", f"Source: {e0a.note}."] if e0a.ok else [f"E0a: {e0a.note}."]
    out.append("")
    present = sorted(p.name for p in results_dir().glob("*.csv")) if results_dir().is_dir() else []
    out += ["## Result files present", ""]
    out += [f"- `{(schema.RESULTS_DIR / p).as_posix()}`" for p in present] or ["- none yet"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- UI helper shared by the pages
def source_banner():
    """A warning when the data are synthetic, otherwise a one-line note on the table used."""
    import dash_bootstrap_components as dbc
    from dash import html

    src = runs_source()
    df = load_runs()
    if src.synthetic:
        return dbc.Alert(
            [
                html.Strong("Synthetic data. "),
                (
                    f"The traits, 3D clouds and results shown are generated for development "
                    f"({src.reason}); they are not measurements. RGB frames, where shown, are real. "
                ),
                html.Span(table_summary(df), className="text-nowrap"),
            ],
            color="warning",
            class_name="py-2 mb-3 small",
        )
    return html.Div(
        f"Data: {src.reason} · {table_summary(df)} · updated {src.updated}",
        className="text-muted small mb-2",
    )


# ---------------------------------------------------------------- figure helpers shared by the pages
HOUR_COLOURS = {6: "#56B4E9", 9: "#009E73", 12: "#E69F00", 15: "#D55E00", 18: "#CC79A7"}
GROUP_COLOURS = {
    "control": "#0072B2",
    "deficit": "#D55E00",
    "W1": "#0072B2",
    "W2": "#D55E00",
    "W3": "#CC79A7",
}
PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#661100",
    "#6699CC",
    "#888888",
]


def hour_colour(hour: int) -> str:
    return HOUR_COLOURS.get(schema.slot_of(int(hour)), "#6c757d")


def group_colour(group: str, i: int) -> str:
    return GROUP_COLOURS.get(group, PALETTE[i % len(PALETTE)])


def rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[k : k + 2], 16) for k in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def value_format(trait: str) -> str:
    """d3 format for hover text and tables: 3 decimals for ratios, none for counts."""
    return {"ratio": ".3f", "count": ",.0f"}.get(schema.TRAITS.get(trait, ""), ".1f")


def fmt_value(trait: str, v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(v):
        return "–"
    return format(v, value_format(trait))


def add_calendar(fig, annotate: bool = True) -> None:
    """Shade the drought window and mark the irrigation cutoff and re-watering (date x axis)."""
    fig.add_vrect(
        x0=schema.CUTOFF.isoformat(),
        x1=schema.REWATER.isoformat(),
        fillcolor="#E69F00",
        opacity=0.07,
        line_width=0,
        layer="below",
    )
    for d, text in ((schema.CUTOFF, "irrigation off"), (schema.REWATER, "re-watered")):
        fig.add_vline(x=d.isoformat(), line={"color": "#6c757d", "width": 1, "dash": "dot"})
        if annotate:
            fig.add_annotation(
                x=d.isoformat(),
                y=1.0,
                xref="x",
                yref="paper",
                text=text,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font={"size": 10, "color": "#6c757d"},
            )


def message_figure(text: str, height: int = 360):
    """An empty figure that says why there is nothing to show."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#6c757d"},
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    return fig
