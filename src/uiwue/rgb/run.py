"""RGB extraction: ADA RGB data -> derived/rgb/traits_runs.parquet + catalogue.csv.

    python -m uiwue.rgb.run --data-dir <ADA data> [--plants W1_A2 W2_J5 ...]
                            [--frames 7] [--crop 0.5] [--jobs 4]

Reads either the downloaded archives directly (<data-dir>/rgb/rgb_<bed>_<plant>.zip,
no extraction needed) or extracted folders (<data-dir>/rgb/<bed>_<plant>/<YYYY_MM_DD>/<HH>/).
The archive wins when both exist, because a part-extracted folder can be incomplete.
One row per bed, plant, date, scheduled slot (Master Plan 6.1).
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from datetime import date
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from uiwue.rgb.features import TRAITS, frame_traits
from uiwue.stats._design import KEYS, SLOTS, derived_dir

DATE_MIN, DATE_MAX = date(2025, 8, 24), date(2025, 9, 22)
TRAIT_COLS = list(TRAITS)
_PLANT_RE = re.compile(r"^(W[123])_([A-Z]+\d+)$")
_ZIP_RE = re.compile(r"^rgb_(W[123])_([A-Z]+\d+)\.zip$")
_DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})$")
_IMG_EXT = (".jpg", ".jpeg", ".png")
_SEP = "::"  # a run inside an archive is addressed as "<archive.zip>::<bed_plant/date/hour>"
RUN_COLS = ["bed", "plant", "date", "folder_hour", "path", "n_images"]


# --------------------------------------------------------------------------- #
# sources: a run folder on disk, or a folder inside a zip
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _zip(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _split(run_path: str) -> tuple[str, str] | None:
    return tuple(run_path.split(_SEP, 1)) if _SEP in run_path else None  # type: ignore[return-value]


def _images(run_path: str) -> list[PurePosixPath | Path]:
    """Image files of a run, sorted by name."""
    z = _split(run_path)
    if z:
        prefix = z[1] + "/"
        names = [n for n in _zip(z[0]).namelist()
                 if n.startswith(prefix) and "/" not in n[len(prefix):]
                 and n.lower().endswith(_IMG_EXT)]
        return [PurePosixPath(n) for n in sorted(names)]
    return sorted(p for p in Path(run_path).iterdir() if p.suffix.lower() in _IMG_EXT)


def _read_bytes(run_path: str, name: PurePosixPath | Path) -> bytes:
    z = _split(run_path)
    if z:
        return _zip(z[0]).read(str(name))
    return Path(name).read_bytes()


def _read_rgb(run_path: str, name) -> np.ndarray:
    data = _read_bytes(run_path, name)
    try:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot decode {name}")
        return bgr[..., ::-1]
    except ImportError:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"))


def _timestamp(run_path: str) -> pd.Timestamp:
    """Median capture time from frames_rgb.csv (epoch ns in column 'timestamp')."""
    z = _split(run_path)
    try:
        if z:
            member = z[1] + "/frames_rgb.csv"
            if member not in set(_zip(z[0]).namelist()):
                return pd.NaT
            t = pd.read_csv(io.BytesIO(_zip(z[0]).read(member)))
        else:
            f = Path(run_path) / "frames_rgb.csv"
            if not f.exists():
                return pd.NaT
            t = pd.read_csv(f)
        col = next((c for c in t.columns if "time" in c.lower()), None)
        if col is None:
            return pd.NaT
        ts = pd.to_datetime(t[col], errors="coerce", utc=True).dropna()
        return ts.sort_values().iloc[len(ts) // 2] if len(ts) else pd.NaT
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return pd.NaT


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def rgb_root(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    return data_dir / "rgb" if (data_dir / "rgb").is_dir() else data_dir


def _parse_date(name: str) -> date | None:
    m = _DATE_RE.match(name)
    if not m:
        return None
    d = date(*map(int, m.groups()))
    return d if DATE_MIN <= d <= DATE_MAX else None


def _zip_runs(zpath: Path, bed: str, plant: str) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    for name in _zip(str(zpath)).namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[3].lower().endswith(_IMG_EXT):
            counts[(parts[1], parts[2])] = counts.get((parts[1], parts[2]), 0) + 1
    rows = []
    for (dname, hname), n in sorted(counts.items()):
        d = _parse_date(dname)
        if d and hname.isdigit():
            rows.append(dict(bed=bed, plant=plant, date=d, folder_hour=int(hname),
                             path=f"{zpath}{_SEP}{bed}_{plant}/{dname}/{hname}", n_images=n))
    return rows


def _dir_runs(pdir: Path, bed: str, plant: str) -> list[dict]:
    rows = []
    for ddir in sorted(pdir.iterdir()):
        d = _parse_date(ddir.name)
        if not d or not ddir.is_dir():
            continue
        for hdir in sorted(ddir.iterdir()):
            if hdir.is_dir() and hdir.name.isdigit():
                rows.append(dict(bed=bed, plant=plant, date=d, folder_hour=int(hdir.name),
                                 path=str(hdir), n_images=len(_images(str(hdir)))))
    return rows


def list_runs(data_dir: Path, plants: list[str] | None = None) -> pd.DataFrame:
    """Every run: bed, plant, date, folder_hour, path, n_images."""
    root = rgb_root(data_dir)
    zips, dirs = {}, {}
    for p in sorted(root.iterdir()):
        if (m := _ZIP_RE.match(p.name)) and p.is_file():
            zips[f"{m[1]}_{m[2]}"] = (p, m[1], m[2])
        elif (m := _PLANT_RE.match(p.name)) and p.is_dir():
            dirs[p.name] = (p, m[1], m[2])
    rows = []
    for pid in sorted(set(zips) | set(dirs)):
        if plants and pid not in plants:
            continue
        if pid in zips:
            rows += _zip_runs(*zips[pid])
        else:
            rows += _dir_runs(*dirs[pid])
    return pd.DataFrame(rows, columns=RUN_COLS)


def assign_slots(runs: pd.DataFrame) -> pd.DataFrame:
    """Map folder hours to scheduled slots (6.1) and keep one folder per slot.

    A folder maps to the nearest slot at most 1 h away. If two folders map to
    one slot: the scheduled folder if it has images, else the one with most.
    """
    if runs.empty:
        return runs.assign(hour=pd.Series(dtype=int))
    slots = np.array(SLOTS)
    fh = runs["folder_hour"].to_numpy()
    nearest = slots[np.abs(fh[:, None] - slots[None, :]).argmin(axis=1)]
    r = runs.assign(hour=nearest)
    r = r[(r["folder_hour"] - r["hour"]).abs() <= 1]
    r = r.assign(
        _sched=(r["folder_hour"] == r["hour"]) & (r["n_images"] > 0),
    ).sort_values(["_sched", "n_images"], ascending=False)
    r = r.drop_duplicates(["bed", "plant", "date", "hour"]).drop(columns="_sched")
    return r.sort_values(KEYS).reset_index(drop=True)


def pick_frames(images: list, frames: tuple[int, ...]) -> list:
    """Images whose trailing number is in `frames`; falls back to list position."""
    by_num = {}
    for p in images:
        m = re.search(r"(\d+)$", p.stem)
        if m:
            by_num[int(m[1])] = p
    out = [by_num[f] for f in frames if f in by_num]
    if not out:
        out = [images[f] for f in frames if f < len(images)]
    if not out and images:
        out = [images[len(images) // 2]]
    return out


# --------------------------------------------------------------------------- #
# per run
# --------------------------------------------------------------------------- #
def process_run(run: dict, frames=(7,), crop: float = 0.5) -> dict:
    path = str(run["path"])
    row = {k: run[k] for k in KEYS}
    row["rgb_capture_hour"] = int(run["folder_hour"])
    row["rgb_timestamp_utc"] = _timestamp(path)
    per = []
    for name in pick_frames(_images(path), tuple(frames)):
        try:
            per.append(frame_traits(_read_rgb(path, name), crop=crop))
        except (OSError, KeyError, zipfile.BadZipFile):
            continue
    ok = [f for f in per if not f["dark"] and not f["mask_failed"]]
    row["rgb_n_frames_ok"] = len(ok)
    use = ok or per
    for c in TRAIT_COLS:
        v = np.array([f[c] for f in use], float) if use else np.array([np.nan])
        row[c] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        if c != "rgb_brightness":
            row[f"{c}_mad"] = (
                float(np.nanmedian(np.abs(v - row[c]))) if np.isfinite(v).sum() > 1 else np.nan
            )
    row["rgb_qc_dark"] = bool(per) and all(f["dark"] for f in per)
    row["rgb_qc_mask_failed"] = bool(per) and all(f["mask_failed"] for f in per)
    row["rgb_qc_missing"] = not per
    row["rgb_qc_pass"] = len(ok) > 0
    return row


def extract(data_dir, plants=None, frames=(7,), crop=0.5, jobs=1) -> pd.DataFrame:
    runs = assign_slots(list_runs(Path(data_dir), plants))
    recs = runs.to_dict("records")
    if jobs != 1:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=jobs, batch_size=16)(
            delayed(process_run)(r, frames, crop) for r in recs)
    else:
        try:
            from tqdm import tqdm

            recs = tqdm(recs, desc="rgb runs")
        except ImportError:
            pass
        rows = [process_run(r, frames, crop) for r in recs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(KEYS).reset_index(drop=True)


def catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [(c, *v) for c, v in TRAITS.items()],
        columns=["column", "label", "unit", "direction", "description"],
    )


def write(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else derived_dir() / "rgb"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "traits_runs.parquet", index=False)
    catalogue().to_csv(out / "catalogue.csv", index=False)
    return out / "traits_runs.parquet"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--plants", nargs="*", help="e.g. W1_A2 W2_J5 (default: all found)")
    ap.add_argument("--frames", nargs="*", type=int, default=[7])
    ap.add_argument("--crop", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    # Call through the importable module, not __main__, so joblib workers can unpickle the task.
    from uiwue.rgb import run as rgb_run  # noqa: PLW0406

    df = rgb_run.extract(a.data_dir, a.plants, tuple(a.frames), a.crop, a.jobs)
    if df.empty:
        raise SystemExit(f"no RGB runs found under {rgb_root(a.data_dir)}")
    path = write(df, a.out)
    print(f"{len(df)} runs, {int(df['rgb_qc_pass'].sum())} pass QC -> {path}")


if __name__ == "__main__":
    main()
