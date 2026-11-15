"""Loaders for the ADA 2025 environmental data and labels used by the 3D experiments.

- Leaf turgor clips (``leaf_readings.parquet``) with the sensor -> plant map from the
  "Sensor Guide" sheet. A higher ``turgic_value`` means LOWER turgor.
- Air (``air_readings.parquet``, two BME280 sensors) with vapour pressure deficit.
- PAR (raw mV, 15 min) from the "PAR" sheet.
- BBCH and written notes per plant ("BBCH & Symptom" sheet) and the first written
  drought symptom per plant.
- The ADA hand-made QC labels (``manually_checked_data.json``), reduced to one row per run.

Facts checked on the real files (2026-11-14):

- Air sensor 1 is W3 and sensor 2 is W1, as the "Sensor Guide" sheet says. Check: until
  2025-09-02 sensor 1 reads about 1 degC cooler and 1 %RH more humid than sensor 2 at midday
  (11-16 h local). From 2025-09-03 it reads 2-4 degC warmer and up to 12 %RH drier, and the gap
  shrinks again after re-watering on 2025-09-15. That is the signature of the deficit bed W3,
  where transpiration stops. W2 has no air sensor; the plan uses W1's sensor for it
  (``AIR_SENSOR_FOR_BED``), which underestimates the VPD at the W2 plants after the cutoff.
- The PAR logger clock is local time (Europe/Berlin, CEST), not UTC. The protocol says the PAR
  sensor was moved at 15:30 on 2025-09-24, and the logged values drop between 15:32 and 15:47.
- Leaf sensors 1-9 are in W3, 10-18 in W2 and 19-21 in W1. Readings start on 2025-09-02 and
  contain drop-outs (values near 0) and spikes (up to 3613); ``load_turgor`` removes values more
  than 20 % away from the sensor's median.
- The ADA QC vocabulary has "Dark", "Medium Dark", "Not Centered" and "no plant" among others.
  "no plant" occurs in one run only (W2_G2, 2025-09-16, 09 h, thermal frames only).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from uiwue.threed import schema
from uiwue.threed.stats import vpd

LEAF_READINGS = schema.DATA_DIR / "leaf_readings.parquet"
AIR_READINGS = schema.DATA_DIR / "air_readings.parquet"
METADATA_XLSX = schema.DATA_DIR / "metadata_rwc_par_en.xlsx"
ADA_QC_JSON = Path("external/ada2025/analysis/dataset_issues/manually_checked_data.json")

# W2 has no air sensor; plan section 7 (E4) uses the W1 sensor for both W1 and W2.
AIR_SENSOR_FOR_BED = {"W1": "W1", "W2": "W1", "W3": "W3"}

# Words in the "Notes" columns that describe a water-deficit symptom. Notes about pests,
# mildew, broken or deformed leaves and leaf counting are not drought symptoms.
DROUGHT_KEYWORDS = ("wilt", "droop", "drap", "yellow", "necro", "limp", "dried", "dry ")

# Raw Protocol, 8.9.25: "many plants in beds 2 and 3 wilting, see image, not just those noted".
# Used as the symptom date for deficit plants without a written symptom of their own.
PROTOCOL_BED_SYMPTOM = {"W2": date(2025, 9, 8), "W3": date(2025, 9, 8)}

# ADA issue label -> our run-level QC flag (plan E0b).
ADA_LABEL_FOR_FLAG = {
    "qc_not_centred": "Not Centered",
    "qc_dark": "Dark",
    "qc_no_plant": "no plant",
}


# ---------------------------------------------------------------- Sensor Guide
def _sensor_section(xlsx: Path, title: str) -> pd.DataFrame:
    """Rows of one block of the "Sensor Guide" sheet ("Leaf Sensors", "Air Sensors", ...)."""
    raw = pd.read_excel(xlsx, sheet_name="Sensor Guide", header=None)
    col0 = raw[0].astype("string").str.strip()
    start = col0[col0 == title].index
    if len(start) == 0:
        raise ValueError(f"'{title}' not found in the Sensor Guide sheet of {xlsx}")
    i = int(start[0]) + 2  # skip the title row and the "Sensor #" header row
    rows = []
    while i < len(raw):
        v = raw.iat[i, 0]
        if pd.isna(v) or not re.fullmatch(r"\d+", str(v).strip()):
            break
        rows.append(raw.iloc[i])
        i += 1
    return pd.DataFrame(rows).reset_index(drop=True)


def leaf_sensor_map(xlsx: Path = METADATA_XLSX) -> pd.DataFrame:
    """Leaf turgor sensor number -> bed and plant. Columns: sensor_number (int), bed, plant."""
    sec = _sensor_section(Path(xlsx), "Leaf Sensors")
    loc = sec[2].astype(str).str.split()
    return pd.DataFrame(
        {
            "sensor_number": sec[0].astype(int),
            "bed": loc.str[0],
            "plant": loc.str[1],
        }
    )


def air_sensor_map(xlsx: Path = METADATA_XLSX) -> pd.DataFrame:
    """Air sensor number -> bed. Columns: sensor_number (int), bed."""
    sec = _sensor_section(Path(xlsx), "Air Sensors")
    return pd.DataFrame(
        {"sensor_number": sec[0].astype(int), "bed": sec[2].astype(str).str.strip()}
    )


# ---------------------------------------------------------------- turgor
def load_turgor(
    path: Path = LEAF_READINGS, xlsx: Path = METADATA_XLSX, max_rel_dev: float = 0.2
) -> pd.DataFrame:
    """Leaf-clip readings with their plant. Columns: utc_timestamp, sensor_number, bed, plant,
    plant_id, turgor_value. A higher value means lower turgor. Values more than ``max_rel_dev``
    (relative) away from the sensor's median are drop-outs or spikes and are removed."""
    df = pd.read_parquet(path, columns=["utc_timestamp", "sensor_number", "turgic_value"])
    df["sensor_number"] = df["sensor_number"].astype(int)
    med = df.groupby("sensor_number")["turgic_value"].transform("median")
    ok = (df["turgic_value"] - med).abs() <= max_rel_dev * med
    df = df[ok].merge(leaf_sensor_map(xlsx), on="sensor_number", how="left")
    df["plant_id"] = df["bed"] + "_" + df["plant"]
    df = df.rename(columns={"turgic_value": "turgor_value"})
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    cols = ["utc_timestamp", "sensor_number", "bed", "plant", "plant_id", "turgor_value"]
    return df[cols].sort_values(["sensor_number", "utc_timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------- air and PAR
def load_air(path: Path = AIR_READINGS, xlsx: Path = METADATA_XLSX) -> pd.DataFrame:
    """Air readings per sensor with its bed and VPD. Columns: utc_timestamp, sensor_number, bed,
    air_temperature_c, rh_pct, pressure_hpa, vpd_kpa."""
    df = pd.read_parquet(path)
    df["sensor_number"] = df["sensor_number"].astype(int)
    df = df.merge(air_sensor_map(xlsx), on="sensor_number", how="left")
    out = pd.DataFrame(
        {
            "utc_timestamp": pd.to_datetime(df["utc_timestamp"], utc=True),
            "sensor_number": df["sensor_number"],
            "bed": df["bed"],
            "air_temperature_c": df["air_temperature_value"].astype(float),
            "rh_pct": df["humidity_value"].astype(float),
            "pressure_hpa": df["air_pressure_value"].astype(float),
        }
    )
    out["vpd_kpa"] = vpd(out["air_temperature_c"], out["rh_pct"])
    return out.sort_values(["bed", "utc_timestamp"]).reset_index(drop=True)


def air_for_bed(air: pd.DataFrame, bed: str) -> pd.DataFrame:
    """The air series used for ``bed`` (W2 uses the W1 sensor)."""
    return air[air["bed"] == AIR_SENSOR_FOR_BED[bed]]


def load_par(xlsx: Path = METADATA_XLSX) -> pd.DataFrame:
    """PAR sensor, raw mV every 15 min. Columns: utc_timestamp, par_mv. The logger clock is
    Europe/Berlin local time and is converted to UTC."""
    raw = pd.read_excel(xlsx, sheet_name="PAR", header=None)
    col0 = raw[0].astype("string").str.strip()
    hdr = col0[col0 == "No.:"].index
    if len(hdr) == 0:
        raise ValueError("PAR sheet: header row 'No.:' not found")
    body = raw.iloc[int(hdr[0]) + 1 :, :3].copy()
    body.columns = ["no", "time", "value"]
    t = pd.to_datetime(body["time"].astype(str), format="%d.%m.%Y %H:%M:%S", errors="coerce")
    t = t.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    out = pd.DataFrame(
        {"utc_timestamp": t, "par_mv": pd.to_numeric(body["value"], errors="coerce")}
    )
    return out.dropna().sort_values("utc_timestamp").reset_index(drop=True)


def daily_window_mean(
    series: pd.DataFrame,
    value_col: str,
    days: list[date] | pd.Series,
    start_hour: int = 9,
    end_hour: int = 15,
) -> pd.Series:
    """Mean of ``value_col`` between ``start_hour`` and ``end_hour`` Europe/Berlin local time on
    each day. Returns a Series indexed by date."""
    local = series["utc_timestamp"].dt.tz_convert("Europe/Berlin")
    hour = local.dt.hour + local.dt.minute / 60
    sel = (hour >= start_hour) & (hour <= end_hour)
    by_day = series.loc[sel, value_col].groupby(local[sel].dt.date).mean()
    return pd.Series([by_day.get(d, np.nan) for d in days], index=list(days), name=value_col)


# ---------------------------------------------------------------- BBCH and symptoms
def _sheet_date(s: str) -> date:
    """'2.9.25' -> date(2025, 9, 2)."""
    return datetime.strptime(str(s).strip(), "%d.%m.%y").date()


def load_bbch(xlsx: Path = METADATA_XLSX) -> pd.DataFrame:
    """The "BBCH & Symptom" sheet in long form. Columns: bed, plant, date, bbch (float, NaN where
    not scored), note (string, NA where empty). The sheet has a two-row header: row 0 holds the
    date above each (BBCH, Notes) pair, row 1 the words "BBCH" and "Notes"."""
    raw = pd.read_excel(xlsx, sheet_name="BBCH & Symptom", header=None)
    pairs = []
    for c in range(raw.shape[1] - 1):
        top, sub = raw.iat[0, c], raw.iat[1, c]
        if isinstance(sub, str) and sub.strip() == "BBCH" and isinstance(top, str):
            pairs.append((_sheet_date(top), c, c + 1))
    rows = []
    for i in range(2, len(raw)):
        pos = raw.iat[i, 0]
        if not isinstance(pos, str) or not re.fullmatch(r"W\d [A-R]\d", pos.strip()):
            continue
        bed, plant = pos.strip().split()
        for d, cb, cn in pairs:
            note = raw.iat[i, cn]
            rows.append(
                {
                    "bed": bed,
                    "plant": plant,
                    "date": d,
                    "bbch": pd.to_numeric(raw.iat[i, cb], errors="coerce"),
                    "note": str(note).strip() if pd.notna(note) and str(note).strip() else pd.NA,
                }
            )
    out = pd.DataFrame(rows)
    out["note"] = out["note"].astype("string")
    return out


def is_drought_note(note: str | None, keywords: tuple[str, ...] = DROUGHT_KEYWORDS) -> bool:
    if note is None or pd.isna(note):
        return False
    s = f" {str(note).lower()} "
    return any(k in s for k in keywords)


def first_symptoms(
    bbch: pd.DataFrame | None = None, keywords: tuple[str, ...] = DROUGHT_KEYWORDS
) -> pd.DataFrame:
    """First written drought symptom per plant. Columns: bed, plant, first_symptom_date, note.
    Plants without such a note are absent."""
    if bbch is None:
        bbch = load_bbch()
    hit = bbch[bbch["note"].map(lambda n: is_drought_note(n, keywords))]
    first = hit.sort_values("date").groupby(["bed", "plant"], as_index=False).first()
    return first.rename(columns={"date": "first_symptom_date"})[
        ["bed", "plant", "first_symptom_date", "note"]
    ]


# ---------------------------------------------------------------- ADA QC labels
def load_ada_qc(path: Path = ADA_QC_JSON) -> pd.DataFrame:
    """The ADA hand labels reduced to one row per run (bed, plant, date, hour).

    Columns per run: counts of frames carrying each label (``n_rgb_dark``, ``n_rgb_medium_dark``,
    ``n_not_centred`` over depth, intensity, RGB and 3D views, ``n_depth_not_centred``,
    ``n_no_plant`` over all views, ``n_depth_missing``), plus the run-level labels used by E0b:

    - ``ada_dark``: at least 8 of the 15 RGB frames are "Dark" (most dark runs have all 15).
    - ``ada_not_centred``: any depth, intensity, RGB or 3D frame is "Not Centered".
    - ``ada_no_plant``: any frame of any camera is "no plant".
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for h in data["hours"]:
        bed, plant = h["plant"].split("_", 1)
        c = dict.fromkeys(
            (
                "n_rgb_dark",
                "n_rgb_medium_dark",
                "n_not_centred",
                "n_depth_not_centred",
                "n_no_plant",
                "n_depth_missing",
            ),
            0,
        )
        for key, flag in h["flags"].items():
            issues = set(flag.get("issues", []))
            view = re.sub(r"_\d+$", "", key)
            if view == "rgb":
                c["n_rgb_dark"] += "Dark" in issues
                c["n_rgb_medium_dark"] += "Medium Dark" in issues
            if "Not Centered" in issues and (
                view in ("depth", "intensity", "rgb") or view.startswith("3d")
            ):
                c["n_not_centred"] += 1
                c["n_depth_not_centred"] += view == "depth"
            c["n_no_plant"] += "no plant" in issues
            c["n_depth_missing"] += view == "depth" and "Missing" in issues
        rows.append(
            {
                "bed": bed,
                "plant": plant,
                "date": schema.folder_date(h["date"]),
                "hour": int(h["hour"]),
                "reviewed_by": h.get("reviewed_by"),
                **c,
            }
        )
    df = pd.DataFrame(rows)
    df["ada_dark"] = df["n_rgb_dark"] >= 8
    df["ada_not_centred"] = df["n_not_centred"] >= 1
    df["ada_no_plant"] = df["n_no_plant"] >= 1
    for k in ("bed", "plant"):
        df[k] = df[k].astype("string")
    return df
