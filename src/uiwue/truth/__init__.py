"""Ground truth from the sensor and notes sheets: derived/truth/{turgor_runs,plants,rwc_leaves}.parquet.

    python -m uiwue.truth [--data-dir DIR]

DIR holds the environmental sub-dataset files `leaf_readings.parquet` and
`metadata_rwc_par_en.xlsx` (osnaData doi:10.26249/FK2/C0HOBN). Default: the
local data folder (UIWUE_DATA or ./data).

Tables
- turgor_runs: one row per leaf-clip plant and scheduled run (21 clip plants, 2-10 Sep onward):
  the clip reading at the run, and its shift from the clip's own pre-cutoff level.
  Higher clip value = lower turgor, so stress is a positive shift.
- plants: one row per plant (216). Date the clip shift began, and the first date a
  water-stress symptom was written down in the notes sheet.
- rwc_leaves: relative water content of the 70 destructively sampled W3 leaves.
"""

import argparse
import re

import numpy as np
import pandas as pd

from uiwue.core import (
    CUTOFF,
    DATA,
    DERIVED,
    KEYS,
    REWATER,
    TRUTH_CLIP_ONSET,
    TRUTH_FIRST_SYMPTOM,
    all_plants,
    read_table,
    write_table,
)

TZ = "Europe/Berlin"
CLIP_BASELINE = (pd.Timestamp("2025-09-02 12:00", tz=TZ), pd.Timestamp("2025-09-03 11:00", tz=TZ))
RUN_WINDOW = pd.Timedelta(minutes=5)  # clip readings within +-5 min of the run (2-min sampling)
ONSET_SHIFT = 5.0  # raw clip units: above the day-to-day noise of the W1 clips (within +-2.5 until 19 Sep)
ONSET_DAYS = 2  # the daily-median shift must stay above ONSET_SHIFT on this many consecutive days

# Notes that mean water stress, as opposed to pests, damage or weeds. "dead fly" is an insect.
STRESS_WORDS = re.compile(r"wilt|yellow|droop|slump|flop|brown|dry stress|necros|blacken|dead leaves|dead,|dead$")
NOT_STRESS = re.compile(r"dead fly|regenerat|mildew")  # mildew notes mention brown/flopped leaves that are disease

CLIP_CATALOGUE = [
    (
        "truth_turgor_raw",
        "Leaf-clip reading",
        "raw units",
        1,
        "AgriHouse clip, median within 5 min of the run; higher = lower turgor",
    ),
    (
        "truth_turgor_shift",
        "Leaf-clip shift",
        "raw units",
        1,
        "reading minus the clip's own median over 2 Sep 12:00 - 3 Sep 11:00 (before the cutoff); positive = turgor lost",
    ),
]
RWC_CATALOGUE = [
    (
        "truth_rwc",
        "Relative water content",
        "%",
        -1,
        "(fresh - dry) / (saturated - dry) x 100; first sampled leaf (smallest fully extended on 15 Sep)",
    ),
    ("truth_rwc_leaf2", "RWC, second leaf", "%", -1, "next-youngest leaf; sampled on 15 Sep only"),
    ("truth_fw_g", "Fresh weight", "g", 0, "first sampled leaf"),
    ("truth_sw_g", "Saturated weight", "g", 0, "first sampled leaf"),
    ("truth_dw_g", "Dry weight", "g", 0, "first sampled leaf"),
]
PLANT_CATALOGUE = [
    (
        TRUTH_CLIP_ONSET,
        "Clip shift onset",
        "date",
        0,
        f"first day the clip's daily-median shift exceeds {ONSET_SHIFT:g} units on {ONSET_DAYS} days in a row; clip plants only",
    ),
    (
        TRUTH_FIRST_SYMPTOM,
        "First stress symptom",
        "date",
        0,
        "first notes-sheet date (2, 6, 8, 10, 12, 15 Sep) with wilting, yellowing or drooping; an upper bound on the true day",
    ),
]


def read_clip_map(xlsx):
    """Sensor number -> (bed, plant), from the Sensor Guide sheet ('W3 Q8')."""
    g = pd.read_excel(xlsx, sheet_name="Sensor Guide", header=None)
    start = g.index[g[0].astype(str).str.strip() == "Leaf Sensors"][0] + 2
    out = {}
    for _, row in g.iloc[start:].iterrows():
        if not str(row[0]).strip().isdigit():
            break
        bed, plant = str(row[2]).split()
        out[int(row[0])] = (bed, plant)
    return out


def clip_runs(leaf, clip_map, run_times):
    """Clip reading and shift for every clip plant at every scheduled run that has readings."""
    leaf = leaf.assign(sensor=leaf.sensor_number.astype(int), t=leaf.berlin_timestamp)
    in_base = leaf.t.between(*CLIP_BASELINE)
    base = leaf[in_base].groupby("sensor").turgic_value.median().rename("base")
    rows = []
    for sensor, g in leaf.groupby("sensor"):
        bed, plant = clip_map[sensor]
        t = g.t.dt.tz_convert("UTC").to_numpy("datetime64[ns]")
        v = g.turgic_value.to_numpy()
        for r in run_times[run_times.bed == bed].itertuples():
            ts = np.datetime64(r.ts.tz_convert("UTC").tz_localize(None), "ns")
            sel = np.abs(t - ts) <= np.timedelta64(int(RUN_WINDOW.total_seconds()), "s")
            if sel.any():
                rows.append((bed, plant, r.date, r.hour, float(np.median(v[sel])), sensor))
    df = pd.DataFrame(rows, columns=[*KEYS, "truth_turgor_raw", "sensor"])
    df["truth_turgor_shift"] = df.truth_turgor_raw - df.sensor.map(base)
    return df.drop(columns="sensor").sort_values(KEYS, ignore_index=True)


def clip_onset(leaf, clip_map):
    """First day of a sustained clip shift per clip plant (NaT if it never happened before re-watering)."""
    leaf = leaf.assign(sensor=leaf.sensor_number.astype(int), t=leaf.berlin_timestamp)
    base = leaf[leaf.t.between(*CLIP_BASELINE)].groupby("sensor").turgic_value.median()
    shift = leaf.turgic_value - leaf.sensor.map(base)
    daily = shift.groupby([leaf.sensor, leaf.t.dt.normalize().dt.tz_localize(None)]).median().unstack(0)
    stress = daily.loc[pd.Timestamp(CUTOFF) : pd.Timestamp(REWATER)] > ONSET_SHIFT
    hold = stress.copy()
    for k in range(1, ONSET_DAYS):
        hold &= stress.shift(-k, fill_value=False)
    out = {}
    for sensor, (bed, plant) in clip_map.items():
        days = hold.index[hold[sensor]]
        out[(bed, plant)] = days[0] if len(days) else pd.NaT
    return out


def first_symptoms(xlsx):
    """First written water-stress symptom per plant from the BBCH & Symptom sheet, with its note."""
    b = pd.read_excel(xlsx, sheet_name="BBCH & Symptom", header=None)
    date_cols = {
        c: pd.to_datetime(str(b.iloc[0, c]), format="%d.%m.%y")
        for c in range(1, b.shape[1])
        if pd.notna(b.iloc[0, c]) and "*" not in str(b.iloc[0, c])
    }
    out = {}
    for r in range(2, len(b)):
        name = b.iloc[r, 0]
        if not isinstance(name, str) or not re.fullmatch(r"W[123] [A-R]\d", name.strip()):
            continue
        bed, plant = name.split()
        for c, day in date_cols.items():
            if day > pd.Timestamp(REWATER):
                break
            note = b.iloc[r, c + 1]
            if isinstance(note, str) and STRESS_WORDS.search(note.lower()) and not NOT_STRESS.search(note.lower()):
                out[(bed, plant)] = (day, note.strip())
                break
    return out


def plants_table(leaf, xlsx, clip_map):
    onset, symptom = clip_onset(leaf, clip_map), first_symptoms(xlsx)
    rows = [
        {
            "bed": bed,
            "plant": p,
            "truth_has_clip": (bed, p) in onset,
            TRUTH_CLIP_ONSET: onset.get((bed, p), pd.NaT),
            TRUTH_FIRST_SYMPTOM: symptom.get((bed, p), (pd.NaT, ""))[0],
            "truth_first_symptom_note": symptom.get((bed, p), (pd.NaT, ""))[1],
        }
        for bed in ("W1", "W2", "W3")
        for p in all_plants()
    ]
    df = pd.DataFrame(rows)
    for c in (TRUTH_CLIP_ONSET, TRUTH_FIRST_SYMPTOM):
        df[c] = pd.to_datetime(df[c]).dt.date
    return df


def rwc_table(xlsx):
    """RWC per sampled leaf. Plant names with _1/_2 are two leaves of one plant (15 Sep)."""
    r = pd.read_excel(xlsx, sheet_name="RWC", header=0).iloc[:, :6]
    r = r.dropna(subset=["plant"]).copy()
    # dates are typed as numbers: 4.09 is 4 Sep, 10.09 is 10 Sep
    r["date"] = pd.to_datetime(r["date"].map(lambda x: f"{x:.2f}") + ".2025", format="%d.%m.%Y")
    leaf = r.plant.str.extract(r"^(?P<p>[A-R]\d)(?:_(?P<n>\d))?$")
    r["plant"], r["leaf"] = leaf.p, leaf.n.fillna("1").astype(int)
    r["rwc"] = (r.fw - r.dw) / (r.sw - r.dw) * 100
    first = r[r.leaf == 1].rename(
        columns={"rwc": "truth_rwc", "fw": "truth_fw_g", "sw": "truth_sw_g", "dw": "truth_dw_g"}
    )
    second = r[r.leaf == 2][["plant", "date", "rwc"]].rename(columns={"rwc": "truth_rwc_leaf2"})
    out = first.merge(second, on=["plant", "date"], how="left")
    out.insert(0, "bed", "W3")  # only W3 was sampled
    cols = ["bed", "plant", "date", "truth_rwc", "truth_rwc_leaf2", "truth_fw_g", "truth_sw_g", "truth_dw_g"]
    return out[cols].sort_values(["date", "plant"], ignore_index=True)


def run_times_from_env():
    """One UTC timestamp per bed, date and scheduled slot, taken from the environment table."""
    env = read_table("env", "env_runs")
    if env is None:
        raise SystemExit("derived/env/env_runs.parquet is missing; pull main first")
    t = env.groupby(["bed", "date", "hour"], as_index=False).env_timestamp_utc.first()
    return t.rename(columns={"env_timestamp_utc": "ts"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=DATA, help="folder with leaf_readings.parquet and metadata_rwc_par_en.xlsx")
    args = ap.parse_args(argv)
    d = pd.io.common.stringify_path(args.data_dir)
    leaf = pd.read_parquet(f"{d}/leaf_readings.parquet")
    xlsx = f"{d}/metadata_rwc_par_en.xlsx"
    clip_map = read_clip_map(xlsx)

    catalogue = CLIP_CATALOGUE + RWC_CATALOGUE + PLANT_CATALOGUE
    runs = clip_runs(leaf, clip_map, run_times_from_env())
    print("wrote", write_table(runs, "truth", "turgor_runs", catalogue))
    print("wrote", write_table(plants_table(leaf, xlsx, clip_map), "truth", "plants"))
    print("wrote", write_table(rwc_table(xlsx), "truth", "rwc_leaves"))
    print("wrote", DERIVED / "truth" / "catalogue.csv")
