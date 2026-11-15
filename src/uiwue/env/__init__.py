"""Greenhouse environment at every scheduled run: derived/env/env_runs.parquet.

    python -m uiwue.env [--data-dir DIR]

One row per plant and scheduled run for all 216 plants, so any module's runs
can be joined to it. Values are taken at the run's nominal time: the slot hour
plus the bed's place in the robot's flyover (W1 first, about 10 min per bed).

Outside light comes from the DWD station Belm (id 00342, 11 km away; CC BY 4.0,
"Source: Deutscher Wetterdienst"), downloaded once to <data dir>/external/.
"""

import argparse
import zipfile
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from uiwue.core import DATA, DATES, SLOTS, all_plants, write_table

SENSOR_W1, SENSOR_W3 = "2", "1"  # sensor 2 sits at the W1 end, sensor 1 at the W3 end; none at W2
BED_OFFSET_MIN = {"W1": 5, "W2": 15, "W3": 25}
WATER_STOP = pd.Timestamp("2025-09-03 11:00")  # last watering of W2/W3 before the stress
WATER_MANUAL = pd.Timestamp("2025-09-15 18:35")  # manual re-watering of W2/W3
WATER_AUTO_AGAIN = pd.Timestamp("2025-09-16")  # schedule (11:00, 19:00) resumes for W2/W3
DWD_SOLAR = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/historical/"
    "10minutenwerte_SOLAR_00342_20200101_20251231_hist.zip"
)
SUNNY = 0.5  # a flyover is "sunny" when the sun shone for at least half of the hour before it
DARK_WM2 = 20  # below this global radiation the leaves get no real sun load

CATALOGUE = [
    (
        "env_air_t",
        "Air temperature",
        "°C",
        0,
        "W1-end sensor, used for every bed (no sensor at W2); 10-min median at the run",
    ),
    (
        "env_air_t_w3",
        "Air temperature, W3 end",
        "°C",
        0,
        "W3-end sensor; reads ~3 °C warmer than the W1 end in the afternoon",
    ),
    ("env_rh", "Relative humidity", "%", 0, "W1-end sensor, 10-min median at the run"),
    ("env_rh_w3", "Relative humidity, W3 end", "%", 0, "W3-end sensor, 10-min median at the run"),
    ("env_vpd", "Vapour pressure deficit", "kPa", 0, "from env_air_t and env_rh; the evaporative demand on the leaves"),
    ("env_vpd_w3", "Vapour pressure deficit, W3 end", "kPa", 0, "from the W3-end sensor"),
    (
        "env_par_mv",
        "Light (PAR sensor)",
        "mV",
        0,
        "raw sensor output, not converted to µmol m⁻² s⁻¹; nearest 15-min reading",
    ),
    (
        "env_vpd_since_water",
        "Demand since last watering",
        "kPa·h",
        0,
        "VPD integrated over time since the bed was last watered (schedule 11:00 and 19:00; W2/W3 dry 3–15 Sep)",
    ),
    ("env_hours_since_water", "Hours since last watering", "h", 0, "time since the bed was last watered"),
    (
        "env_global_rad",
        "Global radiation, outside",
        "W/m²",
        0,
        "DWD station Belm (11 km), 30-min mean around the run",
    ),
    (
        "env_sunshine",
        "Sunshine, last hour",
        "fraction",
        0,
        "DWD station Belm: share of the hour before the run with direct sun",
    ),
    (
        "env_light_wm2",
        "Greenhouse light",
        "W/m²",
        0,
        "light sensor (mV) scaled to DWD global radiation on hourly daytime means; outside-equivalent",
    ),
]


def vpd_kpa(t, rh):
    """Vapour pressure deficit (kPa) from air temperature (°C) and relative humidity (%), Tetens formula."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100)


def _ns(times):
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns").asi8


def at_times(series, times, tol="10min"):
    """Value of a UTC-indexed series nearest each time; NaN where the nearest is further than tol."""
    s = series.dropna().sort_index()
    idx, t = _ns(s.index), _ns(times)
    pos = np.clip(np.searchsorted(idx, t), 1, len(idx) - 1)
    nearest = np.where(np.abs(t - idx[pos - 1]) <= np.abs(idx[pos] - t), pos - 1, pos)
    out = s.values[nearest].astype(float)
    out[np.abs(idx[nearest] - t) > pd.Timedelta(tol).value] = np.nan
    return out


@lru_cache
def air(data_dir=DATA):
    """2-min air series (UTC index): air_t, rh, vpd from the W1-end sensor and *_w3 from the W3 end."""
    a = pd.read_parquet(Path(data_dir) / "air_readings.parquet")
    cols = {}
    for sensor, suffix in ((SENSOR_W1, ""), (SENSOR_W3, "_w3")):
        x = a[a.sensor_number == sensor].set_index("utc_timestamp").sort_index()
        x = x[["air_temperature_value", "humidity_value"]].rolling("10min", center=True).median()
        cols["air_t" + suffix], cols["rh" + suffix] = x.air_temperature_value, x.humidity_value
    df = pd.DataFrame(cols)
    df["vpd"], df["vpd_w3"] = vpd_kpa(df.air_t, df.rh), vpd_kpa(df.air_t_w3, df.rh_w3)
    return df


def par(data_dir=DATA):
    """Raw PAR logger output (mV), UTC index. The sheet has 12 header rows; times are local D.M.Y."""
    p = (
        pd.read_excel(Path(data_dir) / "metadata_rwc_par_en.xlsx", "PAR", header=None, skiprows=13)
        .iloc[:, 1:3]
        .dropna()
    )
    t = pd.to_datetime(p.iloc[:, 0], format="%d.%m.%Y %H:%M:%S").dt.tz_localize("Europe/Berlin").dt.tz_convert("UTC")
    return pd.Series(p.iloc[:, 1].astype(float).values, index=t)


def dwd_frame(raw):
    """DWD 10-min solar product -> global radiation (W/m²) and sunshine fraction, UTC, at each interval's centre.

    GS_10 is J/cm² per 10 min; SD_10 is hours of sunshine per 10 min; MESS_DATUM (UTC) marks the interval end.
    """
    t = pd.to_datetime(raw.MESS_DATUM.astype(str), format="%Y%m%d%H%M").dt.tz_localize("UTC") - pd.Timedelta("5min")
    return pd.DataFrame({"global_rad": raw.GS_10.values * 1e4 / 600, "sunshine": raw.SD_10.values * 6}, index=t)


@lru_cache
def dwd_solar(data_dir=DATA):
    """DWD Belm 10-min radiation and sunshine; downloads the station file (1.5 MB) on first use."""
    path = Path(data_dir) / "external" / DWD_SOLAR.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DWD_SOLAR, path)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.startswith("produkt"))
        raw = pd.read_csv(z.open(name), sep=";", skipinitialspace=True, na_values=[-999], dtype={"MESS_DATUM": str})
    return dwd_frame(raw)


def light_scale(par_mv, solar):
    """(W/m² per mV, hourly r): greenhouse light sensor against outside global radiation, daytime hours only."""
    h = pd.DataFrame({"mv": par_mv.resample("1h").mean(), "gs": solar.global_rad.resample("1h").mean()}).dropna()
    h = h[h.gs > DARK_WM2]
    return (h.gs**2).sum() / (h.mv * h.gs).sum(), h.mv.corr(h.gs)


def watering_times(bed):
    """UTC times the bed was watered (protocol sheet: 8 min per bed at 11:00 and 19:00 daily)."""
    days = pd.date_range("2025-08-01", "2025-09-23", freq="D")
    t = [d + pd.Timedelta(hours=h) for d in days for h in (11, 19)]
    if bed != "W1":
        t = [x for x in t if x <= WATER_STOP or x >= WATER_AUTO_AGAIN] + [WATER_MANUAL]
    return pd.DatetimeIndex(sorted(t)).tz_localize("Europe/Berlin").tz_convert("UTC")


def since_water(times, bed, vpd):
    """(kPa·h of VPD accumulated, hours elapsed) since the bed's last watering before each time."""
    v = vpd.dropna()
    th = _ns(v.index) / 3.6e12
    cum = np.concatenate([[0.0], np.cumsum(np.diff(th) * (v.values[1:] + v.values[:-1]) / 2)])
    w, t = _ns(watering_times(bed)), _ns(times)
    last = w[np.searchsorted(w, t, side="right") - 1]
    demand = np.interp(t / 3.6e12, th, cum) - np.interp(last / 3.6e12, th, cum)
    return demand, (t - last) / 3.6e12


def nominal_times(df):
    """UTC time a bed's plants are scanned in a scheduled run (slot hour + bed offset, local time)."""
    local = df.date + pd.to_timedelta(df.hour, unit="h") + pd.to_timedelta(df.bed.map(BED_OFFSET_MIN), unit="min")
    return local.dt.tz_localize("Europe/Berlin").dt.tz_convert("UTC")


def build(data_dir=DATA):
    days = pd.date_range(*DATES, freq="D")
    g = pd.DataFrame([(b, d, h) for b in BED_OFFSET_MIN for d in days for h in SLOTS], columns=["bed", "date", "hour"])
    g["env_timestamp_utc"] = nominal_times(g)
    a = air(data_dir)
    for col in ("air_t", "air_t_w3", "rh", "rh_w3", "vpd", "vpd_w3"):
        g["env_" + col] = at_times(a[col], g.env_timestamp_utc)
    p, sol = par(data_dir), dwd_solar(data_dir)
    g["env_par_mv"] = at_times(p, g.env_timestamp_utc, tol="20min")
    g["env_light_wm2"] = g.env_par_mv * light_scale(p, sol)[0]
    g["env_global_rad"] = at_times(sol.global_rad.rolling("30min", center=True).mean(), g.env_timestamp_utc)
    g["env_sunshine"] = at_times(sol.sunshine.rolling("60min").mean(), g.env_timestamp_utc)
    # env_sky is a text label, so it stays out of the catalogue (trait pickers list catalogue columns as numbers)
    sky = np.where(g.env_global_rad < DARK_WM2, "dark", np.where(g.env_sunshine >= SUNNY, "sunny", "cloudy"))
    g["env_sky"] = pd.Series(sky, index=g.index).where(g.env_global_rad.notna() & g.env_sunshine.notna())
    for bed, rows in g.groupby("bed"):
        demand, hours = since_water(rows.env_timestamp_utc, bed, a.vpd)
        g.loc[rows.index, "env_vpd_since_water"], g.loc[rows.index, "env_hours_since_water"] = demand, hours
    return g.merge(pd.DataFrame({"plant": all_plants()}), how="cross")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()
    df = build(args.data_dir)
    path = write_table(df, "env", "env_runs", CATALOGUE)
    print(f"{len(df)} rows -> {path}")
    scale, r = light_scale(par(args.data_dir), dwd_solar(args.data_dir))
    print(
        f"greenhouse light vs DWD Belm global radiation: hourly r = {r:.2f} (usable if > 0.8); 1 mV = {scale:.0f} W/m²"
    )
    print("sky at each run (DWD Belm):")
    runs = df.drop_duplicates(["bed", "date", "hour"])
    print(pd.crosstab(runs.hour, runs.env_sky).to_string())
