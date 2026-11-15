"""Hyperspectral traits: derived/spectral/{traits_runs,spectra_runs}.parquet and wavelengths.csv.

    python -m uiwue.spectral --data-dir DIR --black-reference black_reference.npy [--plants paper|core|W2_R7,...]

DIR holds the unzipped hyperspectral dataset (osnaData doi:10.26249/FK2/PK9YDR), one folder per plant and
run: <bed>_<plant>/<YYYY_MM_DD>/<HH>/hsi_000.jp2 and hsi_000.json. For every run the cube is flat-field
corrected, the plant is masked, and the plant-masked mean of each index is written as one row on the
shared keys (bed, plant, date, hour = the scheduled slot).

The steps are plain functions on numpy arrays so each can be tested without the 205 GB of data
(tests/test_spectral.py). Two things need checking against the real data and the authors' code the
first time it runs, and are kept in one place each so a mismatch is a one-line change:
  - INDEX_BANDS: the band positions of each index (taken from the usual definitions, not from the
    authors' calc_spectral_index.py)
  - WHITE_KEYS: the key of the gray-reference spectrum in hsi_000.json
The plant mask is pluggable (`mask_fn`); the default is an NDVI threshold, the fallback named in the
work plan. The authors' segmentation model is meant to replace it where it holds.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from uiwue.core import DATA, DATES, DERIVED, read_table, slot_for_hour, write_table

HSI_SLOTS = (9, 12, 15)  # the 06 h and 18 h hyperspectral runs are mostly dark (Master-Plan 6.5)
FLAT_FIELD_M = 2.0  # gray-reference scaling in C = (R - D) / (W * m - D) (authors' companion README)
DEFAULT_WAVELENGTHS = np.linspace(384.0, 1025.0, 300)  # Pika L, ~384-1025 nm, 300 bands, if the JSON has none
WHITE_KEYS = ("white_reference", "gray_reference", "grey_reference", "white", "gray", "W")
WAVELENGTH_KEYS = ("wavelengths", "wavelength", "bands", "wavelength_nm")
BAND_HALF_WIDTH_NM = 5.0  # each band position is the mean of the bands within +-5 nm (the sensor samples every ~2 nm)
MIN_MASK_FRACTION = 0.005  # a mask smaller than 0.5 % of the image is not a plant
OVEREXPOSED_FRACTION = 0.05  # more than 5 % of the plant's raw values at the ceiling
DARK_RAW_MEAN = 8.0  # image mean raw value (0-255) below which the run is a dark run
NDVI_MASK_THRESHOLD = 0.4

# ASSUMED band positions (nm), the usual definitions. Check against the authors' calc_spectral_index.py.
INDEX_BANDS = dict(
    ndvi=(800, 670),  # (R800 - R670) / (R800 + R670)
    ci_rededge=(780, 720),  # R780 / R720 - 1
    sipi=(800, 445, 680),  # (R800 - R445) / (R800 - R680)
    pri=(531, 570),  # (R531 - R570) / (R531 + R570)
    wbi=(900, 970),  # R900 / R970
    rep=(670, 700, 740, 780),  # red-edge position, Guyot & Baret linear interpolation
)

CATALOGUE = [
    ("hsi_ndvi", "NDVI", "index", -1, "(R800 - R670) / (R800 + R670), plant-masked mean; falls with chlorophyll loss"),
    ("hsi_ci_rededge", "CI red edge", "index", -1, "R780 / R720 - 1, plant-masked mean; falls with chlorophyll loss"),
    ("hsi_sipi", "SIPI", "index", 1, "(R800 - R445) / (R800 - R680), plant-masked mean; rises as pigments shift"),
    ("hsi_pri", "PRI", "index", 0, "(R531 - R570) / (R531 + R570); photoprotection, direction not settled"),
    ("hsi_wbi", "Water band index", "index", -1, "R900 / R970; the 970 nm water dip shallows as the leaf dries"),
    ("hsi_rep", "Red-edge position", "nm", -1, "wavelength of the red-edge inflection; moves blue under stress"),
    ("hsi_mask_frac", "Plant fraction of image", "ratio", 0, "plant pixels / image pixels; canopy size cross-check"),
]
TRAITS = [c[0] for c in CATALOGUE]


# ------------------------------------------------------------------ reading
def read_cube(path):
    """Raw uint8 cube (height, width, bands) from a .jp2 (or .npy) file."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    import rasterio  # imported here: only the reader needs it

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no geotransform": a cube is not a map
        with rasterio.open(path) as src:
            return np.moveaxis(src.read(), 0, -1)


def white_reference(meta):
    """Gray-reference spectrum from a run's JSON, trying WHITE_KEYS in order."""
    for key in WHITE_KEYS:
        if key in meta:
            return np.asarray(meta[key], dtype=np.float32)
    raise KeyError(f"no gray reference in hsi_000.json (tried {WHITE_KEYS}); its keys are {sorted(meta)[:12]}")


def wavelengths(meta, n_bands):
    for key in WAVELENGTH_KEYS:
        if key in meta and len(meta[key]) == n_bands:
            return np.asarray(meta[key], dtype=np.float64)
    return np.interp(np.arange(n_bands), [0, len(DEFAULT_WAVELENGTHS) - 1], DEFAULT_WAVELENGTHS[[0, -1]])


# ------------------------------------------------------------------ steps
def flat_field(raw, white, dark, m=FLAT_FIELD_M):
    """Reflectance C = (R - D) / (W * m - D); a band where the denominator is not positive is NaN."""
    raw, white, dark = (np.asarray(a, dtype=np.float32) for a in (raw, white, dark))
    denom = white * m - dark
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (raw - dark) / denom
    return np.where(denom > 0, np.clip(out, 0, None), np.nan).astype(np.float32)


def band(refl, wl, nm, half=BAND_HALF_WIDTH_NM):
    """Reflectance at a wavelength: mean of the bands within +-half nm, on the last axis."""
    sel = np.abs(wl - nm) <= half
    if not sel.any():
        sel = np.abs(wl - nm) == np.abs(wl - nm).min()
    return np.nanmean(refl[..., sel], axis=-1)


def ratio(a, b, sign=1):
    """(a - b)/(a + b), or a plain a/b for sign=0; non-finite where the denominator is tiny."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b if sign == 0 else (a - b) / (a + b)
        return np.where(np.abs(b if sign == 0 else a + b) > 1e-6, out, np.nan)


def index_values(refl, wl):
    """Every index at every pixel: `refl` is (pixels, bands); returns a dict of (pixels,) arrays."""
    r = lambda nm: band(refl, wl, nm)
    b = INDEX_BANDS
    with np.errstate(divide="ignore", invalid="ignore"):
        sipi = (r(b["sipi"][0]) - r(b["sipi"][1])) / (r(b["sipi"][0]) - r(b["sipi"][2]))
        r670, r700, r740, r780 = (r(nm) for nm in b["rep"])
        rep = 700 + 40 * (((r670 + r780) / 2) - r700) / (r740 - r700)
        out = dict(
            ndvi=ratio(r(b["ndvi"][0]), r(b["ndvi"][1])),
            ci_rededge=ratio(r(b["ci_rededge"][0]), r(b["ci_rededge"][1]), sign=0) - 1,
            sipi=sipi,
            pri=ratio(r(b["pri"][0]), r(b["pri"][1])),
            wbi=ratio(r(b["wbi"][0]), r(b["wbi"][1]), sign=0),
            rep=rep,
        )
    return {f"hsi_{k}": np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def ndvi_mask(refl, wl, threshold=NDVI_MASK_THRESHOLD):
    """Plant pixels by an NDVI threshold: (height, width, bands) reflectance -> boolean (height, width)."""
    ndvi = ratio(band(refl, wl, INDEX_BANDS["ndvi"][0]), band(refl, wl, INDEX_BANDS["ndvi"][1]))
    return np.nan_to_num(ndvi, nan=-1.0) > threshold


def process_run(raw, white, dark, wl, mask_fn=ndvi_mask, m=FLAT_FIELD_M):
    """Traits, mean spectrum and QC flags of one run from its raw cube. Returns (traits, spectrum)."""
    refl = flat_field(raw, white, dark, m)
    mask = mask_fn(refl, wl)
    n = int(mask.sum())
    row = {"hsi_mask_frac": n / mask.size}
    if n:
        pixels = refl[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bands in a few pixels
            row |= {
                k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in index_values(pixels, wl).items()
            }
            spectrum = np.nanmean(pixels, axis=0).astype(np.float32)
        ceiling = float((raw[mask] >= 255).mean())
    else:
        row |= {k: np.nan for k in TRAITS if k != "hsi_mask_frac"}
        spectrum, ceiling = np.full(refl.shape[-1], np.nan, dtype=np.float32), 0.0
    row["hsi_qc_mask_small"] = row["hsi_mask_frac"] < MIN_MASK_FRACTION
    row["hsi_qc_overexposed"] = ceiling > OVEREXPOSED_FRACTION
    row["hsi_qc_dark"] = bool(np.asarray(raw, dtype=np.float32).mean() < DARK_RAW_MEAN)
    row["hsi_qc_pass"] = not (row["hsi_qc_mask_small"] or row["hsi_qc_overexposed"] or row["hsi_qc_dark"])
    return row, spectrum


# ------------------------------------------------------------------ finding the runs
RUN_DIR = re.compile(r"^(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})$")


def paper_plants():
    """The 10 paper plants: A2 A8 J5 R1 R7 in W1 and W2 (Master-Plan 6.5)."""
    return [(bed, p) for bed in ("W1", "W2") for p in ("A2", "A8", "J5", "R1", "R7")]


def core_plants():
    """The 29 core plants: the paper plants plus every leaf-clip plant (needs truth/plants)."""
    plants = read_table("truth", "plants")
    clips = [] if plants is None else list(zip(plants[plants.truth_has_clip].bed, plants[plants.truth_has_clip].plant))
    return sorted(set(paper_plants()) | set(clips))


def plant_folder(root, bed, plant):
    root = Path(root)
    for base in (root, root / "hyperspectral", root / "hsi"):
        if (base / f"{bed}_{plant}").is_dir():
            return base / f"{bed}_{plant}"
    return None


def discover_runs(root, plants, hours=HSI_SLOTS, dates=DATES):
    """One row per plant, date and scheduled slot with its run folder.

    A folder hour maps to the nearest slot (at most 1 h away). If two folders map to one slot, the
    scheduled hour's folder wins if it has a cube, otherwise the re-run with the most files (Master-Plan 6.1).
    """
    rows = []
    for bed, plant in plants:
        base = plant_folder(root, bed, plant)
        if base is None:
            continue
        for day in sorted(p for p in base.iterdir() if p.is_dir() and RUN_DIR.match(p.name)):
            d = pd.Timestamp(day.name.replace("_", "-"))
            if not (pd.Timestamp(dates[0]) <= d <= pd.Timestamp(dates[1])):
                continue
            for hour_dir in (h for h in day.iterdir() if h.is_dir() and h.name.isdigit()):
                slot = slot_for_hour(int(hour_dir.name))
                cube = next((c for c in (hour_dir / "hsi_000.jp2", hour_dir / "hsi_000.npy") if c.exists()), None)
                if slot in hours and cube is not None:
                    n_files = len(list(hour_dir.iterdir()))
                    rows.append(
                        dict(
                            bed=bed,
                            plant=plant,
                            date=d,
                            hour=slot,
                            capture_hour=int(hour_dir.name),
                            folder=hour_dir,
                            n=n_files,
                            cube=cube,
                        )
                    )
    runs = pd.DataFrame(rows, columns=["bed", "plant", "date", "hour", "capture_hour", "folder", "n", "cube"])
    runs["scheduled"] = runs.capture_hour == runs.hour
    runs = runs.sort_values(
        ["bed", "plant", "date", "hour", "scheduled", "n"], ascending=[True, True, True, True, False, False]
    )
    return (
        runs.drop_duplicates(["bed", "plant", "date", "hour"])
        .drop(columns=["n", "scheduled"], errors="ignore")
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------ the tables
def run_tables(runs, black_reference, mask_fn=ndvi_mask, reader=read_cube, progress=None):
    """Process every discovered run: (traits_runs, spectra_runs, wavelengths)."""
    dark = np.load(black_reference) if isinstance(black_reference, (str, Path)) else np.asarray(black_reference)
    traits, spectra, wl = [], [], None
    for i, r in enumerate(runs.itertuples()):
        raw = reader(r.cube)
        meta = json.loads(Path(r.folder, "hsi_000.json").read_text())
        wl = wavelengths(meta, raw.shape[-1]) if wl is None else wl
        row, spectrum = process_run(raw, white_reference(meta), dark, wl, mask_fn)
        key = dict(bed=r.bed, plant=r.plant, date=r.date, hour=r.hour)
        traits.append(key | row | {"hsi_capture_hour": float(r.capture_hour)})
        spectra.append(key | {"hsi_spectrum": spectrum.tolist()})
        if progress:
            progress(i + 1, len(runs))
    return pd.DataFrame(traits), pd.DataFrame(spectra), wl


def write_tables(traits, spectra, wl):
    paths = [
        write_table(traits, "spectral", "traits_runs", CATALOGUE),
        write_table(spectra, "spectral", "spectra_runs"),
    ]
    pd.DataFrame({"wavelength_nm": np.round(wl, 3)}).to_csv(DERIVED / "spectral" / "wavelengths.csv", index=False)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--data-dir", default=DATA, help="folder with the unzipped <bed>_<plant>/<date>/<hour>/ run folders"
    )
    ap.add_argument("--black-reference", required=True, help="black_reference.npy from the companion repo")
    ap.add_argument("--plants", default="paper", help="paper (10), core (29) or a list like W2_R7,W1_J5")
    ap.add_argument("--hours", default="15", help="scheduled slots to process, e.g. 15 or 9,12,15")
    args = ap.parse_args(argv)
    plants = {"paper": paper_plants, "core": core_plants}.get(
        args.plants, lambda: [tuple(p.split("_")) for p in args.plants.split(",")]
    )()
    hours = tuple(int(h) for h in args.hours.split(","))
    runs = discover_runs(args.data_dir, plants, hours)
    if runs.empty:
        raise SystemExit(f"no hsi_000.jp2 cubes found under {args.data_dir} for {len(plants)} plants")
    print(f"{len(runs)} runs for {runs[['bed', 'plant']].drop_duplicates().shape[0]} plants")
    traits, spectra, wl = run_tables(runs, args.black_reference, progress=lambda i, n: print(f"\r{i}/{n}", end=""))
    print()
    for p in write_tables(traits, spectra, wl):
        print("wrote", p)
