"""Hyperspectral pipeline on synthetic cubes: no real data is needed to prove the maths, the folder
handling, the JP2 reading and the tables it writes. What these tests cannot prove is that the index
band positions and the JSON key of the gray reference match the authors' files (see the module docstring).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uiwue import spectral as sp
from uiwue.core.validate import validate

pytest.importorskip("rasterio")

ROOT = Path(__file__).resolve().parents[1]
WL = sp.DEFAULT_WAVELENGTHS
WHITE, DARK_LEVEL, M = 100.0, 5.0, sp.FLAT_FIELD_M  # raw ceiling = 100 * 2 = 200 < 255: no clipping unless a test asks


def leaf_spectrum(chlorophyll=1.0, water=1.0, wl=WL):
    """A plausible green-leaf reflectance: green bump, red absorption, red edge near 720 nm, NIR plateau,
    and a 970 nm water dip. Less chlorophyll = shallower red absorption and an edge further to the blue;
    less water = a shallower 970 dip."""
    edge_nm = 705 + 15 * chlorophyll
    green = 0.08 + 0.07 * np.exp(-(((wl - 550) / 30) ** 2))
    red_absorb = 0.9 * chlorophyll * np.exp(-(((wl - 670) / 25) ** 2))
    shoulder = 1 / (1 + np.exp(-(wl - edge_nm) / 12))
    refl = (green - 0.07 * red_absorb) * (1 - shoulder) + (
        0.55 - 0.1 * np.exp(-(((wl - 970) / 18) ** 2)) * water
    ) * shoulder
    return np.clip(refl, 0.01, 0.95)


def synthetic_cube(chlorophyll=1.0, water=1.0, size=48, radius=15, bright=1.0):
    """Raw uint8 cube with a round plant on bare soil, plus the white and dark references."""
    h = w = size
    yy, xx = np.mgrid[:h, :w]
    plant = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < radius**2
    soil = np.full(len(WL), 0.18) + 0.0004 * (WL - 384)  # flat, rising: low NDVI
    refl = np.where(plant[..., None], leaf_spectrum(chlorophyll, water), soil) * bright
    white, dark = np.full(len(WL), WHITE), np.full(len(WL), DARK_LEVEL)
    raw = np.clip(np.rint(refl * (white * M - dark) + dark), 0, 255).astype(np.uint8)
    return raw, white, dark, plant


# ------------------------------------------------------------------ the maths
def test_flat_field_recovers_reflectance_and_flags_unusable_bands():
    raw, white, dark, plant = synthetic_cube()
    refl = sp.flat_field(raw, white, dark)
    expected = leaf_spectrum()
    assert np.abs(refl[plant].mean(axis=0) - expected).max() < 0.01  # within uint8 quantisation
    bad = sp.flat_field(raw[:2, :2], np.where(np.arange(len(WL)) == 3, 1.0, white), dark)  # W*m - D <= 0 at band 3
    assert np.isnan(bad[..., 3]).all() and np.isfinite(bad[..., 4]).all()


def test_band_is_a_mean_over_a_small_window_with_a_nearest_band_fallback():
    refl = np.arange(len(WL), dtype=float)[None, :]
    near_800 = np.abs(WL - 800) <= sp.BAND_HALF_WIDTH_NM
    assert sp.band(refl, WL, 800)[0] == refl[0, near_800].mean()
    assert sp.band(refl, WL, 800, half=0.01)[0] == refl[0, np.abs(WL - 800).argmin()]


def test_indices_match_their_definitions_on_a_known_spectrum():
    r = leaf_spectrum()[None, :]
    at = lambda nm: sp.band(r, WL, nm)[0]
    got = sp.index_values(r, WL)
    assert got["hsi_ndvi"][0] == pytest.approx((at(800) - at(670)) / (at(800) + at(670)))
    assert got["hsi_ci_rededge"][0] == pytest.approx(at(780) / at(720) - 1)
    assert got["hsi_sipi"][0] == pytest.approx((at(800) - at(445)) / (at(800) - at(680)))
    assert got["hsi_pri"][0] == pytest.approx((at(531) - at(570)) / (at(531) + at(570)))
    assert got["hsi_wbi"][0] == pytest.approx(at(900) / at(970))
    assert 690 < got["hsi_rep"][0] < 750  # the red edge sits between the red and the NIR plateau
    assert 0.5 < got["hsi_ndvi"][0] < 1 and got["hsi_wbi"][0] > 1  # healthy leaf: high NDVI, a real water dip


def test_catalogue_directions_agree_with_a_drying_leaf():
    """As chlorophyll and water fall, each stress trait moves the way its catalogue direction says."""
    healthy, dry = (sp.index_values(leaf_spectrum(c, w)[None, :], WL) for c, w in ((1.0, 1.0), (0.5, 0.3)))
    for column, label, _, direction, _ in sp.CATALOGUE:
        if direction:
            change = float(dry[column][0] - healthy[column][0]) if column in healthy else 0.0
            assert change * direction > 0, (
                f"{column} should move {'up' if direction > 0 else 'down'} under stress, moved {change:+.3f}"
            )


def test_ndvi_mask_picks_the_plant_and_ignores_soil():
    raw, white, dark, plant = synthetic_cube()
    mask = sp.ndvi_mask(sp.flat_field(raw, white, dark), WL)
    assert (mask & plant).sum() / plant.sum() > 0.98 and (mask & ~plant).sum() == 0


def test_process_run_reports_traits_and_quality_flags():
    raw, white, dark, plant = synthetic_cube()
    row, spectrum = sp.process_run(raw, white, dark, WL)
    assert row["hsi_mask_frac"] == pytest.approx(plant.mean(), abs=0.01) and row["hsi_qc_pass"]
    assert spectrum.shape == (300,) and np.abs(spectrum - leaf_spectrum()).max() < 0.02

    dark_run, *_ = synthetic_cube(bright=0.02)
    assert sp.process_run(dark_run, white, dark, WL)[0]["hsi_qc_dark"]
    blown = raw.copy()
    blown[plant[..., None] & (WL > 700)[None, None, :] & np.ones_like(raw, dtype=bool)] = 255  # the NIR bands clip
    flags = sp.process_run(blown, white, dark, WL)[0]
    assert flags["hsi_qc_overexposed"] and not flags["hsi_qc_pass"] and flags["hsi_mask_frac"] > 0.1
    empty = np.full_like(raw, 60)
    row_empty, spec_empty = sp.process_run(empty, white, dark, WL)
    assert row_empty["hsi_qc_mask_small"] and np.isnan(row_empty["hsi_ndvi"]) and np.isnan(spec_empty).all()


def test_white_reference_error_names_the_keys_it_found():
    assert sp.white_reference({"gray_reference": [1, 2]}).tolist() == [1, 2]
    with pytest.raises(KeyError, match="sensor_serial"):
        sp.white_reference({"sensor_serial": 7})


# ------------------------------------------------------------------ the folder tree
def make_run(base, bed, plant, day, hour, cube, white=None, meta_extra=None):
    folder = Path(base) / f"{bed}_{plant}" / day / f"{hour:02d}"
    folder.mkdir(parents=True)
    np.save(folder / "hsi_000.npy", cube)
    meta = {
        "white_reference": (white if white is not None else np.full(len(WL), WHITE)).tolist(),
        "wavelengths": WL.tolist(),
    }
    (folder / "hsi_000.json").write_text(json.dumps(meta | (meta_extra or {})))
    return folder


def test_discovery_maps_reruns_to_the_scheduled_slot_and_prefers_the_scheduled_folder(tmp_path):
    cube = synthetic_cube()[0][:8, :8]
    make_run(tmp_path, "W2", "R7", "2025_09_08", 15, cube)
    make_run(tmp_path, "W2", "R7", "2025_09_08", 16, cube)  # a re-run beside the scheduled run: ignored
    make_run(tmp_path, "W2", "R7", "2025_09_09", 16, cube)  # only the re-run exists: used as slot 15
    make_run(tmp_path, "W2", "R7", "2025_09_10", 6, cube)  # dark slot: not a hyperspectral slot
    make_run(tmp_path, "W2", "R7", "2025_10_30", 15, cube)  # outside the study window
    runs = sp.discover_runs(tmp_path, [("W2", "R7"), ("W1", "J5")])
    got = {(str(r.date.date()), r.hour, r.capture_hour) for r in runs.itertuples()}
    assert got == {("2025-09-08", 15, 15), ("2025-09-09", 15, 16)}


def test_discovery_finds_the_runs_one_folder_down(tmp_path):
    cube = synthetic_cube()[0][:8, :8]
    make_run(tmp_path / "hyperspectral", "W1", "A2", "2025_09_03", 12, cube)
    assert len(sp.discover_runs(tmp_path, [("W1", "A2")])) == 1


def test_plant_sets_follow_the_master_plan():
    paper = sp.paper_plants()
    assert len(paper) == 10 and ("W2", "R7") in paper and ("W3", "A2") not in paper
    core = sp.core_plants()
    assert set(paper) <= set(core) and len(core) in (29, 10)  # 29 once truth/plants has the 21 clip plants


# ------------------------------------------------------------------ end to end, through real JP2 files
@pytest.fixture
def derived(tmp_path, monkeypatch):
    monkeypatch.setattr("uiwue.core.DERIVED", tmp_path / "derived")
    monkeypatch.setattr(sp, "DERIVED", tmp_path / "derived")
    return tmp_path / "derived"


@pytest.fixture
def jp2_tree(tmp_path):
    """Three plants drying over four dates, written as lossless 300-band JP2 cubes next to their JSON."""
    import rasterio

    data = tmp_path / "hsi"
    for bed, plant, health in (
        ("W1", "J5", [1, 1, 1, 1]),
        ("W2", "R7", [1, 0.8, 0.5, 0.3]),
        ("W2", "A2", [1, 0.9, 0.6, 0.4]),
    ):
        for day, h in zip(("2025_09_02", "2025_09_05", "2025_09_08", "2025_09_11"), health):
            raw, white, _dark, _ = synthetic_cube(chlorophyll=h, water=h, size=40, radius=12)
            folder = make_run(data, bed, plant, day, 15, raw[:2, :2], white)
            (folder / "hsi_000.npy").unlink()
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with rasterio.open(
                    folder / "hsi_000.jp2",
                    "w",
                    driver="JP2OpenJPEG",
                    width=40,
                    height=40,
                    count=300,
                    dtype="uint8",
                    QUALITY="100",
                    REVERSIBLE="YES",
                ) as dst:
                    dst.write(np.moveaxis(raw, -1, 0))
    black = tmp_path / "black_reference.npy"
    np.save(black, np.full(len(WL), DARK_LEVEL, dtype=np.float32))
    return data, black


def test_command_line_writes_valid_tables_from_jp2_cubes(jp2_tree, derived):
    data, black = jp2_tree
    sp.main(
        ["--data-dir", str(data), "--black-reference", str(black), "--plants", "W1_J5,W2_R7,W2_A2", "--hours", "15"]
    )
    traits = pd.read_parquet(derived / "spectral" / "traits_runs.parquet")
    spectra = pd.read_parquet(derived / "spectral" / "spectra_runs.parquet")
    wl = pd.read_csv(derived / "spectral" / "wavelengths.csv").wavelength_nm
    assert len(traits) == len(spectra) == 12 and len(wl) == 300 and set(traits.hour) == {15}
    assert traits.hsi_qc_pass.all() and traits.hsi_mask_frac.between(0.2, 0.4).all()
    for table in ("traits_runs", "spectra_runs"):
        assert validate(derived / "spectral" / f"{table}.parquet") == []

    by_date = lambda col, bed, plant: (
        traits[(traits.bed == bed) & (traits.plant == plant)].sort_values("date")[col].to_numpy()
    )
    for col, direction in (("hsi_ndvi", -1), ("hsi_wbi", -1), ("hsi_sipi", 1), ("hsi_rep", -1)):
        stressed, control = by_date(col, "W2", "R7"), by_date(col, "W1", "J5")
        assert (stressed[-1] - stressed[0]) * direction > 0, col  # the drying plant moves toward stress
        assert abs(control[-1] - control[0]) < abs(stressed[-1] - stressed[0]), col  # the control stays put


def test_plant_profile_shows_the_pipelines_spectra(jp2_tree, derived, monkeypatch):
    """The page's spectrum panel reads exactly what the pipeline writes."""
    sys.path.insert(0, str(ROOT / "app"))
    pytest.importorskip("dash")
    import profile_views as pv

    monkeypatch.setattr(pv, "DERIVED", derived)
    data, black = jp2_tree
    sp.main(["--data-dir", str(data), "--black-reference", str(black), "--plants", "W2_R7", "--hours", "15"])
    spec, wl = pv.load_spectra()
    fig = pv.spectrum_figure(spec, wl, "W2", "R7", ["2025-09-02", "2025-09-11"])
    water = np.argmin(abs(wl - 970))
    assert len(fig.data) == 2 and fig.data[1].y[water] > fig.data[0].y[water]  # the dry plant's 970 nm dip is shallower
