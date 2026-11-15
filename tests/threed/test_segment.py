"""Vegetation mask (plan 4.4) and centre-plant isolation (plan 4.5)."""

import numpy as np
import pytest

from uiwue.threed import segment


def test_vegetation_rules():
    h = np.array([0.003, 0.008, 0.008, 0.020, 0.020])
    exg = np.array([20.0, 20.0, 0.0, np.nan, 0.0])
    has = np.isfinite(exg)
    light = segment.vegetation_mask(h, exg, has)
    assert light.tolist() == [False, True, False, True, False]  # no colour -> height only
    dark = segment.vegetation_mask(h, exg, has, dark=True)
    assert dark.tolist() == [False, False, False, True, True]
    assert segment.vegetation_mask(h).tolist() == dark.tolist()
    learned = segment.vegetation_mask(h, exg, has, learned=np.array([1, 1, 0, 0, 1], bool))
    assert learned.tolist() == [False, True, False, False, True]


def test_is_dark():
    assert segment.is_dark(5.0) and not segment.is_dark(30.0) and not segment.is_dark(np.nan)


def blob(cx, cy, r, h, n, seed):
    rng = np.random.default_rng(seed)
    rr = r * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([cx + rr * np.cos(a), cy + rr * np.sin(a), np.full(n, h)])


def test_isolation_keeps_centre_plant_only():
    plant = blob(0.0, 0.0, 0.04, 0.05, 3000, 0)
    neighbour = blob(0.075, 0.0, 0.02, 0.08, 800, 1)  # enters the cylinder, not attached
    loc = np.vstack([plant, neighbour])
    iso = segment.isolate_plant(loc, np.ones(len(loc), bool))
    assert not iso.no_plant and iso.n_clusters == 1
    assert iso.mask[:3000].all() and not iso.mask[3000:].any()
    assert iso.clipped_fraction == 0.0 and iso.centroid_offset < 0.005


def test_isolation_clips_large_canopy_and_flags_overlap():
    canopy = blob(0.0, 0.0, 0.12, 0.10, 20000, 2)
    iso = segment.isolate_plant(canopy, np.ones(len(canopy), bool))
    r = np.hypot(canopy[iso.mask, 0], canopy[iso.mask, 1])
    assert r.max() <= segment.CYLINDER_RADIUS
    assert iso.clipped_fraction > 0.10  # a full cylinder -> qc_overlap
    assert iso.outside_fraction == pytest.approx(1 - (0.085 / 0.12) ** 2, abs=0.03)


def test_isolation_no_plant_and_not_centred():
    off = blob(0.06, 0.0, 0.01, 0.05, 500, 3)
    assert segment.isolate_plant(off, np.ones(len(off), bool)).no_plant
    side = blob(0.045, 0.0, 0.025, 0.05, 2000, 4)  # reaches the centre zone, centroid 4.5 cm off
    iso = segment.isolate_plant(side, np.ones(len(side), bool))
    assert not iso.no_plant and iso.not_centred


def test_multi_frame_isolation_is_consistent():
    plant = blob(0.0, 0.0, 0.04, 0.05, 3000, 5)
    rng = np.random.default_rng(6)
    frames = [plant + rng.normal(0, 0.001, plant.shape) for _ in range(3)]
    isos = segment.isolate_plant_multi(frames, [np.ones(len(f), bool) for f in frames])
    assert all(i.mask.mean() > 0.98 for i in isos)


def test_refine_centre():
    plant = blob(0.01, -0.005, 0.02, 0.05, 2000, 7)
    da, db = segment.refine_centre(plant, np.ones(len(plant), bool))
    assert (da, db) == pytest.approx((0.01, -0.005), abs=0.002)
