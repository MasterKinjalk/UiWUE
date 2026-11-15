"""Learned plant mask (``--plant-model``): cache, lifting, centre selection, fallbacks, the fast
Mask2Former post-processing, and the plumbing through ``run.process_run``."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uiwue.threed import plantmask, run, schema, segment
from uiwue.threed.cloud import Cloud
from uiwue.threed.io import Run

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUN = ("W1_J5", "2025_09_02", "15")
HAVE_DATA = all((DATA / m / RUN[0] / RUN[1] / RUN[2]).is_dir() for m in ("depth", "rgb"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="raw ADA data not on disk")


def test_mask_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    masks = rng.random((3, 7, 13)) > 0.5  # width not a multiple of 8
    f = tmp_path / "a" / "m.npz"
    plantmask.save_masks(f, masks, [0.9, 0.8, 0.7], "model")
    back, scores = plantmask.load_masks(f)
    assert back.dtype == bool and (back == masks).all()
    assert scores == pytest.approx([0.9, 0.8, 0.7])
    plantmask.save_masks(f, np.zeros((0, 7, 13), bool), [])
    assert plantmask.load_masks(f)[0].shape == (0, 7, 13)


def cloud_at(uv, has=None, brightness=50.0):
    n = len(uv)
    has = np.ones(n, bool) if has is None else np.asarray(has, bool)
    return Cloud(
        xyz=np.zeros((n, 3)),
        rc=np.zeros((n, 2), np.int32),
        rgb=np.zeros((n, 3), np.uint8),
        has_rgb=has,
        rgb_uv=np.asarray(uv, np.float32),
        brightness=brightness,
    )


def test_lift_reads_the_half_resolution_mask_at_rgb_uv():
    masks = np.zeros((2, 540, 960), bool)
    masks[0, 100, 200] = True  # half-res pixel (200, 100) = full-res pixels 400..401, 200..201
    masks[1, 539, 959] = True
    c = cloud_at([[400.6, 200.6], [400.6, 200.6], [1919.0, 1079.0], [10.0, 10.0]], has=[1, 0, 1, 1])
    m = plantmask.lift(c, masks)
    assert m.shape == (2, 4)
    assert m[0].tolist() == [True, False, False, False]  # point 1 has no colour
    assert m[1].tolist() == [False, False, True, False]


def ring(n, r0, r1, h, seed, cx=0.0):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(r0**2, r1**2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(a), r * np.sin(a), np.full(n, h)])


def test_select_centre_and_merge_fragments():
    centre = ring(400, 0.0, 0.03, 0.05, 0)  # heart of the plant, at the centre
    fragment = ring(400, 0.03, 0.08, 0.06, 1)  # outer leaves, split into another instance
    neighbour = ring(400, 0.0, 0.07, 0.07, 2, cx=0.17)  # the next pot, leaves reaching over
    soil = ring(200, 0.0, 0.02, 0.0, 3)  # under the centre: below H_MIN, never counts
    local = np.vstack([centre, fragment, neighbour, soil])
    members = np.zeros((4, len(local)), bool)
    members[0, :400] = True
    members[1, 400:800] = True
    members[2, 800:1200] = True
    members[3, 1200:] = True  # an "instance" of soil only
    i, n = plantmask.select_centre(members, local)
    assert (i, n) == (0, 400)
    mask, taken = plantmask.merge_fragments(members, local, i)
    assert taken.tolist() == [True, True, False, False]
    assert mask[:800].all() and not mask[800:].any()
    assert plantmask.select_centre(members[2:], local) == (-1, 0)  # nothing reaches the centre


def test_plant_masks_fallbacks(tmp_path):
    r = Run("W1_J5", "2025_09_02", "15")
    masks = np.zeros((1, 540, 960), bool)
    masks[0, 260:280, 470:490] = True
    uv = np.full((300, 2), [960.0, 540.0], np.float32)  # every point on the instance
    local = np.column_stack([np.zeros(300), np.zeros(300), np.full(300, 0.05)])
    fn = plantmask.PlantMasks(tmp_path)

    def source(c, k=7):
        return fn(r, k, c, local)[1]["source"]

    assert source(cloud_at(uv)) == plantmask.SOURCE_NO_CACHE
    plantmask.save_masks(plantmask.mask_path(tmp_path, r, 7), masks, [0.9])
    plantmask.save_masks(plantmask.mask_path(tmp_path, r, 4), masks[:0], [])
    m, info = fn(r, 7, cloud_at(uv), local)
    assert info["source"] == plantmask.SOURCE_MODEL and m.all() and info["n_instances"] == 1
    assert info["score"] == pytest.approx(0.9)
    assert source(cloud_at(uv, brightness=4.4)) == plantmask.SOURCE_DARK
    assert source(cloud_at(uv, has=np.zeros(300))) == plantmask.SOURCE_NO_COLOUR
    assert source(cloud_at(uv), k=4) == plantmask.SOURCE_NO_INSTANCE  # cached, empty
    far = cloud_at(np.full((300, 2), [100.0, 100.0], np.float32))
    assert source(far) == plantmask.SOURCE_NO_INSTANCE  # an instance, but not at the centre
    # two plants' leaves over the pot centre; the rival has most of its points outside the
    # cylinder (its own pot), so it is not merged
    two = np.zeros((2, 540, 960), bool)
    two[0, 260:280, 470:490] = True
    two[1, 100:110, 100:110] = True
    plantmask.save_masks(plantmask.mask_path(tmp_path, r, 10), two, [0.9, 0.8])

    def scene(n_rival_near):
        uv = [[960.0, 540.0]] * 200 + [[210.0, 210.0]] * (n_rival_near + 400)
        loc = [[0.0, 0.0, 0.05]] * 200 + [[0.01, 0.0, 0.05]] * n_rival_near + [[0.15, 0.0, 0.05]] * 400
        return cloud_at(np.array(uv, np.float32)), np.array(loc)

    m, info = fn(r, 10, *scene(90))  # 90 < 0.8 * 200 points near the axis
    assert info["source"] == plantmask.SOURCE_MODEL and m[:200].all() and not m[200:].any()
    m, info = fn(r, 10, *scene(180))  # 180 >= 0.8 * 200
    assert m is None and info["source"] == plantmask.SOURCE_AMBIGUOUS


def test_vote_keeps_the_frames_consistent():
    rng = np.random.default_rng(0)
    plant = ring(3000, 0.0, 0.04, 0.05, 4)
    neighbour = ring(1500, 0.0, 0.03, 0.05, 5, cx=0.065)  # touches the plant
    loc = np.vstack([plant, neighbour])
    frames = [loc + rng.normal(0, 0.0005, loc.shape) for _ in range(3)]
    is_nb = np.arange(len(loc)) >= 3000
    far_nb = is_nb & (loc[:, 0] > 0.05)
    cands = [np.ones(len(loc), bool)] * 3
    # without votes the touching neighbour is one cluster with the plant
    plain = segment.isolate_plant_multi(frames, cands)
    assert plain[0].mask[far_nb & (np.hypot(loc[:, 0], loc[:, 1]) <= 0.085)].mean() > 0.9
    # frame 0's mask calls the neighbour plant, frames 1 and 2 do not: 1 of 3 -> dropped everywhere
    votes = [np.where(is_nb, 1.0 if i == 0 else 0.0, 1.0) for i in range(3)]
    isos = segment.isolate_plant_multi(frames, cands, votes=votes)
    for iso in isos:
        assert iso.mask[:3000].mean() > 0.95
        assert iso.mask[far_nb].mean() < 0.05
    # points without a vote (no colour) take their voxel's label through connectivity
    votes = [np.where(is_nb, 0.0, np.where(np.arange(len(loc)) % 4 == 0, np.nan, 1.0)) for _ in range(3)]
    iso = segment.isolate_plant_multi(frames, cands, votes=votes)[0]
    assert iso.mask[:3000][np.arange(3000) % 4 == 0].mean() > 0.95


def test_instance_masks_matches_transformers_post_processing():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    from uiwue.ml.leafseg.infer import instance_masks

    g = torch.Generator().manual_seed(0)
    n_q = 40
    cls = torch.randn(1, n_q, 3, generator=g)
    cls[0, :12, 0] += 6.0  # confident plants
    cls[0, 12:20, 1] += 6.0  # confident leaves
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 24), torch.linspace(-1, 1, 40), indexing="ij")
    cx, cy = torch.rand(n_q, generator=g) * 2 - 1, torch.rand(n_q, generator=g) * 2 - 1
    logits = 4 * (0.4 - torch.hypot(xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]))
    out = SimpleNamespace(class_queries_logits=cls, masks_queries_logits=logits[None])
    ref = tr.Mask2FormerImageProcessor().post_process_instance_segmentation(
        out, threshold=0.5, target_sizes=[(54, 96)], return_binary_maps=True
    )[0]
    ref_scores = np.array([s["score"] for s in ref["segments_info"]])
    ref_labels = np.array([s["label_id"] for s in ref["segments_info"]])
    ref_masks = ref["segmentation"].numpy().astype(bool)
    masks, scores, labels = instance_masks(out, (54, 96), 0.5)
    order = np.argsort(-ref_scores, kind="stable")
    assert len(masks) == len(ref_masks) >= 10
    assert scores == pytest.approx(ref_scores[order], abs=1e-5)
    assert (labels == ref_labels[order]).all() and (masks == ref_masks[order]).all()
    plants = instance_masks(out, (54, 96), 0.5, label=0)
    assert (plants[2] == 0).all() and len(plants[0]) == int((ref_labels == 0).sum())


def test_cli_refuses_the_contract_directory():
    with pytest.raises(SystemExit):
        run.main(["--plant-model", "x", "--out", str(schema.DERIVED)])
    with pytest.raises(SystemExit):
        run.main(["--plant-model", "x", "--depth-model", "y"])  # both need an explicit --out


@needs_data
def test_process_run_with_learned_masks(tmp_path):
    r = Run(*RUN)
    raw = run.process_run(r, DATA, None, want_cloud=False)
    plan = run.process_run(r, DATA, None, plan_only=True)
    assert not plan["no_pose"]
    assert raw["run"]["mask_source"] == "rule" and {f["mask_source"] for f in raw["frames"]} <= {"", "rule"}

    # no instance anywhere -> every frame falls back, and the traits equal the rule pipeline
    for k in plan["build"]:
        plantmask.save_masks(plantmask.mask_path(tmp_path, r, k), np.zeros((0, 540, 960), bool), [])
    res = run.process_run(r, DATA, None, want_cloud=False, plant_mask_fn=plantmask.PlantMasks(tmp_path))
    built = [f for f in res["frames"] if f["built"]]
    assert {f["mask_source"] for f in built} == {plantmask.SOURCE_NO_INSTANCE}
    assert res["run"]["mask_source"] == "rule" and res["run"]["mask_model_frac"] == 0
    for t in ("height_p75", "projected_area", "n_points", "clipped_fraction"):
        assert res["run"][t] == raw["run"][t], t

    # one instance over the whole frame -> the model path, a superset of the ExG vegetation
    full = np.ones((1, 540, 960), bool)
    for k in plan["build"]:
        plantmask.save_masks(plantmask.mask_path(tmp_path, r, k), full, [0.99])
    res = run.process_run(r, DATA, None, want_cloud=False, plant_mask_fn=plantmask.PlantMasks(tmp_path))
    assert res["run"]["mask_source"] == "model" and res["run"]["mask_model_frac"] == 1.0
    assert {f["mask_source"] for f in res["frames"] if f["built"]} == {plantmask.SOURCE_MODEL}
    assert res["run"]["n_points"] >= raw["run"]["n_points"]
    assert 50 < res["run"]["height_p75"] < 250


@needs_data
def test_precompute_masks_caches_every_built_frame(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from uiwue.ml.leafseg import infer

    calls = []

    class FakeSegmenter:
        device, threshold = "cpu", 0.5

        def __init__(self, model):
            pass

        def forward(self, rgb, scale, undistort):
            assert rgb.shape == (1080, 1920, 3) and scale == plantmask.MODEL_SCALE and not undistort
            calls.append(1)

    def fake_masks(out, size, threshold, label=None):
        m = np.zeros((1, *size), bool)
        m[0, 200:300, 400:500] = True
        return m, np.array([0.9], np.float32), np.array([0])

    monkeypatch.setattr(infer, "LeafSegmenter", FakeSegmenter)
    monkeypatch.setattr(infer, "instance_masks", fake_masks)
    r = Run(*RUN)
    st = plantmask.precompute_masks([r], {}, DATA, tmp_path, "fake", jobs=1)
    build = run.process_run(r, DATA, None, plan_only=True)["build"]
    assert len(calls) == len(build) and st["missing"] == 0
    for k in build:
        m, s = plantmask.load_masks(plantmask.mask_path(tmp_path, r, k))
        assert m.shape == (1, *plantmask.MASK_SIZE) and m[0, 250, 450] and s[0] == pytest.approx(0.9)
    assert plantmask.precompute_masks([r], {}, DATA, tmp_path, "fake", jobs=1)["frames"] == 0  # resumes
    assert len(calls) == len(build)
