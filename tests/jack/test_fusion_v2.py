"""Fusion v2: control-referenced tokens, hand-made thermal features, the v2 cross-validation,
and v2 fold checkpoints scoring new plants. Tiny synthetic data on CPU, as test_fusion.py."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from test_fusion import _fake_cache
from test_fusion import _keys as _cv_keys
from test_fusion_predict import _keys as _ext_keys

from uiwue import core
from uiwue.fusion.labels import run_labels
from uiwue.fusion.train import RunSet


def _labels(keys):
    return run_labels(keys, turgor=pd.DataFrame(), plants=pd.DataFrame())


def test_loo_median_matches_brute_force():
    from uiwue.fusion.v2 import _loo_median

    rng = np.random.default_rng(0)
    for n in (2, 3, 6, 7):
        x = rng.normal(size=(n, 5)).astype(np.float32)
        x[0, 0] = x[1, 0]  # a tie
        brute = np.stack([np.median(np.delete(x, i, 0), 0) for i in range(n)])
        assert np.allclose(_loo_median(x), brute)


def test_reference_compares_with_training_controls_only():
    from uiwue.fusion import v2

    # one run, 5 controls (4 in the reference, 1 held out) and 2 deficit plants; a second run
    # with only 2 reference controls has no reference
    keys = pd.DataFrame(
        dict(bed=["W1"] * 5 + ["W2"] * 2 + ["W1"] * 2, plant=[f"A{i}" for i in range(1, 8)] + ["A1", "A2"])
    )
    keys["date"] = pd.Timestamp("2025-09-05")
    keys.loc[7:, "date"] = pd.Timestamp("2025-09-06")
    keys["hour"] = 12
    keys["pid"] = keys["bed"] + "_" + keys["plant"]
    d = np.arange(9, dtype=np.float32)[:, None] * np.array([[1.0, -2.0]], np.float32)
    rs = RunSet(keys, ["thermal"], {"thermal": 2}, {("thermal", "full"): d}, {("thermal", "full"): 0 * d},
                {("thermal", "full"): np.ones(9, bool)})  # fmt: skip
    rs.ref, rs.has_ref = {}, {}
    ref_rows = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1], bool)
    medians = v2.reference(rs, ref_rows, min_ref=3)
    out, ok = rs.ref[("thermal", "full")], rs.has_ref[("thermal", "full")]
    med = np.median(d[:4], 0)
    assert np.allclose(out[4:7], d[4:7] - med)  # held-out control and deficit plants: all reference controls
    for i in range(4):  # reference controls: the median of the other reference controls
        assert np.allclose(out[i], d[i] - np.median(np.delete(d[:4], i, 0), 0))
    assert ok[:7].all() and not ok[7:].any()  # 2 controls < min_ref: no reference in that run
    assert list(medians[("thermal", "full")]) == [0] and np.allclose(medians[("thermal", "full")][0], med)


def test_thermal_features_reproduce_the_thermal_module(tmp_path):
    import tifffile

    from uiwue.fusion import features
    from uiwue.thermal import run_row

    rng = np.random.default_rng(3)
    run = tmp_path / "thermal" / "W2_C4" / "2025_09_05" / "12"
    run.mkdir(parents=True)
    for k in range(5):
        tifffile.imwrite(run / f"thermal_00{k}.tif", (25 + 3 * rng.random((48, 64))).astype(np.float32))
    got = features.run_features(run, sorted(run.glob("thermal_*.tif")))
    want = run_row(run, sorted(run.glob("thermal_*.tif")))
    for c in ("thermal_leaf_t", "thermal_leaf_t_mad", "thermal_leaf_t_p10", "thermal_bg_t", "thermal_leaf_t_r160",
              "thermal_leaf_t_r80"):  # fmt: skip
        assert np.isclose(got[c], want[c]), c
    df = features.plant_features(run.parent.parent)
    assert len(df) == 1 and df.loc[0, "hour"] == 12 and df.loc[0, "plant"] == "C4"


def test_v2_removes_a_constant_bed_offset_and_scores_new_plants(tmp_path, fake_derived, monkeypatch):
    """thermal warms in the deficit beds from day 2; RGB differs between beds from the start.
    v1's absolute token lets RGB separate the beds (its own placebo score is high); v2's
    baseline and control-referenced parts cancel the constant offset."""
    from uiwue.core.validate import validate
    from uiwue.fusion import predict, train, v2

    keys = _cv_keys()
    rs = v2.assemble(_fake_cache(tmp_path / "cache", keys), None, exclude=(), labels=_labels(keys))
    assert {"rgb", "thermal"} <= set(rs.sensors) and ("thermal", "r80") in rs.emb
    cfg = v2.make_config("delta+ref", folds=3, epochs=30, patience=5, batch=128, n_boot=50, d_model=32, layers=1,
                         heads=2, min_ref=2)  # fmt: skip
    ckpt = tmp_path / "runs" / "fusion_v2"
    cv = v2.cross_validate(rs, cfg, "cpu", ckpt_dir=ckpt, verbose=False)
    named = v2.shared_setups(cv)
    assert not any(":" in k for k in named["setups"]) and "thermal_r80" in named["setups"]
    res = train.results(rs, named, "fusion_v2", 50, groups=("all",))
    b = res["fusion_benchmark"].set_index("sensors")
    assert b.loc["thermal", "early_auc"] > 0.85 and b.loc["thermal", "onset_day"] <= 5
    assert b.loc["rgb", "placebo_auc_main"] < 0.8 and b.loc["rgb", "early_auc"] < 0.8
    out = tmp_path / "derived" / "detect"
    train.write(res, "fusion_v2", out)
    for name in ("fusion_benchmark", "fusion_auc_by_day"):
        assert validate(out / f"{name}.parquet") == []

    ck = torch.load(ckpt / "fold0.pt", weights_only=True)
    assert ck["design"] == "v2" and ck["tokens"] == ["rgb", "thermal"] and tuple(ck["config"]["parts"]) == cfg.parts
    assert set(ck["train_pids"]).isdisjoint(ck["test_pids"])
    assert torch.load(ckpt / "placebo_fold0.pt", weights_only=True)["target"] == "y_bed"

    # new plants: referenced to each fold's training controls, through predict.run
    ext = _ext_keys("GHIJKL")
    monkeypatch.setattr(predict, "clip_plants", lambda: {"W2_G2"})
    tables, info = predict.run(_fake_cache(tmp_path / "ext", ext), ckpt, "fusion_v2", "cpu", n_boot=50,
                               labels=_labels(ext), features=None)  # fmt: skip
    assert info["design"] == "v2" and info["n_plants"] == 18
    eb = tables["fusion_external_benchmark"]
    allp = eb[(eb["plants"] == "all") & (eb["group"] == "all")].set_index("sensors")
    assert allp.loc["thermal", "early_auc"] > 0.85
    assert set(eb["plants"]) >= {"all", "no_clip"}
    # only some plants: the lock-box path
    t2, i2 = predict.run(_fake_cache(tmp_path / "ext2", ext), ckpt, "fusion_v2", "cpu", n_boot=50,
                         labels=_labels(ext), features=None, only=["W2_G2", "W1_G2", "W3_H3", "W1_H3"])  # fmt: skip
    assert i2["n_plants"] == 4


def test_ablation_variants_build():
    from uiwue.fusion import v2

    for name in v2.VARIANTS:
        cfg = v2.make_config(name)
        assert set(cfg.parts) <= {"now", "delta", "ref"}
    assert v2.make_config("v1").parts == ("now", "delta") and not v2.make_config("v1").tfeat
    m = v2.FusionV2({"thermal": 8, "tfeat": 4, "env": 4}, d_model=16, layers=0, heads=2)
    x = {"thermal": torch.randn(3, 8), "tfeat": torch.randn(3, 4), "env": torch.randn(3, 4)}
    present = torch.tensor([[True, False, True], [False, True, True], [True, True, False]])
    out = m.eval()(x, present)
    assert out["logit"].shape == (3,) and torch.isfinite(out["bed_logit"]).all()
    # absent tokens do not change the output
    x2 = dict(x, tfeat=torch.randn(3, 4) * 100)
    assert torch.allclose(m(x2, present)["logit"][0], out["logit"][0])
    assert core.SLOTS  # keeps the core import used
