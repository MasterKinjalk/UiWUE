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
    _, i2 = predict.run(_fake_cache(tmp_path / "ext2", ext), ckpt, "fusion_v2", "cpu", n_boot=50,
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


def test_late_negatives_are_deficit_runs_of_the_last_day_before_the_cutoff():
    from uiwue.fusion.v2 import late_negative

    k = core.normalize_keys(
        pd.DataFrame(
            dict(
                bed=["W2", "W2", "W2", "W1", "W3", "W3"],
                plant=["A2"] * 6,
                date=["2025-09-02", "2025-09-03", "2025-09-03", "2025-09-02", "2025-09-01", "2025-09-04"],
                hour=[15, 9, 12, 15, 15, 9],
            )
        )
    )
    assert late_negative(k).tolist() == [True, True, False, False, False, False]


def _fake_features(keys, seed=2):
    """Leaf temperature warms from day 2 in the deficit beds; air at 20 C; five frames per run."""
    rng = np.random.default_rng(seed)
    d = core.add_design(keys)
    stress = ((d["treatment"] == "deficit") & d["days_since_cutoff"].between(2, 12)).to_numpy()
    days = d["days_since_cutoff"].clip(0, 12).to_numpy()
    leaf = 24 + rng.normal(0, 0.5, len(keys)) + stress * (0.8 + 0.3 * days)
    f = keys[core.KEYS].copy()
    for sfx in ("", "_r160", "_r80"):
        f[f"thermal_leaf_t{sfx}"] = leaf + rng.normal(0, 0.1, len(keys))
        f[f"thermal_leaf_t_mad{sfx}"] = np.abs(rng.normal(0, 0.2, len(keys)))
        f[f"thermal_leaf_t_p10{sfx}"] = leaf - 1 + rng.normal(0, 0.1, len(keys))
        f[f"thermal_bg_t{sfx}"] = leaf + 3
        f[f"thermal_leaf_air_dt{sfx}"] = leaf - 20
    f["thermal_qc_pass"] = True
    f["thermal_n_frames_ok"] = 5
    return f


def test_hand_made_thermal_token_and_benchmark_on_new_plants(tmp_path, fake_derived):
    from uiwue.fusion import v2

    keys = _cv_keys()
    feats = tmp_path / "thermal_features.parquet"
    _fake_features(keys).to_parquet(feats)
    rs = v2.assemble(_fake_cache(tmp_path / "cache", keys), feats, exclude=(), labels=_labels(keys))
    assert ("tfeat", "full") in rs.emb and ("tfeat", "r80") in rs.emb and rs.dims["tfeat"] == len(v2.TFEAT)
    cfg = v2.make_config("tfeat_only", folds=3, epochs=20, patience=5, batch=128, d_model=16, layers=1, heads=2,
                         min_ref=2)  # fmt: skip
    assert v2.token_list(rs, cfg) == ["tfeat"]
    cv = v2.cross_validate(rs, cfg, "cpu", verbose=False)
    rows = v2.summarise(rs, cv, "tfeat_only", ("thermal",), n_boot=50)
    allg = next(r for r in rows if r["group"] == "all")
    assert allg["early_auc"] > 0.85 and "day_m1_auc" in allg
    # the benchmark fitted on some plants, applied to others; the reference uses the first set only
    pids = sorted(rs.keys["pid"].unique())
    test = {p for p in pids if p.split("_")[1].startswith(("A", "B"))}
    brows, curves = v2.benchmark_external(feats, set(pids) - test, test, n_boot=50)
    b = next(r for r in brows if r["group"] == "all")
    assert b["early_auc"] > 0.85 and b["n_deficit"] == 4 and b["n_control"] == 2 and len(curves) == 3


def test_control_reference_of_the_benchmark_uses_reference_plants_only():
    from uiwue.fusion.v2 import _control_ref

    f = core.normalize_keys(
        pd.DataFrame(
            dict(
                bed=["W1"] * 4 + ["W2"],
                plant=["A2", "A4", "A6", "A8", "A2"],
                date="2025-09-05",
                hour=12,
                thermal_leaf_t=[20.0, 21.0, 22.0, 30.0, 25.0],
                thermal_qc_pass=True,
            )
        )
    )
    ref = _control_ref(f, {"W1_A2", "W1_A4", "W1_A6", "W2_A2"}, min_controls=3)  # W1_A8 is not a reference plant
    assert ref.tolist() == [21.5, 21.0, 20.5, 21.0, 21.0]


def test_v2_cli_writes_valid_tables_and_scores_the_lockbox(tmp_path, fake_derived):
    """`python -m uiwue.fusion.v2 train` then `bench-lockbox`: the shared fusion tables get
    fusion_v2 rows, v2's run scores go to their own table, and everything validates."""
    from uiwue.core.validate import validate
    from uiwue.fusion import v2

    keys = _cv_keys()
    feats = tmp_path / "thermal_features.parquet"
    _fake_features(keys).to_parquet(feats)
    cache = _fake_cache(tmp_path / "cache", keys)
    out, runs = tmp_path / "derived" / "detect", tmp_path / "runs"
    common = ["--features", str(feats), "--n-boot", "20", "--out", str(out), "--min-ref", "2"]
    v2.main(["train", "--cache", str(cache), "--variant", "delta+ref+tfeat", "--epochs", "3", "--runs-dir", str(runs),
             *common])  # fmt: skip
    for name in ("fusion_benchmark", "fusion_auc_by_day", "fusion_v2_scores_runs"):
        assert validate(out / f"{name}.parquet") == [], name
    b = pd.read_parquet(out / "fusion_benchmark.parquet")
    assert set(b["model"]) == {"fusion_v2"} and not b["sensors"].str.contains(":").any()
    sr = pd.read_parquet(out / "fusion_v2_scores_runs.parquet")
    assert {"detect_fusion_v2_prob", "detect_fusion_v2_stage"} <= set(sr.columns)
    assert not (sr["bed"] + "_" + sr["plant"]).isin(v2.lockbox.LOCKBOX).any()  # lock-box plants are not trained on
    assert sorted(p.name for p in (runs / "fusion_v2").glob("fold*.pt")) == [f"fold{k}.pt" for k in range(5)]
    canary = tmp_path / "canary.parquet"
    v2.bench_lockbox(feats, 20, out, canary_path=canary)
    assert canary.exists()
    lb = pd.read_parquet(out / "fusion_lockbox_benchmark.parquet")
    assert set(lb["model"]) == {"logistic"} and set(lb["plants"]) == {"all", "no_clip"}
    assert validate(out / "fusion_lockbox_benchmark.parquet") == []


def test_causal_window_averages_the_same_plants_earlier_runs_of_the_day():
    from uiwue.fusion.v2 import causal_window

    k = pd.DataFrame(
        dict(
            pid=["a"] * 4 + ["b"] * 2,
            date=pd.to_datetime(["2025-09-01"] * 3 + ["2025-09-02"] + ["2025-09-01"] * 2),
            hour=[6, 9, 12, 6, 9, 12],
        )
    )
    x = np.arange(6, dtype=np.float32)[:, None]
    valid = np.array([1, 1, 1, 1, 0, 1], bool)
    # a: 6 h alone, 6-9 h, 6-12 h, next day alone; b: the invalid 9 h run is left out
    assert causal_window(k, x, valid, 6).ravel().tolist() == [0.0, 0.5, 1.0, 3.0, 0.0, 5.0]
    shuffled = np.array([5, 2, 0, 4, 1, 3])
    again = causal_window(k.iloc[shuffled].reset_index(drop=True), x[shuffled], valid[shuffled], 6)
    assert again.ravel().tolist() == causal_window(k, x, valid, 6).ravel()[shuffled].tolist()


def test_a_thermal_only_v2_model_trains_scores_its_oof_subsets_and_new_plants(tmp_path, fake_derived, monkeypatch):
    """A v2 model without image encoders (tfeat_only) has only thermal setups: the shared tables,
    the out-of-fold subsets and the lock-box path must all score just its own camera."""
    from uiwue.core.validate import validate
    from uiwue.fusion import predict, v2

    keys = _cv_keys()
    feats = tmp_path / "thermal_features.parquet"
    _fake_features(keys).to_parquet(feats)
    cache = _fake_cache(tmp_path / "cache", keys)
    out, runs = tmp_path / "derived" / "detect", tmp_path / "runs"
    v2.main(["train", "--cache", str(cache), "--variant", "tfeat_only", "--name", "fusion_v2", "--epochs", "3",
             "--runs-dir", str(runs), "--features", str(feats), "--n-boot", "20", "--out", str(out), "--min-ref", "2"])  # fmt: skip
    b = pd.read_parquet(out / "fusion_benchmark.parquet")
    assert set(b["sensors"]) == {"thermal", "thermal_r160", "thermal_r80"}
    assert validate(out / "fusion_v2_scores_runs.parquet") == []
    ck = torch.load(runs / "fusion_v2" / "fold0.pt", weights_only=True)
    assert ck["tokens"] == ["tfeat"] and ck["sensors"] == ["thermal"]
    monkeypatch.setattr(predict, "clip_plants", lambda: set())
    sub = predict.oof_tables(runs / "fusion_v2", "fusion_v2", n_boot=20)["fusion_cv_subsets_benchmark"]
    assert set(sub["sensors"]) <= {"thermal", "thermal_r160", "thermal_r80", "size"}
    lock = [p for p in v2.lockbox.LOCKBOX if p in set(keys["bed"] + "_" + keys["plant"])]
    tables, info = predict.run(cache, runs / "fusion_v2", "fusion_v2", "cpu", n_boot=20, labels=_labels(keys),
                               features=feats, only=lock, summary_path=tmp_path / "lock.parquet")  # fmt: skip
    assert info["n_plants"] == len(lock) and (tmp_path / "lock.parquet").exists()
    assert set(tables["fusion_external_benchmark"]["sensors"]) <= {"thermal", "thermal_r160", "thermal_r80"}
