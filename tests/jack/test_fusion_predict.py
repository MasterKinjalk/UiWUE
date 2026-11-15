"""External test of the fusion model: fold checkpoints score plants they never saw.

Tiny random encoders and fake embeddings (no downloads), as test_fusion.py.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchgeo")

from test_fusion import _fake_cache

from uiwue import core
from uiwue.fusion.labels import run_labels


def _keys(letters: str, hours=(6, 12, 15), days=26, start=date(2025, 8, 24)):
    rows = []
    for bed in ("W1", "W2", "W3"):
        for i, c in enumerate(letters):
            for d in (start + timedelta(k) for k in range(days)):
                for h in hours:
                    rows.append(dict(bed=bed, plant=f"{c}{2 + i % 2}", date=d, hour=h))
    return core.normalize_keys(pd.DataFrame(rows))


def _labels(keys):
    return run_labels(keys, turgor=pd.DataFrame(), plants=pd.DataFrame())


def _size_traits(keys, seed=0):
    """depth_projected_area: every plant grows 3 units a day; deficit plants stop on day 4."""
    rng = np.random.default_rng(seed)
    d = core.add_design(keys)
    day = (d["date"] - d["date"].min()).dt.days.to_numpy()
    stop = (pd.Timestamp(core.CUTOFF) - d["date"].min()).days + 4
    growth = np.where(d["treatment"] == "deficit", np.minimum(day, stop), day)
    plant, n = pd.factorize(d["bed"] + d["plant"])[0], (d["bed"] + d["plant"]).nunique()
    level = rng.normal(0, 5, n)[plant]
    return keys.assign(depth_projected_area=100 + level + 3.0 * growth + rng.normal(0, 1.5, len(keys)))


def test_fold_checkpoints_score_external_plants(tmp_path, fake_derived, monkeypatch):
    from uiwue.core.validate import validate
    from uiwue.fusion import predict, train

    train_keys = _keys("ABCDEF")
    rs = train.assemble(_fake_cache(tmp_path / "cache", train_keys), labels=_labels(train_keys))
    cfg = train.Config(folds=3, epochs=30, patience=5, batch=128, n_boot=50, d_model=32, layers=1, heads=2)
    ckpt = tmp_path / "runs" / "fusion"
    train.cross_validate(rs, cfg, "cpu", ckpt_dir=ckpt)
    assert sorted(p.name for p in ckpt.glob("*.pt")) == [
        *(f"fold{k}.pt" for k in range(3)),
        *(f"placebo_fold{k}.pt" for k in range(3)),
    ]
    ck = torch.load(ckpt / "fold0.pt", weights_only=True)
    assert (ck["config"]["layers"], ck["config"]["heads"]) == (1, 2)
    assert set(ck["train_pids"]).isdisjoint(ck["test_pids"]) and len(ck["train_pids"]) + len(ck["test_pids"]) == 18
    assert torch.load(ckpt / "placebo_fold0.pt", weights_only=True)["target"] == "y_bed"

    # 18 new plants, and a second cache with the three A2 plants, which the folds trained on:
    # those must be dropped.
    ext_keys, seen_keys = _keys("GHIJKL"), _keys("A")
    caches = [_fake_cache(tmp_path / "ext", ext_keys), _fake_cache(tmp_path / "ext_seen", seen_keys)]
    monkeypatch.setattr(predict, "clip_plants", lambda: {"W2_G2", "W3_G2", "W1_G2"})
    traits = _size_traits(ext_keys[~ext_keys["plant"].isin(["L3"])])  # one plant per bed has no size
    labels = _labels(pd.concat([ext_keys, seen_keys]))
    tables, info = predict.run(caches, ckpt, "fusion", "cpu", n_boot=50, traits=traits, labels=labels)
    assert info["excluded_training_plants"] == ["W1_A2", "W2_A2", "W3_A2"] and info["n_plants"] == 18
    assert info["clip_plants"] == ["W1_G2", "W2_G2", "W3_G2"]

    b = tables["fusion_external_benchmark"]
    allp = b[(b["plants"] == "all") & (b["group"] == "all")].set_index("sensors")
    assert allp.loc["thermal", "early_auc"] > 0.85 and not allp.loc["thermal", "bed_effect"]
    assert allp.loc["rgb", "bed_effect"]  # the bed offset is still exposed on new plants
    assert set(b["plants"]) == {"all", "no_clip", "with_size"}
    assert (b.loc[b["plants"] == "no_clip", "n_plants"] == 15).all()
    assert (b.loc[b["plants"] == "with_size", "n_plants"] == 15).all()
    size = b[b["is_size_baseline"]]
    assert set(size["plants"]) == {"with_size"} and (size["full_auc"] > 0.9).all()  # growth stops on day 4
    cv_cols = {"early_auc", "early_ci_low", "full_auc", "onset_day", "placebo_auc", "placebo_auc_main", "bed_effect"}
    assert cv_cols | {"setup", "cost_rank", "is_size_baseline", "n_deficit", "n_control"} <= set(b.columns)

    out = tmp_path / "derived" / "detect"
    predict.write(tables, "fusion", out)
    runs_path = out / "fusion_external_scores_runs.parquet"
    assert validate(runs_path) == []
    sr = pd.read_parquet(runs_path)
    assert {"detect_fusion_ext_prob", "detect_fusion_ext_stage", "detect_fusion_ext_turgor"} <= set(sr.columns)
    assert not (sr["plant"] == "A2").any()
    for name in ("fusion_external_benchmark", "fusion_external_auc_by_day"):
        assert validate(out / f"{name}.parquet") == []
    # a LoRA stage's rows are added beside stage 1's; the size rows are kept once
    predict.write(
        {"fusion_external_benchmark": b[b["model"] == "fusion"].assign(model="fusion+lora_x")}, "fusion+lora_x", out
    )
    again = pd.read_parquet(out / "fusion_external_benchmark.parquet")
    assert set(again["model"]) == {"fusion", "fusion+lora_x", "size_only"}
    assert len(again[again["model"] == "size_only"]) == len(size)


def test_assemble_concatenates_caches_embedded_in_batches(tmp_path):
    from uiwue.fusion import train

    keys = _keys("GH")
    first = keys["plant"] == "G2"
    _fake_cache(tmp_path / "a", keys[first])
    _fake_cache(tmp_path / "b", keys[~first])
    rs = train.assemble([tmp_path / "a", tmp_path / "b"], labels=_labels(keys))
    assert rs.keys["pid"].nunique() == 6 and len(rs.keys) == len(keys)
    assert rs.has[("thermal", "full")].all() and rs.sensors == ["rgb", "thermal"]


def _tiny():
    from uiwue.fusion.encoders import tiny_dofa

    torch.manual_seed(0)  # the "pretrained" weights: the same in training and at prediction
    return tiny_dofa("rgb")


def _write_rgb(root, keys, rng):
    from PIL import Image

    for r in keys.itertuples():
        d = root / "rgb" / f"{r.bed}_{r.plant}" / f"{r.date:%Y_%m_%d}" / f"{r.hour:02d}"
        d.mkdir(parents=True)
        g = 120 if r.bed == "W1" or r.date < pd.Timestamp("2025-09-04") else 60
        Image.fromarray(np.clip(rng.normal(g, 20, (64, 96, 3)), 0, 255).astype(np.uint8)).save(d / "rgb_007.jpg")


def test_lora_folds_reembed_external_plants_with_their_adapters(tmp_path):
    from uiwue.fusion import inputs, predict, train
    from uiwue.fusion.embed import embed_sensor, save_table

    window = dict(hours=(12,), days=10, start=date(2025, 8, 30))
    train_keys = _keys("AB", **window)
    ext_keys = _keys("GH", **window)
    train_keys, ext_keys = (k[k["bed"] != "W3"].reset_index(drop=True) for k in (train_keys, ext_keys))
    rng = np.random.default_rng(0)
    _write_rgb(tmp_path / "data", pd.concat([train_keys, ext_keys]), rng)
    runs = inputs.list_runs(tmp_path / "data", "rgb")
    pid = runs["bed"] + "_" + runs["plant"]
    is_ext = pid.isin(ext_keys["bed"] + "_" + ext_keys["plant"]).to_numpy()
    emb, ok, _ = embed_sensor(_tiny(), runs, ["full"], "cpu", batch=8)
    assert ok.all()
    save_table(tmp_path / "cache" / "emb_rgb_full.npz", runs[~is_ext], emb["full"][~is_ext])
    save_table(tmp_path / "ext" / "emb_rgb_full.npz", runs[is_ext], emb["full"][is_ext])

    rs = train.assemble(tmp_path / "cache", labels=_labels(train_keys))
    cfg = train.Config(folds=2, epochs=2, lora_epochs=1, lora_batch=8, workers=0, n_boot=0, d_model=16, lora_blocks=2)
    stage = train.LoRAStage(rs, "rgb", tmp_path / "data", cfg)
    stage.make_encoder = _tiny
    ckpt = tmp_path / "runs" / "fusion+lora_rgb"
    train.cross_validate_lora(rs, stage, cfg, "cpu", ckpt_dir=ckpt)
    ck = torch.load(ckpt / "fold1.pt", weights_only=True)
    assert ck["lora_sensor"] == "rgb" and len(ck["lora"]) > 0

    # the saved adapters rebuild the last fold's tuned encoder exactly
    rebuilt = predict._tuned_encoder(ck, "cpu", _tiny)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(rebuilt(x, inputs.RGB_WAVELENGTHS), stage.encoder(x, inputs.RGB_WAVELENGTHS), atol=1e-5)

    tables, info = predict.run(
        tmp_path / "ext", ckpt, "fusion+lora_rgb", "cpu", n_boot=0, data_dir=tmp_path / "data",
        labels=_labels(ext_keys), make_encoder=_tiny, workers=0,
    )  # fmt: skip
    assert info["n_plants"] == 4 and not info["excluded_training_plants"]
    assert tables["fusion_external_scores_runs"]["detect_fusion_ext_prob"].notna().mean() > 0.9
    assert not tables["fusion_external_benchmark"].empty
