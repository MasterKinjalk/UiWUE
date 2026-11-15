"""Fusion track: encoders, LoRA, labels, model, and both training stages end to end.

Uses tiny random encoders (no downloads) so it runs on CPU in seconds.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchgeo")

from uiwue import core
from uiwue.fusion import inputs
from uiwue.fusion.encoders import ImageBindEncoder, LoRALinear, add_lora, lora_parameters, tiny_dofa
from uiwue.fusion.fusion import FusionModel
from uiwue.fusion.labels import run_labels


def test_lora_starts_as_the_base_model_and_only_adapters_train():
    enc = tiny_dofa().eval()
    x = torch.randn(2, 3, 224, 224)
    before = enc(x, inputs.RGB_WAVELENGTHS)
    n = add_lora(enc, last_n=2, rank=4)
    assert n >= 2 * 4  # qkv, proj, fc1, fc2 in each of the last two blocks
    assert torch.allclose(enc(x, inputs.RGB_WAVELENGTHS), before, atol=1e-5)
    trainable = {id(p) for p in enc.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in lora_parameters(enc)}


def test_dofa_takes_any_number_of_bands():
    enc = tiny_dofa("hsi").eval()
    wl300 = np.linspace(0.384, 1.025, 300)
    with torch.no_grad():
        a = enc(torch.randn(1, 3, 224, 224), inputs.RGB_WAVELENGTHS)
        b = enc(torch.randn(1, 300, 224, 224), wl300)
    assert a.shape == b.shape == (1, enc.dim)


class _FakeImageBind(torch.nn.Module):
    """Mimics imagebind_huge's per-modality preprocessor/trunk/head call convention."""

    class Pre(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 16, 32, 32)

        def forward(self, thermal):
            return {"trunk": {"tokens": self.conv(thermal).flatten(2).transpose(1, 2)}, "head": {}}

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)
            self.blocks = torch.nn.ModuleList([layer, torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)])

        def forward(self, tokens):
            for b in self.blocks:
                tokens = b(tokens)
            return tokens

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 1024)

        def forward(self, z):
            return self.proj(z.mean(1))

    def __init__(self):
        super().__init__()
        self.modality_preprocessors = torch.nn.ModuleDict({"thermal": self.Pre()})
        self.modality_trunks = torch.nn.ModuleDict({"thermal": self.Trunk()})
        self.modality_heads = torch.nn.ModuleDict({"thermal": self.Head()})


def test_imagebind_wrapper_and_lora_skip_attention_projections():
    enc = ImageBindEncoder("thermal", _FakeImageBind()).eval()
    x = torch.randn(2, 3, 1, 224, 224)
    assert enc.encode_frames(x).shape == (2, 1024)
    add_lora(enc, last_n=1, rank=2)
    wrapped = [m for m in enc.modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == 2  # linear1, linear2 of the last block; not MultiheadAttention.out_proj
    assert not any(
        isinstance(m.out_proj, LoRALinear) for m in enc.modules() if isinstance(m, torch.nn.MultiheadAttention)
    )


def test_hsi_variants():
    wl = np.linspace(384, 1025, 300).astype(np.float32)
    cube = np.random.default_rng(0).uniform(0, 1, (256, 256, 300)).astype(np.float32)
    s = inputs.load_hsi("", "full", cube=(cube, wl))
    assert s.frames.shape == (1, 300, 224, 224) and len(s.wavelengths) == 300
    assert inputs.load_hsi("", "rgb3", cube=(cube, wl)).frames.shape[1] == 3
    m = inputs.load_hsi("", "mask664", cube=(cube, wl))
    assert 290 <= m.frames.shape[1] < 300 and not any(0.664 <= w < 0.684 for w in m.wavelengths)
    low = inputs.load_hsi("", "s64", cube=(cube, wl)).frames
    assert low.shape == s.frames.shape and low.std() < s.frames.std()  # spatial detail removed
    assert len(inputs.hsi_windows()) == 33


def test_stage_labels():
    rows = []
    for d, h, t in [
        ("2025-09-01", 12, 9.0),  # before cutoff: S0 whatever the clip says
        ("2025-09-05", 6, 1.0), ("2025-09-05", 12, 8.0), ("2025-09-05", 15, 8.0),  # midday loss, recovered: S1
        ("2025-09-08", 6, 7.0), ("2025-09-08", 12, 9.0),  # loss persists at 06 h: S2
        ("2025-09-11", 6, 7.0), ("2025-09-11", 12, 9.0),  # S2 + symptom on 10 Sep: S3
    ]:  # fmt: skip
        rows.append(dict(bed="W2", plant="J5", date=d, hour=h, truth_turgor_shift=t))
    rows.append(dict(bed="W1", plant="A2", date="2025-09-05", hour=12, truth_turgor_shift=0.5))
    turg = pd.DataFrame(rows)
    plants = pd.DataFrame({"bed": ["W2"], "plant": ["J5"], core.TRUTH_FIRST_SYMPTOM: [pd.Timestamp("2025-09-10")]})
    keys = pd.concat([turg[core.KEYS], pd.DataFrame([dict(bed="W2", plant="B1", date="2025-09-05", hour=12)])])
    lab = run_labels(keys, turg, plants).set_index(["plant", "date", "hour"])
    st = lab["stage"]
    assert st[("J5", pd.Timestamp("2025-09-01"), 12)] == 0
    assert st[("J5", pd.Timestamp("2025-09-05"), 12)] == 1
    assert st[("J5", pd.Timestamp("2025-09-08"), 12)] == 2
    assert st[("J5", pd.Timestamp("2025-09-11"), 12)] == 3
    assert st[("A2", pd.Timestamp("2025-09-05"), 12)] == 0  # control
    assert st[("B1", pd.Timestamp("2025-09-05"), 12)] == -1  # no clip: unlabelled
    y = lab["y_deficit"]
    assert y[("J5", pd.Timestamp("2025-09-01"), 12)] == 0 and y[("J5", pd.Timestamp("2025-09-05"), 6)] == 1


def test_fusion_model_ignores_absent_sensors_and_keeps_one_under_dropout():
    m = FusionModel({"rgb": 8, "thermal": 6}, d_model=16, modality_dropout=0.9).eval()
    now = {"rgb": torch.randn(4, 8), "thermal": torch.randn(4, 6)}
    base = {k: torch.zeros_like(v) for k, v in now.items()}
    present = torch.tensor([[True, False]] * 4)
    a = m(now, base, present)["logit"]
    now["thermal"] = torch.randn(4, 6) * 100  # garbage in an absent sensor changes nothing
    assert torch.allclose(m(now, base, present)["logit"], a, atol=1e-5)
    m.train()
    kept = m._drop_modalities(torch.ones(1000, 2, dtype=torch.bool))
    assert kept.any(1).all() and kept.float().mean() < 0.6


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _keys(n_per_bed=6, hours=(6, 12, 15)):
    days = [date(2025, 8, 24) + timedelta(d) for d in range(26)]
    rows = []
    for bed in ("W1", "W2", "W3"):
        for i in range(n_per_bed):
            for d in days:
                for h in hours:
                    rows.append(dict(bed=bed, plant=f"{'ABCDEF'[i]}{2 + i % 2}", date=d, hour=h))
    return core.normalize_keys(pd.DataFrame(rows))


def _fake_cache(tmp_path, keys):
    """thermal: warms from day 2 in deficit beds; rgb: a constant bed offset (a shortcut) + noise."""
    from uiwue.fusion.embed import save_table

    rng = np.random.default_rng(1)
    d = core.add_design(keys)
    stress = ((d["treatment"] == "deficit") & d["days_since_cutoff"].between(2, 12)).to_numpy()
    days = d["days_since_cutoff"].clip(0, 12).to_numpy()
    plant_level = rng.normal(0, 1, (keys["bed"] + keys["plant"]).nunique())[
        pd.factorize(keys["bed"] + keys["plant"])[0]
    ]
    th = rng.normal(0, 1, (len(keys), 12)) + plant_level[:, None]
    th[:, :4] += (stress * (0.6 + 0.25 * days))[:, None]
    offset = rng.normal(0, 1.5, 10)  # deficit beds look different in RGB from the start (lighting, pots)
    rgb = rng.normal(0, 1, (len(keys), 10)) + (d["bed"] != "W1").to_numpy()[:, None] * offset
    save_table(tmp_path / "emb_thermal_full.npz", keys, th)
    save_table(tmp_path / "emb_thermal_r80.npz", keys, th + rng.normal(0, 0.5, th.shape))
    save_table(tmp_path / "emb_rgb_full.npz", keys, rgb)
    return tmp_path


def test_stage1_end_to_end(tmp_path, monkeypatch):
    from uiwue.core.validate import validate
    from uiwue.fusion import train

    keys = _keys()
    cache = _fake_cache(tmp_path / "cache", keys)
    rs = train.assemble(cache, labels=run_labels(keys, turgor=pd.DataFrame(), plants=pd.DataFrame()))
    assert set(rs.sensors) == {"rgb", "thermal"} and ("thermal", "r80") in rs.emb
    cfg = train.Config(folds=3, epochs=30, patience=5, batch=128, n_boot=50, d_model=32)
    cv = train.cross_validate(rs, cfg, "cpu")
    assert {"rgb", "thermal", "rgb+thermal", "thermal_r80", "rgb+thermal_r80"} <= set(cv["setups"])
    res = train.results(rs, cv, "fusion", cfg.n_boot, groups=("all",))
    b = res["fusion_benchmark"].set_index("sensors")
    assert b.loc["thermal", "early_auc"] > 0.85 and b.loc["thermal", "onset_day"] <= 5
    assert not b.loc["thermal", "bed_effect"]
    # RGB carries only a bed offset. Bed identity alone separates the groups after the cutoff
    # (controls are never positive), so RGB's early AUC is high for the wrong reason. Both
    # placebo checks must expose that: the bed model separates the beds before the cutoff,
    # and so does the main model's own score; thermal's does not.
    assert b.loc["rgb", "bed_effect"]
    assert b.loc["rgb", "placebo_auc_main"] > 0.8 > b.loc["thermal", "placebo_auc_main"]
    assert b.loc["thermal_r80", "cost_rank"] < b.loc["thermal", "cost_rank"]

    out = tmp_path / "derived" / "detect"
    train.write(res, "fusion", out)
    assert validate(out / "fusion_scores_runs.parquet") == []
    for name in ("fusion_benchmark", "fusion_auc_by_day"):
        assert validate(out / f"{name}.parquet") == []
    # a second model's rows are added next to the first, not over them
    train.write({"fusion_benchmark": res["fusion_benchmark"].assign(model="fusion+lora_hsi")}, "fusion+lora_hsi", out)
    assert set(pd.read_parquet(out / "fusion_benchmark.parquet")["model"]) == {"fusion", "fusion+lora_hsi"}


def test_lora_stage_end_to_end(tmp_path, monkeypatch):
    """Raw RGB images -> embed with a tiny DOFA -> LoRA stage, 2 folds, 1 epoch."""
    from PIL import Image

    from uiwue.fusion import train
    from uiwue.fusion.embed import embed_sensor, save_table

    keys = _keys(n_per_bed=2, hours=(12,))
    keys = keys[keys["date"].between("2025-08-30", "2025-09-08") & (keys["bed"] != "W3")].reset_index(drop=True)
    rng = np.random.default_rng(0)
    for r in keys.itertuples():
        d = tmp_path / "data" / "rgb" / f"{r.bed}_{r.plant}" / f"{r.date:%Y_%m_%d}" / f"{r.hour:02d}"
        d.mkdir(parents=True)
        g = 120 if r.bed == "W1" or r.date < pd.Timestamp("2025-09-04") else 60
        img = np.clip(rng.normal(g, 20, (64, 96, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(d / "rgb_007.jpg")

    runs = inputs.list_runs(tmp_path / "data", "rgb")
    assert len(runs) == len(keys)
    enc = tiny_dofa("rgb")
    emb, ok, errors = embed_sensor(enc, runs, ["full"], "cpu", batch=8)
    assert ok.all() and not errors
    save_table(tmp_path / "cache" / "emb_rgb_full.npz", runs, emb["full"])

    rs = train.assemble(tmp_path / "cache", labels=run_labels(keys, turgor=pd.DataFrame(), plants=pd.DataFrame()))
    cfg = train.Config(folds=2, epochs=2, lora_epochs=1, lora_batch=8, workers=0, n_boot=0, d_model=16, lora_blocks=2)
    stage = train.LoRAStage(rs, "rgb", tmp_path / "data", cfg)
    assert stage.ok.all()
    stage.make_encoder = lambda: tiny_dofa("rgb")
    cv = train.cross_validate_lora(rs, stage, cfg, "cpu", ckpt_dir=tmp_path / "ckpt")
    assert np.isfinite(cv["prob"]["rgb"]).mean() > 0.9
    assert (tmp_path / "ckpt" / "fold1.pt").exists()


# --------------------------------------------------------------------------- #
# verification 2026-11-14 (docs/fusion.md): real-data fixes
# --------------------------------------------------------------------------- #
def test_dofa_patch_embedding_runs_in_fp32_under_autocast():
    """Wavelength -> kernel generation must not run in bf16 (it garbles the sine code of ~1000 nm)."""
    enc = tiny_dofa("hsi").eval()
    wl = tuple(np.linspace(0.384, 1.025, 20))
    x = torch.randn(2, 20, 224, 224)
    seen = []
    enc.model.patch_embed.register_forward_hook(lambda m, i, o: seen.append(o[0].dtype))
    with torch.no_grad():
        ref = enc.model.forward_features(x, list(wl))  # TorchGeo's own forward, fp32
        assert torch.allclose(enc(x, wl), ref, atol=1e-5)  # the re-implementation is the same model
        with torch.autocast("cpu", dtype=torch.bfloat16):
            enc(x, wl)
    assert seen[-1] == torch.float32


def test_imagebind_model_module_imports_without_package_init():
    """imagebind/__init__ needs pkg_resources, torchaudio and pytorchvideo; the model code does not."""
    import importlib.util

    if importlib.util.find_spec("imagebind") is None:
        pytest.skip("ImageBind not installed")
    from uiwue.fusion.encoders import _imagebind_model_module

    assert hasattr(_imagebind_model_module(), "imagebind_huge")


def test_thermal_uses_the_centre_frame_upright(tmp_path):
    import tifffile

    for k in range(5):  # frame k is 20 + k degC, with a hot pixel stored at the top-left
        img = np.full((48, 64), 20.0 + k, np.float32)
        img[0, 0] = 60.0
        tifffile.imwrite(tmp_path / f"thermal_{k:03d}.tif", img)
    frames = inputs.read_thermal(str(tmp_path), inputs.FRAMES["thermal"])
    assert len(frames) == 1 and frames[0][-1, -1] == 60.0  # stored rotated by 180 degrees
    s = inputs.load_thermal(str(tmp_path))
    assert s.frames.shape == (1, 1, 224, 224)
    assert np.isclose(np.median(s.frames), (22.0 - inputs.THERMAL_SCALE[0]) / inputs.THERMAL_SCALE[1])


def test_depth_disparity_scaling(tmp_path):
    import tifffile

    d = np.full((171, 224), 0.25, np.float32)  # canopy at 0.25 m: disparity 4 / m
    d[:, :100] = 0.5  # soil at 0.5 m: disparity 2 / m
    d[80:90, 150:160] = 0  # invalid
    with tifffile.TiffWriter(tmp_path / "depth_007.tif") as w:
        w.write(np.rot90(d, 2))  # ADA stores depth rotated by 180 degrees
    x = inputs.load_depth(str(tmp_path)).frames[0, 0]
    lo, sd = inputs.DISPARITY_SCALE
    assert np.isclose(np.percentile(x, 5), (2.0 - lo) / sd, atol=1e-3)
    assert np.isclose(np.percentile(x, 95), (4.0 - lo) / sd, atol=1e-3)
    assert x[:, :50].mean() < 0 < x[:, -40:].mean()  # orientation kept: soil on the left
    assert -1.5 < (1 / 0.52 - lo) / sd < 0 < (1 / 0.19 - lo) / sd < 4  # rig range inside ImageBind's


def test_hsi_runs_only_at_9_12_15_and_dark_reference_is_required(tmp_path, monkeypatch):
    for h in ("06", "09", "12", "15", "18"):
        d = tmp_path / "hsi" / "W1_A2" / "2025_09_05" / h
        d.mkdir(parents=True)
        (d / "hsi_000.jp2").write_bytes(b"")
    runs = inputs.list_runs(tmp_path, "hsi")
    assert sorted(runs["hour"]) == [9, 12, 15]
    with pytest.raises(FileNotFoundError):
        inputs.read_hsi(runs["path"].iloc[0], dark_reference=tmp_path / "missing.npy")


def test_hsi_spectral_variants_are_cut_from_the_full_input():
    from uiwue.fusion.embed import _variant

    wl = np.linspace(383.8, 1025.2, 300).astype(np.float32)
    cube = np.random.default_rng(0).uniform(-0.2, 2.5, (256, 256, 300)).astype(np.float32)
    full = inputs.load_hsi("", "full", cube=(cube, wl))
    assert (
        full.frames.min() >= (inputs.REFLECTANCE_RANGE[0] - inputs.REFLECTANCE_SCALE[0]) / inputs.REFLECTANCE_SCALE[1]
    )
    d = {"full": (torch.from_numpy(full.frames), full.wavelengths)}
    for v in ("rgb3", "mask664", "mask1004"):
        direct = inputs.load_hsi("", v, cube=(cube, wl))
        x, w = _variant(d, v)
        assert w == direct.wavelengths and np.allclose(x.numpy(), direct.frames, atol=1e-6)


def test_cutoff_day_afternoon_is_positive_but_not_trained_on():
    keys = pd.DataFrame(
        [dict(bed="W2", plant="J5", date="2025-09-03", hour=h) for h in (6, 9, 12, 15, 18)]
        + [
            dict(bed="W2", plant="J5", date="2025-09-04", hour=6),
            dict(bed="W1", plant="A2", date="2025-09-03", hour=12),
        ]
    )
    lab = run_labels(keys, turgor=pd.DataFrame(), plants=pd.DataFrame()).set_index(["bed", "date", "hour"])
    day0 = lab.loc["W2"].loc[pd.Timestamp("2025-09-03")]
    assert day0["y_deficit"].tolist() == [0, 0, 1, 1, 1]  # 06 and 09 h are before the 11:00 cutoff
    assert day0["train"].tolist() == [True, True, False, False, False]
    assert (
        lab.loc[("W2", pd.Timestamp("2025-09-04"), 6), "train"]
        and lab.loc[("W1", pd.Timestamp("2025-09-03"), 12), "train"]
    )


def test_lora_adapters_follow_the_encoder_device_and_dtype():
    """LoRAStage adds LoRA after moving the encoder to the GPU; adapters must land there too."""
    enc = tiny_dofa().to(torch.float64)
    add_lora(enc, last_n=1, rank=2)
    assert {p.dtype for p in lora_parameters(enc)} == {torch.float64}
    enc = tiny_dofa().to("meta")
    add_lora(enc, last_n=1, rank=2)
    assert {p.device.type for p in lora_parameters(enc)} == {"meta"}


def test_live_images_augment_per_item_not_per_worker(monkeypatch):
    from uiwue.fusion import train

    img = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32).numpy()
    monkeypatch.setattr(train.inputs, "load", lambda sensor, path: inputs.Sample(img.copy(), inputs.RGB_WAVELENGTHS))
    ds = train._LiveImages(["a"] * 6, [["b"]] * 6, "rgb", aug=True, seed=3)
    xs = [ds[i][1] for i in range(6)]
    assert torch.equal(ds[2][1], xs[2])  # reproducible per item
    assert len({tuple(x.flatten()[:50].tolist()) for x in xs}) > 1  # items differ


def test_truncated_hsi_cube_fails_its_run_in_the_loader(tmp_path, monkeypatch):
    """A cube with fewer bands than its wavelength list must raise in read_hsi (the loader reports
    the run as failed); reaching DOFA or the band masks it would stop the whole embedding job."""
    import json

    hyperio = pytest.importorskip("hyperio")
    (tmp_path / "hsi_000.jp2").write_bytes(b"")
    (tmp_path / "hsi_000.json").write_text(json.dumps({"wavelengths": list(np.linspace(384, 1025, 300))}))
    np.save(tmp_path / "dark.npy", np.zeros(300))
    cube = np.full((16, 16, 256), 0.3, np.float32)
    monkeypatch.setattr(hyperio.HSI, "read", staticmethod(lambda *a, **k: cube))
    with pytest.raises(ValueError, match="truncated"):
        inputs.read_hsi(str(tmp_path), dark_reference=tmp_path / "dark.npy")
    cube = np.full((16, 16, 300), 0.3, np.float32)  # 256 x 256 pixels is fine: only the bands count
    assert inputs.read_hsi(str(tmp_path), dark_reference=tmp_path / "dark.npy")[0].shape == (16, 16, 300)
