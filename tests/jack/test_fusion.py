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
