"""Fusion v2: the fusion model read against the control bed (docs/fusion.md section 8).

    python -m uiwue.fusion.v2 ablate --cache <caches> --features <thermal features> --variants v1 ref ...
    python -m uiwue.fusion.v2 train  --cache <caches> --features <thermal features> [--variant <name>]

What changes from v1 (uiwue.fusion.train):

- **Control-referenced tokens** ("ref"). Each sensor's change from the plant's own baseline,
  minus the median change of the control plants of the training fold in the same run:
  a difference in differences. Weather, light and anything else shared by the run cancels,
  as it does in the benchmark's `thermal_leaf_ref_dt`. A control plant of the training
  fold is compared with the other controls (leave-one-out median); held-out plants, of
  either bed, with all the training controls, so a held-out plant's bed is never used.
- **No absolute embedding** by default: the absolute embedding e(t) carries the plant's
  pot, place and light (the placebo bed model found the beds in it), its baseline change
  and the control-referenced change carry the response.
- **Hand-made thermal token** ("tfeat"): the benchmark's leaf temperature, its 10th
  percentile and spread over frames, and leaf minus air (uiwue.fusion.features), with
  the same baseline and control references. It belongs to the thermal camera in every
  sensor setup.
- Optional: weather token ("env": air temperature, humidity, VPD and outside radiation of
  the run, the W1-end values for every bed so the bed's flyover time cannot leak), a bed
  adversary on the pre-cutoff runs (gradient reversal), and a head without attention
  (layers = 0: the mean of the token projections).

Folds, the inner early-stopping split, labels, the placebo models and every score are
those of v1 (train.plant_folds, train.inner_split, labels.run_labels, train.score), so the
numbers sit beside section 7. Lock-box plants (uiwue.fusion.lockbox) are dropped before
anything is fitted.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from uiwue import core
from uiwue.detect.benchmark import FULL_DAYS, TRAIN_DAYS, _curve, _per_plant_auc
from uiwue.fusion import lockbox, train
from uiwue.fusion.embed import CACHE, load_table
from uiwue.fusion.fusion import expected_stage, fusion_loss
from uiwue.fusion.labels import N_STAGES, TURGOR_SCALE, run_labels
from uiwue.fusion.train import Config, RunSet, baseline_of
from uiwue.stats import onset_day
from uiwue.stats.core import EARLY_DAYS, SEED

V2_CACHE = core.REPO / "cache" / "fusion_v2"
FEATURES = V2_CACHE / "thermal_features.parquet"
RUNS_DIR = train.RUNS_DIR
CAMERAS = ("rgb", "depth", "thermal", "hsi")  # cost order, as the benchmark
TOKEN_CAMERA = {"rgb": "rgb", "depth": "depth", "thermal": "thermal", "tfeat": "thermal", "hsi": "hsi", "env": None}
TFEAT = ("thermal_leaf_t", "thermal_leaf_t_mad", "thermal_leaf_t_p10", "thermal_leaf_air_dt")
ENV = ("env_air_t", "env_rh", "env_vpd", "env_global_rad")
LOW_RES = ("r160", "r80")
EMB_VARIANTS = {"rgb": ("full",), "depth": ("full",), "thermal": ("full", *LOW_RES), "hsi": ("full",)}
RECOVERY_DAYS = (14, 19)  # after the re-watering on day 12 (18:35)
# Days before the cutoff scored like the stress days (2 Sep = day -1, inside the baseline window).
# Nothing is stressed yet, so any separation there is a bed or greenhouse difference.
CANARY_DAYS = (-3, -1)


@dataclass
class V2Config(Config):
    parts: tuple = ("delta", "ref")  # per token: "now" e(t), "delta" e(t) - e(base), "ref" delta - control delta
    emb: bool = True  # image-encoder tokens
    thermal_emb: bool = True  # the ImageBind thermal token (False: the thermal camera is the hand-made token only)
    tfeat: bool = True  # hand-made thermal token
    env: bool = False  # weather token
    adv: float = 0.0  # weight of the bed adversary on pre-cutoff runs (0 = off)
    hardneg: float = 1.0  # loss weight of deficit-bed runs on 2 Sep and the 3 Sep morning (negatives)
    min_ref: int = 5  # control plants needed in a run for a reference
    input_dropout: float = 0.0
    window: int = 0  # hours: each run's inputs averaged with the plant's earlier runs that day within it (0 = off)


VARIANTS: dict[str, dict] = {
    # the v1 design on the v2 plants (embeddings only, absolute + baseline change)
    "v1": dict(parts=("now", "delta"), tfeat=False),
    "delta": dict(parts=("delta",), tfeat=False),
    "ref": dict(parts=("ref",), tfeat=False),
    "delta+ref": dict(parts=("delta", "ref"), tfeat=False),
    "now+delta+ref": dict(parts=("now", "delta", "ref"), tfeat=False),
    "ref+tfeat": dict(parts=("ref",), tfeat=True),
    "delta+ref+tfeat": dict(parts=("delta", "ref"), tfeat=True),
    "tfeat_only": dict(parts=("delta", "ref"), tfeat=True, emb=False),
    "delta+ref+tfeat+env": dict(parts=("delta", "ref"), tfeat=True, env=True),
    "delta+ref+tfeat+adv": dict(parts=("delta", "ref"), tfeat=True, adv=0.5),
    "delta+ref+tfeat+adv2": dict(parts=("delta", "ref"), tfeat=True, adv=2.0),
    "delta+ref+tfeat+hn": dict(parts=("delta", "ref"), tfeat=True, hardneg=5.0),
    "delta+ref+tfeat+win6": dict(parts=("delta", "ref"), tfeat=True, window=6),
    "tfeat_only+win6": dict(parts=("delta", "ref"), tfeat=True, emb=False, window=6),
    # every camera, thermal as the hand-made token only (no ImageBind thermal token)
    "fused_tfeat": dict(parts=("delta", "ref"), tfeat=True, thermal_emb=False),
    "delta+ref+tfeat+mean": dict(parts=("delta", "ref"), tfeat=True, layers=0),
    "v1+tfeat": dict(parts=("now", "delta"), tfeat=True),
}


def make_config(name: str, **over) -> V2Config:
    return V2Config(**(VARIANTS[name] | over))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_embeddings(caches, sensors=CAMERAS, variants=EMB_VARIANTS) -> dict[tuple[str, str], tuple]:
    """As train.load_tables, but only the variants v2 uses (the 33 HSI windows are not loaded)."""
    caches = [Path(caches)] if isinstance(caches, str | Path) else [Path(c) for c in caches]
    parts: dict[tuple[str, str], list] = {}
    for cache in caches:
        for s in sensors:
            for v in variants.get(s, ("full",)):
                p = cache / f"emb_{s}_{v}.npz"
                if p.exists():
                    parts.setdefault((s, v), []).append(load_table(p))
    tables = {}
    for sv, got in parts.items():
        keys = pd.concat([k for k, _ in got], ignore_index=True)
        emb = np.concatenate([e for _, e in got])
        first = ~keys.duplicated(core.KEYS).to_numpy()
        tables[sv] = (keys[first].reset_index(drop=True), emb[first])
    return tables


def _align(index: pd.MultiIndex, keys: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = index.get_indexer(pd.MultiIndex.from_frame(keys[core.KEYS]))
    full = np.zeros((len(index), values.shape[1]), np.float32)
    got = np.zeros(len(index), bool)
    full[pos[pos >= 0]], got[pos[pos >= 0]] = values[pos >= 0], True
    bad = ~np.isfinite(full).all(1)
    full[bad], got[bad] = 0, False
    return full, got


def env_table(keys: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Weather per run: the W1 row of each (date, hour) for every bed (no flyover-time offset)."""
    e = core.read_table("env", "env_runs")
    if e is None:
        return keys[core.KEYS].iloc[:0], np.zeros((0, len(ENV)), np.float32)
    w1 = e[e["bed"] == "W1"].groupby(["date", "hour"], as_index=False)[list(ENV)].median()
    k = keys[core.KEYS].merge(w1, on=["date", "hour"], how="left")
    return k[core.KEYS], k[list(ENV)].to_numpy(np.float32)


def assemble(
    caches=CACHE, features: Path | None = FEATURES, exclude=lockbox.LOCKBOX, only=None, labels=None,
    baseline=core.BASELINE,
) -> RunSet:  # fmt: skip
    """Embeddings, hand-made thermal features and weather on one set of runs, with labels.

    Rows of `exclude` plants (the lock-box) are dropped before anything else; `only` keeps
    just those plants (used to score the lock-box)."""
    tables = load_embeddings(caches)
    if not tables:
        raise FileNotFoundError(f"no embeddings in {caches}")
    keys = pd.concat([k for (s, v), (k, _) in tables.items() if v == "full"]).drop_duplicates()
    pid = keys["bed"] + "_" + keys["plant"]
    keep = ~pid.isin(list(exclude or ())) if only is None else pid.isin(list(only))
    keys = keys[keep.to_numpy()].sort_values(core.KEYS, ignore_index=True)
    lab = run_labels(keys) if labels is None else labels
    keys = keys.merge(lab, on=core.KEYS, how="left")
    keys["pid"] = keys["bed"] + "_" + keys["plant"]
    index = pd.MultiIndex.from_frame(keys[core.KEYS])

    emb, base, has, dims = {}, {}, {}, {}
    for (s, v), (k, e) in tables.items():
        full, got = _align(index, k, e)
        b, ok = baseline_of(keys, full, got, baseline)
        emb[(s, v)], base[(s, v)], has[(s, v)], dims[s] = full, b, got & ok, e.shape[1]
    if features is not None and Path(features).exists():
        f = core.normalize_keys(pd.read_parquet(features))
        f = f[f["thermal_qc_pass"].fillna(True).astype(bool)] if "thermal_qc_pass" in f else f
        for v in ("full", *LOW_RES):
            cols = [c if v == "full" else f"{c}_{v}" for c in TFEAT]
            if all(c in f for c in cols):
                full, got = _align(index, f, f[cols].to_numpy(np.float32))
                b, ok = baseline_of(keys, full, got, baseline)
                emb[("tfeat", v)], base[("tfeat", v)], has[("tfeat", v)], dims["tfeat"] = full, b, got & ok, len(cols)
    ek, ev = env_table(keys)
    if len(ek):
        full, got = _align(index, ek, ev)
        emb[("env", "full")], base[("env", "full")], has[("env", "full")], dims["env"] = (
            full, np.zeros_like(full), got, len(ENV),
        )  # fmt: skip
    sensors = [s for s in ("rgb", "depth", "thermal", "tfeat", "hsi", "env") if (s, "full") in emb]
    rs = RunSet(keys, sensors, dims, emb, base, has)
    rs.ref, rs.has_ref = {}, {}
    return rs


def run_groups(keys: pd.DataFrame) -> np.ndarray:
    return keys.groupby(["date", "hour"]).ngroup().to_numpy()


def _loo_median(x: np.ndarray) -> np.ndarray:
    """Median of the other rows for each row of x (n, d), per column."""
    n = len(x)
    s = np.sort(x, 0)
    rank = np.argsort(np.argsort(x, 0, kind="stable"), 0, kind="stable")
    a, b = (n - 2) // 2, (n - 1) // 2
    ia, ib = a + (a >= rank), b + (b >= rank)
    return 0.5 * (np.take_along_axis(s, ia, 0) + np.take_along_axis(s, ib, 0))


def reference(rs: RunSet, ref_rows: np.ndarray, min_ref: int = 5, groups: np.ndarray | None = None) -> dict:
    """Control-referenced change for every (token, variant): the run's baseline change minus
    the median change of the reference plants (`ref_rows`, training-fold controls) in the same
    run; reference plants get the median of the others. Sets rs.ref, rs.has_ref and returns
    the per-run reference medians {(token, variant): {run group: vector}} for predict.py."""
    g = run_groups(rs.keys) if groups is None else groups
    order = np.argsort(g, kind="stable")
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    medians = {}
    for sv in rs.emb:
        if sv[0] == "env":
            continue
        d = rs.emb[sv] - rs.base[sv]
        ok = rs.has[sv]
        out = np.zeros_like(d)
        got = np.zeros(len(d), bool)
        meds = {}
        for rows in np.split(order, bounds):
            r = rows[ref_rows[rows] & ok[rows]]
            if len(r) < min_ref:
                continue
            med = np.median(d[r], 0)
            meds[int(g[rows[0]])] = med.astype(np.float32)
            out[rows] = d[rows] - med
            out[r] = d[r] - _loo_median(d[r])
            got[rows] = True
        rs.ref[sv], rs.has_ref[sv] = out, got & ok
        medians[sv] = meds
    return medians


def causal_window(keys: pd.DataFrame, values: np.ndarray, valid: np.ndarray, hours: int) -> np.ndarray:
    """Each row's mean over the same plant's valid rows from `hours` before it up to it (same
    clock, so with hours < 12 the window stays inside the day). Rows are any order."""
    t = (keys["date"] - pd.Timestamp(core.DATES[0])).dt.days.to_numpy() * 24 + keys["hour"].to_numpy()
    pid = pd.factorize(keys["pid"])[0]
    key = pid.astype(np.int64) * 100_000 + t
    order = np.argsort(key, kind="stable")
    ks, v = key[order], valid[order].astype(np.float64)
    x = np.where(v[:, None] > 0, values[order], 0.0)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    cn = np.concatenate([[0.0], np.cumsum(v)])
    start = np.searchsorted(ks, ks - hours, side="left")
    end = np.arange(1, len(ks) + 1)
    n = (cn[end] - cn[start])[:, None]
    mean = np.where(n > 0, (cs[end] - cs[start]) / np.maximum(n, 1), 0.0)
    out = np.empty_like(values)
    out[order] = mean.astype(values.dtype)
    return out


def smooth_inputs(rs: RunSet, hours: int, parts) -> None:
    """rs.smooth[(part, (token, variant))]: the causal window of each part, after the reference."""
    rs.smooth = {}
    for sv in rs.emb:
        if sv[0] == "env":
            continue
        if "delta" in parts:
            rs.smooth[("delta", sv)] = causal_window(rs.keys, rs.emb[sv] - rs.base[sv], rs.has[sv], hours)
        if "ref" in parts and sv in rs.ref:
            rs.smooth[("ref", sv)] = causal_window(rs.keys, rs.ref[sv], rs.has_ref[sv], hours)
        if "now" in parts:
            rs.smooth[("now", sv)] = causal_window(rs.keys, rs.emb[sv], rs.has[sv], hours)


def apply_reference(rs: RunSet, medians: dict, run_keys: pd.DataFrame) -> None:
    """New plants against saved reference medians (run_keys: date, hour, group of each median)."""
    lookup = {
        (pd.Timestamp(d), int(h)): int(gid) for d, h, gid in run_keys[["date", "hour", "group"]].itertuples(False)
    }
    gid = np.array([lookup.get((pd.Timestamp(d), int(h)), -1) for d, h in rs.keys[["date", "hour"]].itertuples(False)])
    for sv in rs.emb:
        if sv[0] == "env":
            continue
        d = rs.emb[sv] - rs.base[sv]
        out, got = np.zeros_like(d), np.zeros(len(d), bool)
        meds = medians.get(sv, {})
        for i in np.unique(gid):
            if i < 0 or i not in meds:
                continue
            rows = gid == i
            out[rows], got[rows] = d[rows] - meds[i], True
        rs.ref[sv], rs.has_ref[sv] = out, got & rs.has[sv]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class FusionV2(nn.Module):
    """One token per input (sensor embedding, thermal features, weather), each the standardised
    concatenation of its parts -> linear -> d_model + a learned token embedding; a [CLS] token
    and `layers` transformer layers (or, with layers = 0, the mean of the present tokens); heads
    for deficit, stage (ordinal) and turgor, and a bed head behind gradient reversal."""

    def __init__(
        self, in_dims: dict[str, int], d_model=256, layers=2, heads=4, dropout=0.1, modality_dropout=0.3,
        input_dropout=0.0, n_stages=N_STAGES,
    ):  # fmt: skip
        super().__init__()
        self.tokens = list(in_dims)
        self.modality_dropout, self.layers = modality_dropout, layers
        for t, d in in_dims.items():
            self.register_buffer(f"mu_{t}", torch.zeros(d))
            self.register_buffer(f"sd_{t}", torch.ones(d))
        self.inp_drop = nn.Dropout(input_dropout)
        self.proj = nn.ModuleDict({t: nn.Linear(d, d_model) for t, d in in_dims.items()})
        self.token = nn.Parameter(torch.randn(len(in_dims), d_model) * 0.02)
        if layers > 0:
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_deficit = nn.Linear(d_model, 1)
        self.head_stage = nn.Linear(d_model, n_stages - 1)
        self.head_turgor = nn.Linear(d_model, 1)
        self.head_bed = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    @torch.no_grad()
    def fit_standardiser(self, token: str, x: torch.Tensor) -> None:
        getattr(self, f"mu_{token}").copy_(x.mean(0))
        getattr(self, f"sd_{token}").copy_(x.std(0).clamp_min(1e-3) if len(x) > 1 else torch.ones_like(x[0]))

    def _drop(self, present):
        if not self.training or self.modality_dropout <= 0:
            return present
        keep = present & (torch.rand_like(present, dtype=torch.float) >= self.modality_dropout)
        empty = ~keep.any(1) & present.any(1)
        if empty.any():
            score = torch.rand_like(present, dtype=torch.float).masked_fill(~present, -1)
            keep = torch.where(empty[:, None], F.one_hot(score.argmax(1), present.shape[1]).bool(), keep)
        return keep

    def forward(self, x: dict[str, torch.Tensor], present: torch.Tensor, adv_lambda: float = 0.0) -> dict:
        present = self._drop(present)
        toks = []
        for i, t in enumerate(self.tokens):
            z = (x[t] - getattr(self, f"mu_{t}")) / getattr(self, f"sd_{t}")
            z = torch.where(present[:, i : i + 1], z, torch.zeros_like(z))
            toks.append(self.proj[t](self.inp_drop(z)) + self.token[i])
        h = torch.stack(toks, 1)
        if self.layers > 0:
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            pad = torch.cat([torch.zeros_like(present[:, :1]), ~present], 1)
            z = self.transformer(h, src_key_padding_mask=pad)[:, 0]
        else:
            w = present.float().unsqueeze(-1)
            z = (h * w).sum(1) / w.sum(1).clamp_min(1)
        z = self.norm(z)
        return dict(
            logit=self.head_deficit(z).squeeze(-1),
            stage_logits=self.head_stage(z),
            turgor=self.head_turgor(z).squeeze(-1),
            bed_logit=self.head_bed(_GradReverse.apply(z, adv_lambda)).squeeze(-1),
        )


def model_cameras(tokens) -> list[str]:
    """The cameras a model's tokens come from, in cost order (what train.results calls sensors)."""
    cams = {TOKEN_CAMERA[t] for t in tokens}
    return [c for c in train.SENSOR_ORDER if c in cams]


def camera_view(rs: RunSet, tokens) -> RunSet:
    """rs with `sensors` set to the model's cameras, for train.results (its all-sensor scores)."""
    return RunSet(rs.keys, model_cameras(tokens), rs.dims, rs.emb, rs.base, rs.has)


def token_list(rs: RunSet, cfg: V2Config) -> list[str]:
    out = []
    for t in rs.sensors:
        if t == "tfeat" and not cfg.tfeat:
            continue
        if t == "env" and not cfg.env:
            continue
        if t in CAMERAS and not cfg.emb:
            continue
        if t == "thermal" and not cfg.thermal_emb:
            continue
        out.append(t)
    return out


def in_dims(rs: RunSet, cfg: V2Config) -> dict[str, int]:
    return {t: rs.dims[t] * (1 if t == "env" else len(cfg.parts)) for t in token_list(rs, cfg)}


def _parts(rs: RunSet, t: str, v: str, idx: np.ndarray, parts) -> tuple[np.ndarray, np.ndarray]:
    sv = (t, v)
    if t == "env":
        return rs.emb[sv][idx], rs.has[sv][idx]
    cols, ok = [], rs.has[sv][idx].copy()
    smooth = getattr(rs, "smooth", {})
    for p in parts:
        if (p, sv) in smooth:  # the causal same-day window (V2Config.window)
            cols.append(smooth[(p, sv)][idx])
            if p == "ref":
                ok &= rs.has_ref[sv][idx]
        elif p == "now":
            cols.append(rs.emb[sv][idx])
        elif p == "delta":
            cols.append(rs.emb[sv][idx] - rs.base[sv][idx])
        elif p == "ref":
            cols.append(rs.ref[sv][idx])
            ok &= rs.has_ref[sv][idx]
        else:
            raise ValueError(p)
    return np.concatenate(cols, 1), ok


def make_batch(
    rs: RunSet, idx: np.ndarray, setup: dict[str, str], tokens, cfg: V2Config, device: str, target="y_deficit"
):
    x, present = {}, []
    for t in tokens:
        v = setup.get(t)
        d = rs.dims[t] * (1 if t == "env" else len(cfg.parts))
        if v is None or (t, v) not in rs.emb:
            x[t] = torch.zeros(len(idx), d, device=device)
            present.append(np.zeros(len(idx), bool))
        else:
            a, ok = _parts(rs, t, v, idx, cfg.parts)
            x[t] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
            present.append(ok)
    k = rs.keys.iloc[idx]
    return dict(
        x=x,
        present=torch.from_numpy(np.stack(present, 1)).to(device),
        y=torch.from_numpy(k[target].to_numpy(np.float32)).to(device),
        stage=torch.from_numpy(pd.to_numeric(k["stage"]).fillna(-1).to_numpy(np.int64, copy=True)).to(device),
        turgor=torch.from_numpy(k["turgor"].to_numpy(np.float32) / TURGOR_SCALE).to(device),
        bed=torch.from_numpy((k["bed"] != "W1").to_numpy(np.float32)).to(device),
        pre=torch.from_numpy((k["date"] < pd.Timestamp(core.CUTOFF)).to_numpy()).to(device),
        late=torch.from_numpy(late_negative(k)).to(device),
    )


def late_negative(k: pd.DataFrame) -> np.ndarray:
    """Deficit-bed runs of 2 Sep and of the 3 Sep morning: the last negatives before the cutoff.
    They share whatever changed between the beds from 2 Sep (doors closed; Master Plan 6.6)
    with the stress days, so weighting them up asks the model for the stress beyond it."""
    cut = pd.Timestamp(core.CUTOFF)
    day = (k["date"] == cut - pd.Timedelta(days=1)) | ((k["date"] == cut) & (k["hour"] <= 11))
    return (day & (k["bed"] != "W1")).to_numpy()


def full_setup(rs: RunSet, tokens) -> dict[str, str]:
    return {t: "full" for t in tokens}


def usable(rs: RunSet, idx: np.ndarray, setup, tokens, cfg) -> np.ndarray:
    ok = np.zeros(len(idx), bool)
    for t in tokens:
        if t == "env" or (t, setup.get(t)) not in rs.emb:
            continue
        ok |= _parts(rs, t, setup[t], idx, cfg.parts)[1]
    return idx[ok]


def build(rs_or_dims, cfg: V2Config) -> FusionV2:
    dims = rs_or_dims if isinstance(rs_or_dims, dict) else None
    return FusionV2(
        dims, d_model=cfg.d_model, layers=cfg.layers, heads=cfg.heads, modality_dropout=cfg.modality_dropout,
        input_dropout=cfg.input_dropout,
    )  # fmt: skip


def fit(rs: RunSet, train_idx, val_idx, cfg: V2Config, device: str, target="y_deficit", aux=True, adv=None):
    """Train on cached inputs, early stopping on the inner-validation plants' deficit loss."""
    torch.manual_seed(cfg.seed)
    tokens = token_list(rs, cfg)
    setup = full_setup(rs, tokens)
    train_idx, val_idx = usable(rs, train_idx, setup, tokens, cfg), usable(rs, val_idx, setup, tokens, cfg)
    if len(train_idx) == 0:
        return None
    model = build(in_dims(rs, cfg), cfg).to(device)
    for t in tokens:  # standardisation from this fold's training rows only (rule 6)
        a, ok = _parts(rs, t, "full", train_idx, cfg.parts)
        if ok.any():
            model.fit_standardiser(t, torch.from_numpy(np.ascontiguousarray(a[ok], dtype=np.float32)).to(device))
    adv = cfg.adv if adv is None else adv
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best, best_loss, wait = copy.deepcopy(model.state_dict()), np.inf, 0
    for epoch in range(cfg.epochs):
        model.train()
        lam = adv * min(1.0, (epoch + 1) / 10)
        for chunk in np.array_split(rng.permutation(train_idx), max(1, len(train_idx) // cfg.batch)):
            b = make_batch(rs, chunk, setup, tokens, cfg, device, target)
            if not aux:
                b["stage"][:], b["turgor"][:] = -1, float("nan")
            out = model(b["x"], b["present"], adv_lambda=lam)
            loss = fusion_loss(out, b["y"], b["stage"], b["turgor"])
            if cfg.hardneg != 1.0 and b["late"].any():  # the deficit BCE again, late negatives weighted up
                w = torch.where(b["late"], cfg.hardneg, 1.0)
                bce = F.binary_cross_entropy_with_logits(out["logit"], b["y"], reduction="none")
                loss = loss - bce.mean() + (bce * w).sum() / w.sum()
            if adv > 0 and b["pre"].any():
                loss = loss + F.binary_cross_entropy_with_logits(out["bed_logit"][b["pre"]], b["bed"][b["pre"]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if len(val_idx) == 0:
            best = copy.deepcopy(model.state_dict())
            continue
        val = _val_loss(model, rs, val_idx, setup, tokens, cfg, device, target)
        if val < best_loss - 1e-4:
            best, best_loss, wait = copy.deepcopy(model.state_dict()), val, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best)
    return model.eval()


@torch.no_grad()
def _val_loss(model, rs, idx, setup, tokens, cfg, device, target) -> float:
    model.eval()
    losses, n = 0.0, 0
    for start in range(0, len(idx), 8192):
        b = make_batch(rs, idx[start : start + 8192], setup, tokens, cfg, device, target)
        out = model(b["x"], b["present"])
        losses += float(F.binary_cross_entropy_with_logits(out["logit"], b["y"], reduction="sum"))
        n += len(b["y"])
    return losses / max(n, 1)


@torch.no_grad()
def predict(model, rs, idx, setup, cfg, device, chunk=8192):
    """(prob, expected stage, turgor) for rows idx; NaN where no camera of the setup is present."""
    prob, stage, turg = (np.full(len(idx), np.nan) for _ in range(3))
    if model is None:
        return prob, stage, turg
    model.eval()
    tokens = model.tokens
    cam = [i for i, t in enumerate(tokens) if t != "env"]
    for start in range(0, len(idx), chunk):
        part = idx[start : start + chunk]
        b = make_batch(rs, part, setup, tokens, cfg, device)
        out = model(b["x"], b["present"])
        ok = b["present"][:, cam].any(1).cpu().numpy() if cam else np.ones(len(part), bool)
        sl = slice(start, start + len(part))
        prob[sl] = np.where(ok, torch.sigmoid(out["logit"]).cpu().numpy(), np.nan)
        stage[sl] = np.where(ok, expected_stage(out["stage_logits"]).cpu().numpy(), np.nan)
        turg[sl] = np.where(ok, out["turgor"].cpu().numpy() * TURGOR_SCALE, np.nan)
    return prob, stage, turg


# --------------------------------------------------------------------------- #
# sensor setups: cameras -> tokens
# --------------------------------------------------------------------------- #
def setups(rs: RunSet, tokens: list[str]) -> dict[str, dict[str, str]]:
    """Every subset of the cameras among `tokens` (v1 names), low-resolution thermal in place of
    thermal, and, for the ablation, the thermal camera's two token kinds alone ("thermal:emb",
    "thermal:feat"; not written to the shared tables)."""
    cams = [c for c in CAMERAS if any(TOKEN_CAMERA[t] == c for t in tokens)]
    env = {"env": "full"} if "env" in tokens else {}
    out = {}
    for k in range(1, len(cams) + 1):
        for combo in combinations(cams, k):
            toks = {t: "full" for t in tokens if TOKEN_CAMERA[t] in combo}
            out["+".join(combo)] = toks | env
            if "thermal" in combo:
                for v in LOW_RES:
                    lows = {t: (v if TOKEN_CAMERA[t] == "thermal" else "full") for t in toks}
                    if all((t, w) in rs.emb for t, w in lows.items()):
                        out["+".join(f"thermal_{v}" if c == "thermal" else c for c in combo)] = lows | env
    if "thermal" in cams and "thermal" in tokens and "tfeat" in tokens:
        out["thermal:emb"] = {"thermal": "full"} | env
        out["thermal:feat"] = {"tfeat": "full"} | env
    return out


# --------------------------------------------------------------------------- #
# cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(rs: RunSet, cfg: V2Config, device: str, ckpt_dir: Path | None = None, verbose=True) -> dict:
    """Out-of-fold P(deficit) for every setup from one model per fold, plus a placebo bed model
    per fold (same inputs and design, trained on 24-29 Aug to tell W1 from W2/W3)."""
    keys = rs.keys
    train_rows = keys["train"].fillna(False).to_numpy()
    pre_rows = keys["date"].between(*(pd.Timestamp(d) for d in core.PLACEBO)).to_numpy()
    keys["y_bed"] = (keys["bed"] != "W1").astype(int)
    control = (keys["bed"] == "W1").to_numpy()
    groups = run_groups(keys)
    all_setups = setups(rs, token_list(rs, cfg))
    n = len(keys)
    prob = {name: np.full(n, np.nan) for name in all_setups}
    pprob = {name: np.full(n, np.nan) for name in all_setups}
    stage, turg = np.full(n, np.nan), np.full(n, np.nan)
    tokens = token_list(rs, cfg)
    for k, test in enumerate(train.plant_folds(keys, cfg.folds, cfg.seed)):
        t0 = time.time()
        train_all = np.setdiff1d(np.arange(n), test)
        in_train = np.zeros(n, bool)
        in_train[train_all] = True
        medians = reference(rs, in_train & control, cfg.min_ref, groups)
        if cfg.window:
            smooth_inputs(rs, cfg.window, cfg.parts)
        tr, va = train.inner_split(keys, train_all[train_rows[train_all]], cfg.val_frac, cfg.seed + k)
        model = fit(rs, tr, va, cfg, device)
        if model is None:
            raise ValueError(
                f"fold {k}: no usable training runs (fewer than {cfg.min_ref} reference controls per run?)"
            )
        ptr, pva = train.inner_split(keys, train_all[pre_rows[train_all]], cfg.val_frac, cfg.seed + k)
        pmodel = fit(rs, ptr, pva, cfg, device, target="y_bed", aux=False, adv=0.0)
        for name, setup in all_setups.items():
            prob[name][test] = predict(model, rs, test, setup, cfg, device)[0]
            pprob[name][test] = predict(pmodel, rs, test, setup, cfg, device)[0]
        _, stage[test], turg[test] = predict(model, rs, test, full_setup(rs, tokens), cfg, device)
        if ckpt_dir is not None:
            save_fold(ckpt_dir, k, rs, cfg, model, pmodel, medians, groups, train_all, test)
        if verbose:
            print(
                f"  fold {k + 1}/{cfg.folds}: {keys.iloc[test]['pid'].nunique()} held-out plants, {time.time() - t0:.0f} s"
            )
    return dict(prob=prob, placebo=pprob, stage=stage, turgor=turg, setups=all_setups)


def save_fold(ckpt_dir: Path, k: int, rs: RunSet, cfg: V2Config, model, pmodel, medians, groups, train_rows, test_rows):
    """fold<k>.pt: the head, its config and tokens, the fold's plants and the reference medians
    of its training controls per run (predict.py references new plants with them)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    g = pd.DataFrame({"date": rs.keys["date"], "hour": rs.keys["hour"], "group": groups}).drop_duplicates("group")
    ref = {f"{t}|{v}": {int(i): torch.from_numpy(m) for i, m in meds.items()} for (t, v), meds in medians.items()}
    common = dict(
        design="v2", config=asdict(cfg), tokens=list(model.tokens), dims=dict(rs.dims),
        sensors=model_cameras(model.tokens),
        train_pids=sorted(rs.keys["pid"].iloc[train_rows].unique()),
        test_pids=sorted(rs.keys["pid"].iloc[test_rows].unique()),
        runs={"date": g["date"].dt.strftime("%Y-%m-%d").tolist(), "hour": g["hour"].astype(int).tolist(),
              "group": g["group"].astype(int).tolist()},
        reference=ref,
    )  # fmt: skip
    torch.save({"model": model.state_dict(), **common}, ckpt_dir / f"fold{k}.pt")
    if pmodel is not None:
        torch.save({"model": pmodel.state_dict(), **common, "target": "y_bed"}, ckpt_dir / f"placebo_fold{k}.pt")


# --------------------------------------------------------------------------- #
# new plants (external test, lock-box): the fold models, averaged (as predict.ensemble)
# --------------------------------------------------------------------------- #
def config_of(ck: dict) -> V2Config:
    names = {f.name for f in dataclasses.fields(V2Config)}
    return V2Config(**{k: (tuple(v) if k == "parts" else v) for k, v in ck["config"].items() if k in names})


def build_from(ck: dict | None, device: str):
    """The saved head of a fold checkpoint (None if the fold had no such model)."""
    if ck is None:
        return None
    cfg = config_of(ck)
    dims = {t: ck["dims"][t] * (1 if t == "env" else len(cfg.parts)) for t in ck["tokens"]}
    model = build(dims, cfg)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval()


def saved_reference(ck: dict) -> tuple[dict, pd.DataFrame]:
    medians = {tuple(k.split("|")): {int(i): m.numpy() for i, m in d.items()} for k, d in ck["reference"].items()}
    runs = pd.DataFrame(ck["runs"])
    runs["date"] = pd.to_datetime(runs["date"])
    return medians, runs


def ensemble(rs: RunSet, folds: list[dict], device: str) -> dict:
    """Mean over the fold models of P(deficit) per setup, the placebo P(bed), stage and turgor.
    Each fold's model sees the new plants referenced to that fold's training controls."""
    tokens, cfg = list(folds[0]["tokens"]), config_of(folds[0])
    for t in tokens:
        rs.dims.setdefault(t, folds[0]["dims"][t])
    all_setups = setups(rs, tokens)
    n, idx = len(rs.keys), np.arange(len(rs.keys))
    prob = {name: np.zeros((len(folds), n)) for name in all_setups}
    pprob = {name: np.zeros((len(folds), n)) for name in all_setups}
    stage, turg = np.zeros((len(folds), n)), np.zeros((len(folds), n))
    for k, ck in enumerate(folds):
        apply_reference(rs, *saved_reference(ck))
        if cfg.window:
            smooth_inputs(rs, cfg.window, cfg.parts)
        model, pmodel = build_from(ck, device), build_from(ck.get("placebo"), device)
        for name, setup in all_setups.items():
            prob[name][k] = predict(model, rs, idx, setup, cfg, device)[0]
            pprob[name][k] = predict(pmodel, rs, idx, setup, cfg, device)[0]
        _, stage[k], turg[k] = predict(model, rs, idx, full_setup(rs, tokens), cfg, device)
        print(f"  fold model {k + 1}/{len(folds)} applied")
    full = "+".join(c for c in CAMERAS if c in {TOKEN_CAMERA[t] for t in tokens})
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = lambda a: np.nanmean(a, 0)
        return dict(
            prob={k: mean(v) for k, v in prob.items()}, placebo={k: mean(v) for k, v in pprob.items()},
            stage=mean(stage), turgor=mean(turg), setups=all_setups, fold_prob=prob.get(full),
        )  # fmt: skip


def shared_setups(cv: dict) -> dict:
    """The cv dict without the ablation-only setups (names with ":"), for train.results."""
    keep = [k for k in cv["setups"] if ":" not in k]
    return dict(
        cv, setups={k: cv["setups"][k] for k in keep}, prob={k: cv["prob"][k] for k in keep},
        placebo={k: cv["placebo"][k] for k in keep},
    )  # fmt: skip


def run_new(caches, folds: list[dict], model_name: str, device: str, n_boot=2000, traits=None, labels=None,
            features: Path | None = FEATURES, only=None, summary_path: Path | None = None):  # fmt: skip
    """predict.run for v2 fold checkpoints: every plant the folds never saw (or `only` those)."""
    from uiwue.fusion import predict as pr

    rs = assemble(caches, features if features is not None and Path(features).exists() else None, exclude=(),
                  only=only, labels=labels)  # fmt: skip
    seen = pr.seen_plants(folds)
    new = ~rs.keys["pid"].isin(seen).to_numpy()
    dropped = sorted(set(rs.keys["pid"][~new]))
    if dropped:
        print(f"  {len(dropped)} plants were in the training folds and are not scored")
    if not new.any():
        raise ValueError("every plant in the cache was used in training; nothing to test")
    rs = _subset(rs, np.flatnonzero(new))
    pids = sorted(rs.keys["pid"].unique())
    print(f"{len(rs.keys)} runs, {len(pids)} new plants, {len(folds)} fold models ({model_name})")
    cv = ensemble(rs, folds, device)
    present = {s: sorted(rs.keys.loc[rs.has[(s, "full")], "pid"].unique()) for s in rs.sensors if (s, "full") in rs.has}
    rs.sensors = model_cameras(folds[0]["tokens"])  # train.results scores the model's own cameras
    tables = pr.external_tables(rs, shared_setups(cv), model_name, n_boot, traits)
    if summary_path is not None:  # the ablation columns (canary days, recovery, severity) for these plants
        clips = pr.clip_plants()
        rows = summarise(rs, cv, model_name, ABLATION_SETUPS, n_boot)
        sub, cv_sub = subset_cv(rs, cv, ~rs.keys["pid"].isin(list(clips)).to_numpy())
        rows += summarise(sub, cv_sub, model_name, ("thermal", "rgb+depth+thermal+hsi"), n_boot, plants="no_clip")
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(summary_path, index=False)
    info = dict(
        model=model_name, design="v2", plants=pids, n_plants=len(pids), runs=len(rs.keys),
        excluded_training_plants=len(dropped),
        plants_per_bed=rs.keys.drop_duplicates("pid")["bed"].value_counts().sort_index().to_dict(),
        plants_with_sensor={s: len(v) for s, v in present.items()},
        clip_plants=sorted(set(pids) & pr.clip_plants()),
        fold_spread_sd=float(np.nanmean(np.nanstd(cv["fold_prob"], 0))) if cv["fold_prob"] is not None else None,
    )  # fmt: skip
    return tables, info, (rs, cv)


def _subset(rs: RunSet, rows: np.ndarray) -> RunSet:
    from uiwue.fusion.predict import subset

    out = subset(rs, rows)
    out.ref, out.has_ref = {}, {}
    return out


# --------------------------------------------------------------------------- #
# scores for the ablation table
# --------------------------------------------------------------------------- #
def summarise(rs: RunSet, cv: dict, name: str, setups_=None, n_boot=2000, plants: str = "all") -> list[dict]:
    """train.score for each setup (group all, W2, W3), plus day 0-2 AUC, the recovery AUC and
    the within-day Spearman of the severity heads."""
    keys = rs.keys
    rows = []
    for s in setups_ or cv["setups"]:
        if s not in cv["prob"]:
            continue
        summary, curves = train.score(keys, cv["prob"][s], cv["placebo"][s], ("all", "W2", "W3"), n_boot)
        daily = train.daily_rows(keys, cv["prob"][s])
        pre = daily[daily["days_since_cutoff"].between(*CANARY_DAYS)]
        for row, curve in zip(summary, curves, strict=True):
            g = row["group"]
            c = curve.set_index("day")
            pc = _curve(pre, g, 0).set_index("day") if len(pre) else pd.DataFrame(columns=["auc"])
            (rec, rlo, rhi), _, _ = _per_plant_auc(daily, g, n_boot, RECOVERY_DAYS)
            extra = {f"day{d}_auc": c["auc"].get(d, np.nan) for d in (0, 1, 2, 3)}
            extra |= {f"day{d}_ci_low": c["ci_low"].get(d, np.nan) for d in (0, 1, 2, 3)}
            extra |= {f"day_m{-d}_auc": pc["auc"].get(d, np.nan) for d in range(CANARY_DAYS[0], CANARY_DAYS[1] + 1)}
            rows.append(dict(variant=name, plants=plants, sensors=s, **row, **extra, recovery_auc=rec,
                             recovery_ci_low=rlo, recovery_ci_high=rhi))  # fmt: skip
    for target, pred, truth in (
        ("stage", cv["stage"], keys["stage"].where(keys["stage"] >= 0)),
        ("turgor", cv["turgor"], keys["turgor"]),
    ):  # the severity heads, all sensors (one value per variant, repeated on its rows)
        r = train.within_day_rho(keys, pred, truth.to_numpy(float))
        full = "+".join(c for c in CAMERAS if c in "+".join(cv["setups"]).split("+"))
        rp = train.within_day_rho(keys, cv["prob"][full], truth.to_numpy(float)) if full in cv["prob"] else []
        for row in rows:
            row[f"{target}_rho_median"] = float(r["rho"].median()) if len(r) else np.nan
            # the deficit probability itself as a severity score, for comparison with the heads
            row[f"{target}_rho_prob_median"] = float(rp["rho"].median()) if len(rp) else np.nan
    return rows


def subset_cv(rs: RunSet, cv: dict, mask: np.ndarray) -> tuple[RunSet, dict]:
    from uiwue.fusion.predict import subset

    rows = np.flatnonzero(mask)
    sub = subset(rs, rows)
    return sub, dict(
        prob={k: v[rows] for k, v in cv["prob"].items()}, placebo={k: v[rows] for k, v in cv["placebo"].items()},
        stage=cv["stage"][rows], turgor=cv["turgor"][rows], setups=cv["setups"],
    )  # fmt: skip


def clip_pids() -> set[str]:
    from uiwue.fusion.predict import clip_plants

    return clip_plants()


# --------------------------------------------------------------------------- #
# the benchmark and the size baseline on the same plants
# --------------------------------------------------------------------------- #
def benchmark_daily(f: pd.DataFrame, cols: list[str], window=core.BASELINE) -> pd.DataFrame:
    """uiwue.detect.benchmark.daily_features with a chosen baseline window: per plant-day, the median
    over the 12 and 15 h runs passing QC of each feature minus the plant's baseline-window median
    at the same slot."""
    d = core.add_design(core.normalize_keys(f))
    lo, hi = (pd.Timestamp(w) for w in window)
    by = ["bed", "plant", "date", "days_since_cutoff"]
    out = None
    for c in cols:
        ok = d["thermal_qc_pass"].fillna(True).astype(bool) & d[c].notna()
        base = d[ok & d["date"].between(lo, hi)].groupby(["bed", "plant", "hour"])[c].median().rename("_b")
        v = d[c] - d[["bed", "plant", "hour"]].join(base, on=["bed", "plant", "hour"])["_b"]
        keep = d["hour"].isin([12, 15]) & ok & v.notna()
        x = d.loc[keep, by].assign(**{c: v[keep].to_numpy()}).groupby(by, as_index=False)[c].median()
        out = x if out is None else out.merge(x, on=by, how="outer")
    out["y"] = (out["bed"] != "W1").astype(int)
    out["pid"] = out["bed"] + "_" + out["plant"]
    return out.sort_values(["pid", "date"]).reset_index(drop=True)


def benchmark_rows(
    features: Path, pids: set[str], eval_pids: set[str] | None = None, n_boot=2000, groups=("all", "W2", "W3"),
    plants: str = "all", baseline=core.BASELINE,
) -> list[dict]:  # fmt: skip
    """The benchmark's thermal logistic model (uiwue.detect.benchmark.evaluate, leave one plant
    out) on `eval_pids`, with leaf minus control-bed leaf from the controls among `pids` only
    (the lock-box never enters it), plus its scores on the canary days before the cutoff."""
    from uiwue.detect.benchmark import evaluate, lopo_predict
    from uiwue.thermal import control_reference

    f = core.normalize_keys(pd.read_parquet(features))
    f = f[(f["bed"] + "_" + f["plant"]).isin(pids)].reset_index(drop=True)
    passed = f.assign(leaf=f["thermal_leaf_t"].where(f["thermal_qc_pass"]))
    f["thermal_leaf_ref_dt"] = f["thermal_leaf_t"] - control_reference(passed, "leaf")
    if eval_pids is not None:
        f = f[(f["bed"] + "_" + f["plant"]).isin(eval_pids)].reset_index(drop=True)
    sets = {
        "thermal": ["thermal_leaf_t", "thermal_leaf_t_p10", "thermal_leaf_air_dt", "thermal_leaf_ref_dt"],
        "thermal_r80": ["thermal_leaf_t_r80"],
    }
    daily = benchmark_daily(f, sorted({c for v in sets.values() for c in v}), baseline)
    out = []
    for name, feats in sets.items():
        summary, curves = evaluate(daily, feats, "logistic", groups, n_boot)
        stress = daily[daily["days_since_cutoff"].between(*TRAIN_DAYS)]
        pre = daily[daily["days_since_cutoff"].between(*CANARY_DAYS)].copy()
        pre["prob"] = lopo_predict(stress, feats, "logistic", predict=pre)
        # the detection model's own score over the placebo window (train.score's placebo_auc_main)
        plac = daily[daily["date"].between(*(pd.Timestamp(d) for d in core.PLACEBO))].copy()
        plac["prob"] = lopo_predict(stress, feats, "logistic", predict=plac)
        for row, curve in zip(summary, curves, strict=True):
            c = curve.set_index("day")
            pc = _curve(pre, row["group"], 0).set_index("day")
            row["placebo_auc_main"] = _per_plant_auc(plac, row["group"], 0)[0][0]
            extra = {f"day{d}_auc": c["auc"].get(d, np.nan) for d in (0, 1, 2, 3)}
            extra |= {f"day{d}_ci_low": c["ci_low"].get(d, np.nan) for d in (0, 1, 2, 3)}
            extra |= {f"day_m{-d}_auc": pc["auc"].get(d, np.nan) for d in range(CANARY_DAYS[0], CANARY_DAYS[1] + 1)}
            out.append(dict(variant="benchmark_logistic", plants=plants, sensors=name, **row, **extra))
    out += _benchmark_prelabels(daily, sets["thermal"], out, groups, n_boot, plants)
    return out


def _benchmark_prelabels(daily, feats, rows, groups, n_boot, plants) -> list[dict]:
    """The benchmark's features and model trained the way the fusion models are: every plant-day
    before the cutoff is a negative and the deficit beds' day 0 is left out (variant
    benchmark_logistic_prelabels). It separates the effect of the labels from that of the model."""
    from uiwue.detect.benchmark import lopo_predict

    d = daily["days_since_cutoff"]
    deficit = daily["bed"] != "W1"
    lab = daily.assign(y=(deficit & (d >= 1)).astype(int))
    tr = lab[(d <= TRAIN_DAYS[1]) & ~(deficit & (d == 0))]
    frames = {
        "stress": daily[d.between(*TRAIN_DAYS)],
        "pre": daily[d.between(*CANARY_DAYS)],
        "plac": daily[daily["date"].between(*(pd.Timestamp(x) for x in core.PLACEBO))],
    }
    scored = {}
    for k, fr in frames.items():
        fr = fr.copy()
        fr["prob"] = lopo_predict(tr, feats, "logistic", predict=fr)
        scored[k] = fr
    base = {r["group"]: r for r in rows if r["sensors"] == "thermal" and r["variant"] == "benchmark_logistic"}
    out = []
    for g in groups:
        if g not in base:
            continue
        curve = _curve(scored["stress"], g, n_boot)
        (e, elo, ehi), _, _ = _per_plant_auc(scored["stress"], g, n_boot, EARLY_DAYS)
        (fa, _, _), _, _ = _per_plant_auc(scored["stress"], g, 0, FULL_DAYS)
        c, pc = curve.set_index("day"), _curve(scored["pre"], g, 0).set_index("day")
        row = dict(base[g])  # the placebo bed model does not depend on the labels: the benchmark's
        row.update(
            variant="benchmark_logistic_prelabels", plants=plants, early_auc=e, early_ci_low=elo, early_ci_high=ehi,
            full_auc=fa, onset_day=onset_day(curve) if len(curve) else np.nan,
            placebo_auc_main=_per_plant_auc(scored["plac"], g, 0)[0][0],
            **{f"day{x}_auc": c["auc"].get(x, np.nan) for x in (0, 1, 2, 3)},
            **{f"day{x}_ci_low": c["ci_low"].get(x, np.nan) for x in (0, 1, 2, 3)},
            **{f"day_m{-x}_auc": pc["auc"].get(x, np.nan) for x in range(CANARY_DAYS[0], CANARY_DAYS[1] + 1)},
        )  # fmt: skip
        out.append(row)
    return out


def _control_ref(f: pd.DataFrame, ref_pids: set[str], min_controls: int = 3) -> pd.Series:
    """Median leaf temperature of the `ref_pids` controls in the same run (the others for one of
    them): uiwue.thermal.control_reference restricted to reference plants."""
    pid = f["bed"] + "_" + f["plant"]
    leaf = f["thermal_leaf_t"].where(f["thermal_qc_pass"].fillna(True).astype(bool))
    is_ref = ((f["bed"] == "W1") & pid.isin(ref_pids)).to_numpy()
    out = pd.Series(np.nan, index=f.index)
    for g in f.groupby(["date", "hour"]).groups.values():
        g = np.asarray(g)
        r = g[is_ref[f.index.get_indexer(g)]]
        vals = leaf.loc[r].dropna()
        if len(vals) < min_controls:
            continue
        out.loc[g] = float(vals.median())
        for i in vals.index:
            out.loc[i] = float(vals.drop(i).median())
    return out


def benchmark_external(
    features: Path, dev_pids: set[str], test_pids: set[str], n_boot=2000, groups=("all", "W2", "W3"),
    baseline=core.BASELINE,
) -> tuple[list[dict], list[pd.DataFrame]]:  # fmt: skip
    """The benchmark's thermal logistic model fitted on every development plant-day (days 0-12)
    and applied to new plants (the lock-box), with its placebo bed model fitted on the
    development plants' placebo window. Leaf minus control-bed leaf uses development controls."""
    from uiwue.detect.benchmark import make_model

    f = core.normalize_keys(pd.read_parquet(features))
    pid = f["bed"] + "_" + f["plant"]
    f = f[pid.isin(dev_pids | test_pids)].reset_index(drop=True)
    f["thermal_leaf_ref_dt"] = f["thermal_leaf_t"] - _control_ref(f, dev_pids)
    feats = ["thermal_leaf_t", "thermal_leaf_t_p10", "thermal_leaf_air_dt", "thermal_leaf_ref_dt"]
    daily = benchmark_daily(f, feats, baseline)
    dev, test = daily[daily["pid"].isin(dev_pids)], daily[daily["pid"].isin(test_pids)].copy()
    ok = lambda d: d[feats].notna().any(axis=1)
    tr = dev[dev["days_since_cutoff"].between(*TRAIN_DAYS) & ok(dev)]
    m = make_model("logistic").fit(tr[feats].to_numpy(float), tr["y"].to_numpy())
    test["prob"] = np.where(ok(test), m.predict_proba(test[feats].to_numpy(float))[:, 1], np.nan)
    pw = [pd.Timestamp(d) for d in core.PLACEBO]
    ptr = dev[dev["date"].between(*pw) & ok(dev)]
    pm = make_model("logistic").fit(ptr[feats].to_numpy(float), ptr["y"].to_numpy())
    pre = test[test["date"].between(*pw)].copy()
    pre["prob"] = np.where(ok(pre), pm.predict_proba(pre[feats].to_numpy(float))[:, 1], np.nan)
    stress = test[test["days_since_cutoff"].between(*TRAIN_DAYS)]
    canary = test[test["days_since_cutoff"].between(*CANARY_DAYS)]
    rows, curves = [], []
    for g in groups:
        curve = _curve(stress, g, n_boot)
        (e, elo, ehi), npos, nneg = _per_plant_auc(stress, g, n_boot, EARLY_DAYS)
        (fa, _, _), _, _ = _per_plant_auc(stress, g, 0, FULL_DAYS)
        (pa, plo, phi), _, _ = _per_plant_auc(pre, g, n_boot)
        (pmain, _, _), _, _ = _per_plant_auc(test[test["date"].between(*pw)], g, 0)
        c, pc = curve.set_index("day"), _curve(canary, g, 0).set_index("day")
        rows.append(dict(
            sensors="thermal", model="logistic", group=g, early_auc=e, early_ci_low=elo, early_ci_high=ehi,
            full_auc=fa, onset_day=onset_day(curve) if len(curve) else np.nan, placebo_auc=pa,
            placebo_ci_low=plo, placebo_ci_high=phi,
            bed_effect=bool(np.isfinite(plo) and (plo > 0.5 or phi < 0.5) and max(pa, 1 - pa) >= 0.7),
            placebo_auc_main=pmain, n_deficit=npos, n_control=nneg, setup="Thermal", n_sensors=1,
            n_features=len(feats), cost_rank=5, is_size_baseline=False, size_column=None,
            **{f"day{d}_auc": c["auc"].get(d, np.nan) for d in (0, 1, 2, 3)},
            **{f"day_m{-d}_auc": pc["auc"].get(d, np.nan) for d in range(CANARY_DAYS[0], CANARY_DAYS[1] + 1)},
        ))  # fmt: skip
        name = "model:thermal:logistic"
        curves.append(curve.assign(group=g, signal=name, kind="model", column=name, model="logistic"))
    return rows, curves


def bench_lockbox(
    features: Path, n_boot=2000, out_dir: Path | None = None, baseline=core.BASELINE,
    canary_path: Path = V2_CACHE / "lockbox_logistic_summary.parquet",
) -> None:  # fmt: skip
    """benchmark_external on the lock-box (all, no-clip), into fusion_lockbox_{benchmark,auc_by_day}."""
    from uiwue.fusion import predict as pr

    f = pd.read_parquet(features, columns=["bed", "plant"]).drop_duplicates()
    pids = set(f["bed"] + "_" + f["plant"])
    test, dev = pids & set(lockbox.LOCKBOX), pids - set(lockbox.LOCKBOX)
    clips = pr.clip_plants()
    bench, curves = [], []
    for plants, sub in (("all", test), ("no_clip", test - clips)):
        rows, cs = benchmark_external(features, dev, sub, n_boot, baseline=baseline)
        tag = dict(plants=plants, n_plants=len(sub))
        bench += [r | tag for r in rows]
        curves += [c.assign(**tag) for c in cs]
    canary = [c for c in pd.DataFrame(bench).columns if c.startswith("day")]
    tables = {
        "fusion_lockbox_benchmark": pd.DataFrame(bench).drop(columns=canary),
        "fusion_lockbox_auc_by_day": pd.concat(curves, ignore_index=True),
    }
    out = pr.write(tables, "logistic", out_dir)
    Path(canary_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bench).to_parquet(canary_path, index=False)
    print(
        pd.DataFrame(bench)[
            [
                "plants",
                "group",
                "early_auc",
                "early_ci_low",
                "onset_day",
                "day_m1_auc",
                "placebo_auc",
                "n_deficit",
                "n_control",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(f"-> {out}")


RGB_SIZE = V2_CACHE / "rgb_size.parquet"


def size_rows(pids: set[str], n_boot=2000) -> list[dict]:
    """Plant size alone (predict.size_baseline): depth_projected_area where Challenge 3 has it
    (plants "with_size"), and RGB green cover (uiwue.fusion.features --rgb-size) for every plant
    with RGB (plants "all"; the Master Plan's second size column)."""
    from uiwue.fusion.predict import size_baseline

    out = []
    traits = core.read_table("threed", "traits_runs")
    rows, _, have = size_baseline(traits, pids, n_boot=n_boot)
    out += [dict(variant="size_only", plants="with_size", n_plants=len(have), **r) for r in rows]
    if RGB_SIZE.exists():
        green = core.normalize_keys(pd.read_parquet(RGB_SIZE))
        green = green[green["rgb_qc_pass"].fillna(True).astype(bool)] if "rgb_qc_pass" in green else green
        rows, _, have = size_baseline(green[[*core.KEYS, "rgb_green_cover"]], pids, n_boot=n_boot)
        out += [dict(variant="size_only_green_cover", plants="all", n_plants=len(have), **r) for r in rows]
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
ABLATION_SETUPS = (
    "thermal", "thermal:emb", "thermal:feat", "thermal_r80", "depth", "rgb", "hsi", "rgb+depth+thermal",
    "rgb+depth+thermal+hsi",
)  # fmt: skip


def ablate(
    rs: RunSet, names: list[str], device: str, n_boot: int, out: Path, features: Path | None, tag: str = "",
    baseline=core.BASELINE, **over,
) -> None:  # fmt: skip
    rows = []
    clips = clip_pids()
    pids = set(rs.keys["pid"])
    with_size = set()
    for name in names:
        cfg = make_config(name, **over)
        t0 = time.time()
        cv = cross_validate(rs, cfg, device, verbose=False)
        secs = time.time() - t0
        label = name + tag
        rows += [r | dict(seconds=round(secs, 1)) for r in summarise(rs, cv, label, ABLATION_SETUPS, n_boot)]
        sub, cv_sub = subset_cv(rs, cv, ~rs.keys["pid"].isin(list(clips)).to_numpy())
        rows += summarise(sub, cv_sub, label, ("thermal", "rgb+depth+thermal+hsi"), n_boot, plants="no_clip")
        traits = core.read_table("threed", "traits_runs")
        if traits is not None:
            with_size = set(traits["bed"] + "_" + traits["plant"]) & pids
            sub, cv_sub = subset_cv(rs, cv, rs.keys["pid"].isin(list(with_size)).to_numpy())
            rows += summarise(sub, cv_sub, label, ("thermal",), n_boot, plants="with_size")
        b = pd.DataFrame(rows)
        show = b[(b["variant"] == label) & (b["group"] == "all") & (b["plants"] == "all")]
        print(f"{name}: {secs:.0f} s")
        print(
            show[["sensors", "early_auc", "early_ci_low", "onset_day", "day_m1_auc", "day0_auc", "day1_auc",
                  "day2_auc", "placebo_auc", "placebo_auc_main"]].round(3).to_string(index=False)
        )  # fmt: skip
        pd.DataFrame(rows).to_parquet(out, index=False)
    if features is not None:
        bench = lambda ev, plants: [
            r | dict(variant=r["variant"] + tag)
            for r in benchmark_rows(features, pids, ev, n_boot, plants=plants, baseline=baseline)
        ]
        rows += bench(None, "all") + bench(pids - clips, "no_clip")
        if with_size:
            rows += bench(with_size, "with_size")
        rows += size_rows(pids, n_boot)
    df = pd.DataFrame(rows)
    df["n_plants_total"] = len(pids)
    df.to_parquet(out, index=False)
    show = df[(df["group"] == "all") & df["sensors"].isin(["thermal"])]
    print(show[["variant", "plants", "early_auc", "early_ci_low", "onset_day", "placebo_auc_main"]].to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ablate", "train", "bench-lockbox"])
    ap.add_argument("--name", default="fusion_v2", help="train: model name (tables, runs/fusion/<name>)")
    ap.add_argument("--cache", type=Path, nargs="+", default=[CACHE])
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--variant", default="delta+ref+tfeat", help="train: the chosen design")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--max-w1", type=int, default=None, help="ablate: keep at most this many W1 plants (bed balance)")
    ap.add_argument(
        "--baseline", nargs=2, default=None, help="sensitivity: baseline window (two dates) instead of core.BASELINE"
    )
    ap.add_argument("--tag", default="", help="suffix for variant names in the ablation table (e.g. _base0902)")
    ap.add_argument("--no-bench", action="store_true", help="ablate: skip the benchmark and size rows")
    ap.add_argument("--min-ref", type=int, default=V2Config.min_ref, help="control plants needed for a run's reference")
    a = ap.parse_args(argv)
    t0 = time.time()
    window = tuple(pd.Timestamp(d).date() for d in a.baseline) if a.baseline else core.BASELINE
    if a.command == "bench-lockbox":
        bench_lockbox(a.features, a.n_boot, a.out, window)
        print(f"benchmark on the lock-box: {time.time() - t0:.0f} s")
        return
    rs = assemble(a.cache, a.features if a.features.exists() else None, baseline=window)
    if a.max_w1 is not None:
        if a.max_w1 < 0:  # auto: as many W1 plants as the larger deficit bed has
            a.max_w1 = int(rs.keys.drop_duplicates("pid")["bed"].value_counts().drop("W1", errors="ignore").max())
        w1 = sorted(rs.keys.loc[rs.keys["bed"] == "W1", "pid"].unique())
        drop = set(np.random.default_rng(SEED).permutation(w1)[a.max_w1 :])
        rs = _subset(rs, np.flatnonzero(~rs.keys["pid"].isin(drop).to_numpy()))
    per_bed = rs.keys.drop_duplicates("pid")["bed"].value_counts().sort_index().to_dict()
    print(f"{len(rs.keys)} runs, {rs.keys['pid'].nunique()} dev plants {per_bed}, tokens {rs.sensors}")
    if a.command == "ablate":
        out = a.out or (V2_CACHE / "ablation.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        bench = a.features if a.features.exists() and not a.no_bench else None
        ablate(rs, a.variants, a.device, a.n_boot, out, bench, tag=a.tag, baseline=window, epochs=a.epochs,
               min_ref=a.min_ref)  # fmt: skip
        print(f"ablation: {time.time() - t0:.0f} s -> {out}")
        return
    cfg = make_config(a.variant, epochs=a.epochs, n_boot=a.n_boot, min_ref=a.min_ref)
    name = a.name
    cv = cross_validate(rs, cfg, a.device, ckpt_dir=a.runs_dir / name)
    named = shared_setups(cv)
    tables = train.results(camera_view(rs, token_list(rs, cfg)), named, name, cfg.n_boot)
    scores = tables.pop("fusion_scores_runs")  # v1's table keeps v1's scores; v2 writes its own
    out = train.write(tables, name, a.out)
    if name == "fusion_v2":
        scores = scores.rename(columns={k: k.replace("detect_fusion_", "detect_fusion_v2_") for k in scores})
        scores["date"] = pd.to_datetime(scores["date"]).dt.date
        scores.to_parquet(out / "fusion_v2_scores_runs.parquet", index=False)
    train.save_oof(rs, named, a.runs_dir / name)
    info = dict(
        model=name, variant=a.variant, finished=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        seconds=round(time.time() - t0, 1), tokens=token_list(rs, cfg), dims=rs.dims, runs=len(rs.keys),
        plants=int(rs.keys["pid"].nunique()), plants_per_bed=per_bed, config=asdict(cfg), device=a.device,
        lockbox_excluded=len(lockbox.LOCKBOX),
    )  # fmt: skip
    (out / f"{name}_run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"{name}: done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
