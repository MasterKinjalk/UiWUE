"""Train and evaluate the fusion model on 5 folds grouped by plant.

    python -m uiwue.fusion.train                         # stage 1: frozen encoders, head only
    python -m uiwue.fusion.train --stage lora --lora-sensor hsi       # stage 2
    python -m uiwue.fusion.train --stage lora --lora-sensor thermal   # stage 3

Stage 1 reads the cached embeddings (python -m uiwue.fusion.embed). The LoRA
stages read raw images for their sensor and cached embeddings for the others.

Rules (Jonathan-Work-Plan section 3): whole plants per fold, stratified by bed;
no date, bed, hour or position inputs; embeddings enter with their change from
the plant's own 30 Aug - 2 Sep embedding at the same hour; early stopping on an
inner validation split of training plants; a placebo model trained to tell
the beds apart before the cutoff runs beside every result.

Evaluation reuses the benchmark's scoring (plant-day = mean of the 12 and 15 h
runs; per-day AUC with a plant bootstrap; onset day; early and full AUC), so the
two tracks are compared on the same footing. Writes derived/detect/:

  fusion_benchmark.parquet    every sensor setup, as benchmark.parquet (model = fusion stage)
  fusion_auc_by_day.parquet   per-day AUC curves
  fusion_hsi.parquet          Part B: HSI spatial vs spectral detail, 3 vs 300 bands, 20 nm windows
  fusion_severity.parquet     Part C: within-day rank agreement of the stage and turgor heads
  fusion_scores_runs.parquet  detect_fusion_prob / _stage / _turgor per run (out of fold)
  fusion_run_info.json
Checkpoints go to runs/fusion/<stage>/ (not in git): fold<k>.pt (the fusion head, its
standardiser, config and the fold's plants; for LoRA stages also the tuned adapter weights)
and placebo_fold<k>.pt (the bed model). uiwue.fusion.predict scores new plants with them.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uiwue import core
from uiwue.detect.benchmark import FULL_DAYS, TRAIN_DAYS, _curve, _per_plant_auc
from uiwue.detect.data import SETUP_LABEL, cost_rank
from uiwue.detect.run import SCORES_CATALOGUE
from uiwue.fusion import inputs
from uiwue.fusion.embed import CACHE, load_table
from uiwue.fusion.fusion import FusionModel, expected_stage, fusion_loss
from uiwue.fusion.labels import TURGOR_SCALE, run_labels
from uiwue.stats import GROUPS, onset_day
from uiwue.stats.core import EARLY_DAYS, SEED

OUT = "detect"
RUNS_DIR = core.REPO / "runs" / "fusion"
SENSOR_ORDER = ("rgb", "depth", "thermal", "hsi")  # cost order, as the benchmark
LOW_RES = {"thermal": ("r160", "r80")}


@dataclass
class Config:
    folds: int = 5
    epochs: int = 100
    patience: int = 10
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    d_model: int = 256
    layers: int = 2
    heads: int = 4
    modality_dropout: float = 0.3
    val_frac: float = 0.2
    n_boot: int = 2000
    seed: int = SEED
    # LoRA stages
    lora_rank: int = 8
    lora_blocks: int = 4
    lora_lr: float = 1e-4
    lora_epochs: int = 15
    lora_batch: int = 16
    workers: int = 4
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# data: aligned embeddings, baselines and labels
# --------------------------------------------------------------------------- #
@dataclass
class RunSet:
    keys: pd.DataFrame  # bed, plant, date, hour + labels (+ pid)
    sensors: list[str]
    dims: dict[str, int]
    emb: dict[tuple[str, str], np.ndarray]  # (sensor, variant) -> (N, dim), zeros where absent
    base: dict[tuple[str, str], np.ndarray]
    has: dict[tuple[str, str], np.ndarray]  # (N,) bool: embedding and baseline both present

    def variants(self, sensor: str) -> list[str]:
        return [v for s, v in self.emb if s == sensor]


def baseline_of(keys: pd.DataFrame, emb: np.ndarray, has: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each run's baseline: mean embedding of the plant's runs in the baseline window at the
    same hour (any hour if none at that hour). Returns (base, ok)."""
    lo, hi = (pd.Timestamp(d) for d in core.BASELINE)
    in_win = has & keys["date"].between(lo, hi).to_numpy()
    base, ok = np.zeros_like(emb), np.zeros(len(keys), bool)
    plant = (keys["bed"] + "_" + keys["plant"]).to_numpy()
    hour = keys["hour"].to_numpy()
    for pid in np.unique(plant):
        rows = plant == pid
        pool = rows & in_win
        if not pool.any():
            continue
        any_hour = emb[pool].mean(0)
        for h in np.unique(hour[rows]):
            same = pool & (hour == h)
            target = rows & (hour == h)
            base[target] = emb[same].mean(0) if same.any() else any_hour
            ok[target] = True
    return base, ok


def load_tables(caches: Path | list[Path], sensors=SENSOR_ORDER) -> dict[tuple[str, str], tuple]:
    """Every cached (sensor, variant) table; several cache folders (e.g. plants embedded in
    batches) are concatenated, keeping the first copy of a run."""
    caches = [Path(caches)] if isinstance(caches, str | Path) else [Path(c) for c in caches]
    parts: dict[tuple[str, str], list] = {}
    for cache in caches:
        for s in sensors:
            for p in sorted(cache.glob(f"emb_{s}_*.npz")):
                parts.setdefault((s, p.stem.split(f"emb_{s}_", 1)[1]), []).append(load_table(p))
    tables = {}
    order = sorted(parts, key=lambda sv: (list(sensors).index(sv[0]), sv[1] != "full", sv[1]))  # full first
    for sv in order:
        got = parts[sv]
        keys = pd.concat([k for k, _ in got], ignore_index=True)
        emb = np.concatenate([e for _, e in got])
        first = ~keys.duplicated(core.KEYS).to_numpy()
        tables[sv] = (keys[first].reset_index(drop=True), emb[first])
    return tables


def assemble(cache: Path | list[Path] = CACHE, sensors=SENSOR_ORDER, labels: pd.DataFrame | None = None) -> RunSet:
    """Align every cached (sensor, variant) table on the union of runs and attach labels."""
    tables = load_tables(cache, sensors)
    present_sensors = [s for s in sensors if (s, "full") in tables]
    if not present_sensors:
        raise FileNotFoundError(f"no embeddings in {cache}; run python -m uiwue.fusion.embed first")
    keys = pd.concat([tables[(s, "full")][0] for s in present_sensors]).drop_duplicates()
    keys = keys.sort_values(core.KEYS, ignore_index=True)
    lab = run_labels(keys) if labels is None else labels
    keys = keys.merge(lab, on=core.KEYS, how="left")
    keys["pid"] = keys["bed"] + "_" + keys["plant"]

    emb, base, has, dims = {}, {}, {}, {}
    index = pd.MultiIndex.from_frame(keys[core.KEYS])
    for (s, v), (k, e) in tables.items():
        if s not in present_sensors:
            continue
        pos = index.get_indexer(pd.MultiIndex.from_frame(k[core.KEYS]))
        full = np.zeros((len(keys), e.shape[1]), np.float32)
        got = np.zeros(len(keys), bool)
        full[pos[pos >= 0]], got[pos[pos >= 0]] = e[pos >= 0], True
        bad = ~np.isfinite(full).all(1)
        full[bad], got[bad] = 0, False
        b, ok = baseline_of(keys, full, got)
        emb[(s, v)], base[(s, v)], has[(s, v)], dims[s] = full, b, got & ok, e.shape[1]
    return RunSet(keys, present_sensors, dims, emb, base, has)


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def plant_folds(keys: pd.DataFrame, n: int, seed: int) -> list[np.ndarray]:
    """Plants dealt round-robin into n folds within each bed (stratified, grouped)."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, g in keys.drop_duplicates("pid").groupby("bed"):
        pids = g["pid"].to_numpy()
        rng.shuffle(pids)
        start = int(rng.integers(n))
        for j, p in enumerate(pids):
            fold_of[p] = (start + j) % n
    f = keys["pid"].map(fold_of).to_numpy()
    return [np.flatnonzero(f == k) for k in range(n)]


def inner_split(keys: pd.DataFrame, idx: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ~frac of the training plants per bed for early stopping."""
    rng = np.random.default_rng(seed)
    sub = keys.iloc[idx]
    val_p = []
    for _, g in sub.drop_duplicates("pid").groupby("bed"):
        p = g["pid"].to_numpy()
        k = max(1, round(len(p) * frac)) if len(p) > 1 else 0
        val_p += list(rng.choice(p, k, replace=False))
    is_val = sub["pid"].isin(val_p).to_numpy()
    return idx[~is_val], idx[is_val]


# --------------------------------------------------------------------------- #
# batches and training
# --------------------------------------------------------------------------- #
def full_setup(rs: RunSet) -> dict[str, str]:
    return {s: "full" for s in rs.sensors}


def batch(rs: RunSet, idx: np.ndarray, setup: dict[str, str], device: str, target: str = "y_deficit"):
    """Tensors for rows idx; sensors not in `setup` are absent."""
    now, base, present = {}, {}, []
    for s in rs.sensors:
        v = setup.get(s)
        if v is None or (s, v) not in rs.emb:
            now[s] = torch.zeros(len(idx), rs.dims[s], device=device)
            base[s] = torch.zeros_like(now[s])
            present.append(np.zeros(len(idx), bool))
        else:
            now[s] = torch.from_numpy(rs.emb[(s, v)][idx]).to(device)
            base[s] = torch.from_numpy(rs.base[(s, v)][idx]).to(device)
            present.append(rs.has[(s, v)][idx])
    k = rs.keys.iloc[idx]
    return dict(
        now=now,
        base=base,
        present=torch.from_numpy(np.stack(present, 1)).to(device),
        y=torch.from_numpy(k[target].to_numpy(np.float32)).to(device),
        stage=torch.from_numpy(pd.to_numeric(k["stage"]).fillna(-1).to_numpy(np.int64)).to(device),
        turgor=torch.from_numpy(k["turgor"].to_numpy(np.float32) / TURGOR_SCALE).to(device),
    )


def _usable(rs: RunSet, idx: np.ndarray, setup: dict[str, str]) -> np.ndarray:
    ok = np.zeros(len(idx), bool)
    for s, v in setup.items():
        if (s, v) in rs.has:
            ok |= rs.has[(s, v)][idx]
    return idx[ok]


def build_model(dims: dict[str, int], cfg: Config) -> FusionModel:
    return FusionModel(
        dims, d_model=cfg.d_model, layers=cfg.layers, heads=cfg.heads, modality_dropout=cfg.modality_dropout
    )


def fit_head(rs: RunSet, train_idx, val_idx, cfg: Config, device: str, target="y_deficit", aux=True):
    """Stage 1: train the fusion head on cached embeddings, early stopping on val loss."""
    torch.manual_seed(cfg.seed)
    setup = full_setup(rs)
    train_idx, val_idx = _usable(rs, train_idx, setup), _usable(rs, val_idx, setup)
    model = build_model(rs.dims, cfg).to(device)
    if len(train_idx) == 0:
        return None  # nothing to learn from (e.g. no pre-cutoff runs for the placebo model)
    for s in rs.sensors:  # standardisation from this fold's training plants only (rule 6)
        rows = train_idx[rs.has[(s, "full")][train_idx]]
        if len(rows):
            e, b = rs.emb[(s, "full")][rows], rs.base[(s, "full")][rows]
            model.fit_standardiser(s, torch.from_numpy(e).to(device), torch.from_numpy(e - b).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best, best_loss, wait = copy.deepcopy(model.state_dict()), np.inf, 0
    for _ in range(cfg.epochs):
        model.train()
        for chunk in np.array_split(rng.permutation(train_idx), max(1, len(train_idx) // cfg.batch)):
            b = batch(rs, chunk, setup, device, target)
            if not aux:
                b["stage"][:], b["turgor"][:] = -1, float("nan")
            loss = fusion_loss(model(b["now"], b["base"], b["present"]), b["y"], b["stage"], b["turgor"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if len(val_idx) == 0:
            best = copy.deepcopy(model.state_dict())
            continue
        val = _val_loss(model, rs, val_idx, setup, device, target)
        if val < best_loss - 1e-4:
            best, best_loss, wait = copy.deepcopy(model.state_dict()), val, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best)
    return model.eval()


@torch.no_grad()
def _val_loss(model, rs, idx, setup, device, target) -> float:
    model.eval()
    b = batch(rs, idx, setup, device, target)
    out = model(b["now"], b["base"], b["present"])
    return float(torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], b["y"]))


@torch.no_grad()
def predict(model: FusionModel, rs: RunSet, idx: np.ndarray, setup: dict[str, str], device: str, chunk=4096):
    """(prob, expected stage, turgor shift) for rows idx; NaN where no sensor of the setup is present."""
    prob, stage, turg = (np.full(len(idx), np.nan) for _ in range(3))
    if model is None:
        return prob, stage, turg
    model.eval()
    for start in range(0, len(idx), chunk):
        part = idx[start : start + chunk]
        b = batch(rs, part, setup, device)
        out = model(b["now"], b["base"], b["present"])
        ok = b["present"].any(1).cpu().numpy()
        sl = slice(start, start + len(part))
        prob[sl] = np.where(ok, torch.sigmoid(out["logit"]).cpu().numpy(), np.nan)
        stage[sl] = np.where(ok, expected_stage(out["stage_logits"]).cpu().numpy(), np.nan)
        turg[sl] = np.where(ok, out["turgor"].cpu().numpy() * TURGOR_SCALE, np.nan)
    return prob, stage, turg


# --------------------------------------------------------------------------- #
# sensor setups evaluated on the one trained model
# --------------------------------------------------------------------------- #
def setups(rs: RunSet) -> dict[str, dict[str, str]]:
    """Every subset of the sensors present, plus low-resolution thermal in place of thermal."""
    present = [s for s in SENSOR_ORDER if s in rs.sensors]
    out = {}
    for k in range(1, len(present) + 1):
        for combo in combinations(present, k):
            out["+".join(combo)] = {s: "full" for s in combo}
            for s, lows in LOW_RES.items():
                if s in combo:
                    for v in lows:
                        if (s, v) in rs.emb:
                            name = "+".join(f"{c}_{v}" if c == s else c for c in combo)
                            out[name] = {c: (v if c == s else "full") for c in combo}
    return out


def hsi_setups(rs: RunSet) -> dict[str, dict[str, str]]:
    return {f"hsi@{v}": {"hsi": v} for v in rs.variants("hsi")}


# --------------------------------------------------------------------------- #
# scoring (same helpers as the benchmark)
# --------------------------------------------------------------------------- #
def daily_rows(keys: pd.DataFrame, prob: np.ndarray, hours=(12, 15)) -> pd.DataFrame:
    d = keys[["bed", "plant", "pid", "date", "hour", "days_since_cutoff"]].assign(prob=prob)
    d = d[d["hour"].isin(hours)].dropna(subset=["prob"])
    return d.groupby(["bed", "plant", "pid", "date", "days_since_cutoff"], as_index=False)["prob"].mean()


def score(keys, prob, placebo_prob, groups, n_boot) -> tuple[list[dict], list[pd.DataFrame]]:
    daily = daily_rows(keys, prob)
    stress = daily[daily["days_since_cutoff"].between(*TRAIN_DAYS)]
    pre_main = daily[daily["date"].between(*(pd.Timestamp(d) for d in core.PLACEBO))]
    pre = daily_rows(keys, placebo_prob)
    pre = pre[pre["date"].between(*(pd.Timestamp(d) for d in core.PLACEBO))]
    summary, curves = [], []
    for g in groups:
        if not keys["bed"].isin(GROUPS[g]).any() or stress.empty:
            continue
        curve = _curve(stress, g, n_boot)
        (e, elo, ehi), npos, nneg = _per_plant_auc(stress, g, n_boot, EARLY_DAYS)
        (f, _, _), _, _ = _per_plant_auc(stress, g, 0, FULL_DAYS)
        (p, plo, phi), _, _ = _per_plant_auc(pre, g, n_boot) if len(pre) else ((np.nan,) * 3, 0, 0)
        (pm, _, _), _, _ = _per_plant_auc(pre_main, g, 0) if len(pre_main) else ((np.nan,) * 3, 0, 0)
        summary.append(
            dict(
                group=g, early_auc=e, early_ci_low=elo, early_ci_high=ehi, full_auc=f,
                onset_day=onset_day(curve) if len(curve) else np.nan,
                placebo_auc=p, placebo_ci_low=plo, placebo_ci_high=phi,
                bed_effect=bool(np.isfinite(plo) and (plo > 0.5 or phi < 0.5) and max(p, 1 - p) >= 0.7),
                placebo_auc_main=pm, n_deficit=npos, n_control=nneg,
            )
        )  # fmt: skip
        curves.append(curve.assign(group=g))
    return summary, curves


def within_day_rho(keys: pd.DataFrame, pred: np.ndarray, truth: np.ndarray, min_plants: int = 5) -> pd.DataFrame:
    """Spearman between prediction and truth across plants, separately for each day (Part C)."""
    d = keys[["pid", "date", "days_since_cutoff", "hour"]].assign(pred=pred, truth=truth)
    d = d[d["hour"].isin([12, 15])].dropna(subset=["pred", "truth"])
    d = d.groupby(["pid", "date", "days_since_cutoff"], as_index=False)[["pred", "truth"]].mean()
    rows = []
    for (date, day), g in d.groupby(["date", "days_since_cutoff"]):
        if len(g) >= min_plants and g["truth"].nunique() > 1:
            rows.append(dict(date=date, day=int(day), rho=g["pred"].corr(g["truth"], method="spearman"), n=len(g)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# stage 1 cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(
    rs: RunSet,
    cfg: Config,
    device: str,
    fit=fit_head,
    ckpt_dir: Path | None = None,
    fold_state: Callable[[], dict] | None = None,
) -> dict:
    """Out-of-fold predictions for every setup, from one model per fold (+ a placebo model).

    With ckpt_dir, each fold's model and placebo model are saved with the plants they were
    trained on; `fold_state()` adds stage-specific weights (the LoRA adapters) to fold<k>.pt.
    """
    keys = rs.keys
    train_rows = keys["train"].fillna(False).to_numpy()
    pre_rows = keys["date"].between(*(pd.Timestamp(d) for d in core.PLACEBO)).to_numpy()
    keys["y_bed"] = (keys["bed"] != "W1").astype(int)
    all_setups = setups(rs) | hsi_setups(rs)
    prob = {name: np.full(len(keys), np.nan) for name in all_setups}
    pprob = {name: np.full(len(keys), np.nan) for name in all_setups}
    stage, turg = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
    for k, test in enumerate(plant_folds(keys, cfg.folds, cfg.seed)):
        train = np.setdiff1d(np.arange(len(keys)), test)
        tr, va = inner_split(keys, train[train_rows[train]], cfg.val_frac, cfg.seed + k)
        model = fit(rs, tr, va, cfg, device)
        ptr, pva = inner_split(keys, train[pre_rows[train]], cfg.val_frac, cfg.seed + k)
        pmodel = fit_head(rs, ptr, pva, cfg, device, target="y_bed", aux=False)
        for name, setup in all_setups.items():
            prob[name][test] = predict(model, rs, test, setup, device)[0]
            pprob[name][test] = predict(pmodel, rs, test, setup, device)[0]
        _, stage[test], turg[test] = predict(model, rs, test, full_setup(rs), device)
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            plants = dict(
                sensors=list(rs.sensors), dims=dict(rs.dims), config=asdict(cfg),
                train_pids=sorted(keys["pid"].iloc[train].unique()), test_pids=sorted(keys["pid"].iloc[test].unique()),
            )  # fmt: skip
            extra = fold_state() if fold_state is not None else {}
            torch.save({"model": model.state_dict(), **plants, **extra}, ckpt_dir / f"fold{k}.pt")
            if pmodel is not None:
                torch.save(
                    {"model": pmodel.state_dict(), **plants, "target": "y_bed"}, ckpt_dir / f"placebo_fold{k}.pt"
                )
        print(f"  fold {k + 1}/{cfg.folds}: {keys.iloc[test]['pid'].nunique()} held-out plants")
    return dict(prob=prob, placebo=pprob, stage=stage, turgor=turg, setups=all_setups)


def results(rs: RunSet, cv: dict, model_name: str, n_boot: int, groups=("all", "W2", "W3")) -> dict[str, pd.DataFrame]:
    keys = rs.keys
    bench, curves, hsi = [], [], []
    for name, setup in cv["setups"].items():
        summary, cs = score(keys, cv["prob"][name], cv["placebo"][name], groups, n_boot)
        is_hsi = name.startswith("hsi@")
        parts = [] if is_hsi else name.split("+")
        for s in summary:
            row = dict(sensors=name, model=model_name, **s)
            if is_hsi:
                v = setup["hsi"]
                row |= dict(
                    variant=v, analysis=_hsi_analysis(v), wl_lo_nm=float(v[4:]) if v.startswith("mask") else np.nan
                )
                hsi.append(row)
            else:
                row |= dict(
                    setup=" + ".join(SETUP_LABEL[p] for p in parts), n_sensors=len(parts),
                    n_features=int(sum(rs.dims[p.split("_")[0]] for p in parts)), cost_rank=cost_rank(parts),
                    is_size_baseline=False, size_column=None,
                )  # fmt: skip
                bench.append(row)
        for c in cs:
            curves.append(
                c.assign(signal=f"model:{name}:{model_name}", kind="model", column=f"model:{name}:{model_name}")
            )
    hsi = pd.DataFrame(hsi)
    if not hsi.empty:
        ref = hsi[hsi["variant"] == "full"].set_index("group")["early_auc"]
        hsi["early_drop"] = hsi["group"].map(ref) - hsi["early_auc"]

    sev = []
    for target, pred, truth in (
        ("stage", cv["stage"], keys["stage"].where(keys["stage"] >= 0)),
        ("turgor", cv["turgor"], keys["turgor"]),
    ):
        r = within_day_rho(keys, pred, truth.to_numpy(float))
        if not r.empty:
            sev.append(r.assign(target=target, model=model_name))
    full = "+".join(s for s in SENSOR_ORDER if s in rs.sensors)
    scores = keys[core.KEYS].assign(
        detect_fusion_prob=cv["prob"][full], detect_fusion_stage=cv["stage"], detect_fusion_turgor=cv["turgor"]
    )
    return dict(
        fusion_benchmark=pd.DataFrame(bench),
        fusion_auc_by_day=pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        fusion_hsi=hsi,
        fusion_severity=pd.concat(sev, ignore_index=True) if sev else pd.DataFrame(),
        fusion_scores_runs=scores,
    )


def save_oof(rs: RunSet, cv: dict, run_dir: Path) -> None:
    """Out-of-fold P(deficit) and placebo P(bed) of every setup, per run, beside the checkpoints
    (runs/fusion/<model>/oof_{prob,placebo}.parquet, not in git), so other plant subsets (e.g.
    plants without turgor clips) can be scored later without retraining."""
    run_dir.mkdir(parents=True, exist_ok=True)
    keys = rs.keys[[*core.KEYS, "pid", "days_since_cutoff"]].reset_index(drop=True)
    for what in ("prob", "placebo"):
        cols = pd.DataFrame({name: v for name, v in cv[what].items()})
        pd.concat([keys, cols], axis=1).to_parquet(run_dir / f"oof_{what}.parquet", index=False)


def _hsi_analysis(v: str) -> str:
    if v == "full":
        return "all 300 bands, 224 px"
    if v == "rgb3":
        return "3 bands nearest RGB"
    if v.startswith("s"):
        return f"spatial detail of {v[1:]} px"
    return "20 nm window masked"


def write(tables: dict[str, pd.DataFrame], model_name: str, out_dir: Path | None = None) -> Path:
    """Upsert this model's rows (other stages' rows are kept) and the run-level scores."""
    out = Path(out_dir) if out_dir else core.DERIVED / OUT
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        if df is None or df.empty:
            continue
        if name == "fusion_scores_runs":
            if model_name != "fusion":  # run-level scores come from stage 1 (all sensors, frozen)
                continue
            out_df = df.copy()
            out_df["date"] = pd.to_datetime(out_df["date"]).dt.date
        else:
            old = pd.read_parquet(path) if path.exists() else None
            out_df = df if old is None else pd.concat([old[old["model"] != model_name], df], ignore_index=True)
        out_df.to_parquet(path, index=False)
    SCORES_CATALOGUE.to_csv(out / "catalogue.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# stages 2-3: LoRA on one encoder, end to end
# --------------------------------------------------------------------------- #
class _LiveImages(torch.utils.data.Dataset):
    def __init__(self, paths: list[str | None], base_paths: list[list[str]], sensor: str, aug: bool, seed: int):
        self.paths, self.base_paths, self.sensor, self.aug, self.seed = paths, base_paths, sensor, aug, seed

    def __len__(self):
        return len(self.paths)

    def _load(self, path, rng):
        if self.sensor == "thermal":  # one centre frame so runs stack
            s = inputs.load_thermal(path, frames=(2,))
        else:
            s = inputs.load(self.sensor, path)
        x = torch.from_numpy(s.frames)
        return (inputs.augment(x, rng) if self.aug else x), s.wavelengths

    def __getitem__(self, i):
        # One generator per (epoch seed, item): a generator made in __init__ is copied into every
        # DataLoader worker, so all workers drew the same flips, crops and baseline runs.
        rng = np.random.default_rng([self.seed, i])
        try:
            x, wl = self._load(self.paths[i], rng)
            xb, _ = self._load(self.base_paths[i][int(rng.integers(len(self.base_paths[i])))], rng)
            return i, x, xb, wl
        except Exception:  # noqa: BLE001  (an unreadable run is dropped from the batch)
            return i, None, None, None


def _collate_live(items):
    items = [it for it in items if it[1] is not None]
    if not items:
        return None
    return (
        np.array([i for i, *_ in items]),
        torch.stack([x for _, x, _, _ in items]),
        torch.stack([b for _, _, b, _ in items]),
        items[0][3],
    )


class LoRAStage:
    """Fusion head + one encoder with LoRA; the encoder sees raw images, the rest cached embeddings."""

    def __init__(self, rs: RunSet, sensor: str, data_dir: Path, cfg: Config):
        from uiwue.fusion.encoders import add_lora, build_encoder

        self.rs, self.sensor, self.cfg = rs, sensor, cfg
        runs = core.normalize_keys(inputs.list_runs(data_dir, sensor))
        path = rs.keys[core.KEYS].merge(runs[[*core.KEYS, "path"]], on=core.KEYS, how="left")["path"]
        self.paths = path.where(path.notna(), None).tolist()
        lo, hi = (pd.Timestamp(d) for d in core.BASELINE)
        k = rs.keys.assign(path=path)
        pool = k[k["date"].between(lo, hi) & k["path"].notna()]
        same = pool.groupby(["pid", "hour"])["path"].apply(list).to_dict()
        anyh = pool.groupby("pid")["path"].apply(list).to_dict()
        self.base_paths = [same.get((p, h)) or anyh.get(p) or [] for p, h in zip(k["pid"], k["hour"], strict=True)]
        self.ok = np.array([p is not None and len(b) > 0 for p, b in zip(self.paths, self.base_paths, strict=True)])
        self.make_encoder = lambda: build_encoder(sensor)
        self.add_lora = add_lora

    def fit(self, rs: RunSet, train_idx, val_idx, cfg: Config, device: str):
        from uiwue.fusion.encoders import lora_parameters

        model = fit_head(rs, train_idx, val_idx, cfg, device)  # warm start from the frozen-encoder head
        enc = self.make_encoder().to(device)
        self.add_lora(enc, cfg.lora_blocks, cfg.lora_rank)
        params = [
            {"params": model.parameters(), "lr": cfg.lr / 10},
            {"params": lora_parameters(enc), "lr": cfg.lora_lr},
        ]
        opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
        train_idx = train_idx[self.ok[train_idx]]
        best, best_loss, wait = None, np.inf, 0
        for epoch in range(cfg.lora_epochs):
            model.train(), enc.train()
            for b in self._loader(train_idx, aug=True, shuffle=True, seed=cfg.seed + epoch):
                if b is None:
                    continue
                loss = self._step(model, enc, b, device, train=True)
                opt.zero_grad()
                loss.backward()
                opt.step()
            val = self._eval_loss(model, enc, val_idx[self.ok[val_idx]], device)
            if val < best_loss - 1e-4:
                best, best_loss, wait = (copy.deepcopy(model.state_dict()), copy.deepcopy(enc.state_dict())), val, 0
            else:
                wait += 1
                if wait >= max(2, cfg.patience // 3):
                    break
        if best is not None:
            model.load_state_dict(best[0])
            enc.load_state_dict(best[1])
        self.encoder = enc.eval()
        self._refresh_embeddings(device)  # every run, test plants included, through this fold's encoder
        return model.eval()

    def _loader(self, idx, aug, shuffle, seed):
        ds = _LiveImages([self.paths[i] for i in idx], [self.base_paths[i] for i in idx], self.sensor, aug, seed)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.cfg.lora_batch, shuffle=shuffle, num_workers=self.cfg.workers, collate_fn=_collate_live
        )
        for b in dl:
            yield None if b is None else (idx[b[0]], *b[1:])

    def _step(self, model, enc, b, device, train):
        rows, x, xb, wl = b
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            e, eb = enc.encode_frames(x.to(device), wl).float(), enc.encode_frames(xb.to(device), wl).float()
        t = batch(self.rs, rows, full_setup(self.rs), device)
        t["now"][self.sensor], t["base"][self.sensor] = e, eb
        t["present"][:, self.rs.sensors.index(self.sensor)] = True
        return fusion_loss(model(t["now"], t["base"], t["present"]), t["y"], t["stage"], t["turgor"])

    @torch.no_grad()
    def _eval_loss(self, model, enc, idx, device):
        model.eval(), enc.eval()
        losses = [float(self._step(model, enc, b, device, False)) for b in self._loader(idx, False, False, 0) if b]
        return float(np.mean(losses)) if losses else np.inf

    @torch.no_grad()
    def _refresh_embeddings(self, device):
        """Re-embed every run (and baselines) with the tuned encoder, into the RunSet's full table."""
        rows = np.flatnonzero(self.ok)
        emb = self.rs.emb[(self.sensor, "full")].copy()
        for b in self._loader(rows, aug=False, shuffle=False, seed=0):
            if b is None:
                continue
            r, x, _, wl = b
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                emb[r] = self.encoder.encode_frames(x.to(device), wl).float().cpu().numpy()
        has = self.rs.has[(self.sensor, "full")] | self.ok
        base, ok = baseline_of(self.rs.keys, emb, has)
        self.rs.emb[(self.sensor, "full")], self.rs.base[(self.sensor, "full")] = emb, base
        self.rs.has[(self.sensor, "full")] = has & ok


def cross_validate_lora(rs: RunSet, stage: LoRAStage, cfg: Config, device: str, ckpt_dir: Path | None) -> dict:
    """As cross_validate, but each fold's encoder is tuned with LoRA and its test plants are
    re-embedded by that fold's encoder before scoring."""
    frozen = {k: v.copy() for k, v in rs.emb.items()}, {k: v.copy() for k, v in rs.base.items()}, dict(rs.has)

    def fit(rs_, tr, va, cfg_, dev):
        rs_.emb, rs_.base, rs_.has = ({k: v.copy() for k, v in d.items()} for d in frozen)
        return stage.fit(rs_, tr, va, cfg_, dev)

    def fold_state():
        from uiwue.fusion.encoders import lora_state_dict

        return dict(lora=lora_state_dict(stage.encoder), lora_sensor=stage.sensor)

    return cross_validate(rs, cfg, device, fit=fit, ckpt_dir=ckpt_dir, fold_state=fold_state)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["frozen", "lora"], default="frozen")
    ap.add_argument("--lora-sensor", choices=["hsi", "thermal", "rgb", "depth"], default="hsi")
    ap.add_argument("--data-dir", type=Path, default=core.DATA)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lora-epochs", type=int, default=Config.lora_epochs)
    ap.add_argument("--lora-batch", type=int, default=Config.lora_batch)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="where fold checkpoints go (runs/fusion)")
    a = ap.parse_args(argv)
    cfg = Config(
        folds=a.folds, epochs=a.epochs, n_boot=a.n_boot, workers=a.workers,
        lora_epochs=a.lora_epochs, lora_batch=a.lora_batch,
    )  # fmt: skip

    t0 = time.time()
    rs = assemble(a.cache)
    print(f"{len(rs.keys)} runs, {rs.keys['pid'].nunique()} plants, sensors {rs.sensors}")
    if a.stage == "frozen":
        name = "fusion"
        cv = cross_validate(rs, cfg, a.device, ckpt_dir=a.runs_dir / name)
    else:
        name = f"fusion+lora_{a.lora_sensor}"
        cv = cross_validate_lora(rs, LoRAStage(rs, a.lora_sensor, a.data_dir, cfg), cfg, a.device, a.runs_dir / name)
    out = write(results(rs, cv, name, cfg.n_boot), name, a.out)
    save_oof(rs, cv, a.runs_dir / name)
    info = dict(
        model=name, finished=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        seconds=round(time.time() - t0, 1), sensors=rs.sensors, dims=rs.dims, runs=len(rs.keys),
        plants=int(rs.keys["pid"].nunique()), config=asdict(cfg), device=a.device,
    )  # fmt: skip
    (out / f"{name.replace('+', '_')}_run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"{name}: done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
