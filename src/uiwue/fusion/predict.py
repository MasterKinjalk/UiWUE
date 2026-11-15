"""External test: score plants the fusion model never saw with its 5 fold models.

    python -m uiwue.fusion.embed --plants <ids> --hsi-windows --cache cache/fusion_external
    python -m uiwue.fusion.predict --cache cache/fusion_external [--model fusion]
                                   [--size-traits derived/threed/traits_runs.parquet]

Each fold model (runs/fusion/<model>/fold<k>.pt) scores every run of the new plants
and the 5 probabilities are averaged; the stage and turgor heads likewise. The
placebo models (placebo_fold<k>.pt) are averaged the same way. Nothing is fitted
here: the standardisation lives inside the fold models, and each plant's
baseline is the mean embedding of its own 30 Aug - 2 Sep runs at the same hour
(train.baseline_of). Plants that any fold trained or was tested on are dropped.

For a LoRA stage, each fold's tuned encoder (pretrained weights + the saved
adapters) re-embeds the new plants' raw images of its sensor first, as the
cross-validation did for its held-out plants (needs --data-dir).

Scoring is the cross-validation's (train.score / train.results): every sensor
setup, per-day AUC with a plant bootstrap, early (days 1-4) and full AUC, onset
day, and both placebo checks. Each table is written for three sets of plants:

- `all`: every external plant scored.
- `no_clip`: plants without a turgor clip. Clip hardware is on 18 deficit and 3
  control plants from about 2 Sep, after the placebo window, so the placebo
  cannot see it; this AUC cannot use it.
- `with_size`: plants that also have the size trait, beside the size-only
  baseline on the same plants (Master Plan 6.7.3).

Writes derived/detect/fusion_external_{benchmark,auc_by_day,hsi,scores_runs}.parquet
(rows of other models are kept) and fusion_external_<model>_run_info.json.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uiwue import core
from uiwue.detect.benchmark import FULL_DAYS, TRAIN_DAYS
from uiwue.detect.data import size_baseline_column
from uiwue.detect.run import SCORES_CATALOGUE
from uiwue.fusion import train
from uiwue.fusion.train import Config, RunSet

OUT = "detect"
EXTERNAL_CACHE = core.REPO / "cache" / "fusion_external"
GROUPS = ("all", "W2", "W3")
SCORE_COLUMNS = {
    "detect_fusion_prob": "detect_fusion_ext_prob",
    "detect_fusion_stage": "detect_fusion_ext_stage",
    "detect_fusion_turgor": "detect_fusion_ext_turgor",
}


# --------------------------------------------------------------------------- #
# fold models
# --------------------------------------------------------------------------- #
def load_folds(run_dir: Path) -> list[dict]:
    """fold<k>.pt (+ placebo_fold<k>.pt as "placebo") in fold order."""
    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob("fold*.pt"), key=lambda p: int(p.stem[4:]))
    if not paths:
        raise FileNotFoundError(f"no fold checkpoints in {run_dir}; run python -m uiwue.fusion.train first")
    folds = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=True)
        placebo = p.with_name(f"placebo_{p.name}")
        ck["placebo"] = torch.load(placebo, map_location="cpu", weights_only=True) if placebo.exists() else None
        folds.append(ck)
    return folds


def config_of(ck: dict) -> Config:
    names = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in ck["config"].items() if k in names})


def build(ck: dict | None, device: str):
    """The saved fusion head, ready for inference (None if the fold had no such model)."""
    if ck is None:
        return None
    model = train.build_model(ck["dims"], config_of(ck))
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval()


def seen_plants(folds: list[dict]) -> set[str]:
    return {p for ck in folds for p in (*ck.get("train_pids", []), *ck.get("test_pids", []))}


# --------------------------------------------------------------------------- #
# the external RunSet
# --------------------------------------------------------------------------- #
def subset(rs: RunSet, rows: np.ndarray) -> RunSet:
    """The RunSet restricted to `rows` (positions); baselines were computed before, per plant."""
    return RunSet(
        rs.keys.iloc[rows].reset_index(drop=True),
        list(rs.sensors),
        dict(rs.dims),
        {k: v[rows] for k, v in rs.emb.items()},
        {k: v[rows] for k, v in rs.base.items()},
        {k: v[rows] for k, v in rs.has.items()},
    )


def conform(rs: RunSet, sensors: list[str], dims: dict[str, int]) -> RunSet:
    """Sensors in the model's order; a sensor with no embeddings here is added as absent."""
    for s in sensors:
        if s in rs.dims and rs.dims[s] != dims[s]:
            raise ValueError(f"{s}: embeddings are {rs.dims[s]}-d, the model expects {dims[s]}")
        if s not in rs.sensors:
            n = len(rs.keys)
            rs.emb[(s, "full")] = np.zeros((n, dims[s]), np.float32)
            rs.base[(s, "full")] = np.zeros((n, dims[s]), np.float32)
            rs.has[(s, "full")] = np.zeros(n, bool)
    extra = [s for s in rs.sensors if s not in sensors]
    for k in [k for k in rs.emb if k[0] in extra]:
        del rs.emb[k], rs.base[k], rs.has[k]
    rs.sensors, rs.dims = list(sensors), {s: dims[s] for s in sensors}
    return rs


def clip_plants() -> set[str]:
    plants = core.read_table("truth", "plants")
    if plants is None or "truth_has_clip" not in plants:
        return set()
    p = plants[plants["truth_has_clip"].fillna(False).astype(bool)]
    return set(p["bed"] + "_" + p["plant"])


# --------------------------------------------------------------------------- #
# ensemble of the fold models
# --------------------------------------------------------------------------- #
def _tuned_encoder(ck: dict, device: str, make_encoder=None, pretrained=None):
    """The fold's tuned encoder: the pretrained encoder (a copy of `pretrained` if given, since
    building ImageBind reads its 4.8 GB checkpoint) with the fold's saved adapters."""
    from uiwue.fusion.encoders import add_lora, build_encoder, load_lora_state_dict

    cfg = config_of(ck)
    if pretrained is not None:
        enc = copy.deepcopy(pretrained).to(device)
    else:
        enc = (make_encoder or (lambda: build_encoder(ck["lora_sensor"])))().to(device)
    add_lora(enc, cfg.lora_blocks, cfg.lora_rank)  # after .to(device), as LoRAStage.fit
    load_lora_state_dict(enc, ck["lora"])
    return enc.eval()


def ensemble(
    rs: RunSet, folds: list[dict], device: str, data_dir: Path | None = None, make_encoder=None, workers: int = 4
) -> dict:
    """Mean over the fold models of P(deficit) per setup, of the placebo P(bed), stage and turgor.

    Returns the dict train.results() takes (prob, placebo, stage, turgor, setups), plus the
    per-fold full-setup probabilities ("fold_prob", folds x runs) for the spread between folds.
    """
    all_setups = train.setups(rs) | train.hsi_setups(rs)
    n, idx = len(rs.keys), np.arange(len(rs.keys))
    prob = {name: np.zeros((len(folds), n)) for name in all_setups}
    pprob = {name: np.zeros((len(folds), n)) for name in all_setups}
    stage, turg = np.zeros((len(folds), n)), np.zeros((len(folds), n))
    lora_sensor = folds[0].get("lora_sensor")
    if lora_sensor is not None:
        if data_dir is None:
            raise ValueError(f"a LoRA stage ({lora_sensor}) re-embeds raw images: pass data_dir")
        live = train.LoRAStage(rs, lora_sensor, data_dir, dataclasses.replace(config_of(folds[0]), workers=workers))
        frozen = ({k: v.copy() for k, v in rs.emb.items()}, {k: v.copy() for k, v in rs.base.items()}, dict(rs.has))
        print(f"  {lora_sensor}: {int(live.ok.sum())} of {n} runs re-embedded by each fold's tuned encoder")
        from uiwue.fusion.encoders import build_encoder

        pretrained = (make_encoder or (lambda: build_encoder(lora_sensor)))()
    for k, ck in enumerate(folds):
        if lora_sensor is not None:
            rs.emb, rs.base, rs.has = ({key: v.copy() for key, v in d.items()} for d in frozen)
            live.rs = rs
            live.encoder = _tuned_encoder(ck, device, pretrained=pretrained)
            live._refresh_embeddings(device)
            del live.encoder
        model, pmodel = build(ck, device), build(ck["placebo"], device)
        for name, setup in all_setups.items():
            prob[name][k] = train.predict(model, rs, idx, setup, device)[0]
            pprob[name][k] = train.predict(pmodel, rs, idx, setup, device)[0]
        _, stage[k], turg[k] = train.predict(model, rs, idx, train.full_setup(rs), device)
        print(f"  fold model {k + 1}/{len(folds)} applied")
    full = "+".join(s for s in train.SENSOR_ORDER if s in rs.sensors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN where a setup has no sensor present
        mean = lambda a: np.nanmean(a, 0)
        return dict(
            prob={k: mean(v) for k, v in prob.items()},
            placebo={k: mean(v) for k, v in pprob.items()},
            stage=mean(stage),
            turgor=mean(turg),
            setups=all_setups,
            fold_prob=prob[full],
        )


# --------------------------------------------------------------------------- #
# size-only baseline on the same plants
# --------------------------------------------------------------------------- #
def size_baseline(traits: pd.DataFrame | None, pids: set[str], groups=GROUPS, n_boot: int = 2000):
    """Rows in the benchmark layout for plant size alone (no model: one trait, oriented by the
    catalogue, has the AUC a one-feature logistic model would have), and its per-day curves."""
    from uiwue import stats

    if traits is None or traits.empty:
        return [], [], set()
    col = size_baseline_column(traits)
    if col is None:
        return [], [], set()
    t = core.normalize_keys(traits)
    t = t[(t["bed"] + "_" + t["plant"]).isin(pids) & t[col].notna()]
    have = set(t["bed"] + "_" + t["plant"])
    rows, curves = [], []
    for g in groups:
        if not t["bed"].isin(stats.GROUPS[g]).any() or not (t["bed"] == "W1").any():
            continue
        curve = stats.auc_by_day(t, col, group=g, days=TRAIN_DAYS, n_boot=n_boot)
        e, elo, ehi = stats.early_auc(t, col, group=g, n_boot=n_boot)
        f, _, _ = stats.early_auc(t, col, group=g, days=FULL_DAYS, n_boot=0)
        pb = stats.placebo(t, col, group=g, n_boot=n_boot)
        late = curve[curve["day"].between(*TRAIN_DAYS)]
        rows.append(
            dict(
                sensors="size", model="size_only", group=g, early_auc=e, early_ci_low=elo, early_ci_high=ehi,
                full_auc=f, onset_day=stats.onset_day(late) if len(late) else np.nan,
                placebo_auc=pb["auc"], placebo_ci_low=pb["ci_low"], placebo_ci_high=pb["ci_high"],
                bed_effect=pb["bed_effect"], placebo_auc_main=np.nan,
                n_deficit=int(late["n_deficit"].max()) if len(late) else 0,
                n_control=int(late["n_control"].max()) if len(late) else 0,
                setup="Plant size only", n_sensors=0, n_features=1, cost_rank=0, is_size_baseline=True,
                size_column=col,
            )
        )  # fmt: skip
        name = f"model:size:size_only:{col}"
        curves.append(
            curve.drop(columns=["column", "direction"]).assign(
                group=g, signal=name, kind="model", column=name, model="size_only"
            )
        )
    return rows, curves, have


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def external_tables(
    rs: RunSet,
    cv: dict,
    model_name: str,
    n_boot: int,
    traits: pd.DataFrame | None = None,
    groups=GROUPS,
    subsets=("all", "no_clip", "with_size"),
) -> dict[str, pd.DataFrame]:
    """train.results() on the external plants, for `all`, `no_clip` and `with_size` plants."""
    pid = rs.keys["pid"].to_numpy()
    clips = clip_plants()
    size_rows, size_curves, with_size = size_baseline(traits, set(pid), groups, n_boot)
    sets = {"all": np.ones(len(pid), bool), "no_clip": ~np.isin(pid, list(clips))}
    if with_size:
        sets["with_size"] = np.isin(pid, list(with_size))
    sets = {k: v for k, v in sets.items() if k in subsets}
    if "with_size" not in subsets:
        size_rows = []
    scores = None
    bench, curves, hsi = [], [], []
    for plants, mask in sets.items():
        if not mask.any():
            continue
        rows = np.flatnonzero(mask)
        sub = subset(rs, rows)
        cv_sub = dict(
            prob={k: v[rows] for k, v in cv["prob"].items()},
            placebo={k: v[rows] for k, v in cv["placebo"].items()},
            stage=cv["stage"][rows], turgor=cv["turgor"][rows], setups=cv["setups"],
        )  # fmt: skip
        res = train.results(sub, cv_sub, model_name, n_boot, groups=groups)
        tag = dict(plants=plants, n_plants=int(sub.keys["pid"].nunique()))
        bench.append(res["fusion_benchmark"].assign(**tag))
        curves.append(res["fusion_auc_by_day"].assign(**tag))
        hsi.append(res["fusion_hsi"].assign(**tag))
        if plants == "all":
            scores = res["fusion_scores_runs"].rename(columns=SCORE_COLUMNS)
    if size_rows:
        tag = dict(plants="with_size", n_plants=len(with_size))
        bench.append(pd.DataFrame(size_rows).assign(**tag))
        curves.append(pd.concat(size_curves, ignore_index=True).assign(**tag))

    def nonempty(parts):
        parts = [p for p in parts if p is not None and not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else None

    return dict(
        fusion_external_benchmark=nonempty(bench),
        fusion_external_auc_by_day=nonempty(curves),
        fusion_external_hsi=nonempty(hsi),
        fusion_external_scores_runs=scores,
    )


def oof_tables(
    run_dir: Path, model_name: str, n_boot: int = 2000, traits: pd.DataFrame | None = None, groups=GROUPS
) -> dict[str, pd.DataFrame]:
    """The cross-validation's out-of-fold scores (train.save_oof), scored for plant subsets of
    the 39 training plants: `no_clip` (the clip shortcut is out of reach) and `with_size`
    (beside the size-only baseline on the same plants). Written as fusion_cv_subsets_*."""
    from uiwue.fusion.labels import run_labels

    run_dir = Path(run_dir)
    prob = pd.read_parquet(run_dir / "oof_prob.parquet")
    plac = pd.read_parquet(run_dir / "oof_placebo.parquet")
    ck = torch.load(run_dir / "fold0.pt", map_location="cpu", weights_only=True)
    keys = core.normalize_keys(prob[core.KEYS])
    keys = keys.merge(run_labels(keys), on=core.KEYS, how="left", validate="one_to_one")
    keys["pid"] = keys["bed"] + "_" + keys["plant"]
    names = [c for c in prob.columns if c not in (*core.KEYS, "pid", "days_since_cutoff")]
    setups = {n: ({"hsi": n[4:]} if n.startswith("hsi@") else {p: "full" for p in n.split("+")}) for n in names}
    rs = RunSet(keys, list(ck["sensors"]), dict(ck["dims"]), {}, {}, {})
    nan = np.full(len(keys), np.nan)
    cv = dict(
        prob={n: prob[n].to_numpy(float) for n in names}, placebo={n: plac[n].to_numpy(float) for n in names},
        stage=nan, turgor=nan, setups=setups,
    )  # fmt: skip
    t = external_tables(rs, cv, model_name, n_boot, traits, groups, subsets=("no_clip", "with_size"))
    return {
        "fusion_cv_subsets_benchmark": t["fusion_external_benchmark"],
        "fusion_cv_subsets_auc_by_day": t["fusion_external_auc_by_day"],
    }


def write(tables: dict[str, pd.DataFrame], model_name: str, out_dir: Path | None = None) -> Path:
    """As train.write: this model's rows replace its old ones, other models' rows are kept; the
    run-level scores come from stage 1 (model "fusion")."""
    out = Path(out_dir) if out_dir else core.DERIVED / OUT
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        if df is None or df.empty:
            continue
        if name == "fusion_external_scores_runs":
            if model_name != "fusion" or "detect_fusion_ext_prob" not in df:
                continue
            out_df = df.copy()
            out_df["date"] = pd.to_datetime(out_df["date"]).dt.date
        else:
            old = pd.read_parquet(path) if path.exists() else None
            if old is not None:  # the size-only rows are replaced whenever a run brings new ones
                replaced = {model_name} | ({"size_only"} if (df["model"] == "size_only").any() else set())
                old = old[~train.model_of(old).isin(replaced)]
            out_df = df if old is None else pd.concat([old, df], ignore_index=True)
        out_df.to_parquet(path, index=False)
    SCORES_CATALOGUE.to_csv(out / "catalogue.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(
    caches,
    run_dir: Path,
    model_name: str,
    device: str,
    n_boot: int = 2000,
    data_dir: Path | None = None,
    traits: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    make_encoder=None,
    workers: int = 4,
    features: Path | None = None,
    only=None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    folds = load_folds(run_dir)
    if folds[0].get("design") == "v2":  # fusion v2 (uiwue.fusion.v2): its own inputs and references
        from uiwue.fusion import v2

        tables, info, _ = v2.run_new(caches, folds, model_name, device, n_boot, traits, labels, features, only)
        return tables, info
    rs = train.assemble(caches, labels=labels)
    seen = seen_plants(folds)
    new = ~rs.keys["pid"].isin(seen).to_numpy()
    dropped = sorted(set(rs.keys["pid"][~new]))
    if dropped:
        print(f"  {len(dropped)} plants were in the training folds and are not scored: {' '.join(dropped)}")
    if not new.any():
        raise ValueError("every plant in the cache was used in training; nothing to test")
    rs = conform(subset(rs, np.flatnonzero(new)), folds[0]["sensors"], folds[0]["dims"])
    pids = sorted(rs.keys["pid"].unique())
    print(f"{len(rs.keys)} runs, {len(pids)} external plants, {len(folds)} fold models ({model_name})")
    cv = ensemble(rs, folds, device, data_dir=data_dir, make_encoder=make_encoder, workers=workers)
    tables = external_tables(rs, cv, model_name, n_boot, traits)
    present = {s: sorted(rs.keys.loc[rs.has[(s, "full")], "pid"].unique()) for s in rs.sensors if (s, "full") in rs.has}
    info = dict(
        model=model_name, plants=pids, n_plants=len(pids), runs=len(rs.keys), excluded_training_plants=dropped,
        plants_per_bed=rs.keys.drop_duplicates("pid")["bed"].value_counts().sort_index().to_dict(),
        plants_with_sensor={s: len(v) for s, v in present.items()},
        plants_missing_sensor={s: sorted(set(pids) - set(v)) for s, v in present.items()},
        clip_plants=sorted(set(pids) & clip_plants()),
        fold_spread_sd=float(np.nanmean(np.nanstd(cv["fold_prob"], 0))) if len(folds) > 1 else 0.0,
    )  # fmt: skip
    return tables, info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, nargs="+", default=[EXTERNAL_CACHE], help="one or more embedding caches")
    ap.add_argument("--model", default="fusion", help="fusion (stage 1) or fusion+lora_<sensor>")
    ap.add_argument("--runs-dir", type=Path, default=train.RUNS_DIR)
    ap.add_argument("--data-dir", type=Path, default=core.DATA, help="raw images, for LoRA stages only")
    ap.add_argument("--size-traits", type=Path, default=core.DERIVED / "threed" / "traits_runs.parquet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--oof", action="store_true", help="instead: score the CV's out-of-fold scores for no-clip and size plants"
    )
    ap.add_argument(
        "--features",
        type=Path,
        default=core.REPO / "cache" / "fusion_v2" / "thermal_features.parquet",
        help="fusion v2 only: hand-made thermal features (uiwue.fusion.features)",
    )
    ap.add_argument(
        "--lockbox", action="store_true", help="fusion v2: score only the lock-box plants, into fusion_lockbox_* tables"
    )
    a = ap.parse_args(argv)

    t0 = time.time()
    traits = pd.read_parquet(a.size_traits) if a.size_traits.exists() else None
    if a.oof:
        tables = oof_tables(a.runs_dir / a.model, a.model, a.n_boot, traits)
        out = write(tables, a.model, a.out)
        b = tables["fusion_cv_subsets_benchmark"]
        cols = ["plants", "n_plants", "sensors", "model", "early_auc", "early_ci_low", "onset_day", "placebo_auc"]
        print(b[b["group"] == "all"][cols].to_string(index=False))
        print(f"{a.model}: out-of-fold subsets scored in {time.time() - t0:.0f} s -> {out}")
        return
    if a.lockbox:
        from uiwue.fusion.lockbox import LOCKBOX
    tables, info = run(
        a.cache, a.runs_dir / a.model, a.model, a.device, a.n_boot, a.data_dir, traits, workers=a.workers,
        features=a.features, only=LOCKBOX if a.lockbox else None,
    )  # fmt: skip
    if a.lockbox:  # same layout as the external test, own table names
        tables = {k.replace("fusion_external_", "fusion_lockbox_"): v for k, v in tables.items()}
        tables.pop("fusion_lockbox_scores_runs", None)
    out = write(tables, a.model, a.out)
    info |= dict(
        finished=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1),
        caches=[str(c) for c in a.cache], size_traits=str(a.size_traits) if traits is not None else None,
    )  # fmt: skip
    prefix = "fusion_lockbox" if a.lockbox else "fusion_external"
    (out / f"{prefix}_{a.model.replace('+', '_')}_run_info.json").write_text(json.dumps(info, indent=2, default=str))
    b = tables[f"{prefix}_benchmark"]
    if b is not None:
        show = b[(b["group"] == "all")][["plants", "sensors", "early_auc", "early_ci_low", "onset_day", "placebo_auc"]]
        print(show.to_string(index=False))
    print(f"{a.model}: {info['n_plants']} external plants scored in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
