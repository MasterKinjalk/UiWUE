"""Run every detection analysis on whatever tables exist in derived/.

    python -m uiwue.detect.run [--n-boot 2000] [--skip benchmark recovery ...]

Writes derived/detect/:
  auc_by_day.parquet   per-day AUC curves (traits and models)
  onsets.parquet       onset day, robust onset, early AUC, placebo, lead times per signal and group
  onsets_by_slot.parquet  both onsets per signal, group and time-of-day slot
  placebo.parquet      the placebo rows alone (Methods & limits), with the pre-cutoff check
  benchmark.parquet    every sensor combination x model, with size-only baseline, per plant set
  scores_runs.parquet  detect_stress_prob per run (+ catalogue.csv); load_traits() attaches it
  truth_link.parquet   trait vs clip turgor
  recovery.parquet     recovery ratio per deficit plant (plant_id) and trait
  run_info.json        when, on what, with which settings
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from uiwue.detect.benchmark import run_benchmark, stress_scores
from uiwue.detect.data import load_merged, trait_columns
from uiwue.detect.onsets import run_onsets, run_slot_onsets
from uiwue.detect.truth import recovery, truth_link
from uiwue.stats._design import clear_caches, derived_dir

STEPS = ("onsets", "slots", "benchmark", "scores", "truth", "recovery")

SCORES_CATALOGUE = pd.DataFrame(
    [
        (
            "detect_stress_prob",
            "Stress probability",
            "0-1",
            1,
            "Out-of-fold P(deficit) from a leave-one-plant-out logistic model on all sensors' baseline-delta traits.",
        ),
        # Written by the fusion model (python -m uiwue.fusion.train) to fusion_scores_runs.
        (
            "detect_fusion_prob",
            "Stress probability (fusion model)",
            "0-1",
            1,
            (
                "Out-of-fold P(deficit) from the fusion model on DOFA / ImageBind embeddings of all sensors "
                "present; 5 folds grouped by plant."
            ),
        ),
        (
            "detect_fusion_stage",
            "Severity stage (fusion model)",
            "stage",
            1,
            "Expected stage 0-3 from the ordinal head (S0 healthy to S3 severe), out of fold.",
        ),
        (
            "detect_fusion_turgor",
            "Turgor deviation (fusion model)",
            "raw units",
            1,
            "Predicted leaf-clip shift from the turgor head, out of fold.",
        ),
        # Written by python -m uiwue.fusion.predict to fusion_external_scores_runs: plants that the
        # fusion model never saw, scored by the 5 fold models. Own names, so that load_traits can
        # join both scores tables without clashing columns.
        (
            "detect_fusion_ext_prob",
            "Stress probability (fusion model, external plants)",
            "0-1",
            1,
            "Mean P(deficit) of the 5 fold models on plants outside the training set (external test).",
        ),
        (
            "detect_fusion_ext_stage",
            "Severity stage (fusion model, external plants)",
            "stage",
            1,
            "Expected stage 0-3 from the ordinal head, mean of the 5 fold models, external plants.",
        ),
        (
            "detect_fusion_ext_turgor",
            "Turgor deviation (fusion model, external plants)",
            "raw units",
            1,
            "Predicted leaf-clip shift from the turgor head, mean of the 5 fold models, external plants.",
        ),
        # Written by python -m uiwue.fusion.v2 train to fusion_v2_scores_runs: fusion v2 (docs/fusion.md
        # section 8), out of fold on the development plants (the lock-box plants are not in it).
        (
            "detect_fusion_v2_prob",
            "Stress probability (fusion v2)",
            "0-1",
            1,
            (
                "Out-of-fold P(deficit) from fusion v2: each sensor's change from the plant's baseline and from "
                "the control plants in the same run, plus the hand-made thermal features; 5 folds grouped by plant."
            ),
        ),
        (
            "detect_fusion_v2_stage",
            "Severity stage (fusion v2)",
            "stage",
            1,
            "Expected stage 0-3 from fusion v2's ordinal head (S0 healthy to S3 severe), out of fold.",
        ),
        (
            "detect_fusion_v2_turgor",
            "Turgor deviation (fusion v2)",
            "raw units",
            1,
            "Predicted leaf-clip shift from fusion v2's turgor head, out of fold.",
        ),
    ],
    columns=["column", "label", "unit", "direction", "description"],
)


def _save(df: pd.DataFrame, out: Path, name: str) -> None:
    if df is None or df.empty:
        print(f"  {name}: nothing to write")
        return
    df.to_parquet(out / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df)} rows")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    out = a.out or derived_dir() / "detect"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clear_caches()
    df = load_merged()
    cols = trait_columns(df)
    if not cols:
        raise SystemExit(f"no trait tables found under {derived_dir()}")
    print(f"{len(df)} runs, {df[['bed', 'plant']].drop_duplicates().shape[0]} plants, {len(cols)} traits")

    curves = []
    if "onsets" not in a.skip:
        auc, onsets = run_onsets(df, cols, n_boot=a.n_boot)
        curves.append(auc)
        _save(onsets, out, "onsets")
        if not onsets.empty:
            pb = onsets[
                [
                    "signal",
                    "label",
                    "sensor",
                    "group",
                    "placebo_auc",
                    "placebo_ci_low",
                    "placebo_ci_high",
                    "bed_effect",
                    "n_deficit",
                    "n_control",
                    "fires_before_cutoff",
                    "precutoff_fire_day",
                    "null_auc_max",
                ]
            ]
            _save(pb, out, "placebo")
    if "slots" not in a.skip:
        _save(run_slot_onsets(df, cols, n_boot=a.n_boot), out, "onsets_by_slot")
    if "benchmark" not in a.skip:
        bench, bcurves = run_benchmark(df, n_boot=a.n_boot)
        curves.append(bcurves)
        _save(bench, out, "benchmark")
    curves = [c for c in curves if not c.empty]
    if curves:
        _save(pd.concat(curves, ignore_index=True), out, "auc_by_day")
    if "scores" not in a.skip:
        _save(stress_scores(df), out, "scores_runs")
        SCORES_CATALOGUE.to_csv(out / "catalogue.csv", index=False)
        (out / "scores.parquet").unlink(missing_ok=True)  # old name
    if "truth" not in a.skip:
        _save(truth_link(df, cols), out, "truth_link")
    if "recovery" not in a.skip:
        _save(recovery(df, cols), out, "recovery")

    info = dict(
        finished=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        seconds=round(time.time() - t0, 1),
        n_boot=a.n_boot,
        traits=cols,
        plants=int(df[["bed", "plant"]].drop_duplicates().shape[0]),
        skipped=a.skip,
    )
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    clear_caches()
    print(f"done in {info['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
