"""Compare a ``--plant-model`` batch with the ExG batch of the same code (and the contract table).

    python scripts/compare_plant_mask.py \
        --new data/derived/threed_plantmask --base data/derived/threed_holefix \
        --old data/derived/threed --out data/derived/threed_plantmask/compare

Writes into ``--out``: ``qc_rates.csv`` (flag rates per version, all runs and 09-01..09-14 per
slot), ``mask_sources.csv`` (frame mask sources per slot), ``trajectories.csv`` (bed medians of
``height_p75`` and ``projected_area`` per date, 15 h, 09-01..09-14, qc_pass runs of each version),
``e1_<version>.csv`` (the E1 noise floor of each table) and ``summary.md``. Every comparison uses
the runs present in all the versions given (matched on bed, plant, date, hour).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from uiwue.threed import experiments, schema

KEYS = list(schema.KEYS)
WINDOW = (date(2025, 9, 1), date(2025, 9, 14))
FLAGS = ["qc_pass", "qc_overlap", "qc_high_spread", "qc_few_points", "qc_no_plant", "qc_not_centred", "qc_dark"]
TRAJ_TRAITS = ("height_p75", "projected_area")
E1_TRAITS = list(schema.TRAITS) + ["cyl_height_p75", "cyl_projected_area"]


def load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    runs = experiments.prepare_runs(pd.read_parquet(d / "traits_runs.parquet"))
    frames = experiments.load_frames(d / "traits_frames.parquet")
    return runs, frames


def matched(tables: dict[str, tuple]) -> dict[str, tuple]:
    keys = None
    for runs, _ in tables.values():
        k = runs[KEYS].drop_duplicates()
        keys = k if keys is None else keys.merge(k, on=KEYS)
    out = {}
    for name, (runs, frames) in tables.items():
        r = runs.merge(keys, on=KEYS)
        f = None if frames is None else frames.merge(keys, on=KEYS)
        out[name] = (r, f)
    return out


def qc_rates(tables) -> pd.DataFrame:
    rows = []
    for name, (r, _) in tables.items():
        win = r[(r["date"] >= WINDOW[0]) & (r["date"] <= WINDOW[1])]
        for scope, sub in [("all", r), *[(f"window_{s:02d}h", g) for s, g in win.groupby("slot")], ("window", win)]:
            row = {"version": name, "scope": scope, "n_runs": len(sub)}
            row.update({f: float(sub[f].mean()) for f in FLAGS})
            rows.append(row)
    return pd.DataFrame(rows)


def mask_sources(frames: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    f = frames[frames["built"] & frames["used"]].merge(runs[[*KEYS, "slot"]], on=KEYS)
    t = pd.crosstab(f["slot"], f["mask_source"], normalize="index")
    t["n_frames"] = f.groupby("slot").size()
    return t.reset_index()


def trajectories(tables) -> pd.DataFrame:
    rows = []
    for name, (r, _) in tables.items():
        p = r[r["qc_pass"] & (r["slot"] == 15) & (r["date"] >= WINDOW[0]) & (r["date"] <= WINDOW[1])]
        p = p[p["bed"].isin(["W1", "W2"])]
        g = p.groupby(["bed", "date"])
        for (bed, d), sub in g:
            row = {"version": name, "bed": bed, "date": d, "n_runs": len(sub)}
            row.update({t: float(sub[t].median()) for t in TRAJ_TRAITS})
            rows.append(row)
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", type=Path, required=True, help="the --plant-model batch")
    ap.add_argument("--base", type=Path, required=True, help="ExG batch of the same code")
    ap.add_argument("--old", type=Path, help="the contract table directory (older code)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    raw = {"exg": load(a.base), "model": load(a.new)}
    if a.old:
        raw["contract"] = load(a.old)
    tables = matched(raw)
    pairs = matched({k: raw[k] for k in ("exg", "model")})
    n = len(next(iter(tables.values()))[0])

    q = qc_rates(tables)
    q.to_csv(a.out / "qc_rates.csv", index=False, lineterminator="\n")
    ms = mask_sources(pairs["model"][1], pairs["model"][0])
    ms.to_csv(a.out / "mask_sources.csv", index=False, lineterminator="\n")
    tr = trajectories(tables)
    tr.to_csv(a.out / "trajectories.csv", index=False, lineterminator="\n")
    e1s = {}
    for name, (r, f) in tables.items():
        e = experiments.e1(r, f, [t for t in E1_TRAITS if t in r.columns])
        e.to_csv(a.out / f"e1_{name}.csv", index=False, lineterminator="\n")
        e1s[name] = e.set_index("trait")
    e1 = pd.concat({k: v[["noise", "signal", "snr", "reliable"]] for k, v in e1s.items()}, axis=1)
    e1.columns = [f"{t}_{v}" for v, t in e1.columns]
    e1 = e1.reset_index()

    wide = tr.pivot_table(index=["bed", "date"], columns="version", values=list(TRAJ_TRAITS)).round(1)
    wide.columns = [f"{t}_{v}" for t, v in wide.columns]
    runs_src = pairs["model"][0]["mask_source"].value_counts(normalize=True).rename("share").reset_index()
    text = [
        f"# Plant mask comparison\n\n{n} runs present in every version ({', '.join(tables)}).\n",
        "## QC rates\n",
        md(q[q["scope"].isin(["all", "window"])]),
        "\n\n09-01..09-14 per slot:\n",
        md(q[q["scope"].str.startswith("window_")]),
        "\n\n## Mask source of the used frames, per slot (model batch)\n",
        md(ms),
        "\n\nRuns by `mask_source`:\n",
        md(runs_src),
        "\n\n## Bed medians, 15 h, qc_pass runs of each version\n",
        md(wide.reset_index(), "{:.1f}"),
        "\n\n## E1 noise floor\n",
        md(e1),
        "\n",
    ]
    (a.out / "summary.md").write_text("\n".join(text), encoding="utf-8", newline="\n")
    print((a.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
