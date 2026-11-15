"""Per-frame agreement of the learned plant mask with the ExG isolation, by bed and period.

    python scripts/plant_mask_agreement.py --masks data/derived/threed_plantmask \
        --refs data/derived/threed/plant_refs.csv --out data/derived/threed_plantmask/compare

For every lit run of W1 and W2 in 2025-09-01..09-14 (slots 09-18 h), both isolations are run on
the same clouds, plane and centre. Per used frame with a model mask: IoU of the two plants on the
points with colour inside the plant cylinder, the points only ExG keeps and only the model keeps,
and the ratio of plant points. Grouped by bed and period (09-01..09-07, 09-08..09-14: the
deficit bed wilts from about 09-08, and PhenoBench has no wilted plants). Writes
``agreement_frames.parquet`` and ``agreement.csv``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from uiwue.threed import run, segment
from uiwue.threed.io import list_runs
from uiwue.threed.plantmask import SOURCE_MODEL, PlantMasks

SPLIT = date(2025, 9, 8)


def one(r, data_dir: Path, masks: Path, prior) -> list[dict]:
    try:
        g0 = run.process_run(r, data_dir, prior, debug=True, want_cloud=False)["debug"]
        g1 = run.process_run(r, data_dir, prior, debug=True, want_cloud=False, plant_mask_fn=PlantMasks(masks))
    except Exception:  # noqa: BLE001
        return []
    g1 = g1["debug"]
    rows = []
    for k in g1["used"]:
        c, loc = g1["clouds"][k], g1["loc"][k]
        info = g1["mask_info"][k]
        cyl = np.hypot(loc[:, 0], loc[:, 1]) <= segment.CYLINDER_RADIUS
        a = g0["iso"][k].mask & c.has_rgb & cyl
        b = g1["iso"][k].mask & c.has_rgb & cyl
        rows.append(
            dict(
                bed=r.bed,
                plant=r.plant,
                date=pd.Timestamp(r.date.replace("_", "-")).date(),
                hour=int(r.hour),
                frame=k,
                source=info["source"],
                n_instances=info["n_instances"],
                iou=float((a & b).sum() / max((a | b).sum(), 1)),
                exg_only=int((a & ~b).sum()),
                model_only=int((b & ~a).sum()),
                n_exg=int(g0["iso"][k].mask.sum()),
                n_model=int(g1["iso"][k].mask.sum()),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--masks", type=Path, required=True, help="--out of the --plant-model batch")
    ap.add_argument("--refs", type=Path, required=True, help="plant_refs.csv of that batch")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plants", nargs="+")
    ap.add_argument("--hours", nargs="+", default=["09", "12", "15", "18"])
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    refs = run.load_plant_refs(a.refs)
    runs = list_runs(a.data_dir, "depth", plants=a.plants, dates=("2025_09_01", "2025_09_14"), hours=a.hours)
    runs = [r for r in runs if r.bed in ("W1", "W2")]
    rows = Parallel(n_jobs=a.jobs)(delayed(one)(r, a.data_dir, a.masks, refs.get(r.plant_id)) for r in runs)
    df = pd.DataFrame([x for rr in rows for x in rr])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out / "agreement_frames.parquet", index=False)
    df["period"] = np.where(df["date"] >= SPLIT, "09-08..09-14", "09-01..09-07")
    m = df[df["source"] == SOURCE_MODEL]
    out = m.groupby(["bed", "period"]).agg(
        frames=("iou", "size"),
        iou_median=("iou", "median"),
        iou_p25=("iou", lambda x: x.quantile(0.25)),
        exg_only_median=("exg_only", "median"),
        model_only_median=("model_only", "median"),
        points_ratio_median=("n_model", lambda x: float(np.median(x / m.loc[x.index, "n_exg"].clip(lower=1)))),
    )
    fall = df.groupby(["bed", "period"])["source"].value_counts(normalize=True).unstack(fill_value=0)
    out = out.join(fall.add_prefix("share_")).reset_index()
    out.to_csv(a.out / "agreement.csv", index=False, lineterminator="\n")
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
