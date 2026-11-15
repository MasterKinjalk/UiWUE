"""The lock-box: plants kept out of every fusion v2 experiment (docs/fusion.md section 8).

    python -m uiwue.fusion.lockbox      # print the list and check that it re-draws the same

About 20 % of the 216 plants, drawn at random within each bed x turgor-clip stratum
(derived/truth/plants.parquet) with a fixed seed, fixed on 2026-11-14 before any v2
experiment. No v2 model is trained, tuned or selected on these plants, and they are never
a control reference for other plants. The chosen model scores them once, at the end.

The list is written out below so that a change to the truth table cannot change it
silently; `draw()` re-creates it and the tests check that the two agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uiwue import core
from uiwue.stats.core import SEED

FRACTION = 0.2

# draw(core.read_table("truth", "plants")) on 2026-11-14: 15 W1 (1 with a clip),
# 15 W2 (2 with clips), 15 W3 (2 with clips).
LOCKBOX = (
    "W1_B3", "W1_B5", "W1_E6", "W1_G6", "W1_H5", "W1_I6", "W1_J3", "W1_K8", "W1_L3", "W1_M6",
    "W1_N3", "W1_O2", "W1_P7", "W1_R3", "W1_R7",
    "W2_B3", "W2_C4", "W2_E4", "W2_I6", "W2_I8", "W2_J7", "W2_K8", "W2_N1", "W2_N7", "W2_O6",
    "W2_P1", "W2_Q2", "W2_R1", "W2_R3", "W2_R7",
    "W3_A6", "W3_A8", "W3_B1", "W3_C2", "W3_D3", "W3_D7", "W3_H1", "W3_J1", "W3_K4", "W3_K6",
    "W3_K8", "W3_L1", "W3_P1", "W3_R1", "W3_R7",
)  # fmt: skip


def draw(plants: pd.DataFrame, frac: float = FRACTION, seed: int = SEED) -> list[str]:
    """round(frac * n) plants at random from each (bed, has clip) stratum, strata in sorted order."""
    p = plants.assign(
        pid=plants["bed"] + "_" + plants["plant"], clip=plants["truth_has_clip"].fillna(False).astype(bool)
    )
    rng = np.random.default_rng(seed)
    out = []
    for _, g in p.sort_values("pid").groupby(["bed", "clip"], sort=True):
        out += list(rng.permutation(g["pid"].to_numpy())[: round(len(g) * frac)])
    return sorted(out)


def is_lockbox(pids) -> np.ndarray:
    return np.isin(np.asarray(pids, dtype=object), LOCKBOX)


def main() -> None:
    plants = core.read_table("truth", "plants")
    again = draw(plants)
    p = plants.assign(pid=plants["bed"] + "_" + plants["plant"])
    lock = p[p["pid"].isin(LOCKBOX)]
    print(f"{len(LOCKBOX)} lock-box plants of {len(p)}; re-draw matches: {again == sorted(LOCKBOX)}")
    print(lock.groupby(["bed", lock["truth_has_clip"].fillna(False)]).size().to_string())


if __name__ == "__main__":
    main()
