"""The fusion v2 lock-box is fixed, stratified, and re-drawn identically."""

import pandas as pd

from uiwue import core
from uiwue.fusion import lockbox


def test_lockbox_redraws_identically_and_is_stratified():
    plants = core.read_table("truth", "plants")
    if plants is None:  # the truth table ships in derived/; skip only if a checkout lacks it
        return
    assert lockbox.draw(plants) == sorted(lockbox.LOCKBOX)
    p = plants.assign(pid=plants["bed"] + "_" + plants["plant"])
    lock = p[p["pid"].isin(lockbox.LOCKBOX)]
    assert len(lock) == len(lockbox.LOCKBOX) == 45
    assert lock.groupby("bed").size().to_dict() == {"W1": 15, "W2": 15, "W3": 15}
    assert int(lock["truth_has_clip"].fillna(False).sum()) == 5


def test_draw_is_deterministic_per_stratum():
    p = pd.DataFrame(
        {
            "bed": ["W1"] * 10 + ["W2"] * 10,
            "plant": [f"A{i}" for i in range(1, 9)] + ["B1", "B2"] + [f"C{i}" for i in range(1, 9)] + ["D1", "D2"],
            "truth_has_clip": [False] * 20,
        }
    )
    a, b = lockbox.draw(p), lockbox.draw(p.sample(frac=1, random_state=1))
    assert a == b and len(a) == 4
    assert lockbox.is_lockbox(["W1_B3", "W1_A2"]).tolist() == [True, False]
