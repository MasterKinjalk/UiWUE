"""Ground-truth module: the onset rule, the symptom filter and the RWC sheet parsing, on small synthetic inputs."""

import pandas as pd

from uiwue.truth import NOT_STRESS, ONSET_SHIFT, STRESS_WORDS, clip_onset, rwc_table


def _is_stress(note):
    note = note.lower()
    return bool(STRESS_WORDS.search(note)) and not NOT_STRESS.search(note)


def test_symptom_filter_keeps_water_stress_and_drops_pests():
    for note in (
        "wilting",
        "lower leaves wilting",
        "older leaves yellowing, developing brown spots",
        "almost completely slumped over",
    ):
        assert _is_stress(note), note
    for note in (
        "aphids",
        "dead fly on one leaf",
        "powdery mildew, small white splotches on one flopped lying leaf",
        "regenerated, roots as well",
        "leaf-miner damage",
    ):
        assert not _is_stress(note), note


def test_clip_onset_needs_two_days_in_a_row():
    tz = "Europe/Berlin"
    t = pd.date_range("2025-09-02 12:00", "2025-09-12 12:00", freq="12h", tz=tz)
    base = [100.0] * len(t)
    jump_one_day = [100 + (ONSET_SHIFT + 3 if d.day == 8 else 0) for d in t]  # a one-day spike is not an onset
    sustained = [100 + (ONSET_SHIFT + 3 if d.day >= 9 else 0) for d in t]
    leaf = pd.concat(
        [
            pd.DataFrame({"berlin_timestamp": t, "sensor_number": str(i), "turgic_value": v})
            for i, v in ((1, base), (2, jump_one_day), (3, sustained))
        ]
    )
    out = clip_onset(leaf, {1: ("W1", "A2"), 2: ("W2", "A2"), 3: ("W3", "A2")})
    assert pd.isna(out[("W1", "A2")]) and pd.isna(out[("W2", "A2")])
    assert out[("W3", "A2")] == pd.Timestamp("2025-09-09")


def test_rwc_sheet_dates_and_second_leaf(tmp_path):
    sheet = pd.DataFrame(
        {
            "day": [1, 12, 12],
            "date": [4.09, 15.09, 15.09],
            "plant": ["Q6", "D7_1", "D7_2"],
            "fw": [0.5, 0.4, 0.5],
            "sw": [0.6, 0.45, 0.6],
            "dw": [0.1, 0.1, 0.1],
            "rwc": [None] * 3,
        }
    )
    path = tmp_path / "m.xlsx"
    with pd.ExcelWriter(path) as w:
        sheet.to_excel(w, sheet_name="RWC", index=False)
    r = rwc_table(path)
    assert list(r.plant) == ["Q6", "D7"] and list(r.date.dt.day) == [4, 15]
    assert abs(r.truth_rwc[0] - 80.0) < 1e-9  # (0.5 - 0.1) / (0.6 - 0.1)
    assert abs(r.truth_rwc_leaf2[1] - 80.0) < 1e-9 and pd.isna(r.truth_rwc_leaf2[0])
