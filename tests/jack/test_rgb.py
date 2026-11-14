import numpy as np
import pandas as pd
import pytest

from uiwue.rgb.features import frame_traits
from uiwue.rgb.run import assign_slots, extract, pick_frames


def _img(green_frac=0.3, plant=(60, 140, 50), soil=(110, 90, 70), dark=False, size=(1080, 1920)):
    h, w = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = soil
    side = int(min(h, w) * 0.5)
    r = int(side * np.sqrt(green_frac) / 2)
    cy, cx = h // 2, w // 2
    img[cy - r : cy + r, cx - r : cx + r] = plant
    if dark:
        img = (img * 0.1).astype(np.uint8)
    return img


def test_frame_traits_green_plant():
    t = frame_traits(_img(0.3))
    assert t["rgb_green_cover"] == pytest.approx(0.3, abs=0.02)
    assert 90 < t["rgb_hue"] < 120
    assert t["rgb_exg"] > 0 and t["rgb_ngrdi"] > 0
    assert not t["dark"] and not t["mask_failed"]


def test_yellowing_lowers_hue_and_greenness():
    g = frame_traits(_img(0.3, plant=(60, 140, 50)))
    y = frame_traits(_img(0.3, plant=(130, 140, 40)))
    assert y["rgb_hue"] < g["rgb_hue"]
    assert y["rgb_ngrdi"] < g["rgb_ngrdi"]


def test_dark_frame_flagged():
    assert frame_traits(_img(dark=True))["dark"]


def test_assign_slots_prefers_scheduled_then_most_frames():
    runs = pd.DataFrame(
        dict(bed="W1", plant="A2", date=pd.Timestamp("2025-09-01").date(),
             folder_hour=[15, 16, 9, 10, 11, 13], n_images=[0, 15, 15, 3, 15, 15], path="x")
    )
    r = assign_slots(runs).set_index("hour")["folder_hour"]
    assert r[15] == 16  # scheduled folder empty -> re-run
    assert r[9] == 9
    assert r[12] in (11, 13)
    assert set(r.index) == {9, 12, 15}


def test_extract_from_zip_matches_folder(tmp_path):
    pytest.importorskip("PIL")
    import zipfile

    from PIL import Image

    src = tmp_path / "src" / "W1_A2" / "2025_09_05" / "15"
    src.mkdir(parents=True)
    for i in range(15):
        Image.fromarray(_img(0.3, size=(270, 480))).save(src / f"rgb_{i:03d}.png")
    (src / "frames_rgb.csv").write_text("frame_id,timestamp\n0,1757080000000000000\n")
    (tmp_path / "z" / "rgb").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "z" / "rgb" / "rgb_W1_A2.zip", "w") as z:
        for f in src.iterdir():
            z.write(f, f"W1_A2/2025_09_05/15/{f.name}")
    # a part-extracted folder next to the zip must be ignored
    (tmp_path / "z" / "rgb" / "W1_A2" / "2025_09_05").mkdir(parents=True)
    from_zip = extract(tmp_path / "z")
    assert len(from_zip) == 1 and from_zip["rgb_qc_pass"].all()
    assert from_zip.loc[0, "rgb_green_cover"] == pytest.approx(0.3, abs=0.02)
    assert from_zip.loc[0, "rgb_timestamp_utc"].year == 2025


def test_pick_frames(tmp_path):
    imgs = [tmp_path / f"rgb_{i:03d}.jpg" for i in range(15)]
    assert pick_frames(imgs, (7,)) == [imgs[7]]
    assert pick_frames(imgs[:3], (7,)) == [imgs[1]]


def test_extract_end_to_end(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    for bed, plant, frac in [("W1", "A2", 0.3), ("W2", "J5", 0.1)]:
        for hh, dark in [("12", False), ("06", True)]:
            d = tmp_path / "rgb" / f"{bed}_{plant}" / "2025_09_05" / hh
            d.mkdir(parents=True)
            for i in range(15):
                Image.fromarray(_img(frac, dark=dark, size=(270, 480))).save(d / f"rgb_{i:03d}.png")
    df = extract(tmp_path)
    assert len(df) == 4
    assert {"bed", "plant", "date", "hour", "rgb_green_cover", "rgb_qc_pass"} <= set(df.columns)
    day = df[df["hour"] == 12].set_index("plant")
    assert day.loc["A2", "rgb_green_cover"] > day.loc["J5", "rgb_green_cover"]
    assert not df.loc[df["hour"] == 6, "rgb_qc_pass"].any()
    assert df.loc[df["hour"] == 6, "rgb_qc_dark"].all()
