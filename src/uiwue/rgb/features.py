"""RGB colour traits from one webcam frame (Logitech C920, 1920 x 1080 JPEG).

The "visible" baseline signal: what a cheap camera sees. Centre crop, ExG
vegetation mask (paper rule: 2G - R - B > 5 on 8-bit values), then colour
statistics over vegetation pixels.
"""

from __future__ import annotations

import numpy as np

EXG_THRESHOLD = 5.0  # 8-bit ExG, the ADA paper's vegetation rule
# Mean 8-bit brightness below this = dark run. Night frames (06 h, most 18 h) sit
# near 4; daytime frames are dim (dark soil, shade cloth), median ~40 at 12 h,
# and are still segmentable at 20. 40 (the 3D plan's value) failed half of the
# 12 h runs on the first 18 plants, so the cutoff is 12.
DARK_BRIGHTNESS = 12.0
MIN_VEG_PIXELS = 200

TRAITS = {
    # column: (label, unit, direction, description)
    "rgb_green_cover": (
        "Green cover",
        "fraction",
        -1,
        "Share of centre-crop pixels classed as vegetation (2G-R-B > 5).",
    ),
    "rgb_exg": (
        "Greenness (ExG)",
        "",
        -1,
        "Mean excess green 2g-r-b on chromatic coordinates, vegetation pixels.",
    ),
    "rgb_ngrdi": (
        "NGRDI",
        "",
        -1,
        "Mean (G-R)/(G+R) over vegetation pixels.",
    ),
    "rgb_hue": (
        "Mean hue",
        "deg",
        -1,
        "Circular mean HSV hue of vegetation pixels; yellowing moves it down from ~100.",
    ),
    "rgb_brightness": (
        "Brightness",
        "0-255",
        0,
        "Mean brightness of the whole centre crop (exposure check, not a trait).",
    ),
}


def centre_crop(img: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Central `frac` of the height, as a square, so neighbours stay mostly out."""
    h, w = img.shape[:2]
    side = round(min(h, w) * frac)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _hue_deg(rgb: np.ndarray) -> np.ndarray:
    """HSV hue in degrees for an (N, 3) float array in 0-255."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    c = mx - mn
    h = np.zeros(len(rgb))
    nz = c > 0
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / c[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / c[gm] + 2
    h[bm] = (r[bm] - g[bm]) / c[bm] + 4
    return h * 60.0


def frame_traits(img: np.ndarray, crop: float = 0.5) -> dict:
    """Traits for one RGB frame, an (H, W, 3) uint8 array in R, G, B order."""
    x = centre_crop(np.asarray(img), crop).astype(np.float64)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    brightness = float(x.mean())
    veg = (2 * G - R - B) > EXG_THRESHOLD
    n_veg = int(veg.sum())
    out = {
        "rgb_brightness": brightness,
        "rgb_green_cover": n_veg / veg.size,
        "rgb_exg": np.nan,
        "rgb_ngrdi": np.nan,
        "rgb_hue": np.nan,
        "dark": brightness < DARK_BRIGHTNESS,
        "mask_failed": n_veg < MIN_VEG_PIXELS,
    }
    if n_veg >= MIN_VEG_PIXELS:
        px = x[veg]
        s = px.sum(axis=1)
        s[s == 0] = 1
        chrom = px / s[:, None]
        out["rgb_exg"] = float(np.mean(2 * chrom[:, 1] - chrom[:, 0] - chrom[:, 2]))
        gr = px[:, 1] + px[:, 0]
        gr[gr == 0] = 1
        out["rgb_ngrdi"] = float(np.mean((px[:, 1] - px[:, 0]) / gr))
        ang = np.deg2rad(_hue_deg(px))
        out["rgb_hue"] = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    return out
