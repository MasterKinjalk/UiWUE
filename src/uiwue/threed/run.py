"""ADA 2025 depth + RGB runs -> per-frame and per-run plant traits (plan sections 4-6).

Usage::

    python -m uiwue.threed.run --data-dir data --plants W1_J5 W2_J5 \
        --dates 2025-08-12 2025-09-19 --hours 15 --jobs 20
    python -m uiwue.threed.run --data-dir data --parity          # paper-parity heights (E0a)
    python -m uiwue.threed.run --data-dir data --diagnose W1_J5:2025-09-03:15   # figures
    python -m uiwue.threed.run --data-dir data --depth-model runs/promptda-vitl/best \
        --hours 15 --out data/derived/threed_refined                # refined depth (GPU)
    python -m uiwue.threed.run --data-dir data --plant-model runs/leafseg-m2f/best \
        --out data/derived/threed_plantmask                         # learned plant mask (GPU)

Writes into ``--out`` (default ``data/derived/threed``): ``traits_frames.parquet``,
``traits_runs.parquet``, ``clouds/<bed>_<plant>/<date>_<HH>.npz``, ``run_log.csv`` and
``plant_refs.csv`` (each plant's carried soil reference). Runs already in ``traits_runs.parquet``
are skipped, so an interrupted batch resumes where it stopped.

Per run (details in docs/threed/PIPELINE.md):

1. Robot poses of the 15 frames. The plant centre is where the frame-7 RGB optical axis meets the
   soil (it lands on the pot centre); its position in every other frame follows from the robot
   offsets (``cloud.frame_offset``).
2. Frames whose whole plant cylinder lies inside the depth and the RGB views are used.
3. One soil plane per run, RANSAC on the pooled non-vegetation points around the centre of the
   used frames; if rejected, the plant's carried reference (``qc_ref_carried``).
4. Per used frame: vegetation mask (ExG / height rule, or the learned plant mask with
   ``--plant-model``), DBSCAN + cylinder isolation, traits.
5. Median and MAD over the used frames, QC flags; frame 7 is cached as the viewer cloud.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from uiwue.threed import ground, qc, schema, segment, traits
from uiwue.threed.cloud import (
    Cloud,
    build_cloud,
    depth_to_rgb,
    fill_small_holes,
    frame_offset,
    in_depth_fov,
    median3,
    points_to_depth_grid,
    project_to_rgb,
)
from uiwue.threed.io import (
    CENTRE_FRAME,
    Run,
    list_runs,
    load_depth,
    load_rgb_reduced,
    run_frames,
)
from uiwue.threed.plantmask import SOURCE_MODEL, SOURCE_RULE, PlantMasks, precompute_masks

SOIL_Z_GUESS = 0.47  # m, typical pot-soil depth in frame 7 (W1_J5 / W2_J5, 2025-08)
# Plant centre in frame-7 depth coordinates when the run has no poses (median of the RGB-axis
# centres of runs with poses, see plant_refs.csv).
CENTRE_DEFAULT_XY = (0.025, 0.060)
FOOTPRINT_MIN_DEPTH = 0.98  # share of the cylinder footprint inside the ToF view
FOOTPRINT_MIN_RGB = 0.95  # ... and inside the RGB view
RGB_REDUCE = 2  # decode RGB at 1/2 resolution (960 x 540; one ToF pixel is ~3 px there)
CLOUD_RADIUS = 0.15  # m, cached viewer cloud: points within this distance of the centre
PRIOR_DAYS = 7  # the plant's carried reference comes from its first PRIOR_DAYS dates on disk
MIN_PRIOR_RUNS = 3  # ... and needs at least this many accepted soil fits
MAX_POOL = 6000  # pooled soil candidates for the run plane (subsampled beyond)
LOCAL_UTC_OFFSET_H = 2  # CEST, for runs without frame timestamps


# ---------------------------------------------------------------- per-plant reference
@dataclass
class PlantRef:
    plane7: ground.Plane | None  # median accepted soil plane, frame-7 depth coordinates
    centre_xy: tuple[float, float]  # median plant centre, frame-7 depth coordinates
    n_runs: int


def load_plant_refs(path: Path) -> dict[str, PlantRef]:
    if not Path(path).exists():
        return {}
    refs = {}
    for r in pd.read_csv(path).itertuples():
        plane = None
        if np.isfinite(r.nz):
            plane = ground.Plane(np.array([r.nx, r.ny, r.nz]), float(r.d))
        refs[r.plant_id] = PlantRef(plane, (float(r.centre_x), float(r.centre_y)), int(r.n_runs))
    return refs


def save_plant_refs(refs: dict[str, PlantRef], path: Path) -> None:
    rows = []
    for pid, r in sorted(refs.items()):
        n = r.plane7.normal if r.plane7 is not None else [np.nan] * 3
        d = r.plane7.d if r.plane7 is not None else np.nan
        rows.append(
            dict(
                plant_id=pid,
                n_runs=r.n_runs,
                nx=n[0],
                ny=n[1],
                nz=n[2],
                d=d,
                soil_z_centre=r.plane7.z_at(*r.centre_xy) if r.plane7 is not None else np.nan,
                centre_x=r.centre_xy[0],
                centre_y=r.centre_xy[1],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


# ---------------------------------------------------------------- geometry helpers
def nominal_positions() -> dict[int, np.ndarray]:
    """Robot positions of the 15 frames relative to frame 7 on the regular 3 x 5 grid (used
    when a run has no frames_*.csv)."""
    return {k: np.array([(k % 3 - 1) * 0.1, (k // 3 - 2) * 0.1, 0.0]) for k in range(15)}


def rgb_axis_centre(pos_d7, pos_r7, plane7: ground.Plane) -> np.ndarray:
    """Where the frame-7 RGB optical axis meets ``plane7`` (frame-7 depth coordinates)."""
    ti = np.linalg.inv(depth_to_rgb(pos_d7, pos_r7))
    o, dv = ti[:3, 3], ti[:3, :3] @ np.array([0.0, 0.0, 1.0])
    t = -(plane7.d + plane7.normal @ o) / (plane7.normal @ dv)
    return o + t * dv


def on_plane(plane: ground.Plane, xy) -> np.ndarray:
    return np.array([xy[0], xy[1], plane.z_at(xy[0], xy[1])])


def footprint(plane: ground.Plane, centre: np.ndarray, radius: float, n_ring: int = 6) -> np.ndarray:
    """Sample points of the disc of ``radius`` around ``centre`` on ``plane``."""
    e1, e2 = plane.basis()
    pts = [centre]
    for i in range(1, n_ring + 1):
        r = radius * i / n_ring
        m = 8 * i
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        pts.append(centre + r * (np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2)))
    return np.vstack([np.atleast_2d(p) for p in pts])


def coverage(plane_k, centre_k, pos_dk=None, pos_rk=None, radius=segment.CYLINDER_RADIUS):
    """(depth, rgb) share of the plant cylinder footprint inside the ToF / RGB views."""
    fp = footprint(plane_k, centre_k, radius)
    dcov = float(in_depth_fov(fp).mean())
    if pos_dk is None or pos_rk is None:
        return dcov, float("nan")
    _, _, _, ok, _ = project_to_rgb(fp, depth_to_rgb(pos_dk, pos_rk))
    return dcov, float(ok.mean())


# ---------------------------------------------------------------- one run
def _timestamp(ts_ns: list[int], run: Run) -> pd.Timestamp:
    if ts_ns:
        return pd.Timestamp(int(np.median(ts_ns)), unit="ns", tz="UTC")
    d = schema.folder_date(run.date)
    local = datetime(d.year, d.month, d.day, int(run.hour), tzinfo=UTC)
    return pd.Timestamp(local - timedelta(hours=LOCAL_UTC_OFFSET_H))


def _dedupe(depths: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Drop frames whose depth image repeats another frame (edge plants: the robot clamps a
    column, and the clamped frames are stored twice). Frame 7, then centre-column frames win."""
    seen, keep = {}, {}
    for k in sorted(depths, key=lambda k: (k != CENTRE_FRAME, k % 3 != 1, k)):
        h = hashlib.blake2b(depths[k].tobytes(), digest_size=16).digest()
        if h in seen:
            continue
        seen[h] = k
        keep[k] = depths[k]
    return dict(sorted(keep.items()))


def process_run(
    run: Run,
    data_dir: Path,
    prior: PlantRef | None = None,
    *,
    reference_only: bool = False,
    plan_only: bool = False,
    want_cloud: bool = True,
    debug: bool = False,
    plant_mask_fn=None,
    depth_fn=None,
) -> dict:
    """Process one run. Returns ``{"status", "error", "frames", "run", "cloud", "ref"}``.

    ``plant_mask_fn(run, k, cloud, local)`` replaces the ExG + height vegetation rule of frame k
    with a learned per-point plant mask (``plantmask.PlantMasks``, used by ``--plant-model``).
    ``local`` is the frame's plane-local coordinates with the plant centre at a = b = 0. It
    returns None (keep the rule), a bool array, or ``(mask or None, info)``; ``info["source"]``
    and its ``n_instances`` / ``score`` / ``n_merged`` go into the frame columns ``mask_source``,
    ``mask_instances``, ``mask_score`` and ``mask_merged``. The soil candidates of the reference plane always use
    the ExG rule: a centre-plant mask would make the neighbours' leaves over the pot soil.

    ``depth_fn(run, k, depth)`` replaces the ToF depth of a built frame k with a refined map on the
    same 224 x 171 grid. It returns ``(depth, refined_mask)`` (the mask marks pixels whose depth
    is refined; the rest is ToF), a plain depth array (all refined), or None (keep the ToF). The
    map is used as is: no hole filling and no median filter (``RefinedDepth`` applies both to its
    ToF fallback pixels). Used by ``--depth-model``.

    ``plan_only`` returns after frame selection with ``out["build"]``, the frames that would be
    built, and ``out["no_pose"]`` (the refined-depth and plant-mask pre-passes use it).
    """
    data_dir = Path(data_dir)
    out = {"status": "ok", "error": "", "frames": [], "run": None, "cloud": None, "ref": None}
    fr = run_frames(run, data_dir)
    if not fr["depth_files"]:
        out["status"] = "no_depth"
        return out
    pos_d, pos_r = fr["depth_pos"], fr["rgb_pos"]
    k7 = CENTRE_FRAME
    no_pose = not pos_d or not pos_r or k7 not in pos_d or k7 not in pos_r
    if no_pose:
        pos_d, pos_r = nominal_positions(), {}

    depths, nirs = {}, {}
    for k, f in fr["depth_files"].items():
        if k not in pos_d:
            continue
        z, nir = load_depth(f)
        depths[k], nirs[k] = z, nir
    if k7 not in depths:
        out["status"] = "no_frame7"
        return out
    depths = _dedupe(depths)

    # -- frame geometry: plant centre and cylinder coverage per frame
    o7 = frame_offset(pos_d[k7])
    rel = {k: frame_offset(pos_d[k]) - o7 for k in depths}  # p_k = p_7 - rel[k]
    plane7 = (prior.plane7 if prior and prior.plane7 is not None else None) or ground.Plane.horizontal(SOIL_Z_GUESS)

    def centre7_for(plane):
        if not no_pose:
            return rgb_axis_centre(pos_d[k7], pos_r[k7], plane)
        return on_plane(plane, prior.centre_xy if prior else CENTRE_DEFAULT_XY)

    c7 = centre7_for(plane7)
    cov = {}
    for k in depths:
        pk = plane7.shifted(rel[k])
        cov[k] = coverage(pk, c7 - rel[k], pos_d[k], pos_r.get(k) if not no_pose else None)
    used = [k for k in depths if cov[k][0] >= FOOTPRINT_MIN_DEPTH and (no_pose or cov[k][1] >= FOOTPRINT_MIN_RGB)]
    build = sorted(set(used) | {k7})
    if plan_only:
        out["build"] = build
        out["no_pose"] = no_pose
        return out

    # -- clouds (colour only with poses)
    clouds: dict[int, Cloud] = {}
    for k in build:
        rgb, t = None, None
        if not no_pose and k in pos_r:
            rgb = load_rgb_reduced(run.dir(data_dir, "rgb") / f"rgb_{k:03d}.jpg", RGB_REDUCE)
            t = depth_to_rgb(pos_d[k], pos_r[k])
        res = depth_fn(run, k, depths[k]) if depth_fn is not None else None
        if res is None:
            clouds[k] = build_cloud(depths[k], nirs[k], rgb, t, rgb_scale=1 / RGB_REDUCE)
        else:
            dref, rmask = res if isinstance(res, tuple) else (res, res > 0)
            c = build_cloud(dref, nirs[k], rgb, t, rgb_scale=1 / RGB_REDUCE, median=False, fill_holes=False)
            c.refined = np.asarray(rmask, bool)[c.rc[:, 0], c.rc[:, 1]]
            clouds[k] = c

    # -- reference plane: pooled soil candidates of the used frames, in frame-7 coordinates
    pool_frames = used or [k7]
    pts, nv = [], []
    for k in pool_frames:
        c = clouds[k]
        veg0 = c.has_rgb & (np.nan_to_num(c.exg, nan=-1e9) > segment.EXG_THRESHOLD)
        ck = c7 - rel[k]
        near = np.hypot(c.xyz[:, 0] - ck[0], c.xyz[:, 1] - ck[1]) < ground.SOIL_RADIUS
        pts.append(c.xyz[near] + rel[k])
        nv.append(~veg0[near])
    pts, nv = np.vstack(pts), np.concatenate(nv)
    keep_share = 1.0
    if len(pts) > MAX_POOL:
        keep_share = MAX_POOL / len(pts)
        sel = np.random.default_rng(0).choice(len(pts), MAX_POOL, replace=False)
        pts, nv = pts[sel], nv[sel]
    prior_plane = prior.plane7 if prior else None
    # plan 4.3: 150 inliers per frame; the pool holds len(pool_frames) frames, subsampled
    min_inl = max(ground.MIN_INLIERS, int(ground.MIN_INLIERS * len(pool_frames) * keep_share))
    fit = ground.fit_reference(pts, c7, nv, prior=prior_plane, min_inliers=min_inl)
    if fit.accepted:
        plane7, ref_source, carried = fit.plane, "fit", False
    elif prior_plane is not None:
        plane7, ref_source, carried = prior_plane, "carried", True
    elif fit.plane is not None and fit.tilt_deg <= ground.MAX_TILT_DEG:
        plane7, ref_source, carried = fit.plane, "rejected_fit", True
    else:
        plane7, ref_source, carried = ground.Plane.horizontal(SOIL_Z_GUESS), "default", True
    c7 = centre7_for(plane7)
    out["ref"] = {"fit": fit, "plane7": plane7 if fit.accepted else None, "centre7": c7}
    if reference_only:
        return out

    # -- centre refinement on frame 7 (plan 4.5 step 1)
    c = clouds[k7]
    loc = plane7.to_local(c.xyz, c7)
    dark7 = segment.is_dark(c.brightness)
    veg, _ = _veg(run, k7, c, loc, dark7, plant_mask_fn)
    da, db = segment.refine_centre(loc, veg)
    e1, e2 = plane7.basis()
    c7 = c7 + da * e1 + db * e2

    # -- isolation once over all used frames (same plant in every frame), then per-frame traits
    geo = {}
    for k, c in clouds.items():
        pk = plane7.shifted(rel[k])
        loc = pk.to_local(c.xyz, c7 - rel[k])
        dark = segment.is_dark(c.brightness)
        geo[k] = (pk, loc, dark, *_veg(run, k, c, loc, dark, plant_mask_fn))
    iso_frames = used if used else [k7]
    isos = dict(zip(iso_frames, _isolate(iso_frames, clouds, geo), strict=True))
    if k7 not in isos:  # frame 7 not used: isolate it on its own for the viewer cloud
        isos[k7] = _isolate([k7], clouds, geo)[0]
    rows, iso7, loc7 = [], isos[k7], geo[k7][1]
    dbg = {
        "clouds": clouds,
        "loc": {k: g[1] for k, g in geo.items()},
        "iso": isos,
        "veg": {k: g[3] for k, g in geo.items()},
        "mask_info": {k: g[4] for k, g in geo.items()},
        "rel": rel,
        "used": used,
        "fit": fit,
        "plane7": plane7,
        "c7": c7,
        "pool": (pts, nv),
        "cov": cov,
    }
    for k in sorted(depths):
        row = dict(
            bed=run.bed,
            plant=run.plant,
            date=schema.folder_date(run.date),
            hour=int(run.hour),
            frame=k,
            used=k in used,
            built=k in clouds,
            cov_depth=cov[k][0],
            cov_rgb=cov[k][1],
            robot_dx=float(pos_d[k][0] - pos_d[k7][0]),
            robot_dy=float(pos_d[k][1] - pos_d[k7][1]),
        )
        row.update(traits.nan_traits())
        row.update(
            ok=False,
            no_plant=True,
            clipped_fraction=np.nan,
            outside_fraction=np.nan,
            centroid_offset_mm=np.nan,
            n_clusters=0,
            brightness=np.nan,
            dark=False,
            n_veg=0,
            n_filled=0,
            refined_frac=np.nan,
            refined_frac_view=np.nan,
            mask_source="",
            mask_instances=0,
            mask_score=np.nan,
            mask_merged=0,
        )
        if k in clouds:
            c = clouds[k]
            pk, loc, dark, veg, minfo = geo[k]
            row.update(brightness=c.brightness, dark=dark, n_veg=int(veg.sum()))
            row.update(
                mask_source=minfo["source"],
                mask_instances=int(minfo["n_instances"]),
                mask_score=float(minfo["score"]),
                mask_merged=int(minfo["n_merged"]),
            )
            row.update(cylinder_traits(loc, veg, c.pixel_area))
        if k in isos:
            iso = isos[k]
            m = iso.mask
            tr = traits.frame_traits(loc[m], c.pixel_area[m], xyz=c.xyz[m], plane_normal=pk.normal)
            row.update(tr)
            row.update(
                ok=bool(not iso.no_plant and tr["n_points"] >= traits.MIN_POINTS),
                no_plant=iso.no_plant,
                clipped_fraction=iso.clipped_fraction,
                outside_fraction=iso.outside_fraction,
                centroid_offset_mm=1000 * iso.centroid_offset,
                n_clusters=iso.n_clusters,
                n_filled=int(c.filled[m].sum()) if c.filled is not None else 0,
                refined_frac=float(c.refined[m].mean()) if c.refined is not None and m.any() else np.nan,
                refined_frac_view=float(c.refined.mean()) if c.refined is not None else np.nan,
            )
        rows.append(row)
    fdf = pd.DataFrame(rows)

    # -- run row
    ts = [fr["depth_ts"][k] for k in used if k in fr["depth_ts"]]
    usedf = fdf[fdf["used"]]
    bright = float(np.nanmedian(usedf["brightness"])) if usedf["brightness"].notna().any() else np.nan
    d = schema.folder_date(run.date)
    meta = dict(
        bed=run.bed,
        plant=run.plant,
        date=d,
        hour=int(run.hour),
        slot=schema.slot_of(int(run.hour)),
        timestamp_utc=_timestamp(ts, run),
        treatment=schema.BED_TREATMENT.get(run.bed, "unknown"),
        days_since_cutoff=schema.days_since_cutoff(d),
        qc_dark=segment.is_dark(bright),
        qc_ref_carried=carried,
        qc_no_pose=no_pose,
        ref_source=ref_source,
        ref_inliers=fit.n_inliers,
        ref_tilt_deg=plane7.tilt_deg,
        ref_reject_reason=fit.reason,
        soil_z_mm=1000 * float(c7[2]),
        centre_x_mm=1000 * float(c7[0]),
        centre_y_mm=1000 * float(c7[1]),
        centre_shift_mm=1000 * float(np.hypot(da, db)),
        n_frames_on_disk=len(fr["depth_files"]),
        **mask_summary(usedf["mask_source"]),
    )
    out["run"] = qc.aggregate_run(fdf, meta)
    if debug:
        out["debug"] = dbg
    out["frames"] = rows

    # -- viewer cloud: frame 7, plant-centred
    if want_cloud and loc7 is not None:
        c = clouds[k7]
        sel = np.hypot(loc7[:, 0], loc7[:, 1]) < CLOUD_RADIUS
        out["cloud"] = dict(
            xyz=loc7[sel].astype(np.float32),
            rgb=c.rgb[sel],
            height=loc7[sel, 2].astype(np.float32),
            plant=iso7.mask[sel],
            frame=np.int64(k7),
        )
    return out


CYL_TRAITS = (
    "cyl_height_p50",
    "cyl_height_p75",
    "cyl_height_p95",
    "cyl_projected_area",
    "cyl_n_points",
)


def cylinder_traits(loc: np.ndarray, veg: np.ndarray, pixel_area: np.ndarray) -> dict:
    """Canopy inside the plant cylinder without clustering: all vegetation points within
    ``CYLINDER_RADIUS`` of the centre (extra columns ``cyl_*``; more stable than the DBSCAN
    isolation once the canopy is closed, but includes neighbours' leaves over the pot)."""
    m = veg & (np.hypot(loc[:, 0], loc[:, 1]) <= segment.CYLINDER_RADIUS)
    out = dict.fromkeys(CYL_TRAITS, float("nan"))
    out["cyl_n_points"] = float(m.sum())
    if m.sum() >= traits.MIN_POINTS:
        p50, p75, p95 = np.percentile(loc[m, 2], [50, 75, 95])
        out.update(
            cyl_height_p50=1000 * p50,
            cyl_height_p75=1000 * p75,
            cyl_height_p95=1000 * p95,
            cyl_projected_area=1e4 * float(pixel_area[m].sum()),
        )
    return out


def _veg(run, k, cloud: Cloud, local, dark, plant_mask_fn) -> tuple[np.ndarray, dict]:
    """(vegetation mask, mask info) of one frame: the learned plant mask where ``plant_mask_fn``
    gives one, else the ExG / height rule."""
    info = {"source": SOURCE_RULE, "n_instances": 0, "score": float("nan"), "n_merged": 0}
    learned = None
    if plant_mask_fn is not None:
        res = plant_mask_fn(run, k, cloud, local)
        learned, extra = res if isinstance(res, tuple) else (res, {})
        info |= {"source": SOURCE_MODEL if learned is not None else SOURCE_RULE} | dict(extra)
    has_colour = cloud.has_rgb.any()
    veg = segment.vegetation_mask(
        local[:, 2],
        cloud.exg if has_colour else None,
        cloud.has_rgb if has_colour else None,
        dark=dark,
        learned=learned,
    )
    return veg, info


def _isolate(frames: list[int], clouds: dict, geo: dict) -> list:
    """DBSCAN + cylinder isolation over ``frames``. Without a learned mask in any of them this is
    the ExG / height path (``isolate_plant_multi`` on the vegetation points). With one, the
    frames vote per voxel (``votes=``): candidates are points with colour above ``H_MIN`` and
    points without colour above ``H_MIN_DARK``; points with colour vote with their frame's
    vegetation mask (the learned mask, or the rule in frames that fell back), points without
    colour do not vote."""
    locs = [geo[k][1] for k in frames]
    vegs = [geo[k][3] for k in frames]
    if not any(geo[k][4]["source"] == SOURCE_MODEL for k in frames):
        return segment.isolate_plant_multi(locs, vegs)
    cands, votes = [], []
    for k, loc, veg in zip(frames, locs, vegs, strict=True):
        has, h = clouds[k].has_rgb, loc[:, 2]
        cands.append((has & (h > segment.H_MIN)) | (~has & (h > segment.H_MIN_DARK)))
        votes.append(np.where(has, veg, np.nan))
    return segment.isolate_plant_multi(locs, cands, votes=votes)


def mask_summary(sources: pd.Series) -> dict:
    """Run columns from the used frames' ``mask_source``: ``mask_model_frac`` (share of frames with
    the learned mask) and ``mask_source`` ("model" all, "rule" none, "mixed" some)."""
    s = sources.astype(str)
    frac = float((s == SOURCE_MODEL).mean()) if len(s) else float("nan")
    label = SOURCE_RULE if not len(s) or frac == 0 else SOURCE_MODEL if frac == 1 else "mixed"
    return {"mask_source": label, "mask_model_frac": frac}


# ---------------------------------------------------------------- plant references
def plant_reference(plant_id: str, data_dir: Path, days: int = PRIOR_DAYS) -> tuple[PlantRef, list]:
    """Median accepted soil plane of the plant's first ``days`` dates (all hours)."""
    runs = list_runs(data_dir, plants=[plant_id])
    dates = sorted({r.date for r in runs})[:days]
    planes, centres, log = [], [], []
    for r in runs:
        if r.date not in dates:
            continue
        try:
            res = process_run(r, data_dir, None, reference_only=True)
        except Exception as e:  # noqa: BLE001
            log.append((r, f"error: {e}"))
            continue
        ref = res.get("ref")
        if ref is None:
            continue
        log.append((r, ref["fit"].reason))
        if ref["plane7"] is not None:
            planes.append(ref["plane7"])
            centres.append(ref["centre7"][:2])
    plane = ground.median_plane(planes) if len(planes) >= MIN_PRIOR_RUNS else None
    cxy = tuple(np.median(np.array(centres), 0)) if centres else CENTRE_DEFAULT_XY
    return PlantRef(plane, (float(cxy[0]), float(cxy[1])), len(planes)), log


# ---------------------------------------------------------------- refined depth (PromptDA)
DERIVED_REFINED = Path("data/derived/threed_refined")  # default --out with --depth-model
DERIVED_PLANTMASK = Path("data/derived/threed_plantmask")  # default --out with --plant-model
REFINED_DIR = "refined_depth"  # cache of refined maps on the ToF grid, one npz per frame


def refined_path(out_dir: Path, run: Run, k: int) -> Path:
    return Path(out_dir) / REFINED_DIR / run.plant_id / f"{run.date}_{run.hour}_{k:02d}.npz"


class RefinedDepth:
    """``depth_fn`` for :func:`process_run`: the cached refined map of frame k (Prompt Depth
    Anything, resampled onto the ToF grid by :func:`cloud.points_to_depth_grid`), merged with
    the ToF where the refined map has no sample (the ~27 % of the ToF view without RGB, and
    frames without poses). The ToF fallback gets the usual hole filling and 3 x 3 median.
    Frames without a cached map keep the ToF (``refined_frac`` = 0)."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)

    def __call__(self, run: Run, k: int, depth: np.ndarray):
        f = refined_path(self.out_dir, run, k)
        if not f.exists():
            return None
        ref = np.load(f)["depth"].astype(np.float32)
        base = median3(fill_small_holes(np.where(depth > 0, depth, 0).astype(np.float32))[0])
        m = ref > 0
        return np.where(m, ref, base).astype(np.float32), m


def plan_frames(run: Run, data_dir: str, prior: PlantRef | None) -> tuple[Run, list[int], bool]:
    """(run, frames process_run will build, no_pose) for the GPU pre-passes."""
    res = process_run(run, Path(data_dir), prior, plan_only=True)
    if res["status"] != "ok":
        return run, [], True
    return run, res["build"], res["no_pose"]


def precompute_refined(runs: list[Run], refs: dict, data_dir: Path, out_dir: Path, model: str, jobs: int = 8) -> dict:
    """GPU pre-pass: refine every frame :func:`process_run` will build and cache it on the ToF
    grid. Resumable (cached frames are skipped). Returns timing statistics."""
    import torch
    from joblib import Parallel, delayed

    from uiwue.ml.promptda.infer import DepthRefiner

    plans = Parallel(n_jobs=jobs)(delayed(plan_frames)(r, str(data_dir), refs.get(r.plant_id)) for r in runs)
    todo = [(r, k) for r, ks, _ in plans for k in ks if not refined_path(out_dir, r, k).exists()]
    stats = {"frames": 0, "missing": 0, "s_model": 0.0, "s_grid": 0.0, "s_total": 0.0}
    if not todo:
        return stats
    refiner = DepthRefiner(model)  # bf16 on CUDA
    print(f"refining {len(todo)} frames on {refiner.device} ({refiner.dtype})", flush=True)
    t_all = time.perf_counter()
    for i, (r, k) in enumerate(todo, 1):
        t0 = time.perf_counter()
        res = refiner.refine_frame(r, k, data_dir)
        if refiner.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if res is None:
            stats["missing"] += 1
            continue
        grid = points_to_depth_grid(res.points_in_depth_frame())
        t2 = time.perf_counter()
        f = refined_path(out_dir, r, k)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, depth=grid, frame=np.int64(k), model=str(model))
        if i > 1:  # the first call includes CUDA warm-up
            stats["s_model"] += t1 - t0
            stats["s_grid"] += t2 - t1
            stats["frames"] += 1
        if i % 200 == 0:
            print(f"  refined {i}/{len(todo)}", flush=True)
    stats["s_total"] = time.perf_counter() - t_all
    return stats


# ---------------------------------------------------------------- diagnostics
def _norm_rgb(c):
    img = c.image(c.rgb, 0).astype(float)
    hi = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
    return np.clip(img * 255 / hi, 0, 255).astype(np.uint8)


def render_diagnostics(run: Run, data_dir: Path, prior, out_dir: Path, depth_fn=None, plant_mask_fn=None) -> Path:
    """Six-panel verification figure of one run (frame 7 colour, height above reference,
    isolation, side view, plant across frames, reference-plane residuals)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pid, d, h = run.plant_id, run.date, run.hour
    res = process_run(run, data_dir, prior, debug=True, depth_fn=depth_fn, plant_mask_fn=plant_mask_fn)
    g = res["debug"]
    r = res["run"]
    c7 = g["clouds"][7]
    loc = g["loc"][7]
    iso = g["iso"][7]
    veg = g["veg"][7]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    # (a) colour on depth grid + centre/cylinder
    ax[0, 0].imshow(_norm_rgb(c7))
    th = np.linspace(0, 2 * np.pi, 100)
    e1, e2 = g["plane7"].basis()
    cc = g["c7"]
    for rad, col in [(segment.CYLINDER_RADIUS, "w"), (0.05, "c")]:
        ring = cc + rad * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
        u = 213.08 * ring[:, 0] / ring[:, 2] + 106.3
        v = 213.48 * ring[:, 1] / ring[:, 2] + 87.5
        for a_ in (ax[0, 0], ax[0, 1], ax[0, 2]):
            a_.plot(u, v, col, lw=1)
    ax[0, 0].set_title(
        f"{pid} {d} {h}h frame 7: RGB sampled onto ToF points\nwhite: cylinder r=8.5 cm, cyan: soil window r=5 cm"
    )
    # (b) height map
    hm = c7.image(loc[:, 2] * 1000)
    im = ax[0, 1].imshow(hm, cmap="turbo", vmin=-60, vmax=max(60, np.nanpercentile(hm, 99)))
    plt.colorbar(im, ax=ax[0, 1], fraction=0.04, label="height above reference (mm)")
    ax[0, 1].set_title(
        f"height above reference ({r['ref_source']}, {r['ref_reject_reason']}, inliers {r['ref_inliers']}, tilt {r['ref_tilt_deg']:.1f})"
    )
    # (c) isolated plant
    lab = np.zeros(len(loc))
    lab[veg] = 1
    lab[iso.mask] = 2
    ax[0, 2].imshow(
        c7.image(lab, -1),
        cmap=matplotlib.colors.ListedColormap(["white", "lightgray", "orange", "green"]),
        vmin=-1.5,
        vmax=2.5,
        interpolation="nearest",
    )
    ax[0, 2].set_title(
        f"frame 7 isolation: green plant ({iso.mask.sum()} pts), orange other veg\nclipped {iso.clipped_fraction:.3f}, centroid off {1000 * iso.centroid_offset:.1f} mm"
    )
    # (d) side view
    sel = np.hypot(loc[:, 0], loc[:, 1]) < 0.15
    ax[1, 0].scatter(loc[sel, 0] * 1000, loc[sel, 2] * 1000, s=0.5, c=np.where(iso.mask[sel], "g", "0.6"))
    ax[1, 0].axhline(0, color="k")
    ax[1, 0].axvline(-85, ls="--")
    ax[1, 0].axvline(85, ls="--")
    ax[1, 0].set_xlabel("a (mm, in-plane)")
    ax[1, 0].set_ylabel("height (mm)")
    ax[1, 0].set_title("side view, frame 7, |r|<15 cm (green = plant)")
    # (e) plant position across frames in the frame-7 / plant-centred top view
    cols = plt.cm.tab10(np.arange(10))
    for i, k in enumerate(sorted(g["iso"])):
        m = g["iso"][k].mask
        lk = g["loc"][k]
        ax[1, 1].scatter(
            lk[m, 0] * 1000,
            lk[m, 1] * 1000,
            s=0.3,
            color=cols[i % 10],
            label=f"f{k}{'*' if k in g['used'] else ''} n={m.sum()}",
        )
    ax[1, 1].add_patch(plt.Circle((0, 0), 85, fill=False))
    ax[1, 1].set_aspect("equal")
    ax[1, 1].legend(markerscale=10, fontsize=7)
    ax[1, 1].set_title(f"isolated plant per frame, plant-centred top view (mm)\nframes used: {r['frames_used']}")
    # (f) reference plane residuals of pooled soil candidates
    pts, _nv = g["pool"]
    pl = g["plane7"]
    res_ = pl.height(pts) * 1000
    rr = np.hypot(pts[:, 0] - cc[0], pts[:, 1] - cc[1]) < 0.05
    sc = ax[1, 2].scatter(
        (pts[rr, 0] - cc[0]) * 1000,
        (pts[rr, 1] - cc[1]) * 1000,
        s=1,
        c=np.clip(res_[rr], -20, 60),
        cmap="turbo",
    )
    plt.colorbar(sc, ax=ax[1, 2], fraction=0.04, label="height above plane (mm)")
    ax[1, 2].set_aspect("equal")
    ax[1, 2].set_title(f"pooled points within 5 cm (all used frames)\n|h|<5mm: {np.mean(np.abs(res_[rr]) < 5):.2f}")
    txt = (
        f"p50/p75/p95 {r['height_p50']:.0f}/{r['height_p75']:.0f}/{r['height_p95']:.0f} mm (mad p75 {r['height_p75_mad']:.1f}); "
        f"area {r['projected_area']:.0f} cm2; leaf {r['leaf_area_est']:.0f}; radius {r['canopy_radius']:.0f} mm; incl {r['inclination_mean']:.0f}; "
        f"n {r['n_points']:.0f}; frames ok {r['n_frames_ok']}; bright {r['brightness']:.0f}; qc_pass {r['qc_pass']} "
        + " ".join(f for f in r if f.startswith("qc_") and r[f] and f != "qc_pass")
    )
    fig.suptitle(txt, fontsize=9)
    plt.tight_layout()
    f = Path(out_dir) / f"run_{pid}_{d}_{h}.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f, dpi=65)
    plt.close(fig)
    return f


# ---------------------------------------------------------------- batch driver
def _worker(
    run: Run,
    data_dir: str,
    prior: PlantRef | None,
    out_dir: str,
    want_cloud: bool,
    refined: bool = False,
    plant_masks: bool = False,
):
    t0 = time.perf_counter()
    try:
        depth_fn = RefinedDepth(Path(out_dir)) if refined else None
        mask_fn = PlantMasks(Path(out_dir)) if plant_masks else None
        res = process_run(run, Path(data_dir), prior, want_cloud=want_cloud, depth_fn=depth_fn, plant_mask_fn=mask_fn)
    except Exception as e:  # noqa: BLE001
        res = {
            "status": "error",
            "error": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)!r}",
            "frames": [],
            "run": None,
            "cloud": None,
        }
    if res.get("cloud") is not None:
        d = schema.folder_date(run.date)
        p = Path(out_dir) / schema.cloud_path(run.bed, run.plant, d, int(run.hour)).relative_to(schema.DERIVED)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **res["cloud"])
    res["cloud"] = None
    res["seconds"] = time.perf_counter() - t0
    res["key"] = run
    res.pop("ref", None)
    return res


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class Store:
    """Accumulates results and checkpoints the parquet files (merged with what is on disk)."""

    def __init__(self, out: Path):
        self.out = out
        self.runs_path = out / schema.TRAITS_RUNS.name
        self.frames_path = out / schema.TRAITS_FRAMES.name
        self.log_path = out / schema.RUN_LOG.name
        self.old_runs = pd.read_parquet(self.runs_path) if self.runs_path.exists() else None
        self.old_frames = pd.read_parquet(self.frames_path) if self.frames_path.exists() else None
        self.new_runs: list[dict] = []
        self.new_frames: list[dict] = []
        self.log: list[dict] = []

    def done_keys(self) -> set[tuple]:
        if self.old_runs is None:
            return set()
        return {
            (b, p, d.strftime("%Y_%m_%d"), f"{int(h):02d}")
            for b, p, d, h in self.old_runs[schema.KEYS].itertuples(index=False)
        }

    def add(self, res: dict) -> None:
        r: Run = res["key"]
        if res.get("run") is not None:
            self.new_runs.append(res["run"])
            self.new_frames.extend(res["frames"])
        self.log.append(
            dict(
                bed=r.bed,
                plant=r.plant,
                date=r.date,
                hour=r.hour,
                status=res["status"],
                error=res.get("error", "")[:2000],
                n_frames_used=(res["run"] or {}).get("n_frames_used", 0),
                seconds=round(res.get("seconds", np.nan), 3),
                finished_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    def flush(self) -> None:
        if self.new_runs:
            new = qc.runs_table(self.new_runs)
            df = new if self.old_runs is None else pd.concat([self.old_runs, new], ignore_index=True)
            df = df.drop_duplicates(schema.KEYS, keep="last")
            df = df.sort_values(schema.KEYS).reset_index(drop=True)
            schema.validate_runs(df)
            _write_parquet(df, self.runs_path)
            self.old_runs, self.new_runs = df, []
        if self.new_frames:
            new = pd.DataFrame(self.new_frames)
            df = new if self.old_frames is None else pd.concat([self.old_frames, new], ignore_index=True)
            df = df.drop_duplicates(schema.KEYS + ["frame"], keep="last")
            _write_parquet(df.sort_values(schema.KEYS + ["frame"]).reset_index(drop=True), self.frames_path)
            self.old_frames, self.new_frames = df, []
        if self.log:
            pd.DataFrame(self.log).to_csv(
                self.log_path,
                mode="a",
                header=not self.log_path.exists(),
                index=False,
                lineterminator="\n",
            )
            self.log = []


def _parse_date(s: str) -> str:
    """'2025-09-01' or '2025_09_01' -> folder name '2025_09_01'."""
    return s.replace("-", "_")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument(
        "--out",
        help=f"output directory (default {schema.DERIVED}; {DERIVED_REFINED} with --depth-model, "
        f"{DERIVED_PLANTMASK} with --plant-model)",
    )
    p.add_argument("--plants", nargs="+", help="full IDs such as W1_J5 (default: all on disk)")
    p.add_argument("--dates", nargs=2, metavar=("START", "END"), help="inclusive, YYYY-MM-DD")
    p.add_argument("--hours", nargs="+", help="folder hours, e.g. 06 09 15 (default: all)")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--parity", action="store_true", help="paper-parity heights only (E0a)")
    p.add_argument(
        "--depth-model",
        metavar="PATH",
        help="Prompt Depth Anything checkpoint (e.g. runs/promptda-vitl/best): refine the depth "
        "of every built frame on the GPU before the traits; outputs go to a separate --out",
    )
    p.add_argument(
        "--plant-model",
        metavar="PATH",
        help="Mask2Former plant checkpoint (e.g. runs/leafseg-m2f/best): segment every built frame "
        "on the GPU and use the centre-plant mask instead of the ExG rule (the rule stays on dark "
        "frames and where no plant is found); outputs go to a separate --out",
    )
    p.add_argument("--no-clouds", action="store_true")
    p.add_argument("--flush-every", type=int, default=300, help="checkpoint every N runs")
    p.add_argument("--limit", type=int, help="process at most N runs (smoke tests)")
    p.add_argument(
        "--diagnose",
        nargs="+",
        metavar="PLANT:DATE:HOUR",
        help="write diagnostics PNGs for these runs (e.g. W1_J5:2025_09_03:15) and exit",
    )
    a = p.parse_args(argv)

    data_dir = Path(a.data_dir)
    if a.depth_model and a.plant_model and not a.out:
        p.error("--depth-model with --plant-model needs an explicit --out")
    default = DERIVED_PLANTMASK if a.plant_model else DERIVED_REFINED if a.depth_model else schema.DERIVED
    out = Path(a.out) if a.out else default
    contract = {schema.DERIVED.resolve(), (data_dir / "derived" / "threed").resolve()}
    for flag, used in (("--depth-model", a.depth_model), ("--plant-model", a.plant_model)):
        if used and out.resolve() in contract:
            p.error(f"{flag} must not write into {schema.DERIVED} (the ExG / raw-ToF contract files)")
    out.mkdir(parents=True, exist_ok=True)
    dates = (_parse_date(a.dates[0]), _parse_date(a.dates[1])) if a.dates else None
    hours = [f"{int(h):02d}" for h in a.hours] if a.hours else None

    if a.diagnose:
        refs = load_plant_refs(out / "plant_refs.csv")
        druns = []
        for spec in a.diagnose:
            pid, d, h = spec.split(":")
            if pid not in refs:
                refs[pid] = plant_reference(pid, data_dir)[0]
            druns.append(Run(pid, _parse_date(d), h))
        depth_fn = mask_fn = None
        if a.depth_model:
            precompute_refined(druns, refs, data_dir, out, a.depth_model, jobs=1)
            depth_fn = RefinedDepth(out)
        if a.plant_model:
            precompute_masks(druns, refs, data_dir, out, a.plant_model, jobs=1)
            mask_fn = PlantMasks(out)
        for r in druns:
            f = render_diagnostics(r, data_dir, refs[r.plant_id], out / "diagnostics", depth_fn, mask_fn)
            print(f, flush=True)
        return 0

    if a.parity:
        from uiwue.threed import parity

        return parity.main_from_run(data_dir, out, a.plants, dates, hours, a.jobs)

    runs = list_runs(data_dir, "depth", plants=a.plants, dates=dates, hours=hours)
    store = Store(out)
    done = store.done_keys()
    todo = [r for r in runs if (r.bed, r.plant, r.date, r.hour) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(
        f"{len(runs)} runs selected, {len(runs) - len(todo)} already done, {len(todo)} to do",
        flush=True,
    )
    if not todo:
        return 0

    from joblib import Parallel, delayed

    # carried references, one per plant (cached in plant_refs.csv)
    refs_path = out / "plant_refs.csv"
    refs = load_plant_refs(refs_path)
    missing = sorted({r.plant_id for r in todo} - set(refs))
    if missing:
        t0 = time.perf_counter()
        res = Parallel(n_jobs=min(a.jobs, len(missing)))(delayed(plant_reference)(pid, data_dir) for pid in missing)
        for pid, (ref, _log) in zip(missing, res, strict=True):
            refs[pid] = ref
            z = ref.plane7.z_at(*ref.centre_xy) if ref.plane7 is not None else float("nan")
            print(f"  ref {pid}: {ref.n_runs} accepted runs, soil z {z:.4f} m", flush=True)
        save_plant_refs(refs, refs_path)
        print(f"plant references: {time.perf_counter() - t0:.1f} s", flush=True)

    if a.depth_model:
        st = precompute_refined(todo, refs, data_dir, out, a.depth_model, jobs=a.jobs)
        if st["frames"]:
            print(
                f"refined depth: {st['frames']} frames, model {st['s_model'] / st['frames']:.3f} s"
                f"/frame (incl. image I/O), resampling {st['s_grid'] / st['frames']:.3f} s/frame, "
                f"{st['missing']} frames without data, total {st['s_total']:.0f} s",
                flush=True,
            )
    if a.plant_model:
        st = precompute_masks(todo, refs, data_dir, out, a.plant_model, jobs=a.jobs)
        if st["frames"]:
            print(
                f"plant masks: {st['frames']} frames, model {st['s_model'] / st['frames']:.3f} s/frame, "
                f"{st['empty']} frames without a plant instance, {st['missing']} without RGB, "
                f"total {st['s_total']:.0f} s",
                flush=True,
            )

    t0 = time.perf_counter()
    gen = Parallel(n_jobs=a.jobs, return_as="generator_unordered", batch_size=1)(
        delayed(_worker)(
            r,
            str(data_dir),
            refs.get(r.plant_id),
            str(out),
            not a.no_clouds,
            bool(a.depth_model),
            bool(a.plant_model),
        )
        for r in todo
    )
    n_ok = n_err = 0
    for i, res in enumerate(gen, 1):
        store.add(res)
        if res["status"] == "ok":
            n_ok += 1
        else:
            n_err += 1
        if i % a.flush_every == 0 or i == len(todo):
            store.flush()
            el = time.perf_counter() - t0
            print(
                f"[{i}/{len(todo)}] ok={n_ok} other={n_err} {el:.0f} s ({el / i:.2f} s/run wall)",
                flush=True,
            )
    store.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
