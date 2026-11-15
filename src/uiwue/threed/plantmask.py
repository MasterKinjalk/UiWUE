"""Learned plant mask for the 3D pipeline: the Mask2Former centre-plant instance on the ToF points.

``run --plant-model PATH`` does two passes:

1. :func:`precompute_masks` (GPU, resumable): every frame :func:`run.process_run` will build is
   segmented with :class:`uiwue.ml.leafseg.infer.LeafSegmenter` on the raw, distorted RGB frame.
   All plant instances are cached at half RGB resolution in
   ``<out>/plant_masks/<plant>/<date>_<HH>_<k>.npz`` (bit-packed, with their scores). Frames
   where the model finds nothing get an empty file, so the cache records that too.
2. :class:`PlantMasks` is the ``plant_mask_fn`` of :func:`run.process_run` (CPU workers). Per
   frame it lifts the cached instances onto the points through ``Cloud.rgb_uv``, picks the
   instance with the most points within ``ATTACH_RADIUS`` of the plant centre
   (:func:`select_centre`) and merges in the fragments of a split plant (:func:`merge_fragments`).
   It falls back to the ExG / height rule (returns no mask) on dark frames, frames without
   colour, frames without a cache file and frames without an instance at the centre, and says
   why in ``info["source"]``.

Only points with colour can be read from the mask. Points without colour (outside the RGB view
or hidden from it by the z-buffer) keep the height rule, ``h > H_MIN_DARK``
(``segment.vegetation_mask``). The isolation then pools the used frames on voxels, lets the
frames vote per voxel, and runs the DBSCAN + cylinder isolation on the result
(``segment.isolate_plant_multi(votes=...)``, ``run._isolate``; docs/threed/PIPELINE.md).
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from uiwue.threed import calib, segment
from uiwue.threed.cloud import Cloud
from uiwue.threed.io import Run, load_rgb

MASK_DIR = "plant_masks"  # cache of plant instances, one npz per frame
MASK_SCALE = 0.5  # cached masks are at half RGB resolution; the model's mask logits are at 1/8
MASK_SIZE = (calib.RGB_H // 2, calib.RGB_W // 2)  # (540, 960)
MODEL_SCALE = 0.5  # frames go into the model at half resolution (the scale it was evaluated at)
MIN_CENTRE_POINTS = 10  # an instance needs this many points near the centre to be the plant
MERGE_INSIDE = 0.5  # another instance with this share of its points in the cylinder is merged in
# A frame is ambiguous when an instance that is not merged has at least this share of the chosen
# instance's points near the centre (two plants' leaves over the pot centre): it keeps the rule.
AMBIGUOUS = 0.8

# info["source"] values (frame column ``mask_source``)
SOURCE_RULE = "rule"  # no model requested
SOURCE_MODEL = "model"
SOURCE_DARK = "rule_dark"  # dark frame: the model finds no plant there
SOURCE_NO_COLOUR = "rule_no_colour"  # frame without colour (no pose)
SOURCE_NO_CACHE = "rule_no_cache"  # frame not in the mask cache
SOURCE_NO_INSTANCE = "rule_no_instance"  # no plant instance at the plant centre
SOURCE_AMBIGUOUS = "rule_ambiguous"  # two instances reach the centre about equally


def mask_path(out_dir: Path, run: Run, k: int) -> Path:
    return Path(out_dir) / MASK_DIR / run.plant_id / f"{run.date}_{run.hour}_{k:02d}.npz"


def save_masks(path: Path, masks: np.ndarray, scores: np.ndarray, model: str = "") -> None:
    """(n, H, W) bool masks -> bit-packed npz (a 540 x 960 instance takes about 2 kB)."""
    masks = np.asarray(masks, bool)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        bits=np.packbits(masks, axis=-1),
        shape=np.array(masks.shape, np.int64),
        scores=np.asarray(scores, np.float32),
        model=str(model),
    )
    tmp.replace(path)


def load_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """npz from :func:`save_masks` -> (masks (n, H, W) bool, scores (n,))."""
    with np.load(path) as z:
        shape = tuple(int(s) for s in z["shape"])
        masks = np.unpackbits(z["bits"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)
        return masks, z["scores"].astype(np.float32)


def lift(cloud: Cloud, masks: np.ndarray, scale: float = MASK_SCALE) -> np.ndarray:
    """(n, H, W) image masks -> (n, N) instance membership of the cloud's points.

    Each point with colour reads the mask at its full-resolution distorted RGB pixel
    (``Cloud.rgb_uv``), converted to the mask grid with the pixel-centre convention
    (``(u + 0.5) * scale - 0.5``, nearest pixel). Points without colour are False.
    """
    n = len(masks)
    out = np.zeros((n, len(cloud)), bool)
    if n == 0 or cloud.rgb_uv is None or not cloud.has_rgb.any():
        return out
    h, w = masks.shape[1:]
    u = np.rint((cloud.rgb_uv[:, 0] + 0.5) * scale - 0.5).astype(np.int64)
    v = np.rint((cloud.rgb_uv[:, 1] + 0.5) * scale - 0.5).astype(np.int64)
    ok = cloud.has_rgb & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out[:, ok] = masks[:, v[ok], u[ok]]
    return out


def centre_counts(
    members: np.ndarray, local: np.ndarray, radius: float = segment.ATTACH_RADIUS, h_min: float = segment.H_MIN
) -> np.ndarray:
    """Points of each instance within ``radius`` of the plant axis and ``h > h_min`` above the
    reference (``local``: plane-local coordinates, plant centre at a = b = 0)."""
    near = (np.hypot(local[:, 0], local[:, 1]) < radius) & (local[:, 2] > h_min)
    return (members & near).sum(1)


def select_centre(
    members: np.ndarray,
    local: np.ndarray,
    radius: float = segment.ATTACH_RADIUS,
    h_min: float = segment.H_MIN,
    min_points: int = MIN_CENTRE_POINTS,
) -> tuple[int, int]:
    """(index, n) of the instance with the most points within ``radius`` of the plant axis and
    ``h > h_min`` above the reference; (-1, 0) when none has ``min_points``.

    This is the DBSCAN attach rule (a cluster is the plant's if it reaches within 3 cm of the
    centre), applied to the model's instances.
    """
    if len(members) == 0:
        return -1, 0
    counts = centre_counts(members, local, radius, h_min)
    i = int(np.argmax(counts))
    return (i, int(counts[i])) if counts[i] >= min_points else (-1, 0)


def merge_fragments(
    members: np.ndarray,
    local: np.ndarray,
    centre: int,
    min_inside: float = MERGE_INSIDE,
    h_min: float = segment.H_MIN,
    min_points: int = MIN_CENTRE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """(mask, taken): the centre instance plus every other instance with at least ``min_inside``
    of its points (``h > h_min``) inside the plant cylinder; ``taken`` marks the instances used.

    Mask2Former splits large plants into several instances (W1 from about 2025-09-04). A
    neighbour's instance has most of its points around its own pot, 17 cm away, so its share
    inside this plant's cylinder stays low even where its leaves cover the pot.
    """
    r = np.hypot(local[:, 0], local[:, 1])
    up = members & (local[:, 2] > h_min)
    total = up.sum(1)
    inside = (up & (r <= segment.CYLINDER_RADIUS)).sum(1)
    take = (total >= min_points) & (inside >= min_inside * np.maximum(total, 1))
    take[centre] = True
    return members[take].any(0), take


class PlantMasks:
    """``plant_mask_fn`` for :func:`run.process_run`, reading the cache of :func:`precompute_masks`.

    Called as ``fn(run, k, cloud, local)``; returns ``(mask, info)``. ``mask`` is a per-point bool
    array (the centre plant on points with colour, False elsewhere) or None (use the ExG / height
    rule). ``info`` has ``source`` (see the ``SOURCE_*`` constants), ``n_instances``, ``score``
    (of the chosen instance) and ``n_merged`` (fragments merged in).
    """

    def __init__(self, out_dir: Path, dark_threshold: float = segment.DARK_BRIGHTNESS):
        self.out_dir = Path(out_dir)
        self.dark_threshold = dark_threshold
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _load(self, run: Run, k: int):
        key = (run.plant_id, run.date, run.hour, k)
        if key not in self._cache:
            f = mask_path(self.out_dir, run, k)
            if not f.exists():
                return None
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = load_masks(f)
        return self._cache[key]

    def __call__(self, run: Run, k: int, cloud: Cloud, local: np.ndarray):
        info = {"source": SOURCE_MODEL, "n_instances": 0, "score": float("nan"), "n_merged": 0}
        if cloud.rgb_uv is None or not cloud.has_rgb.any():
            return None, info | {"source": SOURCE_NO_COLOUR}
        if segment.is_dark(cloud.brightness, self.dark_threshold):
            return None, info | {"source": SOURCE_DARK}
        cached = self._load(run, k)
        if cached is None:
            return None, info | {"source": SOURCE_NO_CACHE}
        masks, scores = cached
        info["n_instances"] = len(masks)
        members = lift(cloud, masks)
        i, n = select_centre(members, local)
        if i < 0:
            return None, info | {"source": SOURCE_NO_INSTANCE}
        mask, taken = merge_fragments(members, local, i)
        info |= {"score": float(scores[i]), "n_merged": int(taken.sum() - 1)}
        rivals = centre_counts(members[~taken], local)
        if len(rivals) and rivals.max() >= AMBIGUOUS * n:
            return None, info | {"source": SOURCE_AMBIGUOUS}
        return mask, info


def _bounded_prefetch(items, load, depth: int = 8):
    """Yield ``(item, load(item))`` in order, loading at most ``depth`` items ahead in threads."""
    with ThreadPoolExecutor(max_workers=min(depth, 4)) as ex:
        q: deque = deque()
        it = iter(items)
        for item in it:
            q.append((item, ex.submit(load, item)))
            if len(q) >= depth:
                break
        while q:
            item, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(load, nxt)))
            yield item, fut.result()


def precompute_masks(
    runs: list[Run],
    refs: dict,
    data_dir: Path,
    out_dir: Path,
    model: str,
    jobs: int = 8,
) -> dict:
    """GPU pre-pass: segment every frame :func:`run.process_run` will build (runs with poses
    only; without poses the cloud has no colour) and cache the plant instances. Resumable (cached
    frames are skipped). Returns counts and timing statistics."""
    import torch
    from joblib import Parallel, delayed

    from uiwue.ml.leafseg.infer import LeafSegmenter, instance_masks
    from uiwue.threed.run import plan_frames

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    plans = Parallel(n_jobs=jobs)(delayed(plan_frames)(r, str(data_dir), refs.get(r.plant_id)) for r in runs)
    todo = [(r, k) for r, ks, no_pose in plans if not no_pose for k in ks if not mask_path(out_dir, r, k).exists()]
    stats = {"frames": 0, "missing": 0, "empty": 0, "s_model": 0.0, "s_total": 0.0}
    if not todo:
        return stats
    seg = LeafSegmenter(model)
    print(f"segmenting {len(todo)} frames on {seg.device}", flush=True)

    def load(rk):
        r, k = rk
        return load_rgb(r.dir(data_dir, "rgb") / f"rgb_{k:03d}.jpg")

    t_all = time.perf_counter()
    for i, ((r, k), rgb) in enumerate(_bounded_prefetch(todo, load), 1):
        if rgb is None:
            stats["missing"] += 1
            continue
        t0 = time.perf_counter()
        out = seg.forward(rgb, MODEL_SCALE, undistort=False)
        masks, scores, _ = instance_masks(out, MASK_SIZE, seg.threshold, label=0)
        if seg.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        save_masks(mask_path(out_dir, r, k), masks, scores, model)
        stats["empty"] += int(len(masks) == 0)
        if i > 1:  # the first call includes CUDA warm-up
            stats["s_model"] += t1 - t0
            stats["frames"] += 1
        if i % 500 == 0:
            el = time.perf_counter() - t_all
            print(f"  segmented {i}/{len(todo)} ({el / i:.3f} s/frame wall)", flush=True)
    stats["s_total"] = time.perf_counter() - t_all
    return stats
