"""Stage 1: frozen embeddings for every run, one table per sensor and variant.

    python -m uiwue.fusion.embed --data-dir data [--sensors rgb thermal depth hsi]
                                 [--hsi-windows] [--device cuda] [--batch 32] [--workers 4]

Writes cache/fusion/emb_<sensor>_<variant>.npz (keys + float16 embeddings) and
a manifest. Existing tables are skipped unless --overwrite. Each run's frames
are embedded and averaged (thermal: all 5; RGB and depth: the centre frame).
Variants are what Part B swaps in at evaluation: low-resolution thermal, and
hyperspectral with spatial detail removed, only 3 bands, or one 20 nm window masked.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uiwue import core
from uiwue.fusion import inputs
from uiwue.fusion.encoders import Encoder, build_encoder

CACHE = core.REPO / "cache" / "fusion"
KEY_COLS = ["bed", "plant", "date", "hour"]


def table_path(cache: Path, sensor: str, variant: str) -> Path:
    return Path(cache) / f"emb_{sensor}_{variant}.npz"


def save_table(path: Path, keys: pd.DataFrame, emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    k = core.normalize_keys(keys[KEY_COLS])
    np.savez_compressed(
        path,
        bed=k["bed"].to_numpy(str),
        plant=k["plant"].to_numpy(str),
        date=k["date"].dt.strftime("%Y-%m-%d").to_numpy(str),
        hour=k["hour"].to_numpy(int),
        emb=emb.astype(np.float16),
    )


def load_table(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    z = np.load(path)
    keys = core.normalize_keys(pd.DataFrame({c: z[c] for c in KEY_COLS}))
    return keys, z["emb"].astype(np.float32)


class _Runs(torch.utils.data.Dataset):
    """Loads one run's frames for every requested variant (the cube is read once)."""

    def __init__(self, runs: pd.DataFrame, sensor: str, variants: list[str]):
        self.runs, self.sensor, self.variants = runs.reset_index(drop=True), sensor, variants

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, i):
        path = self.runs.loc[i, "path"]
        out = {}
        try:
            if self.sensor == "hsi":
                cube = inputs.read_hsi(path)
                for v in self.variants:
                    out[v] = inputs.load_hsi(path, v, cube=cube)
            else:
                for v in self.variants:
                    out[v] = inputs.load(self.sensor, path, v)
        except Exception as e:  # noqa: BLE001  (one unreadable run must not stop an overnight job; reported)
            return i, None, f"{path}: {e}"
        return i, {v: (torch.from_numpy(s.frames), s.wavelengths) for v, s in out.items()}, None


def _collate(items):
    return items  # runs differ in frame count; batched per variant below


@torch.no_grad()
def embed_sensor(
    encoder: Encoder, runs: pd.DataFrame, variants: list[str], device: str, batch: int = 32, workers: int = 0
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Embeddings per variant (N, dim) for `runs`; returns (embeddings, ok mask, errors)."""
    encoder.eval().to(device)
    n = len(runs)
    out = {v: np.full((n, encoder.dim), np.nan, np.float32) for v in variants}
    ok, errors = np.zeros(n, bool), []
    loader = torch.utils.data.DataLoader(
        _Runs(runs, encoder.sensor, variants), batch_size=batch, num_workers=workers, collate_fn=_collate
    )
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    for items in loader:
        good = [(i, d) for i, d, err in items if d is not None]
        errors += [err for _, d, err in items if d is None]
        for v in variants:
            # group runs with the same frame count and bands so they stack
            groups: dict[tuple, list] = {}
            for i, d in good:
                x, wl = d[v]
                groups.setdefault((x.shape, wl), []).append((i, x))
            for (_, wl), members in groups.items():
                idx = [i for i, _ in members]
                x = torch.stack([m for _, m in members]).to(device)
                with amp:
                    e = encoder.encode_frames(x, wl).float().cpu().numpy()
                out[v][idx] = e
        ok[[i for i, _ in good]] = True
    return out, ok, errors


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=core.DATA)
    ap.add_argument("--sensors", nargs="*", default=list(inputs.SENSORS))
    ap.add_argument("--plants", nargs="*", help="e.g. W1_A2 W2_J5 (default: all on disk)")
    ap.add_argument("--hsi-windows", action="store_true", help="also the 20 nm masked-window HSI variants")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    manifest_path = a.cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for sensor in a.sensors:
        variants = list(inputs.VARIANTS[sensor]) + (inputs.hsi_windows() if sensor == "hsi" and a.hsi_windows else [])
        todo = [v for v in variants if a.overwrite or not table_path(a.cache, sensor, v).exists()]
        if not todo:
            print(f"{sensor}: all variants cached")
            continue
        runs = inputs.list_runs(a.data_dir, sensor, a.plants)
        if runs.empty:
            print(f"{sensor}: no runs under {a.data_dir / sensor}")
            continue
        t0 = time.time()
        enc = build_encoder(sensor)
        emb, ok, errors = embed_sensor(enc, runs, todo, a.device, a.batch, a.workers)
        for v in todo:
            save_table(table_path(a.cache, sensor, v), runs[ok], emb[v][ok])
        manifest[sensor] = dict(
            encoder=type(enc).__name__,
            dim=enc.dim,
            runs=int(ok.sum()),
            failed=len(errors),
            variants=variants,
            seconds=round(time.time() - t0, 1),
        )
        print(f"{sensor}: {ok.sum()} runs x {len(todo)} variants in {time.time() - t0:.0f} s, {len(errors)} failed")
        for e in errors[:10]:
            print("   ", e)
        del enc
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    a.cache.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
