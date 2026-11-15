"""Download the ADA 2025 camera archives in parallel, then extract them.

The ADA download script fetches one archive at a time, and the data server caps each connection
at about 3-5 MB/s. This script splits the archives still missing into byte-balanced, disjoint
chunks and runs one download process per chunk (18 streams reached about 88 MB/s), then extracts
each modality once all its chunks are done. Archives already on disk with the right size are
skipped, and no two processes ever write the same file.

    python scripts/download_parallel.py --data-dir data --streams 18
    python scripts/download_parallel.py --modalities thermal depth --streams 8

Needs the ADA repository at external/ada2025 (scripts/download_data.sh clones it). Logs go to
logs/dl_<modality>_<k>.log and logs/unzip_<modality>.log.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADA_UTILS = ROOT / "external" / "ada2025" / "utils"
MODALITIES = ("thermal", "depth", "hsi", "rgb")


def _ada_downloader():
    spec = importlib.util.spec_from_file_location("ada_download", ADA_UTILS / "download_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def missing_archives(dl, data_dir: Path, modalities) -> dict[str, list[tuple[str, int]]]:
    """(plant ID, size in bytes) of every camera archive not yet on disk with the right size."""
    out = {}
    for m in modalities:
        files, _ = dl.list_files(dl.SUBDATASETS[m]["doi"])
        todo = []
        for f in files:
            name, size = f["dataFile"]["filename"], f["dataFile"]["filesize"]
            plant = dl.plant_of(name)
            p = data_dir / m / name
            if plant and not (p.exists() and p.stat().st_size == size):
                todo.append((plant, size))
        out[m] = todo
    return out


def split_chunks(todo: list[tuple[str, int]], n: int) -> list[list[str]]:
    """Greedy byte balancing: largest archive first, into the lightest chunk."""
    chunks: list[list[str]] = [[] for _ in range(n)]
    load = [0] * n
    for plant, size in sorted(todo, key=lambda x: -x[1]):
        k = load.index(min(load))
        chunks[k].append(plant)
        load[k] += size
    return [c for c in chunks if c]


def run_modality(m: str, chunks: list[list[str]], data_dir: Path, logs: Path) -> str:
    py = sys.executable
    procs = []
    for k, plants in enumerate(chunks):
        log = open(logs / f"dl_{m}_{k}.log", "w")  # noqa: SIM115 - closed below
        cmd = [py, str(ADA_UTILS / "download_dataset.py"), "--modalities", m, "--plants", *plants,
               "--data-dir", str(data_dir), "-y"]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    failed = 0
    for p, log in procs:
        failed += p.wait() != 0
        log.close()
    with open(logs / f"unzip_{m}.log", "w") as log:
        rc = subprocess.call([py, str(ADA_UTILS / "unzip_data.py"), "--data-dir", str(data_dir), "--modalities", m,
                              "--beds", "W1", "W2", "W3", "--plants", "all", "-y"], stdout=log, stderr=subprocess.STDOUT)
    return f"{m}: {len(chunks)} streams, {failed} failed; extraction {'ok' if rc == 0 else 'FAILED'}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    ap.add_argument("--streams", type=int, default=18, help="parallel downloads in total")
    ap.add_argument("--plan-only", action="store_true", help="print the chunks and exit")
    a = ap.parse_args(argv)

    dl = _ada_downloader()
    todo = missing_archives(dl, a.data_dir, a.modalities)
    total = sum(s for t in todo.values() for _, s in t)
    if total == 0:
        print("every archive is on disk")
        return 0
    plan = {}
    for m, t in todo.items():
        if t:
            n = max(1, min(len(t), math.ceil(a.streams * sum(s for _, s in t) / total)))
            plan[m] = split_chunks(t, n)
            print(f"{m}: {len(t)} archives, {sum(s for _, s in t) / 1e9:.0f} GB, {len(plan[m])} streams")
    print(f"total {total / 1e9:.0f} GB")
    if a.plan_only:
        return 0
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with ThreadPoolExecutor(len(plan)) as ex:
        for line in ex.map(lambda m: run_modality(m, plan[m], a.data_dir, logs), plan):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
