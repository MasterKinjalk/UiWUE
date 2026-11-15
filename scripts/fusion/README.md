# Fusion run scripts

Wrappers and helpers used for the fusion runs in `docs/fusion.md` (sections 7 and 8). Every
step is also a plain `python -m uiwue.fusion.<module>` command, listed in the "Commands and
timings" tables there; these scripts only chain them and move data to and from the GPU machine.

Settings come from environment variables (`common.sh`): `GPU_HOST` (user@host, needed by
`sync.sh` and `pull.sh`), `GPU_DIR` (remote folder, default `/tmp/uiwue-fusion`), `GPU_KEY`
(optional SSH key), `PY` (Python) and `BATCHES` (list of extra embedding folders). Nothing about a
machine or a user is hard-coded.

| Step | Script | Where |
|---|---|---|
| Download every ADA camera archive in parallel | `scripts/download_parallel.py --streams 18` | local |
| Embeddings | `python -m uiwue.fusion.embed --plants <ids> --hsi-windows --cache cache/fusion_external/b<n>` | local GPU |
| Hand-made thermal features | `scripts/fusion/features.sh [workers]` | local |
| Send code, embeddings and tables to the GPU machine | `scripts/fusion/sync.sh` | local |
| v1: stage 1 and stage 3 (LoRA thermal) | `scripts/fusion/train_v1.sh` | GPU machine |
| v2: ablation, 17 variants and the 2-Sep rows | `scripts/fusion/ablate.sh` | GPU machine or local |
| Apply the selection rule (fixed before round 2) | `python scripts/fusion/pick_variant.py <ablation parquet ...>` | anywhere |
| Train the chosen design, score the lock-box once | `scripts/fusion/train_final.sh <variant>` | GPU machine |
| Pull results back into `runs/pull/` | `scripts/fusion/pull.sh` | local |
| One table of every round | `python scripts/fusion/combine.py derived/detect/fusion_v2_ablation.parquet final=<files> ...` | local |
| Markdown tables for `docs/fusion.md` | `python scripts/fusion/tables.py ...`, `python scripts/fusion/report.py ...` | local |

On the GPU machine, run the scripts from `GPU_DIR` with `PY` set to its venv's Python.
