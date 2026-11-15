#!/usr/bin/env bash
# Final v2 ablation on every development plant (every embedded plant minus the lock-box), in four
# processes. The benchmark and size rows come with groups 1 and 4. Run where the embeddings are:
# the GPU machine after sync.sh (cd to GPU_DIR, PY=<its venv python>), or locally.
#
# Usage: scripts/fusion/ablate.sh            (NBOOT=2000, TAG=full by default)
# Then:  python scripts/fusion/pick_variant.py cache/fusion_v2/abl_full_*.parquet
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -d cache/fusion ]] || cd "$ROOT"
C="$(caches)"
T="${TAG:-full}"
A=("$PY" -m uiwue.fusion.v2 ablate --cache $C --n-boot "${NBOOT:-2000}")
mkdir -p logs cache/fusion_v2

date > "logs/v2_$T.log"
"${A[@]}" --variants v1 delta ref delta+ref now+delta+ref ref+tfeat \
  --out "cache/fusion_v2/abl_${T}_1.parquet" > "logs/v2_${T}_1.log" 2>&1 &
"${A[@]}" --variants delta+ref+tfeat tfeat_only delta+ref+tfeat+env delta+ref+tfeat+adv delta+ref+tfeat+adv2 \
  delta+ref+tfeat+hn --no-bench --out "cache/fusion_v2/abl_${T}_2.parquet" > "logs/v2_${T}_2.log" 2>&1 &
"${A[@]}" --variants delta+ref+tfeat+win6 tfeat_only+win6 fused_tfeat delta+ref+tfeat+mean v1+tfeat --no-bench \
  --out "cache/fusion_v2/abl_${T}_3.parquet" > "logs/v2_${T}_3.log" 2>&1 &
"${A[@]}" --variants v1 delta+ref+tfeat tfeat_only tfeat_only+win6 fused_tfeat --baseline 2025-09-02 2025-09-02 \
  --tag _base0902 --out "cache/fusion_v2/abl_${T}_base0902.parquet" > "logs/v2_${T}_4.log" 2>&1 &
wait
date >> "logs/v2_$T.log"
echo ABLATE_DONE >> "logs/v2_$T.log"
