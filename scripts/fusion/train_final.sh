#!/usr/bin/env bash
# Train the chosen v2 design on every development plant (5 plant folds, checkpoints in
# runs/fusion/fusion_v2), the v1 design on the same plants (checkpoints only, for the lock-box),
# score the CV subsets, then score the lock-box once.
#
# Usage: scripts/fusion/train_final.sh tfeat_only+win6
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -d cache/fusion ]] || cd "$ROOT"
V="${1:?usage: train_final.sh <variant chosen by pick_variant.py>}"
C="$(caches)"
mkdir -p logs logs/scratch_tables
L=logs/v2_train.log

{ date; echo "chosen: $V"; } > "$L"
"$PY" -m uiwue.fusion.v2 train --cache $C --variant "$V" --name fusion_v2 --n-boot 2000 >> "$L" 2>&1
date >> "$L"
"$PY" -m uiwue.fusion.v2 train --cache $C --variant v1 --name fusion_v1dev --n-boot 200 --out logs/scratch_tables >> "$L" 2>&1
date >> "$L"
"$PY" -m uiwue.fusion.predict --oof --model fusion_v2 >> "$L" 2>&1
{ date; echo LOCKBOX; } >> "$L"
"$PY" -m uiwue.fusion.predict --lockbox --cache $C --model fusion_v2 >> "$L" 2>&1
"$PY" -m uiwue.fusion.predict --lockbox --cache $C --model fusion_v1dev >> "$L" 2>&1
"$PY" -m uiwue.fusion.v2 bench-lockbox >> "$L" 2>&1
{ date; echo FINAL_TRAIN_DONE; } >> "$L"
