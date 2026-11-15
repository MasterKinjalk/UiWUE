#!/usr/bin/env bash
# v1 (docs/fusion.md section 7): stage 1 (frozen encoders) and stage 3 (LoRA on the thermal
# encoder) on the plants in cache/fusion. Stage 3 also needs the raw thermal frames (UIWUE_DATA).
#
# Usage: UIWUE_DATA=<data dir> scripts/fusion/train_v1.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -d cache/fusion ]] || cd "$ROOT"
mkdir -p logs
{ date; "$PY" -m uiwue.fusion.train; date; } > logs/stage1.log 2>&1
{ date; "$PY" -m uiwue.fusion.train --stage lora --lora-sensor thermal --workers "${WORKERS:-6}"; date; } > logs/stage3.log 2>&1
