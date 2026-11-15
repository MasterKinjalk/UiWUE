#!/usr/bin/env bash
# Hand-made thermal features (uiwue.fusion.features) for every plant embedded so far: the plants in
# cache/fusion and in every embedding folder listed in BATCHES (each holds a plants.txt).
#
# Usage: UIWUE_DATA=<data dir> scripts/fusion/features.sh [workers] [extra thermal roots ...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"
W="${1:-4}"; shift || true
plants=""
for d in $(caches); do
  [[ -f "$d/plants.txt" ]] && plants="$plants $(tr -d '\r' < "$d/plants.txt" | xargs)"
done
# shellcheck disable=SC2086
"$PY" -m uiwue.fusion.features --roots "${UIWUE_DATA:?set UIWUE_DATA}/thermal" "$@" --workers "$W" --plants $plants
