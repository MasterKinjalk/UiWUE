#!/usr/bin/env bash
# Pull the v2 results back from the GPU machine into runs/pull/: the shared tables
# (derived/detect), the ablation parquet files, the checkpoints (runs/, not in git) and the logs.
# Copy what you want to keep from runs/pull/ into place afterwards.
#
# Usage: GPU_HOST=user@host scripts/fusion/pull.sh     (see common.sh for the other variables)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"
ssh_cmd

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for i in 1 2 3; do
  "${SSH[@]}" "cd '$GPU_DIR' && tar cf - derived/detect cache/fusion_v2/*.parquet runs/fusion/fusion_v2 runs/fusion/fusion_v1dev logs/v2_* 2>/dev/null" \
    > "$tmp/pull.tar" && break
  echo "pull retry $i"
done
mkdir -p runs/pull
tar --force-local -xf "$tmp/pull.tar" -C runs/pull
ls runs/pull/derived/detect | head -40
