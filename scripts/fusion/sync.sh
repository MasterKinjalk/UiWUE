#!/usr/bin/env bash
# Send code, the embeddings of every complete batch, the thermal features, green cover and the
# derived tables to the GPU machine (tar over SSH), and write the batch list both sides use.
#
# A local batch cache/fusion_external/b<n> is complete when its manifest lists hsi (the last
# sensor embedded); a batch embedded on the GPU machine (g<n>) when it holds a .pulled marker.
#
# Usage: GPU_HOST=user@host scripts/fusion/sync.sh     (see common.sh for the other variables)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"
ssh_cmd

mkdir -p "$(dirname "$BATCHES")"
: > "$BATCHES"
for b in cache/fusion_external/b* cache/fusion_external/g*; do
  [[ -d "$b" ]] || continue
  if [[ -f "$b/.pulled" ]] || grep -q '"hsi"' "$b/manifest.json" 2>/dev/null; then echo "$b" >> "$BATCHES"; fi
done

files=$(ls src/uiwue/*/*.py src/uiwue/*.py cache/fusion_v2/thermal_features.parquet cache/fusion_v2/rgb_size.parquet \
  cache/fusion/emb_{rgb,depth,hsi}_full.npz cache/fusion/emb_thermal_{full,r160,r80}.npz 2>/dev/null)
while read -r b; do
  files="$files $(ls "$b"/emb_{rgb,depth,hsi}_full.npz "$b"/emb_thermal_{full,r160,r80}.npz 2>/dev/null)"
done < "$BATCHES"

for i in 1 2 3; do
  # shellcheck disable=SC2086
  tar cf - $files "$BATCHES" derived pyproject.toml \
    | "${SSH[@]}" "mkdir -p '$GPU_DIR' && cd '$GPU_DIR' && tar xf - && echo synced \$(wc -l < '$BATCHES') complete batches" \
    && break
  echo "sync retry $i"
done
