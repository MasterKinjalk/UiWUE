#!/usr/bin/env bash
# Copy code and prepared training data from this workstation to the GPU machine (tar over SSH).
#
# Usage (Git Bash or Linux, from anywhere):
#   scripts/push_gpu.sh code                 # pyproject, requirements, src/, scripts/, tests/, docs/
#   scripts/push_gpu.sh phenobench           # PhenoBench train + val (images and masks, ~5.4 GB)
#   scripts/push_gpu.sh promptda             # data/derived/promptda (built by uiwue.ml.promptda.prep)
#   scripts/push_gpu.sh code phenobench promptda
#
# Environment:
#   GPU_HOST  required, user@host of the GPU machine
#   GPU_DIR   remote work folder (default /tmp/uiwue-train; pick a disk with room, and note
#             that /tmp is usually cleaned after some days unused)
#   GPU_KEY   optional SSH private key; without it, ssh uses your normal keys/agent
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${GPU_HOST:?set GPU_HOST=user@host of the GPU machine}"
DIR="${GPU_DIR:-/tmp/uiwue-train}"
SSH=(ssh -o BatchMode=yes)
[[ -n "${GPU_KEY:-}" ]] && SSH+=(-i "$GPU_KEY" -o IdentitiesOnly=yes)
SSH+=("$HOST")

push() {  # push <paths relative to ROOT ...>
  tar -C "$ROOT" --exclude='__pycache__' --exclude='*.pyc' -cf - "$@" \
    | "${SSH[@]}" "mkdir -p '$DIR' && tar -xf - -C '$DIR'"
}

items=("$@")
[[ ${#items[@]} -eq 0 ]] && items=(code)
for it in "${items[@]}"; do
  echo "=== push $it -> $HOST:$DIR ($(date '+%T')) ==="
  case "$it" in
    code)
      push pyproject.toml requirements.txt requirements-ml.txt README.md src scripts tests docs
      # Files written on Windows may carry CRLF, which breaks bash on Linux.
      "${SSH[@]}" "find '$DIR/scripts' '$DIR/src' '$DIR/tests' -type f \( -name '*.sh' -o -name '*.py' \) -exec sed -i 's/\r\$//' {} +" ;;
    phenobench)
      paths=()
      for s in train val; do
        for d in images semantics plant_instances leaf_instances; do
          paths+=("data/phenobench/PhenoBench/$s/$d")
        done
      done
      push "${paths[@]}" ;;
    promptda)
      push data/derived/promptda ;;
    *) echo "unknown item: $it (code | phenobench | promptda)"; exit 1 ;;
  esac
done
"${SSH[@]}" "du -sh '$DIR' 2>/dev/null; df -h '$DIR' | tail -1"
