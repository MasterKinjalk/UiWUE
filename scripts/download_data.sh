#!/usr/bin/env bash
# Download the ADA 2025 subsets from the Challenge 3 work plan (plan/Kinjalk-Work-Plan.md, section 3).
#
# Usage:  scripts/download_data.sh [stage ...]
#   env      environmental sub-dataset (sensor parquet files, metadata workbook)   ~1.5 GB
#   dev      depth + rgb of W1_J5 and W2_J5 (development plants)                    ~2.7 GB
#   tier1    depth + rgb + thermal + hsi of A2 A8 J5 R1 R7 in W1, W2 (paper plants)  ~29 GB
#   tier2    depth + rgb + thermal of the 8 hand-labelled QC plants                  ~19 GB
#   tier3    depth + rgb + thermal of the turgor-clip plants (from the Sensor Guide) ~28 GB
#   train    depth + rgb only of the Tier 1-3 plants (model training on a GPU server)  ~45 GB
#   unzip    extract every downloaded archive into data/<modality>/<bed>_<plant>/...
#   all      every stage above, in that order (default)
#
# Environment variables:
#   DATA_DIR  where data goes (default: <repo>/data)
#   PYTHON    interpreter (default: the repo .venv, else python3)
#
# Downloads resume: files already on disk with the right MD5 are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"
ADA="$ROOT/external/ada2025"

if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then PY="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"
else PY="python3"; fi

[[ -d "$ADA" ]] || git clone --depth 1 https://github.com/pippowell/ada2025 "$ADA"
mkdir -p "$DATA"

TIER1="A2 A8 J5 R1 R7"
TIER2="W1_B5 W1_E4 W1_G6 W1_K6 W2_B3 W2_G2 W2_I6 W2_Q6"

dl() { "$PY" "$ADA/utils/download_dataset.py" --data-dir "$DATA" -y "$@"; }

stage_env()   { dl --modalities environmental; }
stage_dev()   { dl --modalities depth rgb --plants W1_J5 W2_J5; }
stage_tier1() { dl --modalities depth rgb thermal hsi --beds W1 W2 --plants $TIER1; }
stage_tier2() { dl --modalities depth rgb thermal --plants $TIER2; }
stage_tier3() {
  local plants
  plants="$("$PY" "$ROOT/scripts/turgor_plants.py" --data-dir "$DATA")"
  echo "Tier 3 turgor-clip plants: $plants"
  # shellcheck disable=SC2086
  dl --modalities depth rgb thermal --plants $plants
}
stage_train() {
  local plants
  plants="$("$PY" "$ROOT/scripts/turgor_plants.py" --data-dir "$DATA")"
  # shellcheck disable=SC2086
  dl --modalities depth rgb --plants $(for b in W1 W2; do for p in $TIER1; do echo -n "${b}_$p "; done; done) $TIER2 $plants
}
stage_unzip() {
  "$PY" "$ADA/utils/unzip_data.py" --data-dir "$DATA" \
    --modalities depth rgb thermal hsi --beds W1 W2 W3 --plants all -y
}

stages=("$@")
[[ ${#stages[@]} -eq 0 || "${stages[0]}" == "all" ]] && stages=(env dev tier1 tier2 tier3 unzip)
for s in "${stages[@]}"; do
  echo "=== stage: $s ($(date '+%F %T')) ==="
  "stage_$s"
done
echo "=== done ($(date '+%F %T')) ==="
