#!/usr/bin/env bash
# Set up UiWUE model training on a Linux GPU machine (tested on 1 x RTX PRO 6000 96 GB; the
# scripts also handle multi-GPU machines such as 4 x A100 80 GB).
#
# Run on the GPU machine, from the repo copy (see scripts/push_gpu.sh):
#   bash scripts/setup_gpu.sh              # env + models
#   bash scripts/setup_gpu.sh env          # only the Python environment
#   bash scripts/setup_gpu.sh data prep    # download raw ADA data + build samples here
#                                          # (needs ~70 GB; otherwise build on the workstation
#                                          #  and push with scripts/push_gpu.sh promptda)
#
# uv, the managed Python 3.12, and the Hugging Face cache all live in <repo>/.cache, so
# nothing is written to a full /home. DATA_DIR overrides where data goes (default <repo>/data).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.cache/uv-python}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/hf}"
PY="$ROOT/.venv/bin/python"
DATA="${DATA_DIR:-$ROOT/data}"

stage_env() {
  if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
  fi
  [[ -x "$PY" ]] || uv venv --python 3.12 .venv
  # --torch-backend auto picks the CUDA wheel that matches the installed driver.
  uv pip install --python "$PY" -e ".[ml]" --torch-backend auto
  "$PY" - <<'EOF'
import torch, transformers, accelerate
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| GPUs", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  {i}: {p.name} {p.total_memory / 2**30:.0f} GiB, sm_{p.major}{p.minor}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", bool(torch.isfinite(x @ x).all()))
print("transformers", transformers.__version__, "| accelerate", accelerate.__version__)
EOF
}

stage_models() {
  "$PY" scripts/download_models.py
}

stage_data() {
  # depth + RGB of the Tier 1-3 plants (env first: the Tier 3 list comes from its workbook),
  # PhenoBench for the leaf model, then extract.
  DATA_DIR="$DATA" PYTHON="$PY" scripts/download_data.sh env train
  "$PY" external/ada2025/utils/unzip_data.py --data-dir "$DATA" --modalities depth rgb \
    --beds W1 W2 --plants all -y
  DATA_DIR="$DATA" PYTHON="$PY" scripts/download_phenobench.sh
}

stage_prep() {
  "$PY" -m uiwue.ml.promptda.prep --data-dir "$DATA" --out "$DATA/derived/promptda" --jobs "$(nproc)"
}

stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && stages=(env models)
for s in "${stages[@]}"; do
  echo "=== $s ($(date '+%F %T')) ==="
  "stage_$s"
done
echo "=== setup done ==="
