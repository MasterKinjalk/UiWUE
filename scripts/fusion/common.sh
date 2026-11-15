# Shared settings for the fusion run scripts. Sourced, not run.
#
# Environment:
#   GPU_HOST  user@host of the GPU machine (needed by sync.sh and pull.sh)
#   GPU_DIR   remote work folder (default /tmp/uiwue-fusion)
#   GPU_KEY   optional SSH private key; without it, ssh uses your normal keys/agent
#   PY        Python to run (default .venv/Scripts/python.exe on Windows, .venv/bin/python else)
#   BATCHES   file listing the extra embedding folders, one per line
#             (default cache/fusion_v2/batches.txt, written by sync.sh)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_DIR="${GPU_DIR:-/tmp/uiwue-fusion}"
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then PY="$ROOT/.venv/Scripts/python.exe"; else PY="$ROOT/.venv/bin/python"; fi
fi
BATCHES="${BATCHES:-cache/fusion_v2/batches.txt}"
export PYTHONUNBUFFERED=1

ssh_cmd() {  # the ssh command line for GPU_HOST, as an array in SSH
  SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=40)
  [[ -n "${GPU_KEY:-}" ]] && SSH+=(-i "$GPU_KEY" -o IdentitiesOnly=yes)
  SSH+=("${GPU_HOST:?set GPU_HOST=user@host of the GPU machine}")
}

caches() {  # cache/fusion plus every folder listed in BATCHES
  echo "cache/fusion $( [[ -f "$BATCHES" ]] && tr -d '\r' < "$BATCHES" | xargs )"
}
