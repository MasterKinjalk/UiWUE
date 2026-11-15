#!/usr/bin/env bash
# Launch fine-tuning on the GPU machine (after scripts/setup_gpu.sh).
#
#   bash scripts/train_gpu.sh leafseg    # Mask2Former Swin-L on PhenoBench, all GPUs
#   bash scripts/train_gpu.sh promptda   # Prompt Depth Anything ViT-L, all GPUs
#   bash scripts/train_gpu.sh both       # both at once: split GPUs in half, or share GPU 0
#   bash scripts/train_gpu.sh seq promptda leafseg   # one after the other, all GPUs each
#
# Runs in the background with nohup; logs in logs/, checkpoints + TensorBoard in runs/.
# Extra arguments after the mode go to the training script(s), e.g. --epochs 10.
# GPUS=0,1 restricts the GPUs used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/hf}"
ACC="$ROOT/.venv/bin/accelerate"
DATA="${DATA_DIR:-$ROOT/data}"
MODE="${1:-both}"
shift || true
mkdir -p logs runs

ALL="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NGPU=$(awk -F, '{print NF}' <<<"$ALL")
NCPU=$(nproc)

launch() {  # launch <gpus csv> <port> <log> <module> [args...]
  local gpus="$1" port="$2" log="$3" module="$4"
  shift 4
  local n multi=()
  n=$(awk -F, '{print NF}' <<<"$gpus")
  [[ "$n" -gt 1 ]] && multi=(--multi_gpu)
  local cmd=(env CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 "$ACC" launch
    "${multi[@]}" --num_processes "$n" --num_machines 1 --mixed_precision bf16
    --dynamo_backend no --main_process_port "$port" -m "$module" "$@")
  if [[ "${WAIT:-0}" == 1 ]]; then  # foreground, used by seq
    echo "running $module on GPUs $gpus ($(date '+%F %T')), log: $log"
    "${cmd[@]}" >"$log" 2>&1 </dev/null
  else
    nohup "${cmd[@]}" >"$log" 2>&1 </dev/null &
    echo "started $module on GPUs $gpus (pid $!), log: $log"
  fi
}

workers() {  # data-loader workers per process, for <jobs> concurrent jobs
  local w=$(( NCPU / ($1 * NGPU) - 1 ))
  (( w < 2 )) && w=2
  (( w > 12 )) && w=12
  echo "$w"
}

promptda_args() {
  echo --data "$DATA/derived/promptda" --model models/promptda-vitl --out runs/promptda-vitl \
    --batch-size 8 --epochs 5 --eval-every 1000 --workers "$1"
}
leafseg_args() {
  echo --data "$DATA/phenobench/PhenoBench" --model models/m2f-swinl-inst --out runs/leafseg-m2f \
    --size 768 --batch-size 8 --epochs 50 --workers "$1"
}

case "$MODE" in
  promptda)
    # shellcheck disable=SC2046
    launch "$ALL" 29501 logs/train_promptda.log uiwue.ml.promptda.train $(promptda_args "$(workers 1)") "$@" ;;
  leafseg)
    # shellcheck disable=SC2046
    launch "$ALL" 29502 logs/train_leafseg.log uiwue.ml.leafseg.train $(leafseg_args "$(workers 1)") "$@" ;;
  both)
    if (( NGPU >= 2 )); then
      half=$(( NGPU / 2 ))
      g1=$(cut -d, -f1-"$half" <<<"$ALL")
      g2=$(cut -d, -f$(( half + 1 ))- <<<"$ALL")
    else
      g1="$ALL"; g2="$ALL"  # one GPU: both jobs share it (check memory with nvidia-smi)
    fi
    w=$(workers 2)
    # shellcheck disable=SC2046
    launch "$g1" 29501 logs/train_promptda.log uiwue.ml.promptda.train $(promptda_args "$w") "$@"
    # shellcheck disable=SC2046
    launch "$g2" 29502 logs/train_leafseg.log uiwue.ml.leafseg.train $(leafseg_args "$w") "$@"
    ;;
  seq)
    # shellcheck disable=SC2016
    # Each mode waits until no other uiwue trainer is running, then runs in the foreground.
    nohup env WAIT=1 bash -c '
      for m in "$@"; do
        while pgrep -f "[u]iwue[.]ml[.].*[.]train" >/dev/null; do sleep 60; done
        bash "$0" "$m" || exit 1
      done' "$0" "$@" >logs/train_seq.log 2>&1 </dev/null &
    echo "queued $* (pid $!), log: logs/train_seq.log"
    ;;
  *) echo "mode: promptda | leafseg | both | seq <mode> ..."; exit 1 ;;
esac
echo "watch: tail -f logs/train_*.log   |   .venv/bin/tensorboard --logdir runs"
