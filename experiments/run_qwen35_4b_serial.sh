#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARTIFACT_ROOT=/home/yuhengyu/vllm/methods/artifacts
MODEL_ROOT="$ARTIFACT_ROOT/models"
PYTHON_BIN=/home/yuhengyu/vllm/.venv/bin/python
MODEL="$MODEL_ROOT/Qwen3.5-4B"
EAGLE3_MODEL="$MODEL_ROOT/Qwen3.5-4B-Eagle3"
DFLASH_MODEL="$MODEL_ROOT/Qwen3.5-4B-DFlash"
DATA_DIR="$ARTIFACT_ROOT/datasets"
BENCHMARKS=${BENCHMARKS:-aime2026}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-0}
DRY_RUN=${DRY_RUN:-0}

METHODS=(
  eagle3
  online_draft+eagle3
  dflash
  online_draft+dflash
  mtp
  online_draft+mtp
)

for path in \
  "$PYTHON_BIN" \
  "$MODEL/config.json" \
  "$EAGLE3_MODEL/config.json" \
  "$DFLASH_MODEL/config.json" \
  "$DATA_DIR/aime2026.jsonl"; do
  [[ -e "$path" ]] || {
    echo "missing required path: $path" >&2
    exit 1
  }
done
[[ -x "$PYTHON_BIN" ]] || {
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
}

common_env=(
  "PATH=$(dirname "$PYTHON_BIN"):$PATH"
  "NO_PROXY=127.0.0.1,localhost"
  "no_proxy=127.0.0.1,localhost"
  "PYTHON_BIN=$PYTHON_BIN"
  "OMP_NUM_THREADS=20"
  "MKL_NUM_THREADS=20"
  "MONITOR=1"
  "RESUME=0"
  "SEED=0"
  "MAX_MODEL_LEN=32768"
  "MAX_NUM_BATCHED_TOKENS=16384"
  "ENABLE_CHUNKED_PREFILL=1"
  "MAX_OUTPUT_TOKENS=$MAX_OUTPUT_TOKENS"
  "MAX_PROMPT_TOKENS=0"
  "ENABLE_THINKING=1"
  "TEMPERATURE=0.6"
  "TOP_P=0.95"
  "TOP_K=20"
  "MIN_P=0.0"
  "PRESENCE_PENALTY=0.0"
  "DATA_DIR=$DATA_DIR"
  "BENCHMARKS=$BENCHMARKS"
  "MODEL=$MODEL"
  "GPU_ATTENTION_BACKEND=FLASH_ATTN"
  "LANGUAGE_MODEL_ONLY=1"
  "DRY_RUN=$DRY_RUN"
)

online_env=(
  "ONLINE_DRAFT_LOSS=position"
  "ONLINE_DRAFT_LR_REFERENCE_ROWS=60"
  "ONLINE_DRAFT_MASTER_WEIGHTS=1"
  "ONLINE_DRAFT_PERSISTENT_GRADS=1"
  "ONLINE_DRAFT_CONTINUOUS_TRAINING=1"
  "ONLINE_DRAFT_DFLASH_LOSS_GAMMA=7"
  "ONLINE_DRAFT_DFLASH_LOSS_NORMALIZATION=weights"
  "ONLINE_DRAFT_DFLASH_LOSS_REJECTION_WEIGHT=1"
  "ONLINE_DRAFT_DFLASH_LOSS_OBJECTIVE=kl"
  "ONLINE_DRAFT_DFLASH_LOSS_TEMPERATURE=0.6"
  "ONLINE_DRAFT_DFLASH_LOSS_TOP_K=0"
  "ONLINE_DRAFT_DFLASH_LOSS_TOP_P=1.0"
  "DFLASH_MAX_TRAIN_ROUNDS=16"
)

current_method=
trap 'status=$?; echo "FAILED: ${current_method:-preflight} (exit $status)" >&2' ERR

for current_method in "${METHODS[@]}"; do
  run_env=("${common_env[@]}")
  case "$current_method" in
    eagle3)
      run_env+=("SPEC_MODEL=$EAGLE3_MODEL" "SPEC_TOKENS=7")
      ;;
    online_draft+eagle3)
      run_env+=("${online_env[@]}" "SPEC_MODEL=$EAGLE3_MODEL" "SPEC_TOKENS=7")
      ;;
    dflash)
      run_env+=("SPEC_MODEL=$DFLASH_MODEL")
      ;;
    online_draft+dflash)
      run_env+=("${online_env[@]}" "SPEC_MODEL=$DFLASH_MODEL")
      ;;
    mtp)
      ;;
    online_draft+mtp)
      run_env+=("${online_env[@]}")
      ;;
    *)
      echo "unknown method: $current_method" >&2
      exit 1
      ;;
  esac

  echo "=== $(date -Is) starting $current_method ==="
  env "${run_env[@]}" bash "$REPO_DIR/experiments/run.sh" "$current_method"
  echo "=== $(date -Is) finished $current_method ==="
done

trap - ERR
echo "=== all configurations finished successfully ==="
