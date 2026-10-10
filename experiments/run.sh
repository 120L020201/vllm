#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
METHOD=${1:-}
PYTHON_BIN=${PYTHON_BIN:-${VIRTUAL_ENV:-$ROOT_DIR/.venv}/bin/python}
DATA_DIR=${DATA_DIR:-/srv/Datasets}
BENCHMARKS=${BENCHMARKS:-aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2}
MODEL=${MODEL:-}
SPEC_MODEL=${SPEC_MODEL:-}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MODEL}
EA_MODEL_PATH_1=${EA_MODEL_PATH_1:-$SPEC_MODEL}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-49152}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-$((MAX_MODEL_LEN - MAX_OUTPUT_TOKENS))}
SPEC_TOKENS=${SPEC_TOKENS:-7}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}
ENABLE_THINKING=${ENABLE_THINKING:-1}
SEED=${SEED:-0}
D_TEMPERATURE=${D_TEMPERATURE:-0}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-}
GPU_MEMORY_HEADROOM_GIB=${GPU_MEMORY_HEADROOM_GIB:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MONITOR=${MONITOR:-0}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}
NSYS=${NSYS:-0}
OUTPUT_DIR=${OUTPUT_DIR:-results}
OUTPUT_FILE=${OUTPUT_FILE:-$OUTPUT_DIR/${METHOD:-run}.jsonl}
SERVER_LOG=${SERVER_LOG:-$OUTPUT_DIR/${METHOD:-run}.server.log}
RUN_LOG=${RUN_LOG:-}
STATS_FILE=${STATS_FILE:-$OUTPUT_DIR/${METHOD:-run}.stats.json}
UPDATE_DELAY=${UPDATE_DELAY:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
OSPEC_TRAIN_MAX_LEN=${OSPEC_TRAIN_MAX_LEN:-}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-1}

usage() {
  cat <<'EOF'
Usage: bash experiments/run.sh {base|eagle3|tts|ospec}

The script starts one OpenAI-compatible vLLM server, waits for readiness,
evaluates DATA_DIR/BENCHMARKS sequentially, and writes JSONL results.
EOF
}

die() {
  echo "run.sh: $*" >&2
  exit 2
}

[[ -n "$METHOD" ]] || {
  usage >&2
  exit 2
}
case "$METHOD" in
  base|eagle3|tts|ospec) ;;
  -h|--help) usage; exit 0 ;;
  *) die "unknown method '$METHOD'" ;;
esac

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -n "$MODEL" || -n "$BASE_MODEL_PATH" ]] || die "MODEL is required"
MODEL=${MODEL:-$BASE_MODEL_PATH}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MODEL}
[[ -d "$MODEL" ]] || die "model directory does not exist: $MODEL"
[[ -d "$DATA_DIR" ]] || die "DATA_DIR does not exist: $DATA_DIR"
model_name=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
if [[ -z "$MAX_GPU_MEMORY_GIB" ]]; then
  case "$model_name" in
    *4b*) MAX_GPU_MEMORY_GIB=24 ;;
    *8b*) MAX_GPU_MEMORY_GIB=32 ;;
    *) die "cannot infer 4B/8B memory limit from MODEL; set MAX_GPU_MEMORY_GIB" ;;
  esac
fi
[[ "$MAX_GPU_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MAX_GPU_MEMORY_GIB must be numeric"
[[ "$GPU_MEMORY_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GPU_MEMORY_HEADROOM_GIB must be numeric"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || die "RESUME must be 0 or 1"
[[ "$ENABLE_THINKING" == 0 || "$ENABLE_THINKING" == 1 ]] || die "ENABLE_THINKING must be 0 or 1"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a nonnegative integer"
((MAX_PROMPT_TOKENS > 0)) || die "MAX_MODEL_LEN must exceed MAX_OUTPUT_TOKENS"

if [[ -n "$RUN_LOG" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  if [[ "$RESUME" == 1 ]]; then
    exec >>"$RUN_LOG" 2>&1
  else
    exec >"$RUN_LOG" 2>&1
  fi
fi

resume_mode=none
if [[ "$RESUME" == 1 ]]; then
  case "$METHOD" in
    base|eagle3) resume_mode=question ;;
    tts|ospec) resume_mode=dataset ;;
  esac
fi
thinking_arg=--enable-thinking
[[ "$ENABLE_THINKING" == 1 ]] || thinking_arg=--no-enable-thinking

if [[ "$METHOD" != base ]]; then
  [[ -n "$SPEC_MODEL" || -n "$EA_MODEL_PATH_1" ]] || die "SPEC_MODEL is required"
  SPEC_MODEL=${SPEC_MODEL:-$EA_MODEL_PATH_1}
  EA_MODEL_PATH_1=${EA_MODEL_PATH_1:-$SPEC_MODEL}
  [[ -d "$SPEC_MODEL" ]] || die "speculative model directory does not exist: $SPEC_MODEL"
fi

if [[ "$NSYS" != 0 ]]; then
  echo "warning: NSYS=$NSYS is accepted for compatibility but profiler wrapping is not implemented" >&2
fi
if [[ "$UPDATE_DELAY" != 0 ]]; then
  echo "warning: UPDATE_DELAY=$UPDATE_DELAY is accepted but online updates are synchronous" >&2
fi
if [[ "$BATCH_SIZE" != 1 ]]; then
  echo "warning: BATCH_SIZE=$BATCH_SIZE is not request concurrency; online training remains single-request ordered" >&2
fi
if [[ -n "$OSPEC_TRAIN_MAX_LEN" ]]; then
  echo "warning: OSPEC_TRAIN_MAX_LEN=$OSPEC_TRAIN_MAX_LEN is recorded but no token truncation is applied" >&2
fi
if [[ "$D_TEMPERATURE" != 0 && "$METHOD" != base ]]; then
  echo "warning: D_TEMPERATURE=$D_TEMPERATURE is ignored; the EAGLE3 draft sampler is greedy" >&2
fi
if ((MAX_MODEL_LEN > 40960)); then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  echo "warning: using Qwen3 RoPE extrapolation from 40960 to $MAX_MODEL_LEN tokens" >&2
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$SERVER_LOG")"
rm -f "$STATS_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  total_gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
  gpu_budget_gib=$(awk -v cap="$MAX_GPU_MEMORY_GIB" -v headroom="$GPU_MEMORY_HEADROOM_GIB" 'BEGIN { print cap - headroom }')
  gpu_utilization=$(awk -v budget="$gpu_budget_gib" -v total="$total_gpu_mib" -v requested="$GPU_MEMORY_UTILIZATION" 'BEGIN { cap=budget*1024/total; if (requested < cap) cap=requested; printf "%.8f", cap }')
else
  die "nvidia-smi is required to enforce the ${MAX_GPU_MEMORY_GIB} GiB GPU limit"
fi
awk -v budget="$gpu_budget_gib" 'BEGIN { if (budget <= 0) exit 1 }' || die "GPU memory cap must exceed headroom"

common_server_args=(
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --max-num-seqs 1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$gpu_utilization"
  --generation-config vllm
  --dtype bfloat16
  --enforce-eager
  --no-enable-chunked-prefill
  --no-enable-prefix-caching
)

if [[ "$METHOD" == base ]]; then
  server_command=(
    "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL"
    "${common_server_args[@]}"
  )
  export OSD_STATS_FILE="$STATS_FILE"
else
  methods_method=$METHOD
  [[ "$METHOD" == eagle3 ]] && methods_method=eagle
  server_command=(
    "$PYTHON_BIN" -m methods.run "$methods_method"
    --target "$MODEL"
    --draft "$SPEC_MODEL"
    --port "$PORT"
    --host "$HOST"
    --spec-tokens "$SPEC_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB"
    --learning-rate "${TTS_LEARNING_RATE:-2e-5}"
    --torch-threads "${OSD_TORCH_THREADS:-20}"
    --update-stride "${TTS_UPDATE_STRIDE:-1}"
    --chunk-size "${CHUNK_SIZE:-5}"
    --probability "${UPDATE_PROBABILITY:-0.5}"
    --seed "$SEED"
    --ensemble-lrs "${OSPEC_ENSEMBLE_LRS:-1e-5,2e-5,3e-5}"
    --epsilon "${OSPEC_EPSILON:-0.1}"
  )
  if [[ "${TTS_RESET_PER_REQUEST:-1}" == 0 || "${OSD_KEEP_WEIGHTS:-0}" == 1 ]]; then
    server_command+=(--keep-weights)
  fi
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'server:'
  printf ' %q' "${server_command[@]}"
  printf '\n'
  printf 'benchmark: %q' "$PYTHON_BIN"
  printf ' %q' "$ROOT_DIR/experiments/benchmark.py"
  printf ' --data-dir %q --benchmarks %q --max-output-tokens %q' \
    "$DATA_DIR" "$BENCHMARKS" "$MAX_OUTPUT_TOKENS"
  printf ' --presence-penalty %q' "$PRESENCE_PENALTY"
  printf ' --seed %q' "$SEED"
  printf ' %q' "$thinking_arg"
  printf ' --resume-mode %q' "$resume_mode"
  printf '\n'
  exit 0
fi

server_pid=
monitor_pid=
cleanup() {
  set +e
  [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null
  [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
  wait "$server_pid" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "method=$METHOD model=$MODEL data=$DATA_DIR"
echo "gpu_memory_limit=${MAX_GPU_MEMORY_GIB}GiB vllm_budget=${gpu_budget_gib}GiB utilization=$gpu_utilization"
echo "run_log=${RUN_LOG:-<external>} server_log=$SERVER_LOG output=$OUTPUT_FILE resume=$resume_mode"
if [[ "$RESUME" == 1 ]]; then
  "${server_command[@]}" >>"$SERVER_LOG" 2>&1 &
else
  "${server_command[@]}" >"$SERVER_LOG" 2>&1 &
fi
server_pid=$!

if [[ "$MONITOR" == 1 ]]; then
  if [[ "$RESUME" == 1 ]]; then
    (
      while kill -0 "$server_pid" 2>/dev/null; do
        date -Is
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
        sleep "$MONITOR_INTERVAL"
      done
    ) >>"${OUTPUT_FILE}.gpu.csv" 2>&1 &
  else
    (
      while kill -0 "$server_pid" 2>/dev/null; do
        date -Is
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
        sleep "$MONITOR_INTERVAL"
      done
    ) >"${OUTPUT_FILE}.gpu.csv" 2>&1 &
  fi
  monitor_pid=$!
fi

for _ in $(seq 1 300); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited before readiness; see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "http://$HOST:$PORT/health" >/dev/null || die "server did not become ready"

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT_DIR/experiments/benchmark.py" \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL" \
  --data-dir "$DATA_DIR" \
  --benchmarks "$BENCHMARKS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --presence-penalty "$PRESENCE_PENALTY" \
  --seed "$SEED" \
  "$thinking_arg" \
  --stats-file "$STATS_FILE" \
  --resume-mode "$resume_mode" \
  --output "$OUTPUT_FILE"
