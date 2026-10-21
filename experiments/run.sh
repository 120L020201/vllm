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
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-0}
SPEC_TOKENS=${SPEC_TOKENS:-7}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
STOP_TOKEN_IDS=${STOP_TOKEN_IDS:-151645,151643}
ENABLE_THINKING=${ENABLE_THINKING:-1}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
SEED=${SEED:-0}
D_TEMPERATURE=${D_TEMPERATURE:-0}
MAX_GPU_MEMORY_GIB=${MAX_GPU_MEMORY_GIB:-}
GPU_MEMORY_HEADROOM_GIB=${GPU_MEMORY_HEADROOM_GIB:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
GPU_ATTENTION_BACKEND=${GPU_ATTENTION_BACKEND:-FLASHINFER}
FLASHINFER_AUTOTUNE=${FLASHINFER_AUTOTUNE:-0}
WARMUP_TOKENS=${WARMUP_TOKENS:-16}
CHECK_GRADIENTS=${CHECK_GRADIENTS:-0}
FROZEN_PARAMETER_PREFIXES=${FROZEN_PARAMETER_PREFIXES:-}
FUSED_OPTIMIZER=${FUSED_OPTIMIZER:-1}
if [[ "$METHOD" == online_draft ]]; then
  LOG_UPDATES=${LOG_UPDATES:-1}
else
  LOG_UPDATES=${LOG_UPDATES:-0}
fi
CPU_TRAINING_ATTENTION_BACKEND=${CPU_TRAINING_ATTENTION_BACKEND:-flash_attention}
VLLM_ONLINE_DRAFT_TRAIN=${VLLM_ONLINE_DRAFT_TRAIN:-}
VLLM_ONLINE_DRAFT_PROFILE=${VLLM_ONLINE_DRAFT_PROFILE:-0}
MONITOR=${MONITOR:-0}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}
OUTPUT_DIR=${OUTPUT_DIR:-results}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
DATASET_TAG=${DATASET_TAG:-${BENCHMARKS// /-}}
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL:-${BASE_MODEL_PATH:-model}}")}
RUN_DIR=${RUN_DIR:-$OUTPUT_DIR/${METHOD:-run}-${DATASET_TAG}-${MODEL_TAG}-${RUN_TIMESTAMP}}
OUTPUT_FILE=${OUTPUT_FILE:-$RUN_DIR/results.jsonl}
SERVER_LOG=${SERVER_LOG:-$RUN_DIR/server.log}
MONITOR_FILE=${MONITOR_FILE:-$RUN_DIR/monitor.csv}
RUN_LOG=${RUN_LOG:-}
STATS_FILE=${STATS_FILE:-$RUN_DIR/stats.json}
UPDATE_DELAY=${UPDATE_DELAY:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
OSPEC_TRAIN_MAX_LEN=${OSPEC_TRAIN_MAX_LEN:-}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-}

usage() {
  cat <<'EOF'
Usage: bash experiments/run.sh {base|eagle3|online_draft|tts|ospec}

The script starts one OpenAI-compatible vLLM server, waits for readiness,
evaluates DATA_DIR/BENCHMARKS sequentially, and writes JSONL results.
In BENCHMARKS, name(N) samples N questions and name[N] selects question N.
online_draft enables CPU online training; eagle3 never trains the draft.
Only base and eagle3 support RESUME=1 (their default).
online_draft, tts, and ospec require RESUME=0 (their default).
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
  base|eagle3|online_draft|tts|ospec) ;;
  -h|--help) usage; exit 0 ;;
  *) die "unknown method '$METHOD'" ;;
esac

case "$METHOD" in
  base|eagle3) RESUME=${RESUME:-1} ;;
  *) RESUME=${RESUME:-0} ;;
esac
if [[ "$METHOD" == online_draft ]]; then
  VLLM_ONLINE_DRAFT_TRAIN=${VLLM_ONLINE_DRAFT_TRAIN:-1}
else
  VLLM_ONLINE_DRAFT_TRAIN=${VLLM_ONLINE_DRAFT_TRAIN:-0}
fi

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -n "$MODEL" || -n "$BASE_MODEL_PATH" ]] || die "MODEL is required"
MODEL=${MODEL:-$BASE_MODEL_PATH}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-$MODEL}
[[ -d "$MODEL" ]] || die "model directory does not exist: $MODEL"
[[ -d "$DATA_DIR" ]] || die "DATA_DIR does not exist: $DATA_DIR"
if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then
  [[ "$MAX_GPU_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MAX_GPU_MEMORY_GIB must be numeric"
fi
[[ "$GPU_MEMORY_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GPU_MEMORY_HEADROOM_GIB must be numeric"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || die "RESUME must be 0 or 1"
[[ "$MONITOR" == 0 || "$MONITOR" == 1 ]] || die "MONITOR must be 0 or 1"
[[ "$MONITOR_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MONITOR_INTERVAL must be numeric"
awk -v interval="$MONITOR_INTERVAL" 'BEGIN { if (interval <= 0) exit 1 }' || die "MONITOR_INTERVAL must be positive"
[[ "$ENABLE_THINKING" == 0 || "$ENABLE_THINKING" == 1 ]] || die "ENABLE_THINKING must be 0 or 1"
[[ "$ENABLE_CHUNKED_PREFILL" == 0 || "$ENABLE_CHUNKED_PREFILL" == 1 ]] || die "ENABLE_CHUNKED_PREFILL must be 0 or 1"
[[ "$TRUST_REMOTE_CODE" == 0 || "$TRUST_REMOTE_CODE" == 1 ]] || die "TRUST_REMOTE_CODE must be 0 or 1"
[[ "$FLASHINFER_AUTOTUNE" == 0 || "$FLASHINFER_AUTOTUNE" == 1 ]] || die "FLASHINFER_AUTOTUNE must be 0 or 1"
[[ "$CHECK_GRADIENTS" == 0 || "$CHECK_GRADIENTS" == 1 ]] || die "CHECK_GRADIENTS must be 0 or 1"
[[ "$FUSED_OPTIMIZER" == 0 || "$FUSED_OPTIMIZER" == 1 ]] || die "FUSED_OPTIMIZER must be 0 or 1"
[[ "$LOG_UPDATES" == 0 || "$LOG_UPDATES" == 1 ]] || die "LOG_UPDATES must be 0 or 1"
[[ "$VLLM_ONLINE_DRAFT_TRAIN" == 0 || "$VLLM_ONLINE_DRAFT_TRAIN" == 1 ]] || die "VLLM_ONLINE_DRAFT_TRAIN must be 0 or 1"
[[ "$VLLM_ONLINE_DRAFT_PROFILE" == 0 || "$VLLM_ONLINE_DRAFT_PROFILE" == 1 ]] || die "VLLM_ONLINE_DRAFT_PROFILE must be 0 or 1"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a nonnegative integer"
[[ "$WARMUP_TOKENS" =~ ^[0-9]+$ ]] || die "WARMUP_TOKENS must be a nonnegative integer"
[[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && ((MAX_MODEL_LEN > 0)) || die "MAX_MODEL_LEN must be a positive integer"
[[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]] && ((MAX_NUM_BATCHED_TOKENS > 0)) || die "MAX_NUM_BATCHED_TOKENS must be a positive integer"
[[ "$MAX_OUTPUT_TOKENS" =~ ^[0-9]+$ ]] || die "MAX_OUTPUT_TOKENS must be a nonnegative integer"
[[ "$MAX_PROMPT_TOKENS" =~ ^[0-9]+$ ]] || die "MAX_PROMPT_TOKENS must be a nonnegative integer"
((MAX_PROMPT_TOKENS == 0 || MAX_PROMPT_TOKENS < MAX_MODEL_LEN)) || die "MAX_PROMPT_TOKENS must be zero or less than MAX_MODEL_LEN"

if [[ "$RESUME" == 1 && "$METHOD" != base && "$METHOD" != eagle3 ]]; then
  die "$METHOD does not support RESUME=1; restart with RESUME=0"
fi
if [[ "$VLLM_ONLINE_DRAFT_TRAIN" == 1 && "$METHOD" != online_draft ]]; then
  die "use 'bash experiments/run.sh online_draft' instead of VLLM_ONLINE_DRAFT_TRAIN=1 with $METHOD"
fi
if [[ "$METHOD" == online_draft && "$VLLM_ONLINE_DRAFT_TRAIN" != 1 ]]; then
  die "online_draft requires CPU training; unset VLLM_ONLINE_DRAFT_TRAIN or set it to 1"
fi
if [[ "$VLLM_ONLINE_DRAFT_PROFILE" == 1 && "$VLLM_ONLINE_DRAFT_TRAIN" != 1 ]]; then
  die "VLLM_ONLINE_DRAFT_PROFILE requires the online_draft entry"
fi
export VLLM_ONLINE_DRAFT_TRAIN VLLM_ONLINE_DRAFT_PROFILE

server_url="http://$HOST:$PORT"
if [[ "$DRY_RUN" != 1 ]]; then
  server_url=$("$PYTHON_BIN" "$ROOT_DIR/experiments/server_guard.py" check-port \
    --host "$HOST" --port "$PORT") || die "server address is unavailable; no server was started"
fi

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
  resume_mode=question
fi
thinking_arg=--enable-thinking
[[ "$ENABLE_THINKING" == 1 ]] || thinking_arg=--no-enable-thinking
chunked_prefill_arg=--enable-chunked-prefill
[[ "$ENABLE_CHUNKED_PREFILL" == 1 ]] || chunked_prefill_arg=--no-enable-chunked-prefill
trust_remote_code_arg=--trust-remote-code
[[ "$TRUST_REMOTE_CODE" == 1 ]] || trust_remote_code_arg=--no-trust-remote-code
check_gradients_arg=--check-gradients
[[ "$CHECK_GRADIENTS" == 1 ]] || check_gradients_arg=--no-check-gradients
fused_optimizer_arg=--fused-optimizer
[[ "$FUSED_OPTIMIZER" == 1 ]] || fused_optimizer_arg=--no-fused-optimizer
log_updates_arg=--log-updates
[[ "$LOG_UPDATES" == 1 ]] || log_updates_arg=--no-log-updates
flashinfer_autotune_arg=--flashinfer-autotune
vllm_flashinfer_autotune_arg=--enable-flashinfer-autotune
if [[ "$FLASHINFER_AUTOTUNE" == 0 ]]; then
  flashinfer_autotune_arg=--no-flashinfer-autotune
  vllm_flashinfer_autotune_arg=--no-enable-flashinfer-autotune
fi

if [[ "$METHOD" != base ]]; then
  [[ -n "$SPEC_MODEL" || -n "$EA_MODEL_PATH_1" ]] || die "SPEC_MODEL is required"
  SPEC_MODEL=${SPEC_MODEL:-$EA_MODEL_PATH_1}
  EA_MODEL_PATH_1=${EA_MODEL_PATH_1:-$SPEC_MODEL}
  [[ -d "$SPEC_MODEL" ]] || die "speculative model directory does not exist: $SPEC_MODEL"
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
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$SERVER_LOG")" "$(dirname "$MONITOR_FILE")"
rm -f "$STATS_FILE"

gpu_utilization=$GPU_MEMORY_UTILIZATION
gpu_budget_gib=
if [[ -n "$MAX_GPU_MEMORY_GIB" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  total_gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
  gpu_budget_gib=$(awk -v cap="$MAX_GPU_MEMORY_GIB" -v headroom="$GPU_MEMORY_HEADROOM_GIB" 'BEGIN { print cap - headroom }')
  gpu_utilization=$(awk -v budget="$gpu_budget_gib" -v total="$total_gpu_mib" -v requested="$GPU_MEMORY_UTILIZATION" 'BEGIN { cap=budget*1024/total; if (requested < cap) cap=requested; printf "%.8f", cap }')
  awk -v budget="$gpu_budget_gib" 'BEGIN { if (budget <= 0) exit 1 }' || die "GPU memory cap must exceed headroom"
elif [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then
  die "nvidia-smi is required to enforce the ${MAX_GPU_MEMORY_GIB} GiB GPU limit"
fi

common_server_args=(
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --max-num-seqs 1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$gpu_utilization"
  --attention-backend "$GPU_ATTENTION_BACKEND"
  "$vllm_flashinfer_autotune_arg"
  --generation-config vllm
  --dtype bfloat16
  --seed "$SEED"
  -cc.cudagraph_mode=FULL_AND_PIECEWISE
  --no-enable-prefix-caching
  "$chunked_prefill_arg"
)
if [[ "$TRUST_REMOTE_CODE" == 1 ]]; then
  common_server_args+=(--trust-remote-code)
fi

export OSD_STATS_FILE="$STATS_FILE"

if [[ "$METHOD" == base ]]; then
  server_command=(
    "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL"
    "${common_server_args[@]}"
  )
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
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --gpu-attention-backend "$GPU_ATTENTION_BACKEND"
    "$flashinfer_autotune_arg"
    --learning-rate "${TTS_LEARNING_RATE:-1e-5}"
    --update-stride "${TTS_UPDATE_STRIDE:-1}"
    --chunk-size "${CHUNK_SIZE:-5}"
    --probability "${UPDATE_PROBABILITY:-0.5}"
    --seed "$SEED"
    --ensemble-lrs "${OSPEC_ENSEMBLE_LRS:-1e-5,2e-5,3e-5}"
    --epsilon "${OSPEC_EPSILON:-0.1}"
    "$chunked_prefill_arg"
    "$trust_remote_code_arg"
    "$check_gradients_arg"
    "$fused_optimizer_arg"
    "$log_updates_arg"
    --frozen-parameter-prefixes "$FROZEN_PARAMETER_PREFIXES"
    --cpu-training-attention-backend "$CPU_TRAINING_ATTENTION_BACKEND"
  )
  if [[ -n "$MAX_GPU_MEMORY_GIB" ]]; then
    server_command+=(--max-gpu-memory-gib "$MAX_GPU_MEMORY_GIB")
  fi
  if [[ "${TTS_RESET_PER_REQUEST:-1}" == 0 || "${OSD_KEEP_WEIGHTS:-0}" == 1 ]]; then
    server_command+=(--keep-weights)
  fi
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'server:'
  printf ' %q' "${server_command[@]}"
  printf '\n'
  printf 'training: cpu_online=%s profile=%s\n' "$VLLM_ONLINE_DRAFT_TRAIN" "$VLLM_ONLINE_DRAFT_PROFILE"
  printf 'benchmark: %q' "$PYTHON_BIN"
  printf ' %q' "$ROOT_DIR/experiments/benchmark.py"
  printf ' --method %q' "$METHOD"
  printf ' --data-dir %q --benchmarks %q --max-model-len %q --max-output-tokens %q' \
    "$DATA_DIR" "$BENCHMARKS" "$MAX_MODEL_LEN" "$MAX_OUTPUT_TOKENS"
  printf ' --max-prompt-tokens %q --temperature %q --top-p %q --top-k %q' \
    "$MAX_PROMPT_TOKENS" "$TEMPERATURE" "$TOP_P" "$TOP_K"
  printf ' --min-p %q --presence-penalty %q --frequency-penalty %q' \
    "$MIN_P" "$PRESENCE_PENALTY" "$FREQUENCY_PENALTY"
  printf ' --repetition-penalty %q --stop-token-ids %q' \
    "$REPETITION_PENALTY" "$STOP_TOKEN_IDS"
  printf ' --warmup-tokens %q' "$WARMUP_TOKENS"
  printf ' --seed %q' "$SEED"
  printf ' %q' "$thinking_arg"
  printf ' --resume-mode %q' "$resume_mode"
  printf '\n'
  printf 'files: run_dir=%q output=%q server_log=%q stats=%q monitor=%q run_log=%q\n' \
    "$RUN_DIR" "$OUTPUT_FILE" "$SERVER_LOG" "$STATS_FILE" "$MONITOR_FILE" \
    "${RUN_LOG:-<terminal>}"
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

read_cpu_counters() {
  awk '
    /^cpu / {
      idle = $5 + $6
      total = 0
      for (field = 2; field <= 9 && field <= NF; field++) {
        total += $field
      }
      printf "%.0f %.0f\n", total, idle
      exit
    }
  ' /proc/stat
}

read_memory_mib() {
  awk '
    /^MemTotal:/ { total = $2 }
    /^MemAvailable:/ { available = $2 }
    END {
      printf "%.2f %.2f\n", (total - available) / 1024, total / 1024
    }
  ' /proc/meminfo
}

read_server_tree_usage() {
  ps -eo pid=,ppid=,pcpu=,rss= | awk -v root="$server_pid" '
    {
      process_id[NR] = $1
      parent[$1] = $2
      cpu[$1] = $3
      rss[$1] = $4
    }
    END {
      selected[root] = 1
      changed = 1
      while (changed) {
        changed = 0
        for (row = 1; row <= NR; row++) {
          current = process_id[row]
          if (!selected[current] && selected[parent[current]]) {
            selected[current] = 1
            changed = 1
          }
        }
      }
      for (row = 1; row <= NR; row++) {
        current = process_id[row]
        if (selected[current]) {
          total_cpu += cpu[current]
          total_rss += rss[current]
        }
      }
      printf "%.2f %.2f\n", total_cpu, total_rss / 1024
    }
  '
}

monitor_resources() {
  local previous_total previous_idle current_total current_idle
  local delta_total delta_idle host_cpu host_memory_used host_memory_total
  local server_cpu server_rss timestamp gpu_rows
  local gpu_index gpu_memory_used gpu_memory_total gpu_utilization

  read -r previous_total previous_idle < <(read_cpu_counters)
  while kill -0 "$server_pid" 2>/dev/null; do
    sleep "$MONITOR_INTERVAL"
    kill -0 "$server_pid" 2>/dev/null || break

    read -r current_total current_idle < <(read_cpu_counters)
    delta_total=$((current_total - previous_total))
    delta_idle=$((current_idle - previous_idle))
    host_cpu=$(awk -v total="$delta_total" -v idle="$delta_idle" 'BEGIN { if (total <= 0) print "0.00"; else printf "%.2f", 100 * (total - idle) / total }')
    previous_total=$current_total
    previous_idle=$current_idle

    read -r host_memory_used host_memory_total < <(read_memory_mib)
    read -r server_cpu server_rss < <(read_server_tree_usage)
    server_cpu=${server_cpu:-0.00}
    server_rss=${server_rss:-0.00}
    timestamp=$(date -Is)
    gpu_rows=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{ gsub(/[[:space:]]/, ""); print }') || gpu_rows=

    if [[ -z "$gpu_rows" ]]; then
      printf '%s,%s,%s,%s,%s,%s,,,,\n' \
        "$timestamp" "$host_cpu" "$host_memory_used" "$host_memory_total" \
        "$server_cpu" "$server_rss"
      continue
    fi
    while IFS=, read -r gpu_index gpu_memory_used gpu_memory_total gpu_utilization; do
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$timestamp" "$host_cpu" "$host_memory_used" "$host_memory_total" \
        "$server_cpu" "$server_rss" "$gpu_index" "$gpu_memory_used" \
        "$gpu_memory_total" "$gpu_utilization"
    done <<<"$gpu_rows"
  done
}

echo "method=$METHOD model=$MODEL data=$DATA_DIR"
echo "gpu_memory_limit=${MAX_GPU_MEMORY_GIB:-<none>} vllm_budget=${gpu_budget_gib:-<uncapped>} utilization=$gpu_utilization"
echo "gpu_attention_backend=$GPU_ATTENTION_BACKEND flashinfer_autotune=$FLASHINFER_AUTOTUNE"
echo "run_log=${RUN_LOG:-<external>} server_log=$SERVER_LOG output=$OUTPUT_FILE resume=$resume_mode"
if [[ "$RESUME" == 1 ]]; then
  "${server_command[@]}" >>"$SERVER_LOG" 2>&1 &
else
  "${server_command[@]}" >"$SERVER_LOG" 2>&1 &
fi
server_pid=$!

if [[ "$MONITOR" == 1 ]]; then
  monitor_header="timestamp,host_cpu_percent,host_memory_used_mib,host_memory_total_mib,server_tree_cpu_percent,server_tree_rss_mib,gpu_index,gpu_memory_used_mib,gpu_memory_total_mib,gpu_utilization_percent"
  if [[ "$RESUME" == 1 ]]; then
    [[ -s "$MONITOR_FILE" ]] || printf '%s\n' "$monitor_header" >"$MONITOR_FILE"
  else
    printf '%s\n' "$monitor_header" >"$MONITOR_FILE"
  fi
  monitor_resources >>"$MONITOR_FILE" 2>&1 &
  monitor_pid=$!
fi

"$PYTHON_BIN" "$ROOT_DIR/experiments/server_guard.py" wait \
  --host "$HOST" --port "$PORT" --pid "$server_pid" \
  || die "server did not become ready with an owned listener; see $SERVER_LOG"
kill -0 "$server_pid" 2>/dev/null || die "server exited before benchmark; see $SERVER_LOG"

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT_DIR/experiments/benchmark.py" \
  --method "$METHOD" \
  --base-url "$server_url" \
  --model "$MODEL" \
  --data-dir "$DATA_DIR" \
  --benchmarks "$BENCHMARKS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p "$MIN_P" \
  --presence-penalty "$PRESENCE_PENALTY" \
  --frequency-penalty "$FREQUENCY_PENALTY" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --stop-token-ids "$STOP_TOKEN_IDS" \
  --warmup-tokens "$WARMUP_TOKENS" \
  --seed "$SEED" \
  "$thinking_arg" \
  --stats-file "$STATS_FILE" \
  --resume-mode "$resume_mode" \
  --output "$OUTPUT_FILE"
