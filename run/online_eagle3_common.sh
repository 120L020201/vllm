#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}

TARGET=${TARGET:-/srv/Models/Qwen3-8B}
DRAFT=${DRAFT:-/srv/Models/Qwen3-8B_eagle3}
MODEL_NAME=${MODEL_NAME:-qwen3-8b}
DATA=${DATA:-/srv/Datasets/TTS/aime2025.jsonl}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BASE=${BASE:-"http://${HOST}:${PORT}"}
RUN_ROOT=${RUN_ROOT:-"${REPO_ROOT}/runs"}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
SPEC_TOKENS=${SPEC_TOKENS:-4}

ONLINE_UPDATE_INTERVAL=${ONLINE_UPDATE_INTERVAL:-1}
ONLINE_LR=${ONLINE_LR:-1e-5}
ONLINE_DTYPE=${ONLINE_DTYPE:-float32}
ONLINE_TORCH_THREADS=${ONLINE_TORCH_THREADS:-8}

SERVER_READY_TRIES=${SERVER_READY_TRIES:-180}
SERVER_READY_SLEEP=${SERVER_READY_SLEEP:-2}

SERVER_PID=""
SERVER_ARGS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

validate_models() {
  [[ -x "$PYTHON_BIN" ]] || die "missing Python executable: $PYTHON_BIN"
  [[ -d "$TARGET" ]] || die "TARGET not found: $TARGET"
  [[ -d "$DRAFT" ]] || die "DRAFT not found: $DRAFT"
  require_command curl
  require_command rg
}

validate_dataset() {
  [[ -f "$DATA" ]] || die "DATA not found: $DATA"
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("pandas") is None:
    print("ERROR: pandas is required by vllm bench custom dataset.", file=sys.stderr)
    print("Run: uv pip install pandas", file=sys.stderr)
    raise SystemExit(1)
PY
}

make_server_args() {
  SERVER_ARGS=(
    "$TARGET"
    --served-model-name "$MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --pipeline-parallel-size 1
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --generation-config vllm
    --spec-method eagle3
    --spec-model "$DRAFT"
    --spec-tokens "$SPEC_TOKENS"
  )
}

start_vllm_server() {
  local online_flag=$1
  local log_file=$2
  shift 2

  make_server_args
  mkdir -p "$(dirname "$log_file")"

  local server_env=("VLLM_ONLINE_EAGLE3=${online_flag}")
  if [[ "$online_flag" == "1" ]]; then
    server_env+=(
      "VLLM_ONLINE_EAGLE3_DRAFT_MODEL=${DRAFT}"
      "VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL=${ONLINE_UPDATE_INTERVAL}"
      "VLLM_ONLINE_EAGLE3_LR=${ONLINE_LR}"
      "VLLM_ONLINE_EAGLE3_DTYPE=${ONLINE_DTYPE}"
      "VLLM_ONLINE_EAGLE3_TORCH_THREADS=${ONLINE_TORCH_THREADS}"
    )
  fi

  echo "Starting server: online=${online_flag} log=${log_file}"
  env "${server_env[@]}" \
    "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve \
    "${SERVER_ARGS[@]}" "$@" >"$log_file" 2>&1 &
  SERVER_PID=$!
}

wait_for_server() {
  local log_file=$1
  local ready_url="${BASE}/v1/models"

  for _ in $(seq 1 "$SERVER_READY_TRIES"); do
    if curl -fsS "$ready_url" >/dev/null 2>&1; then
      echo "Server ready: $ready_url"
      return 0
    fi
    if [[ -z "$SERVER_PID" ]] || ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited during startup. Last log lines:" >&2
      tail -240 "$log_file" >&2 || true
      return 1
    fi
    sleep "$SERVER_READY_SLEEP"
  done

  echo "Server did not become ready. Last log lines:" >&2
  tail -240 "$log_file" >&2 || true
  return 1
}

stop_vllm_server() {
  local pid=${1:-$SERVER_PID}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  if [[ "$pid" == "${SERVER_PID:-}" ]]; then
    SERVER_PID=""
  fi
}

scan_log() {
  local log_file=$1
  rg -n \
    "SpecDecoding metrics|Failed to update online EAGLE3|Failed to reset online EAGLE3|EngineDead|out of memory|oom|SIGKILL|Traceback|ERROR" \
    "$log_file" || true
}

ensure_server_alive() {
  local log_file=$1
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    return 0
  fi
  echo "Server died. Last log lines:" >&2
  tail -240 "$log_file" >&2 || true
  return 1
}

cd "$REPO_ROOT"
