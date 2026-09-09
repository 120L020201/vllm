#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ONLINE_EAGLE3_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ONLINE_EAGLE3_REPO_ROOT=$(cd -- "$ONLINE_EAGLE3_SCRIPT_DIR/.." && pwd)

: "${MODEL:=/srv/Models/Qwen3-8B}"
: "${DRAFT:=/srv/Models/Qwen3-8B_eagle3}"
: "${MODEL_NAME:=qwen3-8b}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${BASE:=http://$HOST:$PORT}"
: "${NUM_SPECULATIVE_TOKENS:=4}"
: "${MAX_NUM_SEQS:=1}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.7}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${MODEL_DTYPE:=auto}"
: "${TRUST_REMOTE_CODE:=1}"
: "${SERVER_WAIT_ATTEMPTS:=180}"
: "${SERVER_WAIT_SECONDS:=2}"
: "${RESULT_ROOT:=runs}"

if [[ -z "${SPECULATIVE_CONFIG:-}" ]]; then
    SPECULATIVE_CONFIG=$(printf \
        '{"method":"eagle3","model":"%s","num_speculative_tokens":%s}' \
        "$DRAFT" \
        "$NUM_SPECULATIVE_TOKENS")
fi

SERVER_ARGS=(
    "$MODEL"
    --served-model-name "$MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
    --speculative-config "$SPECULATIVE_CONFIG"
)

if [[ -n "$MODEL_DTYPE" ]]; then
    SERVER_ARGS+=(--dtype "$MODEL_DTYPE")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    SERVER_ARGS+=(--trust-remote-code)
fi

online_eagle3_cd_repo() {
    cd "$ONLINE_EAGLE3_REPO_ROOT"
}

online_eagle3_result_dir() {
    local name=$1
    mkdir -p "$ONLINE_EAGLE3_REPO_ROOT/$RESULT_ROOT"
    mktemp -d "$ONLINE_EAGLE3_REPO_ROOT/$RESULT_ROOT/${name}_XXXXXXXX"
}

online_eagle3_wait_for_server() {
    local server_pid=$1
    local log_file=$2

    for _ in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
        if curl -fsS "$BASE/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -200 "$log_file" || true
            return 1
        fi
        sleep "$SERVER_WAIT_SECONDS"
    done

    echo "Timed out waiting for $BASE/v1/models" >&2
    tail -200 "$log_file" || true
    return 1
}

online_eagle3_stop_server() {
    local server_pid=${1:-}
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
}

online_eagle3_scan_log() {
    local log_file=$1
    rg -n \
        "Enabled synchronous online EAGLE3|Loading online EAGLE3 CPU draft|Online EAGLE3 CPU runtime|Completed first synchronous online EAGLE3 CPU update|Applied first online EAGLE3 GPU draft weight snapshot|Reset online EAGLE3 CPU draft|Failed to update online EAGLE3|Failed to reset online EAGLE3|EngineDead|out of memory|oom|SIGKILL|Traceback" \
        "$log_file" || true
}

online_eagle3_require_bench_deps() {
    if ! .venv/bin/python - <<'PY'
import pandas  # noqa: F401
PY
    then
        echo "Missing benchmark deps. Install them with:" >&2
        echo "  uv pip install -r requirements/test/cuda.in" >&2
        return 1
    fi
}
