#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=run/online_eagle3_common.sh
source "$SCRIPT_DIR/online_eagle3_common.sh"

online_eagle3_cd_repo

: "${ONLINE_EAGLE3:=1}"
: "${ONLINE_EAGLE3_UPDATE_INTERVAL:=1}"
: "${ONLINE_EAGLE3_LR:=1e-5}"
: "${ONLINE_EAGLE3_DTYPE:=float32}"
: "${ONLINE_EAGLE3_TORCH_THREADS:=8}"
: "${TRACE_MAX_TOKENS:=1024}"
: "${TRACE_MAX_ITERATIONS:=64}"
: "${TRACE_POST_WAIT_SECONDS:=5}"
: "${TRACE_PROMPT:=Solve the following AIME problem step by step: Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.}"

TRACE_LABEL=online
if [[ "$ONLINE_EAGLE3" == "0" ]]; then
    TRACE_LABEL=baseline
fi

RESULT_DIR=$(online_eagle3_result_dir "online_eagle3_trace_${TRACE_LABEL}")
LOG="$RESULT_DIR/server.log"
RESPONSE="$RESULT_DIR/response.json"
SERVER_PID=""

cleanup() {
    online_eagle3_stop_server "$SERVER_PID"
}
trap cleanup EXIT

echo "Trace dir: $RESULT_DIR"

env \
    VLLM_ONLINE_EAGLE3="$ONLINE_EAGLE3" \
    VLLM_ONLINE_EAGLE3_DRAFT_MODEL="$DRAFT" \
    VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL="$ONLINE_EAGLE3_UPDATE_INTERVAL" \
    VLLM_ONLINE_EAGLE3_LR="$ONLINE_EAGLE3_LR" \
    VLLM_ONLINE_EAGLE3_DTYPE="$ONLINE_EAGLE3_DTYPE" \
    VLLM_ONLINE_EAGLE3_TORCH_THREADS="$ONLINE_EAGLE3_TORCH_THREADS" \
    .venv/bin/python -m vllm.entrypoints.cli.main serve \
    "${SERVER_ARGS[@]}" \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir="$RESULT_DIR" \
    --profiler-config.torch_profiler_with_stack=false \
    --profiler-config.ignore_frontend=true \
    --profiler-config.max_iterations="$TRACE_MAX_ITERATIONS" \
    >"$LOG" 2>&1 &
SERVER_PID=$!

online_eagle3_wait_for_server "$SERVER_PID" "$LOG"

curl -fsS -X POST "$BASE/start_profile"

curl -sS "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(printf \
        '{"model":"%s","prompt":"%s","max_tokens":%s,"temperature":0,"ignore_eos":true}' \
        "$MODEL_NAME" \
        "$TRACE_PROMPT" \
        "$TRACE_MAX_TOKENS")" \
    >"$RESPONSE"

sleep "$TRACE_POST_WAIT_SECONDS"
curl -fsS -X POST "$BASE/stop_profile"

online_eagle3_scan_log "$LOG"
find "$RESULT_DIR" -maxdepth 2 -type f -printf "%p %k KB\n" | sort

TRACE_FILE=$(find "$RESULT_DIR" -name "*.pt.trace.json.gz" -print -quit)
if [[ -n "$TRACE_FILE" ]]; then
    .venv/bin/python run/online_eagle3_trace_summary.py "$TRACE_FILE"
fi

echo "Response: $RESPONSE"
echo "Log: $LOG"
