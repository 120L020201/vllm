#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=run/online_eagle3_common.sh
source "$SCRIPT_DIR/online_eagle3_common.sh"

online_eagle3_cd_repo

: "${ONLINE_EAGLE3_UPDATE_INTERVAL:=1}"
: "${ONLINE_EAGLE3_LR:=1e-5}"
: "${ONLINE_EAGLE3_DTYPE:=float32}"
: "${ONLINE_EAGLE3_TORCH_THREADS:=8}"
: "${SMOKE_MAX_TOKENS:=256}"
: "${SMOKE_PROMPT:=Solve step by step: 1+1=}"

RESULT_DIR=$(online_eagle3_result_dir "online_eagle3_smoke")
LOG="$RESULT_DIR/server.log"
RESPONSE="$RESULT_DIR/response.json"
SERVER_PID=""

cleanup() {
    online_eagle3_stop_server "$SERVER_PID"
}
trap cleanup EXIT

echo "Result dir: $RESULT_DIR"

env \
    VLLM_ONLINE_EAGLE3=1 \
    VLLM_ONLINE_EAGLE3_DRAFT_MODEL="$DRAFT" \
    VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL="$ONLINE_EAGLE3_UPDATE_INTERVAL" \
    VLLM_ONLINE_EAGLE3_LR="$ONLINE_EAGLE3_LR" \
    VLLM_ONLINE_EAGLE3_DTYPE="$ONLINE_EAGLE3_DTYPE" \
    VLLM_ONLINE_EAGLE3_TORCH_THREADS="$ONLINE_EAGLE3_TORCH_THREADS" \
    .venv/bin/python -m vllm.entrypoints.cli.main serve "${SERVER_ARGS[@]}" \
    >"$LOG" 2>&1 &
SERVER_PID=$!

online_eagle3_wait_for_server "$SERVER_PID" "$LOG"

curl -sS "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(printf \
        '{"model":"%s","prompt":"%s","max_tokens":%s,"temperature":0,"ignore_eos":true}' \
        "$MODEL_NAME" \
        "$SMOKE_PROMPT" \
        "$SMOKE_MAX_TOKENS")" \
    | tee "$RESPONSE"
echo

online_eagle3_scan_log "$LOG"
echo "Response: $RESPONSE"
echo "Log: $LOG"
