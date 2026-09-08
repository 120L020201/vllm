#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/online_eagle3_common.sh"

MODE=${1:-online}
SMOKE_MAX_TOKENS=${SMOKE_MAX_TOKENS:-256}
SMOKE_POST_WAIT_SECONDS=${SMOKE_POST_WAIT_SECONDS:-75}
SMOKE_PROMPT=${SMOKE_PROMPT:-"Solve step by step: 1+1="}

case "$MODE" in
  baseline)
    ONLINE_FLAG=0
    ;;
  online)
    ONLINE_FLAG=1
    ;;
  *)
    die "usage: $0 [baseline|online]"
    ;;
esac

validate_models

RUN_DIR=${RUN_DIR:-"${RUN_ROOT}/online_eagle3_smoke_$(date +%Y%m%d_%H%M%S)"}
LOG_FILE="${RUN_DIR}/server.log"
RESPONSE_FILE="${RUN_DIR}/response.json"
mkdir -p "$RUN_DIR"

cleanup() {
  stop_vllm_server
}
trap cleanup EXIT

start_vllm_server "$ONLINE_FLAG" "$LOG_FILE"
wait_for_server "$LOG_FILE"

"$PYTHON_BIN" - "$MODEL_NAME" "$SMOKE_PROMPT" "$SMOKE_MAX_TOKENS" <<'PY' \
  | curl -sS "${BASE}/v1/completions" \
    -H "Content-Type: application/json" \
    -d @- >"$RESPONSE_FILE"
import json
import sys

model, prompt, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(
    json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }
    )
)
PY

sleep "$SMOKE_POST_WAIT_SECONDS"
ensure_server_alive "$LOG_FILE"

echo "Run dir: $RUN_DIR"
echo "Response: $RESPONSE_FILE"
echo "Log scan:"
scan_log "$LOG_FILE"
