#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/online_eagle3_common.sh"

TRACE_MAX_TOKENS=${TRACE_MAX_TOKENS:-512}
TRACE_POST_WAIT_SECONDS=${TRACE_POST_WAIT_SECONDS:-10}
TRACE_MAX_ITERATIONS=${TRACE_MAX_ITERATIONS:-64}
TRACE_PROMPT=${TRACE_PROMPT:-"Solve the following AIME problem step by step: Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b."}

validate_models

RUN_DIR=${RUN_DIR:-"${RUN_ROOT}/online_eagle3_trace_$(date +%Y%m%d_%H%M%S)"}
LOG_FILE="${RUN_DIR}/server.log"
RESPONSE_FILE="${RUN_DIR}/response.json"
mkdir -p "$RUN_DIR"

cleanup() {
  stop_vllm_server
}
trap cleanup EXIT

start_vllm_server 1 "$LOG_FILE" \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$RUN_DIR" \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.ignore_frontend=true \
  --profiler-config.max_iterations="$TRACE_MAX_ITERATIONS"
wait_for_server "$LOG_FILE"

curl -fsS -X POST "${BASE}/start_profile"

"$PYTHON_BIN" - "$MODEL_NAME" "$TRACE_PROMPT" "$TRACE_MAX_TOKENS" <<'PY' \
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

sleep "$TRACE_POST_WAIT_SECONDS"
curl -fsS -X POST "${BASE}/stop_profile"
ensure_server_alive "$LOG_FILE"

echo "Run dir: $RUN_DIR"
echo "Response: $RESPONSE_FILE"
echo "Trace files:"
find "$RUN_DIR" -maxdepth 2 -type f -printf "%p %k KB\n"
echo "Log scan:"
scan_log "$LOG_FILE"
