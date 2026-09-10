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
REPEAT_RESPONSE="$RESULT_DIR/response_repeat.json"
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
    VLLM_ONLINE_EAGLE3_WEIGHT_DECAY="$ONLINE_EAGLE3_WEIGHT_DECAY" \
    VLLM_ONLINE_EAGLE3_CHECK_GRADIENTS="$ONLINE_EAGLE3_CHECK_GRADIENTS" \
    VLLM_ONLINE_EAGLE3_DTYPE="$ONLINE_EAGLE3_DTYPE" \
    VLLM_ONLINE_EAGLE3_TORCH_THREADS="$ONLINE_EAGLE3_TORCH_THREADS" \
    .venv/bin/python -m vllm.entrypoints.cli.main serve "${SERVER_ARGS[@]}" \
    >"$LOG" 2>&1 &
SERVER_PID=$!

online_eagle3_wait_for_server "$SERVER_PID" "$LOG"

REQUEST_BODY=$(.venv/bin/python - "$MODEL_NAME" "$SMOKE_PROMPT" "$SMOKE_MAX_TOKENS" <<'PY'
import json
import sys

print(json.dumps({"model": sys.argv[1], "prompt": sys.argv[2],
                  "max_tokens": int(sys.argv[3]), "temperature": 0, "ignore_eos": True}))
PY
)
for response_path in "$RESPONSE" "$REPEAT_RESPONSE"; do
    curl -fsS "$BASE/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$REQUEST_BODY" | tee "$response_path"
    echo
done

.venv/bin/python - "$RESPONSE" "$REPEAT_RESPONSE" <<'PY'
import json
import sys

texts = []
for path in sys.argv[1:]:
    with open(path) as response_file:
        response = json.load(response_file)
    assert "error" not in response, response
    assert response["choices"], response
    assert response["usage"]["completion_tokens"] > 0, response
    texts.append(response["choices"][0]["text"])
assert texts[0] == texts[1], texts
PY
rg -q "S=1 CPU distillation:" "$LOG"
rg -q "Applied first online EAGLE3 GPU draft weight snapshot" "$LOG"
rg -q "Reset online EAGLE3 CPU draft" "$LOG"

online_eagle3_scan_log "$LOG"
echo "Response: $RESPONSE"
echo "Log: $LOG"
