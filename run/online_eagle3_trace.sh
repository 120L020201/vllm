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
: "${TRACE_CONTEXT_TOKENS:=0}"
: "${TRACE_PROMPT:=Solve the following AIME problem step by step: Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.}"

TRACE_LABEL=online
if [[ "$ONLINE_EAGLE3" == "0" ]]; then
    TRACE_LABEL=baseline
fi

RESULT_DIR=$(online_eagle3_result_dir "online_eagle3_trace_${TRACE_LABEL}")
LOG="$RESULT_DIR/server.log"
RESPONSE="$RESULT_DIR/response.json"
REQUEST="$RESULT_DIR/request.json"
SERVER_PID=""

cleanup() {
    online_eagle3_stop_server "$SERVER_PID"
}
trap cleanup EXIT

echo "Trace dir: $RESULT_DIR"

.venv/bin/python - "$REQUEST" "$MODEL_NAME" "$MODEL" "$TRACE_PROMPT" \
    "$TRACE_CONTEXT_TOKENS" "$TRACE_MAX_TOKENS" "$MAX_MODEL_LEN" \
    "$SPECULATIVE_CONFIG" "$ONLINE_EAGLE3_DTYPE" "$ONLINE_EAGLE3_TORCH_THREADS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
context_tokens, max_tokens, max_model_len = map(int, sys.argv[5:8])
if context_tokens < 0 or max_tokens <= 0:
    raise ValueError("Context length must be nonnegative and output length positive")
prompt = sys.argv[4]
if context_tokens:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sys.argv[3], local_files_only=True)
    suffix = tokenizer.encode("\n\n" + prompt, add_special_tokens=False)
    if context_tokens < len(suffix):
        raise ValueError("Requested context is shorter than the question")
    if context_tokens + max_tokens > max_model_len:
        raise ValueError("Context plus output exceeds MAX_MODEL_LEN")
    passage = tokenizer.encode(
        "Reference notes: An integer base b uses digits from 0 to b-1. "
        "The numeral 17 in base b represents b+7; 97 represents 9*b+7.\n",
        add_special_tokens=False,
    )
    prefix_len = context_tokens - len(suffix)
    prompt = (passage * (prefix_len // len(passage) + 1))[:prefix_len] + suffix
request = dict(model=sys.argv[2], prompt=prompt, max_tokens=max_tokens,
               temperature=0, ignore_eos=True)
path.write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
config = dict(context_tokens=context_tokens or None, max_tokens=max_tokens,
              max_model_len=max_model_len, speculative_config=json.loads(sys.argv[8]),
              cpu_dtype=sys.argv[9], cpu_threads=int(sys.argv[10]))
path.with_name("trace_config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8")
print("Trace config:", json.dumps(config))
PY

env \
    VLLM_ONLINE_EAGLE3="$ONLINE_EAGLE3" \
    VLLM_ONLINE_EAGLE3_DRAFT_MODEL="$DRAFT" \
    VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL="$ONLINE_EAGLE3_UPDATE_INTERVAL" \
    VLLM_ONLINE_EAGLE3_LR="$ONLINE_EAGLE3_LR" \
    VLLM_ONLINE_EAGLE3_WEIGHT_DECAY="$ONLINE_EAGLE3_WEIGHT_DECAY" \
    VLLM_ONLINE_EAGLE3_CHECK_GRADIENTS="$ONLINE_EAGLE3_CHECK_GRADIENTS" \
    VLLM_ONLINE_EAGLE3_DTYPE="$ONLINE_EAGLE3_DTYPE" \
    VLLM_ONLINE_EAGLE3_TORCH_THREADS="$ONLINE_EAGLE3_TORCH_THREADS" \
    .venv/bin/python -m vllm.entrypoints.cli.main serve \
    "${SERVER_ARGS[@]}" \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir="$RESULT_DIR" \
    --profiler-config.torch_profiler_with_stack=false \
    --profiler-config.torch_profiler_record_shapes=true \
    --profiler-config.ignore_frontend=true \
    --profiler-config.max_iterations="$TRACE_MAX_ITERATIONS" \
    >"$LOG" 2>&1 &
SERVER_PID=$!

online_eagle3_wait_for_server "$SERVER_PID" "$LOG"

curl -fsS -X POST "$BASE/start_profile"

curl -fsS "$BASE/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@$REQUEST" \
    >"$RESPONSE"

.venv/bin/python - "$RESPONSE" "$TRACE_CONTEXT_TOKENS" "$TRACE_MAX_TOKENS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as response_file:
    response = json.load(response_file)
assert "error" not in response and response["choices"], response
usage = response["usage"]
assert usage["completion_tokens"] == int(sys.argv[3]), response
if int(sys.argv[2]):
    assert usage["prompt_tokens"] == int(sys.argv[2]), response
print("Actual token counts:", usage)
PY

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
