#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=run/online_eagle3_common.sh
source "$SCRIPT_DIR/online_eagle3_common.sh"

online_eagle3_cd_repo
online_eagle3_require_bench_deps

: "${DATA:=/srv/Datasets/TTS/aime2025.jsonl}"
: "${NUM_PROMPTS:=16}"
: "${CUSTOM_OUTPUT_LEN:=2048}"
: "${MAX_CONCURRENCY:=1}"
: "${REQUEST_RATE:=inf}"
: "${BENCH_BACKEND:=openai}"
: "${BENCH_ENDPOINT:=/v1/completions}"
: "${BENCH_SEED:=0}"
: "${IGNORE_EOS:=1}"
: "${SKIP_CHAT_TEMPLATE:=0}"
: "${ONLINE_EAGLE3_UPDATE_INTERVAL:=1}"
: "${ONLINE_EAGLE3_LR:=1e-5}"
: "${ONLINE_EAGLE3_DTYPE:=float32}"
: "${ONLINE_EAGLE3_TORCH_THREADS:=8}"

RESULT_DIR=$(online_eagle3_result_dir "online_eagle3_bench")
SERVER_PID=""

cleanup() {
    online_eagle3_stop_server "$SERVER_PID"
}
trap cleanup EXIT

echo "Result dir: $RESULT_DIR"

BENCH_ARGS=(
    bench serve
    --backend "$BENCH_BACKEND"
    --base-url "$BASE"
    --endpoint "$BENCH_ENDPOINT"
    --model "$MODEL_NAME"
    --tokenizer "$MODEL"
    --trust-remote-code
    --dataset-name custom
    --dataset-path "$DATA"
    --custom-output-len "$CUSTOM_OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$MAX_CONCURRENCY"
    --request-rate "$REQUEST_RATE"
    --seed "$BENCH_SEED"
    --disable-shuffle
    --temperature 0
    --save-result
    --result-dir "$RESULT_DIR"
)

if [[ "$IGNORE_EOS" == "1" ]]; then
    BENCH_ARGS+=(--ignore-eos)
fi

if [[ "$SKIP_CHAT_TEMPLATE" == "1" ]]; then
    BENCH_ARGS+=(--skip-chat-template)
fi

run_one() {
    local label=$1
    local online_enabled=$2
    local log_file="$RESULT_DIR/${label}_server.log"

    echo "Starting $label server..."
    env \
        VLLM_ONLINE_EAGLE3="$online_enabled" \
        VLLM_ONLINE_EAGLE3_DRAFT_MODEL="$DRAFT" \
        VLLM_ONLINE_EAGLE3_UPDATE_INTERVAL="$ONLINE_EAGLE3_UPDATE_INTERVAL" \
        VLLM_ONLINE_EAGLE3_LR="$ONLINE_EAGLE3_LR" \
        VLLM_ONLINE_EAGLE3_DTYPE="$ONLINE_EAGLE3_DTYPE" \
        VLLM_ONLINE_EAGLE3_TORCH_THREADS="$ONLINE_EAGLE3_TORCH_THREADS" \
        .venv/bin/python -m vllm.entrypoints.cli.main serve "${SERVER_ARGS[@]}" \
        >"$log_file" 2>&1 &
    SERVER_PID=$!

    online_eagle3_wait_for_server "$SERVER_PID" "$log_file"

    .venv/bin/python -m vllm.entrypoints.cli.main \
        "${BENCH_ARGS[@]}" \
        --result-filename "${label}.json"

    online_eagle3_scan_log "$log_file"
    online_eagle3_stop_server "$SERVER_PID"
    SERVER_PID=""
}

run_one baseline 0
run_one online 1

.venv/bin/python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
base = json.loads((result_dir / "baseline.json").read_text())
online = json.loads((result_dir / "online.json").read_text())

keys = [
    "spec_decode_acceptance_length",
    "spec_decode_acceptance_rate",
    "spec_decode_num_drafts",
    "spec_decode_accepted_tokens",
    "total_token_throughput",
    "output_throughput",
    "mean_tpot_ms",
]

print("result_dir:", result_dir)
for key in keys:
    baseline_value = base.get(key)
    online_value = online.get(key)
    if baseline_value is None or online_value is None:
        delta = None
    else:
        delta = online_value - baseline_value
    print(
        f"{key}: baseline={baseline_value} "
        f"online={online_value} delta={delta}"
    )
PY
