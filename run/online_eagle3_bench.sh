#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/online_eagle3_common.sh"

MODE=${1:-both}
NUM_PROMPTS=${NUM_PROMPTS:-16}
CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:-512}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
REQUEST_RATE=${REQUEST_RATE:-inf}
SEED=${SEED:-0}
SERVER_STOP_SLEEP=${SERVER_STOP_SLEEP:-5}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-0}

BENCH_CHAT_TEMPLATE_ARGS=()
if [[ "$USE_CHAT_TEMPLATE" != "1" ]]; then
  BENCH_CHAT_TEMPLATE_ARGS+=(--skip-chat-template)
fi

case "$MODE" in
  baseline | online | both)
    ;;
  *)
    die "usage: $0 [baseline|online|both]"
    ;;
esac

validate_models
validate_dataset

RUN_DIR=${RUN_DIR:-"${RUN_ROOT}/online_eagle3_bench_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$RUN_DIR"

cleanup() {
  stop_vllm_server
}
trap cleanup EXIT

run_bench_one() {
  local label=$1
  local online_flag=$2
  local result_file=$3
  local log_file="${RUN_DIR}/${label}_server.log"

  start_vllm_server "$online_flag" "$log_file"
  wait_for_server "$log_file"

  "$PYTHON_BIN" -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --base-url "$BASE" \
    --endpoint /v1/completions \
    --model "$MODEL_NAME" \
    --tokenizer "$TARGET" \
    --dataset-name custom \
    --dataset-path "$DATA" \
    --custom-output-len "$CUSTOM_OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-rate "$REQUEST_RATE" \
    --seed "$SEED" \
    --disable-shuffle \
    "${BENCH_CHAT_TEMPLATE_ARGS[@]}" \
    --ignore-eos \
    --temperature 0 \
    --disable-tqdm \
    --save-result \
    --result-dir "$RUN_DIR" \
    --result-filename "$result_file"

  echo "${label} log scan:"
  scan_log "$log_file"
  stop_vllm_server
  sleep "$SERVER_STOP_SLEEP"
}

compare_results() {
  local baseline_json="${RUN_DIR}/baseline.json"
  local online_json="${RUN_DIR}/online.json"
  [[ -f "$baseline_json" && -f "$online_json" ]] || return 0

  "$PYTHON_BIN" - "$RUN_DIR" <<'PY'
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

print(f"result_dir: {result_dir}")
for key in keys:
    baseline_value = base.get(key)
    online_value = online.get(key)
    delta = (
        None
        if baseline_value is None or online_value is None
        else online_value - baseline_value
    )
    print(
        f"{key}: baseline={baseline_value} "
        f"online={online_value} delta={delta}"
    )
PY
}

case "$MODE" in
  baseline)
    run_bench_one baseline 0 baseline.json
    ;;
  online)
    run_bench_one online 1 online.json
    ;;
  both)
    run_bench_one baseline 0 baseline.json
    run_bench_one online 1 online.json
    compare_results
    ;;
esac

echo "Run dir: $RUN_DIR"
