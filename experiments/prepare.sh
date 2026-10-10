#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

trap 'status=$?; echo "prepare.sh: failed at line ${BASH_LINENO[0]} (exit $status)" >&2; exit "$status"' ERR

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-${VIRTUAL_ENV:-$ROOT_DIR/.venv}/bin/python}
DATA_DIR=${DATA_DIR:-/srv/Datasets}
MODEL_DIR=${MODEL_DIR:-/srv/Models}
DATASETS=${DATASETS-aime2026 gpqa_diamond mmlu-pro-computer_science livecodebench-lite LongBench-v2 LongWriter-6k}
MODEL_SIZES=${MODEL_SIZES-4b 8b}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
MAX_WORKERS=${MAX_WORKERS:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

[[ -x "$PYTHON_BIN" ]] || {
  echo "prepare.sh: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
}

dataset_command=(
  env "HF_ENDPOINT=$HF_ENDPOINT" "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}" PYTHONUNBUFFERED=1
  "$PYTHON_BIN" -u "$ROOT_DIR/experiments/prepare_datasets.py"
  --output-dir "$DATA_DIR"
  --hf-endpoint "$HF_ENDPOINT"
)
for dataset in $DATASETS; do
  dataset_command+=(--dataset "$dataset")
done

model_command=(
  env "HF_ENDPOINT=$HF_ENDPOINT" "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}" PYTHONUNBUFFERED=1
  "$PYTHON_BIN" -u "$ROOT_DIR/experiments/prepare_models.py"
  --output-dir "$MODEL_DIR"
  --max-workers "$MAX_WORKERS"
)
for size in $MODEL_SIZES; do
  model_command+=(--model-size "$size")
done

if [[ "$FORCE" == 1 ]]; then
  dataset_command+=(--force)
  model_command+=(--force)
fi

if [[ "$DRY_RUN" == 1 ]]; then
  if [[ -n "$DATASETS" ]]; then
    printf 'datasets:'
    printf ' %q' "${dataset_command[@]}"
    printf '\n'
  fi
  if [[ -n "$MODEL_SIZES" ]]; then
    printf 'models:'
    printf ' %q' "${model_command[@]}"
    printf '\n'
  fi
  exit 0
fi

echo "prepare.sh: datasets='${DATASETS:-<skip>}' -> $DATA_DIR"
echo "prepare.sh: model_sizes='${MODEL_SIZES:-<skip>}' -> $MODEL_DIR"
echo "prepare.sh: endpoint=$HF_ENDPOINT max_workers=$MAX_WORKERS"

if [[ -n "$DATASETS" ]]; then
  echo "prepare.sh: starting dataset preparation"
  "${dataset_command[@]}"
fi
if [[ -n "$MODEL_SIZES" ]]; then
  echo "prepare.sh: starting model preparation"
  "${model_command[@]}"
fi
echo "prepare.sh: complete"
