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
DATASETS=${DATASETS:-}
MODEL_SIZES=${MODEL_SIZES:-}
DRAFT_METHODS=${DRAFT_METHODS:-${DRAFT_METHOD:-eagle3}}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
MAX_WORKERS=${MAX_WORKERS:-8}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

cli_datasets=()
cli_model_sizes=()
cli_draft_methods=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--model-size|--draft-method)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo "prepare.sh: $1 requires a value" >&2
        exit 2
      fi
      if [[ "$1" == --dataset ]]; then
        cli_datasets+=("$2")
      elif [[ "$1" == --model-size ]]; then
        cli_model_sizes+=("$2")
      else
        cli_draft_methods+=("$2")
      fi
      shift 2
      ;;
    -h|--help)
      printf '%s\n' \
        'Usage: bash experiments/prepare.sh [--dataset NAME] [--model-size 4b|8b|qwen3.5-4b]' \
        '       [--draft-method eagle3|dflash|mtp]' \
        'No datasets or models are selected by default. Options may be repeated.' \
        'DATASETS, MODEL_SIZES, and DRAFT_METHODS accept space-separated selections.' \
        'Draft downloads default to eagle3; MTP uses a target with embedded weights.' \
        'CLI selections override the corresponding environment variable.' \
        'Set DRY_RUN=1 to print commands without downloading.'
      exit 0
      ;;
    *)
      echo "prepare.sh: unknown argument: $1 (see --help)" >&2
      exit 2
      ;;
  esac
done
if [[ ${#cli_datasets[@]} -gt 0 ]]; then
  DATASETS="${cli_datasets[*]}"
fi
if [[ ${#cli_model_sizes[@]} -gt 0 ]]; then
  MODEL_SIZES="${cli_model_sizes[*]}"
fi
if [[ ${#cli_draft_methods[@]} -gt 0 ]]; then
  DRAFT_METHODS="${cli_draft_methods[*]}"
fi

if [[ -z "${DATASETS//[[:space:]]/}" && -z "${MODEL_SIZES//[[:space:]]/}" ]]; then
  echo "prepare.sh: nothing selected; use DATASETS/MODEL_SIZES or --dataset/--model-size"
  exit 0
fi

for size in $MODEL_SIZES; do
  for method in $DRAFT_METHODS; do
    if [[ "$method" == mtp && "$size" != 8b && "$size" != qwen3.5-4b ]]; then
      echo "prepare.sh: $size has no prepared target with embedded MTP; use 8b" >&2
      exit 2
    fi
    if [[ "$size" == qwen3.5-4b && "$method" != mtp ]]; then
      echo "prepare.sh: qwen3.5-4b preparation currently supports only mtp" >&2
      exit 2
    fi
  done
done

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
for method in $DRAFT_METHODS; do
  model_command+=(--draft-method "$method")
done

if [[ "$FORCE" == 1 ]]; then
  dataset_command+=(--force)
  model_command+=(--force)
fi

if [[ "$DRY_RUN" == 1 ]]; then
  if [[ -n "${DATASETS//[[:space:]]/}" ]]; then
    printf 'datasets:'
    printf ' %q' "${dataset_command[@]}"
    printf '\n'
  fi
  if [[ -n "${MODEL_SIZES//[[:space:]]/}" ]]; then
    printf 'models:'
    printf ' %q' "${model_command[@]}"
    printf '\n'
  fi
  exit 0
fi

echo "prepare.sh: datasets='${DATASETS:-<skip>}' -> $DATA_DIR"
echo "prepare.sh: model_sizes='${MODEL_SIZES:-<skip>}' drafts='${DRAFT_METHODS:-<none>}' -> $MODEL_DIR"
echo "prepare.sh: endpoint=$HF_ENDPOINT max_workers=$MAX_WORKERS"

if [[ -n "${DATASETS//[[:space:]]/}" ]]; then
  echo "prepare.sh: starting dataset preparation"
  "${dataset_command[@]}"
fi
if [[ -n "${MODEL_SIZES//[[:space:]]/}" ]]; then
  echo "prepare.sh: starting model preparation"
  "${model_command[@]}"
fi
echo "prepare.sh: complete"
