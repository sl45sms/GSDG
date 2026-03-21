#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$ROOT_DIR"

# Load HF_TOKEN from the local .env file (not committed).
# Avoid printing the token.
if [[ -f .env ]]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

: "${HF_TOKEN:?HF_TOKEN is not set. Put it in .env as HF_TOKEN=...}"

CE_ENVIRONMENT="${CE_ENVIRONMENT:-qwen3-clariden}"
PREFETCH_DATASETS="${PREFETCH_DATASETS:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Some EDF templates expand ${VLLM_HOST_IP}. Dataset prefetch doesn't need it,
# but Pyxis requires it to be defined during EDF expansion.
VLLM_HOST_IP="${VLLM_HOST_IP:-}"
export VLLM_HOST_IP

if [[ -z "$PREFETCH_DATASETS" ]]; then
	echo "Set PREFETCH_DATASETS=glossAPI/foo,glossAPI/bar" >&2
	exit 2
fi

echo "Submitting dataset prefetch: DATASETS=${PREFETCH_DATASETS} SPLIT=${DATASET_SPLIT} CE_ENVIRONMENT=${CE_ENVIRONMENT}" >&2

# IMPORTANT: `sbatch --export=...` uses commas as separators, so passing a
# comma-separated dataset list there will truncate it. Instead, export the
# variables in our environment and submit with `--export=ALL`.
export CE_ENVIRONMENT
export PREFETCH_DATASETS
export DATASET_SPLIT
export HF_HUB_DISABLE_XET
export SKIP_MODEL=1
export SKIP_DATASETS=0

sbatch --export=ALL scripts/prefetch_hf_assets.sh
