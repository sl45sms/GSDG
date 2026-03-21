#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

# Some EDF templates expand ${VLLM_HOST_IP}. Prefetch doesn't need it, but Pyxis
# requires it to be defined during EDF expansion.
VLLM_HOST_IP="${VLLM_HOST_IP:-}"
export VLLM_HOST_IP

# Default: model-only prefetch.
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_DATASETS="${SKIP_DATASETS:-1}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "Submitting prefetch: MODEL_NAME=${MODEL_NAME} SKIP_MODEL=${SKIP_MODEL} SKIP_DATASETS=${SKIP_DATASETS} CE_ENVIRONMENT=${CE_ENVIRONMENT}" >&2

sbatch --export=ALL,CE_ENVIRONMENT="${CE_ENVIRONMENT}",MODEL_NAME="${MODEL_NAME}",DATASET_SPLIT="${DATASET_SPLIT}",SKIP_MODEL="${SKIP_MODEL}",SKIP_DATASETS="${SKIP_DATASETS}",HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
	scripts/prefetch_hf_assets.sh
