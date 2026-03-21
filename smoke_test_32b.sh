#!/bin/bash

# Hugging Face need token for public 32B model, otherwise we get a 401 error. The token is not needed for smaller models like Qwen2.5-0.5B.


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Load HF_TOKEN from the local .env file (not committed).
# We intentionally avoid printing the token.
if [[ -f .env ]]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

: "${HF_TOKEN:?HF_TOKEN is not set. Put it in .env as HF_TOKEN=...}"

# Defaults for a Clariden 4xGH200 node.
CE_ENVIRONMENT="${CE_ENVIRONMENT:-qwen3-clariden}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "Submitting 32B smoke: MODEL_NAME=${MODEL_NAME} TP=${TENSOR_PARALLEL_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN} CE_ENVIRONMENT=${CE_ENVIRONMENT}" >&2

sbatch --export=ALL,CE_ENVIRONMENT="${CE_ENVIRONMENT}",MODEL_NAME="${MODEL_NAME}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}",MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
	scripts/smoke_test_vllm_qwen3.sh