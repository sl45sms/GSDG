#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=gsdg-qwen3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

set -euo pipefail

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to a glossAPI dataset name}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/synthetic_chatml.jsonl}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
MAX_ROWS="${MAX_ROWS:-}"

detect_vllm_host_ip() {
	local interface_name
	for interface_name in hsn0 nmn0; do
		if ip -4 addr show "${interface_name}" >/dev/null 2>&1; then
			ip -4 addr show "${interface_name}" | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1
			return 0
		fi
	done
	return 1
}

if [[ -z "${VLLM_HOST_IP:-}" ]]; then
	VLLM_HOST_IP="$(detect_vllm_host_ip || true)"
	export VLLM_HOST_IP
fi

if [[ -z "${VLLM_HOST_IP:-}" ]]; then
	echo "Failed to determine VLLM_HOST_IP from hsn0/nmn0" >&2
	exit 1
fi

srun --environment=qwen3 --ntasks=1 \
	vllm serve "${MODEL_NAME}" \
	--host 0.0.0.0 \
	--port 8000 \
	--tensor-parallel-size 8 \
	--dtype bfloat16 \
	--max-model-len 32768 \
	--reasoning-parser qwen3 \
	--language-model-only &

until curl -sf http://localhost:8000/health >/dev/null; do
	sleep 2
done

GENERATOR_ARGS=(
	python /workspace/scripts/generate_chatml.py
	--dataset "${DATASET_NAME}"
	--split "${DATASET_SPLIT}"
	--out "${OUTPUT_PATH}"
	--api-base "${API_BASE}"
	--model "${MODEL_NAME}"
)

if [[ -n "${MAX_ROWS}" ]]; then
	GENERATOR_ARGS+=(--max-rows "${MAX_ROWS}")
fi

srun --environment=qwen3 --ntasks=1 "${GENERATOR_ARGS[@]}"
