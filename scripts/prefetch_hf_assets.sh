#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=gsdg-prefetch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

CE_ENVIRONMENT="${CE_ENVIRONMENT:-}"
if [[ -z "${CE_ENVIRONMENT}" ]]; then
	cluster_hint="${SLURM_CLUSTER_NAME:-${SLURM_SUBMIT_HOST:-}}"
	if [[ -z "${cluster_hint}" ]]; then
		cluster_hint="$(hostname -s 2>/dev/null || hostname)"
	fi
	if [[ "${cluster_hint}" == *clariden* ]]; then
		CE_ENVIRONMENT="qwen3-clariden"
	else
		CE_ENVIRONMENT="qwen3"
	fi
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
PREFETCH_DATASETS="${PREFETCH_DATASETS:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
PREFETCH_LOG_LEVEL="${PREFETCH_LOG_LEVEL:-INFO}"

PREFETCH_ARGS=(
	python /workspace/scripts/prefetch_hf_assets.py
	--model "${MODEL_NAME}"
	--split "${DATASET_SPLIT}"
	--log-level "${PREFETCH_LOG_LEVEL}"
)

if [[ "${SKIP_MODEL}" == "1" ]]; then
	PREFETCH_ARGS+=(--skip-model)
fi

if [[ "${SKIP_DATASETS}" == "1" ]]; then
	PREFETCH_ARGS+=(--skip-datasets)
elif [[ -n "${PREFETCH_DATASETS}" ]]; then
	IFS=',' read -r -a dataset_array <<< "${PREFETCH_DATASETS}"
	for dataset_name in "${dataset_array[@]}"; do
		trimmed_name="${dataset_name// /}"
		if [[ -n "${trimmed_name}" ]]; then
			PREFETCH_ARGS+=(--dataset "${trimmed_name}")
		fi
	done
else
	PREFETCH_ARGS+=(--skip-datasets)
fi

echo "Prefetch settings: MODEL_NAME=${MODEL_NAME} DATASET_SPLIT=${DATASET_SPLIT} SKIP_MODEL=${SKIP_MODEL} SKIP_DATASETS=${SKIP_DATASETS} HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}" >&2
if [[ -n "${PREFETCH_DATASETS}" ]]; then
	echo "Prefetch datasets: ${PREFETCH_DATASETS}" >&2
fi

echo "Using CE environment: ${CE_ENVIRONMENT}" >&2

srun --environment="${CE_ENVIRONMENT}" --export=ALL,HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" --ntasks=1 "${PREFETCH_ARGS[@]}"
