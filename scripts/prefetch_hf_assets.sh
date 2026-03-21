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

STAGE_WORKSPACE="${STAGE_WORKSPACE:-1}"
STAGE_ROOT="${STAGE_ROOT:-${SCRATCH}/gsdg_workspace_${SLURM_JOB_ID}}"

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

# Clariden: avoid CXI rendezvous warnings and keep behavior consistent with
# other Clariden scripts.
if [[ "${CE_ENVIRONMENT}" == "qwen3-clariden" ]]; then
	export SLURM_NETWORK=disable_rdzv_get
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
PREFETCH_DATASETS="${PREFETCH_DATASETS:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
PREFETCH_LOG_LEVEL="${PREFETCH_LOG_LEVEL:-INFO}"

PYTHON_BIN="${PYTHON_BIN:-/opt/gsdg-venv/bin/python}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-/workspace/src}"

PREFETCH_ENTRYPOINT="/workspace/scripts/prefetch_hf_assets.py"

if [[ "${STAGE_WORKSPACE}" != "0" ]]; then
	# The CE container image typically bakes the repo into /workspace. To avoid
	# rebuilding the .sqsh for small Python changes, stage the current workspace
	# into $SCRATCH (mounted into the container) and point PYTHONPATH at it.
	rm -rf "${STAGE_ROOT}"
	mkdir -p "${STAGE_ROOT}"
	tar -C /users/p-skarvelis/GSDG -cz requirements.txt src scripts Readme.md TLTR.md Agents.md | tar -xz -C "${STAGE_ROOT}"
	PYTHONPATH_VALUE="${STAGE_ROOT}/src"
	PREFETCH_ENTRYPOINT="${STAGE_ROOT}/scripts/prefetch_hf_assets.py"
fi

PREFETCH_ARGS=(
	"${PYTHON_BIN}" "${PREFETCH_ENTRYPOINT}"
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

srun --environment="${CE_ENVIRONMENT}" \
	--export=ALL,HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}",PYTHONPATH="${PYTHONPATH_VALUE}" \
	--ntasks=1 "${PREFETCH_ARGS[@]}"
