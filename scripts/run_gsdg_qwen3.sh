#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=gsdg-qwen3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
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

# Clariden Enroot hook 89-libfabric-cxi.sh bind-mounts host libfabric into the
# container, which can break torch import for this image due to MPI/libfabric
# symbol-version mismatch. vLLM doesn't require CXI/libfabric for single-node
# serving, so disable the hook.
if [[ "${CE_ENVIRONMENT}" == "qwen3-clariden" ]]; then
	export OCI_ANNOTATION_com__hooks__cxi__enabled=false
	export SLURM_NETWORK=disable_rdzv_get
fi

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to a glossAPI dataset name}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/synthetic_chatml.jsonl}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
MAX_ROWS="${MAX_ROWS:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
REASONING_PARSER="${REASONING_PARSER:-}"

# Only enable the Qwen3 reasoning parser for Qwen3-* models.
if [[ -z "${REASONING_PARSER}" ]]; then
	case "${MODEL_NAME}" in
		*Qwen3*|*qwen3*) REASONING_PARSER="qwen3" ;;
		*) REASONING_PARSER="" ;;
	esac
fi

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

echo "Using CE environment: ${CE_ENVIRONMENT}" >&2
echo "Using VLLM_HOST_IP=${VLLM_HOST_IP}" >&2

PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-}"
GENERATOR_ENTRYPOINT="/workspace/scripts/generate_chatml.py"
if [[ "${STAGE_WORKSPACE}" != "0" ]]; then
	rm -rf "${STAGE_ROOT}"
	mkdir -p "${STAGE_ROOT}"
	tar -C /users/p-skarvelis/GSDG -cz requirements.txt src scripts Readme.md TLTR.md Agents.md | tar -xz -C "${STAGE_ROOT}"
	PYTHONPATH_VALUE="${STAGE_ROOT}/src"
	GENERATOR_ENTRYPOINT="${STAGE_ROOT}/scripts/generate_chatml.py"
	echo "Staged workspace into ${STAGE_ROOT}" >&2
fi

SRUN_EXPORT="ALL"
if [[ -n "${PYTHONPATH_VALUE}" ]]; then
	SRUN_EXPORT+=",PYTHONPATH=${PYTHONPATH_VALUE}"
fi

# Run server + health + generation inside a single Slurm step.
srun --environment="${CE_ENVIRONMENT}" \
	--export="${SRUN_EXPORT}" \
	--ntasks=1 bash -lc '
	set -euo pipefail
	. /opt/gsdg-venv/bin/activate

	SERVER_LOG="${PWD}/vllm-server.log"

	args=(serve "$MODEL_NAME" \
		--host 0.0.0.0 \
		--port 8000 \
		--tensor-parallel-size "'"${TENSOR_PARALLEL_SIZE}"'" \
		--dtype bfloat16 \
		--max-model-len "'"${MAX_MODEL_LEN}"'" \
		--language-model-only)
	if [[ -n "${REASONING_PARSER:-}" ]]; then
		args+=(--reasoning-parser "${REASONING_PARSER}")
	fi

	/opt/gsdg-venv/bin/vllm "${args[@]}" >"$SERVER_LOG" 2>&1 &
	server_pid=$!

	cleanup() {
		if kill -0 "${server_pid}" >/dev/null 2>&1; then
			kill "${server_pid}" || true
			wait "${server_pid}" || true
		fi
	}
	trap cleanup EXIT

	health_url="'"${API_BASE%/v1}"'/health"
	for _ in $(seq 1 360); do
		if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
			echo "vLLM server exited before becoming healthy" >&2
			tail -n 200 "$SERVER_LOG" >&2 || true
			exit 1
		fi
		if curl -sf "$health_url" >/dev/null; then
			break
		fi
		sleep 2
	done
	curl -sf "$health_url" >/dev/null

	generator_args=(
		/opt/gsdg-venv/bin/python "'"${GENERATOR_ENTRYPOINT}"'" \
		--dataset "'"${DATASET_NAME}"'" \
		--split "'"${DATASET_SPLIT}"'" \
		--out "'"${OUTPUT_PATH}"'" \
		--api-base "'"${API_BASE}"'" \
		--model "'"${MODEL_NAME}"'")
	if [[ -n "'"${MAX_ROWS}"'" ]]; then
		generator_args+=(--max-rows "'"${MAX_ROWS}"'")
	fi

	"${generator_args[@]}"

	echo "Server log: $SERVER_LOG" >&2
'
