#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=gsdg-qwen3-397b-clr
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

set -euo pipefail

STAGE_WORKSPACE="${STAGE_WORKSPACE:-1}"
STAGE_ROOT="${STAGE_ROOT:-${SCRATCH}/gsdg_workspace_${SLURM_JOB_ID}}"

CE_ENVIRONMENT="${CE_ENVIRONMENT:-qwen3-clariden}"
if [[ "${CE_ENVIRONMENT}" != "qwen3-clariden" ]]; then
	echo "This script is intended for the Clariden CE environment (qwen3-clariden)." >&2
	exit 1
fi

export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get
unset VLLM_NNODES || true
VLLM_HOST_IP="${VLLM_HOST_IP:-}"
export VLLM_HOST_IP
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING

DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to a glossAPI dataset name}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/synthetic_chatml_397b.jsonl}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
MAX_ROWS="${MAX_ROWS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MASTER_PORT="${MASTER_PORT:-29501}"
RAY_PORT="${RAY_PORT:-6379}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
REASONING_PARSER="${REASONING_PARSER:-}"

if [[ -z "${SLURM_JOB_NODELIST:-}" || -z "${SLURM_NNODES:-}" ]]; then
	echo "This script must run under a Slurm allocation." >&2
	exit 1
fi

EXPECTED_WORLD_SIZE=$((SLURM_NNODES * 4))
if (( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE != EXPECTED_WORLD_SIZE )); then
	echo "This launcher expects TP x PP (${TENSOR_PARALLEL_SIZE} x ${PIPELINE_PARALLEL_SIZE}) to match total allocated GPUs (${EXPECTED_WORLD_SIZE})." >&2
	echo "The current 397B Clariden path is 2 nodes x 4 GPUs with TP=8 and PP=1." >&2
	exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
if [[ -z "${MASTER_ADDR}" ]]; then
	echo "Failed to determine the head node from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}" >&2
	exit 1
fi

if [[ -z "${REASONING_PARSER}" ]]; then
	case "${MODEL_NAME}" in
		*Qwen3*|*qwen3*) REASONING_PARSER="qwen3" ;;
		*) REASONING_PARSER="" ;;
	esac
fi

export DATASET_NAME DATASET_SPLIT OUTPUT_PATH API_BASE MODEL_NAME MAX_ROWS
export MAX_MODEL_LEN MASTER_PORT TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE
export REASONING_PARSER MASTER_ADDR RAY_PORT

PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-}"
GENERATOR_ENTRYPOINT="/workspace/scripts/generate_chatml.py"
RAY_SERVE_ENTRYPOINT="/workspace/scripts/launch_vllm_serve_with_ray.py"
if [[ "${STAGE_WORKSPACE}" != "0" ]]; then
	rm -rf "${STAGE_ROOT}"
	mkdir -p "${STAGE_ROOT}"
	tar -C /users/p-skarvelis/GSDG -cz requirements.txt src scripts Readme.md TLTR.md Agents.md | tar -xz -C "${STAGE_ROOT}"
	PYTHONPATH_VALUE="${STAGE_ROOT}/src"
	GENERATOR_ENTRYPOINT="${STAGE_ROOT}/scripts/generate_chatml.py"
	RAY_SERVE_ENTRYPOINT="${STAGE_ROOT}/scripts/launch_vllm_serve_with_ray.py"
	echo "Staged workspace into ${STAGE_ROOT}" >&2
fi
export GENERATOR_ENTRYPOINT RAY_SERVE_ENTRYPOINT

echo "Using CE environment: ${CE_ENVIRONMENT}" >&2
echo "Using MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}" >&2
echo "Using TP=${TENSOR_PARALLEL_SIZE} PP=${PIPELINE_PARALLEL_SIZE} across ${SLURM_NNODES} node(s)" >&2

SRUN_EXPORT="ALL"
if [[ -n "${PYTHONPATH_VALUE}" ]]; then
	SRUN_EXPORT+=",PYTHONPATH=${PYTHONPATH_VALUE}"
fi
SRUN_EXPORT+=",MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},RAY_PORT=${RAY_PORT},GENERATOR_ENTRYPOINT=${GENERATOR_ENTRYPOINT},VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}"

srun --environment="${CE_ENVIRONMENT}" \
	--export="${SRUN_EXPORT}" \
	--ntasks-per-node=1 bash <<'INNER'
set -euo pipefail

. /opt/gsdg-venv/bin/activate

unset VLLM_NNODES || true
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING}"

detect_vllm_host_ip() {
	local route_target interface_name detected_ip
	if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
		mapfile -t _job_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
		if (( ${#_job_hosts[@]} > 1 )); then
			if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
				route_target="${_job_hosts[1]}"
			else
				route_target="${_job_hosts[0]}"
			fi
		fi
	fi

	if [[ -n "${route_target:-}" ]]; then
		detected_ip="$(ip -4 route get "${route_target}" 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "src") { print $(i + 1); exit }}' | head -n 1)"
		if [[ -n "${detected_ip}" ]]; then
			echo "${detected_ip}"
			return 0
		fi
	fi

	for interface_name in nmn0 hsn0 hs0 bond0 ib0; do
		if ip -4 addr show "${interface_name}" >/dev/null 2>&1; then
			detected_ip="$(ip -4 addr show "${interface_name}" | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1)"
			if [[ -n "${detected_ip}" ]]; then
				echo "${detected_ip}"
				return 0
			fi
		fi
	done

	detected_ip="$(getent ahostsv4 "$(hostname -s)" 2>/dev/null | awk 'NR == 1 { print $1 }')"
	if [[ -n "${detected_ip}" ]]; then
		echo "${detected_ip}"
		return 0
	fi

	detected_ip="$(hostname -I 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i != "127.0.0.1") { print $i; exit }}')"
	if [[ -n "${detected_ip}" ]]; then
		echo "${detected_ip}"
		return 0
	fi

	return 1
}

VLLM_HOST_IP="${VLLM_HOST_IP:-}"
if [[ -z "${VLLM_HOST_IP}" ]]; then
	VLLM_HOST_IP="$(detect_vllm_host_ip || true)"
	export VLLM_HOST_IP
fi

if [[ -z "${VLLM_HOST_IP}" ]]; then
	echo "Failed to determine VLLM_HOST_IP on node ${SLURMD_NODENAME:-unknown}" >&2
	echo "Visible interfaces:" >&2
	ip -o link show >&2 || true
	echo "Visible IPv4 addresses:" >&2
	ip -o -4 addr show >&2 || true
	exit 1
fi

echo "Node ${SLURM_NODEID}: VLLM_HOST_IP=${VLLM_HOST_IP}" >&2
echo "Node ${SLURM_NODEID}: VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}" >&2

SERVER_LOG="${PWD}/vllm-397b-node${SLURM_NODEID}.log"
RAY_LOG="${PWD}/ray-397b-node${SLURM_NODEID}.log"
RAY_HEAD_IP_FILE="${SCRATCH}/ray-head-ip-${SLURM_JOB_ID}.txt"
RAY_STOP_FILE="${SCRATCH}/ray-stop-${SLURM_JOB_ID}.flag"

cleanup() {
	touch "$RAY_STOP_FILE" || true
	if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
		kill "$server_pid" >/dev/null 2>&1 || true
		wait "$server_pid" >/dev/null 2>&1 || true
	fi
	ray stop -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for_file() {
	local path="$1"
	local timeout_secs="$2"
	local elapsed=0
	while [[ ! -f "$path" ]]; do
		if (( elapsed >= timeout_secs )); then
			return 1
		fi
		sleep 2
		elapsed=$((elapsed + 2))
	done
	return 0
}

if [[ "${SLURM_NODEID}" == "0" ]]; then
	rm -f "$RAY_HEAD_IP_FILE" "$RAY_STOP_FILE"
	echo "$VLLM_HOST_IP" >"$RAY_HEAD_IP_FILE"
	ray stop -f >/dev/null 2>&1 || true
	ray start --head --node-ip-address "$VLLM_HOST_IP" --port "$RAY_PORT" >"$RAY_LOG" 2>&1

	for _ in $(seq 1 120); do
		active_nodes="$(/opt/gsdg-venv/bin/python - <<'PY'
import ray
try:
    ray.init(address="auto", logging_level="ERROR")
    print(sum(1 for node in ray.nodes() if node.get("Alive")))
finally:
    if ray.is_initialized():
        ray.shutdown()
PY
		)"
		if [[ "$active_nodes" == "$SLURM_NNODES" ]]; then
			break
		fi
		sleep 5
	done

	if [[ "$active_nodes" != "$SLURM_NNODES" ]]; then
		echo "Ray cluster did not reach ${SLURM_NNODES} node(s)" >&2
		cat "$RAY_LOG" >&2 || true
		exit 1
	fi

	ray_serve_args=(
		/opt/gsdg-venv/bin/python "$RAY_SERVE_ENTRYPOINT"
		--model "$MODEL_NAME"
		--ray-address auto
		--host 0.0.0.0
		--port 8000
		--tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
		--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
		--dtype bfloat16
		--max-model-len "$MAX_MODEL_LEN"
		--language-model-only
	)
	if [[ -n "${REASONING_PARSER}" ]]; then
		ray_serve_args+=(--reasoning-parser "$REASONING_PARSER")
	fi
	RAY_ADDRESS="auto" "${ray_serve_args[@]}" >"$SERVER_LOG" 2>&1 &
	server_pid=$!

	health_url="${API_BASE%/v1}/health"
	for _ in $(seq 1 720); do
		if ! kill -0 "$server_pid" >/dev/null 2>&1; then
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
		/opt/gsdg-venv/bin/python "$GENERATOR_ENTRYPOINT"
		--dataset "$DATASET_NAME"
		--split "$DATASET_SPLIT"
		--out "$OUTPUT_PATH"
		--api-base "$API_BASE"
		--model "$MODEL_NAME"
	)
	if [[ -n "${MAX_ROWS}" ]]; then
		generator_args+=(--max-rows "$MAX_ROWS")
	fi

	"${generator_args[@]}"
	echo "Server log: $SERVER_LOG" >&2
else
	if ! wait_for_file "$RAY_HEAD_IP_FILE" 120; then
		echo "Timed out waiting for Ray head IP file" >&2
		exit 1
	fi
	RAY_HEAD_IP="$(cat "$RAY_HEAD_IP_FILE")"
	ray stop -f >/dev/null 2>&1 || true
	ray start --address "${RAY_HEAD_IP}:${RAY_PORT}" --node-ip-address "$VLLM_HOST_IP" >"$RAY_LOG" 2>&1
	while [[ ! -f "$RAY_STOP_FILE" ]]; do
		sleep 5
	done
fi
INNER