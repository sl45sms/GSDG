#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=gsdg-qwen3-397b-async
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
# ============================================================================
# Async multi-node launcher for Qwen3.5-397B on Clariden.
#
# Replaces the bash-level parallelism of the older *_parallel.sh with a single
# async Python generator that keeps 64-128 requests in flight concurrently,
# saturating vLLM's continuous-batching pipeline.
#
# Usage (no more GENERATOR_PART_INDEX needed — just one job):
#
#   export PARQUET_FILES="${SCRATCH}/.../*.parquet"
#   export OUTPUT_PATH="${SCRATCH}/synthetic_chatml_397b.jsonl"
#   sbatch scripts/run_gsdg_qwen3_397b_clariden_async.sh
#
# Tunables:
#   GENERATOR_CONCURRENCY  – max in-flight requests (default 128)
#   GENERATOR_BATCH_SIZE   – rows per sort/flush batch (default 2000)
# ============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- HF_TOKEN (same as existing wrappers) ----------------------------------
if [[ -z "${HF_TOKEN:-}" && -f "${ROOT_DIR}/.env" ]]; then
	set -a
	# shellcheck disable=SC1091
	source "${ROOT_DIR}/.env"
	set +a
fi

# --- workspace staging (same as existing scripts) --------------------------
STAGE_WORKSPACE="${STAGE_WORKSPACE:-1}"
STAGE_ROOT="${STAGE_ROOT:-${SCRATCH}/gsdg_workspace_${SLURM_JOB_ID}}"

# --- CE environment --------------------------------------------------------
CE_ENVIRONMENT="${CE_ENVIRONMENT:-qwen3-clariden}"
if [[ "${CE_ENVIRONMENT}" != "qwen3-clariden" ]]; then
	echo "This script is intended for the Clariden CE environment (qwen3-clariden)." >&2
	exit 1
fi

# --- networking / NCCL / FI (validated Clariden settings) ------------------
export OCI_ANNOTATION_com__hooks__cxi__enabled=false
export SLURM_NETWORK=disable_rdzv_get
unset VLLM_NNODES || true
VLLM_HOST_IP="${VLLM_HOST_IP:-}"
export VLLM_HOST_IP
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-nmn0}"
export NCCL_SOCKET_IFNAME
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-nmn0}"
export GLOO_SOCKET_IFNAME
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_CROSS_NIC
FI_PROVIDER="${FI_PROVIDER:-cxi}"
export FI_PROVIDER
FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_DEFAULT_CQ_SIZE
FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-16384}"
export FI_CXI_DEFAULT_TX_SIZE
FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
export FI_CXI_DISABLE_HOST_REGISTER
FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"
export FI_CXI_RX_MATCH_MODE
FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
export FI_MR_CACHE_MONITOR
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING
VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM

# --- data & model knobs ----------------------------------------------------
DATASET_NAME="${DATASET_NAME:-}"
HF_PARQUET_REPO="${HF_PARQUET_REPO:-}"
PARQUET_FILES="${PARQUET_FILES:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/synthetic_chatml_397b.jsonl}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
MAX_ROWS="${MAX_ROWS:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MASTER_PORT="${MASTER_PORT:-29501}"
RAY_PORT="${RAY_PORT:-6379}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-}"
REASONING_PARSER="${REASONING_PARSER:-}"

# --- async generator knobs -------------------------------------------------
GENERATOR_CONCURRENCY="${GENERATOR_CONCURRENCY:-128}"
GENERATOR_BATCH_SIZE="${GENERATOR_BATCH_SIZE:-2000}"

# --- validation ------------------------------------------------------------
if [[ -z "${DATASET_NAME}" && -z "${PARQUET_FILES}" ]]; then
	echo "Set DATASET_NAME, or set PARQUET_FILES for local parquet input." >&2
	exit 1
fi

if [[ -n "${DATASET_NAME}" && -n "${PARQUET_FILES}" ]]; then
	echo "Set either DATASET_NAME or PARQUET_FILES, not both." >&2
	exit 1
fi

if [[ -n "${HF_PARQUET_REPO}" && -n "${DATASET_NAME}" ]]; then
	echo "HF_PARQUET_REPO can only be used together with PARQUET_FILES, not DATASET_NAME." >&2
	exit 1
fi

if [[ -z "${SLURM_JOB_NODELIST:-}" || -z "${SLURM_NNODES:-}" ]]; then
	echo "This script must run under a Slurm allocation." >&2
	exit 1
fi

# --- pipeline-parallel default (bf16 → 4 nodes, FP8 → 2 nodes) -------------
if [[ -z "${PIPELINE_PARALLEL_SIZE}" ]]; then
	if [[ "${MODEL_NAME}" == "Qwen/Qwen3.5-397B-A17B" && "${SLURM_NNODES}" -ge 4 ]]; then
		PIPELINE_PARALLEL_SIZE=2
	else
		PIPELINE_PARALLEL_SIZE=1
	fi
fi

EXPECTED_WORLD_SIZE=$((SLURM_NNODES * 4))
if (( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE != EXPECTED_WORLD_SIZE )); then
	echo "TP × PP (${TENSOR_PARALLEL_SIZE} × ${PIPELINE_PARALLEL_SIZE}) must match total GPUs (${EXPECTED_WORLD_SIZE})." >&2
	exit 1
fi

if [[ "${MODEL_NAME}" == "Qwen/Qwen3.5-397B-A17B" && "${EXPECTED_WORLD_SIZE}" -le 8 ]]; then
	echo "The bf16 397B checkpoint OOMs on 2 nodes / 8 GPUs during vLLM profile_run." >&2
	echo "Use the default FP8 checkpoint (Qwen/Qwen3.5-397B-A17B-FP8) or request 4 nodes / 16 GPUs." >&2
	exit 1
fi

# --- head node -------------------------------------------------------------
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
if [[ -z "${MASTER_ADDR}" ]]; then
	echo "Failed to determine head node from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}" >&2
	exit 1
fi

# --- reasoning parser auto-detect ------------------------------------------
if [[ -z "${REASONING_PARSER}" ]]; then
	case "${MODEL_NAME}" in
		*Qwen3*|*qwen3*) REASONING_PARSER="qwen3" ;;
		*) REASONING_PARSER="" ;;
	esac
fi

# --- auto-resume (same logic as existing scripts) --------------------------
RESUME_START_ROW=0
RESUME_RECORD_COUNT=0
RESUME_MAX_ROWS="${MAX_ROWS}"
if [[ "${AUTO_RESUME}" != "0" && -f "${OUTPUT_PATH}" ]]; then
	if ! command -v python3 >/dev/null 2>&1; then
		echo "python3 is required to inspect ${OUTPUT_PATH} for auto-resume." >&2
		exit 1
	fi

	mapfile -t resume_state < <(python3 - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
valid_lines = 0
last_index = None
original_size = path.stat().st_size
truncate_at = original_size

with path.open("rb") as handle:
	while True:
		line_start = handle.tell()
		raw_line = handle.readline()
		if not raw_line:
			break
		if not raw_line.strip():
			truncate_at = handle.tell()
			continue

		try:
			record = json.loads(raw_line.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError):
			truncate_at = line_start
			break

		truncate_at = handle.tell()
		valid_lines += 1

		meta = record.get("meta")
		if isinstance(meta, dict):
			value = meta.get("source_row_index")
			if isinstance(value, int):
				last_index = value
			elif isinstance(value, str):
				try:
					last_index = int(value)
				except ValueError:
					pass

if truncate_at < original_size:
	with path.open("r+b") as handle:
		handle.truncate(truncate_at)

start_row = last_index + 1 if last_index is not None else valid_lines
print(start_row)
print(valid_lines)
print(original_size - truncate_at)
PY
	)

	if (( ${#resume_state[@]} != 3 )); then
		echo "Failed to derive resume state from ${OUTPUT_PATH}." >&2
		exit 1
	fi

	RESUME_START_ROW="${resume_state[0]}"
	RESUME_RECORD_COUNT="${resume_state[1]}"
	RESUME_TRUNCATED_BYTES="${resume_state[2]}"

	if (( RESUME_TRUNCATED_BYTES > 0 )); then
		echo "Auto-resume truncated ${RESUME_TRUNCATED_BYTES} trailing byte(s) from ${OUTPUT_PATH}." >&2
	fi
fi

if [[ -n "${MAX_ROWS}" ]]; then
	if (( RESUME_START_ROW >= MAX_ROWS )); then
		echo "Existing output already covers the requested row window (start_row=${RESUME_START_ROW}, MAX_ROWS=${MAX_ROWS}). Nothing to do." >&2
		exit 0
	fi
	RESUME_MAX_ROWS="$((MAX_ROWS - RESUME_START_ROW))"
fi

# --- export everything for the inner srun ----------------------------------
export DATASET_NAME HF_PARQUET_REPO PARQUET_FILES DATASET_SPLIT OUTPUT_PATH API_BASE MODEL_NAME MAX_ROWS
export MAX_MODEL_LEN MASTER_PORT TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE
export REASONING_PARSER MASTER_ADDR RAY_PORT AUTO_RESUME RESUME_START_ROW RESUME_RECORD_COUNT RESUME_MAX_ROWS
export GENERATOR_CONCURRENCY GENERATOR_BATCH_SIZE

PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-}"
GENERATOR_ENTRYPOINT="/workspace/scripts/generate_chatml_async.py"
RAY_SERVE_ENTRYPOINT="/workspace/scripts/launch_vllm_serve_with_ray.py"
SITECUSTOMIZE_INSTALLER="/workspace/scripts/install_qwen35_gdn_sitecustomize.py"
if [[ "${STAGE_WORKSPACE}" != "0" ]]; then
	rm -rf "${STAGE_ROOT}"
	mkdir -p "${STAGE_ROOT}"
	tar -C /users/p-skarvelis/GSDG -cz requirements.txt src scripts Readme.md TLTR.md Agents.md | tar -xz -C "${STAGE_ROOT}"
	PYTHONPATH_VALUE="${STAGE_ROOT}/src"
	GENERATOR_ENTRYPOINT="${STAGE_ROOT}/scripts/generate_chatml_async.py"
	RAY_SERVE_ENTRYPOINT="${STAGE_ROOT}/scripts/launch_vllm_serve_with_ray.py"
	SITECUSTOMIZE_INSTALLER="${STAGE_ROOT}/scripts/install_qwen35_gdn_sitecustomize.py"
	echo "Staged workspace into ${STAGE_ROOT}" >&2
fi
export GENERATOR_ENTRYPOINT RAY_SERVE_ENTRYPOINT SITECUSTOMIZE_INSTALLER

echo "=== Async launcher ===" >&2
echo "CE environment: ${CE_ENVIRONMENT}" >&2
echo "Model: ${MODEL_NAME}  TP=${TENSOR_PARALLEL_SIZE}  PP=${PIPELINE_PARALLEL_SIZE}  nodes=${SLURM_NNODES}" >&2
echo "Concurrency: ${GENERATOR_CONCURRENCY}  batch-size: ${GENERATOR_BATCH_SIZE}" >&2
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}" >&2
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}  GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}  FI_PROVIDER=${FI_PROVIDER}" >&2
if [[ "${AUTO_RESUME}" != "0" ]]; then
	echo "Auto-resume start row: ${RESUME_START_ROW} (${RESUME_RECORD_COUNT} existing record(s))" >&2
	if [[ -n "${MAX_ROWS}" ]]; then
		echo "Auto-resume remaining MAX_ROWS window: ${RESUME_MAX_ROWS}" >&2
	fi
fi

# --- build srun export string ----------------------------------------------
SRUN_EXPORT="ALL"
if [[ -n "${PYTHONPATH_VALUE}" ]]; then
	SRUN_EXPORT+=",PYTHONPATH=${PYTHONPATH_VALUE}"
fi
SRUN_EXPORT+=",MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},RAY_PORT=${RAY_PORT}"
SRUN_EXPORT+=",GENERATOR_ENTRYPOINT=${GENERATOR_ENTRYPOINT},SITECUSTOMIZE_INSTALLER=${SITECUSTOMIZE_INSTALLER}"
SRUN_EXPORT+=",VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}"
SRUN_EXPORT+=",VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}"
SRUN_EXPORT+=",NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME},GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
SRUN_EXPORT+=",NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
SRUN_EXPORT+=",FI_PROVIDER=${FI_PROVIDER},FI_CXI_DEFAULT_CQ_SIZE=${FI_CXI_DEFAULT_CQ_SIZE}"
SRUN_EXPORT+=",FI_CXI_DEFAULT_TX_SIZE=${FI_CXI_DEFAULT_TX_SIZE}"
SRUN_EXPORT+=",FI_CXI_DISABLE_HOST_REGISTER=${FI_CXI_DISABLE_HOST_REGISTER}"
SRUN_EXPORT+=",FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE}"
SRUN_EXPORT+=",FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR}"
SRUN_EXPORT+=",RESUME_START_ROW=${RESUME_START_ROW},RESUME_MAX_ROWS=${RESUME_MAX_ROWS}"
SRUN_EXPORT+=",GENERATOR_CONCURRENCY=${GENERATOR_CONCURRENCY},GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE}"

# ============================================================================
# Inner script — runs on every allocated node inside the CE container
# ============================================================================
srun --environment="${CE_ENVIRONMENT}" \
	--export="${SRUN_EXPORT}" \
	--ntasks-per-node=1 bash <<'INNER'
set -euo pipefail

. /opt/gsdg-venv/bin/activate

# Bump file-descriptor limit for high concurrency (128+ connections).
ulimit -n 65536 2>/dev/null || true

export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC}"
export FI_PROVIDER="${FI_PROVIDER}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR}"

# --- helpers ---------------------------------------------------------------

maybe_enable_nccl_ofi() {
	if [[ -n "${NCCL_NET:-}" ]]; then
		return 0
	fi
	if find /usr /opt -name 'libnccl-net*.so*' 2>/dev/null | grep -qiE 'ofi|libfabric'; then
		export NCCL_NET='AWS Libfabric'
		echo "Node ${SLURM_NODEID}: enabling NCCL_NET=${NCCL_NET}" >&2
	fi
}

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

# --- detect host IP --------------------------------------------------------

VLLM_HOST_IP="${VLLM_HOST_IP:-}"
if [[ -z "${VLLM_HOST_IP}" ]]; then
	VLLM_HOST_IP="$(detect_vllm_host_ip || true)"
	export VLLM_HOST_IP
fi

if [[ -z "${VLLM_HOST_IP}" ]]; then
	echo "Failed to determine VLLM_HOST_IP on node ${SLURMD_NODENAME:-unknown}" >&2
	exit 1
fi

echo "Node ${SLURM_NODEID}: VLLM_HOST_IP=${VLLM_HOST_IP}" >&2
echo "Node ${SLURM_NODEID}: VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}" >&2
echo "Node ${SLURM_NODEID}: VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}" >&2
echo "Node ${SLURM_NODEID}: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} FI_PROVIDER=${FI_PROVIDER}" >&2

maybe_enable_nccl_ofi

# Install the GDN fallback patch (no-op when FlashInfer is present).
/opt/gsdg-venv/bin/python "$SITECUSTOMIZE_INSTALLER"

# --- paths -----------------------------------------------------------------
SERVER_LOG="${PWD}/vllm-397b-node${SLURM_NODEID}.log"
RAY_LOG="${PWD}/ray-397b-node${SLURM_NODEID}.log"
RAY_HEAD_IP_FILE="${SCRATCH}/ray-head-ip-${SLURM_JOB_ID}.txt"
RAY_STOP_FILE="${SCRATCH}/ray-stop-${SLURM_JOB_ID}.flag"
RAY_SESSION_ARCHIVE_DIR="${SCRATCH}/ray-session-${SLURM_JOB_ID}-node${SLURM_NODEID}"

cleanup() {
	touch "$RAY_STOP_FILE" || true
	if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
		kill "$server_pid" >/dev/null 2>&1 || true
		wait "$server_pid" >/dev/null 2>&1 || true
	fi
	if [[ -d /tmp/ray/session_latest ]]; then
		rm -rf "$RAY_SESSION_ARCHIVE_DIR" || true
		mkdir -p "$RAY_SESSION_ARCHIVE_DIR" || true
		cp -a /tmp/ray/session_latest/. "$RAY_SESSION_ARCHIVE_DIR"/ >/dev/null 2>&1 || true
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

# ============================================================================
# Node 0: Ray head + vLLM server + async generator
# ============================================================================
if [[ "${SLURM_NODEID}" == "0" ]]; then
	rm -f "$RAY_HEAD_IP_FILE" "$RAY_STOP_FILE"
	echo "$VLLM_HOST_IP" >"$RAY_HEAD_IP_FILE"
	ray stop -f >/dev/null 2>&1 || true
	ray start --head --node-ip-address "$VLLM_HOST_IP" --port "$RAY_PORT" >"$RAY_LOG" 2>&1

	# Wait for all Ray workers to join.
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

	# --- launch vLLM server (Ray-backed) -----------------------------------
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

	# --- healthcheck loop --------------------------------------------------
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

	# --- async generator (SINGLE process, high concurrency) ----------------
	generator_args=(
		/opt/gsdg-venv/bin/python "$GENERATOR_ENTRYPOINT"
		--split "$DATASET_SPLIT"
		--start-row "$RESUME_START_ROW"
		--out "$OUTPUT_PATH"
		--api-base "$API_BASE"
		--model "$MODEL_NAME"
		--concurrency "$GENERATOR_CONCURRENCY"
		--batch-size "$GENERATOR_BATCH_SIZE"
		--timeout-seconds 600
	)

	if [[ -n "${DATASET_NAME:-}" ]]; then
		generator_args+=(--dataset "$DATASET_NAME")
	fi
	if [[ -n "${HF_PARQUET_REPO:-}" ]]; then
		generator_args+=(--hf-parquet-repo "$HF_PARQUET_REPO")
	fi
	if [[ -n "${PARQUET_FILES:-}" ]]; then
		IFS=, read -r -a parquet_patterns <<< "$PARQUET_FILES"
		for parquet_pattern in "${parquet_patterns[@]}"; do
			trimmed_pattern="${parquet_pattern#"${parquet_pattern%%[![:space:]]*}"}"
			trimmed_pattern="${trimmed_pattern%"${trimmed_pattern##*[![:space:]]}"}"
			if [[ -n "${trimmed_pattern}" ]]; then
				generator_args+=(--parquet-file "$trimmed_pattern")
			fi
		done
	fi
	if [[ -n "${RESUME_MAX_ROWS}" ]]; then
		generator_args+=(--max-rows "$RESUME_MAX_ROWS")
	fi

	echo "Starting async generator with concurrency=${GENERATOR_CONCURRENCY} batch_size=${GENERATOR_BATCH_SIZE}" >&2
	"${generator_args[@]}"
	echo "Async generator finished.  Server log: $SERVER_LOG" >&2

# ============================================================================
# Other nodes: Ray worker — idle until head signals stop
# ============================================================================
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
