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

# Defaults for a Clariden 4xGH200 node.
CE_ENVIRONMENT="${CE_ENVIRONMENT:-qwen3-clariden}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATASET_NAME="${DATASET_NAME:-glossAPI/Sxolika_vivlia}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_ROWS="${MAX_ROWS:-}"

OUT_BASENAME="${OUT_BASENAME:-synthetic_${DATASET_NAME//\//_}_${MODEL_NAME//\//_}_${SLURM_JOB_ID:-manual}.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/${OUT_BASENAME}}"

echo "Submitting full run: DATASET=${DATASET_NAME} SPLIT=${DATASET_SPLIT} MODEL=${MODEL_NAME} TP=${TENSOR_PARALLEL_SIZE} MAX_MODEL_LEN=${MAX_MODEL_LEN} OUT=${OUTPUT_PATH} CE_ENVIRONMENT=${CE_ENVIRONMENT}" >&2

export CE_ENVIRONMENT MODEL_NAME TENSOR_PARALLEL_SIZE MAX_MODEL_LEN DATASET_NAME DATASET_SPLIT OUTPUT_PATH
if [[ -n "${MAX_ROWS}" ]]; then
	export MAX_ROWS
fi

# Use the latest repo sources without rebuilding the .sqsh
export STAGE_WORKSPACE=1

job_submit_out="$(sbatch --parsable --export=ALL scripts/run_gsdg_qwen3.sh)"
job_id="${job_submit_out%%;*}"
job_id="${job_id%%.*}"

echo "Submitted batch job ${job_id}" >&2

wait_for_job() {
	local jid="$1"
	local state_line=""
	local state=""
	local exit_code=""

	# Wait until the job disappears from the queue.
	while squeue -j "${jid}" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "${jid}" -h -o '%.2t' 2>/dev/null || true)" ]]; do
		sleep 10
	done

	# sacct can lag a bit; poll until it reports something.
	for _ in $(seq 1 60); do
		state_line="$(sacct -j "${jid}" --format=State,ExitCode -n -P 2>/dev/null | head -n 1 || true)"
		if [[ -n "${state_line}" ]]; then
			break
		fi
		sleep 5
	done

	state="${state_line%%|*}"
	exit_code="${state_line#*|}"

	if [[ "${state}" != "COMPLETED" || "${exit_code}" != "0:0" ]]; then
		echo "Job ${jid} did not complete successfully: state=${state:-unknown} exit=${exit_code:-unknown}" >&2
		if [[ -f "slurm-${jid}.out" ]]; then
			tail -n 200 "slurm-${jid}.out" >&2 || true
		fi
		return 1
	fi
}

wait_for_job "${job_id}"

mkdir -p outputs
src_path="${OUTPUT_PATH}"

dst_base="$(basename "${src_path}")"
dst_name="${dst_base%.jsonl}_job${job_id}.jsonl"
dst_path="outputs/${dst_name}"

cp -f "${src_path}" "${dst_path}"
echo "Copied output to ${dst_path}" >&2
