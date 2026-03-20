#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=qwen3-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
HEALTH_URL="${API_BASE%/v1}/health"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

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

echo "Using VLLM_HOST_IP=${VLLM_HOST_IP}" >&2

srun --environment=qwen3 --ntasks=1 \
	vllm serve "${MODEL_NAME}" \
	--host 0.0.0.0 \
	--port 8000 \
	--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
	--dtype bfloat16 \
	--max-model-len "${MAX_MODEL_LEN}" \
	--reasoning-parser "${REASONING_PARSER}" \
	--language-model-only \
	> smoke-vllm-server.log 2>&1 &
server_pid=$!

cleanup() {
	if kill -0 "${server_pid}" >/dev/null 2>&1; then
		kill "${server_pid}" || true
		wait "${server_pid}" || true
	fi
}
trap cleanup EXIT

for _ in $(seq 1 180); do
	if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
		echo "vLLM server exited before becoming healthy" >&2
		wait "${server_pid}"
		exit 1
	fi
	if curl -sf "${HEALTH_URL}" >/dev/null; then
		break
	fi
	sleep 5
done

curl -sf "${HEALTH_URL}" >/dev/null

srun --environment=qwen3 --ntasks=1 python - <<'PY'
import json
import os
import requests

payload = {
	"model": os.environ["MODEL_NAME"],
    "messages": [
        {"role": "system", "content": "Απαντάς σύντομα και ακριβώς."},
        {"role": "user", "content": "Πες μόνο: ok"},
    ],
    "max_tokens": 32,
    "temperature": 0.0,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
    json=payload,
    timeout=120,
)
response.raise_for_status()
body = response.json()
print(json.dumps(body, ensure_ascii=False))
PY