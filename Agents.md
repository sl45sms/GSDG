# Agents / Runbook (CSCS Alps → Bristen)

This repository is designed to run **only** on CSCS Alps, on the **Bristen** cluster, using:

- Slurm
- CSCS Container Engine (CE) with **EDF** files (`~/.edf/*.toml`)

The end goal is:

1. Build (or pull) a container image that can run **Qwen/Qwen3-235B-A22B**.
2. Run the model once (as a server) under Slurm on Bristen.
3. Stream rows from the **GlossAPI** HuggingFace datasets, extract the “best” text field(s), and generate **Greek** synthetic (question, answer) pairs.
4. Write output in **ChatML**-style JSONL.

References:

- Bristen: https://docs.cscs.ch/clusters/bristen/
- Container Engine / EDF: https://docs.cscs.ch/software/container-engine/
- CE quick start (`srun --environment=...`): https://docs.cscs.ch/software/container-engine/#step-2-launch-a-program
- Building images on Alps (Podman + enroot import): https://docs.cscs.ch/build-install/containers/
- GlossAPI datasets: https://huggingface.co/glossAPI/datasets
- Qwen3-235B-A22B model card: https://huggingface.co/Qwen/Qwen3-235B-A22B

---

## 0) Sanity check (known-good)

You already have `~/.edf/test.toml`:

```toml
image = "library/ubuntu:24.04"
mounts = ["${SCRATCH}:${SCRATCH}"]
workdir = "${SCRATCH}"
```

Works via:

```bash
srun -Aa0140 --environment=test echo "Its Alive"
```

---

## 1) Strategy for running Qwen3-235B-A22B on Bristen

Qwen3-235B-A22B is a very large MoE model (235B total / 22B active). For practical inference on Bristen A100 nodes, the usual pattern is:

- Run an inference server (recommended: **vLLM** or **SGLang**).
- Use tensor parallelism (e.g. **8 GPUs** on a single node).
- Stream dataset rows and call the server locally (`http://localhost:8000/v1`).

The model card notes:

- Use **recent Transformers**: `transformers>=4.51.0` (older versions can error with `KeyError: 'qwen3_moe'`).
- Deployment options include `vllm>=0.8.5`.

---

## 2) Build a container image on Alps (Podman → enroot → .sqsh)

CSCS recommends building OCI images with Podman and then importing them into a SquashFS `.sqsh` that CE can run efficiently.

### 2.1 Podman storage (one-time)

Create:

```bash
mkdir -p $HOME/.config/containers
cat > $HOME/.config/containers/storage.conf <<'EOF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF
```

This uses `/dev/shm` (tmpfs), so the built image disappears when the allocation ends; you must **push** or **import** it before exiting the allocation.

### 2.2 Create a Containerfile (example)

Create a `Containerfile` (or `Dockerfile`) that installs:

- Python 3
- `vllm`
- `transformers>=4.51.0`
- `datasets`, `huggingface_hub`, `tokenizers`, `safetensors`

Notes:

- Prefer a CUDA-enabled base image (e.g. NVIDIA PyTorch images) if you want less dependency wrangling.
- Alternatively, you can use Ubuntu and pip-install everything, but be mindful of CUDA/cuDNN compatibility.

### 2.3 Build + import inside a compute allocation

Allocate a node (interactive):

```bash
srun -Aa0140 --partition=normal --pty bash
```

Build the OCI image:

```bash
podman build -t gsdg-qwen3:latest .
```

Pick a target directory on Lustre and tune striping (recommended by CSCS for large `.sqsh` files):

```bash
mkdir -p ${SCRATCH}/images
# optional: lfs setstripe -c <N> ${SCRATCH}/images
```

Import into SquashFS for CE (must be in the *same* allocation as the build):

```bash
enroot import -x mount -o ${SCRATCH}/images/gsdg-qwen3_latest.sqsh podman://gsdg-qwen3:latest
```

---

## 3) Create an EDF for the Qwen3 image

Create `~/.edf/qwen3.toml` pointing to the `.sqsh` you imported:

```toml
image = "${SCRATCH}/images/gsdg-qwen3_latest.sqsh"
mounts = ["${SCRATCH}:${SCRATCH}"]
workdir = "${SCRATCH}"

[env]
# Keep HF cache in SCRATCH
HF_HOME = "${SCRATCH}/hf"
TRANSFORMERS_CACHE = "${SCRATCH}/hf"
HF_DATASETS_CACHE = "${SCRATCH}/hf_datasets"

# To download from the Hub inside the job:
HF_TOKEN = "${HF_TOKEN:-}"

# Note: `HF_TOKEN` is expected to be set in the *host* environment before you run
# `srun`/`sbatch`. In this repo we keep it in a local `.env` file (do not commit it)
# and export it in your shell before submitting jobs:
#
#   set -a
#   source .env
#   set +a
```

Tip: CE also supports pulling remote images automatically (e.g. `image = "nvcr.io#..."`), but for large ML stacks the `.sqsh` path is typically faster and more reproducible.

---

## 4) Run Qwen3 with vLLM on Bristen

### 4.1 Single-node, 8-GPU serving (recommended starting point)

Create a Slurm script (example) that:

- Requests 1 node, 8 GPUs
- Starts a local OpenAI-compatible vLLM server

```bash
#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=qwen3-vllm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail

# Start the server inside the CE environment.
srun --environment=qwen3 \
	vllm serve Qwen/Qwen3-235B-A22B \
	--host 0.0.0.0 --port 8000 \
	--tensor-parallel-size 8 \
	--dtype bfloat16 \
	--max-model-len 32768
```

Submit:

```bash
sbatch run_vllm.sh
```

Notes:

- The model card also shows a vLLM variant with reasoning enabled. If you enable thinking/reasoning, make sure your downstream parsing strips `<think>...</think>`.
- If you do not need reasoning traces for synthetic Q/A generation, prefer **non-thinking mode** prompts (`/no_think`) to reduce verbosity/cost.

### 4.2 (Optional) Multi-node on Clariden

If you need more memory/throughput than a single node, run multi-node on **Clariden** (https://docs.cscs.ch/clusters/clariden/), which provides **GH200** nodes.

Key Clariden differences vs Bristen:

- Clariden nodes have **4 GPUs per node** (GH200). To get 8-way tensor parallelism you typically request **2 nodes**.
- Clariden has `normal` and `debug` partitions (see the Clariden docs for current limits).

Multi-node vLLM setups are more sensitive to networking. Expect to set explicit networking env vars (NCCL/GLOO) on Alps and, if you build your own image, to consider CE hooks / optimized base images for best interconnect performance.

Minimal Slurm resource shape (example, adjust as needed):

```bash
#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=qwen3-vllm-mn
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail

# NOTE: vLLM multi-node launch details vary by vLLM version and backend.
# Follow vLLM’s official multi-node guidance, then run via CE:
# srun --environment=qwen3 <vllm multi-node launcher ...>
```

For Clariden, also see the general Slurm guidance for GH200 nodes: https://docs.cscs.ch/running/slurm/#nvidia-gh200-gpu-nodes

---

## 5) Generate Greek synthetic data from GlossAPI datasets

### 5.1 Recommended flow

1. Start vLLM server on the allocated node.
2. In the same allocation, run a generator script that:
	 - Loads a chosen `glossAPI/<dataset_name>` via `datasets`
	 - Extracts a text payload from each row (datasets vary)
	 - Calls the local OpenAI-compatible endpoint
	 - Writes one JSONL record per row in ChatML format

Because CE shares one container per node for all tasks on that node, you can use separate `srun` commands for “server” and “client” within the same batch script if you want.

#### One-job example (server + generator)

This pattern runs everything in one job allocation on a single node:

```bash
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

# 1) Start vLLM server in the background
srun --environment=qwen3 --ntasks=1 \
	vllm serve Qwen/Qwen3-235B-A22B \
	--host 0.0.0.0 --port 8000 \
	--tensor-parallel-size 8 \
	--dtype bfloat16 \
	--max-model-len 32768 &

# 2) Wait until the health endpoint is up
until curl -sf http://localhost:8000/health >/dev/null; do
	sleep 2
done

# 3) Run the generator (replace with your script)
# Example CLI (you will implement this script in this repo):
srun --environment=qwen3 --ntasks=1 \
	python your_generator.py \
		--dataset glossAPI/<dataset_name> \
		--split train \
		--out ${SCRATCH}/synthetic_chatml.jsonl \
		--api-base http://localhost:8000/v1
```

If you prefer two `srun` steps with different resource shapes (e.g. more CPUs for the generator), split them and adjust `--cpus-per-task`.

### 5.2 Row → text extraction rule (robust defaults)

GlossAPI datasets are heterogeneous. Use this simple, robust approach:

- Prefer a single field if present: `text`, `content`, `document`, `body`, `paragraph`, `sentence`, `ocr`, `transcript`.
- Else concatenate all string-like fields (ignore IDs, URLs, numeric metadata).
- Truncate very long texts to a safe budget (e.g. 2k–6k characters) to keep inference stable.

Always inspect schema first:

```python
from datasets import load_dataset
ds = load_dataset("glossAPI/<dataset_name>", split="train")
print(ds.column_names)
```

### 5.3 Prompt template (Greek Q/A)

Use a strict format so you can parse reliably:

- Input: a text excerpt from the dataset.
- Output: **exactly** a JSON object with keys `question` and `answer` in Greek.

Example user content (include `/no_think`):

```text
/no_think
Με βάση το παρακάτω κείμενο, δημιούργησε ΜΙΑ ερώτηση κατανόησης και την απάντησή της στα Ελληνικά.

Κείμενο:
"""
{extracted_text}
"""

Απάντησε ΜΟΝΟ με JSON της μορφής:
{"question": "...", "answer": "..."}
```

### 5.4 Output format (ChatML JSONL)

Write one line per example:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"meta":{"dataset":"glossAPI/<name>","row_id":"..."}}
```

---

## 6) HuggingFace access + caching

For large assets (model weights + datasets):

- Set HF caches to `${SCRATCH}` (see EDF `[env]`).
- Provide `HF_TOKEN` via the host environment and pass-through via EDF.
	- In this repo, `HF_TOKEN` is stored in `.env`; export it before running `sbatch`/`srun`.

### 6.1 “Download once” behavior (weights reuse)

If you point HuggingFace caches to a **persistent filesystem** (typically `${SCRATCH}` on Alps/MLP) and mount it in the EDF (this runbook does), then:

- The first run downloads the model weights into the cache directory (usually under `${HF_HOME}/hub`).
- Subsequent jobs/container starts reuse the cached files, so weights are **not re-downloaded** unless the cache is deleted/evicted.

Practical tip: do a one-time “prefetch” job so later inference jobs start faster (e.g., a tiny Python script calling `huggingface_hub.snapshot_download(...)`, or an equivalent CLI download).

Caveats:

- Storage policies/cleanup on the filesystem can eventually remove old cache entries.
- Multiple first-time jobs started concurrently can still trigger overlapping downloads, but once the cache is populated later runs should reuse it.

---

## 7) Troubleshooting (Bristen / vLLM)

### 7.1 vLLM hangs on distributed init

On Bristen, vLLM (and/or torch distributed) can hang if it binds to an unusable IP/interface.

If you see hangs at/after torch distributed init, try pinning NCCL/GLOO to the high-speed interface and setting vLLM host IP:

```toml
[env]
NCCL_SOCKET_IFNAME = "hsn0"
GLOO_SOCKET_IFNAME = "hsn0"
VLLM_HOST_IP = "<IPv4 of hsn0>"
```

If `hsn0` is not present, try `nmn0`.

### 7.2 vLLM V1 multiprocessing deadlocks

If the server starts but deadlocks under load, try disabling V1 multiprocessing:

```toml
[env]
VLLM_ENABLE_V1_MULTIPROCESSING = "0"
```

---

## 8) What “done” looks like

- A CE environment `qwen3` exists and points to a `.sqsh` image in `${SCRATCH}`.
- A Slurm job can start `vllm serve Qwen/Qwen3-235B-A22B` on 8×A100.
- A generator script can stream GlossAPI rows and write ChatML JSONL with Greek Q/A pairs.


