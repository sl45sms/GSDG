# Agents / Runbook (CSCS Alps → Bristen)

This repository is designed to run **only** on CSCS Alps, on the **Bristen** cluster, using:

- Slurm
- `uenv` for build-time Python tooling on Alps
- CSCS Container Engine (CE) with **EDF** files (`~/.edf/*.toml`)

The end goal is:

1. Build (or pull) a container image that can run **Qwen/Qwen3.5-397B-A17B**.
2. Run the model once (as a server) under Slurm on Bristen.
3. Stream rows from the **GlossAPI** HuggingFace datasets, extract the “best” text field(s), and generate **Greek** synthetic (question, answer) pairs.
4. Write output in **ChatML**-style JSONL.

Recommended operating split:

- Use `uenv` to get a modern Python and build or validate the repo on Alps.
- Use CE containers as the runtime authority for Slurm jobs.
- Do **not** try to layer `uenv` inside the running CE job.

References:

- Bristen: https://docs.cscs.ch/clusters/bristen/
- Container Engine / EDF: https://docs.cscs.ch/software/container-engine/
- CE quick start (`srun --environment=...`): https://docs.cscs.ch/software/container-engine/#step-2-launch-a-program
- uenv quick start: https://docs.cscs.ch/software/uenv/
- Building images on Alps (Podman + enroot import): https://docs.cscs.ch/build-install/containers/
- GlossAPI datasets: https://huggingface.co/glossAPI/datasets
- Qwen3.5-397B-A17B model card: https://huggingface.co/Qwen/Qwen3.5-397B-A17B

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

## 1) Strategy for running Qwen3.5-397B-A17B on Bristen

Qwen3.5-397B-A17B is a very large multimodal MoE model (397B total / 17B active). For practical inference on Bristen A100 nodes, the usual pattern is:

- Run an inference server (recommended: **vLLM** or **SGLang**).
- Use tensor parallelism (e.g. **8 GPUs** on a single node).
- For this repo, run the model in **text-only** mode and call the server locally (`http://localhost:8000/v1`).

The model card notes:

- Use the **latest Transformers main branch** for native Qwen3.5 support.
- Use **vLLM main/nightly**, not an older released build.
- Qwen3.5 does **not** use the old `/no_think` prompt switch; disable thinking via `chat_template_kwargs.enable_thinking=False` on the API request when you want strict JSON output.

Verified on this Bristen `normal` partition on 2026-03-20:

- Nodes expose **4 GPUs per node**, not 8. `sinfo` reports `gpu:4` and node `CfgTRES` confirms `gres/gpu=4`.
- A single-node 4-GPU smoke test with `tensor_parallel_size=4`, `--language-model-only`, and `--max-model-len 8192` still reached **CUDA OOM during model load** for `Qwen/Qwen3.5-397B-A17B`.
- Practical consequence: this model does **not** fit on one Bristen node in the validated configuration. A real serving run will need a **multi-node setup** on Bristen or a different cluster/resource shape.

---

## 2) Build-time Python via uenv

Use `uenv` to get a recent Python on Alps for local validation and packaging work. This keeps the runtime job clean while avoiding the old system Python.

Pull the image once:

```bash
uenv image pull prgenv-gnu/24.11:v1
```

Sanity check:

```bash
uenv run prgenv-gnu/24.11:v1 --view=default -- bash -lc 'python3 --version; which python3; nvcc --version | head -n 5'
```

Repo helper script:

```bash
./scripts/setup_uenv_python.sh
source .venv-uenv/bin/activate
export PYTHONPATH=$PWD/src
python scripts/generate_chatml.py --help
```

This step is for build-time and validation only. The final Slurm runtime should still go through CE.

---

## 3) Build a container image on Alps (Podman → enroot → .sqsh)

CSCS recommends building OCI images with Podman and then importing them into a SquashFS `.sqsh` that CE can run efficiently.

### 3.1 Podman storage (one-time)

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

The repo helper `scripts/build_container_on_alps.sh` will create this file automatically if it is missing.

### 3.2 Create a Containerfile (example)

Create a `Containerfile` (or `Dockerfile`) that installs:

- Python 3
- `vllm`
- `transformers>=4.51.0`
- `datasets`, `huggingface_hub`, `tokenizers`, `safetensors`

Notes:

- Prefer a CUDA-enabled base image (e.g. NVIDIA PyTorch images) if you want less dependency wrangling.
- Alternatively, you can use Ubuntu and pip-install everything, but be mindful of CUDA/cuDNN compatibility.

### 3.3 Build + import inside a compute allocation

Allocate a node (interactive):

```bash
srun -Aa0140 --partition=normal --pty bash
```

Build the OCI image:

```bash
./scripts/build_container_on_alps.sh
```

The helper script wraps both `podman build` and `enroot import`. It defaults to writing `${SCRATCH}/images/gsdg-qwen3_latest.sqsh`.

---

## 4) Create an EDF for the Qwen3 image

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
HF_TOKEN = "${HF_TOKEN}"

# Note: `HF_TOKEN` is expected to be set in the *host* environment before you run
# `srun`/`sbatch`. In this repo we keep it in a local `.env` file (do not commit it)
# and export it in your shell before submitting jobs:
#
#   set -a
#   source .env
#   set +a
```

Note: CE/Pyxis EDF files do not support shell-style default expansion like `${VAR:-default}`. Use plain `${VAR}` passthroughs or fixed literal values.

Tip: CE also supports pulling remote images automatically (e.g. `image = "nvcr.io#..."`), but for large ML stacks the `.sqsh` path is typically faster and more reproducible.

---

## 5) Prefetch model weights and dataset caches into `${SCRATCH}`

Before the main inference job, warm the HuggingFace caches once so later jobs can reuse them.

### 5.1 Slurm prefetch job

This repo provides a batch script that runs inside the CE environment and downloads:

- the Qwen3 model weights into `${HF_HOME}`
- one or more dataset caches into `${HF_DATASETS_CACHE}`

Example:

```bash
export PREFETCH_DATASETS=glossAPI/<dataset_name>
sbatch scripts/prefetch_hf_assets.sh
```

Multiple datasets:

```bash
export PREFETCH_DATASETS=glossAPI/<dataset_a>,glossAPI/<dataset_b>
sbatch scripts/prefetch_hf_assets.sh
```

Useful prefetch controls:

- `SKIP_MODEL=1` to prefetch datasets only.
- `SKIP_DATASETS=1` to prefetch the model only.
- `HF_HUB_DISABLE_XET=1` to disable the Xet transfer backend for troubleshooting.
- `PREFETCH_LOG_LEVEL=DEBUG` for more verbose Python-side logs.

Dataset-only example:

```bash
export PREFETCH_DATASETS=glossAPI/Sxolika_vivlia
export SKIP_MODEL=1
export HF_HUB_DISABLE_XET=1
sbatch scripts/prefetch_hf_assets.sh
```

Model-only example:

```bash
export SKIP_DATASETS=1
export HF_HUB_DISABLE_XET=1
sbatch scripts/prefetch_hf_assets.sh
```

The Python entry point is:

```bash
python /workspace/scripts/prefetch_hf_assets.py \
	--model Qwen/Qwen3.5-397B-A17B \
	--dataset glossAPI/<dataset_name> \
	--split train
```

This keeps the cache-warming step aligned with the final runtime environment because it uses the same CE EDF and the same `${SCRATCH}`-backed cache directories.

The default prefetch job footprint is intentionally small and queue-friendly: 1 task, 4 CPUs, 32 GB RAM, 0 GPUs. Override it with `sbatch --cpus-per-task=... --mem=...` if you need a larger model-only prefetch.

If a combined model+dataset prefetch appears stuck, split the problem first: run a dataset-only prefetch to validate dataset caching, then a model-only prefetch separately.

---

## 6) Run Qwen3 with vLLM on Bristen

### 6.1 Single-node, 8-GPU serving (recommended starting point)

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
	vllm serve Qwen/Qwen3.5-397B-A17B \
	--host 0.0.0.0 --port 8000 \
	--tensor-parallel-size 8 \
	--dtype bfloat16 \
	--max-model-len 32768 \
	--reasoning-parser qwen3 \
	--language-model-only
```

Submit:

```bash
sbatch run_vllm.sh
```

Notes:

- The model card also shows a vLLM variant with reasoning enabled. If you enable thinking/reasoning, make sure your downstream parsing strips `<think>...</think>`.
- For synthetic Q/A generation, prefer API-side non-thinking mode by sending `chat_template_kwargs={"enable_thinking": False}` rather than using `/no_think` in the prompt.
- `--language-model-only` is recommended here because the pipeline consumes text-only GlossAPI rows and does not need the vision stack loaded.

### 6.2 (Optional) Multi-node on Clariden

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

## 7) Generate Greek synthetic data from GlossAPI datasets

### 7.1 Recommended flow

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
	vllm serve Qwen/Qwen3.5-397B-A17B \
	--host 0.0.0.0 --port 8000 \
	--tensor-parallel-size 8 \
	--dtype bfloat16 \
	--max-model-len 32768 \
	--reasoning-parser qwen3 \
	--language-model-only &

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

### 7.2 Row → text extraction rule (robust defaults)

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

### 7.3 Prompt template (Greek Q/A)

Use a strict format so you can parse reliably:

- Input: a text excerpt from the dataset.
- Output: **exactly** a JSON object with keys `question` and `answer` in Greek.

Example user content:

```text
Με βάση το παρακάτω κείμενο, δημιούργησε ΜΙΑ ερώτηση κατανόησης και την απάντησή της στα Ελληνικά.

Κείμενο:
"""
{extracted_text}
"""

Απάντησε ΜΟΝΟ με JSON της μορφής:
{"question": "...", "answer": "..."}
```

For Qwen3.5 non-thinking mode, disable thinking on the API call rather than by embedding `/no_think` into the prompt.

### 7.4 Output format (ChatML JSONL)

Write one line per example:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"meta":{"dataset":"glossAPI/<name>","row_id":"..."}}
```

---

## 8) HuggingFace access + caching

For large assets (model weights + datasets):

- Set HF caches to `${SCRATCH}` (see EDF `[env]`).
- Provide `HF_TOKEN` via the host environment and pass-through via EDF.
	- In this repo, `HF_TOKEN` is stored in `.env`; export it before running `sbatch`/`srun`.

### 8.1 “Download once” behavior (weights reuse)

If you point HuggingFace caches to a **persistent filesystem** (typically `${SCRATCH}` on Alps/MLP) and mount it in the EDF (this runbook does), then:

- The first run downloads the model weights into the cache directory (usually under `${HF_HOME}/hub`).
- Subsequent jobs/container starts reuse the cached files, so weights are **not re-downloaded** unless the cache is deleted/evicted.

Practical tip: do a one-time prefetch job with `scripts/prefetch_hf_assets.sh` so later inference jobs start faster.

Caveats:

- Storage policies/cleanup on the filesystem can eventually remove old cache entries.
- Multiple first-time jobs started concurrently can still trigger overlapping downloads, but once the cache is populated later runs should reuse it.

---

## 9) Troubleshooting (Bristen / vLLM)

### 9.1 vLLM hangs on distributed init

On Bristen, vLLM (and/or torch distributed) can hang if it binds to an unusable IP/interface.

If you see hangs at/after torch distributed init, try pinning NCCL/GLOO to the high-speed interface and setting vLLM host IP:

```toml
[env]
NCCL_SOCKET_IFNAME = "hsn0"
GLOO_SOCKET_IFNAME = "hsn0"
VLLM_HOST_IP = "<IPv4 of hsn0>"
```

If `hsn0` is not present, try `nmn0`.

### 9.2 vLLM V1 multiprocessing deadlocks

If the server starts but deadlocks under load, try disabling V1 multiprocessing:

```toml
[env]
VLLM_ENABLE_V1_MULTIPROCESSING = "0"
```

---

## 10) What “done” looks like

- A `uenv`-based build-time Python workflow exists for local validation on Alps.
- A CE environment `qwen3` exists and points to a `.sqsh` image in `${SCRATCH}`.
- A prefetch job can populate `${SCRATCH}` caches with model weights and dataset artifacts.
- A Slurm job can start `vllm serve Qwen/Qwen3.5-397B-A17B` on 8×A100.
- A generator script can stream GlossAPI rows and write ChatML JSONL with Greek Q/A pairs.


