# Agents / Runbook (CSCS Alps → Bristen + Clariden)

This repository is designed to run on CSCS Alps, on **Bristen** (A100 / x86_64) and **Clariden** (GH200 / aarch64), using:

- Slurm
- `uenv` for build-time Python tooling on Alps
- CSCS Container Engine (CE) with **EDF** files (`~/.edf/*.toml`)

The end goal is:

1. Build (or pull) a container image that can run **Qwen/Qwen3-32B** and **Qwen/Qwen3.5-397B-A17B**.
2. Run either model under Slurm on Bristen or Clariden, with the right resource shape for that model.
3. Stream rows from the **GlossAPI** HuggingFace datasets, extract the “best” text field(s), and generate **Greek** synthetic (question, answer) pairs.
4. Write output in **ChatML**-style JSONL.

Recommended operating split:

- Use `uenv` to get a modern Python and build or validate the repo on Alps.
- Use CE containers as the runtime authority for Slurm jobs.
- Do **not** try to layer `uenv` inside the running CE job.

References:

- Bristen: https://docs.cscs.ch/clusters/bristen/
- Clariden: https://docs.cscs.ch/clusters/clariden/
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

## 1) Model and cluster strategy

Qwen3.5-397B-A17B is a very large multimodal MoE model (397B total / 17B active). For practical inference, the usual pattern is:

- Run an inference server (recommended: **vLLM** or **SGLang**).
- Use tensor parallelism across **8 GPUs total**.
	- On clusters with 8 GPUs per node, that can be a single node.
	- On **Clariden**, which has **4 GPUs per node**, that means **at least 2 nodes** for an 8-way tensor-parallel starting point.
- For this repo, run the model in **text-only** mode and call the server locally (`http://localhost:8000/v1`).

The model card notes:

- Use the **latest Transformers main branch** for native Qwen3.5 support.
- Use **vLLM main/nightly**, not an older released build.
- Qwen3.5 does **not** use the old `/no_think` prompt switch; disable thinking via `chat_template_kwargs.enable_thinking=False` on the API request when you want strict JSON output.

Practical split for this repo:

- **Qwen/Qwen3-32B on Clariden**: validated **single-node** path on **1 GH200 node / 4 GPUs** with `tensor_parallel_size=4`.
- **Qwen/Qwen3.5-397B-A17B on Clariden**: requires a **multi-node** run because Clariden nodes expose only **4 GPUs per node**.
	- The dedicated Clariden launcher now uses a **Ray-backed** multi-node serve path and defaults to the official **FP8 checkpoint** `Qwen/Qwen3.5-397B-A17B-FP8`.
	- On Clariden, the **bf16** checkpoint `Qwen/Qwen3.5-397B-A17B` is now a **known OOM on 2 nodes / 8 GPUs** during vLLM `profile_run`, even with `--language-model-only` and `--max-model-len 8192`.
	- Recommended starting point on Clariden: **2 nodes / 8 GPUs total** with `tensor_parallel_size=8`, `pipeline_parallel_size=1`, **Ray**, and the **FP8** checkpoint.
	- If you must run the **bf16** checkpoint on Clariden, start from **4 nodes / 16 GPUs total**. In this repo, the intended fallback shape is `tensor_parallel_size=8` and `pipeline_parallel_size=2`.
	- Keep the Clariden-specific container/EDF settings: **aarch64 image**, `qwen3-clariden`, CXI hook disabled, and `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` pinned to `nmn0`.
- The convenience wrappers in the repo root are intentionally for the **32B single-node Clariden workflow**:
	- `smoke_test_32b.sh`
	- `prefetch_32b.sh`
	- `run_single_dataset_32b.sh`
- The generic Slurm scripts in `scripts/` currently use a **single-node** shape by default (`--nodes=1`, `--gpus-per-node=4`, `TENSOR_PARALLEL_SIZE=4`). That matches the 32B Clariden workflow and smoke tests, but it is **not** a turnkey 397B-on-Clariden launcher.

### 1a) Bristen status

Verified on this Bristen `normal` partition on 2026-03-20:

- Nodes expose **4 GPUs per node**, not 8. `sinfo` reports `gpu:4` and node `CfgTRES` confirms `gres/gpu=4`.
- A single-node 4-GPU smoke test with `tensor_parallel_size=4`, `--language-model-only`, and `--max-model-len 8192` still reached **CUDA OOM during model load** for `Qwen/Qwen3.5-397B-A17B`.
- Practical consequence: this model does **not** fit on one Bristen node in the validated configuration. A real serving run will need a **multi-node setup** on Bristen or a different cluster/resource shape.

---

## 1b) Clariden notes (GH200 / aarch64)

- Clariden nodes are GH200, which means the host architecture is **aarch64**.
- You must run an **aarch64-compatible container image** on Clariden.
	- If you try to start an x86_64 `.sqsh` on Clariden, Pyxis/Enroot may fail extremely early in container startup; a common symptom is:
		- `pyxis: [ERROR] Failed to refresh the dynamic linker cache`
		- `/etc/enroot/hooks.d/87-slurm.sh exited with return code 1`

Repo convention:

- Bristen CE environment name: `qwen3` (expects `~/.edf/qwen3.toml`)
- Clariden CE environment name: `qwen3-clariden` (expects `~/.edf/qwen3-clariden.toml`)
- The Slurm scripts in `scripts/` auto-select between these using `SLURM_CLUSTER_NAME` / `SLURM_SUBMIT_HOST`.

Convenience wrappers (repo root):

- `smoke_test_32b.sh`: submits a 32B vLLM smoke on Clariden.
- `prefetch_32b.sh`: prefetches the `Qwen/Qwen3-32B` weights.
- `prefetch_datasets.sh`: prefetches one or more datasets (safe with comma-separated lists).
- `run_single_dataset_32b.sh`: full run for `glossAPI/Sxolika_vivlia` + `Qwen/Qwen3-32B`.

These wrappers are the recommended path when you want the currently validated **single-node Clariden** setup. For **Qwen/Qwen3.5-397B-A17B** on Clariden, keep using the same Clariden image and EDF, but switch to a **multi-node** Slurm allocation.

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

Clariden note: if Podman-based builds are unreliable on Clariden, the repo also provides `scripts/build_clariden_vllm_src_image.sh`, which builds a working Clariden `.sqsh` using an Enroot rootfs + `mksquashfs` export.

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

Note: EDF variable expansion is intentionally limited. Prefer plain `${VAR}` passthroughs.

In this repo, the EDF templates use `${VAR:-}` for a few optional variables (e.g. `HF_TOKEN`, `VLLM_HOST_IP`) so that an unset variable can expand to the empty string on systems where this is supported. If your CE/Pyxis setup rejects `${VAR:-}`, replace it with `${VAR}` and ensure the variable is always defined in the host environment (it can be an empty string).

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

## 6) Run Qwen3 with vLLM

### 6.1 Clariden: single-node 32B workflow

This is the validated path in this repository today:

- **Model**: `Qwen/Qwen3-32B`
- **Cluster**: Clariden
- **Resources**: `1 node`, `4 GPUs`, `tensor_parallel_size=4`
- **Runtime**: `qwen3-clariden` CE environment with the Clariden aarch64 image

Use the existing convenience wrappers:

```bash
./smoke_test_32b.sh
./prefetch_32b.sh
./run_single_dataset_32b.sh
```

The underlying Slurm scripts already match this shape (`--nodes=1`, `--gpus-per-node=4`) and avoid the Clariden multi-step issue by running server + healthcheck + client work inside a single `srun` step.

### 6.2 Clariden: multi-node 397B workflow

For `Qwen/Qwen3.5-397B-A17B`, a **single Clariden node is not enough** for the 8-GPU tensor-parallel shape suggested by the model card, because each Clariden node has only **4 GPUs**.

What needs to be true for the 397B run on Clariden:

- Use the **Clariden aarch64 image** and `qwen3-clariden` EDF.
- Prefer the default **FP8** checkpoint `Qwen/Qwen3.5-397B-A17B-FP8` for the current Clariden path.
- Request **at least 2 nodes** so you have **8 GPUs total**.
- For the current **FP8** starting point, use `tensor_parallel_size=8` and `pipeline_parallel_size=1` on a 2-node job.
- For the **bf16** checkpoint, do **not** reuse the 2-node shape: the repo has now observed a reproducible CUDA OOM during vLLM `profile_run` on `2 nodes / 8 GPUs`. Start from **4 nodes / 16 GPUs** with `tensor_parallel_size=8` and `pipeline_parallel_size=2` instead.
- Keep the Clariden networking/runtime settings:
	- `OCI_ANNOTATION_com__hooks__cxi__enabled=false`
	- `SLURM_NETWORK=disable_rdzv_get`
	- `NCCL_SOCKET_IFNAME=nmn0`
	- `GLOO_SOCKET_IFNAME=nmn0`
	- `VLLM_HOST_IP` set per rank/node
- Use a **multi-node vLLM launch**. This repo now provides [scripts/run_gsdg_qwen3_397b_clariden_multinode.sh](/users/p-skarvelis/GSDG/scripts/run_gsdg_qwen3_397b_clariden_multinode.sh), which currently exercises the 2-node layout correctly but is still blocked on the current image by a vLLM multi-node `mp` startup failure. Rebuild the Clariden image from `Containerfile.clariden` before the next 397B attempt so Ray is available in the runtime image.

Current runtime status on Clariden:

- The old multi-node `mp` path is no longer the main blocker; Ray is now installed in the Clariden image and the launcher uses `--distributed-executor-backend ray`.
- With `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, the Ray-backed launch gets past cluster formation and into checkpoint loading.
- The remaining validated limit is memory: the **bf16** checkpoint still exhausts `8 x GH200 95 GB` during model/profile initialization, so the current repo default is to run the **FP8** checkpoint for the 2-node Clariden path.

Minimal Slurm resource shape (starting point):

```bash
#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=qwen3-397b-clariden
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail

# End-to-end generation against the multi-node vLLM server:
export DATASET_NAME=glossAPI/<dataset_name>
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b.jsonl
sbatch scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

If you want the **bf16** checkpoint instead of the default FP8 one, use a larger fallback shape:

```bash
#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=qwen3-397b-clariden-bf16
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

set -euo pipefail

export MODEL_NAME=Qwen/Qwen3.5-397B-A17B
export TENSOR_PARALLEL_SIZE=8
export PIPELINE_PARALLEL_SIZE=2
export DATASET_NAME=glossAPI/<dataset_name>
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b_bf16.jsonl
sbatch scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

Notes about the current repo state:

- `scripts/smoke_test_vllm_qwen3.sh` and `scripts/run_gsdg_qwen3.sh` are still **single-node scripts**.
- They are appropriate for the **32B Clariden workflow** and for single-node smoke/debug work.
- For **397B on Clariden**, use [scripts/run_gsdg_qwen3_397b_clariden_multinode.sh](/users/p-skarvelis/GSDG/scripts/run_gsdg_qwen3_397b_clariden_multinode.sh) as the base launcher.
- The current launcher default is the **FP8** checkpoint on `2 nodes / 8 GPUs` with `tp=8, pp=1`.
- If you switch that launcher back to the **bf16** checkpoint, do not reuse the 2-node shape; start from `4 nodes / 16 GPUs` with `tp=8, pp=2`.

### 6.3 Bristen: model-card-style 8-GPU launch

Create a Slurm script (example) that:

- Requests 8 GPUs total
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

- Treat this as the **model-card-style launch shape**, not a validated Bristen single-node recipe for the current cluster hardware.
- The model card also shows a vLLM variant with reasoning enabled. If you enable thinking/reasoning, make sure your downstream parsing strips `<think>...</think>`.
- For synthetic Q/A generation, prefer API-side non-thinking mode by sending `chat_template_kwargs={"enable_thinking": False}` rather than using `/no_think` in the prompt.
- `--language-model-only` is recommended here because the pipeline consumes text-only GlossAPI rows and does not need the vision stack loaded.

### 6.4 Clariden or Bristen: multi-node notes

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
# srun --environment=qwen3-clariden <vllm multi-node launcher ...>
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

On Clariden, creating multiple Slurm steps in quick succession can fail transiently with “step creation temporarily disabled”. The repo scripts avoid this by running server + healthcheck + generator in a single `srun` step (see `scripts/smoke_test_vllm_qwen3.sh` and `scripts/run_gsdg_qwen3.sh`).

#### One-job example (server + generator)

This pattern runs everything in one job allocation on a single node. In this repo, that matches the **32B Clariden** workflow, not the **397B multi-node** workflow:

```bash
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

# In this repo, prefer the provided single-job script (it handles cluster
# differences and avoids multi-step issues on Clariden for the single-node path):
export DATASET_NAME=glossAPI/<dataset_name>
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml.jsonl
sbatch scripts/run_gsdg_qwen3.sh
```

If you prefer two `srun` steps with different resource shapes (e.g. more CPUs for the generator), split them and adjust `--cpus-per-task`. For `Qwen/Qwen3.5-397B-A17B` on Clariden, replace this single-node pattern with a dedicated **multi-node** launcher.

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

Note: some datasets on the Hub are “docs-only” (no loadable dataset files). The prefetch script treats those as warnings and continues (e.g. `glossAPI/istorima`).

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
- CE environments exist for the target cluster: `qwen3` on Bristen and `qwen3-clariden` on Clariden.
- A prefetch job can populate `${SCRATCH}` caches with model weights and dataset artifacts.
- A Clariden single-node job can run `Qwen/Qwen3-32B` on `1 node / 4 GPUs`.
- A multi-node job shape is defined for `Qwen/Qwen3.5-397B-A17B` on systems that expose only `4 GPUs per node`, such as Clariden.
- A generator script can stream GlossAPI rows and write ChatML JSONL with Greek Q/A pairs.


