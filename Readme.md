Greek Synthetic Data Generation

This repository generates Greek synthetic question and answer pairs from GlossAPI datasets and writes them in ChatML JSONL format.

The recommended operating model is:

- `uenv` for build-time Python tooling and local validation on Alps.
- CSCS Container Engine for the final runtime under Slurm (Bristen and Clariden).

## What is implemented

- A Python CLI that loads a HuggingFace dataset split, extracts the best text payload from each row, calls an OpenAI-compatible inference endpoint, and writes ChatML records.
- Text extraction heuristics for heterogeneous GlossAPI schemas.
- A strict Greek prompt template that asks Qwen3.5 for exactly one question/answer pair per row.
- `uenv` and container build scaffolding for CSCS Bristen.
- Container and Slurm runtime scaffolding for running the workflow on CSCS Bristen and Clariden.

## Repository layout

- `src/gsdg/text_extraction.py`: row-to-text selection heuristics.
- `src/gsdg/prompting.py`: Greek prompt and ChatML record construction.
- `src/gsdg/openai_client.py`: local OpenAI-compatible API client and JSON parsing.
- `src/gsdg/prefetch.py`: HuggingFace model and dataset cache prefetching.
- `src/gsdg/generator.py`: main CLI entry point.
- `scripts/generate_chatml.py`: script wrapper.
- `scripts/prefetch_hf_assets.py`: cache prefetch script wrapper.
- `scripts/setup_uenv_python.sh`: build-time Python environment setup via `uenv`.
- `scripts/build_container_on_alps.sh`: build and import the CE image on Alps.
- `scripts/prefetch_hf_assets.sh`: Slurm job to warm model and dataset caches in `${SCRATCH}`.
- `scripts/run_gsdg_qwen3.sh`: single-job Slurm example.
- `scripts/run_gsdg_qwen3_397b_clariden_multinode.sh`: multi-node Clariden launcher for `Qwen/Qwen3.5-397B-A17B`.
- `smoke_test_32b.sh`: convenience wrapper for a 32B vLLM smoke test on Clariden.
- `prefetch_32b.sh`: convenience wrapper to prefetch `Qwen/Qwen3-32B` weights.
- `prefetch_datasets.sh`: convenience wrapper to prefetch one or more datasets.
- `run_single_dataset_32b.sh`: convenience wrapper for a full run on `glossAPI/Sxolika_vivlia`.
- `edf/qwen3.toml.example`: CE environment template.
- `edf/qwen3_clariden.toml.example`: CE environment template for Clariden (GH200 / aarch64).
- `Containerfile`: container build recipe.
- `Containerfile.clariden`: container build recipe for Clariden (GH200 / aarch64).

The EDF uses Pyxis-compatible variable expansion only. Keep it simple (plain `${VAR}` passthroughs).

Note: in this repo the EDF templates use `${VAR:-}` for a few optional variables so that an unset variable can expand to the empty string on systems where this is supported. If your CE/Pyxis setup rejects `${VAR:-}`, replace it with `${VAR}` and ensure the variable is always defined in the host environment (it can be an empty string).

## Build-time setup with uenv

Prepare a modern Python environment on Alps with the recommended `uenv` image:

```bash
./scripts/setup_uenv_python.sh
source .venv-uenv/bin/activate
export PYTHONPATH=$PWD/src
```

This gives you a current Python toolchain for validation and development without making `uenv` part of the runtime job.

## Local CLI usage

Run the generator against an already-running local OpenAI-compatible server:

```bash
python scripts/generate_chatml.py \
	--dataset glossAPI/<dataset_name> \
	--split train \
	--out outputs/synthetic_chatml.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B \
	--max-rows 100
```

By default the generator requests Qwen3.5 in non-thinking mode through the OpenAI-compatible API, which is more reliable for strict JSON output. Pass `--enable-thinking` only if you explicitly want reasoning traces.

## Container build on Alps

Build and import the runtime image after you have validated the Python code in `uenv`:

```bash
./scripts/build_container_on_alps.sh
```

This builds the default (Bristen / x86_64) image from `Containerfile` and imports it to `${SCRATCH}/images/gsdg-qwen3_latest.sqsh`.

### Clariden (GH200 / aarch64)

Clariden compute nodes are GH200 (ARM/aarch64). You must use an aarch64-compatible image there.

Build and import the Clariden image using the Clariden Containerfile and a different output path:

```bash
CONTAINERFILE=Containerfile.clariden \
IMAGE_TAG=gsdg-qwen3-clariden:latest \
SQSH_PATH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh \
./scripts/build_container_on_alps.sh
```

Alternative (Clariden-native build): this repo also provides `scripts/build_clariden_vllm_src_image.sh`, which builds a working Clariden `.sqsh` by creating an Enroot rootfs from a base image and installing the Python/vLLM stack into `/opt/gsdg-venv`, then exporting via `mksquashfs`.

If you see Pyxis/Enroot fail very early with messages like "Failed to refresh the dynamic linker cache" on Clariden, it is usually a sign that an x86_64 image is being started on an aarch64 node.

This creates the SquashFS image at the `SQSH_PATH` you set (for Clariden, typically `${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh`).

For Clariden you should use the Clariden output path `${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh` and the corresponding EDF template `edf/qwen3_clariden.toml.example`.

For Clariden, the working image in this repo builds vLLM `v0.17.1` and installs a compatible `transformers` version by default. If you need bleeding-edge Qwen3.5 support, you may need to install Transformers from `main` and/or use a newer vLLM.

If `~/.config/containers/storage.conf` does not exist yet, the helper script creates one that points Podman storage at `/dev/shm/$USER`. This avoids rootless overlay failures on home-backed network filesystems.

## Prefetch weights and datasets into `${SCRATCH}`

Warm the HuggingFace caches before the main run so later jobs can reuse the weights and dataset artifacts:

```bash
export PREFETCH_DATASETS=glossAPI/<dataset_name>
sbatch scripts/prefetch_hf_assets.sh
```

The Slurm scripts default to using `--environment=qwen3` on Bristen and `--environment=qwen3-clariden` on Clariden, based on `SLURM_CLUSTER_NAME` / `SLURM_SUBMIT_HOST`. Override explicitly with `CE_ENVIRONMENT=...` if needed.

You can prefetch multiple datasets by separating them with commas:

```bash
export PREFETCH_DATASETS=glossAPI/<dataset_a>,glossAPI/<dataset_b>
sbatch scripts/prefetch_hf_assets.sh
```

The script runs inside the CE environment, so it uses the same `${SCRATCH}`-backed `HF_HOME` and `HF_DATASETS_CACHE` settings as the runtime job.

The default prefetch job footprint is intentionally small and queue-friendly: 1 task, 4 CPUs, 32 GB RAM, 0 GPUs. You can override that on submission if you need a larger model-only job shape.

Useful prefetch controls:

- `SKIP_MODEL=1`: prefetch datasets only.
- `SKIP_DATASETS=1`: prefetch model weights only.
- `HF_HUB_DISABLE_XET=1`: disable the Xet transfer backend for troubleshooting.
- `PREFETCH_LOG_LEVEL=DEBUG`: emit more verbose Python-side logs.

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

Override Slurm resources at submission time if needed:

```bash
sbatch --cpus-per-task=8 --mem=64G scripts/prefetch_hf_assets.sh
```

If you pass a comma-separated dataset list via a wrapper script, prefer `./prefetch_datasets.sh` (it handles Slurm `--export` comma semantics safely).

## Bristen runtime workflow

1. Use `uenv` to create a current Python environment and validate the generator.
2. Build the runtime container and import it to `${SCRATCH}/images/gsdg-qwen3_latest.sqsh` (Bristen) or `${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh` (Clariden).
3. Copy `edf/qwen3.toml.example` to `~/.edf/qwen3.toml` and fill any environment-specific values.
4. Optionally prefetch the model and datasets into `${SCRATCH}` with `scripts/prefetch_hf_assets.sh`.
5. Submit `scripts/run_gsdg_qwen3.sh` with the required environment variables.

Example submission:

```bash
export DATASET_NAME=glossAPI/<dataset_name>
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml.jsonl
sbatch scripts/run_gsdg_qwen3.sh
```

## Notes

- The intended split is build-time in `uenv`, runtime in CE. Avoid trying to activate `uenv` inside the runtime container.
- Prefetching is designed to run in the CE environment so the warmed caches match the final runtime environment.
- If a combined model+dataset prefetch is hard to diagnose, prefer separate dataset-only and model-only runs first.
- The generator strips `<think>...</think>` blocks before parsing model JSON.
- The runtime launch uses `--language-model-only` because this pipeline is text-only and Qwen3.5 is a multimodal model.
- If a row has no usable text fields, it is skipped and logged.
- Output is appended to the target JSONL file so interrupted jobs can be resumed carefully by changing `--start-row`.

Clariden-specific notes:

- Some Clariden configurations can transiently reject multiple Slurm steps (“step creation temporarily disabled”). The provided scripts run server + health + one request / generation inside a single `srun` step to avoid this.
- The Slurm scripts may “stage” the current repo into `$SCRATCH` and set `PYTHONPATH` inside the container so that small Python changes take effect without rebuilding the `.sqsh`. Disable with `STAGE_WORKSPACE=0`.

See `Agents.md` for the full Bristen runbook and cluster-specific operational guidance.

## Clariden runtime workflow

1. Build/import the Clariden image (see above).
2. Copy `edf/qwen3_clariden.toml.example` to `~/.edf/qwen3-clariden.toml` and adjust the `image = ...` path if needed.
3. For the validated 32B path, use the existing single-node wrappers.
4. For `Qwen/Qwen3.5-397B-A17B`, submit `scripts/run_gsdg_qwen3_397b_clariden_multinode.sh` on a 2-node Clariden allocation shape.


