Greek Synthetic Data Generation

This repository generates Greek synthetic question and answer pairs from GlossAPI datasets and writes them in ChatML JSONL format.

The recommended operating model is:

- `uenv` for build-time Python tooling and local validation on Alps.
- CSCS Container Engine for the final runtime under Slurm (Clariden).

## What is implemented

- A synchronous Python CLI that loads either a HuggingFace dataset split or selected parquet files from a Hugging Face dataset repo or the local filesystem, extracts the best text payload from each row, calls an OpenAI-compatible inference endpoint, and writes ChatML records.
- An async Python CLI (`scripts/generate_chatml_async.py`) with configurable concurrency and batch size, using `aiohttp` for 50–80× higher throughput while preserving row-order output and resumability.
- A synchronous Python CLI that curates existing generated JSONL files, applies deterministic quality filters, optionally asks the model for semantic review/classification, and writes accepted samples into category-specific JSONL files. Includes MinHash-LSH near-duplicate detection via a SQLite-backed deduplication store.
- An async Python CLI (`scripts/curate_jsonl_async.py`) that streams the input JSONL, runs LLM reviews concurrently, and runs dedup before the LLM review to save GPU compute.
- Text extraction heuristics for heterogeneous GlossAPI schemas.
- A strict Greek prompt template that asks Qwen3.5 for exactly one question/answer pair per row.
- Robust JSON parsing with fix-ups for LaTeX backslashes and unescaped quotes inside Greek guillemets.
- A JSONL combine utility with optional Q/A-based deduplication and sequential `row_id` renumbering.
- `uenv` and container build scaffolding for CSCS Clariden.
- Container and Slurm runtime scaffolding for running the workflow on CSCS Clariden, including a Ray-backed multi-node vLLM launcher for the 397B model on Clariden.

## Repository layout

- `src/gsdg/text_extraction.py`: row-to-text selection heuristics.
- `src/gsdg/prompting.py`: Greek prompt and ChatML record construction.
- `src/gsdg/openai_client.py`: synchronous OpenAI-compatible API client and robust JSON parsing.
- `src/gsdg/async_client.py`: async OpenAI-compatible API client built on `aiohttp` (shared by async generator and async curation).
- `src/gsdg/prefetch.py`: HuggingFace model and dataset cache prefetching.
- `src/gsdg/generator.py`: synchronous CLI entry point for ChatML Q/A generation.
- `src/gsdg/async_generator.py`: async batch generator with configurable concurrency for higher throughput.
- `src/gsdg/combine_jsonl.py`: JSONL combine utility with optional Q/A-based dedupe and `row_id` renumbering.
- `src/gsdg/curate_jsonl.py`: synchronous incremental JSONL curation, quality filtering, MinHash-LSH near-deduplication, LLM review, and topic classification.
- `src/gsdg/async_curation.py`: async streaming curation with concurrent LLM reviews and streaming input (never loads full JSONL into memory).
- `scripts/generate_chatml.py`: script wrapper for the synchronous generator.
- `scripts/generate_chatml_async.py`: script wrapper for the async generator (adds `--concurrency` and `--batch-size`).
- `scripts/prefetch_hf_assets.py`: cache prefetch script wrapper.
- `scripts/combine_jsonl.py`: script wrapper for combining two or more JSONL outputs.
- `scripts/curate_jsonl.py`: script wrapper for synchronous curation.
- `scripts/curate_jsonl_async.py`: script wrapper for async curation (adds `--curation-concurrency` and `--curation-batch-size`).
- `scripts/launch_vllm_serve_with_ray.py`: Ray-backed multi-node vLLM serve launcher for Clariden 397B jobs.
- `scripts/install_qwen35_gdn_sitecustomize.py`: GDN-native prefill fallback for older Clariden images without FlashInfer.
- `scripts/setup_uenv_python.sh`: build-time Python environment setup via `uenv`.
- `scripts/build_container_on_alps.sh`: build and import the CE image on Alps.
- `scripts/build_clariden_vllm_src_image.sh`: alternative Clariden-native image build (rootfs + mksquashfs).
- `scripts/prefetch_hf_assets.sh`: Slurm job to warm model and dataset caches in `${SCRATCH}`.
- `scripts/run_gsdg_qwen3_397b_clariden_async.sh`: async generator launcher for 397B on Clariden.
- `scripts/run_curate_qwen3_397b_clariden_async.sh`: async curation launcher for 397B on Clariden.
- `run_uenv.sh`: default uenv entry point — activates the repo venv, sets `PYTHONPATH`, loads `.env`, and runs the given command.
- `prefetch_datasets.sh`: convenience wrapper to prefetch one or more datasets.
- `prefetch_parquet_repo.sh`: convenience wrapper to prefetch parquet files from a HuggingFace dataset repo using `uenv`.
- `edf/qwen3_clariden.toml.example`: CE environment template for Clariden (GH200 / aarch64).
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

**Recommended**: after the venv is created, use `./run_uenv.sh` as the default entry point for all Python commands. It automatically activates the venv, sets `PYTHONPATH`, loads HF cache paths, sources `.env` (for `HF_TOKEN`), and installs dependencies from `requirements.txt` if they changed:

```bash
./run_uenv.sh python scripts/generate_chatml.py --help
./run_uenv.sh python scripts/curate_jsonl.py --help
./run_uenv.sh bash
```

## Local CLI usage

Run the generator against an already-running local OpenAI-compatible server.

Hugging Face dataset mode:

```bash
python scripts/generate_chatml.py \
	--dataset glossAPI/<dataset_name> \
	--split train \
	--out outputs/synthetic_chatml.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B \
	--max-rows 100
```

Specific parquet file from a Hugging Face dataset repo:

```bash
python scripts/generate_chatml.py \
	--hf-parquet-repo fffoivos/glossapi-greek-nanochat-pretraining-dataset \
	--parquet-file 1000_prwta_xronia_ellhnikhs.parquet \
	--out outputs/1000_prwta_xronia_ellhnikhs.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B
```

Multiple parquet shards from a Hugging Face dataset repo via glob:

```bash
python scripts/generate_chatml.py \
	--hf-parquet-repo fffoivos/glossapi-greek-nanochat-pretraining-dataset \
	--parquet-file 'HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet' \
	--out outputs/hplt_clean60.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B
```

Local parquet file or local glob:

```bash
python scripts/generate_chatml.py \
	--parquet-file /path/to/1000_prwta_xronia_ellhnikhs.parquet \
	--out outputs/local_parquet.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B

python scripts/generate_chatml.py \
	--parquet-file '/path/to/HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet' \
	--out outputs/local_hplt.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B
```

For parquet input you can repeat `--parquet-file` to combine multiple exact files or glob patterns into one JSONL. When you select parquet files, `--split` is still accepted and is used as the logical split label written into the output metadata.

By default the generator requests Qwen3.5 in non-thinking mode through the OpenAI-compatible API, which is more reliable for strict JSON output. Pass `--enable-thinking` only if you explicitly want reasoning traces.

### Async generation (higher throughput)

The async generator (`scripts/generate_chatml_async.py`) uses `aiohttp` for non-blocking HTTP requests, delivering 50–80× higher throughput than the synchronous version. It shares the same source arguments as `generate_chatml.py` and adds two controls:

```bash
python scripts/generate_chatml_async.py \
	--dataset glossAPI/<dataset_name> \
	--split train \
	--out outputs/synthetic_chatml.jsonl \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B \
	--max-rows 10000 \
	--concurrency 64 \
	--batch-size 2000
```

- `--concurrency` (default 64): maximum simultaneous in-flight requests to the vLLM server.
- `--batch-size` (default 2000): rows to collect before sorting and flushing to disk. Output is written in row-index order within each batch so resumability is preserved.
- Default `--timeout-seconds` is 600 in async mode (vs 180 in sync mode).

## Combine existing JSONL outputs

Use the combine tool to merge two or more JSONL files produced by this repo:

```bash
python3 scripts/combine_jsonl.py \
	outputs/file_a.jsonl \
	outputs/file_b.jsonl \
	--out outputs/combined.jsonl
```

Optional dedupe keeps the first occurrence of each unique question/answer pair:

```bash
python3 scripts/combine_jsonl.py \
	outputs/file_a.jsonl \
	outputs/file_b.jsonl \
	outputs/file_c.jsonl \
	--out outputs/combined_deduped.jsonl \
	--dedupe
```

Combine behavior summary:

- Input order is preserved in the output.
- Each non-empty line must be valid JSON.
- With `--dedupe`, duplicates are removed only when the extracted `question` and `answer` are identical.
- During combine, `meta.row_id` is renumbered sequentially (`0..N-1`) based on final output order (after dedupe).
- `--out` must be different from every input path.

## Curate existing JSONL outputs

Use the curation tool on a generator-produced JSONL file when you want to filter low-quality Q/A pairs, classify accepted samples by topic, and keep a persistent resume/de-duplication state.

Local or login-node usage expects an already-running OpenAI-compatible API if you want semantic review and topic classification from the model:

```bash
./run_uenv.sh python scripts/curate_jsonl.py \
	outputs/combined_deduped_Wikisource_Greek_texts.jsonl \
	--out-dir "${SCRATCH}/synthetics" \
	--reject-log "${SCRATCH}/synthetics/rejects_wikisource.jsonl" \
	--tokenizer-model Qwen/Qwen3.5-397B-A17B-FP8 \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B-FP8
```

Deterministic-only mode skips the LLM review stage and keeps only the rule-based filters:

```bash
./run_uenv.sh python scripts/curate_jsonl.py \
	outputs/combined_deduped_Wikisource_Greek_texts.jsonl \
	--out-dir "${SCRATCH}/synthetics" \
	--reject-log "${SCRATCH}/synthetics/rejects_wikisource.jsonl" \
	--tokenizer-model Qwen/Qwen3.5-397B-A17B-FP8 \
	--disable-llm-review
```

Curation behavior summary:

- Accepted samples are written into one file per category: `politics.jsonl`, `science.jsonl`, `medicine.jsonl`, `technology.jsonl`, `art.jsonl`, `history.jsonl`, `geography.jsonl`, `religion.jsonl`, `education.jsonl`, `philosophy.jsonl`, `sports.jsonl`, `business.jsonl`, `economics.jsonl`, `law.jsonl`, `mythology.jsonl`, `literature.jsonl`, `music.jsonl`, `general.jsonl`.
- Rejected samples can be logged with reason codes via `--reject-log`.
- Resume state and the MinHash-LSH near-duplicate index are stored in a SQLite DB at `--state-db` or, by default, `${out_dir}/.curation_state.sqlite3`.
- Re-running the same command continues from the highest processed `meta.source_row_index`, so it is safe to use when the source JSONL is still growing.
- If `--tokenizer-model` is set, the tool uses the model tokenizer for context-window token counting; this requires `transformers` in `.venv-uenv`.

### Async curation (higher throughput)

The async curation script (`scripts/curate_jsonl_async.py`) uses `aiohttp` for concurrent LLM reviews and streams the input JSONL one row at a time (never loads the full file into memory). It shares the same arguments as `curate_jsonl.py` and adds two controls:

```bash
./run_uenv.sh python scripts/curate_jsonl_async.py \
	outputs/combined_deduped_Wikisource_Greek_texts.jsonl \
	--out-dir "${SCRATCH}/synthetics" \
	--reject-log "${SCRATCH}/synthetics/rejects_wikisource.jsonl" \
	--tokenizer-model Qwen/Qwen3.5-397B-A17B-FP8 \
	--api-base http://localhost:8000/v1 \
	--model Qwen/Qwen3.5-397B-A17B-FP8 \
	--curation-concurrency 16 \
	--curation-batch-size 200
```

- `--curation-concurrency` (default 16): maximum simultaneous in-flight LLM review requests.
- `--curation-batch-size` (default 200): rows to process before reporting progress.
- Deduplication (exact + near-duplicate via MinHash-LSH) runs **before** the LLM review so duplicates don't consume GPU compute.
- Memory usage is O(batch_size + concurrency), not O(total rows).

## Container build on Alps

Build and import the runtime image after you have validated the Python code in `uenv`:

### Clariden (GH200 / aarch64)

Clariden compute nodes are GH200 (ARM/aarch64). You must use an aarch64-compatible image there.

Build and import the Clariden image using the Clariden Containerfile and a different output path:

```bash
CONTAINERFILE=Containerfile.clariden \
IMAGE_TAG=gsdg-qwen3-clariden:latest \
SQSH_PATH=${SCRATCH}/images/gsdg-qwen3_clariden_flashinfer_latest.sqsh \
./scripts/build_container_on_alps.sh
```

Alternative (Clariden-native build): this repo also provides `scripts/build_clariden_vllm_src_image.sh`, which builds a working Clariden `.sqsh` by creating an Enroot rootfs from a base image and installing the Python/vLLM stack into `/opt/gsdg-venv`, then exporting via `mksquashfs`.

If you see Pyxis/Enroot fail very early with messages like "Failed to refresh the dynamic linker cache" on Clariden, it is usually a sign that an x86_64 image is being started on an aarch64 node.

This creates the SquashFS image at the `SQSH_PATH` you set (for Clariden, typically `${SCRATCH}/images/gsdg-qwen3_clariden_flashinfer_latest.sqsh`).

For Clariden you should use the Clariden output path `${SCRATCH}/images/gsdg-qwen3_clariden_flashinfer_latest.sqsh` and the corresponding EDF template `edf/qwen3_clariden.toml.example`.

For Clariden, the image recipe in this repo builds vLLM `v0.17.1`, installs Transformers from `main`, and includes both Ray and FlashInfer `0.6.4` for the multi-node 397B path.

If `~/.config/containers/storage.conf` does not exist yet, the helper script creates one that points Podman storage at `/dev/shm/$USER`. This avoids rootless overlay failures on home-backed network filesystems.

## Prefetch weights and datasets into `${SCRATCH}`

Warm the HuggingFace caches before the main run so later jobs can reuse the weights and dataset artifacts:

```bash
export PREFETCH_DATASETS=glossAPI/<dataset_name>
sbatch scripts/prefetch_hf_assets.sh
```

The Slurm scripts default to using `--environment=qwen3-clariden` on Clariden, based on `SLURM_CLUSTER_NAME` / `SLURM_SUBMIT_HOST`. Override explicitly with `CE_ENVIRONMENT=...` if needed.

You can prefetch multiple datasets by separating them with commas:

```bash
export PREFETCH_DATASETS=glossAPI/<dataset_a>,glossAPI/<dataset_b>
sbatch scripts/prefetch_hf_assets.sh
```

For gated parquet repos that 401 from inside the CE container, prefetch the
selected parquet files host-side with `uenv` and then point the Slurm job at
the local copies:

```bash
export HF_PARQUET_REPO=fffoivos/glossapi-greek-nanochat-pretraining-dataset
export PARQUET_FILES='HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet'
./prefetch_parquet_repo.sh

unset HF_PARQUET_REPO
export PARQUET_FILES="${SCRATCH}/gsdg_parquet_prefetch/fffoivos__glossapi-greek-nanochat-pretraining-dataset/data/HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet"
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

## Clariden runtime workflow

1. Build/import the Clariden image (see above).
2. Copy `edf/qwen3_clariden.toml.example` to `~/.edf/qwen3-clariden.toml` and adjust the `image = ...` path if needed (defaults to `${SCRATCH}/images/gsdg-qwen3_clariden_flashinfer_latest.sqsh`).
3. For the validated 32B path, use the existing single-node wrappers under `scripts/helpers/`.
4. For the 397B path, use `scripts/run_gsdg_qwen3_397b_clariden_multinode.sh`, which defaults to `Qwen/Qwen3.5-397B-A17B-FP8` on a 2-node Clariden allocation shape.
5. For the async 397B generator (higher throughput), use `scripts/run_gsdg_qwen3_397b_clariden_async.sh`.
6. For the 397B-FP8 convenience wrapper, use `scripts/helpers/run_single_dataset_397b-fp8.sh`.

To curate an existing generator-produced JSONL with the 397B model in the same Slurm allocation, submit the dedicated launcher:

```bash
export INPUT_JSONL=/users/p-skarvelis/GSDG/outputs/combined_deduped_Wikisource_Greek_texts.jsonl
export CURATION_OUT_DIR=${SCRATCH}/synthetics
export REJECT_LOG=${SCRATCH}/synthetics/rejects_wikisource.jsonl
sbatch scripts/run_curate_qwen3_397b_clariden_multinode.sh
```

This launcher starts the Ray-backed 397B API, runs `scripts/curate_jsonl.py` on the head node, writes category outputs under `${CURATION_OUT_DIR}`, and keeps its resume/de-duplication state in `${CURATION_OUT_DIR}/.curation_state.sqlite3`. The input JSONL is snapshotted at job start, so if the source file is still growing you can submit the same job again later and the curation state will continue from the highest processed `source_row_index`.

If you want only deterministic rules and no model review, submit with:

```bash
export INPUT_JSONL=/users/p-skarvelis/GSDG/outputs/combined_deduped_Wikisource_Greek_texts.jsonl
export CURATION_OUT_DIR=${SCRATCH}/synthetics
export DISABLE_LLM_REVIEW=1
sbatch scripts/run_curate_qwen3_397b_clariden_multinode.sh
```

For async curation (concurrent LLM reviews, higher throughput), use:

```bash
export INPUT_JSONL=/users/p-skarvelis/GSDG/outputs/combined_deduped_Wikisource_Greek_texts.jsonl
export CURATION_OUT_DIR=${SCRATCH}/synthetics
export REJECT_LOG=${SCRATCH}/synthetics/rejects_wikisource.jsonl
sbatch scripts/run_curate_qwen3_397b_clariden_async.sh
```

Current 397B Clariden status:

- The launcher now uses a Ray-backed multi-node path instead of the older failing `mp` path.
- On Clariden, the bf16 checkpoint `Qwen/Qwen3.5-397B-A17B` is a known OOM on `2 nodes / 8 GPUs` during vLLM startup.
- The current default is the official FP8 checkpoint `Qwen/Qwen3.5-397B-A17B-FP8` on `2 nodes / 8 GPUs` with `tensor_parallel_size=8`, `pipeline_parallel_size=1`.
- The current Clariden image includes `flashinfer==0.6.4` and `flashinfer-cubin==0.6.4`; the launcher still keeps a defensive worker-side native GDN fallback for older images where `flashinfer` is missing.
- If you must run the bf16 checkpoint on Clariden, start from `4 nodes / 16 GPUs` with `TENSOR_PARALLEL_SIZE=8` and `PIPELINE_PARALLEL_SIZE=2`.


