Greek Synthetic Data Generation

This repository generates Greek synthetic question and answer pairs from GlossAPI datasets and writes them in ChatML JSONL format.

The recommended operating model is:

- `uenv` for build-time Python tooling and local validation on Alps.
- CSCS Container Engine for the final runtime under Slurm (Bristen and Clariden).

## What is implemented

- A Python CLI that loads either a HuggingFace dataset split or selected parquet files from a Hugging Face dataset repo or the local filesystem, extracts the best text payload from each row, calls an OpenAI-compatible inference endpoint, and writes ChatML records.
- A Python CLI that curates existing generated JSONL files, applies deterministic quality filters, optionally asks the model for semantic review/classification, and writes accepted samples into category-specific JSONL files.
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
- `src/gsdg/combine_jsonl.py`: JSONL combine utility with optional Q/A-based dedupe and row_id renumbering.
- `src/gsdg/curate_jsonl.py`: incremental JSONL curation, quality filtering, classification, and de-duplication.
- `scripts/generate_chatml.py`: script wrapper.
- `scripts/prefetch_hf_assets.py`: cache prefetch script wrapper.
- `scripts/combine_jsonl.py`: script wrapper for combining two or more JSONL outputs.
- `scripts/curate_jsonl.py`: script wrapper for curating generated JSONL files.
- `scripts/setup_uenv_python.sh`: build-time Python environment setup via `uenv`.
- `scripts/build_container_on_alps.sh`: build and import the CE image on Alps.
- `scripts/prefetch_hf_assets.sh`: Slurm job to warm model and dataset caches in `${SCRATCH}`.
- `scripts/run_gsdg_qwen3.sh`: single-job Slurm example.
- `scripts/run_gsdg_qwen3_397b_clariden_multinode.sh`: multi-node Clariden launcher for `Qwen/Qwen3.5-397B-A17B`, defaulting to the official FP8 checkpoint on the 397B path.
- `scripts/run_curate_qwen3_397b_clariden_multinode.sh`: multi-node Clariden launcher that starts the 397B API and runs the JSONL curation pass in the same allocation.
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

- Accepted samples are written into one file per category: `politics.jsonl`, `science.jsonl`, `medicine.jsonl`, `technology.jsonl`, `art.jsonl`, `history.jsonl`, `general.jsonl`.
- Rejected samples can be logged with reason codes via `--reject-log`.
- Resume state and the MinHash-LSH near-duplicate index are stored in a SQLite DB at `--state-db` or, by default, `${out_dir}/.curation_state.sqlite3`.
- Re-running the same command continues from the highest processed `meta.source_row_index`, so it is safe to use when the source JSONL is still growing.
- If `--tokenizer-model` is set, the tool uses the model tokenizer for context-window token counting; this requires `transformers` in `.venv-uenv`.

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

The Slurm scripts default to using `--environment=qwen3` on Bristen and `--environment=qwen3-clariden` on Clariden, based on `SLURM_CLUSTER_NAME` / `SLURM_SUBMIT_HOST`. Override explicitly with `CE_ENVIRONMENT=...` if needed.

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

Hugging Face parquet selection via the Slurm launcher uses `PARQUET_FILES` as a comma-separated list and optionally `HF_PARQUET_REPO`:

```bash
unset DATASET_NAME
export HF_PARQUET_REPO=fffoivos/glossapi-greek-nanochat-pretraining-dataset
export PARQUET_FILES=1000_prwta_xronia_ellhnikhs.parquet
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_1000_prwta.jsonl
sbatch scripts/run_gsdg_qwen3.sh
```

```bash
unset DATASET_NAME
export HF_PARQUET_REPO=fffoivos/glossapi-greek-nanochat-pretraining-dataset
export PARQUET_FILES=HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_hplt.jsonl
sbatch scripts/run_gsdg_qwen3.sh
```

For a local parquet file, omit `HF_PARQUET_REPO` and point `PARQUET_FILES` at the local path or glob:

```bash
unset DATASET_NAME
unset HF_PARQUET_REPO
export PARQUET_FILES=/path/to/my_downloaded_file.parquet
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_local.jsonl
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
4. For the 397B path, use `scripts/run_gsdg_qwen3_397b_clariden_multinode.sh`, which now defaults to `Qwen/Qwen3.5-397B-A17B-FP8` on a 2-node Clariden allocation shape.

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

Current 397B Clariden status:

- The launcher now uses a Ray-backed multi-node path instead of the older failing `mp` path.
- On Clariden, the bf16 checkpoint `Qwen/Qwen3.5-397B-A17B` is a known OOM on `2 nodes / 8 GPUs` during vLLM startup.
- The current default is the official FP8 checkpoint `Qwen/Qwen3.5-397B-A17B-FP8` on `2 nodes / 8 GPUs` with `tensor_parallel_size=8`, `pipeline_parallel_size=1`.
- The current Clariden image includes `flashinfer==0.6.4` and `flashinfer-cubin==0.6.4`; the launcher still keeps a defensive worker-side native GDN fallback for older images where `flashinfer` is missing.
- If you must run the bf16 checkpoint on Clariden, start from `4 nodes / 16 GPUs` with `TENSOR_PARALLEL_SIZE=8` and `PIPELINE_PARALLEL_SIZE=2`.


