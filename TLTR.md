# How to run on Clariden (minimum steps)

## 1) Build/import the Clariden (aarch64) image

Build/import an aarch64 image (see the repo runbook for the recommended workflow):

```bash
CONTAINERFILE=Containerfile.clariden \
IMAGE_TAG=gsdg-qwen3-clariden:latest \
SQSH_PATH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh \
./scripts/build_container_on_alps.sh
```

## 2) Create the Clariden CE environment file

```bash
cp edf/qwen3_clariden.toml.example ~/.edf/qwen3-clariden.toml
```

## 3) Smoke test vLLM (32B on 4×GH200)

Put `HF_TOKEN=...` into `.env` (not committed), then:

```bash
./smoke_test_32b.sh
```

## 4) Prefetch (optional, speeds up later jobs)

Model-only prefetch for the 32B weights:

```bash
./prefetch_32b.sh
```

Datasets-only prefetch (comma-separated list):

```bash
PREFETCH_DATASETS='glossAPI/Sxolika_vivlia,glossAPI/istorima' ./prefetch_datasets.sh
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

Note: `glossAPI/istorima` currently has no loadable dataset files on the Hub
(docs-only repo), so it will be skipped with a warning.

## 5) Full run (dataset or parquet → ChatML JSONL)

Default end-to-end run for `glossAPI/Sxolika_vivlia` + `Qwen/Qwen3-32B`:

```bash
./run_single_dataset_32b.sh
```

Specific parquet file from the Hugging Face repo `fffoivos/glossapi-greek-nanochat-pretraining-dataset`:

```bash
unset DATASET_NAME
export HF_PARQUET_REPO=fffoivos/glossapi-greek-nanochat-pretraining-dataset
export PARQUET_FILES=1000_prwta_xronia_ellhnikhs.parquet
./run_single_dataset_32b.sh
```

All matching HPLT parquet shards from the same Hugging Face repo into one JSONL:

```bash
unset DATASET_NAME
export HF_PARQUET_REPO=fffoivos/glossapi-greek-nanochat-pretraining-dataset
export PARQUET_FILES='HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet'
./run_single_dataset_32b.sh
```

Local parquet file or local parquet glob:

```bash
unset DATASET_NAME
unset HF_PARQUET_REPO
export PARQUET_FILES=/path/to/my_downloaded_file.parquet
./run_single_dataset_32b.sh

unset DATASET_NAME
unset HF_PARQUET_REPO
export PARQUET_FILES='/path/to/HPLT__ell_Grek_ge8_no_mt_clean60.part-*.parquet'
./run_single_dataset_32b.sh
```

If you want to combine multiple exact files or patterns in one run, set `PARQUET_FILES` as a comma-separated list.

By default the Slurm scripts stage the current repo into `$SCRATCH` and set
`PYTHONPATH` inside the container, so small Python changes take effect without
rebuilding the `.sqsh`.

## 6) Full run for 397B on Clariden

For the current default 397B path, use the dedicated multi-node launcher:

```bash
export DATASET_NAME=glossAPI/Sxolika_vivlia
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b.jsonl
sbatch scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

The same input selectors also work for the 397B path. For example, a specific HF parquet file:

```bash
unset DATASET_NAME
export HF_PARQUET_REPO=fffoivos/glossapi-greek-nanochat-pretraining-dataset
export PARQUET_FILES=1000_prwta_xronia_ellhnikhs.parquet
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b.jsonl
sbatch scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

Or a local parquet file:

```bash
unset DATASET_NAME
unset HF_PARQUET_REPO
export PARQUET_FILES=/path/to/my_downloaded_file.parquet
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b_local.jsonl
sbatch scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

This launcher now defaults to `Qwen/Qwen3.5-397B-A17B-FP8` and uses 2 Clariden
nodes with 4 GPUs per node, configured as `tensor_parallel_size=8` and
`pipeline_parallel_size=1`.

Current status:

- The launcher uses a Ray-backed multi-node path.
- The current Clariden image includes `flashinfer==0.6.4`; the launcher still keeps a worker-wide Python startup fallback so older images can force Qwen3.5 onto the native GDN prefill path if `flashinfer` is missing.
- The warning about `tensor_parallel_size=8` being larger than the 4 GPUs reserved on each Clariden node is expected on the 2-node FP8 path. It means tensor parallelism is spanning nodes, which is the intended layout here.
- The bf16 checkpoint `Qwen/Qwen3.5-397B-A17B` is a known OOM on Clariden at
	`2 nodes / 8 GPUs` during vLLM startup.
- If you need the bf16 checkpoint, request `4 nodes / 16 GPUs` and submit with:

```bash
export MODEL_NAME=Qwen/Qwen3.5-397B-A17B
export TENSOR_PARALLEL_SIZE=8
export PIPELINE_PARALLEL_SIZE=2
export DATASET_NAME=glossAPI/Sxolika_vivlia
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b_bf16.jsonl
sbatch --nodes=4 scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

### Notes:
  - view the log with `tail -f /iopsstor/scratch/cscs/${USER}/vllm-397b-node0.log` (or `node1.log` for the second node) to confirm the expected checkpoint is being loaded and to monitor GPU memory usage during startup.

## 7) Curate an existing JSONL with 397B on Clariden

To run quality filtering, semantic review, classification, and near-duplicate removal on an existing generator-produced JSONL, submit the dedicated curation launcher:

```bash
export INPUT_JSONL=/users/p-skarvelis/GSDG/outputs/combined_deduped_Wikisource_Greek_texts.jsonl
export CURATION_OUT_DIR=${SCRATCH}/synthetics
export REJECT_LOG=${SCRATCH}/synthetics/rejects_wikisource.jsonl
sbatch scripts/run_curate_qwen3_397b_clariden_multinode.sh
```

This job will:

- start the same Ray-backed `Qwen/Qwen3.5-397B-A17B-FP8` API inside the Slurm allocation,
- run `scripts/curate_jsonl.py` against a snapshot of `INPUT_JSONL`,
- write accepted samples into `${CURATION_OUT_DIR}/politics.jsonl`, `science.jsonl`, `medicine.jsonl`, `technology.jsonl`, `art.jsonl`, `history.jsonl`, and `general.jsonl`,
- write rejects into `${REJECT_LOG}`,
- persist resume and de-duplication state in `${CURATION_OUT_DIR}/.curation_state.sqlite3`.

If the source JSONL is still growing, submit the same command again later. The curation pass continues from the highest processed `meta.source_row_index` instead of starting over.

If you want deterministic-only filtering without model review, add:

```bash
export DISABLE_LLM_REVIEW=1
sbatch scripts/run_curate_qwen3_397b_clariden_multinode.sh
```

Logs:

- follow `slurm-<jobid>.out` for the launcher progress and the final server-log path,
- or tail the vLLM log in the job working directory, typically `${SCRATCH}/vllm-curate-397b-node0.log` (and `node1.log` for the second node).
  