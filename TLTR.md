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

Note: `glossAPI/istorima` currently has no loadable dataset files on the Hub
(docs-only repo), so it will be skipped with a warning.

## 5) Full run (dataset → ChatML JSONL)

End-to-end run for `glossAPI/Sxolika_vivlia` + `Qwen/Qwen3-32B`:

```bash
./run_single_dataset_32b.sh
```

By default the Slurm scripts stage the current repo into `$SCRATCH` and set
`PYTHONPATH` inside the container, so small Python changes take effect without
rebuilding the `.sqsh`.

## 6) Full run for 397B on Clariden (2 nodes)

For `Qwen/Qwen3.5-397B-A17B`, use the dedicated multi-node launcher:

```bash
export DATASET_NAME=glossAPI/Sxolika_vivlia
export OUTPUT_PATH=${SCRATCH}/synthetic_chatml_397b.jsonl
sbatch scripts/run_gsdg_qwen3_397b_clariden_multinode.sh
```

This launcher uses 2 Clariden nodes with 4 GPUs per node, configured as
`tensor_parallel_size=8` and `pipeline_parallel_size=1`.

Current status: the launcher wiring is fixed, but the currently built Clariden
image still fails in vLLM multi-node `mp` worker startup with `inner dp world
group is not initialized`. Rebuild the Clariden image from `Containerfile.clariden`
before the next 397B validation run so Ray is available in the runtime image.