# How to run on Clariden (minimum steps)
Build/import an aarch64 image on Clariden:
CONTAINERFILE=Containerfile.clariden IMAGE_TAG=gsdg-qwen3-clariden:latest SQSH_PATH=${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh ./scripts/build_container_on_alps.sh

Create the Clariden CE environment file:
cp qwen3_clariden.toml.example ~/.edf/qwen3-clariden.toml

Run  prefetch
CE_ENVIRONMENT=qwen3-clariden HF_TOKEN="$HF_TOKEN" MODEL_NAME='Qwen/Qwen3-32B' SKIP_DATASETS=1 HF_HUB_DISABLE_XET=1 sbatch scripts/prefetch_hf_assets.sh
