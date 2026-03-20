#!/bin/bash
#SBATCH -A a0140
#SBATCH --job-name=build-qwen3-clariden
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:30:00

set -euo pipefail

cd /users/p-skarvelis/GSDG

BASE_SQSH="${BASE_SQSH:-${SCRATCH}/images/gsdg-qwen3_clariden_base.sqsh}"
OUT_SQSH="${OUT_SQSH:-${SCRATCH}/images/gsdg-qwen3_clariden_latest.sqsh}"
NAME="${NAME:-gsdg-qwen3-clariden-vllm-src-${SLURM_JOB_ID}}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

log "Using BASE_SQSH=$BASE_SQSH"
log "Writing OUT_SQSH=$OUT_SQSH"
log "Enroot container name: $NAME"

test -f "$BASE_SQSH"
mkdir -p "$(dirname "$OUT_SQSH")"
rm -f "$OUT_SQSH"

log "enroot create"
enroot create -n "$NAME" "$BASE_SQSH"

log "enroot start: install deps + build vLLM from source"
# Disable Enroot hooks that inject host files into the rootfs; those bind mounts
# will otherwise get squashed into the exported image as 0-byte placeholders.
export ENROOT_SLURM_HOOK=off

# Clariden injects host libfabric via an Enroot hook; that breaks torch import
# because HPCX MPI expects newer FABRIC symbol versions. We don't need CXI OFI
# integration to build vLLM, so disable the hook for this build.
export OCI_ANNOTATION_com__hooks__cxi__enabled=false

# Stream the workspace into the container without a persistent bind mount.
tar -C /users/p-skarvelis/GSDG -cz requirements.txt src scripts Readme.md | \
	enroot start --rw "$NAME" bash -lc '
  set -euo pipefail
  set -x

  # Quick connectivity diagnostics (many compute nodes restrict outbound access)
  command -v curl >/dev/null 2>&1 || true
  curl -I -sSf https://pypi.org/simple/ >/dev/null 2>&1 || echo "WARN: cannot reach pypi.org"
  curl -I -sSf https://github.com/ >/dev/null 2>&1 || echo "WARN: cannot reach github.com"

  mkdir -p /workspace
  tar -xz -C /workspace

  python3 -m venv --system-site-packages /opt/gsdg-venv
  . /opt/gsdg-venv/bin/activate

  python -m pip install --upgrade pip
  python -m pip install -r /workspace/requirements.txt
  python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" tokenizers safetensors
  python -m pip install --upgrade setuptools wheel cmake ninja setuptools_scm

  export CUDA_HOME=/usr/local/cuda-13.1
  export VLLM_USE_PRECOMPILED=0
  export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
  export MAX_JOBS=8
  export CMAKE_BUILD_PARALLEL_LEVEL=8
  python -m pip install --no-build-isolation --no-deps "vllm @ git+https://github.com/vllm-project/vllm.git@v0.17.1"

  # Minimal runtime deps for `vllm serve` (keep this lean to avoid huge optional installs).
  python -m pip install fastapi "uvicorn[standard]" pydantic prometheus_client

  python -c "import vllm; print(\"vllm_version=\" + vllm.__version__)"
'

ROOTFS=""
for candidate in "$HOME/.local/share/enroot/$NAME" "/dev/shm/$(id -nu)/enrootdata/$NAME"; do
  if [[ -d "$candidate" ]]; then
    ROOTFS="$candidate"
    break
  fi
done

if [[ -z "$ROOTFS" ]]; then
  log "ERROR: cannot find enroot rootfs for $NAME"
  exit 2
fi

log "mksquashfs export"
mksquashfs "$ROOTFS" "$OUT_SQSH" -comp zstd -b 131072 -noappend -all-root

log "DONE"
