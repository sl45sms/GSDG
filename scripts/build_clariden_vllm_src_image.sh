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
# Bind-mount $SCRATCH at a dedicated path for caching (safe: it does not
# shadow any system paths inside the rootfs).
tar -C /users/p-skarvelis/GSDG -cz requirements.txt src scripts Readme.md | \
	enroot start --rw --mount "${SCRATCH}:/mnt/cache" "$NAME" bash -lc '
  set -euo pipefail
  set -x

  install_flashinfer() {
    if python -c "import flashinfer" >/dev/null 2>&1; then
      return 0
    fi

    if python -m pip install "flashinfer-python==0.6.4" "flashinfer-cubin==0.6.4"; then
      return 0
    fi

    rm -rf /tmp/flashinfer-src
    python -m pip install --upgrade "setuptools>=77" build
    git clone --recursive --branch v0.6.4 https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer-src
    export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-9.0a}"
    (
      cd /tmp/flashinfer-src
      python -m pip install -v .
      cd flashinfer-cubin
      python -m build --no-isolation --wheel
      python -m pip install dist/*.whl
    )
  }

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

  # Keep Transformers within vLLM declared constraint (<5) for stability.
  # If you need bleeding-edge Qwen3.5 support, build vLLM from main and set
  # INSTALL_TRANSFORMERS_MAIN=1.
  if [[ "${INSTALL_TRANSFORMERS_MAIN:-0}" == "1" ]]; then
    python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" tokenizers safetensors
  else
    python -m pip install "transformers>=4.56.0,<5" tokenizers safetensors
  fi

  # vLLM 0.17.1 declares setuptools<81 for Py3.12; keep within that range.
  python -m pip install --upgrade "setuptools<81" wheel cmake ninja setuptools_scm

  export CUDA_HOME=/usr/local/cuda-13.1
  export VLLM_USE_PRECOMPILED=0
  export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
  export MAX_JOBS=8
  export CMAKE_BUILD_PARALLEL_LEVEL=8

  mkdir -p /mnt/cache/wheels
  VLLM_WHEEL_GLOB="/mnt/cache/wheels/vllm-0.17.1+cu131-*.whl"
  if compgen -G "$VLLM_WHEEL_GLOB" >/dev/null; then
    python -m pip install --no-deps $VLLM_WHEEL_GLOB
  else
    python -m pip wheel --no-build-isolation --no-deps -w /mnt/cache/wheels \
      "vllm @ git+https://github.com/vllm-project/vllm.git@v0.17.1"
    python -m pip install --no-deps $VLLM_WHEEL_GLOB
  fi

  # Minimal runtime deps for `vllm serve`.
  # vLLM imports some utilities unconditionally at CLI startup (e.g. cbor2 hashing),
  # so we install a small set of lightweight packages while still avoiding the
  # large optional stack (ray, opencv, torchaudio, etc.).
  python -m pip install \
    fastapi "uvicorn[standard]" pydantic prometheus_client prometheus-fastapi-instrumentator \
    cbor2 blake3 cachetools cloudpickle diskcache msgspec pybase64 setproctitle ijson \
    "gguf>=0.17.0" "compressed-tensors==0.13.0" "depyf==0.20.0"

  # Required for a Ray-backed multi-node vLLM path on Clariden.
  python -m pip install "ray[default]"

  # Qwen3.5 GDN prefill on GH200 currently requires FlashInfer in this vLLM build.
  install_flashinfer

  # Required by the OpenAI-compatible entrypoint.
  python -m pip install "openai>=1.99.1,<2.25.0" "openai-harmony>=0.0.3" mistral-common

  # Imported by the OpenAI server stack (SageMaker router).
  python -m pip install "model-hosting-container-standards>=0.1.13,<1.0.0"

  # Structured-output helpers imported by vLLM V1.
  python -m pip install \
    "llguidance>=1.3.0,<1.4.0" \
    "lm-format-enforcer==0.11.3" \
    partial-json-parser \
    "xgrammar==0.1.29" \
    "outlines_core==0.2.11"

  python -c "import vllm; print(\"vllm_version=\" + vllm.__version__)"
  python -c "import flashinfer; print(\"flashinfer_version=\" + flashinfer.__version__)"
  python -m flashinfer show-config
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
mksquashfs "$ROOTFS" "$OUT_SQSH" -comp zstd -b 131072 -noappend -all-root \
  -e etc/motd etc/xthostname mnt/cache

log "DONE"
