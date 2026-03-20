FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/gsdg-venv \
    PATH=/opt/gsdg-venv/bin:$PATH

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /workspace/requirements.txt \
    && python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" tokenizers safetensors \
    && python -m pip install --extra-index-url https://wheels.vllm.ai/nightly vllm

COPY src /workspace/src
COPY scripts /workspace/scripts
COPY Readme.md /workspace/Readme.md

ENV PYTHONPATH=/workspace/src
