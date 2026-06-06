FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl ca-certificates rsync && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

WORKDIR /opt/nanogpt-kubetorch
COPY pyproject.toml uv.lock ./
COPY nanogpt_kubetorch ./nanogpt_kubetorch
COPY scripts ./scripts
COPY vendor/nanogpt ./vendor/nanogpt
RUN uv pip install --system --compile-bytecode . && \
    uv pip install --system --compile-bytecode numpy transformers datasets tiktoken wandb tqdm requests

RUN uv pip install --system --compile-bytecode \
    'kubetorch[client] @ git+https://github.com/cezarc1/kubetorch.git@47f8b30a07a3d816bdb63805d9828a7ed5bcb39c#subdirectory=python_client'

WORKDIR /workspace
