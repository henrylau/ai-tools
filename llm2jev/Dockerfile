# CUDA + Yinsongxu/LLM2Jev (SGLang) for NVIDIA GPUs
# Default model: huihui-ai/Huihui-Qwen3.5-2B-abliterated (HF VLM)
#
# Build:
#   docker build -t llm2jev-cuda .
# Run (host needs NVIDIA Container Toolkit):
#   docker run --gpus all -p 30000:30000 \
#     -e HF_HOME=/models -v llm2jev-models:/models \
#     llm2jev-cuda

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    MODEL_ID=huihui-ai/Huihui-Qwen3.5-2B-abliterated \
    MODEL_PATH=/models/huihui-ai/Huihui-Qwen3.5-2B-abliterated \
    SERVED_MODEL_NAME=local-model \
    HOST=0.0.0.0 \
    PORT=30000 \
    TP_SIZE=1 \
    MEM_FRACTION_STATIC=0.85 \
    DOWNLOAD_MODEL_ON_START=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      python3-pip \
      build-essential \
      libnuma1 \
      libibverbs1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llm2jev

# Pin to a known-good upstream; bump LLM2JEV_REF to refresh.
ARG LLM2JEV_REPO=https://github.com/Yinsongxu/LLM2Jev.git
ARG LLM2JEV_REF=main
RUN git clone --depth 1 --branch "${LLM2JEV_REF}" "${LLM2JEV_REPO}" . \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install -U pip setuptools wheel \
    && uv pip install -e .[sglang] \
    && uv pip install huggingface_hub pillow \
    && rm -rf /root/.cache/uv /root/.cache/pip

ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

VOLUME ["/models"]
EXPOSE 30000

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/v1/models" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
