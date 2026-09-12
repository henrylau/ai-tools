FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .

# insightface ships a C extension and has no wheels, so it needs a compiler and
# cython/numpy up front. The toolchain is dropped again in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && uv pip install --system --no-cache "setuptools<70" wheel "cython<3" "numpy>=1.24.0,<2.0.0" \
    && uv pip install --system --no-cache --no-build-isolation -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# The image is CPU-only by default. For NVIDIA GPUs build with:
#   docker build --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-gpu .
# That swaps in the CUDA build of onnxruntime *and* the CUDA 13 / cuDNN 9
# runtime libraries it links against — the base image has no CUDA in it, so
# the GPU package alone would silently fall back to CPU. Only libcuda.so.1 is
# left out: that one comes from the host driver via nvidia-container-toolkit.
ARG ONNXRUNTIME_PACKAGE=onnxruntime
ARG ORT_VERSION=1.30.0
RUN if [ "$ONNXRUNTIME_PACKAGE" = "onnxruntime-gpu" ]; then \
        uv pip uninstall --system onnxruntime \
        && uv pip install --system --no-cache \
            "onnxruntime-gpu==${ORT_VERSION}" \
            "nvidia-cuda-runtime>=13,<14" \
            "nvidia-cublas>=13,<14" \
            "nvidia-curand>=10,<11" \
            "nvidia-cudnn-cu13>=9,<10"; \
    elif [ "$ONNXRUNTIME_PACKAGE" != "onnxruntime" ]; then \
        echo "unsupported ONNXRUNTIME_PACKAGE: $ONNXRUNTIME_PACKAGE" >&2; exit 1; \
    fi

# Where the CUDA wheels above put their libraries. Harmless on the CPU image,
# where these directories simply do not exist.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY app.py test.html ./

# Pre-download the buffalo_l model pack at build time (download only, no loading)
ARG MODEL_NAME=buffalo_l
RUN python -c "\
from insightface.utils.storage import ensure_available; \
ensure_available('models', '${MODEL_NAME}', root='/root/.insightface'); \
print('${MODEL_NAME} downloaded successfully')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
