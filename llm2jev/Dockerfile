FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

ENV SERVED_MODEL_NAME=local-model
ENV PORT=30000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN git clone https://github.com/Yinsongxu/LLM2Jev.git && cd LLM2Jev && uv sync --extra transformers && uv add torchvision

ENV PATH="/app/LLM2Jev/.venv/bin:$PATH"
ENV MULTIMODAL=1
ENV HOST=0.0.0.0

COPY serve.py test.html /app/

CMD ["python", "/app/serve.py"]