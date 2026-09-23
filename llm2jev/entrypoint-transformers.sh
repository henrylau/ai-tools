#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${MODEL_ID:-huihui-ai/Huihui-Qwen3.5-2B-abliterated}"
MODEL_PATH="${MODEL_PATH:-/models/${MODEL_ID}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
DTYPE="${DTYPE:-bfloat16}"
DOWNLOAD_MODEL_ON_START="${DOWNLOAD_MODEL_ON_START:-1}"
mkdir -p "$(dirname "$MODEL_PATH")"
if [[ ! -f "${MODEL_PATH}/config.json" && "${DOWNLOAD_MODEL_ON_START}" == "1" ]]; then
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${MODEL_ID}", local_dir="${MODEL_PATH}")
PY
fi
exec llm2jev-serve --backend transformers \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" --port "${PORT}" \
  --dtype "${DTYPE}" --multimodal
