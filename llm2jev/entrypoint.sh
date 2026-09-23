#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-huihui-ai/Huihui-Qwen3.5-2B-abliterated}"
MODEL_PATH="${MODEL_PATH:-/models/${MODEL_ID}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
DOWNLOAD_MODEL_ON_START="${DOWNLOAD_MODEL_ON_START:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$(dirname "$MODEL_PATH")" /models

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  if [[ "${DOWNLOAD_MODEL_ON_START}" == "1" ]]; then
    echo "[entrypoint] Downloading ${MODEL_ID} -> ${MODEL_PATH}"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${MODEL_ID}", local_dir="${MODEL_PATH}")
print("[entrypoint] download complete")
PY
  else
    echo "[entrypoint] Missing model at ${MODEL_PATH} and DOWNLOAD_MODEL_ON_START!=1" >&2
    exit 1
  fi
fi

# SGLang is the default llm2jev backend on Linux/NVIDIA.
# Remaining args are forwarded to SGLang's native server.
# shellcheck disable=SC2086
exec llm2jev-serve --backend sglang \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --tokenizer-worker-num 1 \
  ${EXTRA_ARGS}
