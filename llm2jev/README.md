# LLM2Jev CUDA Service

Vision-language model API powered by [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev) and the default model `huihui-ai/Huihui-Qwen3.5-2B-abliterated`. Runs on NVIDIA GPUs with SGLang for high-throughput inference.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/systemone` | Run structured output generation |

### `POST /v1/systemone`

Generate structured outputs from vision-language model queries.

**Example:**

```bash
curl -s http://127.0.0.1:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "state": "The package never arrived.",
    "questions": {
      "delivery": {"type": "noul", "instructions": "Is this a delivery issue?"}
    }
  }'
```

## Requirements (host)

- Docker Engine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA driver compatible with CUDA 12.8

## Run with Docker

```bash
docker build -t llm2jev-cuda .
docker run --gpus all --rm -p 30000:30000 \
  -e HF_HOME=/models \
  -v llm2jev-models:/models \
  --shm-size=8g \
  llm2jev-cuda
```

Or with compose:

```bash
docker compose up --build
```

First start downloads the HF model into the `llm2jev-models` volume (~4–5 GB).

### Transformers Backend

For environments where SGLang installation fails, use the lighter transformers-only image:

```bash
docker build -f Dockerfile.transformers -t llm2jev-cuda-tf .
docker run --gpus all --rm -p 30000:30000 \
  -v llm2jev-models:/models \
  llm2jev-cuda-tf
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `huihui-ai/Huihui-Qwen3.5-2B-abliterated` | Hugging Face model identifier |
| `MODEL_PATH` | `/models/${MODEL_ID}` | Local path where model is cached |
| `SERVED_MODEL_NAME` | `local-model` | Model name exposed via API |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `30000` | Server port |
| `TP_SIZE` | `1` | Tensor parallelism size (SGLang) |
| `MEM_FRACTION_STATIC` | `0.85` | GPU memory fraction (SGLang) |
| `DOWNLOAD_MODEL_ON_START` | `1` | Auto-download model on first run |
| `DTYPE` | `bfloat16` | Data type (Transformers backend only) |
| `HF_TOKEN` | unset | Hugging Face token for gated models |

## Health Check

The container includes a health check that polls `/v1/models`:

```bash
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

## Notes

- **Linux/NVIDIA** path uses SGLang (`--backend sglang`), matching upstream recommendations for throughput.
- Model weights are downloaded at runtime, not baked into the image.
- For Mac Apple Silicon, use the upstream MLX conversion (`./start-mlx.sh`) instead of this CUDA image.
- The default model is a transformers/safetensors checkpoint (not GGUF or MLX).
