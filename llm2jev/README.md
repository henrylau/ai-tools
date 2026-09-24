# LLM2Jev Decision Service

Jev-style structured decision API powered by [LLM2Jev](https://github.com/Yinsongxu/LLM2Jev). Turns a local language (or vision-language) model into Choice / Score / Noul answers via prefill-only scoring — no token-by-token decoding.

This image runs the Transformers backend with multimodal support enabled.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/test` | Browser test page |
| `GET` | `/v1/models` | List served model names |
| `POST` | `/v1/systemone` | Evaluate a Jev-style decision request |

Set `LLM2JEV_API_KEY` (or pass `--api-key` at serve time) to require a Bearer token on `/v1` routes.

### `POST /v1/systemone`

Submit evidence (`state`) and one or more questions. The request `model` must match `SERVED_MODEL_NAME`.

**Question types:**

| Type | Purpose | Criteria |
|------|---------|----------|
| `choice` | Pick one option from a labeled set | Object of option id → description (≥ 2) |
| `score` | Weighted level on an ordered scale | Array of level labels (≥ 2) |
| `noul` | Yes/no probability for a proposition | Optional |

**Text example:**

```bash
curl http://localhost:30000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "state": "My parcel arrived two weeks late, and my card was charged twice.",
    "questions": {
      "delivery": {
        "type": "noul",
        "instructions": "Is this a delivery issue?"
      },
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "shipping": "Delivery",
          "billing": "Billing",
          "technical": "Product fault"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Low", "Medium", "High"]
      }
    }
  }'
```

**Response (shape):**

```json
{
  "model": "local-model",
  "answers": {
    "delivery": { "type": "noul", "noul": 0.91 },
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.70,
      "probabilities": { "shipping": 0.15, "billing": 0.70, "technical": 0.15 }
    },
    "urgency": {
      "type": "score",
      "score": 1.24,
      "confidence": 0.63,
      "legend": { "0": "Low", "1": "Medium", "2": "High" },
      "probabilities": { "0": 0.10, "1": 0.56, "2": 0.34 }
    }
  },
  "usage": { "input_tokens": 120, "output_tokens": null }
}
```

### Multimodal

With `--multimodal` (enabled in this Dockerfile), `state` or a question's `instructions` can be a multimodal object. Images may be HTTP(S) URLs, local paths, `file://` URIs, or base64 data URLs:

```json
{
  "model": "local-model",
  "state": {
    "type": "multimodal",
    "content": [
      { "type": "text", "text": "Inspect the image" },
      { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }
    ]
  },
  "questions": {
    "person": {
      "type": "noul",
      "instructions": "Is a person visible in the image?"
    }
  }
}
```

Use a vision-language model for image requests.

## Run with Docker

Build from this directory:

```bash
docker build -t llm2jev-service .
```

Run (mount weights and set `MODEL_PATH`):

```bash
docker run --rm --gpus all \
  -p 30000:30000 \
  -e MODEL_PATH=/models/your-model \
  -e SERVED_MODEL_NAME=local-model \
  -v /path/to/your-model:/models/your-model:ro \
  llm2jev-service
```

CPU-only hosts can omit `--gpus all`; Transformers falls back to CPU.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | (required) | Path to a Hugging Face-compatible causal / VLM checkpoint inside the container |
| `SERVED_MODEL_NAME` | `local-model` | Alias clients must send as `model` |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `30000` | HTTP listen port |
| `MULTIMODAL` | `1` | Enable image inputs (`0` / `false` to disable) |
| `LLM2JEV_API_KEY` | unset | Optional Bearer token for `/v1` endpoints |

## Browser test page

The container serves the System One API and the test UI from the same process (`serve.py` wraps LLM2Jev’s Transformers app and adds `/test`).

After the service is up, open:

```text
http://localhost:30000/test
```

Leave **API base URL** blank to call the same origin. You can still open `test.html` from disk for local tweaks — set the base URL to `http://localhost:30000` in that case.

## Upstream

- Project: [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev)
- Usage: [docs/usage.md](https://github.com/Yinsongxu/LLM2Jev/blob/main/docs/usage.md)
- Multimodal: [docs/multimodal.md](https://github.com/Yinsongxu/LLM2Jev/blob/main/docs/multimodal.md)
