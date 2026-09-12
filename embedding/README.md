# Face Embedding Service

Face embedding API powered by the [InsightFace](https://github.com/deepinsight/insightface)
`buffalo_l` model pack (SCRFD detector + ArcFace `w600k_r50`) and FastAPI. Takes face
clips and returns 512-d embedding vectors that can be compared with cosine similarity.

Unlike the other services, `/embed` accepts **more than one face clip per request** —
all clips are detected, aligned and then embedded in a single batched forward pass.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (returns GPU status, model and embedding dim) |
| `POST` | `/embed` | Embed one or more face clips |
| `POST` | `/predict` | Alias of `/embed` |
| `GET` | `/test` | Browser test page (upload clips, view similarity matrix) |

### `POST /embed`

Upload one or more face clips and receive a 512-d embedding for each.

**Parameters (multipart form):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `files` | file[] | required | One or more face clips — repeat the field for a batch |
| `det_threshold` | float | 0.5 | Face detector confidence threshold (0-1) |
| `align` | bool | true | Detect and ArcFace-align the face before embedding |
| `normalize` | bool | true | L2-normalize embeddings (dot product = cosine similarity) |
| `require_face` | bool | false | Report clips with no detected face as errors instead of embedding the raw clip |

Alignment matters: ArcFace was trained on 112×112 crops warped onto five reference
landmarks, so `align=true` (the default) gives noticeably better embeddings. On this
repo's test clips, embedding a tight crop as-is scored **0.53** cosine similarity against
an aligned crop of the same face, while aligning it first scored **0.98**.

Because face clips are usually cropped tight and the SCRFD detector needs some context
around a face, the clip is padded with a replicated border (`DET_PAD`, 25% by default)
before detection — without that padding, tight clips frequently get no detection at all.
Clips missed at the first padding are retried at `DET_PAD_RETRY` (60%). `face_box` is
reported in the coordinates of the clip you uploaded.

When no face is found at any padding, the clip is assumed to already be a face crop and
is resized as-is — such results are flagged with `"aligned": false`. Set
`require_face=true` to reject them instead.

A clip that cannot be decoded is reported in `errors` rather than failing the whole
batch, so one bad file never costs you the rest of the request.

**Example:**

```bash
curl -X POST http://localhost:8000/embed \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg" \
  -F "files=@face3.jpg" \
  -F "det_threshold=0.5"
```

**Response:**

```json
{
  "embeddings": [
    {
      "index": 0,
      "filename": "face1.jpg",
      "aligned": true,
      "det_score": 0.8821,
      "face_box": [12.4, 8.1, 108.7, 121.0],
      "image_size": {"width": 128, "height": 128},
      "dim": 512,
      "norm": 23.4117,
      "embedding": [0.031204, -0.012887, "…"]
    }
  ],
  "errors": [],
  "total": 3,
  "embedded": 3,
  "failed": 0,
  "settings": {
    "model": "buffalo_l",
    "det_threshold": 0.5,
    "align": true,
    "normalize": true,
    "require_face": false
  }
}
```

`embedding` is L2-normalized by default, so the cosine similarity of two faces is just
their dot product. As a rule of thumb with `buffalo_l`, a similarity of **≥ 0.4** means
the same identity and **< 0.3** means different identities. `norm` is the length of the
raw feature before normalization and is a useful quality signal — low values tend to mean
a blurry or badly cropped clip.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The server starts at `http://localhost:8000`. API docs are available at `/docs`, and the
test page at `/test`. The `buffalo_l` pack (~330 MB) downloads on first run to
`~/.insightface/models/`.

## Run with Docker

```bash
docker build -t embedding-service .
docker run --rm -p 8000:8000 embedding-service
```

### GPU

**The default image is CPU-only.** It installs the `onnxruntime` wheel, which has no
CUDA support compiled in at all, so a container started with `--gpus all` still runs
every model on the CPU. CI publishes both variants — pull the `-gpu` tag:

```bash
docker pull ghcr.io/<owner>/<repo>/embedding:latest-gpu
```

or build it yourself:

```bash
docker build --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-gpu -t embedding-service-gpu .
docker run --rm --gpus all -p 8000:8000 embedding-service-gpu
```

Or with compose, which reserves an NVIDIA device and so builds the CUDA image by
default:

```bash
docker compose up --build                                    # GPU
ONNXRUNTIME_PACKAGE=onnxruntime docker compose up --build     # CPU
```

That build swaps `onnxruntime` for `onnxruntime-gpu` **and** installs the CUDA 13 and
cuDNN 9 runtime libraries it links against (`nvidia-cuda-runtime`, `nvidia-cublas`,
`nvidia-curand`, `nvidia-cudnn-cu13`), because the `python:3.11-slim` base has no CUDA
in it — the GPU wheel on its own finds no `libcudart.so.13` and quietly falls back to
the CPU. `libcuda.so.1` is deliberately not installed: that one comes from the host
driver, which `nvidia-container-toolkit` injects when you pass `--gpus all`. The CUDA
libraries add ~1.7 GB to the image.

Check what you actually got — `/health` reports the providers the live session is
running on, not the ones that were requested:

```bash
curl -s localhost:8000/health
{"status":"ok","gpu":true,"providers":["CUDAExecutionProvider","CPUExecutionProvider"],...}
```

If `gpu` is `false` on a GPU build, the startup log says so explicitly ("Requested CUDA
but the session is running on ..."). The usual causes are a missing `--gpus all`, no
NVIDIA driver on the host, or an image built without the build arg.

CI builds `linux/amd64` only: the CPU image under the usual tags, and the CUDA image
under the same tags suffixed with `-gpu` (`latest-gpu`, `sha-<commit>-gpu`). Both
packages publish `aarch64` wheels too, so either image can be built for arm64 locally
with `docker build --platform linux/arm64`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `buffalo_l` | InsightFace model pack (e.g. `buffalo_l`, `buffalo_s`) |
| `INSIGHTFACE_ROOT` | `~/.insightface` | Model cache directory |
| `DET_SIZE` | 640 | Detector input size |
| `DET_PAD` | 0.25 | Border padding added before detection, as a fraction of the clip |
| `DET_PAD_RETRY` | 0.6 | Wider padding retried for clips missed at `DET_PAD` |
| `MAX_FILES` | 64 | Maximum clips accepted per request |
| `MAX_BATCH_SIZE` | 32 | Maximum clips per recognition forward pass |
| `ORT_NUM_THREADS` | CPU count | ONNX Runtime intra-op threads; defaults to the number of cores the container is allowed |

## Troubleshooting

**`pthread_setaffinity_np failed ... error code: 22` repeated at startup.** ONNX Runtime
sizes its thread pool from the host's cores and pins each thread to one of them, which
fails on every thread when the container is restricted to a subset of CPUs (a `cpuset`,
a Kubernetes CPU limit, `--cpuset-cpus`). The service sets the thread count explicitly
from `len(os.sched_getaffinity(0))`, which turns the pinning off and stops the pool from
oversubscribing the cores it is actually allowed. Override with `ORT_NUM_THREADS`.

**The service says it is on CPU.** See [GPU](#gpu) above — the default image has no CUDA
support built in.

## Models

Downloaded automatically at build time (or on first run):

| Model | Source | Purpose |
|-------|--------|---------|
| `buffalo_l` | [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) | SCRFD-10G face detection + ArcFace `w600k_r50` 512-d embeddings |
