import glob
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from insightface.model_zoo.model_zoo import ModelRouter
from insightface.utils import face_align
from insightface.utils.storage import ensure_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")
INSIGHTFACE_ROOT = os.path.expanduser(
    os.environ.get("INSIGHTFACE_ROOT", "~/.insightface")
)
DET_SIZE = int(os.environ.get("DET_SIZE", "640"))
# Face clips are usually cropped tight, which the detector struggles with, so the
# clip is padded with a replicated border before detection. Without it a tight
# clip often yields no detection at all, and the unaligned fallback costs a lot
# of accuracy. The second value is a wider retry for clips missed at the first.
DET_PADS = tuple(sorted({
    float(os.environ.get("DET_PAD", "0.25")),
    float(os.environ.get("DET_PAD_RETRY", "0.6")),
}))
MAX_FILES = int(os.environ.get("MAX_FILES", "64"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))


def _num_threads():
    """Threads for the ONNX Runtime intra-op pool.

    Left to itself, ORT sizes the pool from the host's cores and pins each
    thread to one of them. In a container restricted to a subset of CPUs that
    pinning fails on every thread (pthread_setaffinity_np ... error code: 22)
    and the pool badly oversubscribes the cores it is actually allowed. Sizing
    it explicitly turns the pinning off.
    """
    override = os.environ.get("ORT_NUM_THREADS")
    if override:
        return max(1, int(override))
    try:
        return len(os.sched_getaffinity(0))  # cores this process may run on
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


NUM_THREADS = _num_threads()

# The detector is prepared with a low floor so that the per-request
# det_threshold can filter detections freely in either direction.
MIN_DET_THRESHOLD = 0.1
DEFAULT_DET_THRESHOLD = 0.5

det_model = None
rec_model = None
rec_input_size = (112, 112)
embedding_dim = 512
providers = []


def _build_models(requested_providers, sess_options):
    """Build the detection and recognition sessions from the model pack.

    This is what FaceAnalysis does, minus the models the service has no use for
    and plus session options, which FaceAnalysis gives no way to pass through.
    """
    model_dir = ensure_available("models", MODEL_NAME, root=INSIGHTFACE_ROOT)
    det = rec = None

    for onnx_file in sorted(glob.glob(os.path.join(model_dir, "*.onnx"))):
        model = ModelRouter(onnx_file).get_model(
            sess_options=sess_options, providers=requested_providers
        )
        # buffalo_l also ships landmark and gender/age models — route them the
        # way insightface does, then drop the ones this service never calls.
        if model is None or model.taskname not in ("detection", "recognition"):
            del model
            continue
        if model.taskname == "detection":
            det = model
        else:
            rec = model

    if det is None or rec is None:
        raise RuntimeError(
            f"{MODEL_NAME} does not provide both a detection and a recognition model"
        )
    return det, rec


@asynccontextmanager
async def lifespan(app: FastAPI):
    global det_model, rec_model, rec_input_size, embedding_dim, providers

    # get_available_providers() reports CUDA whenever the GPU build of
    # onnxruntime is installed, even with no CUDA libraries or device present,
    # so it says what to ask for — not what we end up running on.
    requested = [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if p in onnxruntime.get_available_providers()
    ]
    ctx_id = 0 if "CUDAExecutionProvider" in requested else -1

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = NUM_THREADS

    logger.info(
        "Loading %s — requested providers %s, %d intra-op thread(s)",
        MODEL_NAME, requested, NUM_THREADS,
    )

    try:
        det_model, rec_model = _build_models(requested, sess_options)
    except Exception as e:
        if ctx_id < 0:
            raise
        logger.warning("CUDA session failed (%s) — retrying on CPU", e)
        requested, ctx_id = ["CPUExecutionProvider"], -1
        det_model, rec_model = _build_models(requested, sess_options)

    det_model.prepare(
        ctx_id, input_size=(DET_SIZE, DET_SIZE), det_thresh=MIN_DET_THRESHOLD
    )
    rec_model.prepare(ctx_id)

    # What the sessions actually run on, which is not necessarily what was
    # requested: a CUDA session quietly falls back to CPU when the CUDA/cuDNN
    # libraries or the device are missing. Report the real thing so that
    # fallback is visible instead of being mistaken for slow hardware.
    providers = rec_model.session.get_providers()
    if "CUDAExecutionProvider" in requested and "CUDAExecutionProvider" not in providers:
        logger.warning(
            "Requested CUDA but the session is running on %s — check that the image "
            "was built with ONNXRUNTIME_PACKAGE=onnxruntime-gpu and that the "
            "container can see an NVIDIA device (--gpus all)",
            providers,
        )

    rec_input_size = tuple(rec_model.input_size)
    # output_shape is (batch, dim); batch may be a symbolic dim
    try:
        embedding_dim = int(rec_model.output_shape[1])
    except (TypeError, ValueError, IndexError):
        logger.warning("Could not read embedding dim from the model, assuming 512")
    logger.info(
        "Model loaded on %s — input %s, embedding dim %d",
        providers, rec_input_size, embedding_dim,
    )

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Face Embedding Service",
    description="Face clip embedding API using the InsightFace buffalo_l model pack",
    lifespan=lifespan,
)

TEST_HTML = Path(__file__).parent / "test.html"


@app.get("/test", response_class=HTMLResponse)
async def test_page():
    return HTMLResponse(TEST_HTML.read_text())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": "CUDAExecutionProvider" in providers,
        "model": MODEL_NAME,
        "embedding_dim": embedding_dim,
        "providers": providers,
        "intra_op_threads": NUM_THREADS,
    }


def _decode_image(contents: bytes):
    """Decode raw upload bytes into a BGR image, or None if undecodable."""
    img_array = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def _align_face(img, det_threshold: float):
    """Detect the main face in a clip and ArcFace-align it to the model input.

    Detection runs on a padded copy of the clip (see DET_PADS) and the landmarks
    it returns are warped straight from that padded image; only the reported box
    is mapped back to clip coordinates.

    Returns (aligned_crop, det_score, face_box), or (None, None, None) when no
    face passes det_threshold at any padding.
    """
    for pad_ratio in DET_PADS:
        pad = int(round(max(img.shape[:2]) * pad_ratio))
        padded = cv2.copyMakeBorder(
            img, pad, pad, pad, pad, cv2.BORDER_REPLICATE
        ) if pad > 0 else img

        bboxes, kpss = det_model.detect(padded, max_num=1, metric="default")
        if bboxes is None or len(bboxes) == 0 or kpss is None or len(kpss) == 0:
            continue

        score = float(bboxes[0][4])
        if score < det_threshold:
            continue

        aligned = face_align.norm_crop(
            padded, landmark=kpss[0], image_size=rec_input_size[0]
        )
        box = [round(float(v) - pad, 1) for v in bboxes[0][:4]]
        return aligned, round(score, 4), box

    return None, None, None


def _get_feats(crops):
    """Run the recognition model over aligned crops, batched where possible."""
    feats = []
    for start in range(0, len(crops), MAX_BATCH_SIZE):
        chunk = crops[start:start + MAX_BATCH_SIZE]
        try:
            out = rec_model.get_feat(chunk)
        except Exception as e:
            logger.warning(
                "Batched embedding failed (%s) — falling back to one crop at a time", e
            )
            out = np.concatenate([rec_model.get_feat(c) for c in chunk], axis=0)
        feats.append(np.asarray(out))
    return np.concatenate(feats, axis=0)


@app.post("/embed")
async def embed(
    files: List[UploadFile] = File(...),
    det_threshold: float = Form(DEFAULT_DET_THRESHOLD),
    align: bool = Form(True),
    normalize: bool = Form(True),
    require_face: bool = Form(False),
):
    """Embed one or more face clips into 512-d ArcFace vectors.

    - **files**: one or more face crops (repeat the field to send a batch)
    - **det_threshold**: face detector confidence threshold (0-1, default 0.5)
    - **align**: detect and ArcFace-align the face before embedding (default true)
    - **normalize**: L2-normalize embeddings so dot product is cosine similarity
    - **require_face**: report clips with no detected face as errors instead of
      embedding the resized clip as-is
    """
    if not 0 <= det_threshold <= 1:
        raise HTTPException(status_code=400, detail="det_threshold must be between 0 and 1")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"Too many files: {len(files)} (max {MAX_FILES})"
        )

    crops = []
    meta = []
    errors = []

    for index, file in enumerate(files):
        if not file.content_type or not file.content_type.startswith("image/"):
            errors.append({
                "index": index,
                "filename": file.filename,
                "error": "File must be an image",
            })
            continue

        img = _decode_image(await file.read())
        if img is None:
            errors.append({
                "index": index,
                "filename": file.filename,
                "error": "Could not decode image",
            })
            continue

        aligned, det_score, face_box = (None, None, None)
        if align:
            aligned, det_score, face_box = _align_face(img, det_threshold)

        if aligned is None and require_face:
            errors.append({
                "index": index,
                "filename": file.filename,
                "error": "No face detected above det_threshold",
            })
            continue

        # A clip that the detector misses is already a face crop — embed it as-is.
        crops.append(aligned if aligned is not None else cv2.resize(img, rec_input_size))
        meta.append({
            "index": index,
            "filename": file.filename,
            "aligned": aligned is not None,
            "det_score": det_score,
            "face_box": face_box,
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
        })

    embeddings = []
    if crops:
        feats = _get_feats(crops)
        for entry, feat in zip(meta, feats):
            norm = float(np.linalg.norm(feat))
            vector = feat / norm if normalize and norm > 0 else feat
            entry["dim"] = int(vector.shape[0])
            entry["norm"] = round(norm, 4)
            entry["embedding"] = [round(float(v), 6) for v in vector]
            embeddings.append(entry)

    return {
        "embeddings": embeddings,
        "errors": errors,
        "total": len(files),
        "embedded": len(embeddings),
        "failed": len(errors),
        "settings": {
            "model": MODEL_NAME,
            "det_threshold": det_threshold,
            "align": align,
            "normalize": normalize,
            "require_face": require_face,
        },
    }


# Keep /predict as an alias for /embed, matching the other services
@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    det_threshold: float = Form(DEFAULT_DET_THRESHOLD),
    align: bool = Form(True),
    normalize: bool = Form(True),
    require_face: bool = Form(False),
):
    return await embed(
        files=files,
        det_threshold=det_threshold,
        align=align,
        normalize=normalize,
        require_face=require_face,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
