import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

from viewer import router as viewer_router, _dataset_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DETECT_MODEL_PATH = os.environ.get("DETECT_MODEL_PATH", os.environ.get("MODEL_PATH", "yolo26x.pt"))
CLS_MODEL_PATH = os.environ.get("CLS_MODEL_PATH", "")

DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

detect_model = None
cls_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detect_model, cls_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading detection model: %s", DETECT_MODEL_PATH)
    detect_model = YOLO(DETECT_MODEL_PATH)
    detect_model.to(device)
    logger.info("Detection model loaded on %s", device)

    if CLS_MODEL_PATH:
        logger.info("Loading classification model: %s", CLS_MODEL_PATH)
        cls_model = YOLO(CLS_MODEL_PATH)
        cls_model.to(device)
        logger.info("Classification model loaded on %s", device)
    else:
        logger.info("No CLS_MODEL_PATH set — /classify endpoint disabled")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="YOLO Detection & Classification Service",
    description="Object detection and image classification API using YOLO26",
    lifespan=lifespan,
)

if _dataset_available:
    app.include_router(viewer_router)

TEST_HTML = Path(__file__).parent / "test.html"


@app.get("/test", response_class=HTMLResponse)
async def test_page():
    return HTMLResponse(TEST_HTML.read_text())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "detect_model": DETECT_MODEL_PATH,
        "cls_model": CLS_MODEL_PATH or None,
    }


async def _read_image(file: UploadFile):
    """Read and decode an uploaded image file."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD),
    classes: str = Form(None),
):
    if not 0 <= score_threshold <= 1:
        raise HTTPException(status_code=400, detail="score_threshold must be between 0 and 1")
    if not 0 <= iou_threshold <= 1:
        raise HTTPException(status_code=400, detail="iou_threshold must be between 0 and 1")

    img = await _read_image(file)

    filter_classes = None
    if classes:
        try:
            filter_classes = [int(c.strip()) for c in classes.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid class indices format")

    results = detect_model.predict(
        img,
        conf=score_threshold,
        iou=iou_threshold,
        classes=filter_classes,
        verbose=False,
    )

    predictions = []
    result = results[0]
    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        predictions.append({
            "class_id": cls_id,
            "class_name": names[cls_id],
            "confidence": round(conf, 4),
            "box": [round(v, 1) for v in xyxy],
        })

    return {
        "predictions": predictions,
        "total_detected": len(predictions),
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "settings": {
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "classes": filter_classes,
        },
    }


# Keep /predict as alias for /detect for backwards compatibility
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD),
    classes: str = Form(None),
):
    return await detect(file=file, score_threshold=score_threshold,
                        iou_threshold=iou_threshold, classes=classes)


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):
    if cls_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded. Set CLS_MODEL_PATH env var.")

    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    img = await _read_image(file)

    results = cls_model.predict(img, verbose=False)
    result = results[0]
    probs = result.probs
    names = result.names

    # Get top-k predictions
    top_indices = probs.top5 if top_k >= 5 else probs.top5[:top_k]
    top_confs = probs.top5conf.tolist()

    predictions = []
    for i, idx in enumerate(top_indices):
        if i >= top_k:
            break
        predictions.append({
            "class_id": idx,
            "class_name": names[idx],
            "confidence": round(top_confs[i], 4),
        })

    return {
        "predictions": predictions,
        "top_class": names[probs.top1],
        "top_confidence": round(float(probs.top1conf), 4),
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
