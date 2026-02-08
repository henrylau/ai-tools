import io
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "yolo26x.pt")

DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

yolo_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model

    logger.info("Loading YOLO26 model: %s", MODEL_PATH)
    yolo_model = YOLO(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model.to(device)
    logger.info("Model loaded on %s", device)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="YOLO Object Detection Service",
    description="Object detection API using YOLO26",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "model": MODEL_PATH,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD),
    classes: str = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if not 0 <= score_threshold <= 1:
        raise HTTPException(status_code=400, detail="score_threshold must be between 0 and 1")
    if not 0 <= iou_threshold <= 1:
        raise HTTPException(status_code=400, detail="iou_threshold must be between 0 and 1")

    contents = await file.read()
    img_array = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    filter_classes = None
    if classes:
        try:
            filter_classes = [int(c.strip()) for c in classes.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid class indices format")

    results = yolo_model.predict(
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
