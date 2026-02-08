import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import cv2
import huggingface_hub
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Global model references
mivolo_model = None
image_processor = None
mivolo_config = None
yolo_model = None

MIVOLO_REPO = "iitolstykh/mivolo_v2"
YOLO_DETECTOR_REPO = "iitolstykh/YOLO-Face-Person-Detector"
YOLO_DETECTOR_FILE = "yolov8x_person_face.pt"

# Class indices for the face+person YOLO model
YOLO_FACE_CLASS = 1
YOLO_PERSON_CLASS = 0

# Default thresholds matching the official demo
DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.7


class InferenceMode(str, Enum):
    persons_and_faces = "persons_and_faces"
    persons_only = "persons_only"
    faces_only = "faces_only"


def load_models():
    global mivolo_model, image_processor, mivolo_config, yolo_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading models on device=%s dtype=%s", device, dtype)

    # MiVoLo v2
    mivolo_config = AutoConfig.from_pretrained(MIVOLO_REPO, trust_remote_code=True)
    mivolo_model = AutoModelForImageClassification.from_pretrained(
        MIVOLO_REPO, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    mivolo_model.eval()
    image_processor = AutoImageProcessor.from_pretrained(
        MIVOLO_REPO, trust_remote_code=True
    )

    # Download and load the official YOLO face+person detector
    detector_path = huggingface_hub.hf_hub_download(
        YOLO_DETECTOR_REPO, YOLO_DETECTOR_FILE
    )
    yolo_model = YOLO(detector_path)
    logger.info("All models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="MiVoLo v2 Age & Gender Estimation", lifespan=lifespan)


def detect_persons_and_faces(
    image: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
):
    """Detect persons and faces using the official YOLO face+person detector."""
    results = yolo_model(
        image, verbose=False, conf=score_threshold, iou=iou_threshold
    )[0]

    persons = []
    faces = []
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf[0]), 3)
        entry = (x1, y1, x2, y2, conf)
        if cls == YOLO_PERSON_CLASS:
            persons.append(entry)
        elif cls == YOLO_FACE_CLASS:
            faces.append(entry)

    return persons, faces


def match_face_to_person(face, persons):
    """Find the person bounding box that contains or best overlaps the face."""
    fx1, fy1, fx2, fy2, _ = face
    face_cx, face_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

    for px1, py1, px2, py2, _ in persons:
        if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
            return (px1, py1, px2, py2)
    return None


def estimate_face_from_person(person_box, image_shape):
    """Estimate a face region from the upper portion of a person bounding box."""
    px1, py1, px2, py2, _ = person_box
    pw = px2 - px1
    ph = py2 - py1
    margin = int(pw * 0.15)
    fx1 = max(0, px1 + margin)
    fy1 = max(0, py1)
    fx2 = min(image_shape[1], px2 - margin)
    fy2 = min(image_shape[0], py1 + int(ph * 0.25))
    if fx2 > fx1 and fy2 > fy1:
        return (fx1, fy1, fx2, fy2)
    return None


def run_mivolo(face_crop, body_crop):
    """Run MiVoLo v2 inference on a single face/body pair."""
    faces_list = [face_crop] if face_crop is not None else [None]
    bodies_list = [body_crop] if body_crop is not None else [None]

    faces_input = image_processor(images=faces_list)["pixel_values"]
    body_input = image_processor(images=bodies_list)["pixel_values"]

    faces_input = faces_input.to(dtype=mivolo_model.dtype, device=mivolo_model.device)
    body_input = body_input.to(dtype=mivolo_model.dtype, device=mivolo_model.device)

    with torch.no_grad():
        output = mivolo_model(faces_input=faces_input, body_input=body_input)

    age = round(output.age_output[0].item(), 1)
    id2label = mivolo_config.gender_id2label
    gender = id2label[output.gender_class_idx[0].item()]
    gender_confidence = round(output.gender_probs[0].item() * 100, 1)

    return {"age": age, "gender": gender, "gender_confidence": gender_confidence}


@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    score_threshold: float = Form(DEFAULT_SCORE_THRESHOLD),
    iou_threshold: float = Form(DEFAULT_IOU_THRESHOLD),
    mode: InferenceMode = Form(InferenceMode.persons_and_faces),
):
    """Upload an image and get age/gender predictions for all detected persons.

    - **score_threshold**: Detector confidence threshold (0-1, default 0.4)
    - **iou_threshold**: NMS IoU threshold (0-1, default 0.7)
    - **mode**: persons_and_faces | persons_only | faces_only
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if not 0 <= score_threshold <= 1:
        raise HTTPException(status_code=400, detail="score_threshold must be between 0 and 1")
    if not 0 <= iou_threshold <= 1:
        raise HTTPException(status_code=400, detail="iou_threshold must be between 0 and 1")

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    persons, faces = detect_persons_and_faces(image, score_threshold, iou_threshold)

    # Filter based on inference mode
    use_persons = mode in (InferenceMode.persons_and_faces, InferenceMode.persons_only)
    use_faces = mode in (InferenceMode.persons_and_faces, InferenceMode.faces_only)

    if not use_persons:
        persons = []
    if not use_faces:
        faces = []

    if not persons and not faces:
        return {"predictions": [], "message": "No persons or faces detected"}

    results = []
    used_faces = set()

    # Process each detected person
    for person_box in persons:
        matched_face = None
        if use_faces:
            for i, face in enumerate(faces):
                if i not in used_faces and match_face_to_person(face, [person_box]):
                    matched_face = face
                    used_faces.add(i)
                    break

        # If no face detected, estimate from person box
        if matched_face is None and use_faces:
            est = estimate_face_from_person(person_box, image.shape)
            if est:
                matched_face = (*est, 0.0)

        face_crop = None
        if matched_face is not None:
            fx1, fy1, fx2, fy2 = matched_face[:4]
            face_crop = image[fy1:fy2, fx1:fx2]

        px1, py1, px2, py2, _ = person_box
        body_crop = image[py1:py2, px1:px2]

        try:
            prediction = run_mivolo(face_crop, body_crop)
            prediction["person_box"] = [px1, py1, px2, py2]
            if matched_face:
                prediction["face_box"] = list(matched_face[:4])
            results.append(prediction)
        except Exception as e:
            logger.warning("MiVoLo inference failed for a person: %s", e)

    # Process unmatched faces (no person body found)
    if use_faces:
        for i, face in enumerate(faces):
            if i in used_faces:
                continue
            fx1, fy1, fx2, fy2, _ = face
            face_crop = image[fy1:fy2, fx1:fx2]
            try:
                prediction = run_mivolo(face_crop, None)
                prediction["face_box"] = [fx1, fy1, fx2, fy2]
                results.append(prediction)
            except Exception as e:
                logger.warning("MiVoLo inference failed for a face: %s", e)

    return {
        "predictions": results,
        "total_detected": len(results),
        "persons_detected": len(persons),
        "faces_detected": len(faces),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "settings": {
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "mode": mode.value,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
