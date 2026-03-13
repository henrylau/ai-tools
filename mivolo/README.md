# MiVoLo Age & Gender Estimation Service

Age and gender estimation API powered by [MiVoLo v2](https://github.com/WildChlamydia/MiVOLO) and FastAPI. Uses a YOLO face+person detector to locate people in images, then estimates age and gender for each detected person.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (returns GPU status) |
| `POST` | `/predict` | Estimate age and gender for people in an image |

### `POST /predict`

Upload an image and receive age/gender predictions for all detected persons and faces.

**Parameters (multipart form):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Image file to analyze |
| `score_threshold` | float | 0.4 | Detector confidence threshold (0-1) |
| `iou_threshold` | float | 0.7 | NMS IoU threshold (0-1) |
| `mode` | string | `persons_and_faces` | Detection mode (see below) |

**Detection modes:**

- `persons_and_faces` — detect both persons and faces, match faces to bodies
- `persons_only` — detect only person bounding boxes
- `faces_only` — detect only faces

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@photo.jpg" \
  -F "score_threshold=0.4" \
  -F "mode=persons_and_faces"
```

**Response:**

```json
{
  "predictions": [
    {
      "age": 32.5,
      "gender": "male",
      "gender_confidence": 97.3,
      "person_box": [120, 45, 350, 480],
      "face_box": [180, 50, 290, 160]
    }
  ],
  "total_detected": 1,
  "persons_detected": 1,
  "faces_detected": 1,
  "image_size": {"width": 640, "height": 480},
  "settings": {"score_threshold": 0.4, "iou_threshold": 0.7, "mode": "persons_and_faces"}
}
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git
python app.py
```

The server starts at `http://localhost:8000`. API docs are available at `/docs`.

## Run with Docker

```bash
docker build -t mivolo-service .
docker run --rm -p 8000:8000 mivolo-service
```

For GPU support:

```bash
docker run --rm --gpus all -p 8000:8000 mivolo-service
```

## Models

The following models are downloaded automatically at build time (or on first run):

| Model | Source | Purpose |
|-------|--------|---------|
| MiVoLo v2 | [iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2) | Age & gender estimation |
| YOLO Face+Person | [iitolstykh/YOLO-Face-Person-Detector](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector) | Person and face detection |
