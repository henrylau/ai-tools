# YOLO Object Detection Service

Object detection API powered by [YOLO26](https://docs.ultralytics.com/) and FastAPI. Includes tools for training on custom datasets and a built-in dataset viewer.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (returns GPU status and model info) |
| `POST` | `/predict` | Run object detection on an uploaded image |
| `GET` | `/viewer` | Dataset viewer UI (available when a dataset is present) |

### `POST /predict`

Upload an image and receive bounding box predictions.

**Parameters (multipart form):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Image file to analyze |
| `score_threshold` | float | 0.25 | Minimum confidence score (0-1) |
| `iou_threshold` | float | 0.45 | NMS IoU threshold (0-1) |
| `classes` | string | all | Comma-separated class IDs to filter (e.g. `0,2,5`) |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@photo.jpg" \
  -F "score_threshold=0.3"
```

**Response:**

```json
{
  "predictions": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9213,
      "box": [120.5, 45.0, 350.2, 480.0]
    }
  ],
  "total_detected": 1,
  "image_size": {"width": 640, "height": 480},
  "settings": {"score_threshold": 0.3, "iou_threshold": 0.45, "classes": null}
}
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The server starts at `http://localhost:8000`. API docs are available at `/docs`.

## Run with Docker

```bash
docker build -t yolo-service .
docker run --rm -p 8000:8000 yolo-service
```

For GPU support:

```bash
docker run --rm --gpus all -p 8000:8000 yolo-service
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `yolo26x.pt` | Path to YOLO model weights |
| `DATASET_PATH` | `datasets/aquarium-combined` | Path to dataset for the viewer |

## Training

### 1. Download a dataset

Requires a [Roboflow](https://roboflow.com) API key:

```bash
export ROBOFLOW_API_KEY="your_key"
python download_dataset.py
```

Or specify a custom dataset:

```bash
python download_dataset.py \
  --workspace roboflow-universe-projects \
  --project hard-hat-universe-0dy7t \
  --version 1
```

### 2. Train

```bash
python train.py --data datasets/aquarium-combined/data.yaml --epochs 50 --name my_model
```

**Training options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolo26n.pt` | Model variant (yolo26n/s/m/l/x.pt) |
| `--data` | `coco8.yaml` | Dataset config YAML |
| `--epochs` | 10 | Training epochs |
| `--batch` | 16 | Batch size (-1 for auto) |
| `--imgsz` | 640 | Input image size |
| `--device` | auto | Device (`cpu`, `0`, `0,1`) |

Results are saved to `runs/train/<name>/`.

### Train Face + Person Detector

`train_face_person.py` downloads WIDERFace and COCO person datasets, merges them into a unified 2-class dataset (`person`, `face`), and trains a YOLO26 model.

```bash
# Full pipeline: download datasets + train (yolo26n by default)
python train_face_person.py

# Use yolo26s for better accuracy
python train_face_person.py --model yolo26s.pt

# Skip download if datasets already exist
python train_face_person.py --skip-download

# Custom parameters
python train_face_person.py --model yolo26s.pt --epochs 100 --batch 8 --device mps
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `yolo26n.pt` | Model variant (`yolo26n.pt` or `yolo26s.pt`) |
| `--epochs` | 50 | Training epochs |
| `--batch` | 16 | Batch size (-1 for auto) |
| `--imgsz` | 640 | Input image size |
| `--device` | auto | Device (`cpu`, `0`, `mps`) |
| `--skip-download` | false | Skip dataset download |

Results are saved to `runs/face-person/<model>_face_person/weights/best.pt`.

## Dataset Viewer

When a dataset is present at `DATASET_PATH`, the `/viewer` endpoint serves an interactive UI for browsing images and annotations across train/valid/test splits.
