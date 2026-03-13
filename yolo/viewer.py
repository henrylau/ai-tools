import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

DATASET_DIR = Path(os.environ.get("DATASET_PATH", "datasets/aquarium-combined"))
VIEWER_HTML = Path(__file__).parent / "viewer.html"

SPLITS = {"train": "train", "valid": "valid", "test": "test"}

router = APIRouter(prefix="/viewer")

# ── dataset loaded once at import time ────────────────────────────────────────

def _load_dataset():
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        raise RuntimeError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    classes = [{"id": i, "name": n} for i, n in enumerate(cfg["names"])]

    splits = {}
    for split, folder in SPLITS.items():
        img_dir = DATASET_DIR / folder / "images"
        if img_dir.exists():
            splits[split] = sorted(p.name for p in img_dir.iterdir() if p.is_file())

    return classes, splits


try:
    _classes, _split_images = _load_dataset()
    _dataset_available = True
except RuntimeError:
    _classes, _split_images = [], {}
    _dataset_available = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_image_path(split: str, index: int) -> Path:
    images = _split_images.get(split)
    if not images:
        raise HTTPException(status_code=404, detail=f"Split '{split}' not found")
    if index < 0 or index >= len(images):
        raise HTTPException(status_code=404, detail="Index out of range")
    return DATASET_DIR / split / "images" / images[index]


def _get_annotations(split: str, index: int) -> list[dict]:
    images = _split_images[split]
    stem = Path(images[index]).stem
    label_path = DATASET_DIR / split / "labels" / f"{stem}.txt"

    if not label_path.exists():
        return []

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append({
                "class_id": cls_id,
                "class_name": _classes[cls_id]["name"] if cls_id < len(_classes) else str(cls_id),
                "bbox_norm": {"cx": cx, "cy": cy, "w": w, "h": h},
            })
    return annotations


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def viewer_page():
    return HTMLResponse(VIEWER_HTML.read_text())


@router.get("/info")
async def dataset_info():
    return {
        "name": DATASET_DIR.name,
        "classes": _classes,
        "splits": {split: len(images) for split, images in _split_images.items()},
    }


@router.get("/annotations")
async def annotations(
    split: str = Query(...),
    index: int = Query(...),
):
    if split not in _split_images:
        raise HTTPException(status_code=404, detail=f"Split '{split}' not found")

    images = _split_images[split]
    total = len(images)

    if index < 0 or index >= total:
        raise HTTPException(status_code=404, detail="Index out of range")

    return {
        "split": split,
        "index": index,
        "total": total,
        "filename": images[index],
        "annotations": _get_annotations(split, index),
    }


@router.get("/image")
async def image(
    split: str = Query(...),
    index: int = Query(...),
):
    img_path = _get_image_path(split, index)
    return FileResponse(img_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    _app = FastAPI()
    _app.include_router(router)
    uvicorn.run(_app, host="0.0.0.0", port=8000)
