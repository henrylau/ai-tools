import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", "datasets"))
VIEWER_HTML = Path(__file__).parent / "viewer.html"

SPLIT_FOLDERS = ["train", "valid", "test"]

router = APIRouter(prefix="/viewer")

# ── dataset discovery and loading ─────────────────────────────────────────────

def _discover_datasets():
    """Find all datasets under DATASETS_ROOT that contain a data.yaml."""
    datasets = {}
    if not DATASETS_ROOT.exists():
        return datasets
    for yaml_path in sorted(DATASETS_ROOT.rglob("data.yaml")):
        dataset_dir = yaml_path.parent
        name = str(dataset_dir.relative_to(DATASETS_ROOT))
        datasets[name] = dataset_dir
    return datasets


def _load_dataset(dataset_dir):
    """Load classes and split image lists from a dataset directory."""
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", {})
    if isinstance(names, list):
        classes = [{"id": i, "name": n} for i, n in enumerate(names)]
    else:
        classes = [{"id": i, "name": names[i]} for i in sorted(names.keys())]

    splits = {}
    for folder in SPLIT_FOLDERS:
        img_dir = dataset_dir / folder / "images"
        if img_dir.exists():
            splits[folder] = sorted(p.name for p in img_dir.iterdir() if p.is_file())

    return classes, splits


_available_datasets = _discover_datasets()
_dataset_available = len(_available_datasets) > 0

# Cache loaded datasets
_cache = {}


def _get_dataset(name):
    """Get or load a dataset by name."""
    if name not in _available_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    if name not in _cache:
        _cache[name] = _load_dataset(_available_datasets[name])
    return _available_datasets[name], _cache[name][0], _cache[name][1]


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_image_path(dataset_dir, split_images, split: str, index: int) -> Path:
    images = split_images.get(split)
    if not images:
        raise HTTPException(status_code=404, detail=f"Split '{split}' not found")
    if index < 0 or index >= len(images):
        raise HTTPException(status_code=404, detail="Index out of range")
    return dataset_dir / split / "images" / images[index]


def _get_annotations(dataset_dir, classes, split_images, split: str, index: int) -> list[dict]:
    images = split_images[split]
    stem = Path(images[index]).stem
    label_path = dataset_dir / split / "labels" / f"{stem}.txt"

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
                "class_name": classes[cls_id]["name"] if cls_id < len(classes) else str(cls_id),
                "bbox_norm": {"cx": cx, "cy": cy, "w": w, "h": h},
            })
    return annotations


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def viewer_page():
    return HTMLResponse(VIEWER_HTML.read_text())


@router.get("/datasets")
async def list_datasets():
    """List all available datasets."""
    result = []
    for name, dataset_dir in _available_datasets.items():
        classes, splits = _load_dataset(dataset_dir) if name not in _cache else _cache[name]
        if name not in _cache:
            _cache[name] = (classes, splits)
        result.append({
            "name": name,
            "classes": classes,
            "splits": {s: len(imgs) for s, imgs in splits.items()},
        })
    return result


@router.get("/info")
async def dataset_info(dataset: str = Query(None)):
    if dataset is None:
        # Default to first available dataset
        if not _available_datasets:
            raise HTTPException(status_code=404, detail="No datasets found")
        dataset = next(iter(_available_datasets))

    dataset_dir, classes, splits = _get_dataset(dataset)
    return {
        "name": dataset,
        "classes": classes,
        "splits": {split: len(images) for split, images in splits.items()},
    }


@router.get("/annotations")
async def annotations(
    split: str = Query(...),
    index: int = Query(...),
    dataset: str = Query(None),
):
    if dataset is None:
        if not _available_datasets:
            raise HTTPException(status_code=404, detail="No datasets found")
        dataset = next(iter(_available_datasets))

    dataset_dir, classes, split_images = _get_dataset(dataset)

    if split not in split_images:
        raise HTTPException(status_code=404, detail=f"Split '{split}' not found")

    images = split_images[split]
    total = len(images)

    if index < 0 or index >= total:
        raise HTTPException(status_code=404, detail="Index out of range")

    return {
        "dataset": dataset,
        "split": split,
        "index": index,
        "total": total,
        "filename": images[index],
        "annotations": _get_annotations(dataset_dir, classes, split_images, split, index),
    }


@router.get("/image")
async def image(
    split: str = Query(...),
    index: int = Query(...),
    dataset: str = Query(None),
):
    if dataset is None:
        if not _available_datasets:
            raise HTTPException(status_code=404, detail="No datasets found")
        dataset = next(iter(_available_datasets))

    dataset_dir, _, split_images = _get_dataset(dataset)
    img_path = _get_image_path(dataset_dir, split_images, split, index)
    return FileResponse(img_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    _app = FastAPI()
    _app.include_router(router)
    uvicorn.run(_app, host="0.0.0.0", port=8000)
