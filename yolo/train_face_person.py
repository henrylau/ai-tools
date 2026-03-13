"""
Train a YOLO26 model to detect faces and persons.

Downloads the WIDERFace dataset (YOLO format) and COCO person-only subset,
merges them into a unified 2-class dataset, then trains a YOLO26 model.

Classes:
    0: person
    1: face

Usage:
    # Download datasets and train with defaults (yolo26n)
    python train_face_person.py

    # Use a larger model
    python train_face_person.py --model yolo26s.pt

    # Skip download if datasets already exist
    python train_face_person.py --skip-download

    # Custom training parameters
    python train_face_person.py --model yolo26s.pt --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO

DATASET_DIR = Path("datasets/face-person")
WIDERFACE_DIR = Path("datasets/_widerface_raw")
COCO_PERSON_DIR = Path("datasets/_coco_person_raw")

WIDERFACE_URLS = {
    "train": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
    "val": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",
    "split": "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip",
}

# COCO 2017 person subset - we use val set for manageable size, train for more data
COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

CLASS_NAMES = {0: "person", 1: "face"}


def download_file(url, dest):
    """Download a file with progress."""
    dest = Path(dest)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done ({dest.stat().st_size / 1e6:.1f} MB)")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def parse_widerface_annotations(annot_path):
    """Parse WIDERFace annotation file into {image_path: [(x, y, w, h), ...]}."""
    annotations = {}
    with open(annot_path) as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            image_path = line
            count = int(f.readline().strip())
            boxes = []
            if count == 0:
                # WIDERFace uses "0 0 0 0 0 0 0 0 0 0" as placeholder for no faces
                f.readline()
            else:
                for _ in range(count):
                    parts = f.readline().strip().split()
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    if w > 0 and h > 0:
                        boxes.append((x, y, w, h))
            if boxes:
                annotations[image_path] = boxes
    return annotations


def convert_widerface(split, image_subdir):
    """Convert WIDERFace split to YOLO format as face (class 1)."""
    annot_file = WIDERFACE_DIR / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    if not annot_file.exists():
        print(f"  Annotation file not found: {annot_file}")
        return 0

    annotations = parse_widerface_annotations(annot_file)
    out_split = "train" if split == "train" else "valid"
    img_out = DATASET_DIR / out_split / "images"
    lbl_out = DATASET_DIR / out_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for rel_path, boxes in annotations.items():
        src_img = WIDERFACE_DIR / image_subdir / rel_path
        if not src_img.exists():
            continue

        # Read image dimensions
        from PIL import Image
        with Image.open(src_img) as im:
            iw, ih = im.size

        # Use flat filename to avoid collisions
        flat_name = rel_path.replace("/", "_")
        stem = Path(flat_name).stem

        # Copy image
        dst_img = img_out / flat_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write YOLO label (class 1 = face)
        label_path = lbl_out / f"{stem}.txt"
        lines = []
        # Preserve existing person labels if file exists
        if label_path.exists():
            lines = label_path.read_text().strip().split("\n")

        for x, y, w, h in boxes:
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines.append(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines) + "\n")
        count += 1

    return count


def convert_coco_persons(split):
    """Extract person annotations from COCO and convert to YOLO format (class 0)."""
    import json

    coco_split = "train2017" if split == "train" else "val2017"
    annot_name = f"instances_{coco_split}.json"
    annot_path = COCO_PERSON_DIR / "annotations" / annot_name
    img_dir = COCO_PERSON_DIR / coco_split

    if not annot_path.exists():
        print(f"  COCO annotations not found: {annot_path}")
        return 0

    with open(annot_path) as f:
        coco = json.load(f)

    # Find person category id
    person_cat_id = None
    for cat in coco["categories"]:
        if cat["name"] == "person":
            person_cat_id = cat["id"]
            break

    if person_cat_id is None:
        print("  'person' category not found in COCO")
        return 0

    # Build image id -> info mapping
    images = {img["id"]: img for img in coco["images"]}

    # Group person annotations by image
    person_annots = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != person_cat_id:
            continue
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in person_annots:
            person_annots[img_id] = []
        person_annots[img_id].append(ann["bbox"])  # [x, y, w, h]

    out_split = "train" if split == "train" else "valid"
    img_out = DATASET_DIR / out_split / "images"
    lbl_out = DATASET_DIR / out_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, boxes in person_annots.items():
        img_info = images[img_id]
        src_img = img_dir / img_info["file_name"]
        if not src_img.exists():
            continue

        iw, ih = img_info["width"], img_info["height"]
        dst_img = img_out / f"coco_{img_info['file_name']}"

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        stem = f"coco_{Path(img_info['file_name']).stem}"
        label_path = lbl_out / f"{stem}.txt"

        lines = []
        for x, y, w, h in boxes:
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            count += 1

    return count


def download_and_prepare_dataset():
    """Download WIDERFace + COCO person data and merge into a single dataset."""
    print("\n=== Step 1: Download WIDERFace ===")
    download_file(WIDERFACE_URLS["train"], WIDERFACE_DIR / "WIDER_train.zip")
    download_file(WIDERFACE_URLS["val"], WIDERFACE_DIR / "WIDER_val.zip")
    download_file(WIDERFACE_URLS["split"], WIDERFACE_DIR / "wider_face_split.zip")

    extract_zip(WIDERFACE_DIR / "WIDER_train.zip", WIDERFACE_DIR)
    extract_zip(WIDERFACE_DIR / "WIDER_val.zip", WIDERFACE_DIR)
    extract_zip(WIDERFACE_DIR / "wider_face_split.zip", WIDERFACE_DIR)

    print("\n=== Step 2: Download COCO (person subset) ===")
    download_file(COCO_URLS["val_images"], COCO_PERSON_DIR / "val2017.zip")
    download_file(COCO_URLS["annotations"], COCO_PERSON_DIR / "annotations_trainval2017.zip")
    print("  Note: COCO train2017 images (~18GB) are optional. Using val2017 for training.")
    print("  To include train2017, download manually to datasets/_coco_person_raw/train2017.zip")

    extract_zip(COCO_PERSON_DIR / "val2017.zip", COCO_PERSON_DIR)
    extract_zip(COCO_PERSON_DIR / "annotations_trainval2017.zip", COCO_PERSON_DIR)

    # If train2017 images exist, extract them too
    train_zip = COCO_PERSON_DIR / "train2017.zip"
    if train_zip.exists():
        extract_zip(train_zip, COCO_PERSON_DIR)

    print("\n=== Step 3: Convert WIDERFace to YOLO format ===")
    # Clean output directory
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    n = convert_widerface("train", "WIDER_train/images")
    print(f"  WIDERFace train: {n} images")
    n = convert_widerface("val", "WIDER_val/images")
    print(f"  WIDERFace val: {n} images")

    print("\n=== Step 4: Convert COCO persons to YOLO format ===")
    # Use COCO val2017 as our training data (5K images, manageable size)
    n = convert_coco_persons("val")
    print(f"  COCO val (person): {n} images -> merged into valid split")

    # If train images available, use them
    if (COCO_PERSON_DIR / "train2017").exists():
        n = convert_coco_persons("train")
        print(f"  COCO train (person): {n} images -> merged into train split")

    print("\n=== Step 5: Write data.yaml ===")
    write_data_yaml()

    # Print dataset stats
    for split in ["train", "valid"]:
        img_dir = DATASET_DIR / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.iterdir()))
            print(f"  {split}: {count} images")


def write_data_yaml():
    """Write the YOLO dataset configuration file."""
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": CLASS_NAMES,
    }
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Written: {yaml_path}")


def train(args):
    """Train the YOLO model."""
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found. Run without --skip-download first.")
        return

    print(f"\n=== Training {args.model} ===")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device or 'auto'}")
    print()

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    print(f"\nTo use in mivolo service:")
    print(f"  Set YOLO_DETECTOR_PATH={best_weights.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26 face+person detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt",
                        help="Model variant: yolo26n.pt or yolo26s.pt (default: yolo26n.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size, -1 for auto (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', '0', 'mps', or None for auto")
    parser.add_argument("--project", type=str, default="runs/face-person",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (default: derived from model name)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download, use existing data")
    args = parser.parse_args()

    if args.name is None:
        model_stem = Path(args.model).stem  # e.g. "yolo26n"
        args.name = f"{model_stem}_face_person"

    if not args.skip_download:
        download_and_prepare_dataset()

    train(args)


if __name__ == "__main__":
    main()
