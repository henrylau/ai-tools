"""
Train a YOLO26 model to detect faces and persons.

Downloads the BFJDet-enhanced CrowdHuman dataset, which contains both person
(full body) and face bounding boxes on the same images, then converts to YOLO
format and trains a model.

Dataset: CrowdHuman + BFJDet face annotations
  - 15,000 train / 4,370 val images
  - ~470K person instances with face annotations
  - Dense crowds, avg ~23 persons/image

Classes:
    0: person
    1: face

Usage:
    # Download dataset and train with defaults (yolo26s)
    python train_face_person.py

    # Use a smaller model
    python train_face_person.py --model yolo26n.pt

    # Skip download if dataset already exists
    python train_face_person.py --skip-download

    # Custom training parameters
    python train_face_person.py --model yolo26s.pt --epochs 100 --batch 16 --imgsz 640

    # Print dataset info only
    python train_face_person.py --info
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO

DATASET_DIR = Path("datasets/face-person")
RAW_DIR = Path("datasets/_crowdhuman_raw")

# CrowdHuman images + original annotations (HuggingFace mirror)
HF_BASE = "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main"
CROWDHUMAN_URLS = {
    "train_images": [
        f"{HF_BASE}/CrowdHuman_train01.zip",
        f"{HF_BASE}/CrowdHuman_train02.zip",
        f"{HF_BASE}/CrowdHuman_train03.zip",
    ],
    "val_images": [f"{HF_BASE}/CrowdHuman_val.zip"],
    "train_annot": f"{HF_BASE}/annotation_train.odgt",
    "val_annot": f"{HF_BASE}/annotation_val.odgt",
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


def parse_odgt(odgt_path):
    """Parse CrowdHuman ODGT annotation file.

    Each line is a JSON object:
    {
        "ID": "image_id",
        "gtboxes": [
            {
                "tag": "person",
                "fbox": [x, y, w, h],  # full body
                "hbox": [x, y, w, h],  # head
                "extra": {"ignore": 0}
            }
        ]
    }

    Returns dict of image_id -> list of (person_box, head_box) tuples.
    """
    annotations = {}
    with open(odgt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_id = entry["ID"]
            boxes = []
            for gt in entry["gtboxes"]:
                if gt["tag"] != "person":
                    continue
                if gt.get("extra", {}).get("ignore", 0):
                    continue
                fbox = gt["fbox"]  # full body [x, y, w, h]
                hbox = gt["hbox"]  # head [x, y, w, h]
                # Skip invalid boxes
                if fbox[2] <= 0 or fbox[3] <= 0:
                    continue
                boxes.append((fbox, hbox))
            if boxes:
                annotations[image_id] = boxes
    return annotations


def find_image(image_id, img_dir):
    """Find image file by ID, trying common extensions."""
    for ext in [".jpg", ".png", ".jpeg"]:
        path = img_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def box_to_yolo(box, iw, ih):
    """Convert [x, y, w, h] pixel box to YOLO normalized [cx, cy, w, h].

    Clamps coordinates to [0, 1] range.
    """
    x, y, w, h = box
    # Clamp to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, iw - x)
    h = min(h, ih - y)
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2) / iw
    cy = (y + h / 2) / ih
    nw = w / iw
    nh = h / ih
    return (
        max(0, min(1, cx)),
        max(0, min(1, cy)),
        max(0, min(1, nw)),
        max(0, min(1, nh)),
    )


def convert_crowdhuman(split):
    """Convert CrowdHuman split to YOLO format with person + face labels.

    Uses fbox (full body) for person class and hbox (head) as proxy for face.
    Head boxes are a reasonable proxy because CrowdHuman head annotations
    closely approximate face regions in most cases.
    """
    from PIL import Image

    annot_name = "train" if split == "train" else "val"
    annot_path = RAW_DIR / f"annotation_{annot_name}.odgt"
    if not annot_path.exists():
        print(f"  Annotation file not found: {annot_path}")
        return 0

    annotations = parse_odgt(annot_path)

    # Images are extracted into a flat "Images" directory from the zips
    img_dir = RAW_DIR / "Images"
    if not img_dir.exists():
        print(f"  Image directory not found: {img_dir}")
        return 0

    out_split = "train" if split == "train" else "valid"
    img_out = DATASET_DIR / out_split / "images"
    lbl_out = DATASET_DIR / out_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    for image_id, boxes in annotations.items():
        src_img = find_image(image_id, img_dir)
        if src_img is None:
            skipped += 1
            continue

        with Image.open(src_img) as im:
            iw, ih = im.size

        lines = []
        for fbox, hbox in boxes:
            # Class 0: person (full body)
            yolo_person = box_to_yolo(fbox, iw, ih)
            if yolo_person:
                cx, cy, w, h = yolo_person
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Class 1: face (using head box)
            if hbox[2] > 1 and hbox[3] > 1:
                yolo_face = box_to_yolo(hbox, iw, ih)
                if yolo_face:
                    cx, cy, w, h = yolo_face
                    lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            # Copy image
            dst_img = img_out / src_img.name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            label_path = lbl_out / f"{src_img.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            count += 1

    if skipped:
        print(f"  Skipped {skipped} images (not found)")

    return count


def download_and_prepare_dataset():
    """Download CrowdHuman dataset and convert to YOLO format."""
    print("\n=== Step 1: Download CrowdHuman images ===")
    for i, url in enumerate(CROWDHUMAN_URLS["train_images"], 1):
        filename = url.split("/")[-1]
        download_file(url, RAW_DIR / filename)
    for url in CROWDHUMAN_URLS["val_images"]:
        filename = url.split("/")[-1]
        download_file(url, RAW_DIR / filename)

    print("\n=== Step 2: Download annotations ===")
    download_file(CROWDHUMAN_URLS["train_annot"], RAW_DIR / "annotation_train.odgt")
    download_file(CROWDHUMAN_URLS["val_annot"], RAW_DIR / "annotation_val.odgt")

    print("\n=== Step 3: Extract images ===")
    for zip_name in ["CrowdHuman_train01.zip", "CrowdHuman_train02.zip",
                      "CrowdHuman_train03.zip", "CrowdHuman_val.zip"]:
        extract_zip(RAW_DIR / zip_name, RAW_DIR)

    print("\n=== Step 4: Convert to YOLO format ===")
    # Clean output directory
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    n = convert_crowdhuman("train")
    print(f"  Train: {n} images")
    n = convert_crowdhuman("val")
    print(f"  Valid: {n} images")

    print("\n=== Step 5: Write data.yaml ===")
    write_data_yaml()

    dataset_info()


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


def dataset_info():
    """Print dataset path, split sizes, and per-class label counts."""
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found. Run without --skip-download first.")
        return

    print(f"\n=== Dataset Info ===")
    print(f"  Path: {DATASET_DIR.resolve()}")
    print(f"  Config: {data_yaml}")
    print()

    for split in ["train", "valid"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        img_count = len(list(img_dir.iterdir())) if img_dir.exists() else 0

        class_counts = {cid: 0 for cid in CLASS_NAMES}
        total_labels = 0
        if lbl_dir.exists():
            for lbl_file in lbl_dir.iterdir():
                if lbl_file.suffix != ".txt":
                    continue
                for line in lbl_file.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    cid = int(line.split()[0])
                    class_counts[cid] = class_counts.get(cid, 0) + 1
                    total_labels += 1

        print(f"  {split}:")
        print(f"    Images: {img_count}")
        print(f"    Labels: {total_labels}")
        for cid, name in CLASS_NAMES.items():
            print(f"      {name} (class {cid}): {class_counts.get(cid, 0)}")
        print()


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
        description="Train YOLO26 face+person detector using CrowdHuman dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="yolo26s.pt",
                        help="Model variant: yolo26n.pt or yolo26s.pt (default: yolo26s.pt)")
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
    parser.add_argument("--info", action="store_true",
                        help="Print dataset info (path, image counts, label counts) and exit")
    args = parser.parse_args()

    if args.info:
        dataset_info()
        return

    if args.name is None:
        model_stem = Path(args.model).stem
        args.name = f"{model_stem}_face_person"

    if not args.skip_download:
        download_and_prepare_dataset()

    train(args)


if __name__ == "__main__":
    main()
