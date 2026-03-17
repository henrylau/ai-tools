"""
YOLO26 Training Script

Usage:
    python train.py                          # Train with defaults (COCO8, yolo26n)
    python train.py --model yolo26s.pt       # Use a different model size
    python train.py --data coco128.yaml      # Use a larger sample dataset
    python train.py --epochs 50 --batch 8    # Custom training parameters
    python train.py --prepare-dataset /path/to/yolov8_dataset  # Prepare & split dataset, then train
"""

import argparse
import json
import os
import random
import shutil
import glob
import yaml
from ultralytics import YOLO


def prepare_dataset(input_path, output_path=None, train_ratio=0.8, seed=42, classes_yaml=None):
    """
    Prepare a dataset from YOLOv8 format and split into train/val sets.

    Expects input_path to contain:
      - images/  (with .jpg, .jpeg, .png files)
      - labels/  (with matching .txt annotation files)

    Produces output_path with:
      - images/train/, images/val/
      - labels/train/, labels/val/
      - dataset.yaml
    """
    input_path = os.path.abspath(input_path)
    if output_path is None:
        output_path = os.path.join(input_path, "split")

    images_dir = os.path.join(input_path, "images")
    labels_dir = os.path.join(input_path, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Collect image files
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files.sort()

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    # Filter to only images that have matching label files
    paired = []
    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.isfile(label_path):
            paired.append((img_path, label_path))

    if not paired:
        raise ValueError("No image/label pairs found. Ensure label .txt filenames match image filenames.")

    print(f"Found {len(paired)} image/label pairs (out of {len(image_files)} images)")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(paired)
    split_idx = int(len(paired) * train_ratio)
    train_pairs = paired[:split_idx]
    val_pairs = paired[split_idx:]

    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")

    # Create output directories
    for subset in ("train", "val"):
        os.makedirs(os.path.join(output_path, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", subset), exist_ok=True)

    # Copy files
    for subset, pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, os.path.join(output_path, "images", subset, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(output_path, "labels", subset, os.path.basename(lbl_path)))

    # Read class names from Label Studio notes.json or existing yaml
    class_names = None

    # Check for Label Studio notes.json first
    notes_files = glob.glob(os.path.join(input_path, "**", "notes.json"), recursive=True)
    for notes_path in notes_files:
        with open(notes_path) as f:
            notes = json.load(f)
        if isinstance(notes, dict) and "categories" in notes:
            # Label Studio format: {"categories": [{"id": 0, "name": "cat"}, ...]}
            class_names = {cat["id"]: cat["name"] for cat in notes["categories"]}
            print(f"Using class names from: {notes_path}")
            print(f"Classes: {class_names}")
            break
        elif isinstance(notes, list):
            # Simple list format: ["cat", "dog", ...]
            class_names = {i: name for i, name in enumerate(notes)}
            print(f"Using class names from: {notes_path}")
            print(f"Classes: {class_names}")
            break

    # Fallback: check for yaml/yml files
    if class_names is None:
        yaml_files = glob.glob(os.path.join(input_path, "**", "*.yaml"), recursive=True) + \
                     glob.glob(os.path.join(input_path, "**", "*.yml"), recursive=True)
        for yaml_candidate in yaml_files:
            with open(yaml_candidate) as f:
                existing = yaml.safe_load(f)
            if existing and "names" in existing:
                class_names = existing["names"]
                if isinstance(class_names, list):
                    class_names = {i: name for i, name in enumerate(class_names)}
                print(f"Using class names from: {yaml_candidate}")
                print(f"Classes: {class_names}")
                break

    if class_names is None:
        print("WARNING: No notes.json or YAML with class names found in dataset folder.")
        max_class = -1
        for _, lbl_path in paired:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        max_class = max(max_class, int(parts[0]))
        nc = max_class + 1
        class_names = {i: f"class_{i}" for i in range(nc)}

    # Write dataset.yaml
    dataset_yaml = os.path.join(output_path, "dataset.yaml")
    dataset_config = {
        "path": output_path,
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    with open(dataset_yaml, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"Dataset prepared at: {output_path}")
    print(f"Dataset config: {dataset_yaml}")
    return dataset_yaml


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Training")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model variant (yolo26n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Dataset config (coco8.yaml, coco128.yaml, etc.)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', '0', '0,1', or None for auto")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory")
    parser.add_argument("--name", type=str, default="yolo26n_coco8", help="Experiment name")
    parser.add_argument("--prepare-dataset", type=str, default=None, metavar="INPUT_PATH",
                        help="Prepare dataset from YOLOv8 format: split into train/val and generate dataset.yaml")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--split-output", type=str, default=None, help="Output path for split dataset (default: INPUT_PATH/split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split")
    args = parser.parse_args()

    # Prepare dataset if requested
    if args.prepare_dataset:
        dataset_yaml = prepare_dataset(
            input_path=args.prepare_dataset,
            output_path=args.split_output,
            train_ratio=args.split_ratio,
            seed=args.seed,
        )
        # Use the generated yaml for training unless --data was explicitly set
        if args.data == "coco8.yaml":
            args.data = dataset_yaml

    # Load pretrained model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # Validate on the trained model
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Export to ONNX (optional, handy for deployment)
    print(f"\nTraining complete. Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
