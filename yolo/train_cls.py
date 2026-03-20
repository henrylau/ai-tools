"""
YOLO26 Classification Training Script

Prepares a classification dataset from Label Studio JSON-MIN export,
organizes images into class folders, and trains a YOLO classification model.

Label Studio setup:
    Use a <Choices> tag for single-label image classification.

Export:
    Project → Export → JSON-MIN

    JSON-MIN format (single choice):
    [
        {"id": 1, "image": "/data/upload/img.jpg", "choice_name": "cat"},
        {"id": 2, "image": "/data/upload/img2.jpg", "choice_name": "dog"},
        ...
    ]

    The field name for the label matches the "from_name" in your Label Studio config.

Usage:
    # Prepare dataset from Label Studio export and train
    python train_cls.py --export export.json --images /path/to/images

    # Specify which JSON field contains the label
    python train_cls.py --export export.json --images /path/to/images --label-field choice

    # Print dataset info and exit
    python train_cls.py --info /path/to/dataset

    # Skip preparation, train on existing dataset
    python train_cls.py --data /path/to/dataset --epochs 50
"""

import argparse
import json
import os
import random
import shutil

from ultralytics import YOLO


def find_label_field(tasks):
    """Auto-detect the label field name from JSON-MIN tasks.

    Skips known non-label fields and picks the first remaining field.
    """
    skip = {"id", "image", "annotator", "annotation_id", "created_at", "updated_at", "lead_time"}
    for task in tasks:
        for key in task:
            if key not in skip:
                return key
    return None


def prepare_cls_dataset(export_path, images_dir, output_path, label_field=None,
                        train_ratio=0.8, seed=42):
    """Convert Label Studio JSON-MIN export to YOLO classification folder structure.

    Produces:
        output_path/
        ├── train/
        │   ├── class_a/
        │   └── class_b/
        └── val/
            ├── class_a/
            └── class_b/
    """
    with open(export_path) as f:
        tasks = json.load(f)

    if not tasks:
        raise ValueError(f"No tasks found in {export_path}")

    # Auto-detect label field if not specified
    if label_field is None:
        label_field = find_label_field(tasks)
        if label_field is None:
            raise ValueError("Could not auto-detect label field. Use --label-field to specify it.")
        print(f"Auto-detected label field: '{label_field}'")

    # Collect (image_path, label) pairs
    pairs = []
    skipped_no_label = 0
    skipped_no_image = 0
    for task in tasks:
        label = task.get(label_field)
        if not label:
            skipped_no_label += 1
            continue

        # Resolve image path
        image_ref = task.get("image", "")
        # Label Studio stores paths like /data/upload/1/filename.jpg — extract filename
        filename = os.path.basename(image_ref)
        img_path = os.path.join(images_dir, filename)

        if not os.path.isfile(img_path):
            skipped_no_image += 1
            continue

        pairs.append((img_path, label))

    if not pairs:
        raise ValueError("No valid image/label pairs found.")

    print(f"Found {len(pairs)} labeled images")
    if skipped_no_label:
        print(f"Skipped {skipped_no_label} tasks with no label")
    if skipped_no_image:
        print(f"Skipped {skipped_no_image} tasks with missing image file")

    # Collect class names
    class_names = sorted(set(label for _, label in pairs))
    print(f"Classes ({len(class_names)}): {class_names}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")

    # Create output directories and copy images
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, label in split_pairs:
            dest_dir = os.path.join(output_path, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

    print(f"Dataset prepared at: {output_path}")
    dataset_info(output_path)
    return output_path


def dataset_info(dataset_path):
    """Print dataset path, split sizes, and per-class image counts."""
    dataset_path = os.path.abspath(dataset_path)

    print(f"\n=== Dataset Info ===")
    print(f"  Path: {dataset_path}")
    print()

    for split in ("train", "val"):
        split_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(split_dir):
            print(f"  {split}: not found")
            continue

        class_dirs = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        total = 0
        class_counts = {}
        for cls_name in class_dirs:
            cls_dir = os.path.join(split_dir, cls_name)
            count = len([f for f in os.listdir(cls_dir) if not f.startswith(".")])
            class_counts[cls_name] = count
            total += count

        print(f"  {split}:")
        print(f"    Images: {total}")
        print(f"    Classes: {len(class_dirs)}")
        for cls_name, count in class_counts.items():
            print(f"      {cls_name}: {count}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26 classification model from Label Studio export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset preparation
    parser.add_argument("--export", type=str, default=None,
                        help="Path to Label Studio JSON-MIN export file")
    parser.add_argument("--images", type=str, default=None,
                        help="Directory containing the source images")
    parser.add_argument("--label-field", type=str, default=None,
                        help="JSON field name for the label (auto-detected if omitted)")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset split")

    # Training
    parser.add_argument("--data", type=str, default=None,
                        help="Path to prepared dataset (skip preparation)")
    parser.add_argument("--model", type=str, default="yolo26n-cls.pt",
                        help="Model variant (default: yolo26n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', '0', 'mps', or None for auto")
    parser.add_argument("--project", type=str, default="runs/classify",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (default: derived from model name)")

    # Info
    parser.add_argument("--info", type=str, default=None, metavar="DATASET_PATH",
                        help="Print dataset info and exit")

    args = parser.parse_args()

    if args.info:
        dataset_info(args.info)
        return

    # Determine dataset path
    dataset_path = args.data
    if args.export:
        if not args.images:
            parser.error("--images is required when using --export")
        output_path = args.data or "datasets/classification"
        dataset_path = prepare_cls_dataset(
            export_path=args.export,
            images_dir=args.images,
            output_path=output_path,
            label_field=args.label_field,
            train_ratio=args.split_ratio,
            seed=args.seed,
        )

    if dataset_path is None:
        parser.error("Provide --export (to prepare dataset) or --data (to use existing dataset)")

    # Train
    if args.name is None:
        from pathlib import Path
        args.name = f"{Path(args.model).stem}_cls"

    print(f"\n=== Training {args.model} ===")
    print(f"  Dataset: {dataset_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device or 'auto'}")
    print()

    model = YOLO(args.model)
    model.train(
        data=dataset_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    metrics = model.val()
    print(f"\nTop-1 accuracy: {metrics.top1:.4f}")
    print(f"Top-5 accuracy: {metrics.top5:.4f}")

    from pathlib import Path
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
