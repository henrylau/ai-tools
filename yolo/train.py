"""
YOLO26 Training Script

Usage:
    python train.py                          # Train with defaults (COCO8, yolo26n)
    python train.py --model yolo26s.pt       # Use a different model size
    python train.py --data coco128.yaml      # Use a larger sample dataset
    python train.py --epochs 50 --batch 8    # Custom training parameters
"""

import argparse
from ultralytics import YOLO


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
    args = parser.parse_args()

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
