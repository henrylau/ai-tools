"""
Download a labeled dataset from Roboflow Universe for YOLO26 training.

Setup:
    1. Create a free Roboflow account at https://roboflow.com
    2. Go to Settings > API Keys and copy your Private API Key
    3. Set it as an environment variable:
         export ROBOFLOW_API_KEY="your_api_key_here"
       Or create a .env file in this directory:
         ROBOFLOW_API_KEY=your_api_key_here

Usage:
    # Download the default sample dataset (Hard Hat Workers)
    python download_dataset.py

    # Download a specific Roboflow Universe dataset
    python download_dataset.py --workspace <workspace> --project <project> --version <version>

Popular public datasets on Roboflow Universe (https://universe.roboflow.com):
    - Hard Hat Workers:  --workspace roboflow-universe-projects --project hard-hat-universe-0dy7t --version 1
    - Road Signs:        --workspace roboflow-universe-projects --project road-sign-detection-wwhat --version 2
    - Aquarium:          --workspace brad-dwyer --project aquarium-combined --version 6
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download dataset from Roboflow")
    parser.add_argument("--workspace", type=str, default="brad-dwyer",
                        help="Roboflow workspace name")
    parser.add_argument("--project", type=str, default="aquarium-combined",
                        help="Roboflow project name")
    parser.add_argument("--version", type=int, default=6,
                        help="Dataset version number")
    parser.add_argument("--format", type=str, default="yolov8",
                        help="Export format (yolov8 = YOLO format)")
    parser.add_argument("--location", type=str, default=None,
                        help="Download location (default: datasets/<project>)")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not set.")
        print()
        print("1. Sign up at https://roboflow.com (free)")
        print("2. Go to Settings > Roboflow API > Private API Key")
        print("3. Run: export ROBOFLOW_API_KEY='your_key_here'")
        sys.exit(1)

    location = args.location or os.path.join("datasets", args.project)

    print(f"Downloading: {args.workspace}/{args.project} v{args.version}")
    print(f"Format: {args.format}")
    print(f"Location: {location}")
    print()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download(args.format, location=location)

    print(f"\nDataset downloaded to: {dataset.location}")
    print(f"\nTo train, run:")
    print(f"  .venv/bin/python train.py --data {dataset.location}/data.yaml --name custom_train --epochs 50")


if __name__ == "__main__":
    main()
