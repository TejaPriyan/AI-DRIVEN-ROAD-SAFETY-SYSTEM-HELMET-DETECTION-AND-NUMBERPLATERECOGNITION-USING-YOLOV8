"""
YOLOv8 Road Safety - Custom Dataset Training Script
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Custom Road Safety Dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model base (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset configuration YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="CUDA device ID ('0') or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train", help="Save directory")
    parser.add_argument("--name", type=str, default="road_safety_model", help="Experiment name")
    args = parser.parse_args()

    print(f"[INFO] Initializing YOLOv8 model from {args.model}...")
    model = YOLO(args.model)

    print(f"[INFO] Starting training on {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name
    )

    print("[INFO] Training completed successfully!")
    print(f"[INFO] Best weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
