"""
YOLOv8 Road Safety - Validation & Model Evaluation Script
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 Road Safety Model")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Trained model weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--device", default="", help="Device: '0' or 'cpu'")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device if args.device else None)

    print(f"Validation mAP@0.50: {metrics.box.map50:.4f}")
    print(f"Validation mAP@0.50:0.95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
