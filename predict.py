"""
YOLOv8 Road Safety - Quick Prediction & Evaluation Script
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Quick Prediction")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="Source image or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", default=True, help="Save prediction results")
    args = parser.parse_args()

    print(f"Loading YOLOv8 model from {args.weights}...")
    model = YOLO(args.weights)

    print(f"Running inference on {args.source}...")
    results = model.predict(source=args.source, conf=args.conf, save=args.save)

    for i, r in enumerate(results):
        print(f"Result {i+1}: {len(r.boxes)} detections found.")
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f" - {cls_name}: {conf:.2f}")

if __name__ == "__main__":
    main()
