"""
YOLOv8 Road Safety - Quick Prediction & Evaluation Script
"""

import argparse
from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 Quick Prediction")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="Source image or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save prediction results (use --no-save to disable)",
    )
    return parser.parse_args(argv)


def _get_class_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def main(argv=None):
    args = parse_args(argv)

    print(f"Loading YOLOv8 model from {args.weights}...")
    model = YOLO(args.weights)

    print(f"Running inference on {args.source}...")
    results = model.predict(source=args.source, conf=args.conf, save=args.save)

    if not results:
        print("No prediction results returned.")
        return results

    model_names = getattr(model, "names", {})
    for i, r in enumerate(results):
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            print(f"Result {i + 1}: 0 detections found.")
            continue
        print(f"Result {i + 1}: {len(boxes)} detections found.")
        for box in boxes:
            cls_name = _get_class_name(model_names, int(box.cls[0]))
            conf = float(box.conf[0])
            print(f" - {cls_name}: {conf:.2f}")

    return results


if __name__ == "__main__":
    main()
