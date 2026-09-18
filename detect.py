"""
AI-Driven Road Safety System: Real-Time Helmet Detection & Automated Number Plate Recognition (ANPR)
Powered by Ultralytics YOLOv8.

Usage Examples:
    # Run detection on a video file
    $ python detect.py --source traffic_video.mp4 --weights yolov8n.pt --save

    # Run detection on webcam feed (ID: 0)
    $ python detect.py --source 0 --conf 0.35

    # Run detection with Number Plate OCR extraction
    $ python detect.py --source test_image.jpg --ocr --save
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser(description="YOLOv8 Road Safety: Helmet & License Plate Detection")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to trained YOLOv8 model weights (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Source: webcam ID ('0'), image path, video path, or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (pixels)")
    parser.add_argument("--device", default="", help="Device: '0', '0,1', 'cpu', etc.")
    parser.add_argument("--save", action="store_true", help="Save detection visual results to disk")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped license plate and rider images")
    parser.add_argument("--ocr", action="store_true", help="Attempt OCR extraction on detected license plates")
    parser.add_argument("--output", type=str, default="runs/detect", help="Output directory")
    return parser.parse_args()

def run_pipeline(opt):
    source = opt.source
    if source.isdigit():
        source = int(source)

    print(f"[INFO] Initializing YOLOv8 Road Safety Pipeline...")
    print(f"[INFO] Model Weights: {opt.weights}")
    print(f"[INFO] Input Source: {opt.source}")

    # Load YOLOv8 model
    model = YOLO(opt.weights)

    # Optional OCR loader (EasyOCR if installed)
    reader = None
    if opt.ocr:
        try:
            import easyocr
            print("[INFO] Loading EasyOCR English reader...")
            reader = easyocr.Reader(['en'], gpu=bool(opt.device and opt.device != 'cpu'))
        except ImportError:
            print("[WARNING] 'easyocr' not found. Run 'pip install easyocr' for plate text reading.")

    os.makedirs(opt.output, exist_ok=True)
    if opt.save_crop:
        os.makedirs(os.path.join(opt.output, "crops", "plates"), exist_ok=True)
        os.makedirs(os.path.join(opt.output, "crops", "violations"), exist_ok=True)

    # Run inference stream
    results = model.predict(
        source=source,
        conf=opt.conf,
        iou=opt.iou,
        imgsz=opt.imgsz,
        device=opt.device if opt.device else None,
        stream=True,
        save=opt.save,
        project=opt.output,
        name="road_safety",
        exist_ok=True
    )

    frame_idx = 0
    total_violations = 0

    for result in results:
        frame_idx += 1
        orig_img = result.orig_img
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Heuristic check for road safety classes
            is_no_helmet = "no-helmet" in cls_name.lower() or "without helmet" in cls_name.lower()
            is_plate = "plate" in cls_name.lower() or "license" in cls_name.lower()

            if is_no_helmet:
                total_violations += 1
                print(f"[VIOLATION] Frame {frame_idx}: Detected '{cls_name}' (conf: {conf:.2f}) at {xyxy}")
                if opt.save_crop:
                    x1, y1, x2, y2 = xyxy
                    crop = orig_img[max(0, y1):min(orig_img.shape[0], y2), max(0, x1):min(orig_img.shape[1], x2)]
                    crop_name = os.path.join(opt.output, "crops", "violations", f"violation_{frame_idx}_{i}.jpg")
                    cv2.imwrite(crop_name, crop)

            if is_plate and reader is not None:
                x1, y1, x2, y2 = xyxy
                plate_crop = orig_img[max(0, y1):min(orig_img.shape[0], y2), max(0, x1):min(orig_img.shape[1], x2)]
                if plate_crop.size > 0:
                    ocr_results = reader.readtext(plate_crop)
                    plate_text = " ".join([text for (_, text, score) in ocr_results if score > 0.3]).strip()
                    if plate_text:
                        print(f"[ANPR] Frame {frame_idx}: Detected Number Plate: '{plate_text}'")

    print(f"[INFO] Inference complete. Total violations detected: {total_violations}")
    if opt.save:
        print(f"[INFO] Annotated outputs saved to: {os.path.join(opt.output, 'road_safety')}")

if __name__ == "__main__":
    opt = parse_opt()
    run_pipeline(opt)
