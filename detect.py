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

import cv2
from ultralytics import YOLO


def parse_opt(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 Road Safety: Helmet & License Plate Detection")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to trained YOLOv8 model weights (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Source: webcam ID ('0'), image path, video path, or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (pixels)")
    parser.add_argument("--device", default="", help="Device: '0', '0,1', 'cpu', etc.")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save detection visual results to disk",
    )
    parser.add_argument("--save-crop", action="store_true", help="Save cropped license plate and violation images")
    parser.add_argument("--ocr", action="store_true", help="Attempt OCR extraction on detected license plates")
    parser.add_argument("--output", type=str, default="runs/detect", help="Output directory")
    return parser.parse_args(argv)


def _should_use_easyocr_gpu(device: str) -> bool:
    """Determine whether EasyOCR should use GPU based on --device or CUDA availability."""
    normalized = str(device or "").strip().lower()
    if normalized:
        return normalized != "cpu"
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        return bool(is_available()) if callable(is_available) else False
    except Exception:
        return False


def _scalar_value(val):
    return val.item() if hasattr(val, "item") else val


def _to_xyxy_ints(raw_xyxy) -> tuple[int, int, int, int]:
    coords = raw_xyxy
    if hasattr(coords, "cpu"):
        coords = coords.cpu()
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    vals = [int(v) for v in coords]
    return vals[0], vals[1], vals[2], vals[3]


def _get_class_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _is_no_helmet_class(cls_name: str) -> bool:
    norm = cls_name.lower().replace("_", "-").replace(" ", "-")
    return "no-helmet" in norm or "without-helmet" in norm or "nohelmet" in norm


def _is_rider_class(cls_name: str) -> bool:
    norm = cls_name.lower().replace("_", "-").replace(" ", "-")
    return "rider" in norm or "motorcycle" in norm or "motorbike" in norm


def _is_plate_class(cls_name: str) -> bool:
    norm = cls_name.lower().replace("_", "-").replace(" ", "-")
    return "plate" in norm or "license" in norm


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    inter = _intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _union_box(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    )


def _center_distance_sq(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    return (ax - bx) ** 2 + (ay - by) ** 2


def crop_box(orig_img, xyxy: tuple[int, int, int, int]):
    """Safely slice a bounding box crop from an image, returning an empty slice if degenerate."""
    h, w = orig_img.shape[:2]
    x1, y1, x2, y2 = xyxy
    cx1 = max(0, min(x1, w))
    cy1 = max(0, min(y1, h))
    cx2 = max(0, min(x2, w))
    cy2 = max(0, min(y2, h))
    if cx2 <= cx1 or cy2 <= cy1:
        return orig_img[0:0, 0:0]
    return orig_img[cy1:cy2, cx1:cx2]


def associate_plates_to_violations(
    no_helmet_boxes: list[dict],
    rider_boxes: list[dict],
    plate_boxes: list[dict],
) -> list[dict]:
    """Spatially associate detected number plates with No-Helmet / Rider violation boxes."""
    if not no_helmet_boxes:
        return []

    violation_entries = []
    violating_rider_indices = set()

    for nh in no_helmet_boxes:
        best_rider = None
        best_rider_score = None
        nh_area = max(1, _box_area(nh["xyxy"]))

        for rider in rider_boxes:
            inter = _intersection_area(nh["xyxy"], rider["xyxy"])
            if inter <= 0:
                continue
            score = (
                inter / nh_area,
                _box_iou(nh["xyxy"], rider["xyxy"]),
                -_center_distance_sq(nh["xyxy"], rider["xyxy"]),
            )
            if best_rider_score is None or score > best_rider_score:
                best_rider_score = score
                best_rider = rider

        if best_rider is not None:
            violating_rider_indices.add(best_rider["index"])
            region = _union_box(nh["xyxy"], best_rider["xyxy"])
        else:
            region = nh["xyxy"]

        violation_entries.append(
            {
                "violation": nh,
                "rider": best_rider,
                "region": region,
                "plates": [],
            }
        )

    non_violating_riders = [r for r in rider_boxes if r["index"] not in violating_rider_indices]

    for plate in plate_boxes:
        plate_area = max(1, _box_area(plate["xyxy"]))
        best_entry = None
        best_entry_score = None

        for entry in violation_entries:
            inter = _intersection_area(plate["xyxy"], entry["region"])
            if inter <= 0:
                continue
            score = (
                inter / plate_area,
                _box_iou(plate["xyxy"], entry["region"]),
                -_center_distance_sq(plate["xyxy"], entry["region"]),
            )
            if best_entry_score is None or score > best_entry_score:
                best_entry_score = score
                best_entry = entry

        if best_entry is None:
            continue

        best_non_violating_overlap = max(
            (
                _intersection_area(plate["xyxy"], r["xyxy"]) / plate_area
                for r in non_violating_riders
            ),
            default=0.0,
        )
        if best_non_violating_overlap > best_entry_score[0]:
            continue

        best_entry["plates"].append(plate)

    return violation_entries


def run_pipeline(opt):
    source = opt.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print("[INFO] Initializing YOLOv8 Road Safety Pipeline...")
    print(f"[INFO] Model Weights: {opt.weights}")
    print(f"[INFO] Input Source: {opt.source}")

    # Load YOLOv8 model
    model = YOLO(opt.weights)

    model_names = getattr(model, "names", None)
    if isinstance(model_names, (dict, list, tuple)) and len(model_names) > 0:
        names_list = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
        if not any(_is_no_helmet_class(str(n)) or _is_plate_class(str(n)) for n in names_list):
            print(
                "[WARNING] Loaded model classes do not include helmet violation or number plate labels. "
                "Provide custom-trained weights via --weights for helmet/ANPR detection."
            )

    # Optional OCR loader (EasyOCR if installed)
    reader = None
    if opt.ocr:
        try:
            import easyocr

            print("[INFO] Loading EasyOCR English reader...")
            reader = easyocr.Reader(["en"], gpu=_should_use_easyocr_gpu(opt.device))
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
        exist_ok=True,
    )

    frame_idx = 0
    total_violations = 0
    plates_recognized = []

    for result in results:
        frame_idx += 1
        orig_img = getattr(result, "orig_img", None)
        boxes = getattr(result, "boxes", None)

        if orig_img is None or getattr(orig_img, "size", 0) == 0 or boxes is None or len(boxes) == 0:
            continue

        no_helmet_boxes = []
        rider_boxes = []
        plate_boxes = []

        for i, box in enumerate(boxes):
            cls_id = int(_scalar_value(box.cls[0]))
            cls_name = _get_class_name(model_names, cls_id)
            conf = float(_scalar_value(box.conf[0]))
            xyxy = _to_xyxy_ints(box.xyxy[0])

            det = {
                "index": i,
                "cls_name": cls_name,
                "conf": conf,
                "xyxy": xyxy,
            }
            if _is_no_helmet_class(cls_name):
                no_helmet_boxes.append(det)
            elif _is_rider_class(cls_name):
                rider_boxes.append(det)
            elif _is_plate_class(cls_name):
                plate_boxes.append(det)

        violation_entries = associate_plates_to_violations(no_helmet_boxes, rider_boxes, plate_boxes)

        for entry in violation_entries:
            nh = entry["violation"]
            total_violations += 1
            print(
                f"[VIOLATION] Frame {frame_idx}: Detected '{nh['cls_name']}' "
                f"(conf: {nh['conf']:.2f}) at {list(nh['xyxy'])}"
            )

            if opt.save_crop:
                crop = crop_box(orig_img, nh["xyxy"])
                if crop.size > 0:
                    crop_name = os.path.join(
                        opt.output, "crops", "violations", f"violation_{frame_idx}_{nh['index']}.jpg"
                    )
                    cv2.imwrite(crop_name, crop)

            for plate in entry["plates"]:
                plate_crop = crop_box(orig_img, plate["xyxy"])
                if plate_crop.size <= 0:
                    continue

                if opt.save_crop:
                    plate_crop_name = os.path.join(
                        opt.output, "crops", "plates", f"plate_{frame_idx}_{plate['index']}.jpg"
                    )
                    cv2.imwrite(plate_crop_name, plate_crop)

                if reader is not None:
                    ocr_results = reader.readtext(plate_crop)
                    plate_text = " ".join([text for (_, text, score) in ocr_results if score > 0.3]).strip()
                    if plate_text:
                        plates_recognized.append(plate_text)
                        print(f"[ANPR] Frame {frame_idx}: Detected Number Plate: '{plate_text}'")

    print(f"[INFO] Inference complete. Total violations detected: {total_violations}")
    if opt.save:
        print(f"[INFO] Annotated outputs saved to: {os.path.join(opt.output, 'road_safety')}")

    return {
        "frames": frame_idx,
        "violations": total_violations,
        "plates": plates_recognized,
    }


if __name__ == "__main__":
    opt = parse_opt()
    run_pipeline(opt)
