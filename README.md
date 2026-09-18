# 🛡️ AI-Driven Road Safety System
### Real-Time Helmet Detection & Automated Number Plate Recognition (ANPR) using YOLOv8

> An end-to-end intelligent computer vision system engineered by **[Teja Priyan](https://github.com/TejaPriyan)** to detect motorcycle helmet violations and automatically recognize license plates in real-time traffic surveillance streams.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

Traffic violations involving non-compliance with helmet laws are a leading contributor to severe road accidents and fatalities. This system addresses automated enforcement by combining **YOLOv8 Deep Learning Object Detection** with **Automated Number Plate Recognition (ANPR)**.

When a motorcycle rider without a helmet is detected in surveillance or live CCTV video, the system immediately triggers a violation event, crops the vehicle's license plate region, and performs Optical Character Recognition (OCR) to extract the vehicle registration number for ticketing and auditing.

---

## 🚀 System Architecture & Pipeline

```
  +-----------------------+
  | Live Video / CCTV /   |
  | Traffic Camera Stream |
  +-----------+-----------+
              |
              v
  +-----------------------+
  |     YOLOv8 Model      |
  | Multiclass Detection  |
  +-----------+-----------+
              |
      +-------+-------+
      |               |
      v               v
 [Rider / Helmet]  [Number Plate]
      |               |
      v               v
  Violation?       Bounding Box
  (No Helmet)        Crop
      |               |
      +-------+-------+
              |
              v
  +-----------------------+
  |  OCR Text Extraction  |
  |   (ANPR with EasyOCR) |
  +-----------+-----------+
              |
              v
  +-----------------------+
  | Violation Log Saved   |
  | Image Snapshot + Plate|
  +-----------------------+
```

---

## ✨ Key Features

- ⚡ **High FPS Real-Time Inference**: Powered by Ultralytics YOLOv8 architecture optimized for both GPU acceleration and edge computing.
- 🎯 **Multi-Class Detection**: Identifies riders, helmet compliance, no-helmet violations, and license plates concurrently.
- 🔍 **Integrated ANPR / OCR**: Extracts alphanumeric characters from detected license plates automatically.
- 📸 **Automated Violation Cropping**: Archives high-resolution snapshots of rule violations for administrative review.
- 🛠️ **Multi-Source Support**: Accepts live webcams, RTSP IP camera streams, local video files (`.mp4`, `.avi`), and static image batches.

---

## 🛠️ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/TejaPriyan/AI-DRIVEN-ROAD-SAFETY-SYSTEM-HELMET-DETECTION-AND-NUMBERPLATERECOGNITION-USING-YOLOV8.git
cd AI-DRIVEN-ROAD-SAFETY-SYSTEM-HELMET-DETECTION-AND-NUMBERPLATERECOGNITION-USING-YOLOV8
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Inference on Video or Camera Feed
```bash
# Run detection on a video file with annotated visual output
python detect.py --source demo_traffic.mp4 --weights yolov8n.pt --save

# Run detection on live webcam (Device 0) with Plate OCR enabled
python detect.py --source 0 --conf 0.35 --ocr

# Run inference on a directory of traffic images
python detect.py --source path/to/images/ --save --save-crop
```

---

## 🏋️ Training on a Custom Dataset

To train the YOLOv8 road safety model on your custom dataset:

1. Prepare your `data.yaml` file specifying `train` and `val` image paths and class labels:
```yaml
names:
  0: Helmet
  1: No-Helmet
  2: Rider
  3: Number-Plate
nc: 4
train: data/images/train
val: data/images/val
```

2. Execute the training script:
```bash
python train.py --model yolov8n.pt --data data.yaml --epochs 50 --batch 16 --imgsz 640
```

3. Validate model performance metrics:
```bash
python val.py --weights runs/train/road_safety_model/weights/best.pt --data data.yaml
```

---

## 📊 Performance & Evaluation

| Metric | Target / Benchmark |
| :--- | :--- |
| **Model Family** | YOLOv8 (Nano / Small / Medium) |
| **Inference Speed** | ~35-60 FPS (NVIDIA GPU) / ~18-25 FPS (CPU) |
| **Detection Precision** | > 92% mAP@0.50 on standard traffic benchmarks |
| **Plate Recognition** | Clean alphanumeric parsing with EasyOCR / OpenCV preprocessing |

---

## 👨‍💻 Author & Contact

**Teja Priyan (Tejapriyan)**
* 🌐 **Portfolio:** [portfoliotejapriyan.vercel.app](https://portfoliotejapriyan.vercel.app/)
* 🐙 **GitHub:** [@TejaPriyan](https://github.com/TejaPriyan)
* 🤗 **Hugging Face:** [@teja161615](https://huggingface.co/teja161615)
* ✉️ **Email:** [teja1616150@gmail.com](mailto:teja1616150@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © 2026 **Teja Priyan**.
