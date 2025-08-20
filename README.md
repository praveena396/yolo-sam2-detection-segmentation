# yolo-sam2-detection-segmentation
 An automated pipeline for face and hand segmentation using YOLO for detection and Meta's SAM2 for high-precision mask generation. Includes Streamlit interface, bounding box-to-prompt conversion, and segmentation overlay.
This project implements an automated pipeline for **real-time face and hand detection and segmentation** using **YOLO** for object detection and **SAM2 (Segment Anything Model v2)** for high-precision segmentation masks. The system provides a **FastAPI backend** and **Streamlit interface** for interactive visualization and low-latency image processing.

---

## Features
- **Object Detection**: Detects faces and hands in images using YOLO.
- **Segmentation**: Generates precise segmentation masks for detected objects using SAM2.
- **Interactive Interface**: Real-time visualization using Streamlit.
- **Pipeline Automation**: Converts bounding boxes from YOLO into prompts for SAM2 automatically.
- **FastAPI Backend**: Serves low-latency API endpoints for inference and image processing.
- **Supports Multiple Image Formats**: Works with jpg, png, and other common image formats.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/praveena396/yolo-sam2-detection-segmentation.git
cd yolo-sam2-detection-segmentation
