"""Детекция объектов (YOLO): bbox и нижняя центральная точка (cx, cy)."""
import cv2
import torch
from ultralytics import YOLO

from app.core.config import MODEL_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)


def run_inference(frame: cv2.Mat) -> list:
    """Детекции: список кортежей (x1, y1, x2, y2, cx, cy). (cx, cy) — низ центра bbox."""
    results = model(frame, device=device)
    boxes = results[0].boxes
    detections = []

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in xyxy:
            cx = (x1 + x2) / 2.0
            cy = float(y2)
            detections.append(
                (float(x1), float(y1), float(x2), float(y2), float(cx), float(cy))
            )

    return detections

