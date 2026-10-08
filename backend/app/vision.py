"""Replaceable perception adapters. HOG is a limited CPU baseline, not YOLO."""

import os
import cv2
import numpy as np


class HogDetector:
    name = "OpenCV HOG + IoU baseline (people only)"

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.previous = {}
        self.next_id = 0

    def detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        current = {}
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            box = [int(x), int(y), int(x + w), int(y + h)]
            candidates = [
                (iou(box, old), key)
                for key, old in self.previous.items()
                if key not in current
            ]
            overlap, key = max(candidates, default=(0, 0))
            if overlap < 0.2:
                self.next_id += 1
                key = self.next_id
            current[key] = box
            # HOG SVM margins are uncalibrated; don't relabel as probabilities.
            detections.append(
                {
                    "track_id": key,
                    "box": box,
                    "object_type": "person",
                    "confidence": None,
                    "raw_margin": float(weight),
                }
            )
        self.previous = current
        return detections


class YoloDetector:
    name = "YOLO + ByteTrack"

    def __init__(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO unavailable. Install backend/requirements-models.txt or set PERCEPTION_BACKEND=hog for the limited CPU baseline."
            ) from exc
        self.model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))

    def detect(self, frame):
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=int(os.getenv("DETECTION_SIZE", "640")),
            conf=float(os.getenv("DETECTION_CONFIDENCE", "0.3")),
            verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        return [
            {
                "track_id": int(track),
                "box": [int(v) for v in box],
                "object_type": self.model.names[int(cls)],
                "confidence": float(conf),
            }
            for box, track, cls, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            )
        ]


def detector():
    choice = os.getenv("PERCEPTION_BACKEND", "hog")
    if choice not in ("hog", "yolo"):
        raise RuntimeError(f"Unknown detector: {choice}")
    return YoloDetector() if choice == "yolo" else HogDetector()


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(1, union)


def appearance(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    return (hist / max(float(np.linalg.norm(hist)), 1e-9)).tolist()


def dominant_color(image):
    if image.size < 30:
        return "unknown"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = np.median(hsv.reshape(-1, 3), axis=0)
    if v < 45:
        return "black"
    if s < 35:
        return "white" if v > 190 else "gray"
    if h < 12 or h > 170:
        return "red"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"


class ClipEncoder:
    """Lazy optional OpenCLIP adapter. Model failure is surfaced, never hidden."""

    def __init__(self):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def image(self, image):
        from PIL import Image

        with self.torch.no_grad():
            vector = self.model.encode_image(
                self.preprocess(
                    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ).unsqueeze(0)
            )
            return (vector / vector.norm(dim=-1, keepdim=True))[0].tolist()

    def text(self, text):
        with self.torch.no_grad():
            vector = self.model.encode_text(self.tokenizer([text]))
            return (vector / vector.norm(dim=-1, keepdim=True))[0].tolist()
