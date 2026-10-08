#!/usr/bin/env python3
"""Evaluation-only person-centered ROI object probe; no annotation-guided crops."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.temporal_events import PORTABLE  # noqa: E402
from meva.annotations import parse_geometry, parse_types  # noqa: E402
from meva.dataset import load_manifest, resolve, sha256  # noqa: E402
from meva.evaluation import box_iou  # noqa: E402
from meva.perception_experiment import requirements  # noqa: E402


def person_crop(box, width, height):
    """Expand a detected person box without consulting annotation geometry."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    radius_x = max(32, x2 - x1)
    radius_y = max(32, (y2 - y1) * 0.8)
    return (
        max(0, int(cx - radius_x)), max(0, int(cy - radius_y)),
        min(width, int(cx + radius_x)), min(height, int(cy + radius_y)),
    )


def predict_boxes(model, pixels, size, confidence):
    boxes = model.predict(pixels, imgsz=size, conf=confidence, verbose=False)[0].boxes
    if boxes is None:
        return []
    return [
        (model.names[int(cls)], float(score), box.tolist())
        for box, cls, score in zip(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(),
        )
    ]


def run(args):
    from ultralytics import YOLO

    manifest = load_manifest(args.manifest)
    diagnosis = json.loads((args.baseline_run / "failure-diagnostics.json").read_text())
    videos, indexed, actors = requirements(manifest, diagnosis, 2.0)
    model = YOLO(str(args.model))
    details = defaultdict(list)
    frames = person_crops = 0
    begun = time.perf_counter()
    for video_id, video in videos.items():
        target = {
            frame: [key for key in keys if actors[key]["annotation_actor_type"] == "other"]
            for frame, keys in indexed.get(video_id, {}).items()
        }
        target = {frame: keys for frame, keys in target.items() if keys}
        if not target:
            continue
        geometry = parse_geometry(
            resolve(video["geometry_path"]), video,
            parse_types(resolve(video["types_path"])),
        )
        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video['filename']}")
        try:
            for frame, keys in sorted(target.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode {video['filename']} frame {frame}")
                frames += 1
                full = predict_boxes(model, pixels, args.size, args.confidence)
                found = []
                for cls, _, box in full:
                    if cls != "person":
                        continue
                    left, top, right, bottom = person_crop(box, video["width"], video["height"])
                    if right - left < 32 or bottom - top < 32:
                        continue
                    person_crops += 1
                    crop = pixels[top:bottom, left:right]
                    for name, score, local in predict_boxes(model, crop, args.size, args.confidence):
                        if name in {"person", "car", "truck", "bus", "motorcycle"}:
                            continue
                        found.append((name, score, [local[0] + left, local[1] + top, local[2] + left, local[3] + top]))
                for key in keys:
                    reference = geometry.get(actors[key]["annotation_actor_id"], {}).get(frame)
                    if reference is None:
                        continue
                    matches = sorted(
                        ((box_iou(box, reference), name, score)
                         for name, score, box in found if box_iou(box, reference) >= 0.1),
                        reverse=True,
                    )
                    if matches:
                        overlap, name, score = matches[0]
                        details[key].append({
                            "frame": frame, "class": name,
                            "confidence": round(score, 4), "iou": round(overlap, 4),
                            "event_supported": name in PORTABLE,
                        })
        finally:
            cap.release()
    rows = [
        {**actor, "hits": details[key]}
        for key, actor in actors.items() if actor["annotation_actor_type"] == "other"
    ]
    report = {
        "method": "YOLO full frame person detections generate all expanded person crops; YOLO reruns on each crop. MEVA boxes are used only after all crops and predictions, to score overlaps at IoU >= 0.1.",
        "model": args.model.name, "model_sha256": sha256(args.model),
        "size": args.size, "confidence": args.confidence,
        "unique_source_frames": frames, "person_crops": person_crops,
        "wall_seconds": round(time.perf_counter() - begun, 3),
        "broad_object_covered": sum(bool(row["hits"]) for row in rows),
        "event_supported_object_covered": sum(
            any(hit["event_supported"] for hit in row["hits"]) for row in rows
        ),
        "annotations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "unique_source_frames", "person_crops", "wall_seconds",
        "broad_object_covered", "event_supported_object_covered",
    )}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "data/meva/improved-2fps")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--output", type=Path, default=ROOT / "data/meva/perception-experiment/person-roi.json")
    run(parser.parse_args())
