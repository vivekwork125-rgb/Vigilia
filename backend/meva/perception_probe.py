"""Independent raw-detector probe; ground truth never enters runtime inference."""

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2

from .annotations import parse_geometry, parse_types
from .dataset import resolve, sha256
from .diagnostics import actor_compatible
from .evaluation import box_iou


def sampled_annotation_frames(start, end, source_fps, sample_fps):
    """Frames that production's requested time grid visits inside a label."""
    last = round(end * sample_fps / source_fps) + 2
    return sorted({
        round(n * source_fps / sample_fps)
        for n in range(last + 1)
        if start <= round(n * source_fps / sample_fps) <= end
    })


def probe(manifest, directory):
    """Compare raw YOLO boxes with stored tracked observations at exact frames."""
    from ultralytics import YOLO

    directory = Path(directory)
    config = json.loads((directory / "run-config.json").read_text())
    diagnosis = json.loads((directory / "failure-diagnostics.json").read_text())
    from .dataset import ROOT

    model_path = Path(os.environ.get("YOLO_MODEL", str(ROOT / "models" / config["model"])))
    if not model_path.is_file() or sha256(model_path) != config["model_sha256"]:
        raise ValueError("Raw detector probe requires the recorded local YOLO weights")
    model = YOLO(str(model_path))
    by_video = defaultdict(list)
    for row in diagnosis["annotations"]:
        by_video[row["video_id"]].append(row)
    records = []
    decoded = 0
    for video in manifest["videos"]:
        rows = by_video[video["video_id"]]
        if not rows:
            continue
        types = parse_types(resolve(video["types_path"]))
        geometry = parse_geometry(resolve(video["geometry_path"]), video, types)
        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        if not cap.isOpened():
            raise ValueError(f"Cannot open source for raw detector probe: {video['filename']}")
        frame_requirements = defaultdict(list)
        for row in rows:
            for actor in row["actors"]:
                for frame in sampled_annotation_frames(
                    row["start_frame"], row["end_frame_inclusive"],
                    video["fps"], config["sample_fps"],
                ):
                    frame_requirements[frame].append((row["key"], actor["annotation_actor_id"]))
        actor_hits = defaultdict(list)
        try:
            for frame in sorted(frame_requirements):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError(f"Raw detector probe cannot decode {video['filename']} frame {frame}")
                decoded += 1
                result = model.predict(
                    pixels, imgsz=config["detection_size"],
                    conf=config["confidence_threshold"], verbose=False,
                )[0]
                boxes = result.boxes
                detections = [
                    (model.names[int(cls)], float(conf), box.tolist())
                    for box, cls, conf in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.cls.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                    )
                ] if boxes is not None else []
                for key, actor in frame_requirements[frame]:
                    reference = geometry.get(actor, {}).get(frame)
                    if reference is None:
                        continue
                    annotation_type = types[actor]
                    compatible = [
                        (box_iou(box, reference), name, confidence)
                        for name, confidence, box in detections
                        if actor_compatible(annotation_type, name)
                    ]
                    best = max(compatible, default=(0.0, None, None))
                    if best[0] >= 0.1:
                        actor_hits[(key, actor)].append({
                            "frame": frame, "iou": round(best[0], 4),
                            "detector_class": best[1], "confidence": round(best[2], 4),
                        })
        finally:
            cap.release()
        for row in rows:
            actors = []
            for actor in row["actors"]:
                raw = actor_hits[(row["key"], actor["annotation_actor_id"])]
                actors.append({
                    "annotation_actor_id": actor["annotation_actor_id"],
                    "annotation_actor_type": actor["annotation_actor_type"],
                    "raw_detected": bool(raw),
                    "tracked": actor["detected_during_activity"],
                    "raw_hits": raw,
                    "matching_track_ids": actor["track_ids"],
                })
            records.append({
                "key": row["key"],
                "camera_id": row["camera_id"],
                "activity_type": row["activity_type"],
                "event_type": row["event_type"],
                "all_raw_detected": all(a["raw_detected"] for a in actors),
                "all_tracked": all(a["tracked"] for a in actors),
                "actors": actors,
            })
    summary = []
    for activity in sorted({r["activity_type"] for r in records}):
        subset = [r for r in records if r["activity_type"] == activity]
        summary.append({
            "activity_type": activity,
            "gt": len(subset),
            "all_required_raw_detected": sum(r["all_raw_detected"] for r in subset),
            "all_required_tracked": sum(r["all_tracked"] for r in subset),
            "raw_detected_but_not_all_tracked": sum(r["all_raw_detected"] and not r["all_tracked"] for r in subset),
            "actor_roles": {
                role: {
                    "raw_detected": sum(a["raw_detected"] for r in subset for a in r["actors"] if a["annotation_actor_type"] == role),
                    "tracked": sum(a["tracked"] for r in subset for a in r["actors"] if a["annotation_actor_type"] == role),
                }
                for role in sorted({a["annotation_actor_type"] for r in subset for a in r["actors"]})
            },
        })
    report = {
        "method": "Independent YOLO.predict on exact production sample frames at recorded model, size and confidence; annotation boxes compared only after prediction. Track support reads completed production observations. A one-frame hit is coverage, not a verified action or identity.",
        "decoded_unique_frames": decoded,
        "sample_fps": config["sample_fps"],
        "model_sha256": config["model_sha256"],
        "detection_size": config["detection_size"],
        "confidence_threshold": config["confidence_threshold"],
        "summary": summary,
        "annotations": records,
    }
    (directory / "detector-coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
