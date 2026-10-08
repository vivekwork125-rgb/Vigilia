"""Evaluation-only, fixed-frame detector experiment against MEVA geometry.

Ground truth is consulted only after YOLO has predicted on source pixels. This
module is deliberately outside ``app`` and is never imported by processing.
"""

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
from app.temporal_events import PORTABLE as EVENT_PORTABLE

from .annotations import parse_geometry, parse_types
from .dataset import resolve, sha256
from .diagnostics import actor_compatible
from .evaluation import box_iou
from .perception_probe import sampled_annotation_frames

def requirements(manifest, diagnosis, sample_fps):
    """Use every DIRECT case and the exact production sampling grid."""
    videos = {video["video_id"]: video for video in manifest["videos"]}
    indexed = defaultdict(lambda: defaultdict(list))
    actors = {}
    for row in diagnosis["annotations"]:
        video = videos[row["video_id"]]
        for actor in row["actors"]:
            ident = actor["annotation_actor_id"]
            key = (row["key"], ident)
            actors[key] = {
                "annotation_key": row["key"],
                "activity_type": row["activity_type"],
                "camera_id": row["camera_id"],
                "video_id": row["video_id"],
                "annotation_actor_id": ident,
                "annotation_actor_type": actor["annotation_actor_type"],
                "baseline_tracked": actor["detected_during_activity"],
                "sample_frames": sampled_annotation_frames(
                    row["start_frame"], row["end_frame_inclusive"],
                    video["fps"], sample_fps,
                ),
            }
            for frame in actors[key]["sample_frames"]:
                indexed[row["video_id"]][frame].append(key)
    return videos, indexed, actors


def longest_run(hits, sample_frames):
    """Consecutive hits on the sample grid, without joining a missed sample."""
    found = set(hits)
    best = run = 0
    for frame in sample_frames:
        run = run + 1 if frame in found else 0
        best = max(best, run)
    return best


def summarize(actors, frame_count, detections, unmatched, elapsed):
    groups = {}
    for activity in sorted({a["activity_type"] for a in actors}):
        rows = [a for a in actors if a["activity_type"] == activity]
        groups[activity] = {
            "actors": len(rows),
            "covered": sum(bool(a["hits"]) for a in rows),
            "event_supported_object_covered": sum(
                any(h["class"] in EVENT_PORTABLE for h in a["hits"])
                for a in rows if a["annotation_actor_type"] == "other"
            ),
            "sampled_actor_frames": sum(len(a["sample_frames"]) for a in rows),
            "actor_frames_with_geometry": sum(a["geometry_sample_count"] for a in rows),
            "detected_actor_frames": sum(len(a["hits"]) for a in rows),
            "mean_longest_run": round(statistics.mean(a["longest_detection_run"] for a in rows), 3),
            "median_box_area_fraction": round(statistics.median(a["median_box_area_fraction"] for a in rows if a["median_box_area_fraction"] is not None), 7),
        }
    return {
        "activity": groups,
        "unique_source_frames": frame_count,
        "detections": detections,
        "detections_per_frame": round(detections / frame_count, 4),
        "unmatched_boxes_per_frame": round(unmatched / frame_count, 4),
        "unmatched_box_note": "Not a false-positive rate: MEVA activity geometry is not exhaustive object ground truth.",
        "inference_wall_seconds": round(elapsed, 3),
        "inference_fps_on_this_host": round(frame_count / elapsed, 3),
    }


def run(manifest, diagnosis, model_path, size, confidence, output, sample_fps=2.0):
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.is_file():
        raise ValueError(f"Missing local model: {model_path}")
    videos, indexed, actors = requirements(manifest, diagnosis, sample_fps)
    model = YOLO(str(model_path))
    geometry = {}
    for video_id in indexed:
        video = videos[video_id]
        types = parse_types(resolve(video["types_path"]))
        geometry[video_id] = parse_geometry(resolve(video["geometry_path"]), video, types)
    hits = defaultdict(list)
    near_misses = defaultdict(list)
    sizes = defaultdict(list)
    decoded = detections = unmatched = 0
    inference_seconds = 0.0
    begun = time.perf_counter()
    for video_id, video in videos.items():
        if video_id not in indexed:
            continue
        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video['filename']}")
        try:
            for frame, keys in sorted(indexed[video_id].items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode {video['filename']} frame {frame}")
                decoded += 1
                tick = time.perf_counter()
                prediction = model.predict(pixels, imgsz=size, conf=confidence, verbose=False)[0]
                inference_seconds += time.perf_counter() - tick
                boxes = prediction.boxes
                found = [
                    (model.names[int(cls)], float(score), box.tolist())
                    for box, cls, score in zip(
                        boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                    )
                ] if boxes is not None else []
                detections += len(found)
                references = []
                for key in keys:
                    actor = actors[key]
                    reference = geometry[video_id].get(actor["annotation_actor_id"], {}).get(frame)
                    if reference is None:
                        continue
                    references.append(reference)
                    width = reference[2] - reference[0]
                    height = reference[3] - reference[1]
                    sizes[key].append((width, height, width * height / (video["width"] * video["height"])))
                    candidates = sorted(
                        ((box_iou(box, reference), name, score, box)
                         for name, score, box in found), reverse=True,
                    )
                    compatible = next(
                        (item for item in candidates
                         if item[0] >= 0.1 and actor_compatible(actor["annotation_actor_type"], item[1])),
                        None,
                    )
                    if compatible:
                        iou, name, score, box = compatible
                        hits[key].append({
                            "frame": frame, "class": name, "confidence": round(score, 4),
                            "iou": round(iou, 4), "box": [round(v, 1) for v in box],
                        })
                    elif candidates:
                        iou, name, score, box = candidates[0]
                        near_misses[key].append({
                            "frame": frame, "class": name, "confidence": round(score, 4),
                            "iou": round(iou, 4), "box": [round(v, 1) for v in box],
                        })
                unmatched += sum(
                    not any(box_iou(box, reference) >= 0.1 for reference in references)
                    for _, _, box in found
                )
        finally:
            cap.release()
        print(f"{model_path.name} {size}px conf={confidence:.2f}: {video['camera_id']} {len(indexed[video_id])} frames", flush=True)
    details = []
    for key, actor in actors.items():
        extent = sizes[key]
        if not extent:
            raise ValueError(f"No MEVA geometry on sampled frames for {actor['annotation_key']} actor {actor['annotation_actor_id']}")
        scores = [h["confidence"] for h in hits[key]]
        details.append({
            **actor,
            "geometry_sample_count": len(extent),
            "median_box_width": round(statistics.median(x[0] for x in extent), 2) if extent else None,
            "median_box_height": round(statistics.median(x[1] for x in extent), 2) if extent else None,
            "median_box_area_fraction": statistics.median(x[2] for x in extent) if extent else None,
            "hit_count": len(hits[key]),
            "event_supported_object_hit_count": sum(
                h["class"] in EVENT_PORTABLE for h in hits[key]
            ) if actor["annotation_actor_type"] == "other" else None,
            "longest_detection_run": longest_run((x["frame"] for x in hits[key]), actor["sample_frames"]),
            "mean_confidence": round(statistics.mean(scores), 4) if scores else None,
            "median_confidence": round(statistics.median(scores), 4) if scores else None,
            "hits": hits[key],
            "nearest_boxes_when_missed": near_misses[key],
        })
    report = {
        "method": "YOLO.predict on every exact 2 FPS frame inside all 89 DIRECT intervals. Annotation geometry and class compatibility applied only after prediction. IoU >= 0.1 is one-frame coverage, not event recognition; activity annotations are not exhaustive negative labels.",
        "model": model_path.name,
        "model_sha256": sha256(model_path),
        "detection_size": size,
        "confidence_threshold": confidence,
        "sample_fps": sample_fps,
        "summary": summarize(details, decoded, detections, unmatched, inference_seconds),
        "total_wall_seconds": round(time.perf_counter() - begun, 3),
        "actors": details,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
