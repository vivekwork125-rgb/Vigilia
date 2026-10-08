#!/usr/bin/env python3
"""Probe runtime-derived person-track ROIs on fixed MEVA diagnostic frames."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

SKIP = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


def predict(model, pixels, size, confidence):
    output = model.predict(pixels, imgsz=size, conf=confidence, verbose=False)[0].boxes
    if output is None:
        return []
    return [
        (model.names[int(cls)], float(score), box.tolist())
        for box, cls, score in zip(
            output.xyxy.cpu().numpy(), output.cls.cpu().numpy(), output.conf.cpu().numpy()
        )
    ]


def run(args):
    from app.temporal_events import PORTABLE
    from meva.annotations import parse_geometry, parse_types
    from meva.dataset import load_manifest, resolve, sha256
    from meva.perception_experiment import requirements
    from meva.selective_objects import (
        context_frames,
        deduplicate,
        link_objects,
        person_roi,
        person_samples,
        score_tracklets,
        supported_tracklets,
    )
    from ultralytics import YOLO

    manifest = load_manifest(args.manifest)
    diagnosis = json.loads((args.baseline_run / "failure-diagnostics.json").read_text())
    videos, indexed, actors = requirements(manifest, diagnosis, 2.0)
    object_actors = {
        key: actor for key, actor in actors.items()
        if actor["annotation_actor_type"] == "other"
    }
    by_annotation = {
        row["key"]: [track for actor in row["actors"] if actor["annotation_actor_type"] == "person" for track in actor["track_ids"]]
        for row in diagnosis["annotations"]
    }
    for actor in object_actors.values():
        actor["person_track_ids"] = by_annotation[actor["annotation_key"]]
    people = person_samples(args.baseline_run / "vigilia.db")
    model = YOLO(str(args.model))
    detections, persons, geometry = defaultdict(lambda: defaultdict(list)), {}, {}
    source_frames = crops = detections_total = 0
    begun = time.perf_counter()
    for video_id, video in videos.items():
        frames = {
            frame: keys for frame, keys in indexed.get(video_id, {}).items()
            if any(key in object_actors for key in keys)
        }
        if not frames:
            continue
        if args.context_seconds:
            for neighbor in context_frames(
                frames, video["frame_count"], video["fps"], args.context_seconds
            ):
                frames.setdefault(neighbor, [])
        geometry[video_id] = parse_geometry(
            resolve(video["geometry_path"]), video,
            parse_types(resolve(video["types_path"])),
        )
        persons[video_id] = people[video_id]
        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        if not cap.isOpened():
            raise ValueError(video["filename"])
        try:
            for frame in sorted(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode {video['filename']} frame {frame}")
                source_frames += 1
                found = []
                for person in people[video_id].get(frame, []):
                    left, top, right, bottom = person_roi(
                        person["box"], video["width"], video["height"],
                        args.strategy, args.factor,
                    )
                    if right - left < 32 or bottom - top < 32:
                        continue
                    crops += 1
                    for name, confidence, box in predict(
                        model, pixels[top:bottom, left:right], args.size, args.confidence
                    ):
                        if name in SKIP:
                            continue
                        found.append({
                            "class": name, "confidence": confidence,
                            "box": [box[0] + left, box[1] + top, box[2] + left, box[3] + top],
                            "frame": frame, "t": frame / video["fps"],
                        })
                detections[video_id][frame] = deduplicate(found)
                detections_total += len(detections[video_id][frame])
        finally:
            cap.release()
    raw, tracked, associated = set(), set(), set()
    portable_raw, portable_tracked, portable_associated = set(), set(), set()
    tracklet_total = supported_total = 0
    supported_examples = []
    for video_id, frames in detections.items():
        tracks = link_objects(frames, videos[video_id]["fps"])
        tracklet_total += len(tracks)
        supported = supported_tracklets(tracks, persons[video_id])
        supported_total += len(supported)
        supported_examples.extend({
            "video_id": video_id, "class": track[0]["class"],
            "frames": [hit["frame"] for hit in track],
        } for track in supported)
        a, b, c = score_tracklets(
            tracks,
            {k: v for k, v in object_actors.items() if v["video_id"] == video_id},
            geometry[video_id], persons[video_id],
        )
        raw.update(a)
        tracked.update(b)
        associated.update(c)
        a, b, c = score_tracklets(
            tracks,
            {k: v for k, v in object_actors.items() if v["video_id"] == video_id},
            geometry[video_id], persons[video_id], PORTABLE,
        )
        portable_raw.update(a)
        portable_tracked.update(b)
        portable_associated.update(c)
    result = {
        "strategy": args.strategy, "factor": args.factor,
        "context_seconds": args.context_seconds,
        "model": args.model.name, "model_sha256": sha256(args.model),
        "size": args.size, "confidence": args.confidence,
        "source_frames": source_frames, "person_track_crops": crops,
        "object_detections_after_crop_dedup": detections_total,
        "object_tracklets": tracklet_total,
        "persistent_portable_associated_tracklets": supported_total,
        "wall_seconds": round(time.perf_counter() - begun, 3),
        "annotations": len(object_actors),
        "raw_object_coverage": len(raw),
        "object_track_coverage": len(tracked),
        "person_object_association_coverage": len(associated),
        "portable_raw_coverage": len(portable_raw),
        "portable_track_coverage": len(portable_tracked),
        "portable_association_coverage": len(portable_associated),
        "pickup_matches": 0, "placement_matches": 0,
        "pickup_placement_note": "Diagnostic crops are not runtime events; no production pickup or placement emitted.",
        "raw_keys": sorted(key[0] for key in raw),
        "tracked_keys": sorted(key[0] for key in tracked),
        "associated_keys": sorted(key[0] for key in associated),
        "supported_examples": supported_examples,
        "annotation_use": "Frames selected from benchmark labels. ROI, detector, linking and association use stored person tracks only. GT boxes score outputs after inference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"raw_keys", "tracked_keys", "associated_keys", "supported_examples"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "data/meva/improved-2fps")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--strategy", choices=("expanded", "lower", "interaction"), default="expanded")
    parser.add_argument("--factor", type=float, choices=(1.25, 1.5, 2.0), default=1.5)
    parser.add_argument("--context-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=ROOT / "data/meva/selective-objects/expanded-1.5.json")
    run(parser.parse_args())
