"""Independent, human-authored object-identity pilot metadata and validation.

This module never imports detector outputs or production observations. It only
validates a reviewer's annotations against the source-video clip manifest.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from .dataset import load_manifest

DATASET_VERSION = "1.0"
VISIBILITY = {"visible", "partially_visible", "occluded", "not_visible", "uncertain"}
OCCLUSION = {"none", "partial", "heavy", "uncertain"}
IDENTITY_CONFIDENCE = {"high", "medium", "uncertain"}
REVIEW_STATE = {"partial", "complete"}
CLASSES = {"person", "vehicle", "bag", "bottle", "cup", "phone",
           "other_visible_object", "unknown_object"}


def clip_index(clips_path, source_manifest=None):
    """Validate deterministic clips against the independently stored MEVA manifest."""
    data = json.loads(Path(clips_path).read_text())
    source = load_manifest(source_manifest) if source_manifest else load_manifest()
    videos = {v["video_id"]: v for v in source["videos"]}
    if data.get("dataset_version") != DATASET_VERSION or not data.get("clips"):
        raise ValueError("Invalid object-identity dataset version or empty clips")
    ids, used_frames = set(), defaultdict(set)
    for clip in data["clips"]:
        ident = clip["clip_id"]
        if ident in ids:
            raise ValueError(f"Duplicate clip ID: {ident}")
        ids.add(ident)
        video = videos.get(clip["video_id"])
        if not video or clip["camera_id"] != video["camera_id"]:
            raise ValueError(f"Invalid source video/camera for {ident}")
        if clip.get("source_video") != video["local_path"]:
            raise ValueError(f"Invalid source video reference for {ident}")
        for key in ("source_sha256", "width", "height", "fps"):
            expected = video["sha256"] if key == "source_sha256" else video[key]
            if clip.get(key) != expected:
                raise ValueError(f"Source {key} mismatch for {ident}")
        frames = clip.get("frame_indices")
        rate = clip.get("sample_fps")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid sample FPS for {ident}")
        step = video["fps"] / rate
        if (abs(step-round(step)) > 1e-6 or not isinstance(frames, list)
                or not frames or any(type(frame) is not int for frame in frames)):
            raise ValueError(f"Invalid sampling grid for {ident}")
        if frames != list(range(frames[0], frames[0] + len(frames)*round(step), round(step))):
            raise ValueError(f"Nonconsecutive sampled frames for {ident}")
        if frames[0] < 0 or frames[-1] >= video["frame_count"]:
            raise ValueError(f"Out-of-range frames for {ident}")
        if (clip.get("start_seconds") != frames[0]/video["fps"]
                or clip.get("duration_seconds") != len(frames)/rate):
            raise ValueError(f"Clip clock/duration mismatch for {ident}")
        if used_frames[clip["video_id"]].intersection(frames):
            raise ValueError(f"Overlapping clips for {ident}")
        used_frames[clip["video_id"]].update(frames)
    return data, {c["clip_id"]: c for c in data["clips"]}


def frame_record(clip, frame_index, *, annotations=None, review_state="partial", note=""):
    if frame_index not in clip["frame_indices"]:
        raise ValueError("Frame is outside selected clip")
    return {
        "dataset_version": DATASET_VERSION,
        "clip_id": clip["clip_id"],
        "video_id": clip["video_id"],
        "camera_id": clip["camera_id"],
        "source_sha256": clip["source_sha256"],
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / clip["fps"],
        "review_state": review_state,
        "review_note": note,
        "annotations": annotations or [],
    }


def validate_record(row, clips):
    if not isinstance(row, dict):
        raise TypeError("Frame annotation must be an object")
    clip = clips.get(row.get("clip_id"))
    if clip is None:
        raise ValueError("Unknown clip ID")
    frame = row.get("frame_index")
    if type(frame) is not int or frame not in clip["frame_indices"]:
        raise ValueError("Invalid frame index for clip")
    for key in ("dataset_version", "video_id", "camera_id", "source_sha256"):
        expected = DATASET_VERSION if key == "dataset_version" else clip[key]
        if row.get(key) != expected:
            raise ValueError(f"Invalid {key} for clip/frame")
    if (type(row.get("timestamp_seconds")) not in (int, float)
            or not math.isfinite(row["timestamp_seconds"])
            or abs(row["timestamp_seconds"] - frame/clip["fps"]) > 1e-6):
        raise ValueError("Invalid source timestamp")
    if row.get("review_state") not in REVIEW_STATE:
        raise ValueError("Invalid review state")
    if not isinstance(row.get("review_note"), str):
        raise TypeError("Review note must be text")
    anns = row.get("annotations")
    if not isinstance(anns, list):
        raise TypeError("Annotations must be a list")
    if row["review_state"] == "complete" and not anns and row["review_note"] != "EMPTY_SCENE_CONFIRMED":
        raise ValueError("An empty complete frame needs EMPTY_SCENE_CONFIRMED")
    seen = set()
    for ann in anns:
        if not isinstance(ann, dict):
            raise TypeError("Object annotation must be an object")
        object_id = ann.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("Missing object ID")
        if object_id in seen:
            raise ValueError(f"Duplicate object ID in frame: {object_id}")
        seen.add(object_id)
        if ann.get("class_label") not in CLASSES:
            raise ValueError(f"Invalid class for {object_id}")
        visibility = ann.get("visibility")
        if visibility not in VISIBILITY or ann.get("occlusion") not in OCCLUSION:
            raise ValueError(f"Invalid visibility/occlusion for {object_id}")
        if ann.get("identity_confidence") not in IDENTITY_CONFIDENCE:
            raise ValueError(f"Invalid identity confidence for {object_id}")
        if not isinstance(ann.get("annotator_note", ""), str):
            raise TypeError(f"Invalid annotator note for {object_id}")
        if not isinstance(ann.get("reidentification_note", ""), str):
            raise TypeError(f"Invalid reidentification note for {object_id}")
        box = ann.get("bbox_xyxy")
        if visibility == "not_visible":
            if box is not None:
                raise ValueError(f"Absent object has a box: {object_id}")
        else:
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                    or not 0 <= box[0] < box[2] <= clip["width"]
                    or not 0 <= box[1] < box[3] <= clip["height"]):
                raise ValueError(f"Invalid source-frame box for {object_id}")
            if visibility == "visible" and ann["occlusion"] not in {"none", "uncertain"}:
                raise ValueError(f"Visible object has contradictory occlusion: {object_id}")
            if visibility == "occluded" and ann["occlusion"] not in {"heavy", "partial"}:
                raise ValueError(f"Occluded object needs occlusion severity: {object_id}")
        if object_id.startswith("UNRESOLVED-") and ann["identity_confidence"] != "uncertain":
            raise ValueError("Unresolved identity must have uncertain confidence")
    return row


def load_annotations(path):
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["clip_id"], row["frame_index"])
        if key in records:
            raise ValueError(f"Duplicate annotation frame at line {number}: {key}")
        records[key] = row
    return records


def save_annotations(path, records):
    """Write a complete JSONL snapshot atomically so interrupted saves can resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    content = "".join(json.dumps(records[key], sort_keys=True) + "\n" for key in sorted(records))
    temp.write_text(content)
    temp.replace(path)


def validate_dataset(clips_path, annotations_path, source_manifest=None):
    _, clips = clip_index(clips_path, source_manifest)
    records = load_annotations(annotations_path)
    errors = []
    for key, row in records.items():
        try:
            validate_record(row, clips)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
    by_object = defaultdict(list)
    for (clip_id, frame), row in records.items():
        if clip_id not in clips:
            continue
        for ann in row.get("annotations", []):
            if isinstance(ann, dict) and ann.get("object_id"):
                by_object[(clip_id, ann["object_id"])].append((frame, ann))
    for (clip_id, object_id), history in by_object.items():
        history.sort()
        labels = {ann.get("class_label") for _, ann in history}
        if len(labels) > 1 and not all(ann.get("annotator_note") for _, ann in history):
            errors.append(f"{clip_id}/{object_id}: class changed without notes")
        if object_id.startswith("UNRESOLVED-") and len(history) > 1:
            errors.append(f"{clip_id}/{object_id}: unresolved ID reused across frames")
        indices = clips[clip_id]["frame_indices"]
        rank = {frame: i for i, frame in enumerate(indices)}
        for (previous, _), (current, ann) in pairwise(history):
            if rank[current]-rank[previous] > 1 and not ann.get("reidentification_note"):
                errors.append(f"{clip_id}/{object_id}: reappearance at {current} needs a note")
    planned = sum(len(c["frame_indices"]) for c in clips.values())
    complete = sum(row.get("review_state") == "complete" for row in records.values())
    visible = [ann for row in records.values() for ann in row.get("annotations", [])
               if ann.get("visibility") != "not_visible"]
    report = {
        "valid_schema": not errors,
        "complete_pilot": not errors and complete == planned,
        "clip_count": len(clips), "camera_count": len({c["camera_id"] for c in clips.values()}),
        "planned_frames": planned, "saved_frames": len(records), "complete_frames": complete,
        "partial_frames": len(records)-complete,
        "visible_instances": len(visible),
        "unique_object_ids": len(by_object),
        "uncertain_instances": sum(ann.get("identity_confidence") == "uncertain"
                                   for row in records.values() for ann in row.get("annotations", [])),
        "visibility": dict(Counter(ann.get("visibility") for row in records.values()
                                   for ann in row.get("annotations", []))),
        "errors": errors,
    }
    return report
