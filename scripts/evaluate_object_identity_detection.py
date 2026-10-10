"""Score cached runtime detections against independent manual object boxes.

Incomplete reviews yield target-box coverage only. Precision/recall require
every pilot frame to be marked complete after exhaustive manual review.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.object_identity_benchmark import (
    clip_index,
    load_annotations,
    validate_dataset,
)
from meva.object_identity_metrics import match_frame, target_track_diagnostics
from meva.object_presence import deduplicate_frame

DEFAULT_RUNS = (
    ROOT / "data/meva/object-presence-yolo11n-640-c030",
    ROOT / "data/meva/object-presence-yolo11n-interaction2-640-c030",
)
EVALUATED_CLASSES = {"car", "truck", "bus", "motorcycle", "backpack",
                     "handbag", "suitcase", "bottle", "cup", "cell phone"}


def evaluate(clips_path, annotation_path, run_dirs, *, allow_partial=False):
    status = validate_dataset(clips_path, annotation_path)
    if not status["valid_schema"]:
        raise ValueError("Manual annotation schema has errors: " + "; ".join(status["errors"]))
    if not status["complete_pilot"] and not allow_partial:
        raise ValueError("Manual pilot review is incomplete; use --allow-partial-diagnostic "
                         "for target-box coverage only, with no precision/recall")
    _, clips = clip_index(clips_path)
    records = load_annotations(annotation_path)
    results = {}
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg.get("protocol") != "full-video sampled runtime-only detection; no activity annotations":
            raise ValueError(f"Not a runtime-only cache: {run_dir}")
        video_cache = {}
        tracks = json.loads((run_dir / "evaluation.json").read_text())["object_tracks"]
        track_by_detection = {did: tr["track_id"] for tr in tracks for did in tr["detection_ids"]}
        rows = []
        for (clip_id, frame), record in sorted(records.items()):
            clip = clips[clip_id]
            vid = clip["video_id"]
            if vid not in video_cache:
                cache = json.loads((run_dir / f"{vid}.json").read_text())
                if (cache["source_sha256"] != clip["source_sha256"]
                        or cache["model_sha256"] != cfg["model_sha256"]
                        or cache["sample_fps"] != clip["sample_fps"]):
                    raise ValueError(f"Source/config mismatch in {run_dir}/{vid}")
                indexed = defaultdict(list)
                for det in cache["detections"]:
                    indexed[det["frame"]].append(det)
                video_cache[vid] = indexed
            raw = [d for d in video_cache[vid].get(frame, []) if d["class"] in EVALUATED_CLASSES]
            detections = deduplicate_frame(raw)
            object_annotations = [ann for ann in record["annotations"]
                                  if ann["class_label"] != "person"]
            matches = match_frame(object_annotations, detections)
            rows.append({"clip_id": clip_id, "camera_id": clip["camera_id"],
                         "frame_index": frame, "review_state": record["review_state"],
                         "annotations": object_annotations, "detections": detections,
                         "raw_detection_count": len(raw), "matches": matches})
        visible = sum(ann["visibility"] != "not_visible" for row in rows
                      for ann in row["annotations"])
        matched = sum(len(row["matches"]) for row in rows)
        by_camera = {}
        for camera in sorted({row["camera_id"] for row in rows}):
            subset = [row for row in rows if row["camera_id"] == camera]
            by_camera[camera] = {
                "saved_review_frames": len(subset),
                "visible_target_boxes": sum(ann["visibility"] != "not_visible"
                                            for row in subset for ann in row["annotations"]),
                "matched_target_boxes": sum(len(row["matches"]) for row in subset),
            }
        full = status["complete_pilot"]
        detection_count = sum(len(row["detections"]) for row in rows)
        entry = {
            "model": cfg["model"], "model_sha256": cfg["model_sha256"],
            "imgsz": cfg["imgsz"], "confidence_threshold": cfg["confidence"],
            "roi_strategy": cfg.get("roi_strategy"), "iou_threshold": .5,
            "saved_review_frames": len(rows),
            "visible_target_boxes": visible,
            "matched_target_boxes": matched,
            "raw_evaluated_class_detections": sum(row["raw_detection_count"] for row in rows),
            "after_same_class_dedup": detection_count,
            "duplicate_suppressed": sum(row["raw_detection_count"]-len(row["detections"])
                                        for row in rows),
            "by_camera": by_camera,
            "target_track_diagnostics": target_track_diagnostics(rows, track_by_detection),
            "precision": matched/detection_count if full and detection_count else None,
            "recall": matched/visible if full and visible else None,
            "false_positives": detection_count-matched if full else None,
            "false_negatives": visible-matched if full else None,
            "review_complete": full,
            "metric_scope": ("full reviewed pilot; selected object-class inventory"
                             if full else "partial target-box diagnostic; no precision/recall"),
        }
        results[run_dir.name] = entry
    return {"dataset_validation": status, "results": results,
            "class_policy": "bag=backpack/handbag/suitcase; vehicle=car/truck/bus/motorcycle; "
                            "unknown object is class-agnostic within evaluated mobile-object "
                            "detector classes. One-to-one IoU >=0.5."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path,
                        default=ROOT / "datasets/meva/object-identity/clips.json")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "datasets/meva/object-identity/annotations.jsonl")
    parser.add_argument("--run-dir", type=Path, action="append", dest="run_dirs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/meva/object-identity/detection-diagnostic.json")
    parser.add_argument("--allow-partial-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(args.clips, args.annotations, args.run_dirs or DEFAULT_RUNS,
                          allow_partial=args.allow_partial_diagnostic)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, str(exc) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in row.items()
                            if k not in {"target_track_diagnostics"}}
                      for key, row in result["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
