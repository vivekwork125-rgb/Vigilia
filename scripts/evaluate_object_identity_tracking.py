"""Summarize target-only track continuity; withhold MOT scores until review completes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.object_identity_benchmark import load_annotations, validate_dataset


def evaluate(clips_path, annotations_path, detection_diagnostic, *, allow_partial=False):
    validation = validate_dataset(clips_path, annotations_path)
    if not validation["valid_schema"]:
        raise ValueError("Manual annotation schema invalid")
    if not validation["complete_pilot"] and not allow_partial:
        raise ValueError("Manual pilot review incomplete; use --allow-partial-diagnostic "
                         "for reviewed-target continuity only")
    detection = json.loads(Path(detection_diagnostic).read_text())
    if detection["dataset_validation"]["complete_frames"] != validation["complete_frames"]:
        raise ValueError("Detection diagnostic uses a different annotation completion state")
    records = load_annotations(annotations_path)
    counts = Counter((clip, ann["object_id"]) for (clip, _), row in records.items()
                     for ann in row["annotations"] if ann["visibility"] != "not_visible"
                     and ann["class_label"] != "person")
    results = {}
    for config, row in detection["results"].items():
        trajectories = {}
        for (clip_id, object_id), count in sorted(counts.items()):
            name = f"{clip_id}:{object_id}"
            matched = row["target_track_diagnostics"].get(name, {})
            trajectories[name] = {
                "manually_reviewed_visible_frames": count,
                "matched_frames": matched.get("matched_frames", 0),
                "provisional_track_ids": matched.get("provisional_track_ids", []),
                "provisional_id_changes_among_matched_pairs": matched.get(
                    "provisional_id_changes_among_matched_pairs", 0),
                "unlinked_matched_detections": matched.get("unlinked_matched_detections", 0),
            }
        results[config] = {
            "reviewed_target_trajectories": len(trajectories),
            "target_tracks_with_multiple_provisional_ids": sum(
                len(t["provisional_track_ids"]) > 1 for t in trajectories.values()),
            "total_provisional_id_changes_among_matched_pairs": sum(
                t["provisional_id_changes_among_matched_pairs"] for t in trajectories.values()),
            "trajectories": trajectories,
            "idf1": None, "hota": None,
            "metric_scope": ("reviewed pilot target continuity; no formal IDF1/HOTA implementation"
                             if validation["complete_pilot"] else
                             "partial target-only diagnostic; no global tracking precision/recall"),
        }
    return {"dataset_validation": validation, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path,
                        default=ROOT / "datasets/meva/object-identity/clips.json")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "datasets/meva/object-identity/annotations.jsonl")
    parser.add_argument("--detection-diagnostic", type=Path,
                        default=ROOT / "data/meva/object-identity/detection-diagnostic.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/meva/object-identity/tracking-diagnostic.json")
    parser.add_argument("--allow-partial-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(args.clips, args.annotations, args.detection_diagnostic,
                          allow_partial=args.allow_partial_diagnostic)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, str(exc) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
