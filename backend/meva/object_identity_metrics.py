"""One-to-one spatial matching for independently reviewed object boxes."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
from scipy.optimize import linear_sum_assignment

from .object_presence import box_iou

CLASS_COMPATIBILITY = {
    "person": {"person"},
    "vehicle": {"car", "truck", "bus", "motorcycle"},
    "bag": {"backpack", "handbag", "suitcase"},
    "bottle": {"bottle"},
    "cup": {"cup"},
    "phone": {"cell phone"},
}


def class_compatible(annotation, detection):
    label = annotation["class_label"]
    if label in {"unknown_object", "other_visible_object"}:
        return detection["class"] != "person"
    return detection["class"] in CLASS_COMPATIBILITY[label]


def match_frame(annotations, detections, *, iou_threshold=.5):
    """Global one-to-one assignment: each visible target/detection matches at most once."""
    visible = [(i, row) for i, row in enumerate(annotations)
               if row["visibility"] != "not_visible" and row["bbox_xyxy"] is not None]
    if not visible or not detections:
        return []
    weights = np.zeros((len(visible), len(detections)), dtype=float)
    for i, (_, ann) in enumerate(visible):
        for j, det in enumerate(detections):
            if class_compatible(ann, det):
                iou = box_iou(ann["bbox_xyxy"], det["box"])
                if iou >= iou_threshold:
                    weights[i, j] = 1. + iou
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [{"annotation_index": visible[int(i)][0], "detection_index": int(j),
             "iou": round(float(weights[i, j]-1), 4)}
            for i, j in zip(rows, cols) if weights[i, j] > 0]


def target_track_diagnostics(frame_rows, matched_detection_ids_to_tracks):
    """Target-only continuity. No HOTA/IDF1 or global identity-accuracy claim."""
    by_identity = defaultdict(list)
    for row in frame_rows:
        for match in row["matches"]:
            annotation = row["annotations"][match["annotation_index"]]
            detection = row["detections"][match["detection_index"]]
            by_identity[(row["clip_id"], annotation["object_id"])].append(
                (row["frame_index"], matched_detection_ids_to_tracks.get(detection["detection_id"])))
    result = {}
    for (clip_id, object_id), history in by_identity.items():
        history.sort()
        known_tracks = [track for _, track in history if track]
        switches_among_matched_pairs = sum(
            a is not None and b is not None and a != b
            for (_, a), (_, b) in pairwise(history)
        )
        result[f"{clip_id}:{object_id}"] = {
            "matched_frames": len(history),
            "provisional_track_ids": sorted(set(known_tracks)),
            "provisional_id_changes_among_matched_pairs": switches_among_matched_pairs,
            "unlinked_matched_detections": len(history)-len(known_tracks),
        }
    return result
