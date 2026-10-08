"""Evaluation-only selective small-object experiment from stored person tracks.

Annotations are consulted only after ROI generation, inference, and object
linking. Nothing in this module is imported by production processing.
"""

import json
import sqlite3
import statistics
from collections import defaultdict
from itertools import pairwise
from math import hypot
from pathlib import Path

from app.temporal_events import PORTABLE

from .evaluation import box_iou


def context_frames(anchors, frame_count, fps, seconds, sample_fps=2.0):
    """Add sampled source frames around diagnostic anchors, clamped to video."""
    if seconds < 0 or fps <= 0 or sample_fps <= 0:
        raise ValueError("Context, source FPS and sample FPS must be valid")
    step = round(fps / sample_fps)
    radius = round(seconds * sample_fps)
    return {
        anchor + offset * step
        for anchor in anchors
        for offset in range(-radius, radius + 1)
        if 0 <= anchor + offset * step < frame_count
    }


def person_roi(box, width, height, strategy="expanded", factor=1.5):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    if strategy == "lower":
        cy += 0.25 * (y2 - y1)
        h *= 0.75
    elif strategy == "interaction":
        w *= 1.4
        h *= 0.9
    elif strategy != "expanded":
        raise ValueError(strategy)
    return (
        max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
        min(width, int(cx + w / 2)), min(height, int(cy + h / 2)),
    )


def person_samples(database):
    db = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT o.video_id,o.track_id,o.boxes FROM observations o "
            "JOIN entities e ON o.entity_id=e.id WHERE e.object_type='person'"
        ).fetchall()
    finally:
        db.close()
    indexed = defaultdict(lambda: defaultdict(list))
    for video, track, boxes in rows:
        for sample in json.loads(boxes):
            indexed[video][sample["frame"]].append({"track_id": track, **sample})
    return indexed


def deduplicate(detections):
    """Repeated crops may see one object; retain its strongest detection."""
    kept = []
    for item in sorted(detections, key=lambda row: -row["confidence"]):
        if not any(
            other["class"] == item["class"]
            and box_iou(other["box"], item["box"]) >= 0.5
            for other in kept
        ):
            kept.append(item)
    return kept


def link_objects(frames, fps=30.0):
    """Conservative adjacent-sample, class-consistent object tracklets.

    These are diagnostic identities only. No identity is asserted from one hit.
    """
    tracklets, active = [], []
    for frame in sorted(frames):
        for detection in frames[frame]:
            candidates = []
            for track in active:
                prev = track[-1]
                if detection["class"] != prev["class"] or detection["t"] - prev["t"] > 1.0:
                    continue
                a, b = prev["box"], detection["box"]
                pa = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
                pb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                scale = max(1, a[3] - a[1], b[3] - b[1])
                distance = hypot(pb[0] - pa[0], pb[1] - pa[1]) / scale
                if distance <= 1.0:
                    candidates.append((distance, track))
            if candidates:
                _, chosen = min(candidates, key=lambda item: item[0])
                if chosen[-1]["frame"] != frame:
                    chosen.append(detection)
                    continue
            tracklets.append([detection])
        active = [track for track in tracklets if frame / fps - track[-1]["t"] <= 1]
    return tracklets


def relationship_metrics(tracklet, persons_by_frame):
    """Measured proximity/motion descriptors; no contact or ownership claim."""
    paired = defaultdict(list)
    for hit in tracklet:
        obj = hit["box"]
        ox, oy = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
        for person in persons_by_frame.get(hit["frame"], []):
            box = person["box"]
            px, py = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            distance = hypot(ox - px, oy - py) / max(1, box[3] - box[1])
            if distance <= 1.5:
                paired[person["track_id"]].append((hit, person, distance))
    result = {}
    for key, pairs in paired.items():
        if len(pairs) < 2:
            continue
        residuals, relative_speeds = [], []
        for (obj_a, person_a, _), (obj_b, person_b, _) in pairwise(pairs):
            if obj_b["t"] - obj_a["t"] > 1:
                continue
            a, b = obj_a["box"], obj_b["box"]
            c, d = person_a["box"], person_b["box"]
            ov = ((b[0] + b[2] - a[0] - a[2]) / 2, (b[1] + b[3] - a[1] - a[3]) / 2)
            pv = ((d[0] + d[2] - c[0] - c[2]) / 2, (d[1] + d[3] - c[1] - c[3]) / 2)
            difference = hypot(ov[0] - pv[0], ov[1] - pv[1])
            residuals.append(difference / max(hypot(*ov), hypot(*pv), 1))
            relative_speeds.append(difference / max(1, c[3] - c[1]) / (obj_b["t"] - obj_a["t"]))
        distances = [distance for _, _, distance in pairs]
        result[key] = {
            "persistence_samples": len(pairs),
            "mean_normalized_distance": sum(distances) / len(distances),
            "median_normalized_distance": round(statistics.median(distances), 4),
            "median_relative_speed": round(statistics.median(relative_speeds), 4) if relative_speeds else None,
            "median_motion_residual": round(statistics.median(residuals), 4) if residuals else None,
            "approach_distance_change": round(max(0, distances[0] - distances[-1]), 4),
            "separation_distance_change": round(max(0, distances[-1] - distances[0]), 4),
        }
    return result


def associate(tracklet, persons_by_frame):
    """Return one persistent spatial candidate, or None for ambiguous proximity.

    Motion descriptors are available for audit but proximity alone is never
    promoted to a pickup, placement, ownership or contact claim.
    """
    metrics = relationship_metrics(tracklet, persons_by_frame)
    qualified = sorted(
        (row["mean_normalized_distance"], key)
        for key, row in metrics.items()
    )
    if not qualified or len(tracklet) < 2:
        return None
    if len(qualified) > 1 and qualified[1][0] - qualified[0][0] < 0.25:
        return None
    return qualified[0][1]


def score_tracklets(tracklets, actors, geometry, persons_by_frame, classes=None):
    """Use GT only after links/associations have been generated."""
    hits, tracked, associated = set(), set(), set()
    for tracklet in tracklets:
        if classes is not None and tracklet[0]["class"] not in classes:
            continue
        owner = associate(tracklet, persons_by_frame)
        for key, actor in actors.items():
            reference = geometry.get(actor["annotation_actor_id"], {})
            if any(
                hit["frame"] in actor["sample_frames"]
                and hit["frame"] in reference
                and box_iou(hit["box"], reference[hit["frame"]]) >= 0.1
                for hit in tracklet
            ):
                hits.add(key)
                if len(tracklet) >= 2:
                    tracked.add(key)
                    if owner and owner in actor.get("person_track_ids", ()):
                        associated.add(key)
    return hits, tracked, associated


def supported_tracklets(tracklets, persons_by_frame):
    return [
        track for track in tracklets
        if len(track) >= 2 and track[0]["class"] in PORTABLE
        and associate(track, persons_by_frame)
    ]
