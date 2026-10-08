"""Offline vehicle-rule comparison on stored tracks; never imported by runtime."""

import json
import sqlite3
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from app.temporal_events import Candidate, center, speed, vehicle_transitions

from .annotations import parse_activities, parse_geometry, parse_types
from .dataset import load_manifest, resolve
from .evaluation import (
    detection_metrics,
    match,
    spatial_compatibility,
)
from .mapping import mapping

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def offline_transitions(key, samples, method):
    """Fixed, predeclared formulations using only observed track geometry.

    A transition needs a low/high state on both sides, cumulative displacement,
    and at least two samples of confirmation. Gaps greater than 1.5 seconds
    cannot supply continuity. These are diagnostic proposals, not runtime rules.
    """
    if method == "current":
        return vehicle_transitions(key, samples)
    if len(samples) < 7:
        return []
    scale = max(1, statistics.median(s["box"][3] - s["box"][1] for s in samples))
    raw = [speed(a, b, scale) for a, b in pairwise(samples)]
    smooth = [statistics.median(raw[max(0, i - 1):min(len(raw), i + 2)]) for i in range(len(raw))]
    result = []
    for pivot in range(3, len(smooth) - 2):
        before = smooth[pivot - 3:pivot]
        after = smooth[pivot:pivot + 3]
        if samples[pivot + 3]["t"] - samples[pivot - 3]["t"] > 4.5:
            continue
        if method == "velocity_transition":
            low = max(before) <= 0.12
            high = min(after[1:]) >= 0.28
            reverse_low = max(after) <= 0.12
            reverse_high = min(before[:2]) >= 0.28
        elif method == "change_point":
            low = statistics.median(before) <= 0.12
            high = statistics.median(after) >= 0.28
            reverse_low = statistics.median(after) <= 0.12
            reverse_high = statistics.median(before) >= 0.28
        elif method == "multi_window":
            low = sum(v <= 0.12 for v in before) >= 2
            high = sum(v >= 0.28 for v in after) >= 2
            reverse_low = sum(v <= 0.12 for v in after) >= 2
            reverse_high = sum(v >= 0.28 for v in before) >= 2
        else:
            raise ValueError(method)
        dx = center(samples[pivot + 3])[0] - center(samples[pivot])[0]
        dy = center(samples[pivot + 3])[1] - center(samples[pivot])[1]
        after_displacement = (dx * dx + dy * dy) ** 0.5 / scale
        dx = center(samples[pivot])[0] - center(samples[pivot - 3])[0]
        dy = center(samples[pivot])[1] - center(samples[pivot - 3])[1]
        before_displacement = (dx * dx + dy * dy) ** 0.5 / scale
        if low and high and after_displacement >= 0.3:
            result.append(Candidate("started_moving", (key,), samples[pivot]["t"], samples[pivot + 3]["t"]))
        if reverse_high and reverse_low and before_displacement >= 0.3:
            result.append(Candidate("stopped", (key,), samples[pivot - 3]["t"], samples[pivot]["t"]))
    # Nearby pivots describe the same physical transition. Keep the first
    # interval in a two-second neighborhood; no annotation information is used.
    unique = []
    for candidate in result:
        if not any(c.kind == candidate.kind and abs(c.start - candidate.start) <= 2 for c in unique):
            unique.append(candidate)
    return unique


def stored_tracks(database):
    db = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT o.track_id,o.boxes,e.object_type,o.video_id "
            "FROM observations o JOIN entities e ON o.entity_id=e.id "
            "WHERE e.object_type IN ('car','truck','bus','motorcycle')"
        ).fetchall()
    finally:
        db.close()
    return [{"track_id": k, "boxes": json.loads(b), "object_type": t, "video_id": v} for k, b, t, v in rows]


def compare(run_dir, manifest_path):
    manifest = load_manifest(manifest_path)
    videos = {v["video_id"]: v for v in manifest["videos"]}
    truth_by_video, geometry_by_video = defaultdict(list), {}
    for video in videos.values():
        types = parse_types(resolve(video["types_path"]))
        geometry_by_video[video["video_id"]] = parse_geometry(resolve(video["geometry_path"]), video, types)
        for row in parse_activities(resolve(video["annotation_path"]), video, types):
            row.update(mapping(row["activity_type"]))
            if row["activity_type"] in {"vehicle_starts", "vehicle_stops"} and row["status"] == "DIRECT":
                row["key"] = f"{row['video_id']}:{row['annotation_id']}:{row['span_index']}"
                truth_by_video[video["video_id"]].append(row)
    tracks = stored_tracks(Path(run_dir) / "vigilia.db")
    result = {}
    truths = [truth for video_truths in truth_by_video.values() for truth in video_truths]
    for method in ("current", "velocity_transition", "change_point", "multi_window"):
        predictions = []
        for track in tracks:
            video = videos[track["video_id"]]
            for candidate in offline_transitions(track["track_id"], track["boxes"], method):
                pred = {
                    "event_type": candidate.kind, "video_id": track["video_id"],
                    "camera_id": video["camera_id"], "start_seconds": candidate.start,
                    "end_seconds": candidate.end, "entity_types": [track["object_type"]],
                    "supporting_tracks": [track], "spatial_scores": {},
                }
                for truth in truth_by_video[track["video_id"]]:
                    if truth["event_type"] == candidate.kind:
                        pred["spatial_scores"][truth["key"]] = spatial_compatibility(
                            pred, truth, geometry_by_video[track["video_id"]], video["fps"], 2.0
                        )
                predictions.append(pred)
        pairs = match(predictions, truths)
        result[method] = {
            **detection_metrics(predictions, truths, pairs),
            "by_type": {
                typ: detection_metrics(
                    [p for p in predictions if p["event_type"] == typ],
                    [t for t in truths if t["event_type"] == typ],
                    match(
                        [p for p in predictions if p["event_type"] == typ],
                        [t for t in truths if t["event_type"] == typ],
                    ),
                )
                for typ in ("started_moving", "stopped")
            },
        }
    return result
