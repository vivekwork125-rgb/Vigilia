"""Evaluation-only attribution of MEVA misses to perception, rules, and search.

Ground-truth geometry is read here, after production processing. This module is
never imported by the video processing or event generation code.
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from math import hypot

from sqlalchemy import select

from .annotations import parse_geometry, parse_types
from .dataset import resolve
from .evaluation import (
    QUERIES,
    VEHICLES,
    box_iou,
    class_compatible,
    interval_errors,
    match,
    predictions_and_integrity,
    spatial_compatibility,
)


def actor_compatible(annotation_type, runtime_type):
    if annotation_type == "person":
        return runtime_type == "person"
    if annotation_type == "vehicle":
        return runtime_type in VEHICLES
    # MEVA's generic "other" does not identify a COCO class. A spatial match
    # is still required; class alone never establishes an object detection.
    return runtime_type not in VEHICLES | {"person"}


def motion_summary(hits, sample_fps):
    """Track geometry diagnostics, never a source of production events."""
    if not hits:
        return {}
    height = statistics.median(max(1, x["box"][3] - x["box"][1]) for x in hits)
    centers = [
        ((x["box"][0] + x["box"][2]) / 2, (x["box"][1] + x["box"][3]) / 2)
        for x in hits
    ]
    speeds, gaps, longest, run, stationary = [], 0, 1, 1, 0.0
    for i in range(1, len(hits)):
        elapsed = hits[i]["t"] - hits[i - 1]["t"]
        if elapsed > 1.5 / sample_fps:
            gaps += 1
            longest = max(longest, run)
            run = 1
            continue
        run += 1
        velocity = hypot(
            centers[i][0] - centers[i - 1][0],
            centers[i][1] - centers[i - 1][1],
        ) / height / max(elapsed, 1e-9)
        speeds.append((hits[i]["t"], velocity))
        if velocity <= 0.12:
            stationary += elapsed
    longest = max(longest, run)
    accelerations = [
        (v - u) / max(t - prior_t, 1e-9)
        for (prior_t, u), (t, v) in zip(speeds, speeds[1:])
    ]
    return {
        "normalized_displacement_box_heights": round(
            hypot(centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]) / height, 4
        ),
        "median_normalized_speed_box_heights_per_second": round(statistics.median(v for _, v in speeds), 4) if speeds else None,
        "max_normalized_acceleration_box_heights_per_second2": round(max(accelerations), 4) if accelerations else None,
        "max_normalized_deceleration_box_heights_per_second2": round(min(accelerations), 4) if accelerations else None,
        "observed_stationary_seconds_at_speed_at_most_0_12": round(stationary, 4),
        "gaps_longer_than_1_5_sample_periods": gaps,
        "longest_contiguous_segment_samples": longest,
    }


def actor_coverage(truth, actor, geometry, observations, entity_types, fps, sample_fps):
    start, end = truth["start_frame"], truth["end_frame"]
    margin = round(2 * fps)
    annotation_type = truth["actor_types"][str(actor)]
    matching = []
    in_event_frames = set()
    for observation in observations:
        if not actor_compatible(annotation_type, entity_types[observation.entity_id]):
            continue
        hits = []
        for sample in observation.boxes:
            frame = sample["frame"]
            if not start - margin <= frame <= end + margin:
                continue
            annotation_box = geometry.get(actor, {}).get(frame)
            if annotation_box is None or box_iou(sample["box"], annotation_box) < 0.1:
                continue
            hits.append(sample)
            if start <= frame <= end:
                in_event_frames.add(frame)
        if hits:
            matching.append(
                {
                    "track_id": observation.track_id,
                    "first_frame": min(x["frame"] for x in hits),
                    "last_frame": max(x["frame"] for x in hits),
                    "matched_samples": len(hits),
                    "matched_event_samples": sum(start <= x["frame"] <= end for x in hits),
                    "mean_track_confidence": observation.confidence,
                    **motion_summary(hits, sample_fps),
                }
            )
    matching.sort(key=lambda row: (-row["matched_event_samples"], -row["matched_samples"]))
    last_sample = math.ceil(end * sample_fps / fps) + 1
    expected = sum(
        start <= round(number * fps / sample_fps) <= end
        for number in range(last_sample + 1)
    )
    return {
        "annotation_actor_id": actor,
        "annotation_actor_type": annotation_type,
        "perception_support": "COCO vehicle/person" if annotation_type in ("person", "vehicle") else "generic MEVA other; COCO category unknown",
        "detected_in_window": bool(matching),
        "detected_during_activity": bool(in_event_frames),
        "matched_event_sample_frames": sorted(in_event_frames),
        "sampled_event_coverage": len(in_event_frames) / expected if expected else None,
        "fragments": len(matching),
        "longest_fragment_samples": max((r["matched_event_samples"] for r in matching), default=0),
        "track_ids": [r["track_id"] for r in matching],
        "tracks": matching,
    }


def analyze(manifest, directory):
    """Write a complete, annotation-level diagnostic alongside benchmark output."""
    from app.db import Entity, Observation, session

    directory = Path(directory)
    benchmark = json.loads((directory / "benchmark-results.json").read_text())
    truths = json.loads((directory / "ground-truth.json").read_text())
    videos = {v["video_id"]: v for v in manifest["videos"]}
    with session() as s:
        observations = list(s.scalars(select(Observation)))
        entity_types = {e.id: e.object_type for e in s.scalars(select(Entity))}
        predictions, integrity = predictions_and_integrity(s, videos)
    if integrity["invalid"]:
        raise ValueError("Evidence chain failed during diagnostic analysis")
    by_video = defaultdict(list)
    for observation in observations:
        by_video[observation.video_id].append(observation)
    geometry = {}
    for video in videos.values():
        types = parse_types(resolve(video["types_path"]))
        geometry[video["video_id"]] = parse_geometry(
            resolve(video["geometry_path"]), video, types
        )
    direct = [g for g in truths if g["status"] == "DIRECT"]
    selected = [
        p for p in predictions
        if p["event_type"] in {"started_moving", "stopped", "picked_up", "placed_object"}
        and (p["event_type"] not in {"started_moving", "stopped"} or set(p["entity_types"]) & VEHICLES)
    ]
    for truth in direct:
        video = videos[truth["video_id"]]
        for prediction in selected:
            if prediction["video_id"] == truth["video_id"] and prediction["event_type"] == truth["event_type"] and class_compatible(prediction, truth):
                prediction.setdefault("spatial_scores", {})[truth["key"]] = spatial_compatibility(
                    prediction, truth, geometry[truth["video_id"]], video["fps"], 2
                )
    pairs = match(selected, direct)
    matched_truth = {j: i for i, j in pairs}
    activities_by_query = {query: activity for query, _, activity in QUERIES if activity}
    query_by_camera_activity = {
        (row["camera_id"], activities_by_query[row["query"]]): row
        for row in benchmark["queries"] if row["query"] in activities_by_query
    }
    details = []
    for j, truth in enumerate(direct):
        video = videos[truth["video_id"]]
        actors = [
            actor_coverage(
                truth, actor, geometry[truth["video_id"]],
                by_video[truth["video_id"]], entity_types, video["fps"],
                benchmark["metadata"]["sample_fps"],
            ) for actor in truth["actor_ids"]
        ]
        all_visible = all(a["detected_during_activity"] for a in actors)
        actor_tracks = [set(a["track_ids"]) for a in actors]
        related = [
            p for p in selected
            if p["video_id"] == truth["video_id"]
            and p["event_type"] == truth["event_type"]
            and all(set(p["track_ids"]) & ids for ids in actor_tracks)
            and p["start_seconds"] <= truth["end_seconds"] + 2
            and p["end_seconds"] >= truth["start_seconds"] - 2
        ] if all_visible else []
        prediction = selected[matched_truth[j]] if j in matched_truth else None
        query = query_by_camera_activity.get((truth["camera_id"], truth["activity_type"]))
        rank = next(
            (i + 1 for i, hit in enumerate(query["retrieved"]) if prediction and hit["event_id"] == prediction["event_id"]),
            None,
        ) if query else None
        source_start = datetime.fromisoformat(video["recording_start"])
        details.append({
            "key": truth["key"],
            "camera_id": truth["camera_id"],
            "video_id": truth["video_id"],
            "activity_type": truth["activity_type"],
            "event_type": truth["event_type"],
            "annotation_id": truth["annotation_id"],
            "source_annotation": truth["source_annotation"],
            "start_frame": truth["start_frame"],
            "end_frame_inclusive": truth["end_frame"],
            "fps": video["fps"],
            "start_seconds": truth["start_seconds"],
            "end_seconds_inclusive": truth["end_seconds"],
            "end_exclusive_seconds": truth["end_exclusive_seconds"],
            "source_clock_start": (source_start + timedelta(seconds=truth["start_seconds"])).isoformat(),
            "source_clock_end": (source_start + timedelta(seconds=truth["end_seconds"])).isoformat(),
            "source_clock_timezone": video["recording_timezone"],
            "actors": actors,
            "all_required_entities_detected_during_activity": all_visible,
            "same_actor_event_in_temporal_window": bool(related),
            "same_actor_event_ids": [p["event_id"] for p in related],
            "matched_event_id": prediction["event_id"] if prediction else None,
            "matched_event_errors": interval_errors(prediction, truth) if prediction else None,
            "matched_event_search_rank_top5": rank,
            "failure_stage": (
                "matched" if prediction else
                "perception" if not all_visible else
                "event_extraction" if not related else
                "temporal_or_spatial_matching"
            ),
        })
    per_type = []
    for activity in sorted({g["activity_type"] for g in direct}):
        rows = [row for row in details if row["activity_type"] == activity]
        event_type = rows[0]["event_type"]
        event_predictions = [p for p in selected if p["event_type"] == event_type]
        matched = [r for r in rows if r["matched_event_id"]]
        errors = [r["matched_event_errors"] for r in matched]
        tracked = [r for r in rows if r["all_required_entities_detected_during_activity"]]
        extracted = [r for r in tracked if r["same_actor_event_in_temporal_window"]]
        per_type.append({
            "activity_type": activity,
            "event_type": event_type,
            "gt": len(rows),
            "predictions": len(event_predictions),
            "matched": len(matched),
            "false_negatives": len(rows) - len(matched),
            "false_positives": len(event_predictions) - len(matched),
            "precision": len(matched) / len(event_predictions) if event_predictions else None,
            "recall": len(matched) / len(rows),
            "mean_temporal_iou": statistics.mean(e["temporal_iou"] for e in errors) if errors else None,
            "mean_start_error_seconds": statistics.mean(e["start_error_seconds"] for e in errors) if errors else None,
            "median_start_error_seconds": statistics.median(e["start_error_seconds"] for e in errors) if errors else None,
            "mean_end_error_seconds": statistics.mean(e["end_error_seconds"] for e in errors) if errors else None,
            "median_end_error_seconds": statistics.median(e["end_error_seconds"] for e in errors) if errors else None,
            "detection_tracking_coverage": len(tracked) / len(rows),
            "event_extraction_success_given_tracking": len(extracted) / len(tracked) if tracked else None,
            "retrieved_top5_of_matched": sum(r["matched_event_search_rank_top5"] is not None for r in matched),
            "failure_stage_counts": {stage: sum(r["failure_stage"] == stage for r in rows) for stage in ("perception", "event_extraction", "temporal_or_spatial_matching", "matched")},
        })
    result = {
        "definition": "Actor coverage requires class-compatible runtime observation boxes with IoU >=0.1 at exact MEVA frame coordinates during the activity. Fragment counts are overlapping runtime tracks, not verified ID switches. Event extraction requires same actor track(s), event type and a +/-2s temporal window. No ground truth enters runtime.",
        "per_type": per_type,
        "annotations": details,
        "unsupported_activities": benchmark["unsupported_activities"],
        "evidence_integrity": integrity,
    }
    (directory / "failure-diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (directory / "failure-diagnostics.csv").open("w", newline="") as handle:
        columns = ["camera_id", "video_id", "annotation_id", "activity_type", "event_type", "start_frame", "end_frame_inclusive", "start_seconds", "end_seconds_inclusive", "all_required_entities_detected_during_activity", "same_actor_event_in_temporal_window", "matched_event_id", "matched_event_search_rank_top5", "failure_stage"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in details)
    return result
