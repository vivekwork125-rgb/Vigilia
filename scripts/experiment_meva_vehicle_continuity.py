#!/usr/bin/env python3
"""Evaluate same-camera vehicle track continuity across gap thresholds.

Measures candidate merges, ground truth correctness, downstream event hypotheses,
and independent MEVA matches on stored production tracks.
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.temporal_events import vehicle_transitions
from meva.annotations import parse_activities, parse_geometry, parse_types
from meva.dataset import load_manifest, resolve
from meva.evaluation import box_iou, detection_metrics, match, spatial_compatibility
from meva.mapping import mapping
from meva.vehicle_continuity import (
    build_continuous_tracks,
    find_safe_merges,
)


def load_stored_tracks(db_path: Path):
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT o.track_id, o.boxes, e.object_type, o.video_id, v.camera_id, o.id "
            "FROM observations o "
            "JOIN entities e ON o.entity_id=e.id "
            "JOIN videos v ON o.video_id=v.id "
            "WHERE e.object_type IN ('car', 'truck', 'bus', 'motorcycle')"
        ).fetchall()
    finally:
        db.close()
    return [
        {
            "track_id": r[0],
            "boxes": json.loads(r[1]),
            "object_type": r[2],
            "video_id": r[3],
            "camera_id": r[4],
            "observation_id": r[5],
        }
        for r in rows
    ]


def map_tracks_to_actors(tracks_by_video, geometry_by_video):
    """Map each stored track to best overlapping MEVA ground truth actor."""
    track_actor = {}
    for vid, vtracks in tracks_by_video.items():
        geom = geometry_by_video.get(vid, {})
        for trk in vtracks:
            best_actor, best_iou = None, 0.0
            for actor_id, frames in geom.items():
                ious = [
                    box_iou(s["box"], frames[s["frame"]])
                    for s in trk["boxes"]
                    if s["frame"] in frames
                ]
                if ious and max(ious) > best_iou:
                    best_iou = max(ious)
                    best_actor = actor_id
            if best_iou >= 0.15:
                track_actor[trk["track_id"]] = (best_actor, best_iou)
    return track_actor


def evaluate_gap(
    tracks: list,
    max_gap: float,
    videos: dict,
    truth_by_video: dict,
    geometry_by_video: dict,
    track_actor_map: dict,
    min_score: float = 0.65,
):
    truths = [t for v_truths in truth_by_video.values() for t in v_truths]

    # Find safe candidate merges
    merges = find_safe_merges(tracks, max_gap=max_gap, min_score=min_score)

    # Check merge correctness against ground truth actors
    correct_merges = 0
    incorrect_merges = 0
    unannotated_merges = 0
    merge_details = []

    for m in merges:
        act_a = track_actor_map.get(m.track_a_id)
        act_b = track_actor_map.get(m.track_b_id)
        if act_a and act_b:
            if act_a[0] == act_b[0]:
                correct_merges += 1
                status = "CORRECT"
            else:
                incorrect_merges += 1
                status = "INCORRECT"
        else:
            unannotated_merges += 1
            status = "UNANNOTATED"

        merge_details.append({
            "track_a": m.track_a_id,
            "track_b": m.track_b_id,
            "gap_seconds": m.gap_seconds,
            "score": m.score,
            "status": status,
            "actor_a": act_a[0] if act_a else None,
            "actor_b": act_b[0] if act_b else None,
        })

    # Build continuous tracks
    continuous_tracks = build_continuous_tracks(tracks, merges)

    # Run unchanged vehicle transitions
    predictions = []
    starts_count = 0
    stops_count = 0

    for trk in continuous_tracks:
        vid = trk["video_id"]
        video = videos[vid]
        candidates = vehicle_transitions(trk["track_id"], trk["boxes"])
        for cand in candidates:
            if cand.kind == "started_moving":
                starts_count += 1
            elif cand.kind == "stopped":
                stops_count += 1

            pred = {
                "event_type": cand.kind,
                "video_id": vid,
                "camera_id": video["camera_id"],
                "start_seconds": cand.start,
                "end_seconds": cand.end,
                "entity_types": [trk["object_type"]],
                "supporting_tracks": [trk],
                "spatial_scores": {},
            }
            for truth in truth_by_video[vid]:
                if truth["event_type"] == cand.kind:
                    pred["spatial_scores"][truth["key"]] = spatial_compatibility(
                        pred, truth, geometry_by_video[vid], video["fps"], 2.0
                    )
            predictions.append(pred)

    pairs = match(predictions, truths)
    metrics = detection_metrics(predictions, truths, pairs)

    by_type = {}
    for typ in ("started_moving", "stopped"):
        p_sub = [p for p in predictions if p["event_type"] == typ]
        t_sub = [t for t in truths if t["event_type"] == typ]
        sub_pairs = match(p_sub, t_sub)
        by_type[typ] = detection_metrics(p_sub, t_sub, sub_pairs)

    return {
        "max_gap": max_gap,
        "candidate_merges": len(merges),
        "correct_merges": correct_merges,
        "incorrect_merges": incorrect_merges,
        "unannotated_merges": unannotated_merges,
        "merged_tracks_count": len(continuous_tracks),
        "start_hypotheses": starts_count,
        "stop_hypotheses": stops_count,
        "total_predictions": len(predictions),
        "total_matches": metrics["matched_events"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mean_temporal_iou": metrics["mean_temporal_iou_on_matched"],
        "mean_start_error": metrics["mean_start_error_seconds_on_matched"],
        "mean_end_error": metrics["mean_end_error_seconds_on_matched"],
        "start_matches": by_type["started_moving"]["matched_events"],
        "stop_matches": by_type["stopped"]["matched_events"],
        "by_type": by_type,
        "merges": merge_details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/improved-2fps")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/meva/vehicle-continuity-results.json")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    videos = {v["video_id"]: v for v in manifest["videos"]}
    truth_by_video = defaultdict(list)
    geometry_by_video = {}

    for video in videos.values():
        types = parse_types(resolve(video["types_path"]))
        geometry_by_video[video["video_id"]] = parse_geometry(
            resolve(video["geometry_path"]), video, types
        )
        for row in parse_activities(resolve(video["annotation_path"]), video, types):
            row.update(mapping(row["activity_type"]))
            if row["activity_type"] in {"vehicle_starts", "vehicle_stops"} and row["status"] == "DIRECT":
                row["key"] = f"{row['video_id']}:{row['annotation_id']}:{row['span_index']}"
                truth_by_video[video["video_id"]].append(row)

    tracks = load_stored_tracks(args.run_dir / "vigilia.db")
    tracks_by_video = defaultdict(list)
    for t in tracks:
        tracks_by_video[t["video_id"]].append(t)

    track_actor_map = map_tracks_to_actors(tracks_by_video, geometry_by_video)

    # First evaluate baseline (max_gap = 0.0, no merges)
    baseline_result = evaluate_gap(
        tracks, 0.0, videos, truth_by_video, geometry_by_video, track_actor_map
    )

    gap_windows = [0.5, 1.0, 1.5, 2.0, 3.0]
    gap_results = {}
    for gap in gap_windows:
        gap_results[f"{gap}s"] = evaluate_gap(
            tracks, gap, videos, truth_by_video, geometry_by_video, track_actor_map
        )

    output = {
        "baseline": baseline_result,
        "gap_experiments": gap_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Results written to {args.output}")

    print("\nSUMMARY TABLE:")
    print(f"{'Gap':<8} | {'Merges':<8} | {'Correct':<8} | {'Incorr':<8} | {'Preds':<6} | {'Match':<6} | {'Starts':<8} | {'Stops':<8} | {'Prec %':<8} | {'Rec %':<8} | {'mIoU':<8}")
    print("-" * 95)
    b = baseline_result
    print(f"{'Baseline':<8} | {b['candidate_merges']:<8} | {b['correct_merges']:<8} | {b['incorrect_merges']:<8} | {b['total_predictions']:<6} | {b['total_matches']:<6} | {b['start_matches']}/32{'':<4} | {b['stop_matches']}/27{'':<4} | {b['precision']*100:.2f}%{'':<2} | {b['recall']*100:.2f}%{'':<2} | {b['mean_temporal_iou']:.4f}")

    for gap_str, gr in gap_results.items():
        prec_str = f"{gr['precision']*100:.2f}%" if gr['precision'] else "N/A"
        rec_str = f"{gr['recall']*100:.2f}%" if gr['recall'] else "N/A"
        miou_str = f"{gr['mean_temporal_iou']:.4f}" if gr['mean_temporal_iou'] else "N/A"
        print(f"{gap_str:<8} | {gr['candidate_merges']:<8} | {gr['correct_merges']:<8} | {gr['incorrect_merges']:<8} | {gr['total_predictions']:<6} | {gr['total_matches']:<6} | {gr['start_matches']}/32{'':<4} | {gr['stop_matches']}/27{'':<4} | {prec_str:<8} | {rec_str:<8} | {miou_str}")


if __name__ == "__main__":
    main()
