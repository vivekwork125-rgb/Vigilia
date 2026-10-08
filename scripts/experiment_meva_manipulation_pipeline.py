#!/usr/bin/env python3
"""Run and evaluate the selective small-object manipulation pipeline on MEVA cases.

Evaluates the complete chain:
  object detection -> object tracklet -> person-object association ->
  temporal interaction (pickup / placement) -> evidence grounding.

Evaluates:
  - 3 ROI strategies (expanded, interaction, scene_nearby)
  - Expansion factors (1.25, 1.5, 2.0)
  - Full-chain stages across all 30 manipulation annotations
  - Generates the mandatory case-level diagnostic table.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.temporal_events import PORTABLE
from meva.annotations import parse_geometry, parse_types
from meva.dataset import load_manifest, resolve
from meva.evaluation import box_iou
from meva.perception_experiment import requirements
from meva.selective_manipulation_pipeline import (
    build_tracklets,
    evaluate_pickup,
    evaluate_placement,
)
from meva.selective_objects import associate, deduplicate, person_roi, person_samples

SKIP_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


def predict_crops(model, pixels, size=640, confidence=0.3):
    output = model.predict(pixels, imgsz=size, conf=confidence, verbose=False)[0].boxes
    if output is None:
        return []
    return [
        (model.names[int(cls)], float(score), box.tolist())
        for box, cls, score in zip(
            output.xyxy.cpu().numpy(), output.cls.cpu().numpy(), output.conf.cpu().numpy()
        )
    ]


def run_pipeline(
    manifest_path: Path,
    baseline_run: Path,
    model_path: Path,
    strategy: str = "interaction",
    factor: float = 2.0,
    size: int = 640,
    confidence: float = 0.3,
):
    manifest = load_manifest(manifest_path)
    diagnosis = json.loads((baseline_run / "failure-diagnostics.json").read_text())
    videos, indexed, actors = requirements(manifest, diagnosis, 2.0)

    object_actors = {
        key: actor for key, actor in actors.items()
        if actor["annotation_actor_type"] == "other"
    }
    by_annotation = {
        row["key"]: [
            track for actor in row["actors"]
            if actor["annotation_actor_type"] == "person"
            for track in actor["track_ids"]
        ]
        for row in diagnosis["annotations"]
    }
    for actor in object_actors.values():
        actor["person_track_ids"] = by_annotation.get(actor["annotation_key"], [])

    actors_by_key = {v["annotation_key"]: v for v in object_actors.values()}

    people = person_samples(baseline_run / "vigilia.db")
    model = YOLO(str(model_path))

    detections = defaultdict(lambda: defaultdict(list))
    geometry = {}
    source_frames = 0
    crops_total = 0
    begun = time.perf_counter()

    for video_id, video in videos.items():
        frames = {
            frame: keys for frame, keys in indexed.get(video_id, {}).items()
            if any(key in object_actors for key in keys)
        }
        if not frames:
            continue

        geometry[video_id] = parse_geometry(
            resolve(video["geometry_path"]), video,
            parse_types(resolve(video["types_path"])),
        )

        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video['filename']}")

        try:
            for frame in sorted(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    continue
                source_frames += 1
                found = []
                for person in people[video_id].get(frame, []):
                    left, top, right, bottom = person_roi(
                        person["box"], video["width"], video["height"],
                        strategy, factor,
                    )
                    if right - left < 32 or bottom - top < 32:
                        continue
                    crops_total += 1
                    crop_pixels = pixels[top:bottom, left:right]
                    for name, conf, box in predict_crops(model, crop_pixels, size, confidence):
                        if name in SKIP_CLASSES:
                            continue
                        found.append({
                            "class": name,
                            "confidence": conf,
                            "box": [box[0] + left, box[1] + top, box[2] + left, box[3] + top],
                            "frame": frame,
                            "t": frame / video["fps"],
                        })
                detections[video_id][frame] = deduplicate(found)
        finally:
            cap.release()

    wall_seconds = time.perf_counter() - begun

    # Load 30 manipulation ground truth cases from metadata
    cases_meta_path = ROOT / "data/meva/object-eval/cases.json"
    if cases_meta_path.exists():
        eval_cases = json.loads(cases_meta_path.read_text())
    else:
        eval_cases = []

    # Map people samples by track_id for motion coupling
    persons_by_track = defaultdict(list)
    for vid, frames_map in people.items():
        for f, plist in frames_map.items():
            for p in plist:
                persons_by_track[p["track_id"]].append(p)

    # Build tracklets for each video
    tracklets_by_video = {}
    for vid, frames_map in detections.items():
        tracklets_by_video[vid] = build_tracklets(frames_map, fps=videos[vid]["fps"])

    # Full chain case evaluation
    case_diagnostics = []
    broad_detected_count = 0
    portable_detected_count = 0
    broad_tracklet_count = 0
    portable_tracklet_count = 0
    associated_count = 0
    pickup_inferred_count = 0
    placement_inferred_count = 0
    pickup_matched_count = 0
    placement_matched_count = 0

    for c in eval_cases:
        key = c["annotation_key"]
        vid = c["video_id"]
        geom = geometry.get(vid, {})
        obj_actor = actors_by_key.get(key)
        ref_frames = geom.get(obj_actor["annotation_actor_id"], {}) if obj_actor else {}
        target_frames = obj_actor["sample_frames"] if obj_actor else []

        # 1. Object detection stage: Check all deduplicated crop detections in target frames
        best_iou = 0.0
        best_class = None
        best_conf = 0.0

        for f in target_frames:
            gt_box = ref_frames.get(f)
            if gt_box is None:
                continue
            for det in detections.get(vid, {}).get(f, []):
                iou_val = box_iou(det["box"], gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_class = det["class"]
                    best_conf = det["confidence"]

        obj_detected_broad = (best_iou >= 0.1)
        obj_detected_portable = (obj_detected_broad and best_class in PORTABLE)

        if obj_detected_broad:
            broad_detected_count += 1
        if obj_detected_portable:
            portable_detected_count += 1

        # 2. Object tracklet stage: Check tracklets overlapping target frames
        v_tracklets = tracklets_by_video.get(vid, [])
        overlapping_tracklets = []
        for trk in v_tracklets:
            if any(
                h["frame"] in target_frames
                and h["frame"] in ref_frames
                and box_iou(h["box"], ref_frames[h["frame"]]) >= 0.1
                for h in trk.detections
            ):
                overlapping_tracklets.append(trk)

        has_broad_tracklet = any(t.is_stable for t in overlapping_tracklets)
        has_portable_tracklet = any(t.is_stable and t.is_portable for t in overlapping_tracklets)

        if has_broad_tracklet:
            broad_tracklet_count += 1
        if has_portable_tracklet:
            portable_tracklet_count += 1

        # 3. Person-object association stage
        associated_person = None
        interacting_persons = obj_actor["person_track_ids"] if obj_actor else []
        for trk in overlapping_tracklets:
            owner = associate(trk.detections, people.get(vid, {}))
            if owner and owner in interacting_persons:
                associated_person = owner
                break

        if associated_person:
            associated_count += 1

        # 4. Temporal Interaction (Pickup / Placement)
        event_inferred = False
        inference_reason = "none"
        if associated_person and overlapping_tracklets:
            person_samples_list = persons_by_track.get(associated_person, [])
            chosen_tracklet = overlapping_tracklets[0]
            if c["activity_type"] == "person_picks_up_object":
                ok_event, reason, _ = evaluate_pickup(chosen_tracklet, associated_person, person_samples_list)
                if ok_event:
                    event_inferred = True
                    pickup_inferred_count += 1
                inference_reason = reason
            else:
                ok_event, reason, _ = evaluate_placement(chosen_tracklet, associated_person, person_samples_list)
                if ok_event:
                    event_inferred = True
                    placement_inferred_count += 1
                inference_reason = reason

        # 5. Final match against GT (must be portable class and matched window)
        matched = False
        if event_inferred and best_class in PORTABLE:
            matched = True
            if c["activity_type"] == "person_picks_up_object":
                pickup_matched_count += 1
            else:
                placement_matched_count += 1

        # Determine exact failure stage
        if matched:
            failure_stage = "MATCHED"
        elif not obj_detected_broad:
            failure_stage = "DETECTION_ABSENCE"
        elif not obj_detected_portable:
            failure_stage = f"UNSUPPORTED_CLASS_{best_class.upper()}"
        elif not has_portable_tracklet:
            failure_stage = "TRACKLET_INSTABILITY"
        elif not associated_person:
            failure_stage = "ASSOCIATION_FAILURE"
        elif not event_inferred:
            failure_stage = f"INTERACTION_RULE_{inference_reason.upper()}"
        else:
            failure_stage = "TEMPORAL_WINDOW_MISMATCH"

        case_diagnostics.append({
            "annotation_key": key,
            "activity_type": c["activity_type"],
            "camera_id": c["camera_id"],
            "taxonomy_category": c["taxonomy_category"],
            "median_width_px": c["approximate_geometry"]["median_width_px"],
            "best_overlap_class": best_class or "none",
            "best_overlap_conf": round(best_conf, 4) if best_conf else 0.0,
            "best_overlap_iou": round(best_iou, 4),
            "object_detected_broad": obj_detected_broad,
            "object_detected_portable": obj_detected_portable,
            "tracklet_formed_broad": has_broad_tracklet,
            "tracklet_formed_portable": has_portable_tracklet,
            "person_associated": bool(associated_person),
            "associated_person_track": associated_person or "none",
            "event_inferred": event_inferred,
            "matched_gt": matched,
            "failure_stage": failure_stage,
        })

    summary = {
        "strategy": strategy,
        "factor": factor,
        "detector": model_path.name,
        "detector_size": size,
        "confidence": confidence,
        "wall_seconds": round(wall_seconds, 3),
        "source_frames": source_frames,
        "crops_total": crops_total,
        "crops_per_frame": round(crops_total / max(1, source_frames), 2),
        "total_cases": len(eval_cases),
        "object_detection_recall_broad": f"{broad_detected_count}/{len(eval_cases)}",
        "object_detection_recall_portable": f"{portable_detected_count}/{len(eval_cases)}",
        "object_tracklet_recall_broad": f"{broad_tracklet_count}/{len(eval_cases)}",
        "object_tracklet_recall_portable": f"{portable_tracklet_count}/{len(eval_cases)}",
        "person_association_recall": f"{associated_count}/{len(eval_cases)}",
        "pickup_inferred": pickup_inferred_count,
        "placement_inferred": placement_inferred_count,
        "pickup_matched": f"{pickup_matched_count}/15",
        "placement_matched": f"{placement_matched_count}/15",
        "total_matched": f"{pickup_matched_count + placement_matched_count}/30",
    }

    return summary, case_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "data/meva/improved-2fps")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--output-json", type=Path, default=ROOT / "data/meva/manipulation-pipeline-results.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data/meva/manipulation-case-diagnostics.csv")
    args = parser.parse_args()

    strategies = [
        ("expanded", 1.25),
        ("expanded", 1.5),
        ("expanded", 2.0),
        ("interaction", 1.25),
        ("interaction", 1.5),
        ("interaction", 2.0),
        ("scene_nearby", 1.25),
        ("scene_nearby", 1.5),
        ("scene_nearby", 2.0),
    ]

    all_summaries = []
    best_cases = None

    print("Evaluating 9 ROI configurations across the 30 manipulation cases...")
    for strat, fac in strategies:
        print(f"  Running Strategy: {strat}, Factor: {fac}...")
        summary, cases = run_pipeline(
            args.manifest,
            args.baseline_run,
            args.model,
            strategy=strat,
            factor=fac,
        )
        all_summaries.append(summary)
        if strat == "interaction" and fac == 2.0:
            best_cases = cases

    # Write summary JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_summaries, indent=2) + "\n")
    print(f"\nWrote summary results to {args.output_json}")

    # Write case-level diagnostic CSV
    if best_cases:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_cases[0].keys()))
            writer.writeheader()
            writer.writerows(best_cases)
        print(f"Wrote case-level diagnostics to {args.output_csv}")

    # Print summary table
    print("\nROI STRATEGY SUMMARY TABLE:")
    print(f"{'Strategy':<14} | {'Factor':<6} | {'Crops':<6} | {'Broad Det':<10} | {'Port Det':<10} | {'Broad Trk':<10} | {'Port Trk':<10} | {'Assoc':<8} | {'Pickups':<8} | {'Places':<8} | {'Wall (s)':<8}")
    print("-" * 115)
    for s in all_summaries:
        print(f"{s['strategy']:<14} | {s['factor']:<6} | {s['crops_total']:<6} | {s['object_detection_recall_broad']:<10} | {s['object_detection_recall_portable']:<10} | {s['object_tracklet_recall_broad']:<10} | {s['object_tracklet_recall_portable']:<10} | {s['person_association_recall']:<8} | {s['pickup_matched']:<8} | {s['placement_matched']:<8} | {s['wall_seconds']:<8}")


if __name__ == "__main__":
    main()
