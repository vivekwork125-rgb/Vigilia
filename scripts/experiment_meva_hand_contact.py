#!/usr/bin/env python3
"""Run and evaluate hand-object interaction and contact perception on MEVA.

Evaluation-only experiment investigating whether hand-object interaction / contact perception
provides a reliable intermediate signal for manipulation events (pickup / placement).

STRICT NO-LEAKAGE:
  Operates on runtime video frames and runtime person tracks from vigilia.db.
  MEVA ground-truth annotations are consulted only during evaluation scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from meva.dataset import load_manifest, resolve
from meva.hand_object_contact import (
    ContactObservation,
    ContactStateMachine,
    compute_person_roi,
    evaluate_hand_contact_state,
    extract_pose_keypoints,
)
from meva.selective_objects import person_samples


def load_eval_data(cases_path: Path, manifest_path: Path):
    cases = json.loads(cases_path.read_text())
    manifest = load_manifest(manifest_path)
    video_map = {
        v["video_id"]: {
            "local_path": resolve(v["local_path"]),
            "fps": v["fps"],
            "width": v["width"],
            "height": v["height"],
            "camera_id": v["camera_id"],
        }
        for v in manifest["videos"]
    }
    return cases, video_map


def run_hand_contact_evaluation(
    model_name: str = "yolo11n-pose.pt",
    roi_strategy: str = "expanded",
    roi_factor: float = 1.25,
    persist_threshold: float = 0.5,
    contact_threshold: float = 0.45,
    cases_path: Path = ROOT / "data/meva/object-eval/cases.json",
    manifest_path: Path = ROOT / "datasets/meva/manifest.json",
    db_path: Path = ROOT / "data/meva/improved-2fps/vigilia.db",
    hand_conf_min: float = 0.30,
) -> dict[str, Any]:
    """Execute full evaluation of hand/contact perception across MEVA manipulation cases and negative controls."""
    cases, video_map = load_eval_data(cases_path, manifest_path)
    people = person_samples(db_path)

    model_path = ROOT / "models" / model_name
    if not model_path.exists():
        # Fallback to local root if not yet moved
        model_path = Path(model_name)
    load_start = time.perf_counter()
    model = YOLO(str(model_path))
    model_load_time = time.perf_counter() - load_start

    # Track timing
    crop_latencies: list[float] = []

    # Case-level results
    case_diagnostics: list[dict[str, Any]] = []

    # Aggregates across all 30 cases
    total_cases = len(cases)
    person_covered_cases = 0
    hand_covered_cases = 0
    contact_covered_cases = 0
    persistent_contact_cases = 0
    pickup_candidates_total = 0
    placement_candidates_total = 0
    pickup_matched_total = 0
    placement_matched_total = 0

    # Video cache to minimize reopening
    video_caps: dict[str, cv2.VideoCapture] = {}

    try:
        for idx, c in enumerate(cases):
            vid_id = c["video_id"]
            vid_info = video_map[vid_id]
            fps = vid_info["fps"]
            step = round(fps / 2.0)  # 2 FPS sampling

            if vid_id not in video_caps:
                cap = cv2.VideoCapture(str(vid_info["local_path"]))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open {vid_info['local_path']}")
                video_caps[vid_id] = cap
            else:
                cap = video_caps[vid_id]

            # Activity window + 2.0s context
            gt_start = c["start_seconds"]
            gt_end = c["end_seconds"]
            activity_type = c["activity_type"]

            eval_start_f = max(0, int((gt_start - 2.0) * fps))
            eval_end_f = int((gt_end + 2.0) * fps)

            # Quantize to 2 FPS sampled frames
            f_min = (eval_start_f // step) * step
            f_max = (eval_end_f // step) * step
            sample_frames = [
                f for f in range(f_min, f_max + 1, step)
                if f in people[vid_id]
            ]

            # Case observation tracking
            case_persons_found = 0
            case_hands_found = 0
            case_contacts_found = 0
            case_persistent_found = 0
            case_candidates: list[Any] = []

            # State machines for runtime person tracks in this window
            state_machines: dict[str, ContactStateMachine] = defaultdict(
                lambda: ContactStateMachine(
                    persist_duration=persist_threshold,
                    contact_threshold=contact_threshold,
                )
            )

            # Process each sampled frame
            for f in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue

                t = f / fps
                p_list = people[vid_id].get(f, [])
                if p_list:
                    case_persons_found += len(p_list)

                for p in p_list:
                    pid = p["track_id"]
                    p_box = p["box"]

                    # Compute runtime ROI
                    roi = compute_person_roi(
                        p_box,
                        vid_info["width"],
                        vid_info["height"],
                        strategy=roi_strategy,
                        factor=roi_factor,
                    )
                    rx1, ry1, rx2, ry2 = roi
                    if rx2 - rx1 < 16 or ry2 - ry1 < 16:
                        continue

                    crop = frame[ry1:ry2, rx1:rx2]
                    if crop.size == 0:
                        continue

                    t_crop_0 = time.perf_counter()
                    res = model.predict(crop, imgsz=640, conf=0.20, verbose=False)[0]
                    crop_latencies.append(time.perf_counter() - t_crop_0)

                    if res.keypoints is None or len(res.keypoints) == 0:
                        continue

                    # Unproject keypoints to full frame
                    kps = extract_pose_keypoints(
                        res.keypoints.data.cpu().numpy(),
                        crop_offset=(rx1, ry1),
                        min_conf=hand_conf_min,
                    )

                    # Evaluate contact state for each detected pose in crop
                    for kp_dict in kps:
                        hand_states = evaluate_hand_contact_state(
                            p_box,
                            kp_dict,
                            frame_pixels=frame,
                            hand_conf_threshold=hand_conf_min,
                        )
                        for hs in hand_states:
                            case_hands_found += 1
                            if hs["contact_probability"] >= contact_threshold:
                                case_contacts_found += 1

                            obs = ContactObservation(
                                timestamp=t,
                                frame_index=f,
                                person_track_id=pid,
                                hand_side=hs["hand_side"],
                                hand_box=hs["hand_box"],
                                interaction_region=hs["interaction_region"],
                                contact_probability=hs["contact_probability"],
                                model_confidence=hs["wrist_conf"],
                                source_video=vid_id,
                                source_frame=f,
                                level=hs["level"],
                                attributes={
                                    "reach_score": hs["reach_score"],
                                    "patch_contrast": hs["patch_contrast"],
                                    "rel_y": hs["rel_y"],
                                },
                            )
                            emitted = state_machines[pid].process_observation(obs)
                            case_candidates.extend(emitted)

            # Analyze persistence and temporal matching
            for sm in state_machines.values():
                for runs in sm._extract_contact_runs(
                    [obs for h in sm.track_history.values() for obs in h]
                ):
                    dur = runs[-1].timestamp - runs[0].timestamp
                    if dur >= persist_threshold:
                        case_persistent_found += 1

            # Check matching against ground truth window [gt_start, gt_end] (+/- 2.0s)
            matched_event = False
            match_iou = 0.0
            matched_candidate_id = ""

            for cand in case_candidates:
                if (activity_type == "person_picks_up_object" and cand.event_type == "pickup") or (
                    activity_type == "person_puts_down_object" and cand.event_type == "placement"
                ):
                    # Temporal IoU calculation
                    s_max = max(gt_start, cand.start_seconds)
                    e_min = min(gt_end, cand.end_seconds)
                    inter = max(0.0, e_min - s_max)
                    s_min = min(gt_start, cand.start_seconds)
                    e_max = max(gt_end, cand.end_seconds)
                    union = max(1e-6, e_max - s_min)
                    tiou = inter / union

                    # Check proximity window match (overlap or within 2.0s)
                    if inter > 0 or abs(cand.start_seconds - gt_start) <= 2.0:
                        matched_event = True
                        match_iou = max(match_iou, tiou)
                        matched_candidate_id = cand.candidate_id

            if case_persons_found > 0:
                person_covered_cases += 1
            if case_hands_found > 0:
                hand_covered_cases += 1
            if case_contacts_found > 0:
                contact_covered_cases += 1
            if case_persistent_found > 0:
                persistent_contact_cases += 1

            if activity_type == "person_picks_up_object":
                p_cands = [c for c in case_candidates if c.event_type == "pickup"]
                pickup_candidates_total += len(p_cands)
                if matched_event:
                    pickup_matched_total += 1
            else:
                pl_cands = [c for c in case_candidates if c.event_type == "placement"]
                placement_candidates_total += len(pl_cands)
                if matched_event:
                    placement_matched_total += 1

            # Record case diagnostic row
            diag = {
                "case_index": idx,
                "annotation_key": c["annotation_key"],
                "activity_type": activity_type,
                "camera_id": c["camera_id"],
                "video_id": vid_id,
                "gt_start": round(gt_start, 2),
                "gt_end": round(gt_end, 2),
                "object_size_under_40px": c["approximate_geometry"]["under_40px_width"],
                "persons_detected": case_persons_found,
                "hands_detected": case_hands_found,
                "contacts_detected": case_contacts_found,
                "persistent_contacts": case_persistent_found,
                "candidates_emitted": len(case_candidates),
                "matched": matched_event,
                "temporal_iou": round(match_iou, 4),
                "matched_candidate_id": matched_candidate_id,
                "primary_limitation": _diagnose_limitation(
                    case_persons_found,
                    case_hands_found,
                    case_contacts_found,
                    case_persistent_found,
                    matched_event,
                    c["approximate_geometry"]["under_40px_width"],
                ),
            }
            case_diagnostics.append(diag)

    finally:
        for cap in video_caps.values():
            cap.release()

    # Controlled Negative Sequence Evaluation
    negative_results = _evaluate_negative_controls(
        model=model,
        people=people,
        video_map=video_map,
        roi_strategy=roi_strategy,
        roi_factor=roi_factor,
        persist_threshold=persist_threshold,
        contact_threshold=contact_threshold,
        hand_conf_min=hand_conf_min,
    )

    # Compute overall metrics
    pickup_cases_count = sum(1 for c in cases if c["activity_type"] == "person_picks_up_object")
    placement_cases_count = sum(1 for c in cases if c["activity_type"] == "person_puts_down_object")

    pickup_recall = pickup_matched_total / max(1, pickup_cases_count)
    placement_recall = placement_matched_total / max(1, placement_cases_count)
    pickup_precision = (
        pickup_matched_total
        / max(1, pickup_candidates_total + negative_results["false_pickups"])
    )
    placement_precision = (
        placement_matched_total
        / max(1, placement_candidates_total + negative_results["false_placements"])
    )

    avg_crop_latency = float(np.mean(crop_latencies)) if crop_latencies else 0.0

    return {
        "model_name": model_name,
        "roi_strategy": roi_strategy,
        "roi_factor": roi_factor,
        "persist_threshold_s": persist_threshold,
        "contact_threshold": contact_threshold,
        "model_load_time_s": round(model_load_time, 3),
        "avg_crop_latency_ms": round(avg_crop_latency * 1000.0, 2),
        "effective_fps": round(1.0 / max(1e-4, avg_crop_latency), 1),
        "total_cases": total_cases,
        "person_covered_cases": person_covered_cases,
        "person_coverage_pct": round(person_covered_cases / total_cases * 100.0, 2),
        "hand_covered_cases": hand_covered_cases,
        "hand_coverage_pct": round(hand_covered_cases / total_cases * 100.0, 2),
        "contact_covered_cases": contact_covered_cases,
        "contact_coverage_pct": round(contact_covered_cases / total_cases * 100.0, 2),
        "persistent_contact_cases": persistent_contact_cases,
        "persistent_coverage_pct": round(persistent_contact_cases / total_cases * 100.0, 2),
        "pickup_metrics": {
            "gt_cases": pickup_cases_count,
            "candidates_emitted": pickup_candidates_total,
            "matched": pickup_matched_total,
            "recall": round(pickup_recall * 100.0, 2),
            "precision": round(pickup_precision * 100.0, 2),
        },
        "placement_metrics": {
            "gt_cases": placement_cases_count,
            "candidates_emitted": placement_candidates_total,
            "matched": placement_matched_total,
            "recall": round(placement_recall * 100.0, 2),
            "precision": round(placement_precision * 100.0, 2),
        },
        "negative_controls": negative_results,
        "case_diagnostics": case_diagnostics,
    }


def _diagnose_limitation(
    persons: int,
    hands: int,
    contacts: int,
    persistent: int,
    matched: bool,
    under_40px: bool,
) -> str:
    if persons == 0:
        return "A_PERSON_DETECTION_ABSENT"
    if hands == 0:
        return "B_HAND_VISIBILITY_LIMIT"
    if contacts == 0:
        return "C_CONTACT_VISIBILITY_LIMIT"
    if persistent == 0:
        return "D_TEMPORAL_SPARSITY_2FPS"
    if not matched:
        return "E_FALSE_PROGRESSION_REJECTED"
    if under_40px:
        return "F_SUB_40PX_RESOLUTION_AMBIGUOUS"
    return "G_SUCCESSFUL_CONTACT_DETECTION"


def _evaluate_negative_controls(
    model: YOLO,
    people: dict[str, dict[int, list[dict]]],
    video_map: dict[str, dict],
    roi_strategy: str,
    roi_factor: float,
    persist_threshold: float,
    contact_threshold: float,
    hand_conf_min: float,
) -> dict[str, Any]:
    """Evaluate contact state on non-manipulation intervals (walking, standing, gesturing)."""
    # Define 10 controlled 10-second non-manipulation windows across cameras
    negative_windows = [
        {"video_id": "MEVA-G421-20180315-1555", "start_s": 10.0, "end_s": 20.0, "type": "walking_plaza"},
        {"video_id": "MEVA-G421-20180315-1555", "start_s": 25.0, "end_s": 35.0, "type": "talking_group"},
        {"video_id": "MEVA-G331-20180315-1555", "start_s": 15.0, "end_s": 25.0, "type": "standing_bus_stop"},
        {"video_id": "MEVA-G331-20180315-1555", "start_s": 40.0, "end_s": 50.0, "type": "walking_crosswalk"},
        {"video_id": "MEVA-G336-20180315-1555", "start_s": 10.0, "end_s": 20.0, "type": "corridor_transit"},
        {"video_id": "MEVA-G336-20180315-1555", "start_s": 30.0, "end_s": 40.0, "type": "walking_past_door"},
        {"video_id": "MEVA-G424-20180315-1555", "start_s": 15.0, "end_s": 25.0, "type": "outdoor_path_transit"},
        {"video_id": "MEVA-G328-20180315-1555", "start_s": 10.0, "end_s": 20.0, "type": "school_entrance_walk"},
        {"video_id": "MEVA-G638-20180315-1555", "start_s": 20.0, "end_s": 30.0, "type": "parking_transit"},
        {"video_id": "MEVA-G436-20180315-1555", "start_s": 15.0, "end_s": 25.0, "type": "hospital_curb_walk"},
    ]

    false_hands = 0
    false_contacts = 0
    false_persistent = 0
    false_pickups = 0
    false_placements = 0
    total_negative_frames = 0

    for win in negative_windows:
        vid_id = win["video_id"]
        vid_info = video_map[vid_id]
        fps = vid_info["fps"]
        step = round(fps / 2.0)
        f_start = int(win["start_s"] * fps)
        f_end = int(win["end_s"] * fps)

        frames_to_eval = [
            f for f in range((f_start // step) * step, (f_end // step) * step + 1, step)
            if f in people[vid_id]
        ]
        if not frames_to_eval:
            continue

        cap = cv2.VideoCapture(str(vid_info["local_path"]))
        if not cap.isOpened():
            continue

        state_machines: dict[str, ContactStateMachine] = defaultdict(
            lambda: ContactStateMachine(
                persist_duration=persist_threshold,
                contact_threshold=contact_threshold,
            )
        )

        try:
            for f in frames_to_eval:
                total_negative_frames += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue

                t = f / fps
                for p in people[vid_id].get(f, []):
                    pid = p["track_id"]
                    roi = compute_person_roi(
                        p["box"],
                        vid_info["width"],
                        vid_info["height"],
                        strategy=roi_strategy,
                        factor=roi_factor,
                    )
                    rx1, ry1, rx2, ry2 = roi
                    if rx2 - rx1 < 16 or ry2 - ry1 < 16:
                        continue
                    crop = frame[ry1:ry2, rx1:rx2]
                    if crop.size == 0:
                        continue

                    res = model.predict(crop, imgsz=640, conf=0.20, verbose=False)[0]
                    if res.keypoints is None or len(res.keypoints) == 0:
                        continue

                    kps = extract_pose_keypoints(
                        res.keypoints.data.cpu().numpy(),
                        crop_offset=(rx1, ry1),
                        min_conf=hand_conf_min,
                    )
                    for kp_dict in kps:
                        states = evaluate_hand_contact_state(
                            p["box"],
                            kp_dict,
                            frame_pixels=frame,
                            hand_conf_threshold=hand_conf_min,
                        )
                        for hs in states:
                            false_hands += 1
                            if hs["contact_probability"] >= contact_threshold:
                                false_contacts += 1
                            obs = ContactObservation(
                                timestamp=t,
                                frame_index=f,
                                person_track_id=pid,
                                hand_side=hs["hand_side"],
                                hand_box=hs["hand_box"],
                                interaction_region=hs["interaction_region"],
                                contact_probability=hs["contact_probability"],
                                model_confidence=hs["wrist_conf"],
                                source_video=vid_id,
                                source_frame=f,
                                level=hs["level"],
                            )
                            emitted = state_machines[pid].process_observation(obs)
                            for cand in emitted:
                                if cand.event_type == "pickup":
                                    false_pickups += 1
                                elif cand.event_type == "placement":
                                    false_placements += 1

            for sm in state_machines.values():
                for run in sm._extract_contact_runs(
                    [obs for h in sm.track_history.values() for obs in h]
                ):
                    if run[-1].timestamp - run[0].timestamp >= persist_threshold:
                        false_persistent += 1

        finally:
            cap.release()

    return {
        "negative_intervals_evaluated": len(negative_windows),
        "total_negative_frames": total_negative_frames,
        "false_hand_detections": false_hands,
        "false_contact_detections": false_contacts,
        "false_persistent_contact": false_persistent,
        "false_pickups": false_pickups,
        "false_placements": false_placements,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hand-Object Contact Perception on MEVA")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Pose/hand model file")
    parser.add_argument("--roi-strategy", default="expanded", choices=["full_person", "upper_body", "expanded", "interaction_region"])
    parser.add_argument("--roi-factor", type=float, default=1.25)
    parser.add_argument("--persist", type=float, default=0.5, help="Persistence duration threshold (seconds)")
    parser.add_argument("--contact-thresh", type=float, default=0.45, help="Contact probability threshold")
    parser.add_argument("--output-json", default="data/meva/hand-contact-results.json")
    parser.add_argument("--output-csv", default="data/meva/hand-contact-case-diagnostics.csv")
    args = parser.parse_args()

    print("=== VIGILIA Hand-Object Interaction Evaluation ===")
    print(f"Model: {args.model}")
    print(f"ROI Strategy: {args.roi_strategy} (factor {args.roi_factor})")
    print(f"Persistence Threshold: {args.persist}s")
    print(f"Contact Threshold: {args.contact_thresh}")

    results = run_hand_contact_evaluation(
        model_name=args.model,
        roi_strategy=args.roi_strategy,
        roi_factor=args.roi_factor,
        persist_threshold=args.persist,
        contact_threshold=args.contact_thresh,
    )

    # Save JSON summary
    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote summary to {out_json}")

    # Save CSV diagnostics
    out_csv = ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if results["case_diagnostics"]:
        keys = list(results["case_diagnostics"][0].keys())
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results["case_diagnostics"])
        print(f"Wrote case diagnostics to {out_csv}")

    # Print summary
    print("\n--- Results Summary ---")
    print(f"Person Coverage: {results['person_covered_cases']}/{results['total_cases']} ({results['person_coverage_pct']}%)")
    print(f"Hand Coverage:   {results['hand_covered_cases']}/{results['total_cases']} ({results['hand_coverage_pct']}%)")
    print(f"Contact Coverage:{results['contact_covered_cases']}/{results['total_cases']} ({results['contact_coverage_pct']}%)")
    print(f"Persistent Cont: {results['persistent_contact_cases']}/{results['total_cases']} ({results['persistent_coverage_pct']}%)")
    print(f"Pickup Matches:  {results['pickup_metrics']['matched']}/{results['pickup_metrics']['gt_cases']} (Recall: {results['pickup_metrics']['recall']}%, Prec: {results['pickup_metrics']['precision']}%)")
    print(f"Placement Match: {results['placement_metrics']['matched']}/{results['placement_metrics']['gt_cases']} (Recall: {results['placement_metrics']['recall']}%, Prec: {results['placement_metrics']['precision']}%)")
    print(f"False Pickups (Negatives): {results['negative_controls']['false_pickups']}")
    print(f"False Placements (Negatives): {results['negative_controls']['false_placements']}")
    print(f"Avg Crop Latency: {results['avg_crop_latency_ms']} ms (~{results['effective_fps']} FPS)")


if __name__ == "__main__":
    main()
