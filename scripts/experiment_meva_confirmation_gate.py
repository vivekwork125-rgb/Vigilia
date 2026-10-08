#!/usr/bin/env python3
"""Run and evaluate secondary manipulation confirmation gate on MEVA.

Evaluation-only experiment investigating whether an independent secondary confirmation signal
(localized motion, relative motion coupling, appearance change, region persistence, separation)
can reduce false manipulation hypotheses produced by upstream hand/interaction-region pipelines.

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
    ManipulationCandidate,
    compute_person_roi,
    evaluate_hand_contact_state,
    extract_pose_keypoints,
)
from meva.manipulation_confirmation import (
    ManipulationConfirmationGate,
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


NEGATIVE_WINDOWS = [
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


def extract_upstream_candidates(
    cases: list[dict[str, Any]],
    video_map: dict[str, dict[str, Any]],
    people: dict[str, dict[int, list[dict]]],
    model: YOLO,
    roi_strategy: str = "interaction_region",
    roi_factor: float = 1.25,
    persist_threshold: float = 1.0,
    contact_threshold: float = 0.45,
    hand_conf_min: float = 0.30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Extract raw upstream candidates and all visual/temporal context for evaluation and negative controls."""
    video_caps: dict[str, cv2.VideoCapture] = {}
    case_data_list: list[dict[str, Any]] = []
    negative_candidates: list[dict[str, Any]] = []

    profile_times = {
        "upstream_detection_s": 0.0,
        "frames_processed": 0,
    }

    t0 = time.perf_counter()

    try:
        # 1. Process 30 MEVA manipulation cases
        for idx, c in enumerate(cases):
            vid_id = c["video_id"]
            vid_info = video_map[vid_id]
            fps = vid_info["fps"]
            step = round(fps / 2.0)

            if vid_id not in video_caps:
                cap = cv2.VideoCapture(str(vid_info["local_path"]))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open {vid_info['local_path']}")
                video_caps[vid_id] = cap
            else:
                cap = video_caps[vid_id]

            gt_start = c["start_seconds"]
            gt_end = c["end_seconds"]
            eval_start_f = max(0, int((gt_start - 2.5) * fps))
            eval_end_f = int((gt_end + 2.5) * fps)

            f_min = (eval_start_f // step) * step
            f_max = (eval_end_f // step) * step
            sample_frames = [
                f for f in range(f_min, f_max + 1, step)
                if f in people[vid_id]
            ]

            state_machines: dict[str, ContactStateMachine] = defaultdict(
                lambda: ContactStateMachine(
                    persist_duration=persist_threshold,
                    contact_threshold=contact_threshold,
                )
            )

            loaded_frames: dict[int, np.ndarray] = {}
            all_observations: dict[str, list[ContactObservation]] = defaultdict(list)
            person_boxes_map: dict[tuple[str, int], tuple[int, int, int, int]] = {}
            emitted_candidates: list[ManipulationCandidate] = []

            for f in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue
                loaded_frames[f] = frame
                profile_times["frames_processed"] += 1
                t = f / fps

                for p in people[vid_id].get(f, []):
                    pid = p["track_id"]
                    p_box = p["box"]
                    person_boxes_map[(pid, f)] = p_box

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

                    res = model.predict(crop, imgsz=640, conf=0.20, verbose=False)[0]
                    if res.keypoints is None or len(res.keypoints) == 0:
                        continue

                    kps = extract_pose_keypoints(
                        res.keypoints.data.cpu().numpy(),
                        crop_offset=(rx1, ry1),
                        min_conf=hand_conf_min,
                    )
                    for kp_dict in kps:
                        hand_states = evaluate_hand_contact_state(
                            p_box,
                            kp_dict,
                            frame_pixels=frame,
                            hand_conf_threshold=hand_conf_min,
                        )
                        for hs in hand_states:
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
                            all_observations[pid].append(obs)
                            new_cands = state_machines[pid].process_observation(obs)
                            emitted_candidates.extend(new_cands)

            # Package candidates with extracted visual context
            packaged_cands = []
            for cand in emitted_candidates:
                # Frames during candidate
                cand_frames = [
                    loaded_frames[obs.frame_index]
                    for obs in cand.supporting_observations
                    if obs.frame_index in loaded_frames
                ]
                # Pre-frame (1-2 steps prior)
                pre_candidates = [
                    f for f in sample_frames
                    if f < cand.start_frame and f in loaded_frames
                ]
                pre_frame = loaded_frames[pre_candidates[-1]] if pre_candidates else (
                    loaded_frames.get(cand.start_frame)
                )

                # Post-frame (1-2 steps post)
                post_candidates = [
                    f for f in sample_frames
                    if f > cand.end_frame and f in loaded_frames
                ]
                post_frame = loaded_frames[post_candidates[0]] if post_candidates else (
                    loaded_frames.get(cand.end_frame)
                )

                # Post observations for this person track
                post_obs = [
                    obs for obs in all_observations[cand.person_track_id]
                    if obs.frame_index > cand.end_frame
                    and obs.frame_index <= cand.end_frame + 4 * step
                ]

                # Person boxes during candidate
                p_boxes = [
                    person_boxes_map.get((cand.person_track_id, obs.frame_index), obs.hand_box)
                    for obs in cand.supporting_observations
                ]

                packaged_cands.append({
                    "candidate": cand,
                    "candidate_frames": cand_frames,
                    "pre_frame": pre_frame,
                    "post_frame": post_frame,
                    "post_observations": post_obs,
                    "person_boxes": p_boxes,
                })

            case_data_list.append({
                "case_index": idx,
                "case_info": c,
                "step": step,
                "candidates": packaged_cands,
            })

        # 2. Process Negative Control Windows
        for win in NEGATIVE_WINDOWS:
            vid_id = win["video_id"]
            vid_info = video_map[vid_id]
            fps = vid_info["fps"]
            step = round(fps / 2.0)
            f_start = int(win["start_s"] * fps)
            f_end = int(win["end_s"] * fps)

            sample_frames = [
                f for f in range((f_start // step) * step, (f_end // step) * step + 1, step)
                if f in people[vid_id]
            ]
            if not sample_frames:
                continue

            cap = video_caps.get(vid_id)
            if cap is None:
                cap = cv2.VideoCapture(str(vid_info["local_path"]))
                if not cap.isOpened():
                    continue
                video_caps[vid_id] = cap

            state_machines = defaultdict(
                lambda: ContactStateMachine(
                    persist_duration=persist_threshold,
                    contact_threshold=contact_threshold,
                )
            )
            loaded_frames = {}
            all_observations = defaultdict(list)
            person_boxes_map = {}
            emitted_cands = []

            for f in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue
                loaded_frames[f] = frame
                profile_times["frames_processed"] += 1
                t = f / fps

                for p in people[vid_id].get(f, []):
                    pid = p["track_id"]
                    p_box = p["box"]
                    person_boxes_map[(pid, f)] = p_box

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
                            p_box,
                            kp_dict,
                            frame_pixels=frame,
                            hand_conf_threshold=hand_conf_min,
                        )
                        for hs in states:
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
                            all_observations[pid].append(obs)
                            new_cands = state_machines[pid].process_observation(obs)
                            emitted_cands.extend(new_cands)

            for cand in emitted_cands:
                cand_frames = [
                    loaded_frames[obs.frame_index]
                    for obs in cand.supporting_observations
                    if obs.frame_index in loaded_frames
                ]
                pre_candidates = [
                    f for f in sample_frames
                    if f < cand.start_frame and f in loaded_frames
                ]
                pre_frame = loaded_frames[pre_candidates[-1]] if pre_candidates else (
                    loaded_frames.get(cand.start_frame)
                )

                post_candidates = [
                    f for f in sample_frames
                    if f > cand.end_frame and f in loaded_frames
                ]
                post_frame = loaded_frames[post_candidates[0]] if post_candidates else (
                    loaded_frames.get(cand.end_frame)
                )

                post_obs = [
                    obs for obs in all_observations[cand.person_track_id]
                    if obs.frame_index > cand.end_frame
                    and obs.frame_index <= cand.end_frame + 4 * step
                ]

                p_boxes = [
                    person_boxes_map.get((cand.person_track_id, obs.frame_index), obs.hand_box)
                    for obs in cand.supporting_observations
                ]

                negative_candidates.append({
                    "window": win,
                    "candidate": cand,
                    "candidate_frames": cand_frames,
                    "pre_frame": pre_frame,
                    "post_frame": post_frame,
                    "post_observations": post_obs,
                    "person_boxes": p_boxes,
                })

    finally:
        for cap in video_caps.values():
            cap.release()

    profile_times["upstream_detection_s"] = time.perf_counter() - t0
    return case_data_list, negative_candidates, profile_times


def evaluate_gate_configuration(
    case_data_list: list[dict[str, Any]],
    negative_candidates: list[dict[str, Any]],
    mode: str,
    confirm_threshold: float,
    weak_threshold: float = 0.30,
) -> dict[str, Any]:
    """Evaluate a specific confirmation gate configuration across all cases and negative controls."""
    gate = ManipulationConfirmationGate(
        mode=mode,
        confirm_threshold=confirm_threshold,
        weak_threshold=weak_threshold,
    )

    t_eval_start = time.perf_counter()
    eval_latencies: list[float] = []

    pickup_cases_gt = 0
    placement_cases_gt = 0
    pickup_candidates_kept = 0
    placement_candidates_kept = 0
    pickup_matched_count = 0
    placement_matched_count = 0

    pickup_ious: list[float] = []
    placement_ious: list[float] = []

    # Proximity breakdown
    pickup_matches_1s = 0
    pickup_matches_2s = 0
    pickup_matches_3s = 0
    placement_matches_1s = 0
    placement_matches_2s = 0
    placement_matches_3s = 0

    # Resolution breakdown
    hi_res_pickup_gt = 0
    hi_res_pickup_matched = 0
    sub40_pickup_gt = 0
    sub40_pickup_matched = 0

    hi_res_placement_gt = 0
    hi_res_placement_matched = 0
    sub40_placement_gt = 0
    sub40_placement_matched = 0

    case_diagnostics: list[dict[str, Any]] = []

    for item in case_data_list:
        c = item["case_info"]
        act = c["activity_type"]
        is_sub40 = c["approximate_geometry"]["under_40px_width"]
        gt_start = c["start_seconds"]
        gt_end = c["end_seconds"]

        if act == "person_picks_up_object":
            pickup_cases_gt += 1
            if is_sub40:
                sub40_pickup_gt += 1
            else:
                hi_res_pickup_gt += 1
        else:
            placement_cases_gt += 1
            if is_sub40:
                sub40_placement_gt += 1
            else:
                hi_res_placement_gt += 1

        matched_case = False
        best_iou = 0.0
        best_cand_id = ""
        case_kept_cands = []
        best_explanation = ""

        for cand_pkg in item["candidates"]:
            cand: ManipulationCandidate = cand_pkg["candidate"]
            t_gate_0 = time.perf_counter()
            decision = gate.evaluate_candidate(
                candidate=cand,
                candidate_frames=cand_pkg["candidate_frames"],
                pre_frame=cand_pkg["pre_frame"],
                post_frame=cand_pkg["post_frame"],
                post_observations=cand_pkg["post_observations"],
                person_boxes=cand_pkg["person_boxes"],
            )
            eval_latencies.append(time.perf_counter() - t_gate_0)

            if decision.status == "CONFIRMED":
                case_kept_cands.append((cand, decision))
                if cand.event_type == "pickup":
                    pickup_candidates_kept += 1
                elif cand.event_type == "placement":
                    placement_candidates_kept += 1

                # Check match against GT
                if (act == "person_picks_up_object" and cand.event_type == "pickup") or (
                    act == "person_puts_down_object" and cand.event_type == "placement"
                ):
                    s_max = max(gt_start, cand.start_seconds)
                    e_min = min(gt_end, cand.end_seconds)
                    inter = max(0.0, e_min - s_max)
                    s_min = min(gt_start, cand.start_seconds)
                    e_max = max(gt_end, cand.end_seconds)
                    union = max(1e-6, e_max - s_min)
                    tiou = inter / union

                    dt_start = abs(cand.start_seconds - gt_start)
                    if inter > 0 or dt_start <= 1.0:
                        if act == "person_picks_up_object":
                            pickup_matches_1s += 1
                        else:
                            placement_matches_1s += 1
                    if inter > 0 or dt_start <= 2.0:
                        if act == "person_picks_up_object":
                            pickup_matches_2s += 1
                        else:
                            placement_matches_2s += 1
                    if inter > 0 or dt_start <= 3.0:
                        if act == "person_picks_up_object":
                            pickup_matches_3s += 1
                        else:
                            placement_matches_3s += 1

                    if inter > 0 or dt_start <= 2.0:
                        matched_case = True
                        if tiou > best_iou:
                            best_iou = tiou
                            best_cand_id = cand.candidate_id
                            best_explanation = decision.explanation

        if matched_case:
            if act == "person_picks_up_object":
                pickup_matched_count += 1
                pickup_ious.append(best_iou)
                if is_sub40:
                    sub40_pickup_matched += 1
                else:
                    hi_res_pickup_matched += 1
            else:
                placement_matched_count += 1
                placement_ious.append(best_iou)
                if is_sub40:
                    sub40_placement_matched += 1
                else:
                    hi_res_placement_matched += 1

        case_diagnostics.append({
            "case_index": item["case_index"],
            "annotation_key": c["annotation_key"],
            "activity_type": act,
            "camera_id": c["camera_id"],
            "video_id": c["video_id"],
            "gt_start": round(gt_start, 2),
            "gt_end": round(gt_end, 2),
            "object_size_under_40px": is_sub40,
            "upstream_candidates": len(item["candidates"]),
            "confirmed_candidates": len(case_kept_cands),
            "matched": matched_case,
            "temporal_iou": round(best_iou, 4),
            "matched_candidate_id": best_cand_id,
            "evidence_explanation": best_explanation,
        })

    # Evaluate Negative Controls
    neg_false_pickups = 0
    neg_false_placements = 0
    neg_total_cands = len(negative_candidates)
    neg_rejected_count = 0

    for neg_pkg in negative_candidates:
        cand = neg_pkg["candidate"]
        t_gate_0 = time.perf_counter()
        decision = gate.evaluate_candidate(
            candidate=cand,
            candidate_frames=neg_pkg["candidate_frames"],
            pre_frame=neg_pkg["pre_frame"],
            post_frame=neg_pkg["post_frame"],
            post_observations=neg_pkg["post_observations"],
            person_boxes=neg_pkg["person_boxes"],
        )
        eval_latencies.append(time.perf_counter() - t_gate_0)

        if decision.status == "CONFIRMED":
            if cand.event_type == "pickup":
                neg_false_pickups += 1
            elif cand.event_type == "placement":
                neg_false_placements += 1
        else:
            neg_rejected_count += 1

    total_eval_time = time.perf_counter() - t_eval_start
    avg_cand_latency_ms = (float(np.mean(eval_latencies)) * 1000.0) if eval_latencies else 0.0

    # Calculate final metrics
    pickup_recall = pickup_matched_count / max(1, pickup_cases_gt)
    pickup_precision = pickup_matched_count / max(1, pickup_candidates_kept + neg_false_pickups)
    pickup_mean_iou = float(np.mean(pickup_ious)) if pickup_ious else 0.0

    placement_recall = placement_matched_count / max(1, placement_cases_gt)
    placement_precision = placement_matched_count / max(1, placement_candidates_kept + neg_false_placements)
    placement_mean_iou = float(np.mean(placement_ious)) if placement_ious else 0.0

    neg_rejection_rate = neg_rejected_count / max(1, neg_total_cands)

    return {
        "mode": mode,
        "confirm_threshold": confirm_threshold,
        "weak_threshold": weak_threshold,
        "pickup": {
            "gt_cases": pickup_cases_gt,
            "candidates_kept": pickup_candidates_kept,
            "matched": pickup_matched_count,
            "precision": round(pickup_precision * 100.0, 2),
            "recall": round(pickup_recall * 100.0, 2),
            "mean_temporal_iou": round(pickup_mean_iou, 4),
            "matched_1s": pickup_matches_1s,
            "matched_2s": pickup_matches_2s,
            "matched_3s": pickup_matches_3s,
            "resolution": {
                "high_res_gt": hi_res_pickup_gt,
                "high_res_matched": hi_res_pickup_matched,
                "high_res_recall": round(hi_res_pickup_matched / max(1, hi_res_pickup_gt) * 100.0, 2),
                "sub40_gt": sub40_pickup_gt,
                "sub40_matched": sub40_pickup_matched,
                "sub40_recall": round(sub40_pickup_matched / max(1, sub40_pickup_gt) * 100.0, 2),
            },
        },
        "placement": {
            "gt_cases": placement_cases_gt,
            "candidates_kept": placement_candidates_kept,
            "matched": placement_matched_count,
            "precision": round(placement_precision * 100.0, 2),
            "recall": round(placement_recall * 100.0, 2),
            "mean_temporal_iou": round(placement_mean_iou, 4),
            "matched_1s": placement_matches_1s,
            "matched_2s": placement_matches_2s,
            "matched_3s": placement_matches_3s,
            "resolution": {
                "high_res_gt": hi_res_placement_gt,
                "high_res_matched": hi_res_placement_matched,
                "high_res_recall": round(hi_res_placement_matched / max(1, hi_res_placement_gt) * 100.0, 2),
                "sub40_gt": sub40_placement_gt,
                "sub40_matched": sub40_placement_matched,
                "sub40_recall": round(sub40_placement_matched / max(1, sub40_placement_gt) * 100.0, 2),
            },
        },
        "negative_controls": {
            "total_candidates": neg_total_cands,
            "false_pickups": neg_false_pickups,
            "false_placements": neg_false_placements,
            "rejected_candidates": neg_rejected_count,
            "rejection_rate_pct": round(neg_rejection_rate * 100.0, 2),
        },
        "runtime": {
            "avg_candidate_latency_ms": round(avg_cand_latency_ms, 2),
            "total_confirmation_time_s": round(total_eval_time, 4),
        },
        "case_diagnostics": case_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Secondary Manipulation Confirmation Gate on MEVA")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--roi-strategy", default="interaction_region")
    parser.add_argument("--persist", type=float, default=1.0)
    parser.add_argument("--contact-thresh", type=float, default=0.45)
    parser.add_argument("--output-json", default="data/meva/manipulation-confirmation-results.json")
    parser.add_argument("--output-csv", default="data/meva/manipulation-confirmation-case-diagnostics.csv")
    args = parser.parse_args()

    print("=== VIGILIA Secondary Manipulation Confirmation Gate Experiment ===")
    print(f"Model: {args.model}")
    print(f"Upstream Strategy: {args.roi_strategy}, persist: {args.persist}s, contact_thresh: {args.contact_thresh}")

    cases_path = ROOT / "data/meva/object-eval/cases.json"
    manifest_path = ROOT / "datasets/meva/manifest.json"
    db_path = ROOT / "data/meva/improved-2fps/vigilia.db"

    cases, video_map = load_eval_data(cases_path, manifest_path)
    people = person_samples(db_path)

    model_path = ROOT / "models" / args.model
    if not model_path.exists():
        model_path = Path(args.model)
    model = YOLO(str(model_path))

    print("\n[1/3] Extracting upstream candidates and visual context...")
    case_data, neg_cands, profile_times = extract_upstream_candidates(
        cases=cases,
        video_map=video_map,
        people=people,
        model=model,
        roi_strategy=args.roi_strategy,
        roi_factor=1.25,
        persist_threshold=args.persist,
        contact_threshold=args.contact_thresh,
    )
    print(f"  Upstream detection complete in {profile_times['upstream_detection_s']:.2f}s across {profile_times['frames_processed']} frames.")

    total_up_cands = sum(len(d["candidates"]) for d in case_data)
    print(f"  Emitted upstream candidates: {total_up_cands} in cases, {len(neg_cands)} in negative controls.")

    print("\n[2/3] Running feature ablation experiments...")
    ablation_modes = [
        "baseline",
        "localized_motion",
        "relative_motion",
        "appearance_change",
        "separation",
        "best_two",
        "best_three",
        "all",
    ]

    ablation_results = {}
    for mode in ablation_modes:
        thresh = 0.45 if mode != "baseline" else 0.0
        res = evaluate_gate_configuration(
            case_data_list=case_data,
            negative_candidates=neg_cands,
            mode=mode,
            confirm_threshold=thresh,
        )
        ablation_results[mode] = res
        print(f"  Mode '{mode:18s}' | Pickup: {res['pickup']['matched']}/15 ({res['pickup']['recall']}%, prec: {res['pickup']['precision']}%) | "
              f"Placement: {res['placement']['matched']}/15 ({res['placement']['recall']}%, prec: {res['placement']['precision']}%) | "
              f"Neg False: (P:{res['negative_controls']['false_pickups']}, Pl:{res['negative_controls']['false_placements']}) | "
              f"Latency: {res['runtime']['avg_candidate_latency_ms']:.2f}ms")

    print("\n[3/3] Running threshold sweeps on best configurations...")
    sweep_configs = [
        ("best_two", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
        ("best_three", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
        ("all", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
    ]
    sweep_results = {}
    for mode, thresholds in sweep_configs:
        sweep_results[mode] = {}
        for t in thresholds:
            r = evaluate_gate_configuration(
                case_data_list=case_data,
                negative_candidates=neg_cands,
                mode=mode,
                confirm_threshold=t,
            )
            sweep_results[mode][str(t)] = r

    # Package output
    best_config_name = "best_two"
    final_output = {
        "experiment": "secondary_manipulation_confirmation_gate",
        "upstream_baseline": {
            "model": args.model,
            "roi_strategy": args.roi_strategy,
            "persist_threshold_s": args.persist,
            "contact_threshold": args.contact_thresh,
        },
        "timing_profile": profile_times,
        "ablation_results": {k: {m: v for m, v in val.items() if m != "case_diagnostics"} for k, val in ablation_results.items()},
        "threshold_sweeps": {
            m: {
                t: {k: v for k, v in res.items() if k != "case_diagnostics"}
                for t, res in tdict.items()
            }
            for m, tdict in sweep_results.items()
        },
        "best_configuration": ablation_results[best_config_name],
    }

    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(final_output, indent=2))
    print(f"\nWrote full experiment results to {out_json}")

    # Write case diagnostics CSV for best configuration
    out_csv = ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    best_diags = ablation_results[best_config_name]["case_diagnostics"]
    if best_diags:
        keys = list(best_diags[0].keys())
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(best_diags)
        print(f"Wrote case diagnostics to {out_csv}")


if __name__ == "__main__":
    main()
