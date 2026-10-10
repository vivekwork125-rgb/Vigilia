"""Leakage-controlled, full-video audit of the evaluation-only confirmation gate.

Extraction reads only the MEVA manifest, video pixels, and frozen runtime person tracks.
Evaluation reads annotations only after the extraction artifact is complete. The two
commands are deliberately separate so ground truth cannot select inference frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
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
from meva.manipulation_confirmation import ManipulationConfirmationGate
from meva.selective_objects import person_samples

MODES = {
    "baseline": {},
    "localized_motion": {"localized_motion": 1.0},
    "relative_motion": {"relative_motion_coupling": 1.0},
    "appearance_change": {"appearance_change": 1.0},
    "separation": {"separation_evidence": 1.0},
    "best_two": {"appearance_change": .4, "separation_evidence": .6},
    "best_three": {"localized_motion": .25, "appearance_change": .35, "separation_evidence": .4},
    "all": {"localized_motion": .2, "relative_motion_coupling": .15,
            "appearance_change": .25, "region_persistence": .15,
            "separation_evidence": .25},
}
NEGATIVE_WINDOWS = [
    ("MEVA-G421-20180315-1555", 10, 20),
    ("MEVA-G421-20180315-1555", 25, 35),
    ("MEVA-G331-20180315-1555", 15, 25),
    ("MEVA-G331-20180315-1555", 40, 50),
    ("MEVA-G336-20180315-1555", 10, 20),
    ("MEVA-G336-20180315-1555", 30, 40),
    ("MEVA-G424-20180315-1555", 15, 25),
    ("MEVA-G328-20180315-1555", 10, 20),
    ("MEVA-G638-20180315-1555", 20, 30),
    ("MEVA-G436-20180315-1555", 15, 25),
]


def read_frame(cap, frame_no):
    if frame_no < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    return frame if ok else None


def extract(args):
    manifest = load_manifest(args.manifest)
    people = person_samples(args.database)
    model = YOLO(str(args.model))
    gate = ManipulationConfirmationGate(mode="all")
    rows = []
    total_frames = total_crops = 0
    feature_seconds = 0.0
    begun = time.perf_counter()
    for video in manifest["videos"]:
        vid = video["video_id"]
        fps = video["fps"]
        step = round(fps / 2.0)
        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video['local_path']}")
        machines = defaultdict(lambda: ContactStateMachine(
            persist_duration=1.0, contact_threshold=.45))
        observations = defaultdict(list)
        person_boxes = {}
        candidates = []
        try:
            # Every sampled frame with a stored runtime person is considered, across
            # each full video; annotation intervals are unavailable in this process.
            for frame_no in sorted(people.get(vid, {})):
                if frame_no % step:
                    continue
                frame = read_frame(cap, frame_no)
                if frame is None:
                    raise RuntimeError(f"Undecodable {vid} frame {frame_no}")
                total_frames += 1
                for person in people[vid][frame_no]:
                    pid = str(person["track_id"])
                    pbox = tuple(person["box"])
                    person_boxes[(pid, frame_no)] = pbox
                    x1, y1, x2, y2 = compute_person_roi(
                        pbox, video["width"], video["height"],
                        strategy="interaction_region", factor=1.25)
                    if x2-x1 < 16 or y2-y1 < 16:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    total_crops += 1
                    result = model.predict(crop, imgsz=640, conf=.20, verbose=False)[0]
                    if result.keypoints is None or len(result.keypoints) == 0:
                        continue
                    poses = extract_pose_keypoints(
                        result.keypoints.data.cpu().numpy(), (x1, y1), min_conf=.30)
                    for pose in poses:
                        for state in evaluate_hand_contact_state(
                            pbox, pose, frame_pixels=frame, hand_conf_threshold=.30):
                            obs = ContactObservation(
                                timestamp=frame_no/fps, frame_index=frame_no,
                                person_track_id=pid, hand_side=state["hand_side"],
                                hand_box=state["hand_box"],
                                interaction_region=state["interaction_region"],
                                contact_probability=state["contact_probability"],
                                model_confidence=state["wrist_conf"],
                                source_video=vid, source_frame=frame_no,
                                level=state["level"],
                                attributes={"reach_score": state["reach_score"],
                                            "patch_contrast": state["patch_contrast"]},
                            )
                            observations[pid].append(obs)
                            candidates.extend(machines[pid].process_observation(obs))
            for cand in candidates:
                obs = cand.supporting_observations
                frames = [read_frame(cap, o.frame_index) for o in obs]
                if any(f is None for f in frames):
                    raise RuntimeError(f"Missing evidence frame for {cand.candidate_id}")
                pre = read_frame(cap, cand.start_frame-step)
                post = read_frame(cap, cand.end_frame+step)
                later = [o for o in observations[cand.person_track_id]
                         if cand.end_frame < o.frame_index <= cand.end_frame+3*step]
                boxes = [person_boxes.get((cand.person_track_id, o.frame_index), o.hand_box)
                         for o in obs]
                t0 = time.perf_counter()
                decision = gate.evaluate_candidate(
                    cand, frames, pre_frame=pre, post_frame=post,
                    post_observations=later, person_boxes=boxes)
                feature_seconds += time.perf_counter()-t0
                rows.append({
                    "candidate_id": cand.candidate_id, "video_id": vid,
                    "camera_id": video["camera_id"], "event_type": cand.event_type,
                    "person_track_id": cand.person_track_id,
                    "start_seconds": cand.start_seconds, "end_seconds": cand.end_seconds,
                    "start_frame": cand.start_frame, "end_frame": cand.end_frame,
                    "source_sha256": video["sha256"],
                    "person_boxes": [{"frame": o.frame_index, "box": list(box)}
                                     for o, box in zip(obs, boxes)],
                    "observations": [asdict(o) for o in obs],
                    "features": decision.features.to_dict(),
                })
        finally:
            cap.release()
        print(f"{vid}: {len(candidates)} candidates", flush=True)
    output = {
        "protocol": "full-video runtime-only extraction; annotations unavailable",
        "model": str(args.model), "model_sha256": None,
        "roi": "interaction_region", "persistence_seconds": 1.0,
        "pose_confidence": .20, "contact_threshold": .45,
        "source_database": str(args.database), "sampling_fps": 2,
        "sampled_person_frames": total_frames, "pose_crops": total_crops,
        "wall_seconds": time.perf_counter()-begun,
        "feature_seconds": feature_seconds, "candidates": rows,
    }
    from meva.dataset import sha256
    output["model_sha256"] = sha256(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Saved {len(rows)} candidates to {args.output}")


def load_truth(manifest_path, cases_path):
    from meva.annotations import parse_activities, parse_geometry, parse_types

    manifest = load_manifest(manifest_path)
    selected = {c["annotation_key"]: c for c in json.loads(cases_path.read_text())}
    truths = []
    geometry = {}
    all_activities = defaultdict(list)
    for video in manifest["videos"]:
        vid = video["video_id"]
        types = parse_types(resolve(video["types_path"]))
        geo = parse_geometry(resolve(video["geometry_path"]), video, types)
        activities = parse_activities(resolve(video["annotation_path"]), video, types)
        all_activities[vid] = activities
        for act in activities:
            key = f"{vid}:{act['annotation_id']}:{act['span_index']}"
            if key in selected:
                act["key"] = key
                act["under_40px"] = selected[key]["approximate_geometry"]["under_40px_width"]
                truths.append(act)
                geometry[key] = geo.get(act["actor_id"], {})
    if len(truths) != len(selected):
        raise RuntimeError(f"Found {len(truths)} of {len(selected)} evaluation cases")
    return truths, geometry, all_activities


def temporal_iou(a, b):
    inter = max(0., min(a["end_seconds"], b["end_seconds"])-max(a["start_seconds"], b["start_seconds"]))
    union = max(a["end_seconds"], b["end_seconds"])-min(a["start_seconds"], b["start_seconds"])
    return inter/union if union > 0 else 0.


def eligible(pred, truth, geometry):
    from meva.evaluation import box_iou

    act = "person_picks_up_object" if pred["event_type"] == "pickup" else "person_puts_down_object"
    if pred["video_id"] != truth["video_id"] or act != truth["activity_type"]:
        return False
    iou = temporal_iou(pred, truth)
    ends = (abs(pred["start_seconds"]-truth["start_seconds"]) <= 2 and
            abs(pred["end_seconds"]-truth["end_seconds"]) <= 2)
    if iou < .1 and not ends:
        return False
    actor_boxes = geometry[truth["key"]]
    spatial = max((box_iou(row["box"], actor_boxes[row["frame"]])
                   for row in pred["person_boxes"] if row["frame"] in actor_boxes), default=0.)
    return spatial >= .1


def score(row, mode):
    if mode == "baseline":
        return 1.0
    features = row["features"]
    return sum(features[k]*v for k, v in MODES[mode].items())


def evaluate_set(rows, truths, geometry, mode, threshold):
    selected = [r for r in rows if score(r, mode) >= threshold]
    weights = np.zeros((len(selected), len(truths)))
    for i, pred in enumerate(selected):
        for j, truth in enumerate(truths):
            if eligible(pred, truth, geometry):
                weights[i, j] = 100 + temporal_iou(pred, truth)
    pairs = []
    if weights.size:
        ii, jj = linear_sum_assignment(weights, maximize=True)
        pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if weights[i, j] > 0]
    breakdown = {}
    for kind, act in (("pickup", "person_picks_up_object"),
                      ("placement", "person_puts_down_object")):
        pred_indices = [i for i, r in enumerate(selected) if r["event_type"] == kind]
        gt_indices = [j for j, t in enumerate(truths) if t["activity_type"] == act]
        local = [(i, j) for i, j in pairs if i in pred_indices and j in gt_indices]
        breakdown[kind] = {
            "gt": len(gt_indices), "candidates": len(pred_indices), "matches": len(local),
            "precision": len(local)/len(pred_indices) if pred_indices else 0.,
            "recall": len(local)/len(gt_indices) if gt_indices else 0.,
            "mean_temporal_iou": float(np.mean([temporal_iou(selected[i], truths[j])
                                                   for i, j in local])) if local else None,
            "sub40_gt": sum(truths[j]["under_40px"] for j in gt_indices),
            "sub40_matches": sum(truths[j]["under_40px"] for _, j in local),
            "highres_matches": sum(not truths[j]["under_40px"] for _, j in local),
            "both_endpoints_within_1s": sum(
                abs(selected[i]["start_seconds"]-truths[j]["start_seconds"]) <= 1
                and abs(selected[i]["end_seconds"]-truths[j]["end_seconds"]) <= 1
                for i, j in local),
            "both_endpoints_within_2s": sum(
                abs(selected[i]["start_seconds"]-truths[j]["start_seconds"]) <= 2
                and abs(selected[i]["end_seconds"]-truths[j]["end_seconds"]) <= 2
                for i, j in local),
            "both_endpoints_within_3s": sum(
                abs(selected[i]["start_seconds"]-truths[j]["start_seconds"]) <= 3
                and abs(selected[i]["end_seconds"]-truths[j]["end_seconds"]) <= 3
                for i, j in local),
        }
    return selected, pairs, breakdown


def evaluate(args):
    artifact = json.loads(args.input.read_text())
    if artifact.get("protocol") != "full-video runtime-only extraction; annotations unavailable":
        raise ValueError("Refusing biased or unknown extraction artifact")
    rows = artifact["candidates"]
    truths, geometry, activities = load_truth(args.manifest, args.cases)
    # The ten windows are post-hoc controls only; exclude any window with a MEVA
    # pickup/placement annotation before treating emitted candidates as false.
    controls = []
    for vid, start, end in NEGATIVE_WINDOWS:
        overlapping = [a for a in activities[vid]
                       if a["activity_type"] in {"person_picks_up_object", "person_puts_down_object"}
                       and a["start_seconds"] < end and a["end_seconds"] > start]
        if not overlapping:
            controls.append((vid, start, end))
    result = {"extraction": {k: v for k, v in artifact.items() if k != "candidates"},
              "truth_count": len(truths), "negative_windows_verified": len(controls),
              "protocol": "one-to-one video/type/time/actor-box matching; IoU >= .1 or both endpoints <= 2 s; actor IoU >= .1",
              "ablations": {}, "threshold_sweep": {}}
    for mode in MODES:
        thresholds = [0.] if mode == "baseline" else ([.35,.4,.45,.5,.55,.6] if mode in {"all", "best_two", "best_three"} else [.45])
        for threshold in thresholds:
            selected, pairs, breakdown = evaluate_set(rows, truths, geometry, mode, threshold)
            matched_indices = {i for i, _ in pairs}
            controls_in = [r for r in rows if any(
                r["video_id"] == vid and r["start_seconds"] < end and r["end_seconds"] > start
                for vid, start, end in controls)]
            controls_kept = [r for r in controls_in if score(r, mode) >= threshold]
            counts = {kind: sum(r["event_type"] == kind for r in controls_kept)
                      for kind in ("pickup", "placement")}
            entry = {"pickup": breakdown["pickup"], "placement": breakdown["placement"],
                     "negative_controls": {"baseline_candidates": len(controls_in),
                                           "confirmed_candidates": len(controls_kept),
                                           "false_pickups": counts["pickup"],
                                           "false_placements": counts["placement"],
                                           "rejection_rate": 1-len(controls_kept)/len(controls_in) if controls_in else None},
                     "matched_candidate_ids": [selected[i]["candidate_id"] for i in sorted(matched_indices)]}
            if threshold == .45 and mode != "baseline" or mode == "baseline":
                result["ablations"][mode] = entry
            if mode in {"all", "best_two", "best_three"}:
                result["threshold_sweep"].setdefault(mode, {})[str(threshold)] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"baseline": result["ablations"]["baseline"],
                      "relative_motion": result["ablations"]["relative_motion"],
                      "all_050": result["threshold_sweep"]["all"]["0.5"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--manifest", type=Path, default=ROOT/"datasets/meva/manifest.json")
    ex.add_argument("--database", type=Path, default=ROOT/"data/meva/improved-2fps/vigilia.db")
    ex.add_argument("--model", type=Path, default=ROOT/"models/yolo11n-pose.pt")
    ex.add_argument("--output", type=Path, default=ROOT/"data/meva/confirmation-full-video-candidates.json")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--manifest", type=Path, default=ROOT/"datasets/meva/manifest.json")
    ev.add_argument("--cases", type=Path, default=ROOT/"data/meva/object-eval/cases.json")
    ev.add_argument("--input", type=Path, default=ROOT/"data/meva/confirmation-full-video-candidates.json")
    ev.add_argument("--output", type=Path, default=ROOT/"data/meva/confirmation-unbiased-results.json")
    args = parser.parse_args()
    {"extract": extract, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
