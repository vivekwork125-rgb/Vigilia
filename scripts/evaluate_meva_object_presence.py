"""Post-hoc evaluation of full-video object detections against MEVA geometry.

No labels or review notes are read by the extractor. A box overlap with generic
MEVA `other` is diagnostic, not proof of a semantic class or physical identity.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.annotations import parse_activities, parse_geometry, parse_types
from meva.dataset import load_manifest, resolve
from meva.object_presence import box_iou, build_object_tracks, deduplicate_frame
from meva.selective_objects import associate, person_samples

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def load_runtime(run_dir, manifest):
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("protocol") != "full-video sampled runtime-only detection; no activity annotations":
        raise ValueError("Not a runtime-only extraction artifact")
    videos = {}
    for video in manifest["videos"]:
        p = run_dir / f"{video['video_id']}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Incomplete run: {p}")
        data = json.loads(p.read_text())
        expected = len(range(0, video["frame_count"], round(video["fps"] / cfg["sample_fps"])))
        if (data["source_sha256"] != video["sha256"] or data["sampled_frames"] != expected
                or data["model_sha256"] != cfg["model_sha256"]
                or data["imgsz"] != cfg["imgsz"] or data["confidence"] != cfg["confidence"]):
            raise ValueError(f"Incomplete or inconsistent runtime video: {p}")
        videos[video["video_id"]] = data
    return cfg, videos


def assign_frame(detections, references, threshold=.3, compatibility=None):
    """One-to-one box matching per frame; no detector hit can claim two GT actors."""
    if not detections or not references:
        return []
    weights = np.zeros((len(detections), len(references)))
    for i, det in enumerate(detections):
        for j, ref in enumerate(references):
            if compatibility and not compatibility(det, ref):
                continue
            iou = box_iou(det["box"], ref["box"])
            if iou >= threshold:
                weights[i, j] = 1 + iou
    row, col = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j), float(weights[i, j]-1))
            for i, j in zip(row, col) if weights[i, j] > 0]


def reference_actors(video, fps, source_types, geometry):
    step = round(fps / 2)
    return {actor_id: {f: box for f, box in boxes.items() if f % step == 0}
            for actor_id, boxes in geometry.items()
            if source_types.get(actor_id) in {"other", "bicycle", "vehicle"}}


def actor_evidence(actor_id, actor_type, geometry, by_frame, tracks, class_filter=None):
    """Frame coverage and provisional track fragments at exact annotated frames."""
    annotated = geometry.get(actor_id, {})
    hits = []
    classes = Counter()
    matched_track_ids = set()
    for frame, box in annotated.items():
        options = [(box_iou(det["box"], box), det) for det in by_frame.get(frame, [])
                   if det["class"] != "person" and
                   (class_filter is None or det["class"] in class_filter)]
        best_iou, best = max(options, key=lambda item: item[0], default=(0., None))
        if best and best_iou >= .3:
            hits.append(frame)
            classes[best["class"]] += 1
    for track in tracks:
        if class_filter is not None and track["class"] not in class_filter:
            continue
        if any(det["frame"] in annotated and box_iou(det["box"], annotated[det["frame"]]) >= .3
               for det in track["detections"]):
            matched_track_ids.add(track["track_id"])
    return {"actor_id": actor_id, "actor_type": actor_type,
            "annotated_sample_frames": len(annotated), "matched_sample_frames": len(hits),
            "coverage": len(hits)/len(annotated) if annotated else None,
            "matched_classes": dict(classes),
            "provisional_track_fragments": len(matched_track_ids),
            "matched_track_ids": sorted(matched_track_ids)}


def evaluate(args):
    manifest = load_manifest(args.manifest)
    cfg, runtime = load_runtime(args.run_dir, manifest)
    review = json.loads(args.visibility_review.read_text())
    review_by_key = {row["annotation_key"]: row for row in review["cases"]}
    if len(review_by_key) != 30:
        raise ValueError("Visibility review must retain all 30 manipulation cases")
    cases = json.loads(args.cases.read_text())
    cases_by_key = {c["annotation_key"]: c for c in cases}
    if set(cases_by_key) != set(review_by_key):
        raise ValueError("Visibility review and evaluation cases differ")
    people = person_samples(args.database)
    per_video = []
    case_rows = []
    control_rows = []
    track_rows = []
    total_track_s = 0.
    for video in manifest["videos"]:
        vid = video["video_id"]
        cached = runtime[vid]
        # Runtime association uses only the cached detections. All GT parsing below
        # occurs after tracks are formed and never mutates boxes or track links.
        by_frame = defaultdict(list)
        for det in cached["detections"]:
            if det["class"] != "person":
                by_frame[det["frame"]].append(det)
        deduped = {f: deduplicate_frame(rows) for f, rows in by_frame.items()}
        object_dets = [d for frame in sorted(deduped) for d in deduped[frame]]
        t0 = time.perf_counter()
        tracks = build_object_tracks(object_dets)
        tracking_s = time.perf_counter()-t0
        total_track_s += tracking_s
        persistent = [t for t in tracks if t["persistent"]]
        person_assoc = {t["track_id"]: associate(t["detections"], people.get(vid, {}))
                        for t in persistent}
        track_rows.extend(tracks)
        types = parse_types(resolve(video["types_path"]))
        geometry = parse_geometry(resolve(video["geometry_path"]), video, types)
        activities = parse_activities(resolve(video["annotation_path"]), video, types)
        activity_by_key = {f"{vid}:{a['annotation_id']}:{a['span_index']}": a
                           for a in activities}
        actor_boxes = reference_actors(video, video["fps"], types, geometry)
        for key, case in cases_by_key.items():
            if case["video_id"] != vid:
                continue
            act = activity_by_key.get(key)
            if act is None or act["object_id"] is None:
                raise ValueError(f"Missing GT object actor for {key}")
            actor_id = act["object_id"]
            object_geo = actor_boxes.get(actor_id, {})
            activity_frames = {f: box for f, box in object_geo.items()
                               if act["start_frame"] <= f <= act["end_frame"]}
            best = (0., None)
            strict_hit_frames = []
            loose_hit_frames = []
            for frame, box in activity_frames.items():
                options = [(box_iou(det["box"], box), det)
                           for det in deduped.get(frame, [])]
                local = max(options, key=lambda row: row[0], default=(0., None))
                if local[0] > best[0]:
                    best = local
                if local[0] >= .3:
                    strict_hit_frames.append(frame)
                if local[0] >= .1:
                    loose_hit_frames.append(frame)
            track_ids = []
            matched_frames_by_track = {}
            for tr in tracks:
                matched_frames = {det["frame"] for det in tr["detections"]
                                  if det["frame"] in activity_frames
                                  and box_iou(det["box"], activity_frames[det["frame"]]) >= .3}
                if matched_frames:
                    track_ids.append(tr["track_id"])
                    matched_frames_by_track[tr["track_id"]] = len(matched_frames)
            case_rows.append({
                "annotation_key": key, "video_id": vid, "activity_type": case["activity_type"],
                "object_actor_id": actor_id,
                "visibility": review_by_key[key]["visibility"],
                "approx_width_px": review_by_key[key]["approx_width_px"],
                "object_actor_type": types[actor_id],
                "annotated_sample_frames": len(activity_frames),
                "overlap_0_1": bool(loose_hit_frames),
                "overlap_0_3": bool(strict_hit_frames),
                "matched_frames_0_3": len(strict_hit_frames),
                "best_iou": round(best[0], 4),
                "best_detector_class": best[1]["class"] if best[1] else None,
                "best_confidence": best[1]["confidence"] if best[1] else None,
                "provisional_track_ids": track_ids,
                "same_track_matched_annotated_frames_max": max(
                    matched_frames_by_track.values(), default=0),
                "persistent_provisional_tracks": sum(
                    tr["track_id"] in track_ids and tr["persistent"] for tr in tracks),
                "person_associations": [person_assoc[tid] for tid in track_ids
                                        if person_assoc.get(tid)],
            })
        for actor_id, typ in types.items():
            if typ not in {"bicycle", "vehicle"}:
                continue
            filt = {"bicycle"} if typ == "bicycle" else VEHICLE_CLASSES
            control_rows.append({"video_id": vid, **actor_evidence(
                actor_id, typ, actor_boxes, deduped, tracks, class_filter=filt)})
        per_video.append({
            "video_id": vid, "duration_seconds": video["duration"],
            "source_frames": video["frame_count"], "sampled_frames": cached["sampled_frames"],
            "all_detections": len(cached["detections"]),
            "nonperson_detections": len(object_dets),
            "class_counts": dict(Counter(d["class"] for d in object_dets)),
            "tentative_tracks": len(tracks), "persistent_tracks": len(persistent),
            "persistent_track_median_duration_seconds": round(statistics.median(
                t["duration_seconds"] for t in persistent), 3) if persistent else None,
            "persistent_tracks_with_gap": sum(t["gap_count"] > 0 for t in persistent),
            "ambiguous_predecessor_tracks": sum(t["ambiguous_predecessor"] for t in tracks),
            "uniquely_near_person_tracks": sum(bool(x) for x in person_assoc.values()),
            "wall_seconds": cached["wall_seconds"],
            "inference_seconds": cached["inference_seconds"],
            "tracking_seconds": round(tracking_s, 4),
            "device": cached["device"],
        })
    group = defaultdict(list)
    for row in case_rows:
        group[row["visibility"]].append(row)
    stratified = {cat: {"cases": len(rows), "box_overlap_0_1": sum(r["overlap_0_1"] for r in rows),
                        "box_overlap_0_3": sum(r["overlap_0_3"] for r in rows),
                        "persistent_tracks": sum(r["persistent_provisional_tracks"] > 0 for r in rows),
                        "frames_with_geometry": sum(r["annotated_sample_frames"] for r in rows)}
                  for cat, rows in group.items()}
    controls = {}
    for typ in ("bicycle", "vehicle"):
        rows = [r for r in control_rows if r["actor_type"] == typ and r["annotated_sample_frames"]]
        controls[typ] = {"actors": len(rows),
                         "annotated_sample_frames": sum(r["annotated_sample_frames"] for r in rows),
                         "matched_sample_frames": sum(r["matched_sample_frames"] for r in rows),
                         "actors_with_detection": sum(r["matched_sample_frames"] > 0 for r in rows),
                         "actors_with_provisional_track": sum(r["provisional_track_fragments"] > 0 for r in rows),
                         "actors_with_multiple_provisional_tracks": sum(r["provisional_track_fragments"] > 1 for r in rows)}
    result = {
        "protocol": "inference/cache first; post-hoc GT geometry and visual-review scoring",
        "configuration": cfg,
        "videos": per_video,
        "summary": {
            "videos": len(per_video),
            "video_duration_seconds": round(sum(v["duration_seconds"] for v in per_video), 3),
            "sampled_frames": sum(v["sampled_frames"] for v in per_video),
            "all_detections": sum(v["all_detections"] for v in per_video),
            "nonperson_detections": sum(v["nonperson_detections"] for v in per_video),
            "tentative_tracks": sum(v["tentative_tracks"] for v in per_video),
            "persistent_tracks": sum(v["persistent_tracks"] for v in per_video),
            "persistent_track_median_duration_seconds": round(statistics.median(
                t["duration_seconds"] for t in track_rows if t["persistent"]), 3),
            "persistent_tracks_with_gap": sum(t["gap_count"] > 0 for t in track_rows if t["persistent"]),
            "ambiguous_predecessor_tracks": sum(t["ambiguous_predecessor"] for t in track_rows),
            "uniquely_near_person_tracks": sum(v["uniquely_near_person_tracks"] for v in per_video),
            "total_detection_wall_seconds": round(sum(v["wall_seconds"] for v in per_video), 3),
            "total_inference_seconds": round(sum(v["inference_seconds"] for v in per_video), 3),
            "total_tracking_seconds": round(total_track_s, 3),
            "manipulation_cases": len(case_rows),
            "manipulation_box_overlap_0_1": sum(r["overlap_0_1"] for r in case_rows),
            "manipulation_box_overlap_0_3": sum(r["overlap_0_3"] for r in case_rows),
            "manipulation_persistent_provisional_tracks": sum(r["persistent_provisional_tracks"] > 0 for r in case_rows),
            "manipulation_cases_with_two_matched_frames_same_track": sum(
                r["same_track_matched_annotated_frames_max"] >= 2 for r in case_rows),
        },
        "visibility_strata": stratified,
        "known_type_controls": controls,
        "cases": case_rows,
        "control_actors": control_rows,
        "object_tracks": [{
            "track_id": tr["track_id"], "video_id": tr["video_id"],
            "camera_id": tr["camera_id"], "source_sha256": tr["source_sha256"],
            "detector_class": tr["class"],
            "association_method": tr["association_method"],
            "detection_ids": [det["detection_id"] for det in tr["detections"]],
            "source_frames": [det["frame"] for det in tr["detections"]],
            "start_seconds": tr["start_seconds"], "end_seconds": tr["end_seconds"],
            "sample_count": tr["sample_count"], "persistent": tr["persistent"],
            "gap_count": tr["gap_count"],
            "ambiguous_predecessor": tr["ambiguous_predecessor"],
        } for tr in track_rows],
        "metric_limits": [
            "MEVA other has no reliable semantic class; overlap is class-agnostic and not detection precision.",
            "Activity actor geometry is sparse and may not cover every visible object; false-detection rate cannot be calculated.",
            "A class-consistent tracklet is provisional identity; true ID switches/false associations are not verified by this audit.",
            "Single-frame visual-review labels do not certify object visibility across a full activity interval.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [k for k in case_rows[0] if k not in {"provisional_track_ids", "person_associations"}]
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in case_rows)
    print(json.dumps({"summary": result["summary"], "visibility": stratified,
                      "known_type_controls": controls}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"datasets/meva/manifest.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=ROOT/"data/meva/improved-2fps/vigilia.db")
    parser.add_argument("--cases", type=Path, default=ROOT/"data/meva/object-eval/cases.json")
    parser.add_argument("--visibility-review", type=Path, default=ROOT/"datasets/meva/object-visibility-review.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
