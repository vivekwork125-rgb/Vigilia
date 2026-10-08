#!/usr/bin/env python3
"""Combine fixed-frame MEVA detector probes into auditable CSV tables."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from meva.annotations import parse_geometry, parse_types  # noqa: E402
from meva.dataset import load_manifest, resolve  # noqa: E402


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(directory):
    reports = {
        path.stem: json.loads(path.read_text())
        for path in sorted(directory.glob("yolo*-c*.json"))
    }
    if "yolo11n-640-c030" not in reports:
        raise ValueError("Frozen YOLO11n 640/.30 raw baseline is required")
    baseline = reports["yolo11n-640-c030"]
    videos = {video["video_id"]: video for video in load_manifest()["videos"]}
    geometry = {}
    for video_id in {a["video_id"] for a in baseline["actors"]}:
        video = videos[video_id]
        geometry[video_id] = parse_geometry(
            resolve(video["geometry_path"]), video,
            parse_types(resolve(video["types_path"])),
        )
    summaries = []
    for name, report in sorted(reports.items()):
        actors = report["actors"]
        vehicles = [a for a in actors if a["annotation_actor_type"] == "vehicle"]
        objects = [a for a in actors if a["annotation_actor_type"] == "other"]
        vehicle_hits = [hit for actor in vehicles for hit in actor["hits"]]
        summaries.append({
            "configuration": name,
            "model_sha256": report["model_sha256"],
            "size": report["detection_size"],
            "confidence": report["confidence_threshold"],
            "source_frames": report["summary"]["unique_source_frames"],
            "start_actors_detected_of_32": sum(bool(a["hits"]) for a in vehicles if a["activity_type"] == "vehicle_starts"),
            "stop_actors_detected_of_27": sum(bool(a["hits"]) for a in vehicles if a["activity_type"] == "vehicle_stops"),
            "vehicles_detected_of_59": sum(bool(a["hits"]) for a in vehicles),
            "vehicles_with_at_least_two_hits_of_59": sum(len(a["hits"]) >= 2 for a in vehicles),
            "objects_broad_overlap_of_30": sum(bool(a["hits"]) for a in objects),
            "objects_in_production_supported_classes_of_30": sum(a["event_supported_object_hit_count"] > 0 for a in objects),
            "vehicle_mean_confidence": round(sum(h["confidence"] for h in vehicle_hits) / len(vehicle_hits), 4) if vehicle_hits else "",
            "vehicle_median_confidence": round(statistics.median(h["confidence"] for h in vehicle_hits), 4) if vehicle_hits else "",
            "detections_per_frame": report["summary"]["detections_per_frame"],
            "unmatched_boxes_per_frame_not_false_positive_rate": report["summary"]["unmatched_boxes_per_frame"],
            "inference_wall_seconds": report["summary"]["inference_wall_seconds"],
            "total_probe_wall_seconds": report["total_wall_seconds"],
        })
    write_rows(directory / "configuration-summary.csv", summaries)
    by_config = {
        name: {(a["annotation_key"], a["annotation_actor_id"]): a for a in report["actors"]}
        for name, report in reports.items()
    }
    object_rows = []
    vehicle_rows = []
    for actor in baseline["actors"]:
        key = (actor["annotation_key"], actor["annotation_actor_id"])
        video = videos[actor["video_id"]]
        reference = next(
            (geometry[actor["video_id"]].get(actor["annotation_actor_id"], {}).get(frame)
             for frame in actor["sample_frames"]
             if frame in geometry[actor["video_id"]].get(actor["annotation_actor_id"], {})),
            None,
        )
        center_x = round((reference[0] + reference[2]) / (2 * video["width"]), 3) if reference else ""
        center_y = round((reference[1] + reference[3]) / (2 * video["height"]), 3) if reference else ""
        if actor["annotation_actor_type"] == "other":
            best = [
                (name, hit)
                for name, table in by_config.items() for hit in table[key]["hits"]
            ]
            best.sort(key=lambda pair: (pair[1]["iou"], pair[1]["confidence"]), reverse=True)
            object_rows.append({
                "annotation_key": actor["annotation_key"],
                "activity_type": actor["activity_type"],
                "camera_id": actor["camera_id"],
                "annotation_actor_id": actor["annotation_actor_id"],
                "sampled_frames": len(actor["sample_frames"]),
                "median_box_width_px": actor["median_box_width"],
                "median_box_height_px": actor["median_box_height"],
                "median_image_fraction": round(actor["median_box_area_fraction"], 7),
                "image_center_x_fraction": center_x,
                "image_center_y_fraction": center_y,
                "baseline_hits": actor["hit_count"],
                "best_overlap_configuration": best[0][0] if best else "",
                "best_overlap_class": best[0][1]["class"] if best else "",
                "best_overlap_confidence": best[0][1]["confidence"] if best else "",
                "best_overlap_iou": best[0][1]["iou"] if best else "",
                "any_production_supported_class": any(table[key]["event_supported_object_hit_count"] > 0 for table in by_config.values()),
                "visual_review": "",
                "main_failure": "",
            })
        elif actor["annotation_actor_type"] == "vehicle" and not actor["hits"]:
            row = {
                "annotation_key": actor["annotation_key"],
                "activity_type": actor["activity_type"],
                "camera_id": actor["camera_id"],
                "annotation_actor_id": actor["annotation_actor_id"],
                "median_box_width_px": actor["median_box_width"],
                "median_box_height_px": actor["median_box_height"],
                "image_center_x_fraction": center_x,
                "image_center_y_fraction": center_y,
                "baseline_stored_track": actor["baseline_tracked"],
            }
            for name, table in by_config.items():
                row[name + "_hit_count"] = len(table[key]["hits"])
            row["visual_review"] = ""
            row["main_failure"] = ""
            vehicle_rows.append(row)
    write_rows(directory / "manipulation-objects.csv", sorted(object_rows, key=lambda r: r["annotation_key"]))
    write_rows(directory / "baseline-vehicle-misses.csv", sorted(vehicle_rows, key=lambda r: r["annotation_key"]))
    print(f"Wrote {len(summaries)} configurations, {len(object_rows)} object cases, {len(vehicle_rows)} baseline vehicle misses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/meva/perception-experiment"))
    run(parser.parse_args().run_dir)
