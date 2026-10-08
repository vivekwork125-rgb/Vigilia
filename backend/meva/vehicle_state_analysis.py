"""Evaluation-only vehicle state traces from completed VIGILIA observations.

MEVA labels select tracks for diagnosis and are never used to produce runtime
events. Historical observations retain mean confidence only, not frame scores.
"""

import csv
import itertools
import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

from app.temporal_events import center, speed, vehicle_transitions


def median_or_none(values):
    return round(statistics.median(values), 4) if values else None


def state_trace(samples, track_confidence):
    """Describe measured motion; states are diagnostic estimates, not GT."""
    samples = sorted(samples, key=lambda sample: sample["t"])
    if not samples:
        return []
    scale = max(1, statistics.median(s["box"][3] - s["box"][1] for s in samples))
    edges = [speed(a, b, scale) for a, b in itertools.pairwise(samples)]
    smooth = [
        statistics.median(edges[max(0, i - 1):min(len(edges), i + 2)])
        for i in range(len(edges))
    ]
    trace = []
    for i, sample in enumerate(samples):
        gap = sample["t"] - samples[i - 1]["t"] if i else None
        velocity = edges[i - 1] if i else None
        smoothed = smooth[i - 1] if i else None
        prior = smooth[i - 2] if i >= 2 else None
        acceleration = (smoothed - prior) / gap if prior is not None and gap else None
        if i == 0 or gap > 1.5:
            state = "UNKNOWN"
        elif smoothed <= 0.12:
            state = "STATIONARY"
        elif smoothed >= 0.28:
            state = "MOVING"
        else:
            state = "UNKNOWN"
        if acceleration is not None and state != "STATIONARY":
            if acceleration >= 0.12:
                state = "ACCELERATING"
            elif acceleration <= -0.12:
                state = "DECELERATING"
        x, y = center(sample)
        trace.append({
            "timestamp": sample["t"], "frame": sample["frame"],
            "box": sample["box"], "box_height": sample["box"][3] - sample["box"][1],
            "center": [x, y],
            "normalized_speed": round(velocity, 4) if velocity is not None else None,
            "smoothed_speed": round(smoothed, 4) if smoothed is not None else None,
            "acceleration": round(acceleration, 4) if acceleration is not None else None,
            "deceleration": round(min(0, acceleration), 4) if acceleration is not None else None,
            "confidence": track_confidence,
            "confidence_source": "track_mean; per-frame scores not persisted in this run",
            "gap_seconds": round(gap, 4) if gap is not None else None,
            "state": state,
        })
    return trace


def duration_in_state(trace, state, start, end):
    return round(sum(
        min(end, row["timestamp"]) - max(start, previous["timestamp"])
        for previous, row in itertools.pairwise(trace)
        if row["state"] == state
        and row["gap_seconds"] is not None and row["gap_seconds"] <= 1.5
        and max(start, previous["timestamp"]) < min(end, row["timestamp"])
    ), 3)


def trace_metrics(trace, gt_start, gt_end):
    speeds = [r["smoothed_speed"] for r in trace if r["smoothed_speed"] is not None]
    before = [r["smoothed_speed"] for r in trace if r["smoothed_speed"] is not None and gt_start - 2 <= r["timestamp"] < gt_start]
    after = [r["smoothed_speed"] for r in trace if r["smoothed_speed"] is not None and gt_end < r["timestamp"] <= gt_end + 2]
    first_half = [r["smoothed_speed"] for r in trace if r["smoothed_speed"] is not None and gt_start <= r["timestamp"] <= (gt_start + gt_end) / 2]
    second_half = [r["smoothed_speed"] for r in trace if r["smoothed_speed"] is not None and (gt_start + gt_end) / 2 < r["timestamp"] <= gt_end]
    acceleration = [r["acceleration"] for r in trace if r["acceleration"] is not None]
    return {
        "track_start": trace[0]["timestamp"], "track_end": trace[-1]["timestamp"],
        "track_duration": round(trace[-1]["timestamp"] - trace[0]["timestamp"], 3),
        "median_speed": median_or_none(speeds),
        "max_speed": max(speeds, default=None),
        "speed_before": median_or_none(before),
        "speed_after": median_or_none(after),
        "speed_event_first_half": median_or_none(first_half),
        "speed_event_second_half": median_or_none(second_half),
        "max_acceleration": max(acceleration, default=None),
        "max_deceleration": min(acceleration, default=None),
        "stationary_before": duration_in_state(trace, "STATIONARY", gt_start - 4, gt_start),
        "stationary_after": duration_in_state(trace, "STATIONARY", gt_end, gt_end + 4),
        "longest_gap": max((r["gap_seconds"] for r in trace if r["gap_seconds"] is not None), default=0),
        "sample_count": len(trace),
        "samples_during_activity": sum(gt_start <= r["timestamp"] <= gt_end for r in trace),
    }


def failure_reason(activity, metrics, candidates, gt_start, gt_end, matched):
    if matched:
        return "MATCHED"
    kind = "started_moving" if activity == "vehicle_starts" else "stopped"
    same = [c for c in candidates if c.kind == kind]
    if same and any(c.start <= gt_end + 2 and c.end >= gt_start - 2 for c in same):
        return "EVENT_BOUNDARY_OR_SPATIAL_MISMATCH"
    if same:
        return "TEMPORAL_WINDOW_MISMATCH"
    if metrics["sample_count"] < 7 or metrics["track_duration"] < 3:
        return "INSUFFICIENT_TRACK_DURATION"
    if metrics["longest_gap"] > 1.5:
        return "DETECTION_GAP"
    if activity == "vehicle_starts":
        if metrics["track_start"] > gt_start + 1:
            return "NO_STATIONARY_PRECEDING_START"
        if metrics["stationary_before"] < 1:
            return "NO_STATIONARY_PRECEDING_START"
        if max(metrics["speed_event_second_half"] or 0, metrics["speed_after"] or 0) < 0.28:
            return "NO_SUSTAINED_MOVEMENT"
        return "RULE_ABSTENTION"
    if metrics["track_end"] < gt_end - 1:
        return "NO_STATIONARY_AFTER_STOP"
    if metrics["stationary_after"] < 1:
        return "NO_STATIONARY_AFTER_STOP"
    if max(metrics["speed_before"] or 0, metrics["speed_event_first_half"] or 0) < 0.28:
        return "MOVEMENT_TOO_SLOW_OR_UNOBSERVED"
    return "RULE_ABSTENTION"


def load_observations(database):
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    try:
        observations = {
            row[0]: {"observation_id": row[1], "samples": json.loads(row[2]), "confidence": row[3]}
            for row in connection.execute("SELECT track_id,id,boxes,confidence FROM observations")
        }
        event_tracks = {
            row[0]: row[1] for row in connection.execute(
                "SELECT events.id, observations.track_id FROM events "
                "JOIN evidence ON events.evidence_id=evidence.id "
                "JOIN observations ON evidence.observation_id=observations.id"
            )
        }
        return observations, event_tracks
    finally:
        connection.close()


def analyze_run(directory):
    directory = Path(directory)
    observations, event_tracks = load_observations(directory / "vigilia.db")
    diagnoses = json.loads((directory / "failure-diagnostics.json").read_text())["annotations"]
    output, traces = [], []
    for gt in diagnoses:
        if gt["activity_type"] not in {"vehicle_starts", "vehicle_stops"}:
            continue
        for track_id in gt["actors"][0]["track_ids"]:
            observation = observations.get(track_id)
            if observation is None:
                raise ValueError(f"Missing observation for {track_id}")
            samples = observation["samples"]
            trace = state_trace(samples, observation["confidence"])
            metrics = trace_metrics(trace, gt["start_seconds"], gt["end_seconds_inclusive"])
            candidates = vehicle_transitions(track_id, [
                {**sample, "object_type": "car"} for sample in samples
            ])
            matched = event_tracks.get(gt["matched_event_id"]) == track_id
            output.append({
                "run": directory.name, "video": gt["video_id"], "camera": gt["camera_id"],
                "annotation_id": gt["annotation_id"], "track_id": track_id,
                "matched_event_id": gt["matched_event_id"] if matched else "",
                "activity": gt["activity_type"],
                "gt_start": round(gt["start_seconds"], 4), "gt_end": round(gt["end_seconds_inclusive"], 4),
                **metrics,
                "mean_confidence": observation["confidence"],
                "confidence_source": "track_mean",
                "current_rule_result": json.dumps([
                    {"kind": c.kind, "start": c.start, "end": c.end} for c in candidates
                    if c.kind in {"started_moving", "stopped"}
                ], separators=(",", ":")),
                "failure_reason": failure_reason(
                    gt["activity_type"], metrics, candidates,
                    gt["start_seconds"], gt["end_seconds_inclusive"], matched,
                ),
            })
            traces.append({
                "run": directory.name, "annotation_key": gt["key"],
                "track_id": track_id, "observation_id": observation["observation_id"],
                "model_derived_trace": trace,
            })
    return output, traces


def write_report(directories, csv_path, trace_path):
    rows, traces = [], []
    for directory in directories:
        run_rows, run_traces = analyze_run(directory)
        rows.extend(run_rows)
        traces.extend(run_traces)
    if not rows:
        raise ValueError("No MEVA vehicle tracks found")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(traces, indent=2) + "\n")
    return {run: dict(Counter(r["failure_reason"] for r in rows if r["run"] == run)) for run in sorted({r["run"] for r in rows})}
