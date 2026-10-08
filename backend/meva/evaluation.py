"""Independent camera/time/class/spatial matching; annotation IDs never match event IDs."""

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .annotations import parse_activities, parse_geometry, parse_types
from .dataset import resolve, validate
from .mapping import ACTIVITIES, mapping

VEHICLES = {"car", "truck", "bus", "motorcycle"}
QUERIES = (
    ("person picked up object", "picked_up", "person_picks_up_object"),
    ("person put down object", "placed_object", "person_puts_down_object"),
    ("person approached object", "approached_object", None),
    ("unattended bag", "unattended", None),
    ("vehicle stopped", "stopped", "vehicle_stops"),
    ("vehicle started", "started_moving", "vehicle_starts"),
    ("person entered scene", "appeared", "person_enters_scene_through_structure"),
    ("person exited scene", "disappeared", "person_exits_scene_through_structure"),
    ("person talked to another person", "talks_to_person", "person_talks_to_person"),
    ("person used phone", "talks_on_phone", "person_talks_on_phone"),
    ("vehicle turned left", "turned_left", "vehicle_turns_left"),
    ("vehicle turned right", "turned_right", "vehicle_turns_right"),
    ("vehicle reversed", "reversed", "vehicle_reverses"),
)


def average(values):
    return round(statistics.mean(values), 6) if values else None


def temporal_iou(prediction, truth):
    a, b = prediction["start_seconds"], prediction["end_seconds"]
    c, d = truth["start_seconds"], truth["end_seconds"]
    union = max(b, d) - min(a, c)
    return max(0.0, min(b, d) - max(a, c)) / union if union else float(a == c)


def interval_errors(prediction, truth):
    return {
        "temporal_iou": temporal_iou(prediction, truth),
        "start_error_seconds": abs(
            prediction["start_seconds"] - truth["start_seconds"]
        ),
        "end_error_seconds": abs(prediction["end_seconds"] - truth["end_seconds"]),
    }


def class_compatible(prediction, truth):
    classes = set(prediction["entity_types"])
    if truth["activity_type"].startswith("vehicle_"):
        return bool(classes & VEHICLES)
    if "person" not in classes:
        return False
    if truth["activity_type"] in ("person_picks_up_object", "person_puts_down_object"):
        return bool(classes - {"person"})
    return True


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1)


def spatial_compatibility(prediction, truth, geometry, fps, tolerance):
    """Require a visible compatible actor, and object for manipulation events.

    Evaluate annotation geometry only at actual runtime observation sample frames.
    No reidentification and no identity claims follow from this overlap.
    """
    people, objects = [], []
    for actor in truth["actor_ids"]:
        typ = truth["actor_types"][str(actor)]
        best = 0.0
        for track in prediction["supporting_tracks"]:
            runtime_type = track["object_type"]
            compatible = (
                (typ == "person" and runtime_type == "person")
                or (typ == "vehicle" and runtime_type in VEHICLES)
                or (typ not in ("person", "vehicle") and runtime_type != "person")
            )
            if not compatible:
                continue
            for sample in track["boxes"]:
                if (
                    not truth["start_seconds"] - tolerance
                    <= sample["t"]
                    <= truth["end_seconds"] + tolerance
                ):
                    continue
                frame = sample["frame"]
                box = geometry.get(actor, {}).get(frame)
                if box is not None:
                    best = max(best, box_iou(sample["box"], box))
        (people if typ == "person" else objects).append(best)
    if truth["activity_type"] in ("person_picks_up_object", "person_puts_down_object"):
        return min(max(people, default=0), max(objects, default=0))
    return max(people + objects, default=0)


def eligible(prediction, truth, tolerance=2.0, min_iou=0.1, min_spatial_iou=0.1):
    if (
        prediction["camera_id"] != truth["camera_id"]
        or prediction["video_id"] != truth["video_id"]
        or prediction["event_type"] != truth["event_type"]
        or not class_compatible(prediction, truth)
    ):
        return False
    errors = interval_errors(prediction, truth)
    temporal = errors["temporal_iou"] >= min_iou or (
        errors["start_error_seconds"] <= tolerance
        and errors["end_error_seconds"] <= tolerance
    )
    return (
        temporal
        and prediction.get("spatial_scores", {}).get(truth["key"], 0) >= min_spatial_iou
    )


def match(predictions, truths, tolerance=2.0, min_iou=0.1, min_spatial_iou=0.1):
    """Maximum cardinality assignment first; IoU/timing break ties, never IDs."""
    if not predictions or not truths:
        return []
    weights = np.zeros((len(predictions), len(truths)))
    cardinality_bonus = min(len(predictions), len(truths)) + 1
    for i, prediction in enumerate(predictions):
        for j, truth in enumerate(truths):
            if eligible(prediction, truth, tolerance, min_iou, min_spatial_iou):
                errors = interval_errors(prediction, truth)
                weights[i, j] = (
                    cardinality_bonus
                    + errors["temporal_iou"]
                    + 0.001
                    / (1 + errors["start_error_seconds"] + errors["end_error_seconds"])
                )
    indices, columns = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j)) for i, j in zip(indices, columns) if weights[i, j] > 0]


def detection_metrics(predictions, truths, pairs):
    errors = [interval_errors(predictions[i], truths[j]) for i, j in pairs]
    tp, fp, fn = len(pairs), len(predictions) - len(pairs), len(truths) - len(pairs)
    return {
        "ground_truth_events": len(truths),
        "predicted_events": len(predictions),
        "matched_events": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / len(predictions) if predictions else None,
        "recall": tp / len(truths) if truths else None,
        "false_positive_rate": fp / len(predictions) if predictions else None,
        "false_positive_rate_definition": "unmatched predictions / predictions (false discovery rate); true negative count unavailable",
        "mean_temporal_iou_on_matched": average([e["temporal_iou"] for e in errors]),
        "mean_start_error_seconds_on_matched": average(
            [e["start_error_seconds"] for e in errors]
        ),
        "mean_end_error_seconds_on_matched": average(
            [e["end_error_seconds"] for e in errors]
        ),
    }


def predictions_and_integrity(s, videos):
    from app.db import Entity, Event, EventEntity, Evidence, Observation, Track
    import hashlib
    from datetime import datetime, timedelta
    from sqlalchemy import select

    entities = {e.id: e for e in s.scalars(select(Entity))}
    observations = {o.id: o for o in s.scalars(select(Observation))}
    tracks = {t.id: t for t in s.scalars(select(Track))}
    evidence = {e.id: e for e in s.scalars(select(Evidence))}
    links = defaultdict(list)
    entity_obs = defaultdict(list)
    for link in s.scalars(select(EventEntity)):
        links[link.event_id].append(link.entity_id)
    for obs in observations.values():
        entity_obs[obs.entity_id].append(obs)
    result, invalid = [], []
    for event in s.scalars(select(Event)):
        ev = evidence.get(event.evidence_id)
        if ev is None or ev.video_id not in videos:
            invalid.append(
                {"event_id": event.id, "reason": "missing source evidence/video"}
            )
            continue
        video, obs = videos[ev.video_id], observations.get(ev.observation_id)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "video": ev.video_id,
                    "source_sha256": video["sha256"],
                    "frame_start": ev.frame_start,
                    "frame_end": ev.frame_end,
                    "method": ev.method,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        refs = []
        for key in links[event.id]:
            for supporting in entity_obs[key]:
                if supporting.video_id == ev.video_id:
                    refs.append(
                        {
                            "observation_id": supporting.id,
                            "track_id": supporting.track_id,
                            "entity_id": key,
                            "object_type": entities[key].object_type,
                            "boxes": supporting.boxes,
                        }
                    )
        valid = (
            obs is not None
            and obs.track_id in tracks
            and obs.video_id == ev.video_id
            and tracks[obs.track_id].video_id == ev.video_id
            and event.camera_id == video["camera_id"]
            and bool(refs)
            and bool(ev.method)
            and ev.sha256 == signature
            and datetime.fromisoformat(event.start)
            == datetime.fromisoformat(video["recording_start"])
            + timedelta(seconds=ev.timestamp_start)
            and datetime.fromisoformat(event.end)
            == datetime.fromisoformat(video["recording_start"])
            + timedelta(seconds=ev.timestamp_end)
            and 0 <= ev.frame_start <= ev.frame_end < video["frame_count"]
            and 0 <= ev.timestamp_start <= ev.timestamp_end <= video["duration"]
            and abs(ev.frame_start / video["fps"] - ev.timestamp_start)
            <= 1 / video["fps"] + 1e-6
            and abs(ev.frame_end / video["fps"] - ev.timestamp_end)
            <= 1 / video["fps"] + 1e-6
            and all(
                tracks.get(r["track_id"])
                and tracks[r["track_id"]].entity_id == r["entity_id"]
                for r in refs
            )
        )
        if not valid:
            invalid.append(
                {
                    "event_id": event.id,
                    "reason": "broken observation/track/frame/timestamp chain",
                }
            )
        result.append(
            {
                "event_id": event.id,
                "event_type": event.event_type,
                "category": event.category,
                "video_id": ev.video_id,
                "camera_id": event.camera_id,
                "start_seconds": ev.timestamp_start,
                "end_seconds": ev.timestamp_end,
                "start_frame": ev.frame_start,
                "end_frame": ev.frame_end,
                "evidence_id": ev.id,
                "observation_id": ev.observation_id,
                "track_ids": [r["track_id"] for r in refs],
                "entity_ids": links[event.id],
                "entity_types": [entities[k].object_type for k in links[event.id]],
                "source_video_sha256": video["sha256"],
                "method": ev.method,
                "supporting_tracks": refs,
                "valid_evidence": valid,
            }
        )
    return result, {
        "predictions": len(result),
        "invalid": invalid,
        "valid_source_evidence_percent": 100
        * (len(result) - len(invalid))
        / len(result)
        if result
        else None,
    }


def review_record(kind, prediction, truth, reason=None):
    row = {
        "kind": kind,
        "prediction": public_prediction(prediction) if prediction else None,
        "ground_truth": truth,
        "reason": reason,
    }
    if prediction and truth:
        row.update(interval_errors(prediction, truth))
        row["spatial_iou"] = prediction.get("spatial_scores", {}).get(truth["key"], 0)
    return row


def public_prediction(prediction):
    return {
        k: v
        for k, v in prediction.items()
        if k not in ("supporting_tracks", "spatial_scores")
    }


def evaluate(
    manifest,
    directory,
    tolerance=2.0,
    min_iou=0.1,
    min_spatial_iou=0.1,
    verify_api=False,
):
    if tolerance < 0 or not 0 < min_iou <= 1 or not 0 < min_spatial_iou <= 1:
        raise ValueError("Tolerance must be >=0 and IoU thresholds must be in (0,1]")
    validation = validate(manifest)
    from app.db import Video, session
    from app.retrieval import search
    from sqlalchemy import select

    videos = {v["video_id"]: v for v in manifest["videos"]}
    with session() as s:
        stored = {v.id: v for v in s.scalars(select(Video))}
        if set(stored) != set(videos) or any(
            v.status != "completed" for v in stored.values()
        ):
            raise ValueError(
                "All manifest videos must be successfully indexed in the isolated DB"
            )
        if any(stored[k].sha256 != v["sha256"] for k, v in videos.items()):
            raise ValueError("Processed source hash differs from manifest")
        predictions, integrity = predictions_and_integrity(s, videos)
        if integrity["invalid"]:
            raise ValueError(f"Evidence validation failed: {integrity['invalid'][:5]}")
        ground_truth = []
        for video in videos.values():
            types = parse_types(resolve(video["types_path"]))
            rows = parse_activities(resolve(video["annotation_path"]), video, types)
            geometry = parse_geometry(resolve(video["geometry_path"]), video, types)
            for truth in rows:
                truth.update(mapping(truth["activity_type"]))
                truth["key"] = (
                    f"{truth['video_id']}:{truth['annotation_id']}:{truth['span_index']}"
                )
                ground_truth.append(truth)
                if truth["status"] != "UNSUPPORTED":
                    for prediction in predictions:
                        if (
                            prediction["video_id"] == truth["video_id"]
                            and prediction["event_type"] == truth["event_type"]
                            and class_compatible(prediction, truth)
                        ):
                            prediction.setdefault("spatial_scores", {})[
                                truth["key"]
                            ] = spatial_compatibility(
                                prediction, truth, geometry, video["fps"], tolerance
                            )
            del geometry
        errors, breakdown, groups = [], [], {}
        for status in ("DIRECT", "APPROXIMATE"):
            truths = [g for g in ground_truth if g["status"] == status]
            event_types = {
                mapping(a)["event_type"]
                for a in ACTIVITIES
                if mapping(a)["status"] == status
            }
            selected = [
                p
                for p in predictions
                if p["event_type"] in event_types
                and (
                    p["event_type"] not in ("stopped", "started_moving")
                    or set(p["entity_types"]) & VEHICLES
                )
                and (status != "APPROXIMATE" or "person" in p["entity_types"])
            ]
            pairs = match(selected, truths, tolerance, min_iou, min_spatial_iou)
            groups[status] = detection_metrics(selected, truths, pairs)
            matched_pred, matched_gt = {i for i, _ in pairs}, {j for _, j in pairs}
            for i, prediction in enumerate(selected):
                if i not in matched_pred:
                    compatible = [
                        g
                        for g in truths
                        if g["camera_id"] == prediction["camera_id"]
                        and g["event_type"] == prediction["event_type"]
                    ]
                    nearest = (
                        min(
                            compatible,
                            key=lambda g: sum(
                                interval_errors(prediction, g)[k]
                                for k in ("start_error_seconds", "end_error_seconds")
                            ),
                        )
                        if compatible
                        else None
                    )
                    errors.append(
                        review_record(
                            "FALSE_POSITIVE",
                            prediction,
                            nearest,
                            "Unmatched prediction; nearest label is diagnostic only",
                        )
                    )
            for j, truth in enumerate(truths):
                if j not in matched_gt:
                    compatible = [
                        p
                        for p in selected
                        if p["camera_id"] == truth["camera_id"]
                        and p["event_type"] == truth["event_type"]
                    ]
                    nearest = (
                        min(
                            compatible,
                            key=lambda p: abs(
                                p["start_seconds"] - truth["start_seconds"]
                            ),
                        )
                        if compatible
                        else None
                    )
                    errors.append(
                        review_record(
                            "FALSE_NEGATIVE",
                            nearest,
                            truth,
                            "No compatible prediction within matching criteria",
                        )
                    )
            for dimension in ("camera_id", "activity_type", "event_type"):
                values = sorted(
                    {g[dimension] for g in truths}
                    | (
                        {p[dimension] for p in selected}
                        if dimension != "activity_type"
                        else set()
                    )
                )
                for value in values:
                    gt_indices = [
                        j for j, g in enumerate(truths) if g[dimension] == value
                    ]
                    relevant_types = {truths[j]["event_type"] for j in gt_indices}
                    pred_indices = [
                        i
                        for i, p in enumerate(selected)
                        if (
                            p.get(dimension) == value
                            if dimension != "activity_type"
                            else p["event_type"] in relevant_types
                        )
                    ]
                    local_pairs = [
                        (pred_indices.index(i), gt_indices.index(j))
                        for i, j in pairs
                        if i in pred_indices and j in gt_indices
                    ]
                    breakdown.append(
                        {
                            "mapping_status": status,
                            "dimension": dimension,
                            "value": value,
                            **detection_metrics(
                                [selected[i] for i in pred_indices],
                                [truths[j] for j in gt_indices],
                                local_pairs,
                            ),
                        }
                    )
        prediction_by_id = {p["event_id"]: p for p in predictions}
        query_rows, query_integrity_failures = [], []
        for video in videos.values():
            for query, event_type, activity in QUERIES:
                begun = time.perf_counter()
                retrieved = search(s, query, camera=video["camera_id"], limit=5)
                latency = (time.perf_counter() - begun) * 1000
                hits = retrieved["results"]
                truths = [
                    g
                    for g in ground_truth
                    if g["video_id"] == video["video_id"]
                    and g["activity_type"] == activity
                ]
                status = (
                    mapping(activity)["status"] if activity else "UNLABELED_IN_MEVA"
                )
                selected = [prediction_by_id[h["id"]] for h in hits]
                pairs = (
                    match(selected, truths, tolerance, min_iou, min_spatial_iou)
                    if status in ("DIRECT", "APPROXIMATE")
                    else []
                )
                ranks = [i + 1 for i, _ in pairs]
                row = {
                    "query": query,
                    "camera_id": video["camera_id"],
                    "mapping_status": status,
                    "ground_truth_count": len(truths),
                    "parsed": retrieved["parsed"],
                    "latency_ms": latency,
                    "retrieved": [public_prediction(p) for p in selected],
                    "matched_count": len(pairs),
                    "precision_at_1": float(1 in ranks)
                    if truths and status != "UNSUPPORTED"
                    else None,
                    "precision_at_5": len(pairs) / 5
                    if truths and status != "UNSUPPORTED"
                    else None,
                    "recall_at_5": len(pairs) / len(truths)
                    if truths and status != "UNSUPPORTED"
                    else None,
                    "mrr": 1 / min(ranks)
                    if ranks
                    else (0 if truths and status != "UNSUPPORTED" else None),
                    "negative_query_false_positive": bool(hits)
                    if not truths and status in ("DIRECT", "APPROXIMATE")
                    else None,
                }
                for hit in hits:
                    prediction = prediction_by_id[hit["id"]]
                    # Exact expected type (including documented appearance proxy), source and ranking signals.
                    valid = (
                        hit["event_type"] == event_type
                        and hit["camera_id"] == video["camera_id"]
                        and prediction["valid_evidence"]
                        and bool(hit["explanations"])
                        and bool(hit["signals"])
                    )
                    if not valid:
                        query_integrity_failures.append(
                            {
                                "query": query,
                                "event_id": hit["id"],
                                "reason": "wrong event/camera or missing evidence/explanation",
                            }
                        )
                query_rows.append(row)
        retrieval_metrics = {}
        for status in ("DIRECT", "APPROXIMATE"):
            positives = [
                r
                for r in query_rows
                if r["mapping_status"] == status and r["ground_truth_count"]
            ]
            negatives = [
                r
                for r in query_rows
                if r["mapping_status"] == status and not r["ground_truth_count"]
            ]
            retrieval_metrics[status] = {
                k: average([r[k] for r in positives])
                for k in ("precision_at_1", "precision_at_5", "recall_at_5", "mrr")
            }
            retrieval_metrics[status].update(
                {
                    "positive_queries": len(positives),
                    "negative_queries": len(negatives),
                    "negative_query_false_positive_rate": average(
                        [float(r["negative_query_false_positive"]) for r in negatives]
                    ),
                }
            )
    api_checks = (
        verify_evidence_api(predictions) if verify_api else {"performed": False}
    )
    unsupported = [
        {"activity_type": a, "ground_truth_count": n, **mapping(a)}
        for a, n in sorted(
            Counter(
                g["activity_type"] for g in ground_truth if g["status"] == "UNSUPPORTED"
            ).items()
        )
    ]
    report = {
        "dataset": "MEVA drops-123-r13 selected eight videos",
        "validation": validation,
        "metadata": json.loads((directory / "run-config.json").read_text()),
        "matching": {
            "tolerance_seconds": tolerance,
            "minimum_temporal_iou": min_iou,
            "minimum_spatial_iou": min_spatial_iou,
            "policy": "same camera/video/type, compatible classes, temporal IoU OR both endpoints within tolerance, annotation/runtime actor box overlap; maximum cardinality one-to-one assignment",
        },
        "annotation_activity_count": len(ground_truth),
        "all_predictions": len(predictions),
        "prediction_type_counts": dict(Counter(p["event_type"] for p in predictions)),
        "detection": groups,
        "retrieval": retrieval_metrics,
        "breakdown": breakdown,
        "unsupported_activities": unsupported,
        "mapping": {a: mapping(a) for a in ACTIVITIES},
        "evidence_integrity": integrity,
        "api_checks": api_checks,
        "search_integrity_failures": query_integrity_failures,
        "queries": query_rows,
        "performance": json.loads((directory / "processing-results.json").read_text()),
        "search_mean_latency_ms": average([r["latency_ms"] for r in query_rows]),
        "limitations": [
            "Selected subset is an evaluation set, not a held-out generalization claim.",
            "Approximate appearance/disappearance proxies are reported separately from direct activities.",
            "False positives mean unmatched predictions, not proof that an unannotated activity did not occur.",
            "MEVA has no matching approached_object/unattended activity labels; those query accuracies are unmeasured.",
            "Unattended does not equal the intentional MEVA person_abandons_package activity.",
            "Spatial overlap constrains local matching, not identity verification across cameras.",
            "No cross-camera identity benchmark; MEVA actor IDs are never runtime identities.",
            "P@5 uses denominator five. Queries without labels are excluded from positive retrieval averages.",
            "Phone-use query does not independently measure texting; unsupported queries receive no successful score.",
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark-results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (directory / "error-analysis.json").write_text(json.dumps(errors, indent=2) + "\n")
    (directory / "ground-truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n"
    )
    (directory / "predictions.json").write_text(
        json.dumps([public_prediction(p) for p in predictions], indent=2) + "\n"
    )
    write_csv(directory / "benchmark-results.csv", breakdown)
    flat_errors = []
    for e in errors:
        p, g = e["prediction"] or {}, e["ground_truth"] or {}
        row = {
            "kind": e["kind"],
            "camera_id": g.get("camera_id", p.get("camera_id")),
            "video_id": g.get("video_id", p.get("video_id")),
            "event_type": g.get("event_type", p.get("event_type")),
            "reason": e["reason"],
        }
        row.update(
            {
                "prediction_" + k: p.get(k)
                for k in (
                    "event_id",
                    "evidence_id",
                    "track_ids",
                    "start_seconds",
                    "end_seconds",
                    "start_frame",
                    "end_frame",
                )
            }
        )
        row.update(
            {
                "ground_truth_" + k: g.get(k)
                for k in (
                    "annotation_id",
                    "activity_type",
                    "start_seconds",
                    "end_seconds",
                    "start_frame",
                    "end_frame",
                    "source_annotation",
                )
            }
        )
        row.update(
            {
                k: e.get(k)
                for k in (
                    "temporal_iou",
                    "start_error_seconds",
                    "end_error_seconds",
                    "spatial_iou",
                )
            }
        )
        flat_errors.append(row)
    write_csv(directory / "error-analysis.csv", flat_errors)
    if query_integrity_failures:
        raise ValueError(
            f"Search validation found {len(query_integrity_failures)} wrong-type/unsupported query results; diagnostic report saved"
        )
    return report


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def verify_evidence_api(predictions):
    """Exercise real ASGI endpoints without starting a worker or seeding fixtures."""
    import cv2
    from app.main import app
    from fastapi.testclient import TestClient

    failures = []
    from app.config import TOKEN
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    with TestClient(
        app, raise_server_exceptions=True, headers={"Authorization": "Bearer " + TOKEN}
    ) as client:
        # DEMO_MODE=false; indexed DB has no queued work.
        for prediction in predictions:
            response = client.get("/events/" + prediction["event_id"])
            if response.status_code != 200:
                failures.append(
                    {
                        "event_id": prediction["event_id"],
                        "endpoint": "event",
                        "status": response.status_code,
                    }
                )
            response = client.get("/evidence/" + prediction["evidence_id"])
            if (
                response.status_code != 200
                or not response.json().get("observation")
                or not response.json().get("source_boxes")
                or response.json().get("source", {}).get("sha256")
                != prediction["source_video_sha256"]
            ):
                failures.append(
                    {
                        "event_id": prediction["event_id"],
                        "endpoint": "evidence",
                        "status": response.status_code,
                    }
                )
            graph = client.get("/graph", params={"event_id": prediction["event_id"]})
            if graph.status_code != 200 or not any(
                n["id"] == prediction["event_id"] for n in graph.json()["nodes"]
            ):
                failures.append(
                    {
                        "event_id": prediction["event_id"],
                        "endpoint": "source graph",
                        "status": graph.status_code,
                    }
                )
        for video_id in sorted({p["video_id"] for p in predictions}):
            p = next(p for p in predictions if p["video_id"] == video_id)
            response = client.get("/evidence/" + p["evidence_id"] + "/thumbnail")
            if (
                response.status_code != 200
                or cv2.imdecode(
                    np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                is None
            ):
                failures.append(
                    {
                        "video_id": video_id,
                        "endpoint": "source thumbnail",
                        "status": response.status_code,
                    }
                )
    if failures:
        raise ValueError(f"Evidence API verification failed: {failures[:5]}")
    return {
        "performed": True,
        "event_endpoints": len(predictions),
        "evidence_endpoints": len(predictions),
        "scoped_graph_endpoints": len(predictions),
        "decoded_source_thumbnails": len({p["video_id"] for p in predictions}),
        "failures": [],
    }
