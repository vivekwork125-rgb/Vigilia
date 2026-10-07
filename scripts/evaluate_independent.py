#!/usr/bin/env python3
"""Evaluate retrieval against labels made by reviewing source footage.

Ground truth names video/time/type/entity, never generated event IDs. The
30-query guard prevents a one-clip smoke test being presented as a benchmark.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.db import session
from app.retrieval import search

FIELDS = (
    "query_id",
    "query",
    "camera_id",
    "video_id",
    "event_type",
    "start_seconds",
    "end_seconds",
    "tolerance_seconds",
    "entity_id",
    "negative",
    "case",
)


def read_labels(path, allow_small=False):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS) <= set(reader.fieldnames or []):
            raise ValueError(
                "Label CSV is missing required columns: "
                + ", ".join(sorted(set(FIELDS) - set(reader.fieldnames or [])))
            )
        groups = defaultdict(list)
        for row in reader:
            if not row["query_id"] or not row["query"]:
                raise ValueError("query_id and query are required")
            groups[row["query_id"]].append(row)
    if not allow_small and len(groups) < 30:
        raise ValueError(
            f"Need at least 30 independently labeled queries; found {len(groups)}. Use --allow-small-fixture only for smoke tests."
        )
    for query_id, rows in groups.items():
        if len({row["query"] for row in rows}) != 1:
            raise ValueError(f"Inconsistent query text for {query_id}")
        if any(row["negative"].lower() == "true" for row in rows) and len(rows) != 1:
            raise ValueError(f"Negative query {query_id} must have exactly one row")
        for row in rows:
            if row["negative"].lower() != "true":
                if not row["video_id"] or not row["event_type"]:
                    raise ValueError(
                        f"Positive label {query_id} requires video_id and event_type"
                    )
                if float(row["end_seconds"]) < float(row["start_seconds"]):
                    raise ValueError(f"Reversed interval in {query_id}")
                if float(row["tolerance_seconds"]) < 0:
                    raise ValueError(f"Negative tolerance in {query_id}")
    return groups


def overlap(a, b, c, d):
    intersection = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return intersection / union if union else float(a == c)


def label_match(hit, label):
    evidence = hit["evidence"]
    if hit["video_id"] != label["video_id"] or hit["event_type"] != label["event_type"]:
        return False
    if label["entity_id"] and label["entity_id"] not in {
        e["id"] for e in hit["entities"]
    }:
        return False
    tol = float(label["tolerance_seconds"])
    return (
        abs(evidence["timestamp_start"] - float(label["start_seconds"])) <= tol
        and abs(evidence["timestamp_end"] - float(label["end_seconds"])) <= tol
    )


def evaluate(groups):
    rows, p1, p5, r5, rr, localization = [], [], [], [], [], []
    with session() as s:
        for query_id, labels in groups.items():
            query = labels[0]["query"]
            camera = labels[0]["camera_id"] or None
            result = search(s, query, camera=camera, limit=5)
            hits = result["results"]
            negative = labels[0]["negative"].lower() == "true"
            if negative:
                rows.append(
                    {
                        "query_id": query_id,
                        "case": labels[0]["case"],
                        "negative": True,
                        "false_positive": bool(hits),
                        "retrieved": [h["id"] for h in hits],
                    }
                )
                continue
            matched = set()
            ranks = []
            for rank, hit in enumerate(hits, 1):
                for index, label in enumerate(labels):
                    if index not in matched and label_match(hit, label):
                        matched.add(index)
                        ranks.append(rank)
                        ev = hit["evidence"]
                        localization.append(
                            {
                                "query_id": query_id,
                                "start_error_seconds": abs(
                                    ev["timestamp_start"]
                                    - float(label["start_seconds"])
                                ),
                                "end_error_seconds": abs(
                                    ev["timestamp_end"] - float(label["end_seconds"])
                                ),
                                "temporal_iou": overlap(
                                    ev["timestamp_start"],
                                    ev["timestamp_end"],
                                    float(label["start_seconds"]),
                                    float(label["end_seconds"]),
                                ),
                            }
                        )
                        break
            p1.append(float(1 in ranks))
            p5.append(len(ranks) / 5)
            r5.append(len(ranks) / len(labels))
            rr.append(1 / min(ranks) if ranks else 0)
            rows.append(
                {
                    "query_id": query_id,
                    "case": labels[0]["case"],
                    "negative": False,
                    "matched_labels": len(matched),
                    "label_count": len(labels),
                    "retrieved": [h["id"] for h in hits],
                }
            )
    mean = lambda x: round(statistics.mean(x), 4) if x else None
    negatives = [x for x in rows if x["negative"]]
    return {
        "scope": "Independent labels for indexed footage; validity depends on manual review and dataset sampling. No field-accuracy claim follows from a synthetic fixture.",
        "query_count": len(groups),
        "positive_queries": len(p1),
        "negative_queries": len(negatives),
        "metrics": {
            "precision_at_1": mean(p1),
            "precision_at_5": mean(p5),
            "recall_at_5": mean(r5),
            "mrr": mean(rr),
            "negative_false_positive_rate": mean(
                [float(x["false_positive"]) for x in negatives]
            ),
            "mean_temporal_iou_on_matched": mean(
                [x["temporal_iou"] for x in localization]
            ),
            "mean_start_error_seconds_on_matched": mean(
                [x["start_error_seconds"] for x in localization]
            ),
            "mean_end_error_seconds_on_matched": mean(
                [x["end_error_seconds"] for x in localization]
            ),
        },
        "localization": localization,
        "queries": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--allow-small-fixture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(read_labels(args.labels, args.allow_small_fixture))
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
