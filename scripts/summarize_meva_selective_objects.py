#!/usr/bin/env python3
"""Summarize person-triggered ROI diagnostics without changing runtime inference."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    results = [
        row for path in sorted(args.run_dir.glob("*.json"))
        if "strategy" in (row := json.loads(path.read_text()))
    ]
    if len(results) != 9:
        raise ValueError(f"Expected nine strategy/scale results, got {len(results)}")
    rows = list(csv.DictReader(args.ontology.open(newline="")))
    for row in rows:
        key = row["annotation_key"]
        for stage, field in (
            ("raw", "raw_keys"),
            ("tracked", "tracked_keys"),
            ("associated", "associated_keys"),
        ):
            row[f"selective_{stage}_strategies"] = ";".join(
                f"{run['strategy']}-{run['factor']}" for run in results
                if key in run[field]
            )
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    with args.cases.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "fixed_baseline": "YOLO11n 640/0.30, ByteTrack, 2 FPS; person tracks from frozen ba838cf run",
        "evaluation_only": True,
        "configurations": [
            {k: run[k] for k in (
                "strategy", "factor", "source_frames", "person_track_crops",
                "object_detections_after_crop_dedup", "object_tracklets",
                "raw_object_coverage", "object_track_coverage",
                "person_object_association_coverage", "portable_raw_coverage",
                "portable_track_coverage", "portable_association_coverage",
                "wall_seconds",
            )} for run in results
        ],
        "ontology_cases": len(rows),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/selective-objects")
    parser.add_argument("--ontology", type=Path, default=ROOT / "docs/meva-perception-objects.csv")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/meva-small-object-cases.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/meva/selective-objects/summary.json")
    main(parser.parse_args())
