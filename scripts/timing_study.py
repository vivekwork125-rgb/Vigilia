#!/usr/bin/env python3
"""Record actual paired manual/assisted investigation timings to a local CSV."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

FIELDS = (
    "recorded_at",
    "task",
    "participant",
    "trial",
    "mode",
    "first_relevant_seconds",
    "correct_seconds",
    "total_seconds",
    "notes",
)


def record(path, args):
    a, b, c = args.first_relevant, args.correct, args.total
    if not 0 < a <= b <= c:
        raise ValueError(
            "Require 0 < first relevant <= correct evidence <= total seconds"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "task": args.task,
                "participant": args.participant,
                "trial": args.trial,
                "mode": args.mode,
                "first_relevant_seconds": a,
                "correct_seconds": b,
                "total_seconds": c,
                "notes": args.notes,
            }
        )


def summarize(path):
    if not path.exists():
        return {
            "paired_trials": 0,
            "metrics": None,
            "message": "No human timings recorded.",
        }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = {}
    for row in rows:
        key = (row["task"], row["participant"], row["trial"])
        entry = pairs.setdefault(key, {})
        if row["mode"] in entry:
            raise ValueError(f"Duplicate {row['mode']} recording for {key}")
        entry[row["mode"]] = row
    complete = [
        (key, modes["manual"], modes["assisted"])
        for key, modes in pairs.items()
        if {"manual", "assisted"} <= modes.keys()
    ]
    if not complete:
        return {
            "paired_trials": 0,
            "unpaired_recordings": len(rows),
            "metrics": None,
            "message": "At least one matched participant/task/trial pair is required.",
        }
    metrics = {}
    for field in ("first_relevant_seconds", "correct_seconds", "total_seconds"):
        manual = [float(m[field]) for _, m, _ in complete]
        assisted = [float(a[field]) for _, _, a in complete]
        metrics[field] = {
            "manual_mean": round(statistics.mean(manual), 3),
            "assisted_mean": round(statistics.mean(assisted), 3),
            "mean_paired_reduction": round(
                statistics.mean(1 - a / m for m, a in zip(manual, assisted)), 4
            ),
        }
    return {
        "paired_trials": len(complete),
        "unpaired_recordings": len(rows) - len(complete) * 2,
        "metrics": metrics,
        "pairs": [
            {"task": task, "participant": participant, "trial": trial}
            for (task, participant, trial), _, _ in complete
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("data/timing-study.csv"))
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("record")
    for name in ("task", "participant", "trial"):
        add.add_argument(f"--{name}", required=True)
    add.add_argument("--mode", required=True, choices=("manual", "assisted"))
    add.add_argument("--first-relevant", required=True, type=float)
    add.add_argument("--correct", required=True, type=float)
    add.add_argument("--total", required=True, type=float)
    add.add_argument("--notes", default="")
    commands.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "record":
        record(args.file, args)
        print(
            f"Recorded measured {args.mode} trial for {args.task} / {args.participant} / {args.trial}"
        )
    else:
        print(json.dumps(summarize(args.file), indent=2))


if __name__ == "__main__":
    main()
