"""Validate human object-identity annotations; fail if the pilot is incomplete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.object_identity_benchmark import validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path,
                        default=ROOT / "datasets/meva/object-identity/clips.json")
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "datasets/meva/object-identity/annotations.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/meva/object-identity/validation-report.json")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Validate saved records while reporting outstanding manual review")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.clips, args.annotations)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, str(exc) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid_schema"] or (not args.allow_incomplete and not report["complete_pilot"]):
        parser.exit(1, "Object-identity review incomplete or invalid\n")


if __name__ == "__main__":
    main()
