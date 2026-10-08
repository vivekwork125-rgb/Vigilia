#!/usr/bin/env python3
"""Run one controlled, evaluation-only MEVA raw-detector configuration."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from meva.dataset import load_manifest  # noqa: E402
from meva.perception_experiment import run  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "data/meva/improved-2fps")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or args.size < 32:
        parser.error("confidence must be (0,1) and size at least 32")
    diagnosis = json.loads((args.baseline_run / "failure-diagnostics.json").read_text())
    report = run(load_manifest(args.manifest), diagnosis, args.model, args.size, args.confidence, args.output)
    print(json.dumps({"model": report["model"], "size": args.size, "confidence": args.confidence, "summary": report["summary"]}, indent=2))
