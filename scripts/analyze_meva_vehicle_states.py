#!/usr/bin/env python3
"""Export diagnostic vehicle state traces and per-track failure attribution."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

if __name__ == "__main__":
    from meva.vehicle_state_analysis import write_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[
        ROOT / "data/meva/improved-2fps",
        ROOT / "data/meva/perception-yolo11s-640-c030",
        ROOT / "data/meva/perception-yolo11n-960-c030",
    ])
    parser.add_argument("--csv", type=Path, default=ROOT / "docs/meva-vehicle-state-analysis.csv")
    parser.add_argument("--traces", type=Path, default=ROOT / "data/meva/vehicle-state-traces.json")
    args = parser.parse_args()
    print(json.dumps(write_report(args.runs, args.csv, args.traces), indent=2))
