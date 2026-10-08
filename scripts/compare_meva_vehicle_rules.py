#!/usr/bin/env python3
"""Compare offline vehicle transitions with unchanged MEVA independent matching."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

if __name__ == "__main__":
    from meva.vehicle_rule_experiment import compare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/improved-2fps")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/meva/vehicle-rule-comparison.json")
    args = parser.parse_args()
    result = compare(args.run_dir, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
