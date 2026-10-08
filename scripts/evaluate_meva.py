#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from meva.dataset import DEFAULT_MANIFEST, ROOT, load_manifest
from meva.evaluation import evaluate
from meva.processing import configure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent MEVA evaluation; writes metrics, predictions, GT and error analysis"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/baseline")
    parser.add_argument("--tolerance", type=float, default=2)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--min-spatial-iou", type=float, default=0.1)
    parser.add_argument("--verify-api", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(
            load_manifest(args.manifest),
            configure(args.run_dir),
            args.tolerance,
            args.min_iou,
            args.min_spatial_iou,
            args.verify_api,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "detection",
                        "retrieval",
                        "evidence_integrity",
                        "api_checks",
                        "search_integrity_failures",
                    )
                },
                indent=2,
            )
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")
